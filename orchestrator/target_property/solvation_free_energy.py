import numpy as np
# import yaml
import subprocess
import shutil
# import matplotlib.pyplot as plt
import random
# from datetime import datetime
import os
# from glob import glob
from os.path import isfile  # split, getmtime
from typing import Union, Optional, Any, Dict
from ..simulator import simulator_builder
from ..potential import Potential
from . import TargetProperty
from ..scheduler import Scheduler
from ..storage import Storage
from orchestrator.target_property.analysis import AnalyzeLammpsLog
from ..utils.restart import restarter
# from ..utils.isinstance import isinstance_no_import


class SolvationFreeEnergy(TargetProperty):
    """
    Class to compute solvation free energy for a solute in a pure or
    multi-component solvent.

    This is done by thermodynamic integration in a three step process:
    1. First, the charges of the solute molecule are turned off
    (scaled to 0).
    2. Next, the long-range van der Waals interactions are turned off
    (scaled to 0).
    3. Finally, the charges of the isolated solute are turned back on
    (re-scaled to 1).

    This module reports the solvation free energy, or the change in free
    energy for solute to move FROM ideal gas INTO solution. A negative
    result indicates a favorable dissolution, and a positive results
    indicates an unfavorable dissolution.

    Simulation parameters and required paths are read from json file.

    :param simulator_type: name of the simulator to perform simulations
    :type simulator_type: str
    :param simulator_path: path to the simulator executable
    :type simulator_path: str
    :param elements: list of elements which are present in the simulation
    :type elements: list
    :param input_template: LAMMPS template input files. This is a dictionary
        of key-value pairs that can be used to define any simulation input
        template files required for conducting simulations (e.g. one input
        simulations).
    :type input_template: dict
    :param job_details: optional parameters for running the job
    :type job_details: dict
    :param npt_job_details: optional parameters for running the NPT job
    :type npt_job_details: dict
    :param nph_job_details: optional parameters for running the NPH job
    :type nph_job_details: dict

    """

    def __init__(
        self,
        simulator_type: str,
        simulator_path: str,
        job_details: dict,
        input_template: dict,
        elements: list = [],
        min_job_details: dict = [],
        equil_job_details: dict = [],
        ti_job_details: dict = [],
        **kwargs: Any,
    ):
        """
        Initialization of the MeltingPoint class with args dict

        :param simulator_type: name of the simulator to perform simulations
        :type simulator_type: str
        :param simulator_path: path to the simulator executable
        :type simulator_path: str
        :param elements: list of elements which are present in the simulation
        :type elements: list
        :param input_template: LAMMPS template input file. This is a dictionary
            of key-value pairs that can be used to define any simulation input
            template files required for conducting simulations. One template is
            required per key: ``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
            ``ti_vacuum``
        :type input_template: dict
        :param job_details: optional parameters for running the job
        :type job_details: dict
        :param min_job_details: optional parameters for running the packed
        structure minimization job
        :type min_job_details: dict
        :param equil_job_details: optional parameters for running the
        equilibration job
        :type equil_job_details: dict
        :param ti_job_details: optional parameters for running the
        thermodynamic integration job
        :type ti_job_details: dict
        """

        # Define default simulation parameters
        self.default_sim_params = {
            "units": "real",
            "atom_style": "full",
            "pair_style": "unknown",
            "pair_style_args": {
                "cutoff": "none",
                "cutoff2": "none",
                "softcore_n": "none",
                "alpha_vdw": "none",
                "alpha_elec": "none",
                "inner": "none",
                "outer": "none"
            },
            "pair_modify": "mix arithmetic",
            "special_bonds": "none",
            "bond_style": "none",
            "angle_style": "none",
            "dihedral_style": "none",
            "kspace_style": "pppm 1.0e-4",
            "improper_style": "none",
            "temp": 298.15,
            "press": 1.0,
            "temp_damp": 100.0,
            "press_damp": 250.0,
            "timestep": 0.1,
            "equil_steps": 50000,
            "ti_steps": 100000
        }

        self.default_solvation_params = {
            "free_energy": "gibbs",
            "pack_tol": 2.0,
            "pack_boxlen": 40.0,
            "solute": "unknown",
            "solvent": ["unknown"],
            "lambda_values": 11,
            "lambda_diff": 0.002
        }

        # Set default seed
        self.default_seed = 4928302

        # If differences are found in pair_style or pair_style_args,
        # should they be overwritten?
        self.override_with_init = True

        self.min_job_details = {**job_details, **min_job_details}
        self.equil_job_details = {**job_details, **equil_job_details}
        self.ti_job_details = {**job_details, **ti_job_details}

        self.input_template = input_template

        self.input_template_min = input_template['min']
        self.input_template_equil = input_template['equil']
        self.input_template_ti_elec = input_template['ti_elec']
        self.input_template_ti_vdw = input_template['ti_vdw']
        self.input_template_ti_vacuum = input_template['ti_vacuum']

        simulator_args = {'code_path': simulator_path, 'elements': elements}
        self.built_simulator = simulator_builder.build(simulator_type,
                                                       simulator_args)

        self.progress_flag = 'init'
        self.current_state = {}
        self.pack_dir = None
        self.min_dir = []
        self.no_charge_dir = []

        self.elec_lambda_calcs = []
        self.vdw_lambda_calcs = []
        self.vacuum_lambda_calcs = []
        super().__init__(**kwargs)

    def checkpoint_property(self) -> None:
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        save_dict = {
            self.checkpoint_name: {
                'progress_flag': self.progress_flag,
                'current_state': self.current_state,
                'elec_lambda_calcs': self.elec_lambda_calcs,
                'vdw_lambda_calcs': self.vdw_lambda_calcs,
                'vacuum_lambda_calcs': self.vacuum_lambda_calcs,
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

    def restart_property(self) -> None:
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.progress_flag = restart_dict.get('progress_flag',
                                              self.progress_flag)
        self.current_state = restart_dict.get('current_state',
                                              self.current_state)
        self.elec_lambda_calcs = restart_dict.get('elec_lambda_calcs',
                                                  self.elec_lambda_calcs)
        self.vdw_lambda_calcs = restart_dict.get('vdw_lambda_calcs',
                                                 self.vdw_lambda_calcs)
        self.vacuum_lambda_calcs = restart_dict.get('vacuum_lambda_calcs',
                                                    self.vacuum_lambda_calcs)

        # ADD THIS STUFF LATER FOR PROGRESS TRACKING IN A WAY THAT MAKES SENSE

        # if len(self.elec_lambda_calcs) > 0:
        #     self.outstanding_npt = self.npt_calcs[-1]
        # if len(self.nph_calcs) > 0:
        #     self.outstanding_nph = self.nph_calcs[-1]
        # if self.progress_flag != 'init':
        #     if self.progress_flag == 'done':
        #         # restart information exists but the last calculation ended
        #         self.restart = False
        #         self.npt_calcs = []
        #         self.nph_calcs = []
        #     else:
        #         # restart information exists and we actually want to restart
        #         self.restart = True
        # else:
        #     # no restart information exists
        #     self.restart = False

    def calculate_property(
        self,
        # modified_params: Optional[Dict[str, Any]] = None,
        path_type: str,
        sim_params: dict,
        solvation_params: dict,
        executables: None,
        iter_num: int = 0,
        random_seed_use: bool = False,
        model_path: str = None,
        potential: Optional[Union[str,
                                  Potential]] = None,  # NOT CURRENTLY USED!!
        scheduler: Optional[Scheduler] = None,
        storage: Optional[Storage] = None,
        **kwargs,
    ):
        """
        Determine the solvation free energy of a solute in a pure
        or multi-component solvent using Thermodynamic Integration (TI).
        This is computed by turning an initially solvated system into a
        system of solvent + ideal gas solute. Specifically, this is done by
        first turning off solute charges, then turning off van der Waals
        solute-solvent interactions, and finally turning the ideal gas
        solute charges back on (now in the absence of solvent).

        This pipeline uses packmol and moltemplate to generate appropriate
        initial configurations of the solvated system. For now, the packing,
        minimization, & equilibration steps are required.

        :param path_type: path to perform solvation free energy calculations
        :type path_type: str
        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :solvation_params:                   # NEED TO UPDATE THIS LINE!!
        :type solvation_params: dict
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: boolean
        :param model_path: path to store the potential file
        :type model_path: str
        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: a dictionary with property output, errors, and calc ids as
            a tuple (different indices can correspond to different calc types)
        :rtype: dict
        """

        # ensure restart is properly read
        self.restart_property()

        # Get default scheduler if one is not provided & get its root_dir
        if scheduler is None:
            scheduler = self.default_scheduler
        root_dir = scheduler.root_directory

        # Make path for coefficient files to be generated, stored, & copied
        if model_path is None:
            model_path = f'{os.path.realpath(root_dir)}/model/'
        os.makedirs(model_path, exist_ok=True)

        # Validate all inputs for phsyical constraints, override sim_params
        # and solvation_params with completed versions and check that
        # executables exist and are executable
        sim_params, solvation_params = self._validate_inputs(
            sim_params, solvation_params, executables)

        # Check for /omp pairstyle
        pair_style = sim_params.get('pair_style')
        if pair_style.split('/')[-1] == 'omp':
            self.built_simulator.code_path += ' -sf omp'

        # --- Packing procedure!! ----

        # Define parameters needed for packing
        self.logger.info('\nAttempting to pack solvated system...')
        pack_tol = solvation_params.get('pack_tol')
        pack_boxlen = solvation_params.get('pack_boxlen')
        packmol_exe = executables.get('packmol')

        # Make path for packing
        self.pack_dir = f'{os.path.realpath(root_dir)}/pack/{iter_num}/'
        sim_params['pack_dir'] = self.pack_dir
        pack_inp = f'{self.pack_dir}/system_packed.inp'
        self.logger.info(f'\tmaking packing directory, {self.pack_dir}')
        os.makedirs(self.pack_dir, exist_ok=True)

        # Make list that includes both solute and solvents
        solute_params = solvation_params.get('solute')
        solvent_params = solvation_params.get('solvent')
        self.logger.info(f'\tcreating list of molecules in system')
        solute_entry = {
            **solute_params, 'num_molecs': solute_params.get('num_molecs', 1)
        }
        molec_list = [solute_entry] + solvent_params

        self.logger.info(f'\twriting packmol input file...')
        with open(pack_inp, 'w') as f:
            f.write(f'tolerance {pack_tol}\noutput system_packed.pdb\n'
                    'filetype pdb\n\n')
            for i, molec in enumerate(molec_list):
                pdb = molec.get('pdb')
                num_molecs = molec.get('num_molecs')
                f.write(f'structure {pdb}\n'
                        f'  number {num_molecs}\n'
                        f'  inside cube 0.0 0.0 0.0 {pack_boxlen}\n'
                        f'end structure\n')

        self.logger.info(f'\tfinished writing packmol input script,'
                         'running packmol...')

        with open(pack_inp) as stdin_file:
            self.logger.info(
                f"Running: {packmol_exe} < {pack_inp}  (in {self.pack_dir})")
            result = subprocess.run(
                packmol_exe,
                stdin=stdin_file,
                cwd=self.pack_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.logger.error(
                    f"Packmol failed with return code {result.returncode}")
                self.logger.error(result.stdout)
                self.logger.error(result.stderr)
                raise RuntimeError(
                    f"Packmol failed (return code {result.returncode});"
                    "see log for details")

        self.logger.info(f'Finished packing system!')

        # --- Packing Complete!! ---

        # --- Run moltemplate now ---
        self.logger.info(
            f'\nUsing moltemplate to generate parameter file & packed_system.data'
        )

        # Check that moltemp executable exists & is executable
        moltemp_exe = executables.get('moltemp')
        if moltemp_exe is None:
            raise ValueError(
                "moltemp_exe must be specified in solvation_params")
        if shutil.which(moltemp_exe) is None:
            raise ValueError(
                f"moltemp executable not found or not executable: {moltemp_exe}"
            )

        self.logger.info(f'\tgathering .lt files')
        # Copy over lt files into , saving solvent as solv{i}.{name of lt}.lt
        # and solute as solu.{name of lt}, checking that each is found
        lt_header = []
        lt_body = []
        for i, mol in enumerate(molec_list):
            lt = mol.get('lt')
            if lt is None or not isfile(lt):
                raise ValueError(
                    f".lt file {lt} for does not exist or was not provided.")
            num_molecs = mol.get('num_molecs')
            mol_class = mol.get('class', 'unknown')
            if mol_class == 'unknown':
                raise ValueError(F"unknown class in .lt file for file {lt}")
            name = f'comp_{i}'
            lt_name = f'{name}.{os.path.basename(lt)}'
            shutil.copy(mol["lt"], f"{self.pack_dir}/{lt_name}")
            subprocess.run([
                "sed", "-i", f"s/{mol_class}/{name}/g",
                f"{self.pack_dir}/{lt_name}"
            ])
            lt_header.append(f'import "{lt_name}"\n')
            lt_body.append(f'{name} = new {name}[{num_molecs}]\n')

        self.logger.info(f'\twriting combined system_packed.lt file')
        with open(f'{self.pack_dir}/system_packed.lt', "w") as f:
            f.writelines(lt_header)
            f.write('\n')
            f.writelines(lt_body)
            f.write('\n')
            f.write('write_once("Data Boundary"){\n')
            f.write(f'  0 {pack_boxlen} xlo xhi\n')
            f.write(f'  0 {pack_boxlen} ylo yhi\n')
            f.write(f'  0 {pack_boxlen} zlo zhi\n')
            f.write('}\n\n')

        # Get atom style, so that moltemplate will write the output file correctly
        atom_style = sim_params.get('atom_style')
        cmd = [
            moltemp_exe, "-pdb", "system_packed.pdb", "-atomstyle", atom_style,
            "system_packed.lt"
        ]
        self.logger.info(
            f"\trunning moltemplate: {' '.join(cmd)}  (in {self.pack_dir})")
        result = subprocess.run(
            cmd,
            cwd=self.pack_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(
                f"Moltemplate failed with return code {result.returncode}")
            self.logger.error(result.stdout)
            self.logger.error(result.stderr)
            raise RuntimeError(
                f"Moltemplate failed (return code {result.returncode});"
                "see log for details")

        # Running cleanup_moltemplate.sh, assumed to be in the same directory
        # as moltemplate.sh
        moltemp_cleanup_exe = executables.get('moltemp_cleanup')
        if moltemp_cleanup_exe is None:
            raise ValueError(
                "moltemp_cleanup_exe must be specified in solvation_params")
        if shutil.which(moltemp_cleanup_exe) is None:
            raise ValueError(
                "moltemp_cleanup_exe executable not found or not executable:"
                f"{moltemp_cleanup_exe}")

        # Run moltemplate_cleanup.sh
        result = subprocess.run(
            [moltemp_cleanup_exe, "-base", "system_packed"],
            cwd=self.pack_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(
                f"Moltemplate cleanup failed with return code {result.returncode}"
            )
            self.logger.error(result.stdout)
            self.logger.error(result.stderr)
            raise RuntimeError(
                f"Moltemplate failed (return code {result.returncode});"
                "see log for details")

        # --- moltemplate complete! ---

        # --- Process parameters, charges, and build required input blocks
        # before starting any simulations ---

        # Read the .in.init file & check for difference with passed sim_params
        init_args, init_ps_args = self._parse_init_file(
            f"{self.pack_dir}/system_packed.in.init")
        for style in init_args.keys():
            if style in sim_params.keys(
            ) and sim_params[style] != init_args[style]:
                self.logger.warning(
                    f"\ndifference found in passed sim_param and .lt file for {style} "
                    f"\npassed style: {sim_params[style]},"
                    f".in.init style: {init_args[style]}"
                    f"\ntaking .in.init value instead? -> {self.override_with_init}"
                )
                if self.override_with_init:
                    sim_params[style] = init_args[style]

        ps_args = sim_params.get("pair_style_args")
        for arg in init_ps_args.keys():
            if arg in ps_args.keys() and ps_args[arg] != init_ps_args[arg]:
                self.logger.warning(
                f"\ndifference found in passed pair_style argument {arg} " \
                f"\npassed style: {ps_args[arg]}, .in.init style: {init_ps_args[arg]}" \
                f"\ntaking .in.init value instead? -> {self.override_with_init}" \
                )
                if self.override_with_init:
                    sim_params["pair_style_args"][arg] = init_ps_args[arg]

        # Read coefficients writeen by moltemplate
        self.logger.info(
            f'\tparsing pair/bond/angle/dihedral/improper coefficients from moltemplate .in.settings'
        )
        pack_settings_path = f'{self.pack_dir}/system_packed.in.settings'
        pair_coeffs, other_coeffs = self._parse_settings_file(
            pack_settings_path)
        mix_rule = sim_params.get('pair_modify').split()[-1]

        # Parse solute vs. solvent types and charges
        self.logger.info(
            f'\tparsing atom types and charges from system_packed.data')
        solvent_types, solute_types, all_charges = self._parse_types_and_charges_from_data(
            f'{self.pack_dir}/system_packed.data', 1, atom_style, self.logger)
        charge_file = f'{self.pack_dir}/system_packed.in.charges'
        if isfile(charge_file):
            self.logger.info(
                f'\tfound {charge_file}, overriding charges from Data Atoms')
            _, chg_file_charges = self._parse_charges_file(charge_file)
            self.logger.info(
                f'\t\t{len(chg_file_charges)} charge override(s) parsed: {chg_file_charges}'
            )
            all_charges = {**all_charges, **chg_file_charges}
        else:
            self.logger.info(
                f'\tno {charge_file} found, using charges from Data Atoms only'
            )

        # Replace .lt placeholder with 0's & check if ANY charges are present
        has_charges = any(abs(q) > 0.01 for q in all_charges.values())
        self.logger.info(f'\thas_charges: {has_charges}')
        if has_charges:
            charge_params = self._build_charge_modify(
                sorted(set(solute_types + solvent_types)), all_charges)
            sim_params = {**sim_params, **charge_params}

        # Check if solute is charged -> If so, need to run TI_ELEC & TI_VDW
        charged_solute = False
        if has_charges:
            for solu in solute_types:
                if all_charges[solu] > 1e-8:
                    charged_solute = True
        sim_params['charged_solute'] = charged_solute

        self.logger.info(f'\tcharged_solute: {charged_solute}')
        if has_charges and not charged_solute:
            self.logger.warning(
                '\tSystem has charges, but the solute itself appears '
                'uncharged. TI_ELEC leg may be unnecessary — check solute_types '
                f'{sorted(solute_types)} against all_charges.')

        # Prepare pair_style_args_str & soft_pair_style_args from pair_style_args,
        # then add them to sim_params
        main_pair_style = sim_params.get('pair_style')
        pair_style_args = sim_params.get('pair_style_args')
        _, soft_pair_style = self._resolve_soft_style(main_pair_style)
        sim_params['pair_style_args_str'] = self._resolve_pair_style_args(
            main_pair_style, pair_style_args)
        sim_params['soft_pair_style_args_str'] = self._resolve_pair_style_args(
            soft_pair_style, pair_style_args)
        sim_params['soft_pair_style'] = soft_pair_style

        # Write pretty version of system_packed.in.settings (pair_coeffs)
        main_settings_file = f'{model_path}/system_packed.in.settings'
        self._write_settings_file("elec", pair_coeffs, other_coeffs,
                                  solvent_types, solute_types, None, mix_rule,
                                  main_pair_style, soft_pair_style,
                                  main_settings_file)

        # Write complete version of system_packed.in.charges
        main_charge_file = f'{model_path}/system_packed.in.charges'
        self._write_charge_file(sorted(set(solute_types + solvent_types)),
                                all_charges, main_charge_file)

        self.logger.info(f'\tbuilding vdW soft-core modification lines')
        vdw_params = self._build_vdw_modification_lines(
            solute_types, solvent_types, soft_pair_style)
        sim_params = {**sim_params, **vdw_params}

        # --- End of processing parameters ---

        # --- Start of simulation procedures ---

        # --- Minimization procedure!! ---
        self.logger.info(f'\nAttempting to run minimization job')
        min_job = self._conduct_sim(sim_params, [
            main_settings_file, main_charge_file,
            f'{self.pack_dir}/system_packed.data'
        ], scheduler, path_type + f'/{iter_num}/min', "min", random_seed_use)
        scheduler.block_until_completed(min_job)
        # min_status = scheduler.update_job_status([min_job])
        self.min_dir = os.path.realpath(scheduler.get_job_path(min_job))

        # --- End of minimization ---

        # --- Equilibration procedure!! ---

        if not has_charges and sim_params.get('kspace_style') != "none":
            sim_params['kspace_style'] = "none"
            self.logger.warning(
                'no charges in system, kspace_style has been changed to none.')

        self.logger.info('\nAttempting to run equilibration job')
        equil_job = self._conduct_sim(
            sim_params, 
            [
                main_settings_file, 
                main_charge_file,
                f'{self.min_dir}/minimized.data'
            ], 
            scheduler, 
            path_type + f'/{iter_num}/equil', 
            "equil",
            random_seed_use
        )
        scheduler.block_until_completed(equil_job)
        self.equil_dir = os.path.realpath(scheduler.get_job_path(equil_job))

        # --- End of equilibration ---

        # --- TI procedural setup ---
        self.logger.info('\nGathering info for TI jobs')
        lambda_values = solvation_params.get('lambda_values')
        lambda_diff = solvation_params.get('lambda_diff')
        sim_params['lambda_diff'] = lambda_diff
        self.logger.info(f'\tusing lambda_diff: {lambda_diff}')

        # For tracking jobs
        lam_jobs = []
        log_files = {'ti_elec': [], 'ti_vdw': [], 'ti_vacuum': []}

        # --- TI Electronic Jobs ---
        if charged_solute:
            self.logger.info('\nPreparing & running TI_ELEC jobs')
            for lam in lambda_values:
                sim_params['lambda_c'] = lam
                sim_params['lambda_vdw'] = 1.00
                lam_job = self._conduct_sim(
                    sim_params, [
                        main_settings_file, main_charge_file,
                        f"{self.equil_dir}/npt_equil.data"
                    ], scheduler,
                    path_type + f'/{iter_num}/ti_elec/lambda_{lam:.5f}',
                    'ti_elec', random_seed_use)
                jobpath = scheduler.get_job_path(lam_job)
                log_files['ti_elec'].append(f'{jobpath}/lammps.out')
                if lam == 0:
                    self.no_charge_dir = scheduler.get_job_path(lam_job)
                lam_jobs.append(lam_job)
            scheduler.block_until_completed(lam_jobs)
        else:
            self.logger.info('\nno charge file -> skipping TI_ELEC jobs')

        # --- TI van der Waals ---
        self.logger.info('\nPreparing & running TI_VDW jobs')
        for lam in lambda_values:
            sim_params['lambda_c'] = 0.00
            sim_params['lambda_vdw'] = lam
            sim_params['lambda_vdw_str'] = f'{lam:.5f}'
            vdw_settings_file = f'{model_path}/ti_vdw_{lam:.5f}.in.settings'
            self._write_settings_file("vdw", pair_coeffs, other_coeffs,
                                      solvent_types, solute_types, lam,
                                      mix_rule, main_pair_style,
                                      soft_pair_style, vdw_settings_file)
            lam_job = self._conduct_sim(
                sim_params, [
                    vdw_settings_file, main_charge_file,
                    f"{self.equil_dir}/npt_equil.data",
                    f"{self.no_charge_dir}/ti_elec.restart"
                ], scheduler,
                path_type + f'/{iter_num}/ti_vdw/lambda_{lam:.5f}', 'ti_vdw',
                random_seed_use)
            jobpath = scheduler.get_job_path(lam_job)
            log_files['ti_vdw'].append(f'{jobpath}/lammps.out')
            lam_jobs.append(lam_job)
        scheduler.block_until_completed(lam_jobs)
        # Don't have to wait for VDW to run VACUUM

        # --- TI electronic solute in vacuum ---
        if charged_solute:
            self.logger.info('\nPreparing & running TI_VACUUM jobs')
            for lam in lambda_values:
                sim_params['lambda_c'] = lam
                sim_params['lambda_vdw'] = 1.00
                lam_job = self._conduct_sim(
                    sim_params, [
                        main_settings_file, main_charge_file,
                        f"{self.equil_dir}/npt_equil.data"
                    ], scheduler,
                    path_type + f'/{iter_num}/ti_vacuum/lambda_{lam:.5f}',
                    'ti_vacuum', random_seed_use)
                jobpath = scheduler.get_job_path(lam_job)
                log_files['ti_vacuum'].append(f'{jobpath}/lammps.out')
                lam_jobs.append(lam_job)
        else:
            self.logger.info('\nno charge file -> skipping TI_VACCUM jobs')
        scheduler.block_until_completed(lam_jobs)

        # Make analysis directory
        analysis_dir = f'{root_dir}/analysis'
        os.makedirs(analysis_dir, exist_ok=True)

        # Analyze results of leg
        dg_total = 0
        for leg in ('elec', 'vdw', 'vacuum'):
            dudl, dudl_std, dudl_err = [], [], []
            for lam, f in zip(lambda_values, log_files[f"ti_{leg}"]):
                time_series, _, avg, std = \
                    AnalyzeLammpsLog.extract_property([f, 'f_dudl_avg'])
                dudl.append(avg)
                dudl_std.append(std)
                dudl_err.append(std / np.sqrt(len(time_series)))
            lams, dudl, dudl_std, dudl_err = self._write_results_file(
                lambda_values, dudl, dudl_std, dudl_err,
                f'{analysis_dir}/results_{leg}.dat')

            # Integrate dudl for each leg
            dg_leg = np.trapezoid(dudl, lams)
            if leg in ("vdw", "elec"):
                dg_total += dg_leg
            else:
                dg_total -= dg_leg

        # Return results (finally!)
        results_dict = {
            'property_value': dg_total,
            'property_std': None,
            'calc_ids': (min_job, equil_job, lam_jobs),
            'success': True,
        }

        return results_dict

    def _conduct_sim(
        self,
        sim_params: Dict[str, Any],
        sim_files: Union[str, list[str], None],
        scheduler: Scheduler,
        sim_path: str,
        sim_type: str,  # ("min", "equil", "ti_elec", "ti_vdw", "ti_vacuum")
        random_seed_use: bool
    ) -> int:
        """
        Perform the simulation for the target property calculations

        sim_params is a dictionary of key-value pairs that can
        be used to define various parameters related to conducting
        simulations (e.g. temperature, pressure, random seed, etc..).
        The dictionary is described in the input json file.

        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :param sim_files: simulation files that are needed in the path to
        # be included
        :type sim_files: str, list, or None
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :param sim_path: path to perform simulations for
            target property calculations
        :type sim_path: str
        """

        template_fill = sim_params

        # Seed for calculation
        random_seed = random.randint(
            1, 10000) if random_seed_use else self.default_seed
        template_fill['random_seed'] = random_seed

        # If no charges are present, use kspace_modify = none
        if sim_type in ('ti_elec',
                        'ti_vacuum') and sim_params['lambda_c'] == 0:
            template_fill['kspace_modify'] = "none"

        if sim_type == "min":
            self.logger.info("Attempting to run minimization.")
            selected_job_details = self.min_job_details
        elif sim_type == "equil":
            self.logger.info("Attempting to run equilibration")
            selected_job_details = self.equil_job_details
        elif sim_type == "ti_elec":
            self.logger.info("Attempting to run TI_ELEC")
            selected_job_details = self.ti_job_details
        elif sim_type == "ti_vdw":
            self.logger.info("Attempting to run TI_VDW")
            selected_job_details = self.ti_job_details
        elif sim_type == "ti_vacuum":
            self.logger.info("Attempting to run TI_VAC")
            selected_job_details = self.ti_job_details
        else:
            raise ValueError(
                f"unknown simulation type: {sim_type}, expected one of:"
                "('min', 'equil', 'ti_elec', 'ti_vdw', 'ti_vacuum')")

        calc_id = self.built_simulator.run(
            sim_path,
            sim_files,
            template_fill,
            input_template=self.input_template[sim_type],
            scheduler=scheduler,
            job_details=selected_job_details)
        return calc_id

    def calculate_with_error(
        self,
        n_calc: int,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        scheduler: Optional[Scheduler] = None,
    ):
        """
        Calculate a target property with mean and standard deviation
        Derived classes should list explicit arguments required
        to calculate their properties.

        Mean and standard deviation will be obtained from multiple
        number of calculations (n_calc)

        :param n_calc: total number of calculations to perform
        :type n_calc: int
        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :returns: mean and standard deviation of the calculated property
        """
        pass

    def validate_sim_inputs(self):
        """
        Use this function to verify that ALL inputs are physically correct.
        Wouldn't want to waste our time preparing a system that's going to
        error out down the line anyhow!
        """
        pass

    def _resolve_soft_style(self, base_pair_style: str) -> str:
        """
        Look up the FEP soft-core pair style for a base pair_style.
        """
        fep_soft_map = {
            "lj/cut": {
                "elec": "",
                "vdw": "lj/cut/soft"
            },
            "lj/cut/coul/cut": {
                "elec": "lj/cut/coul/cut",
                "vdw": "lj/cut/soft"
            },
            "lj/cut/coul/long": {
                "elec": "lj/cut/coul/long",
                "vdw": "lj/cut/soft"
            },
            "lj/charmm/coul/long": {
                "elec": "lj/charmm/coul/long",
                "vdw": "lj/cut/soft"
            },
            "lj/class2/coul/cut": {
                "elec": "lj/class2/coul/cut",
                "vdw": "lj/class2/soft"
            },
            "lj/class2/coul/long": {
                "elec": "lj/class2/coul/long",
                "vdw": "lj/class2/soft"
            },
        }

        use_omp = base_pair_style.endswith('/omp')
        if use_omp:
            lookup_style = base_pair_style[:-len('/omp')]
        else:
            lookup_style = base_pair_style

        styles = fep_soft_map.get(lookup_style)
        if styles is None:
            raise ValueError("No FEP soft-core mapping for pair_style"
                             f"{base_pair_style}."
                             f"Supported base styles: {sorted(fep_soft_map)}")

        def _with_omp(style):
            if not style:
                return style
            return f"{style}/omp" if use_omp else style

        elec = _with_omp(styles["elec"])
        vdw = _with_omp(styles["vdw"])

        return elec, vdw

    def _resolve_pair_style_args(self, pair_style, pair_style_args):
        """
        Resolve the ordered pair_style argument values for a given LAMMPS
        pair_style, based on a lookup of required argument names.

        :param pair_style: LAMMPS pair_style name, with or without an '/omp'
            suffix
        :type pair_style: str
        :param pair_style_args: dict mapping argument name -> value for this
            pair_style (e.g. {'cutoff': 12.0, 'n': 2})
        :type pair_style_args: dict
        :returns: space-separated string of argument values in the order
            required by the pair_style
        :rtype: str
        :raises ValueError: if the pair_style is unrecognized or a required
            argument is missing from pair_style_args
        """

        pair_style_args_map = {
            "lj/cut": ["cutoff"],
            "lj/cut/soft": ["softcore_n", "alpha_vdw", "cutoff"],
            "lj/cut/coul/long": ["cutoff", "cutoff2"],
            "lj/cut/coul/long/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "coul/long": ["cutoff"],
            "coul/long/soft": ["softcore_n", "alpha_elec", "cutoff"],
            "coul/cut": ["cutoff"],
            "coul/cut/soft": ["softcore_n", "alpha_elec", "cutoff"],
            "lj/cut/coul/cut": ["cutoff", "cutoff2"],
            "lj/cut/coul/cut/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "lj/charmm/coul/long": ["inner", "outer", "cutoff"],
            "lj/charmm/coul/long/soft": [
                "softcore_n", "alpha_vdw", "alpha_elec", "inner", "outer",
                "cutoff"
            ],
            "lj/class2": ["cutoff"],
            "lj/class2/coul/cut": ["cutoff", "cutoff2"],
            "lj/class2/coul/long": ["cutoff", "cutoff2"],
            "lj/class2/soft": ["softcore_n", "alpha_vdw", "cutoff"],
            "lj/class2/coul/cut/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "lj/class2/coul/long/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
        }

        lookup_style = pair_style[:-len('/omp')] if pair_style.endswith(
            '/omp') else pair_style

        required_args = pair_style_args_map.get(lookup_style)
        if required_args is None:
            raise ValueError(
                f"No pair_style argument mapping for pair_style "
                f"``{pair_style}``. "
                f"Supported base styles: {sorted(pair_style_args_map)}")

        values = []
        skipped_arg = None
        for arg_name in required_args:
            value = pair_style_args.get(arg_name)
            is_missing = value is None or (isinstance(value, str)
                                           and value.lower() == 'none')

            if is_missing:
                skipped_arg = arg_name
                self.logger.warning(
                    f"Skipping pair_style arg '{arg_name}' for pair_style "
                    f"'{pair_style}' - presumably optional argument to LAMMPS "
                    f"(value was {value!r})")
                continue

            if skipped_arg is not None:
                raise ValueError(
                    f"pair_style arg '{arg_name}' was provided for "
                    f"pair_style '{pair_style}', but an earlier "
                    f"required arg '{skipped_arg}' was missing/none. "
                    f"LAMMPS pair_style args are positional, so gaps "
                    f"aren't allowed — all args after a skipped one "
                    f"must also be skipped.")

            values.append(str(value))

        return ' '.join(values)

    def _lookup_pair_style_args(self, pair_style):
        """
        Look up the ordered list of required argument names for a given
        LAMMPS pair_style.

        :param pair_style: LAMMPS pair_style name, with or without an '/omp'
            suffix
        :type pair_style: str
        :returns: ordered list of required argument names for the pair_style
        :rtype: list[str]
        :raises ValueError: if the pair_style is unrecognized
        """

        pair_style_args_map = {
            "lj/cut": ["cutoff"],
            "lj/cut/soft": ["softcore_n", "alpha_vdw", "cutoff"],
            "lj/cut/coul/long": ["cutoff", "cutoff2"],
            "lj/cut/coul/long/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "coul/long": ["cutoff"],
            "coul/long/soft": ["softcore_n", "alpha_elec", "cutoff"],
            "coul/cut": ["cutoff"],
            "coul/cut/soft": ["softcore_n", "alpha_elec", "cutoff"],
            "lj/cut/coul/cut": ["cutoff", "cutoff2"],
            "lj/cut/coul/cut/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "lj/charmm/coul/long": ["inner", "outer", "cutoff"],
            "lj/charmm/coul/long/soft": [
                "softcore_n", "alpha_vdw", "alpha_elec", "inner", "outer",
                "cutoff"
            ],
            "lj/class2": ["cutoff"],
            "lj/class2/coul/cut": ["cutoff", "cutoff2"],
            "lj/class2/coul/long": ["cutoff", "cutoff2"],
            "lj/class2/soft": ["softcore_n", "alpha_vdw", "cutoff"],
            "lj/class2/coul/cut/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
            "lj/class2/coul/long/soft":
            ["softcore_n", "alpha_vdw", "alpha_elec", "cutoff", "cutoff2"],
        }

        lookup_style = pair_style[:-len('/omp')] if pair_style.endswith(
            '/omp') else pair_style

        required_args = pair_style_args_map.get(lookup_style)
        if required_args is None:
            raise ValueError(
                "No pair_style argument mapping for pair_style "
                f"'{pair_style}'. "
                f"Supported base styles: {sorted(pair_style_args_map)}")

        return required_args

    def _parse_types_and_charges_from_data(
        self,
        data_file_path,
        solute_molecule_id=1,
        atom_style="full",
        logger=None,
    ):
        """
        Parse atom types from a LAMMPS data file.

        Supports 'full' (atom-ID mol-ID atom-type charge x y z) and 'molecular'
        (atom-ID mol-ID atom-type x y z) atom styles.

        :returns: solvent_types (set), solute_types (set),
            solute_charges (dict,empty if atom_style has no charge column)
        """
        log = logger.info if logger is not None else print

        if atom_style not in ("full", "molecular"):
            raise ValueError(
                f"parse_atom_types_from_data only supports 'full' or "
                f"'molecular' atom_style, got '{atom_style}'")
        has_charge = atom_style == "full"

        solute_types = set()
        solvent_types = set()
        all_charges = {}
        num_atoms = None

        with open(data_file_path) as f:
            lines = f.readlines()

        # Find declared atom count from the header, e.g. "1001 atoms"
        for line in lines:
            parts = line.split("#")[0].split()
            if len(parts) >= 2 and parts[1] == "atoms":
                num_atoms = int(parts[0])
                break

        if num_atoms is None:
            raise ValueError(
                f"Could not find 'N atoms' header line in {data_file_path}")

        # Find the Atoms section and read exactly num_atoms non-blank lines
        atoms_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Atoms"):
                atoms_start = i + 1
                break

        if atoms_start is None:
            raise ValueError(f"No 'Atoms' section found in {data_file_path}")

        parsed = 0
        i = atoms_start
        while parsed < num_atoms and i < len(lines):
            stripped = lines[i].split("#")[0].strip()
            i += 1
            if not stripped:
                continue

            parts = stripped.split()
            mol_id = int(parts[1])
            atom_type = int(parts[2])
            if has_charge:
                all_charges[atom_type] = float(parts[3])

            if mol_id == solute_molecule_id:
                solute_types.add(atom_type)
            else:
                solvent_types.add(atom_type)
            parsed += 1

        if parsed < num_atoms:
            raise ValueError(
                f"Expected {num_atoms} atoms but only parsed {parsed} in "
                f"{data_file_path}")
        if not solute_types:
            raise ValueError(
                f"No atoms found for solute molecule ID {solute_molecule_id}. "
                f"Check solute_molecule_id in ti_settings.")
        if not solvent_types:
            raise ValueError("No solvent atoms found. Check your data file.")

        log(f"Parsed atom types from {os.path.basename(data_file_path)}:")
        log(f"  Solute  (molecule {solute_molecule_id}):"
            f"{sorted(solute_types)}")
        log(f"  Solvent (all others): {sorted(solvent_types)}")

        return sorted(solvent_types), sorted(solute_types), all_charges

    def _parse_charges_file(self, charge_file_path):
        types = set()
        charges = {}
        with open(charge_file_path) as f:
            for line in f:
                stripped = line.strip().split()
                if not stripped:
                    continue
                type_id = int(stripped[2])
                types.add(type_id)
                charge_val = float(stripped[4])  # 5th column (index 4)
                charges[type_id] = charge_val
        return types, charges

    def _build_charge_modify(self, solute_types, all_charges):

        charge_modification_block = ""
        delta_charge_block = ""
        compute_fep_forward = ""
        compute_fep_backward = ""

        prefix = "                "
        for type in solute_types:
            charge = all_charges[type]
            suffix = '\n' if (type == max(solute_types)) else ' &\n'
            if type in solute_types:
                charge_modification_block += (
                    f'set type {type} charge $( {charge} * v_lambda_c )\n')
                delta_charge_block += (f'variable        dq{type} equal '
                                       f'{charge}*(v_lambda_diff)\n')
                delta_charge_block += (f'variable        ndq{type}      equal '
                                       f'{charge}*(v_nlambda_diff)\n')
                compute_fep_forward += (
                    f'{prefix}atom charge {type} v_dq{type}{suffix}')
                compute_fep_backward += (
                    f'{prefix}atom charge {type} v_ndq{type}{suffix}')

        return {
            "charge_modification_block": charge_modification_block,
            "delta_charge_block": delta_charge_block,
            "fep_elec_forward": compute_fep_forward,
            "fep_elec_backward": compute_fep_backward
        }

    def _write_charge_file(self, all_types, all_charges, output_file):
        """
        Write a plain LAMMPS settings file of 'set type <type> charge <val>'
        lines for every atom type, based on parsed charges. This file can
        be `include`d in any LAMMPS input script (min, equil, ti_elec,
        ti_vdw, ti_vacuum) to apply the same base charges consistently,
        regardless of whether those charges originally came from
        Data Atoms or a .in.charges override.

        :param all_types: iterable of all atom types (solute + solvent)
        :type all_types: iterable
        :param all_charges: dict {atom_type: charge}
        :type all_charges: dict
        :param output_file: path to write the charge-setting file to
        :type output_file: str
        :returns: output_file, for convenience when chaining into sim_params
        :rtype: str
        """
        lines = []
        for atom_type in sorted(all_types):
            charge = all_charges[atom_type]
            lines.append(f"set type {atom_type} charge {charge}\n")

        with open(output_file, "w") as f:
            f.writelines(lines)
        self.logger.info(f"\tWrote {os.path.basename(output_file)}")

        return output_file

    def _build_vdw_modification_lines(self, solute_types, solvent_types,
                                      soft_pair_style):

        compute_fep_forward = ""
        compute_fep_backward = ""

        prefix = "                "
        for i, solu in enumerate(solute_types):
            for j, solv in enumerate(solvent_types):
                suffix = '\n' if (i == len(solute_types) - 1) and (
                    j == len(solvent_types) - 1) else ' &\n'
                compute_fep_forward += (
                    f'{prefix}pair {soft_pair_style} '
                    f'lambda {solu} {solv} v_lambda_diff{suffix}')
                compute_fep_backward += (
                    f'{prefix}pair {soft_pair_style} '
                    f'lambda {solu} {solv} v_nlambda_diff{suffix}')

        return {
            "fep_vdw_forward": compute_fep_forward,
            "fep_vdw_backward": compute_fep_backward
        }

    def _parse_settings_file(self, settings_path):
        """
        Parse the .in.settings from equilibrium data and extract:
        - like-like pair coefficients (epsilon, sigma) for each atom type
        - bond, angle, dihedral, improper coefficients (stored as raw strings)

        Returns:
            pair_coeffs : dict  {atom_type: (epsilon, sigma)}
            other_coeffs: dict  {"bond": [...], "angle": [...],
                                "dihedral": [...], "improper": [...]}
        """
        pair_coeffs = {}  # {atom_type: (epsilon, sigma)}
        other_coeffs = {
            "bond": [],
            "angle": [],
            "dihedral": [],
            "improper": [],
        }

        with open(settings_path) as f:
            for line in f:
                # Strip inline comments and whitespace
                clean = line.split("#")[0].strip()
                if not clean:
                    continue
                parts = clean.split()
                keyword = parts[0]
                if keyword == "pair_coeff":
                    i = int(parts[1])
                    j = int(parts[2])
                    if i == j:
                        epsilon = float(parts[3])
                        sigma = float(parts[4])
                        pair_coeffs[i] = (epsilon, sigma)
                elif keyword == "bond_coeff":
                    other_coeffs["bond"].append(clean)
                elif keyword == "angle_coeff":
                    other_coeffs["angle"].append(clean)
                elif keyword == "dihedral_coeff":
                    other_coeffs["dihedral"].append(clean)
                elif keyword == "improper_coeff":
                    other_coeffs["improper"].append(clean)
        if not pair_coeffs:
            raise ValueError(
                f"No direct pair_coeff entries found in {settings_path}")

        return pair_coeffs, other_coeffs

    def _parse_init_file(self, init_path):
        styles = {}
        pair_style_args = {}
        with open(init_path) as f:
            for line in f:
                clean = line.split("#")[0].strip().split()
                if (not clean) or (len(clean) < 2):
                    continue
                if clean[0] == "pair_style":
                    styles["pair_style"] = clean[1]
                    if len(clean) > 2:
                        required_args = self._lookup_pair_style_args(
                            styles["pair_style"])
                        for arg, value in zip(required_args, clean[2:]):
                            pair_style_args[arg] = value
                else:
                    styles[clean[0]] = " ".join(clean[1:])

        return styles, pair_style_args

    def _compute_mixed_params(self, eps_i, sig_i, eps_j, sig_j, mix_rule):
        """
        Compute epsilon_{ij} and sigma_{ij} using user specified mixing rule.

        Arithmetic:
            epsilon_ij = sqrt(eps_i * eps_j)
            sigma_ij   = 0.5 * (sig_i + sig_j)
        Geometric:
            epsilon_ij = sqrt(eps_i * eps_j)
            sigma_ij   = sqrt(sig_i * sig_j)
        """
        eps_ij = np.sqrt(eps_i * eps_j)

        if mix_rule == "arithmetic":
            sig_ij = 0.5 * (sig_i + sig_j)
        elif mix_rule == "geometric":
            sig_ij = np.sqrt(sig_i * sig_j)
        else:
            raise ValueError(f"Unknown mix rule: '{mix_rule}'. "
                             f"Use 'arithmetic' or 'geometric'.")

        if (eps_ij != 0.0) or (sig_ij != 0.0):
            return eps_ij, sig_ij
        else:
            return eps_ij, 1.00

    def _write_settings_file(self, leg, pair_coeffs, other_coeffs,
                             solvent_types, solute_types, lam, mix_rule,
                             main_pair_style, soft_pair_style, output_path):
        """
        Compute all cross terms using the mixing rule and write a TI
        .in.settings file with explicit pair coefficients.

        Same-molecule pairs:      main_pair_style (e.g. lj/cut/coul/long/omp)
        Cross-molecule LJ:        soft_pair_style (e.g. lj/cut/soft/omp)

        Cross-molecule Coulomb:   scaled by fix adapt in elec leg

        Called with leg="elec", lam=1.0 for the shared, fully-coupled
        (non-scaled) settings file used by min, equil, ti_elec, and
        ti_vacuum. Called with leg="vdw" per-lambda for the soft-core
        scaled settings file used by ti_vdw.

        :param leg: 'elec' or 'vdw', which TI leg this settings file is for
        :param pair_coeffs: dict {atom_type: (epsilon, sigma)}
        from _parse_settings_file
        :param other_coeffs: dict {"bond": [...], "angle": [...], ...}
        from _parse_settings_file
        :param solvent_types: iterable of solvent atom types
        :param solute_types: iterable of solute atom types
        :param lam: current lambda value for this leg (1.0 for the shared
            fully-coupled file)
        :param mix_rule: 'arithmetic' or 'geometric'
        :param main_pair_style: the base (non-soft) LAMMPS pair_style string
        :param soft_pair_style: the FEP soft-core pair_style string
        :param output_path: path to write the .in.settings file to
        :returns: output_path, for convenience when chaining into sim_params
        :rtype: str
        """
        is_elec = leg == "elec"
        lj_lam = 1.0 if is_elec else lam
        lj_lam_str = f"{lj_lam:.5f}"

        solvent_sorted = sorted(solvent_types)
        solute_sorted = sorted(solute_types)

        lines = []
        if leg != "elec":
            lines.append("# " + 35 * "=" + "\n")
            lines.append(
                f"# TI force field coeffs — {leg} leg, lambda={lam:.5f}\n")
            lines.append(f"# Mixing rule: {mix_rule}\n")
            lines.append("# " + 35 * "=" + "\n\n")
        else:
            lines.append("# " + 35 * "=" + "\n")
            lines.append(
                "# TI force field coeffs — Pretty system_packed.in.settings\n")
            lines.append(f"# Mixing rule: {mix_rule}\n")
            lines.append("# " + 35 * "=" + "\n\n")

        # --- Solvent like-like and cross terms ---

        lines.append("# -- Solvent: regular style, never scaled ---\n")
        if is_elec:
            for i in solvent_sorted:
                for j in solvent_sorted:
                    if j < i:
                        continue
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    if i == j:
                        eps_ij, sig_ij = eps_i, sig_i
                    else:
                        eps_ij, sig_ij = self._compute_mixed_params(
                            eps_i, sig_i, eps_j, sig_j, mix_rule)
                    if eps_ij == 0.0:
                        sig_ij = max(sig_ij, 1.0)
                    lines.append(
                        f"pair_coeff      {i} {j} {eps_ij:.3f} {sig_ij:.3f}\n")
        else:
            for i in solvent_sorted:
                for j in solvent_sorted:
                    if j < i:
                        continue
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    if i == j:
                        eps_ij, sig_ij = eps_i, sig_i
                    else:
                        eps_ij, sig_ij = self._compute_mixed_params(
                            eps_i, sig_i, eps_j, sig_j, mix_rule)
                    lines.append(f"pair_coeff      {i} {j} {main_pair_style} "
                                 f"{eps_ij:.3f} {sig_ij:.3f}\n")
        lines.append("\n")

        # --- Solute like-like and intramolecular cross terms ---
        lines.append(
            "# -- Solute: like-like and intramolecular cross terms ---\n")
        if is_elec:
            for i in solute_sorted:
                for j in solute_sorted:
                    if j < i:
                        continue
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    if i == j:
                        eps_ij, sig_ij = eps_i, sig_i
                    else:
                        eps_ij, sig_ij = self._compute_mixed_params(
                            eps_i, sig_i, eps_j, sig_j, mix_rule)
                    if eps_ij == 0.0:
                        sig_ij = max(sig_ij, 1.0)
                    lines.append(f"pair_coeff      {i} {j} "
                                 f"{eps_ij:.3f} {sig_ij:.3f}\n")
        else:
            for i in solute_sorted:
                for j in solute_sorted:
                    if j < i:
                        continue
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    if i == j:
                        eps_ij, sig_ij = eps_i, sig_i
                    else:
                        eps_ij, sig_ij = self._compute_mixed_params(
                            eps_i, sig_i, eps_j, sig_j, mix_rule)
                    lines.append(f"pair_coeff      {i} {j}"
                                 f" {main_pair_style} "
                                 f"{eps_ij:.3f} {sig_ij:.3f}\n")
        lines.append("\n")

        # --- Solvent-solute cross terms ---

        if is_elec:
            lines.append(
                "# -- Solvent-solute cross terms for coulomb leg---\n")
            for i in solvent_sorted:
                for j in solute_sorted:
                    ii, jj = min(i, j), max(i, j)
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    eps_ij, sig_ij = self._compute_mixed_params(
                        eps_i, sig_i, eps_j, sig_j, mix_rule)
                    if eps_ij == 0.0:
                        sig_ij = max(sig_ij, 1.0)
                    lines.append(f"pair_coeff      {ii} {jj} "
                                 f"{eps_ij:.3f} {sig_ij:.3f}\n")
            lines.append("\n")
        else:
            lines.append("# -- Solvent-solute cross terms "
                         "for LJ leg--- \n")
            for i in solvent_sorted:
                for j in solute_sorted:
                    ii, jj = min(i, j), max(i, j)
                    eps_i, sig_i = pair_coeffs[i]
                    eps_j, sig_j = pair_coeffs[j]
                    eps_ij, sig_ij = self._compute_mixed_params(
                        eps_i, sig_i, eps_j, sig_j, mix_rule)
                    if eps_ij == 0.0:
                        sig_ij = max(sig_ij, 1.0)
                    lines.append(
                        f"pair_coeff      {ii} {jj} {soft_pair_style} "
                        f"{eps_ij:.3f} {sig_ij:.3f} {lj_lam_str}\n")
            lines.append("\n")
            lines.append("# -- Explicit zero out of other pair style"
                         " for cross terms--- \n")
            for i in solvent_sorted:
                for j in solute_sorted:
                    ii, jj = min(i, j), max(i, j)
                    lines.append(
                        f"pair_coeff      {ii} {jj} {main_pair_style} "
                        f"0.00 1.00 \n")
            lines.append("\n")

        # --- Bond, angle, dihedral, improper coefficients ---
        for section, header in [
            ("bond", "# -- Bond coefficients ---\n"),
            ("angle", "# -- Angle coefficients ---\n"),
            ("dihedral", "# -- Dihedral coefficients ---\n"),
            ("improper", "# -- Improper coefficients ---\n"),
        ]:
            if other_coeffs[section]:
                lines.append(header)
                for coeff in other_coeffs[section]:
                    lines.append(f"{coeff}\n")
                lines.append("\n")

        with open(output_path, "w") as f:
            f.writelines(lines)
        self.logger.info(f"\tWrote {os.path.basename(output_path)}")

        return output_path

    def _validate_inputs(self, sim_params, solvation_params, executables):
        """
        Validate all inputs before any packing/moltemplate/simulation work
        begins, so obviously bad configuration fails fast rather than after
        expensive upstream steps have already run.

        :raises ValueError: on any invalid or missing required input
        """
        errors = []

        # Load all inputs from sim_params + use defaults
        # where nothing was specified
        sim_params = {**self.default_sim_params, **sim_params}

        # Check that none of these quantities are negative or missing
        for k in ('temp', 'press', 'temp_damp', 'press_damp', 'timestep',
                  'equil_steps', 'ti_steps'):
            v = sim_params.get(k)
            if v is None:
                errors.append(f"Simulation parameter {k} is missing")
            elif v < 0:
                errors.append(
                    f"Simulation parameter {k} cannot be < 0, got {v}")

        # Load all solvation_params + use defaults where nothing was specified
        solvation_params = {
            **self.default_solvation_params,
            **solvation_params
        }
        solute_params = solvation_params.get('solute')
        if solute_params is None:
            errors.append("Solute must be specified in solvation_params")

        solvent_params = solvation_params.get('solvent')
        if solvent_params is None:
            errors.append("Solvent must be specified in solvation_params")
        elif not isinstance(solvent_params, list):
            errors.append(
                "Solvent should be specified as a list of molecular components"
            )

        for label, mol in [("solute", solute_params)] + [
            (f"solvent[{i}]", m) for i, m in enumerate(solvent_params or [])
        ]:
            if mol is None:
                continue
            pdb = mol.get('pdb')
            if pdb is None or not isfile(pdb):
                errors.append(f"{label}: pdb file '{pdb}' does not exist")
            lt = mol.get('lt')
            if lt is None or not isfile(lt):
                errors.append(f"{label}: .lt file '{lt}' does not exist")
            if mol.get('class', 'unknown') == 'unknown':
                errors.append(
                    f"{label}: 'class' not specified in molecule entry")
            num_molecs = mol.get('num_molecs',
                                 1 if label == "solute" else None)
            if not isinstance(num_molecs, int) or num_molecs <= 0:
                errors.append(
                    f"{label}: num_molecs must be a positive integer,"
                    " got {num_molecs!r}")

        pack_tol = solvation_params.get('pack_tol')
        pack_boxlen = solvation_params.get('pack_boxlen')
        if not (pack_tol > 0 and pack_boxlen > 0):
            errors.append("pack_tol and pack_boxlen must both be > 0")

        free_energy = solvation_params.get('free_energy')
        if free_energy in ('gibbs', 'helmholtz'):
            sim_params['free_energy'] = free_energy
        else:
            errors.append(f"free_energy must be ``helmholtz`` or "
                          f"``gibbs``, got {free_energy}")

        for exe_key in ('packmol', 'moltemp', 'moltemp_cleanup'):
            exe = executables.get(exe_key)
            if exe is None:
                errors.append(
                    f"{exe_key} must be specified in solvation_params")
            elif shutil.which(exe) is None:
                errors.append(f"{exe_key} '{exe}' not found or not executable")

        lambda_values = solvation_params.get('lambda_values')
        if isinstance(lambda_values, int):
            if lambda_values <= 1:
                errors.append(
                    f"lambda_values as int must be >= 2, got {lambda_values}")
            else:
                lambda_values = list(np.linspace(0.00, 1.00,
                                                 num=lambda_values))
        elif isinstance(lambda_values, list):
            if not all(isinstance(x, (int, float)) for x in lambda_values):
                errors.append("lambda_values list must contain only numbers")
            else:
                lambda_values = [float(x) for x in lambda_values]
        else:
            errors.append(
                "lambda_values must be a list of floats or an int count")

        if not errors:
            solvation_params['lambda_values'] = lambda_values
            self.logger.info(
                f"\tlambda windows: {[f'{lam:.5f}' for lam in lambda_values]}")

        lambda_diff = solvation_params.get('lambda_diff')
        if lambda_diff is None or lambda_diff <= 0:
            errors.append(f"lambda_diff must be > 0, got {lambda_diff!r}")

        # --- sim_params ---
        atom_style = sim_params.get('atom_style')
        if atom_style not in ("full", "molecular"):
            errors.append(
                f"atom_style must be 'full' or 'molecular', got {atom_style!r}"
            )

        pair_style = sim_params.get('pair_style')
        if pair_style is None:
            errors.append("pair_style must be specified in sim_params")
        else:
            try:
                self._lookup_pair_style_args(pair_style)
                self._resolve_soft_style(pair_style)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError("Invalid solvation free energy inputs:\n  - "
                             + "\n  - ".join(errors))
        else:
            return sim_params, solvation_params

    def _write_results_file(self, lambda_values, dudl, dudl_std, dudl_err,
                            filepath):
        """
        lambda_values, dudl, dudl_std, dudl_err are parallel lists:
        dudl_std  = standard deviation of the per-step/per-block dU/dL samples
        dudl_err  = standard error of the mean (accounting for autocorrelation)
        """
        # read prior results, if any
        lambda_values_old, dudl_old, dudl_std_old, dudl_err_old = \
            self._read_results_file(filepath)

        # merge old and new, letting new values overwrite old ones
        # at matching lambda
        combined = {}
        for lam, val, std, err in zip(lambda_values_old, dudl_old,
                                      dudl_std_old, dudl_err_old):
            combined[lam] = (val, std, err)
        for lam, val, std, err in zip(lambda_values, dudl, dudl_std, dudl_err):
            combined[lam] = (val, std, err)

        sorted_lambdas = sorted(combined.keys())

        with open(filepath, 'w') as f:
            f.write(f'# {"lambda":>10s}'
                    f'{"dudl":>15s}'
                    f'{"stddev":>15s}'
                    f'{"stderr":>15s}\n')
            for lam in sorted_lambdas:
                val, std, err = combined[lam]
                f.write(f'{lam:10.5f} {val:15.6f} {std:15.6f} {err:15.6f}\n')

        # build the merged, sorted output lists from combined
        # (not the raw inputs)
        merged_lambda_values = sorted_lambdas
        merged_dudl = [combined[lam][0] for lam in sorted_lambdas]
        merged_dudl_std = [combined[lam][1] for lam in sorted_lambdas]
        merged_dudl_err = [combined[lam][2] for lam in sorted_lambdas]

        return (merged_lambda_values, merged_dudl, merged_dudl_std,
                merged_dudl_err)

    def _read_results_file(self, filepath):
        lambda_values, dudl, dudl_std, dudl_err = [], [], [], []

        if not os.path.exists(filepath):
            return lambda_values, dudl, dudl_std, dudl_err

        with open(filepath, 'r') as r:
            for line in r:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                lambda_values.append(float(parts[0]))
                dudl.append(float(parts[1]))
                dudl_std.append(float(parts[2]))
                dudl_err.append(float(parts[3]))

        return lambda_values, dudl, dudl_std, dudl_err
