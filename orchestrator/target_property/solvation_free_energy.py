import numpy as np
import json
import copy
from time import sleep
import subprocess
import shutil
import matplotlib.pyplot as plt
import random
import os
from os.path import isfile
from typing import Union, Optional, Any, Dict
from ..simulator import simulator_builder
from . import TargetProperty
from ..scheduler import Scheduler
from ..storage import Storage
from orchestrator.target_property.analysis import AnalyzeLammpsLog
from ..utils.restart import restarter
from .pack_molecules import (pack_system, read_lammps_data,
                             make_molecule_unique)
from .forcefield_parser import (
    forcefield_merger,
    write_parameter_file,
    build_style_strings,
)


class SolvationFreeEnergy(TargetProperty):
    """
    Class to compute solvation free energy for a solute in a pure or
    multi-component solvent.

    This is done by thermodynamic integration in a three step process:
    1. First, the charges of the solute molecule are turned off
    (scaled 1 to 0).
    2. Next, the long-range van der Waals interactions are turned off
    (scaled 1 to 0).
    3. Finally, the charges of the isolated solute are turned back on
    (re-scaled 0 to 1).

    This module reports the solvation free energy, or the change in free
    energy for solute to move from ideal gas into solution. A negative
    result indicates a favorable dissolution, and a positive results
    indicates an unfavorable dissolution.

    Simulation parameters and required paths are read from json file.

    :param simulator_type: name of the simulator to perform simulations
    :type simulator_type: str
    :param simulator_path: path to the simulator executable
    :type simulator_path: str
    :param job_details: parameters for running the jobs
    :type job_details: dict
    :param input_template: Optional LAMMPS input templates. This should
        be a dictionary of paths to simulation input template files
        required for conducting simulations. Example templates can be
        found in the target_property module in the solvation_ti_defaults
        folder.
        Possible keys: ``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
        ``ti_vacuum``
    :type input_template: dict
    :param ti_job_details: optional job parameters for specific jobs.
        Values that differ from job_details will override for that
        type of job.
        Possible keys: ``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
        ``ti_vacuum``
    :type ti_job_details: dict

    """

    def __init__(
        self,
        simulator_type: str,
        simulator_path: str,
        job_details: dict,
        input_template: Optional[Dict] = None,
        ti_job_details: Optional[Dict] = None,
        **kwargs: Any,
    ):
        """
        Initialization of the SolvationFreeEnergy class with args dict

        :param simulator_type: name of the simulator to perform simulations
        :type simulator_type: str
        :param simulator_path: path to the simulator executable
        :type simulator_path: str
        :param job_details: parameters for running the jobs
        :type job_details: dict
        :param input_template: Optional LAMMPS input templates. This should
            be a dictionary of paths to simulation input template files
            required for conducting simulations. Example templates can be
            found in the target_property module in the solvation_ti_defaults
            folder.
            Possible keys: ``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
            ``ti_vacuum``
        :type input_template: dict
        :param ti_job_details: optional job parameters for specific jobs.
            Values that differ from job_details will override for that
            type of job.
            Possible keys: ``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
            ``ti_vacuum``
        :type ti_job_details: dict
        """

        self.default_sim_params = {
            "units": "real",
            "atom_style": "full",
            "soft_args": {
                "softcore_n": 1,
                "alpha_vdw": 0.50,
                "alpha_elec": 10
            },
            "pair_modify": "none",
            "special_bonds": "none",
            "extra_pair_styles": None,
            "extra_coeff_lines": None,
            "pair_style": "none",
            "bond_style": "none",
            "angle_style": "none",
            "dihedral_style": "none",
            "improper_style": "none",
            "kspace_modify": "none",
            "kspace_style": "none",
            "temp": 298.15,
            "press": 1.0,
            "temp_damp": 100.0,
            "press_damp": 250.0,
            "timestep": 0.1,
            "equil_steps": 50000,
            "ti_steps": 100000
        }

        self.default_system_params = {
            "pack_tol": 2.0,
            "pack_boxlen": 40.0,
            "solute_molecule_id": 1
        }

        self.default_ti_params = {
            "free_energy": "gibbs",
            "solute": "unknown",
            "solvent": ["unknown"],
            "lambda_values": 11,
            "lambda_diff": 0.002,
            "equil_frac": 0.25
        }

        self.fep_soft_map = {
            "lj/cut": {
                "vdw": "lj/cut/soft",
                "soft_args": ["softcore_n", "alpha_vdw"]
            },
            "lj/cut/coul/cut": {
                "vdw": "lj/cut/coul/soft",
                "soft_args": ["softcore_n", "alpha_vdw", "alpha_elec"]
            },
            "lj/cut/coul/long": {
                "vdw": "lj/cut/coul/long/soft",
                "soft_args": ["softcore_n", "alpha_vdw", "alpha_elec"]
            },
            "lj/charmm/coul/long": {
                "vdw": "lj/charmm/coul/long/soft",
                "soft_args": ["softcore_n", "alpha_vdw", "alpha_elec"]
            },
            "lj/class2/coul/cut": {
                "vdw": "lj/class2/coul/cut/soft",
                "soft_args": ["softcore_n", "alpha_vdw", "alpha_elec"]
            },
            "lj/class2/coul/long": {
                "vdw": "lj/class2/coul/long/soft",
                "soft_args": ["softcore_n", "alpha_vdw", "alpha_elec"]
            },
        }

        # Set default seed
        self.default_seed = 4928302

        # Fill in defaults
        self.ti_job_details = {}
        if ti_job_details:
            for k in ("min", "equil", "ti_elec", "ti_vdw", "ti_vacuum"):
                v = ti_job_details.get(k)
                if v is not None:
                    self.ti_job_details[k] = {
                        **job_details,
                        **ti_job_details[k]
                    }
                else:
                    self.ti_job_details[k] = dict(job_details)

        self.input_template = input_template
        if self.input_template is None:
            source_file_location = os.path.dirname(os.path.abspath(__file__))
            self.input_template = {
                "min":
                f"{source_file_location}/solvation_ti_defaults/min.in",
                "equil":
                f"{source_file_location}/solvation_ti_defaults/equil.in",
                "ti_elec":
                f"{source_file_location}/solvation_ti_defaults/ti_elec.in",
                "ti_vdw":
                f"{source_file_location}/solvation_ti_defaults/ti_vdw.in",
                "ti_vacuum":
                f"{source_file_location}/solvation_ti_defaults/ti_vacuum.in"
            }

        simulator_args = {'code_path': simulator_path, 'elements': []}
        self.built_simulator = simulator_builder.build(simulator_type,
                                                       simulator_args)

        self.progress_flag = 'init'
        self.forcefield_path = None
        self.system_mode = None
        self.init_structure = None
        self.analysis_dir = None
        self.min_structure = None
        self.equil_structure = None
        self.elec_restart = None
        self.lambdas_to_run = {'ti_elec': [], 'ti_vdw': [], 'ti_vacuum': []}
        self.running_jobs = {'ti_elec': [], 'ti_vdw': [], 'ti_vacuum': []}
        self.finished_jobs = {'ti_elec': [], 'ti_vdw': [], 'ti_vacuum': []}
        self.job_ids = {
            'min': [],
            'equil': [],
            'ti_elec': [],
            'ti_vdw': [],
            'ti_vacuum': []
        }

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
                'forcefield_path': self.forcefield_path,
                'system_mode': self.system_mode,
                'init_structure': self.init_structure,
                'min_structure': self.min_structure,
                'equil_structure': self.equil_structure,
                'analysis_dir': self.analysis_dir,
                'elec_restart': self.elec_restart,
                'job_ids': self.job_ids,
                'lambdas_to_run': self.lambdas_to_run,
                'running_jobs': self.running_jobs,
                'finished_jobs': self.finished_jobs
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

        for attr, value in restart_dict.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

        if self.progress_flag != 'init':
            self.restart = True
            self.logger.info('\tRestart data found (progress_flag='
                             f'{self.progress_flag}); will attempt to'
                             'resume from last checkpoint.')
        else:
            self.restart = False

        # Save finished_jobs as the union of unique jobs found
        # in the analysis .json file and the restart file
        if self.analysis_dir is not None:
            manifest_path = os.path.join(self.analysis_dir,
                                         "finished_jobs.json")
            try:
                with open(manifest_path) as f:
                    json_finished_jobs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(
                    'analysis_dir set but finished_jobs.json missing or '
                    f'unreadable at {manifest_path} ({e}); '
                    'using checkpoint value only')
                json_finished_jobs = {}

            merged = {}
            for leg in set(self.finished_jobs) | set(json_finished_jobs):
                checkpoint_jobs = self.finished_jobs.get(leg, [])
                manifest_jobs = json_finished_jobs.get(leg, [])

                seen_ids = set()
                unique_jobs = []
                for job in checkpoint_jobs + manifest_jobs:
                    if job['job_id'] in seen_ids:
                        continue
                    seen_ids.add(job['job_id'])
                    unique_jobs.append(job)
                merged[leg] = unique_jobs

            self.finished_jobs = merged

    def calculate_property(
        self,
        path_type: str,
        sim_params: dict,
        system_params: dict,
        ti_params: dict,
        executables: Optional[Dict] = None,
        iter_num: Optional[int] = 0,
        random_seed_use: Optional[bool] = False,
        max_submit: Optional[int] = None,
        user: Optional[str] = None,
        accelerator: Optional[str] = None,
        scheduler: Optional[Scheduler] = None,
        storage: Optional[Storage] = None,  # CURRENTLY NOT IMPLEMENTED
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
        :param system_params: parameters required for initial system
            preparation
        :type system_params: dict
        :param ti_params: parameters required for thermodynamic integration
            analysis
        :type ti_params: dict
        :param executables: dictionary for paths to required executables. For
            system_mode = ``prepare``, no executables are required. For
            system_mode = ``pack``, packmol is required. If
            ff_mode = ``moltemplate`` for any molecule, moltemplate and
            moltemplate_cleanup are required.
            Possible keys: ``packmol``, ``moltemp``, ``moltemp_cleanup``
        :type executables: dict
        :param iter_num: unique iteration number for running replicates
        :type iter_num: int
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: boolean
        :param max_submit: maximum number of jobs to submit to queue
        :type max_submit: str
        :param user: HPC user to check for current number of submitted jobs
        :type user: str
        :param scheduler: Orchestrator scheduler class for submitting jobs
        :type scheduler: Scheduler
        :param storage: Orchestrator storage class to create datasets
            CURRENTLY NOT USED!!
        :type storage: Storage

        :returns: a dictionary with solvation free energy, sampling error,
            calc_ids as dictionary of lists with keys corresponding to
            simulation types (``min``, ``equil``, ``ti_elec``, ``ti_vdw``,
            and ``ti_vacuum``), and success
        :rtype: dict
        """

        # ensure restart is properly read
        self.restart_property()

        # Get default scheduler if one is not provided
        if scheduler is None:
            scheduler = self.default_scheduler

        # Make path for coefficient files to be generated, stored, & copied
        if self.forcefield_path is None:  # i.e. is a restart
            self.forcefield_path = scheduler.make_path_base(
                self.__class__.__name__, f'{path_type}/ForceField')

        if self.init_structure is None:
            system_dir = scheduler.make_path_base(self.__class__.__name__,
                                                  f'{path_type}/InitSystem')
            self.init_structure = f'{system_dir}/system_packed.data'

        # Validate all inputs for phsyical constraints, override sim_params
        # and solvation_params with completed versions and check that
        # executables exist and are executable
        self.logger.info('Validaing calculate property inputs...')
        sim_params, system_params, ti_params = self._validate_inputs(
            sim_params, system_params, ti_params, executables)

        # Seed for velocity initialization during ``equil``
        random_seed = random.randint(
            1, 10000) if random_seed_use else self.default_seed
        sim_params['random_seed'] = random_seed

        if self.progress_flag == 'init':
            sim_params = self._prepare_solute_solvent_system(
                system_dir, sim_params, system_params, executables, scheduler,
                path_type, random_seed_use)
            self.progress_flag = 'ff_done'
            self.checkpoint_property()

        # Always read saved_params.json and merge with sim_params
        self.logger.info('Reading simulation parameters from '
                         f'{self.forcefield_path}/saved_params.json')
        with open(f'{self.forcefield_path}/saved_params.json') as f:
            save_params = json.load(f)
        sim_params = {**sim_params, **save_params}
        charged_solute = save_params.get('charged_solute')

        charge_file = f'{self.forcefield_path}/system_packed.in.charges'
        param_file = f'{self.forcefield_path}/system_packed.in.settings'
        soft_param_file = f"{self.forcefield_path}/ti_vdw.in.settings"

        # Find accelerator, if one is provided. If not, search
        # *_style lines for suffix (e.g. lj/cut/kk -> '/kk')
        self.add_accelerator_flags(sim_params, accelerator)

        # --- Minimization procedure!! ---
        if (self.progress_flag == 'ff_done' and self.system_mode == 'pack'):
            self.logger.info(
                'Attempting to run minimization of packed structure')
            self.job_ids['min'] = self._conduct_sim(
                sim_params,
                [param_file, charge_file, f'{self.init_structure}'], scheduler,
                f'{path_type}/min', "min", max_submit, user)
            scheduler.block_until_completed(self.job_ids['min'])
            job_status = scheduler.check_completed_job_status(
                self.job_ids['min'])
            min_dir = \
                os.path.realpath(scheduler.get_job_path(self.job_ids['min']))
            self.min_structure = f"{min_dir}/minimized.data"
            if not isfile(self.min_structure) or job_status != 'done':
                raise ValueError("Minimization job finished unsuccessfully.")
            self.logger.info('Minimization job finished successfully')
            self.progress_flag = 'min_done'
            self.checkpoint_property()
        elif (self.progress_flag == 'ff_done'
              and self.system_mode == 'prepared'):
            self.logger.info(
                'Skipping minimization procedure for prepared system')
            self.min_structure = f'{system_dir}/minimized.data'
            shutil.copy(self.init_structure, self.min_structure)
            self.logger.info(
                f'Copied {self.init_structure} to {self.min_structure}')
            self.progress_flag = 'min_done'
            self.checkpoint_property()

        # --- Equilibration procedure!! ---
        if self.progress_flag == 'min_done':
            self.logger.info('Attempting to run equilibration job')
            self.job_ids['equil'] = self._conduct_sim(
                sim_params, [param_file, charge_file, self.min_structure],
                scheduler, f'{path_type}/equil', "equil", max_submit, user)
            scheduler.block_until_completed(self.job_ids['equil'])
            job_status = scheduler.check_completed_job_status(
                self.job_ids['equil'])
            equil_dir = os.path.realpath(
                scheduler.get_job_path(self.job_ids['equil']))
            self.equil_structure = f'{equil_dir}/npt_equil.data'
            if not isfile(self.equil_structure) or job_status != 'done':
                raise ValueError("Equilibration job finished unsuccessfully")
            self.progress_flag = 'equil_done'
            self.checkpoint_property()

        # --- TI procedural setup ---
        self.logger.info('Gathering info for TI jobs')
        lambda_values = ti_params.get('lambda_values')
        equil_frac = ti_params.get("equil_frac")
        lambda_diff = ti_params.get('lambda_diff')
        sim_params['lambda_diff'] = lambda_diff
        if self.restart is not True:
            for leg in ('elec', 'vdw', 'vacuum'):
                self.lambdas_to_run[f'ti_{leg}'] = list(lambda_values)
        self.checkpoint_property()

        self.logger.info(
            f'Starting TI w/ initial lambda array: {lambda_values}')
        self.logger.info(f'Using lambda_diff: {lambda_diff}')

        # Make analysis directory if one does not exist from failed run
        if self.analysis_dir is None:
            self.analysis_dir = scheduler.make_path_base(
                self.__class__.__name__, f'{path_type}/Analysis/{iter_num}')
            self.checkpoint_property()

        # --- TI Electronic Jobs ---
        if charged_solute and (self.lambdas_to_run['ti_elec']
                               or self.running_jobs['ti_elec']):
            self.logger.info('Preparing & running electronic leg')
            sim_params['lambda_vdw'] = 1.00
            self._run_ti('elec', iter_num, 'lambda_c', sim_params,
                         [param_file, charge_file, self.equil_structure],
                         scheduler, f'{path_type}/ti_elec/{iter_num}',
                         max_submit, user)
            scheduler.block_until_completed(self.job_ids['ti_elec'])
            self.check_ti_jobs('elec', iter_num, equil_frac, scheduler)
            self.plot_leg_diagnostics('elec')
            self.progress_flag = 'ti_elec_done'

        elif not charged_solute:
            self.logger.info('Solute is uncharged, skipping electronic TI leg')
            self.progress_flag = 'ti_elec_skip'
        self.checkpoint_property()

        # TI van der Waals + vacuum legs, submitted & blocked together
        vdw_remaining = bool(self.lambdas_to_run['ti_vdw']) or bool(
            self.running_jobs['ti_vdw'])
        vacuum_remaining = charged_solute and (
            bool(self.lambdas_to_run['ti_vacuum'])
            or bool(self.running_jobs['ti_vacuum']))

        if vdw_remaining or vacuum_remaining:
            if charged_solute:
                starting_structure = self.elec_restart
            else:
                starting_structure = self.equil_structure

            self.logger.info('Preparing & running dispersion (VDW) leg')
            self.logger.info(f'Starting from structure: {starting_structure}')
            sim_params['lambda_c'] = 0.00
            self._run_ti('vdw', iter_num, 'lambda_vdw', sim_params,
                         [soft_param_file, charge_file, starting_structure],
                         scheduler, f'{path_type}/ti_vdw/{iter_num}',
                         max_submit, user)

            if charged_solute:
                self.logger.info('Preparing & running TI_VACUUM jobs')
                sim_params['lambda_vdw'] = 1.00
                self._run_ti('vacuum', iter_num, 'lambda_c', sim_params,
                             [param_file, charge_file, self.equil_structure],
                             scheduler, f'{path_type}/ti_vacuum/{iter_num}',
                             max_submit, user)

            combined_job_ids = list(self.job_ids['ti_vdw'])
            if charged_solute:
                combined_job_ids += self.job_ids['ti_vacuum']
            scheduler.block_until_completed(combined_job_ids)

            self.check_ti_jobs('vdw', iter_num, equil_frac, scheduler)
            self.plot_leg_diagnostics('vdw')
            self.progress_flag = 'ti_vdw_done'

            if charged_solute:
                self.check_ti_jobs('vacuum', iter_num, equil_frac, scheduler)
                self.plot_leg_diagnostics('vacuum')
                self.progress_flag = 'ti_vacuum_done'
            else:
                self.progress_flag = 'ti_vacuum_skip'
        self.checkpoint_property()

        # Analyze results of legs
        if self.progress_flag in ('ti_vacuum_done', 'ti_vacuum_skip', 'done'):

            equil_frac = ti_params.get("equil_frac")
            dg_total = 0
            stat_err_total = 0

            legs = ('elec', 'vdw', 'vacuum')
            fig, axes = plt.subplots(2,
                                     len(legs),
                                     figsize=(4 * len(legs), 6),
                                     sharex='col')

            leg_data = {}
            for col, leg in enumerate(legs):
                leg_data[leg] = {}

                results_file = f'{self.analysis_dir}/results_{leg}.dat'
                lams, lens, dudl, dudl_std, dudl_err = \
                    self._read_results_file(results_file)

                # Integrate dudl for each leg
                dg_leg, stat_err = self.compute_integral(lams, dudl, dudl_err)
                if leg in ("vdw", "elec"):
                    dg_total += dg_leg
                else:
                    dg_total -= dg_leg
                stat_err_total += stat_err**2

                self.plot_leg_diagnostics(leg)

                print("=" * 75 + '\n',
                      f"dF ({leg}) = {dg_leg:.4f} +/- {stat_err:.4f}\n",
                      "=" * 75 + '\n\n')

            stat_err_total = np.sqrt(stat_err_total)
            self.plot_dudl_vs_lambda(self.analysis_dir)

            print(
                "=" * 75 + '\n', f"{ti_params.get('free_energy').upper()} "
                f"FREE ENERGY = {dg_total:.4f} +/- {stat_err_total:.4f}\n",
                "=" * 75 + '\n\n')

            self.progress_flag = 'done'
            self.checkpoint_property()

            return {
                'property_value': dg_total,
                'property_std': stat_err_total,
                'calc_ids': self.job_ids,
                'success': True
            }

    def _conduct_sim(self,
                     sim_params: Dict[str, Any],
                     sim_files: Union[str, list[str], None],
                     scheduler: Scheduler,
                     sim_path: str,
                     sim_type: str,
                     max_submit: Optional[int] = None,
                     user: Optional[str] = None) -> int:
        """
        Performs a single simulation of the passed sim_type

        :param sim_params: simulation specific parameters. See
            calculate_property() function doc string for specific keys
            to be included.
        :type sim_params: dict
        :param sim_files: simulation files that are needed in the path
        :type sim_files: str, list, or None
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :param sim_path: path to perform simulation
        :type sim_path: str
        :param sim_type: type of simulation to be run.
            Possible options:
            ``min``, ``equil``, ``ti_elec``, ``ti_vdw``, or ``ti_vacuum``.
        :type sim_type: str
        :param max_submit: max number of jobs allowed to run concurrently
        :type max_submit: int
        """

        if sim_type not in ("min", "equil", "ti_elec", "ti_vdw", "ti_vacuum"):
            raise ValueError(
                f"unknown simulation type: {sim_type}, expected one of:"
                "(``min``, ``equil``, ``ti_elec``, ``ti_vdw``, ``ti_vacuum``)")

        if max_submit is not None and user is not None:
            scheduler._block_max_submit(max_submit, user)

        calc_id = self.built_simulator.run(
            sim_path,
            sim_files,
            template_fill=sim_params,
            input_template=self.input_template[sim_type],
            scheduler=scheduler,
            job_details=self.ti_job_details[sim_type])

        return calc_id

    def _run_ti(self, leg, iter_num, lambda_parameter, sim_params, sim_files,
                scheduler, sim_path, max_submit, user):
        """
        Run a single TI leg (``elec``, ``vdw``, or ``vacuum``) at the lambda
        values that correspoind to those in the self.lambdas_to_run dictionary.
        Per leg results are written to results_<leg>.dat, using the timeseries
        equilibration cutoff passed in sim_params (see below for details).

        For lambda values with a matching entry in self.finished_jobs (i.e.
        already run in a prior iter_num), the corresponding .data file is
        used to restart that lambda window rather than starting from
        sim_params['input_data']. For the ``vdw`` leg, if no restart data
        file is found for a given lambda and self.elec_restart is set, jobs
        are instead started from self.elec_restart. As each job is
        submitted, its lambda is removed from self.lambdas_to_run and it is
        appended to self.running_jobs; the checkpoint is written after each
        submission.

        :param leg: TI leg that is being run: ``elec``, ``vdw``, or ``vacuum``
        :type leg: str
        :param iter_num: unique number for running replicates
        :type iter_num: int
        :param lambda_parameter: the sim_params key set to the current
            lambda value for each job. This should be ``lambda_vdw``
            for ti_vdw jobs and ``lambda_c`` for ti_elec and ti_vacuum.
        :type lambda_parameter: str
        :param sim_params: Simulation parameters required to run jobs. This
            is used to fill out the simulation templates passed during
            SolvationFreeEnergy class initialization.
        :type sim_params: dict
        :param sim_files: list of files needed for every job in this leg
        :type sim_files: list[str]
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :param sim_path: base path for this leg's jobs, without a lambda
            suffix
        :type sim_path: str
        :param max_submit: max number of jobs allowed to run concurrently
        :type max_submit: int
        :param user: HPC user to check for current number of submitted jobs
        :type user: str
        """

        if leg not in ('elec', 'vdw', 'vacuum'):
            raise ValueError(
                f"leg must be 'elec', 'vdw', or 'vacuum', got {leg!r}")
        sim_type = f'ti_{leg}'

        # Create lambda -> old iter_num mapping, overriding with newest
        # lambda runs for duplicates
        restart_datafile = {}
        for job in self.finished_jobs[f'ti_{leg}']:
            if job['lambda'] in self.lambdas_to_run[f'ti_{leg}']:
                restart_datafile[job['lambda']] = (
                    f"{job['job_path']}/ti_{leg}_"
                    f"{job['lambda']:.5f}_{job['iter_num']:03d}.data")

        self.logger.info(
            f"[{sim_type}]: running "
            f"{len(self.lambdas_to_run[f'ti_{leg}'])} lambda window(s): "
            f"{[f'{lam:.5f}' for lam in self.lambdas_to_run[f'ti_{leg}']]}")

        lam_list = list(self.lambdas_to_run[f'ti_{leg}'])
        for lam in lam_list:
            job_sim_params = copy.deepcopy(sim_params)
            job_sim_files = list(sim_files)
            job_sim_params[lambda_parameter] = lam
            if lam not in restart_datafile.keys():
                job_sim_params['input_data'] = 'npt_equil.data'
                if leg == 'vdw' and self.elec_restart is not None:
                    job_sim_params['input_data'] = os.path.basename(
                        self.elec_restart)
                    job_sim_files.append(self.elec_restart)
            else:
                df = restart_datafile[lam]
                job_sim_params['input_data'] = os.path.basename(df)
                job_sim_files.append(df)

            job_sim_params[
                'output_data'] = f'ti_{leg}_{lam:.5f}_{iter_num:03d}.data'
            if (leg == 'vacuum' and lam == 0.0):
                job_sim_params['kspace_modify'] = 'gewald 0.01'
            job_id = self._conduct_sim(job_sim_params, job_sim_files,
                                       scheduler,
                                       f'{sim_path}/lambda_{lam:.5f}',
                                       sim_type, max_submit, user)
            job_path = scheduler.get_job_path(job_id)
            self.lambdas_to_run[f'ti_{leg}'].remove(lam)
            self.running_jobs[f'ti_{leg}'].append({
                "iter_num": iter_num,
                "lambda": lam,
                "job_id": job_id,
                "job_path": job_path
            })
            self.job_ids[f'ti_{leg}'].append(job_id)
            self.checkpoint_property()
            sleep(5)

    def check_ti_jobs(self, leg, iter_num, equil_frac, scheduler):
        """
        Check whether TI jobs have finished, collect statistics from finished
        jobs, update results files with equilibrium stats, and move jobs
        between self.running_jobs and self.finished_jobs.

        :param leg: TI leg that is being run: ``elec``, ``vdw``, or ``vacuum``
        :type leg: str
        :param iter_num: unique number for running replicates
        :type iter_num: int
        :param equil_frac: fraction of the time series to discard as
        equilibration. default values is 0.25, discarding first 25 percent.
        :type equil_frac: float
        :param scheduler: Orchestrator scheduler class for submitting jobs
        :type scheduler: Scheduler
        """
        newly_completed = []
        for job in self.running_jobs[f'ti_{leg}']:
            job_iter = job['iter_num']
            lam = job['lambda']
            job_id = job['job_id']
            job_path = job['job_path']

            if job_iter != iter_num:
                continue
            job_status = scheduler.check_completed_job_status(job_id)
            if job_status != 'done':
                self.logger.warning(
                    f'[ti_{leg}] lambda={lam:.5f} did not complete '
                    f'successfully (status={job_status}), will retry on '
                    'next restart')
                continue
            if leg == 'elec' and lam == 0.0:
                outfile = f"{job_path}/ti_elec_0.00000_{iter_num:03d}.data"
                self.elec_restart = outfile
            newly_completed.append(job)

        # Move running jobs to finished jobs
        for job in newly_completed:
            self.running_jobs[f'ti_{leg}'].remove(job)
            self.finished_jobs[f'ti_{leg}'].append(job)
        self._write_finished_jobs()
        self.checkpoint_property()

        # Analyze new lambda values
        lambda_values, dudl, dudl_std, dudl_err, lens = [], [], [], [], []
        for job in newly_completed:
            lam = job['lambda']
            job_path = job['job_path']
            log_file = f'{job_path}/lammps.out'
            _, n, avg, std, err = self._get_production_stats(
                log_file, 'f_dUdL_avg', equil_frac)
            lambda_values.append(lam)
            dudl.append(avg)
            dudl_std.append(std)
            dudl_err.append(err)
            lens.append(n)

        # Write results to lastest analysis_dir
        lams, lens, dudl, dudl_std, dudl_err = self._write_results_file(
            lambda_values, lens, dudl, dudl_std, dudl_err,
            f'{self.analysis_dir}/results_{leg}.dat')

        return None

    def calculate_with_error(
        self,
        n_calc: int,
        modified_params: Optional[Dict[str, Any]] = None,
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
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :returns: mean and standard deviation of the calculated property
        """
        raise NotImplementedError("To accumulate stats, run compute_property()"
                                  " multiple times")

    def add_accelerator_flags(self, sim_params, provided_acc):
        """
        Detect LAMMPS accelerator suffixes (``/kk`` or ``/omp``) among the
        ``*_style`` values in sim_params and append the corresponding
        command-line flags to self.built_simulator.code_path.

        Every ``*_style`` key in sim_params is scanned for tokens ending in
        ``/kk`` or ``/omp``; the first such suffix found determines which
        accelerator flags are appended (``-k on -sf kk`` for ``/kk``, or
        ``-sf omp`` for ``/omp``). If ``provided_acc`` is given, it is
        checked for consistency against the detected suffix (raising if they
        disagree), or, if no suffix was detected in sim_params, its
        corresponding flags are appended instead.

        :param sim_params: Simulation parameters to scan for accelerator
            suffixes on ``*_style`` keys.
        :type sim_params: dict
        :param provided_acc: An explicitly requested accelerator to
            cross-check against (or fall back on if none is found in
            sim_params). Expected values are "kk", "omp", or None.
        :type provided_acc: str or None

        :returns: None. self.built_simulator.code_path is mutated in place.
        :rtype: None
        """

        found_acc = False
        for s in sim_params:
            if not s.endswith('_style'):
                continue
            value = sim_params.get(s)
            if not isinstance(value, str):
                continue
            for part in value.split():
                if part.endswith('/kk'):
                    self.logger.info('Found /kk suffix -> using -k on -sf kk')
                    self.built_simulator.code_path += ' -k on -sf kk'
                    found_acc = 'kk'
                    break
                elif part.endswith('/omp'):
                    self.logger.info('Found /omp suffix -> using -sf omp')
                    self.built_simulator.code_path += ' -sf omp'
                    found_acc = 'omp'
                    break
            if found_acc is not False:
                break

        if provided_acc is not None and found_acc is not False:
            if found_acc != provided_acc:
                raise ValueError(
                    f'Provided accelerator {provided_acc} '
                    f'does not match found accelerator {found_acc}')

        if found_acc == 'kk':
            self.logger.info('Found /kk suffix -> using -k on -sf kk')
            self.built_simulator.code_path += ' -k on -sf kk'
        elif found_acc == 'omp':
            self.logger.info('Found /omp suffix -> using -sf omp')
            self.built_simulator.code_path += ' -sf omp'

    def plot_dudl_vs_lambda(self, analysis_dir):
        """
        Plot <dU/dλ> as a function of lambda values for each thermodynamic
        integration leg. This requires that results are already accumulated
        into a results_<leg>.dat file via _write_results_file().

        :param analysis_dir: dictory where results_<leg>.dat files live
        :type analysis_dir: str
        """
        lam_elec, _, du_elec, _, se_elec = \
            self._read_results_file(f'{analysis_dir}/results_elec.dat')
        lam_vdw, _, du_vdw, _, se_vdw = \
            self._read_results_file(f'{analysis_dir}/results_vdw.dat')
        lam_vac, _, du_vac, _, se_vac = \
            self._read_results_file(f'{analysis_dir}/results_vacuum.dat')

        fig, ax = plt.subplots()
        if len(lam_elec) > 0:
            ax.errorbar(lam_elec,
                        du_elec,
                        yerr=se_elec,
                        label='elec',
                        marker='o',
                        color='blue')
        if len(lam_vdw) > 0:
            ax.errorbar(lam_vdw,
                        du_vdw,
                        yerr=se_vdw,
                        label='vdw',
                        marker='o',
                        color='red')
        if len(lam_vac) > 0:
            ax.errorbar(lam_vac,
                        du_vac,
                        yerr=se_vac,
                        label='vacuum',
                        marker='o',
                        color='black')

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'dU/d$\lambda$')
        ax.legend()
        fig.tight_layout()

        fig.savefig(f'{analysis_dir}/dudl_vs_lambda.png', dpi=300)
        plt.close(fig)

    def plot_leg_diagnostics(self, leg):
        """
        Plot and save per-lambda timeseries and histogram grids for a
        single TI leg (``elec``, ``vdw``, or ``vacuum``), using whatever
        lambda windows are currently recorded in self.finished_jobs.

        :param leg: TI leg that is being run: ``elec``, ``vdw``, or ``vacuum``
        :type leg: str
        """

        # Load jobs from analysis directory
        with open(os.path.join(self.analysis_dir, "finished_jobs.json")) as f:
            finished_jobs = json.load(f)
        jobs = finished_jobs[f'ti_{leg}']
        lambdas = [job['lambda'] for job in jobs]
        logs = [f"{job['job_path']}/lammps.out" for job in jobs]
        n = len(lambdas)

        if n == 0:
            self.logger.warning(f'[ti_{leg}] no completed lambdas to plot, '
                                'skipping diagnostics')
            return

        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols  # ceil division

        fig_ts, axes_ts = plt.subplots(nrows,
                                       ncols,
                                       figsize=(4 * ncols, 3 * nrows),
                                       squeeze=False)
        fig_hist, axes_hist = plt.subplots(nrows,
                                           ncols,
                                           figsize=(4 * ncols, 3 * nrows),
                                           squeeze=False)
        axes_ts_flat = axes_ts.flatten()
        axes_hist_flat = axes_hist.flatten()

        for i, (lam, log) in enumerate(zip(lambdas, logs)):
            AnalyzeLammpsLog.plot_timeseries(logfile=log,
                                             quantity='f_dUdL_avg',
                                             name='<dUdL>',
                                             ax=axes_ts_flat[i],
                                             logger=self.logger)
            axes_ts_flat[i].set_title(f'\u03bb = {lam:.5f}')

            AnalyzeLammpsLog.plot_histogram(logfile=log,
                                            quantity='f_dUdL_avg',
                                            name='<dUdL>',
                                            ax=axes_hist_flat[i],
                                            bins=50,
                                            logger=self.logger)
            axes_hist_flat[i].set_title(f'\u03bb = {lam:.5f}')

        for j in range(i + 1, len(axes_ts_flat)):
            axes_ts_flat[j].set_visible(False)
            axes_hist_flat[j].set_visible(False)

        fig_ts.tight_layout()
        fig_hist.tight_layout()

        fig_ts.savefig(f'{self.analysis_dir}/ti_{leg}_timeseries.png',
                       dpi=300,
                       bbox_inches='tight')
        fig_hist.savefig(f'{self.analysis_dir}/ti_{leg}_histograms.png',
                         dpi=300,
                         bbox_inches='tight')
        plt.close(fig_ts)
        plt.close(fig_hist)

    def _prepare_solute_solvent_system(self, system_dir, sim_params,
                                       system_params, executables, scheduler,
                                       path_type, random_seed_use):
        """
        Wrapper function for preparing forcefield files and an initial
        structure for system_mode = ``pack``. Specifically, this wraper runs
        moltemplate where required, parses and rewrites forcefield files
        generated by RadonPy / Moltemplate, merges forcefields with
        forcefield_merger, prepares required TI ``blocks`` to write in
        simulation templates (e.g. compute fep arguments, charge set, etc.),
        and saves parameters that are built so they can be re-read on restart
        without re-processing files.

        :param system_dir:
        :type system_dir:
        :param sim_params: Simulation parameters required to run jobs. This
            is used to fill out the simulation templates passed during
            SolvationFreeEnergy class initialization.
        :type sim_params: dict
        :param system_params: parameters required for initial system
            preparation
        :type system_params: dict
        :param executables: dictionary for paths to required executables. For
            system_mode = ``prepare``, no executables are required. For
            system_mode = ``pack``, packmol is required. If
            ff_mode = ``moltemplate`` for any molecule, moltemplate and
            moltemplate_cleanup are required.
            Possible keys: ``packmol``, ``moltemp``, ``moltemp_cleanup``
        :type executables: dict
        :param scheduler: Orchestrator scheduler class for submitting jobs
        :type scheduler: Scheduler
        :param path_type: path to perform solvation free energy calculations
        :type path_type: str
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: boolean
        """

        save_params = {}
        atom_style = sim_params.get('atom_style')
        extra_pair_styles = sim_params.get("extra_pair_styles")
        extra_coeff_lines = sim_params.get("extra_coeff_lines")

        if self.system_mode == "pack":

            self.logger.info('SYSTEM MODE = Pack')
            solute_params = system_params.get('solute')
            solvent_params = system_params.get('solvent')
            molecule_list = [solute_params] + solvent_params

            # Run moltemplate for any molecules that require it
            mt_dir = self._run_moltemplate_pass(molecule_list, executables,
                                                scheduler, path_type)

            # Build solvent-only forcefield
            self.logger.info('Building solvent-only forcefield')
            solvent_styles, solvent_coeffs, solvent_reduced_names = \
                self._build_solvent_forcefield(
                    solvent_params, mt_dir, atom_style, extra_pair_styles,
                    extra_coeff_lines)

            # Merge with solute forcefield
            self.logger.info(
                'Merging solvent forcefield with solute forcefield')
            styles, coeffs, reduced_names = self._build_solute_forcefield(
                solute_params, mt_dir, solvent_styles, solvent_reduced_names,
                atom_style, sim_params, extra_pair_styles, extra_coeff_lines)

            style_strings = build_style_strings(styles)
            save_params = {**save_params, **style_strings}

            self.logger.info('Packing solvated system')
            pack_result = self._pack_solute_solvent_system(
                system_dir, molecule_list, reduced_names, system_params,
                atom_style, executables, random_seed_use)

            solute_types = pack_result["solute_types"]
            solvent_types = pack_result["solvent_types"]
            all_charges = pack_result["all_charges"]

            save_params["solute_molecule_id"] = 1

        elif self.system_mode == "prepared":

            self.logger.info('SYSTEM MODE = Prepared')

            solute_id = system_params.get('solute_molecule_id')
            save_params["solute_molecule_id"] = solute_id

            style_file = system_params.get('style_file')
            param_file = system_params.get('param_file')
            data_file = system_params.get('data_file')

            # Read forcefield from passed inputs
            self.logger.info('Reading prepared forcefield')
            cross_ps = sim_params.get("cross_pairstyle")
            override_cross = sim_params.get("override_cross_pairstyle")
            styles, coeffs, datafiles = forcefield_merger(
                style_files=style_file,
                params_files=param_file,
                data_files=data_file,
                atom_style=atom_style,
                extra_pair_styles=extra_pair_styles,
                extra_coeff_lines=extra_coeff_lines,
                mixing_rule="arithmetic",
                cross_pairstyle=cross_ps,
                override_cross_ps=override_cross,
                stack_ff=False,
                outparams=f"{self.forcefield_path}/solvent.in.settings",
                outstyle=f"{self.forcefield_path}/solvent.in.init")

            self.logger.info('Reading prepared LAMMPs .data file')
            construction, topology, box = read_lammps_data(
                datafiles[0], atom_style=atom_style)

            # Make solute_molecule_id into unique type in .data file
            self.logger.info(
                'Reconfiguring .data file so solute atom types are unique')
            costruction, topology, mapping = make_molecule_unique(
                construction=construction,
                topology=topology,
                target_mol_id=solute_id,
                filepath=f'{system_dir}/system_packed.data',
                box=box,
                atom_style=atom_style)

            self.logger.info(
                'Altering _coeff parameters to account for new types')
            coeffs["pair"] = self._clone_pair_coefficients(coeffs, mapping)
            write_parameter_file(
                styles=styles,
                coeffs=coeffs,
                extra_coeff_lines=extra_coeff_lines,
                outfile=f'{self.forcefield_path}/system_packed.in.settings')

            style_strings = build_style_strings(styles)
            save_params = {**save_params, **style_strings}

            self.logger.info('Re-reading solute, solvent types & charges')
            solvent_types, solute_types, all_types, all_charges = \
                self._parse_types_and_charges_from_data(
                    f"{self.init_structure}",
                    solute_id,
                    atom_style,
                    self.logger)

        # Regardless of mode, these are needed
        has_charges = any(abs(q) > 0.01 for q in all_charges.values())
        charged_solute = has_charges and any(
            abs(all_charges[s]) > 0.01 for s in solute_types)
        save_params['charged_solute'] = charged_solute

        charge_file = f'{self.forcefield_path}/system_packed.in.charges'
        self.logger.info(f'Writing charge file: {charge_file}')
        self._write_charge_file(sorted(set(solute_types + solvent_types)),
                                all_charges, charge_file)

        self.logger.info(
            'Altering forcefield parameters to include soft pairstyles')
        self._adjust_soft_pair_coeffs(styles, coeffs, solvent_types,
                                      solute_types, sim_params)

        soft_param_file = f"{self.forcefield_path}/ti_vdw.in.settings"
        self.logger.info('Writing new forcefield with soft pairstyles')
        write_parameter_file(styles=styles,
                             coeffs=coeffs,
                             extra_coeff_lines=extra_coeff_lines,
                             outfile=soft_param_file)
        save_params["soft_pair_style"] = build_style_strings(styles).get(
            "pair_style")

        self.logger.info('Building FEP purturbation lines')
        fep_lines = self.build_soft_fep_lines(coeffs, solvent_types,
                                              solute_types)
        charge_lines = self._build_charge_modify(solute_types, all_charges)

        save_params = {**save_params, **fep_lines}
        save_params = {**save_params, **charge_lines}

        self.logger.info(f'Writing {self.forcefield_path}/saved_params.json'
                         ' in case of restart')
        with open(f"{self.forcefield_path}/saved_params.json", 'w') as f:
            json.dump(save_params, f, indent=2)

        return sim_params

    def _clone_pair_coefficients(self, coeffs, mapping):
        """
        Alter pair coefficients to create a new, unique atom type which is a
        direct copy of an existing type.

        :param coeffs: dictionary of coefficients with keys ``pair``, ``bond``,
        ``angle``, ``dihedral``, ``improper``. The ``pair`` dictionary has a
        two-layered dictionary structure with the first layer corresponding to
        atom type i, and the second layer corresponding to atom type j, where i
        is always less than or equal to j. The other dictionaries have integer
        keys with the integer corresponding to the correspond bond_type,
        angle_type, etc. The final nested dictionary has keys (substyle, list)
        where substyle is the substyle that the coefficeints are used for
        (i.e. lj/cut for ``pair``, harmonic for ``bond``, etc.), and list is a
        list of strings in the *_coeff line, minus the substyle (when included
        in the *_coeff line for hybrid *_styles).
        :type coeffs: dict
        :param mapping: dictinary with keys being the new atom types and
            their corresponding values being the original atom types
        :type mapping: dict

        Duplicate pair coefficients so that any new (mapped) atom type
        inherits the coefficients of the original type it was split from.
        """
        new_pair = {}
        for old_i, row in coeffs["pair"].items():
            for old_j, coeff in row.items():
                substyle, parts = coeff
                i_variants = [old_i] + ([mapping[old_i]]
                                        if old_i in mapping else [])
                j_variants = [old_j] + ([mapping[old_j]]
                                        if old_j in mapping else [])

                for i in i_variants:
                    for j in j_variants:
                        new_i, new_j = min(i, j), max(i, j)
                        new_parts = list(parts)
                        new_parts[1] = str(i)
                        new_parts[2] = str(j)
                        new_pair.setdefault(new_i,
                                            {})[new_j] = (substyle, new_parts)

        return {
            i: dict(sorted(row.items()))
            for i, row in sorted(new_pair.items())
        }

    def _run_moltemplate_pass(self, molecule_list, executables, scheduler,
                              path_type):
        """
        Run moltemplate for any molecule in molecule_list whose ff_mode is
        ``moltemplate``. Mutates each such molecule dict in place, replacing
        the ``structure`` field with the generated .data file path, and
        appending ``base_name`` for later use.

        :param molecule_list: list of dictionaries, each of which including:
            ``ff_mode``: 'moltemplate' or anything else
            ``forcefield``: path to .lt file
            ``class``: name of class to use from .lt file,
            ``structure``: .pdb file to use for molecule structure.
        :type molecule_list: list of dict
        :param executables: dictionary for paths to required executables. For
            system_mode = ``prepare``, no executables are required. For
            system_mode = ``pack``, packmol is required. If
            ff_mode = ``moltemplate`` for any molecule, moltemplate and
            moltemplate_cleanup are required.
            Possible keys: ``packmol``, ``moltemp``, ``moltemp_cleanup``
        :type executables: dict
        :param scheduler: Orchestrator scheduler class for submitting jobs
            and constructing working-directory paths.
        :type scheduler: Scheduler
        :param path_type: path to perform solvation free energy calculations,
            used to build the moltemplate working directory.
        :type path_type: str

        :returns: The moltemplate working directory, or None if no molecules
            required moltemplate.
        :rtype: str or None
        """
        mt_dir = None
        mt_exe = None
        mt_clean_exe = None
        mt_count = 1

        for m in molecule_list:
            ff_mode = m.get('ff_mode')
            if ff_mode.lower() != "moltemplate":
                continue

            if mt_dir is None:
                mt_dir = scheduler.make_path_base(self.__class__.__name__,
                                                  f'{path_type}/Moltemplate')
                mt_exe = executables.get('moltemp')
                mt_clean_exe = executables.get('moltemp_cleanup')

            datafile = self.run_single_moltemplate(mt_dir,
                                                   m.get('forcefield'),
                                                   m.get('class'),
                                                   m.get('structure'),
                                                   structure_type="pdb",
                                                   atom_style="full",
                                                   name=f"mol{mt_count}",
                                                   moltemp_exe=mt_exe,
                                                   cleanup_exe=mt_clean_exe)

            m['structure'] = os.path.realpath(datafile)
            m['base_name'] = f"mol{mt_count}"
            mt_count += 1

        return mt_dir

    def _build_solvent_forcefield(self, solvent_params, mt_dir, atom_style,
                                  extra_pair_styles, extra_coeff_lines):
        """
        Build the merged forcefield for all solvent molecules, grouping
        molecules that share the same forcefield.

        For each solvent molecule, style/parameter/data files are resolved
        based on its ``ff_mode`` (``radonpy`` or ``moltemplate``). Molecules
        sharing the same ``forcefield`` value are grouped so their data files
        are merged under a single set of style/parameter files. The resulting
        unique forcefields are combined into a single merged solvent forcefield
        via ``forcefield_merger``, written to ``solvent.in.settings`` and
        ``solvent.in.init`` under ``self.forcefield_path``.

        :param solvent_params: List of solvent molecule dictionaries, each
            including ``ff_mode``, ``forcefield``, ``structure``, and (for
            ``ff_mode="moltemplate"``) ``base_name``.
        :type solvent_params: list of dict
        :param mt_dir: Moltemplate working directory containing generated
            ``.in.init``/``.in.settings`` files, used when a molecule's
            ``ff_mode`` is ``moltemplate``.
        :type mt_dir: str
        :param atom_style: LAMMPS atom style to use when merging forcefields.
        :type atom_style: str
        :param extra_pair_styles: Additional pair styles to include in the
            merged forcefield.
        :type extra_pair_styles: list
        :param extra_coeff_lines: Additional coefficient lines to include in
            the merged forcefield.
        :type extra_coeff_lines: list

        :returns: The ``(styles, coeffs, reduced_names)`` tuple produced by
            ``forcefield_merger`` for the merged solvent forcefield.
        :rtype: tuple
        """
        unique_forcefields = {}

        for m in solvent_params:
            ff_mode = m.get('ff_mode')
            forcefield = m.get('forcefield')
            structure = m.get('structure')

            if forcefield in unique_forcefields:
                unique_forcefields[forcefield]["data_files"].append(structure)
                continue

            if ff_mode.lower() == "radonpy":
                style_file = f"{forcefield}/lammps_forcefield_style.lmp"
                paramslist = f"{forcefield}/lammps_forcefield_paramlist.lmp"
                if not (isfile(style_file) and isfile(paramslist)):
                    raise ValueError(
                        f"ff_mode={ff_mode}, but lammps_forcefield_style.lmp"
                        "and lammps_forcefield_paramslist.lmp appear not to"
                        " exist")
                style_files = style_file
                params_files = self.parse_paramslist_file(paramslist)
                data_files = [structure]

            elif ff_mode.lower() == "moltemplate":
                base_name = m.get("base_name")
                style_files = f"{mt_dir}/{base_name}.in.init"
                params_files = [f"{mt_dir}/{base_name}.in.settings"]
                if not (isfile(style_files) and isfile(params_files[0])):
                    raise ValueError(
                        f"ff_mode={ff_mode}, but {mt_dir}/{base_name}.in.init"
                        f" and {params_files[0]} appear not to exist")
                data_files = [structure]

            else:
                raise ValueError(f"Unknown ff_mode, got {ff_mode}, "
                                 "expected ``moltemplate`` or ``radonpy``")

            unique_forcefields[forcefield] = {
                "style_files": style_files,
                "params_files": params_files,
                "data_files": data_files
            }

        solvent_data_files = [
            tuple(ff["data_files"]) for ff in unique_forcefields.values()
        ]
        solvent_style_files = [
            ff["style_files"] for ff in unique_forcefields.values()
        ]
        solvent_params_files = [
            tuple(ff["params_files"]) for ff in unique_forcefields.values()
        ]

        return forcefield_merger(
            style_files=solvent_style_files,
            params_files=solvent_params_files,
            data_files=solvent_data_files,
            atom_style=atom_style,
            extra_pair_styles=extra_pair_styles,
            extra_coeff_lines=extra_coeff_lines,
            stack_ff=False,
            outparams=f"{self.forcefield_path}/solvent.in.settings",
            outstyle=f"{self.forcefield_path}/solvent.in.init")

    def _build_solute_forcefield(self, solute_params, mt_dir, solvent_styles,
                                 solvent_reduced_names, atom_style, sim_params,
                                 extra_pair_styles, extra_coeff_lines):
        """
        Get the solute's forcefield files and merge them with the already-built
        solvent forcefield, forcing the solute to remain unique (stack_ff=True)

        Style/parameter/data files for the solute are resolved based on its
        ``ff_mode`` (``radonpy`` or ``moltemplate``) and merged, via
        ``forcefield_merger``, with the previously written solvent forcefield
        (``solvent.in.init`` / ``solvent.in.settings``) using arithmetic
        mixing rules. The cross pair style for solute-solvent interactions is
        taken from ``sim_params["cross_pairstyle"]`` if provided, otherwise
        defaults to the last solvent pair style. The merged result is written
        to ``system_packed.in.settings`` and ``system_packed.in.init`` under
        ``self.forcefield_path``.

        :param solute_params: Solute molecule dictionary, including
            ``ff_mode``, ``forcefield``, ``structure``, and (for
            ``ff_mode="moltemplate"``) ``base_name``.
        :type solute_params: dict
        :param mt_dir: Moltemplate working directory containing generated
            ``.in.init``/``.in.settings`` files, used when the solute's
            ``ff_mode`` is ``moltemplate``.
        :type mt_dir: str
        :param solvent_styles: Merged solvent style dictionary (as returned by
            ``_build_solvent_forcefield``), used to determine the default
            cross pair style.
        :type solvent_styles: dict
        :param solvent_reduced_names: Reduced solvent data file names/paths
            (as returned by ``_build_solvent_forcefield``) to merge against.
        :type solvent_reduced_names: list
        :param atom_style: LAMMPS atom style to use when merging forcefields.
        :type atom_style: str
        :param sim_params: Simulation parameters, used to look up
            ``cross_pairstyle`` and ``override_cross_pairstyle`` overrides.
        :type sim_params: dict
        :param extra_pair_styles: Additional pair styles to include in the
            merged forcefield.
        :type extra_pair_styles: list
        :param extra_coeff_lines: Additional coefficient lines to include in
            the merged forcefield.
        :type extra_coeff_lines: list

        :returns: The ``(styles, coeffs, reduced_names)`` tuple produced by
            ``forcefield_merger`` for the merged solute/solvent forcefield.
        :rtype: tuple
        """
        ff_mode = solute_params.get('ff_mode')
        forcefield = solute_params.get('forcefield')
        structure = solute_params.get('structure')

        if ff_mode.lower() == "radonpy":
            solute_styles_file = f"{forcefield}/lammps_forcefield_style.lmp"
            solute_params_file = tuple(
                self.parse_paramslist_file(
                    f"{forcefield}/lammps_forcefield_paramlist.lmp"))
            solute_data_file = tuple([structure])
        elif ff_mode.lower() == "moltemplate":
            base_name = solute_params.get("base_name")
            solute_styles_file = f"{mt_dir}/{base_name}.in.init"
            solute_params_file = tuple([f"{mt_dir}/{base_name}.in.settings"])
            solute_data_file = tuple([structure])
        else:
            raise ValueError(f"Unknown ff_mode for solute, got {ff_mode}, "
                             "expected ``moltemplate`` or ``radonpy``")

        cross_ps = sim_params.get("cross_pairstyle",
                                  solvent_styles["pair"]["styles"][-1])
        override_cross = sim_params.get("override_cross_pairstyle")

        return forcefield_merger(
            style_files=[
                solute_styles_file, f"{self.forcefield_path}/solvent.in.init"
            ],
            params_files=[
                solute_params_file,
                tuple([f"{self.forcefield_path}/solvent.in.settings"])
            ],
            data_files=[solute_data_file,
                        tuple(solvent_reduced_names)],
            atom_style=atom_style,
            extra_pair_styles=extra_pair_styles,
            extra_coeff_lines=extra_coeff_lines,
            mixing_rule="arithmetic",
            cross_pairstyle=cross_ps,
            override_cross_ps=override_cross,
            stack_ff=True,
            outparams=f"{self.forcefield_path}/system_packed.in.settings",
            outstyle=f"{self.forcefield_path}/system_packed.in.init")

    def _pack_solute_solvent_system(self, system_dir, molecule_list,
                                    reduced_names, system_params, atom_style,
                                    executables, random_seed_use):
        """
        Pack the reduced molecules into a box with packmol, then parse types
        and charges from the resulting data file.

        Molecule counts from ``molecule_list`` are paired with their
        corresponding reduced data files in ``reduced_names`` and packed into
        a cubic box of side length ``system_params["pack_boxlen"]`` using the
        given packing tolerance. Atom types and charges are then parsed from
        the resulting ``system_packed.data`` file, treating molecule ID 1 as
        the solute.

        :param system_dir: Directory in which to run packmol and write the
            packed system files.
        :type system_dir: str
        :param molecule_list: List of molecule dictionaries (solute followed
            by solvents, in the same order as ``reduced_names``), each
            including a ``number`` of copies to pack.
        :type molecule_list: list of dict
        :param reduced_names: Reduced data file names/paths corresponding
            elementwise to ``molecule_list``.
        :type reduced_names: list
        :param system_params: System/packing parameters, used to look up
            ``pack_tol`` and ``pack_boxlen``.
        :type system_params: dict
        :param atom_style: LAMMPS atom style used for the packed system and
            for subsequent type/charge parsing.
        :type atom_style: str
        :param executables: Mapping of executable names to paths; used to
            look up the ``packmol`` executable.
        :type executables: dict
        :param scheduler: Orchestrator scheduler class (currently unused
            directly, reserved for job submission).
        :type scheduler: Scheduler
        :param path_type: Path to perform solvation free energy calculations
            (currently unused directly, reserved for path construction).
        :type path_type: str
        :param random_seed_use: If True, a random packing seed is generated;
            if False, ``self.default_seed`` is used.
        :type random_seed_use: bool

        :returns: A dictionary with keys ``system_dir``, ``solvent_types``,
            ``solute_types``, ``all_types``, and ``all_charges``, as parsed
            from the packed system data file.
        :rtype: dict
        """

        pack_tol = system_params.get('pack_tol')
        pack_boxlen = system_params.get('pack_boxlen')

        packing_list = [{
            'structure': data,
            'number': mol['number']
        } for data, mol in zip(reduced_names, molecule_list)]

        packmol_exe = executables.get('packmol')
        random_seed = random.randint(
            1, 10000) if random_seed_use else self.default_seed

        pack_system(molecules=packing_list,
                    box=(0, pack_boxlen, 0, pack_boxlen, 0, pack_boxlen),
                    tolerance=pack_tol,
                    pack_dir=system_dir,
                    packmol_exe=packmol_exe,
                    atom_style=atom_style,
                    seed=random_seed,
                    lammps_file=True)

        solvent_types, solute_types, all_types, all_charges = \
            self._parse_types_and_charges_from_data(
                f"{system_dir}/system_packed.data", 1, atom_style, self.logger)

        return {
            "solvent_types": solvent_types,
            "solute_types": solute_types,
            "all_types": all_types,
            "all_charges": all_charges,
        }

    def _adjust_soft_pair_coeffs(self, styles, coeffs, solvent_types,
                                 solute_types, sim_params):
        """
        Rewrite solvent-solute cross pair coefficients to use the FEP soft-core
        style, adding a new hybrid/overlay pair style entry when needed.

        For every solute-solvent cross pair type combination whose base pair
        style (after stripping any accelerator suffix) appears in
        ``self.fep_soft_map``, the pair coefficient entry is rewritten to use
        the corresponding soft-core vdW style, with a ``${lambda_vdw}``
        argument appended. If that soft-core style is not already declared
        among the existing pair styles, a new pair style line is constructed
        by copying the matching base-style declaration, substituting the
        soft-core style name, and inserting the configured soft-core
        arguments (from ``sim_params["soft_args"]``); the pair style section
        is switched to ``hybrid/overlay`` if it was not already a hybrid
        style. Mutates ``styles`` and ``coeffs`` in place, and may append an
        accelerator suffix (``-sf omp`` / ``-k on -sf kk``) to
        ``self.built_simulator.code_path``.

        :param styles: Merged forcefield style dictionary (as produced by
            ``forcefield_merger``), containing the ``"pair"`` styles list and
            hybrid flag. Mutated in place.
        :type styles: dict
        :param coeffs: Merged forcefield pair coefficients dictionary (as
            produced by ``forcefield_merger``), keyed by atom-type pair.
            Mutated in place.
        :type coeffs: dict
        :param solvent_types: Atom type IDs belonging to the solvent.
        :type solvent_types: iterable of int
        :param solute_types: Atom type IDs belonging to the solute.
        :type solute_types: iterable of int
        :param sim_params: Simulation parameters, used to look up
            ``soft_args`` values for each required soft-core argument.
        :type sim_params: dict

        :returns: None. ``styles`` and ``coeffs`` are mutated in place.
        :rtype: None
        """
        base_names = [s.split()[0] for s in styles["pair"]["styles"]]

        for i in coeffs["pair"]:
            for j in coeffs["pair"][i]:
                is_cross = (i in solvent_types and j in solute_types) or \
                    (j in solvent_types and i in solute_types)
                if not is_cross:
                    continue

                substyle, args = coeffs["pair"][i][j]
                base_style, suffix = self._strip_accelerator_suffix(substyle)

                if base_style not in self.fep_soft_map:
                    continue

                substyle = self.fep_soft_map[base_style]['vdw']
                args.append("${lambda_vdw}")
                coeffs["pair"][i][j] = (substyle, args)

                if substyle in base_names:
                    continue

                copy_style = next((style for style in styles["pair"]["styles"]
                                   if self._strip_accelerator_suffix(
                                       style.split()[0])[0] == base_style),
                                  None)
                if copy_style is None:
                    raise ValueError(
                        f"Could not find an existing pair style matching "
                        f"base style '{base_style}' to copy.")

                parts = copy_style.split()
                parts[0] = substyle
                for offset, arg in enumerate(
                        self.fep_soft_map[base_style]["soft_args"]):
                    parts.insert(1 + offset,
                                 str(sim_params.get("soft_args").get(arg)))
                styles["pair"]["styles"].append(" ".join(parts))

                if styles["pair"]["hybrid"] is False:
                    styles["pair"]["hybrid"] = "hybrid/overlay"
                base_names = [s.split()[0] for s in styles["pair"]["styles"]]

    def _parse_types_and_charges_from_data(
        self,
        data_file_path,
        solute_molecule_id=1,
        atom_style="full",
        logger=None,
    ):
        """
        Parse atom types and charges from a LAMMPS data file's Atoms section.

        Supports 'full' (atom-ID mol-ID atom-type charge x y z) and 'molecular'
        (atom-ID mol-ID atom-type x y z) atom styles; accelerator suffixes
        (``/omp``, ``/kk``) on ``atom_style`` are stripped before checking.
        Atom counts and atom-type counts are read from the data file's header
        lines, and exactly that many non-blank atom lines are then read from
        the Atoms section. Atoms belonging to ``solute_molecule_id`` are
        classified as solute; all other atoms are classified as solvent.

        :param data_file_path: Path to the LAMMPS data file to parse.
        :type data_file_path: str
        :param solute_molecule_id: Molecule ID in the data file's Atoms
            section that identifies the solute; all other molecule IDs are
            treated as solvent.
        :type solute_molecule_id: int, optional
        :param atom_style: LAMMPS atom style used in the data file. Supported
            base styles are "full" and "molecular" (optionally suffixed with
            ``/omp`` or ``/kk``).
        :type atom_style: str, optional
        :param logger: Logger whose ``.info`` method is used to report parsed
            atom types. If None, results are printed instead.
        :type logger: logging.Logger or None, optional

        :returns:
            A tuple ``(solvent_types, solute_types, all_types, all_charges)``:
            sorted lists of solvent atom types, solute atom types, and all
            atom types (1 through the declared number of atom types), and a
            dict mapping atom type to charge (empty if ``atom_style`` has no
            charge column).
        :rtype: tuple
        """

        log = logger.info if logger is not None else print

        atom_style = atom_style[:-len('/omp')] if atom_style.endswith(
            '/omp') else atom_style
        atom_style = atom_style[:-len('/kk')] if atom_style.endswith(
            '/kk') else atom_style

        if atom_style not in ("full", "molecular"):
            raise ValueError("_parse_types_and_charges_from_data only supports"
                             "'full' or 'molecular' atom_style, "
                             f"got '{atom_style}'")
        has_charge = atom_style == "full"

        solute_types = set()
        solvent_types = set()
        all_charges = {}
        num_atoms = None
        num_atom_types = None

        with open(data_file_path) as f:
            lines = f.readlines()

        # Find declared counts from the header
        for line in lines:
            parts = line.split("#")[0].split()
            if len(parts) >= 2:
                if parts[1] == "atoms":
                    num_atoms = int(parts[0])
                elif parts[1] == "atom" and len(
                        parts) >= 3 and parts[2] == "types":
                    num_atom_types = int(parts[0])

            if num_atoms is not None and num_atom_types is not None:
                break

        if num_atoms is None:
            raise ValueError(
                f"Could not find 'N atoms' header line in {data_file_path}")
        if num_atom_types is None:
            raise ValueError(f"Could not find 'N atom types' header line "
                             f"in {data_file_path}")
        all_types = set(range(1, num_atom_types + 1))

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

        return sorted(solvent_types), sorted(solute_types), sorted(
            all_types), all_charges

    def _build_charge_modify(self, solute_types, all_charges):
        """
        Build LAMMPS input-script snippets for scaling solute atom charges
        during a free-energy perturbation run.

        For each solute atom type, a ``set type`` command scaling the charge
        by ``v_lambda_c`` is generated, along with LAMMPS variables for the
        forward and backward per-lambda-step charge increments
        (``v_lambda_diff`` / ``v_nlambda_diff``), and corresponding
        ``atom charge`` lines for the forward and backward FEP ``compute
        fep`` blocks.

        :param solute_types: Atom type IDs belonging to the solute.
        :type solute_types: iterable of int
        :param all_charges: Mapping of atom type to its base charge.
        :type all_charges: dict

        :returns:
            A dictionary with keys ``"charge_modification_block"`` (the
            ``set type ... charge`` commands scaling solute charges by
            ``v_lambda_c``), ``"delta_charge_block"`` (variable definitions
            for the forward/backward charge increments), ``"fep_elec_forward"``
            (the forward ``atom charge`` lines for a ``compute fep`` block),
            and ``"fep_elec_backward"`` (the corresponding backward lines).
        :rtype: dict
        """

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

    def parse_paramslist_file(self, file):
        """
        Extract the paths of files referenced by ``include`` directives in a
        LAMMPS parameter list file.

        Each line is stripped of comments (text following ``#``); blank lines
        are skipped. Lines whose first token is ``include`` contribute their
        second token, resolved relative to the directory containing ``file``,
        to the returned list.

        :param file: Path to the parameter list file to parse.
        :type file: str

        :returns: Paths of the files referenced by ``include`` directives,
            each resolved relative to the directory containing ``file``.
        :rtype: list of str
        """
        dir = os.path.dirname(file)
        files = []
        with open(file) as f:
            for line in f:
                clean = line.split('#')[0].strip().split()
                if not clean:
                    continue
                if clean[0] == "include":
                    files.append(f'{dir}/{clean[1]}')
        return files

    def _strip_accelerator_suffix(self, style):
        """
        Remove a supported LAMMPS accelerator suffix from a pair style.

        The ``/kk`` and ``/omp`` suffixes are recognized. If no supported
        suffix is present, the original style is returned unchanged.

        :param style: Pair style name that may contain an accelerator suffix.
        :type style: str

        Returns the base pair style and the removed suffix.

        :returns:
            A tuple ``(base_style, suffix)``. ``suffix`` is an empty string
            when no supported accelerator suffix is present.
        :rtype: tuple
        """
        for suffix in ("/kk", "/omp"):
            if style.endswith(suffix):
                return style[:-len(suffix)], suffix
        return style, ""

    def build_soft_fep_lines(self, coeffs, solvent_types, solute_types):
        """
        Write pair coefficients and free-energy parameters for a soft-core
        LAMMPS simulation.

        Pair styles are assigned to solute-solvent interactions and converted
        to the corresponding soft-core styles defined in ``fep_soft_map``.
        Forward and backward free-energy perturbation ``pair`` commands are
        generated for each solute-solvent type pair. The resulting parameter
        file is then written using the configured coefficient data.

        The soft pair style is also constructed from the configured base pair
        style. Soft-core parameters are added according to the corresponding
        entry in ``fep_soft_map``, while applicable existing pair-style
        arguments are preserved. Accelerator suffixes such as ``/kk`` and
        ``/omp`` are removed when determining the corresponding soft-core
        mapping.

        :param coeffs: Mapping of atom-type pairs to pair coefficient
            entries. Entries have the form
            ``(substyle, parts)``.
        :type coeffs: dict
        :param solvent_types: Atom type IDs belonging to the solvent.
        :type solvent_types: list
        :param solute_types: Atom type IDs belonging to the solute.
        :type solute_types: list

        Returns the free-energy pair commands and soft pair style needed by
        the subsequent LAMMPS simulation.

        :returns:
            A dictionary containing ``"fep_vdw_forward"``,
            and ``"fep_vdw_backward"``.
        :rtype: dict
        """

        prefix = "                "
        compute_fep_forward = ""
        compute_fep_backward = ""
        pairs = [(min(i, j), max(i, j)) for i in solute_types
                 for j in solvent_types]
        for idx, (i, j) in enumerate(pairs):
            i, j = min(i, j), max(i, j)
            soft_style, parts = coeffs["pair"][i][j]
            suffix = '\n' if idx == len(pairs) - 1 else ' &\n'
            compute_fep_forward += (
                f'{prefix}pair {soft_style} lambda {i} {j} '
                f'v_lambda_diff{suffix}')
            compute_fep_backward += (
                f'{prefix}pair {soft_style} lambda {i} {j} '
                f'v_nlambda_diff{suffix}')

        return {
            "fep_vdw_forward": compute_fep_forward,
            "fep_vdw_backward": compute_fep_backward
        }

    def _try_float(self, val):
        """
        Try taking ``float()`` of an input value without raising an error.

        :param val: Value to attempt conversion to a float.
        :type val: str

        Returns the converted float when conversion succeeds, or ``None``
        when the value cannot be interpreted as a float.

        :rtype: float or None
        """
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _try_int(self, val):
        """
        Try taking ``int()`` of an input value without raising an error.

        :param val: Value to attempt conversion to a int.
        :type val: str

        Returns the converted float when conversion succeeds, or ``None``
        when the value cannot be interpreted as a int.

        :rtype: float or None
        """
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _tokens_equal(self, a, b):
        """
        Compare two whitespace-delimited strings token by token for equality.

        Each string is split on whitespace, and the resulting tokens are
        compared pairwise: tokens that both parse as floats are compared
        numerically (so ``"1.0"`` and ``"1"`` are considered equal), and all
        other tokens are compared as plain strings.

        :param a: First string to compare.
        :type a: str
        :param b: Second string to compare.
        :type b: str

        :returns: True if ``a`` and ``b`` have the same number of tokens and
            every corresponding pair of tokens is equal (numerically for
            tokens that parse as floats, otherwise as strings), False
            otherwise.
        :rtype: bool
        """
        a_parts = a.split()
        b_parts = b.split()

        if len(a_parts) != len(b_parts):
            return False

        for x, y in zip(a_parts, b_parts):
            try:
                if float(x) != float(y):
                    return False
            except (ValueError, TypeError):
                if x != y:
                    return False

        return True

    def _validate_inputs(self, sim_params, system_params, ti_params,
                         executables):
        """
        Validate simulation, solvation, and executable inputs before any
        packing, moltemplate, or simulation work begins.

        Simulation and solvation parameters are merged with their corresponding
        class defaults before validation. Required molecular structure files,
        simulation parameters, packing parameters, free-energy settings,
        lambda windows, and external executables are checked for valid values.
        Lambda values specified as an integer are converted to evenly spaced
        values between 0 and 1.

        :param sim_params: Simulation parameters to validate. Values not
            specified are taken from the class's default simulation parameters.
        :type sim_params: dict
        :param system_params: System/solvation parameters to validate,
            including packing settings and solute/solvent definitions (or,
            in "prepared" system mode, paths to pre-built LAMMPS input
            files). Values not specified are taken from the class's default
            system parameters.
        :type system_params: dict
        :param ti_params: Free-energy and thermodynamic-integration parameters
            to validate, including ``free_energy``, ``equil_frac``,
            ``lambda_values``, and ``lambda_diff``. Values not specified are
            taken from the class's default TI parameters.
        :type ti_params: dict
        :param executables: Mapping of external executable names (e.g.
            ``packmol``, ``moltemp``, ``moltemp_cleanup``) to the executable
            path or command to validate and use.
        :type executables: dict

        :returns:
            A tuple ``(sim_params, system_params, ti_params)`` containing the
            validated and normalized parameter dictionaries, including
            defaults and any normalized values such as lambda windows. The
            ``free_energy`` and ``equil_frac`` values are also copied into
            ``sim_params`` for downstream use.
        :rtype: tuple
        """
        errors = []

        # Merge inputs with defaults
        sim_params = {**self.default_sim_params, **sim_params}
        system_params = {**self.default_system_params, **system_params}
        ti_params = {**self.default_ti_params, **ti_params}

        # Check that none of these quantities are negative or missing
        for k in ('temp', 'press', 'temp_damp', 'press_damp', 'timestep',
                  'equil_steps', 'ti_steps'):
            v = sim_params.get(k)
            if v is None:
                errors.append(f"Simulation parameter {k} is missing")
            elif v < 0:
                errors.append(
                    f"Simulation parameter {k} cannot be < 0, got {v}")

        pack_tol = system_params.get('pack_tol')
        pack_boxlen = system_params.get('pack_boxlen')
        if not (pack_tol > 0 and pack_boxlen > 0):
            errors.append("pack_tol and pack_boxlen must both be > 0")

        free_energy = ti_params.get('free_energy')
        if free_energy in ('gibbs', 'helmholtz'):
            sim_params['free_energy'] = free_energy
        else:
            errors.append(f"free_energy must be ``helmholtz`` or "
                          f"``gibbs``, got {free_energy}")

        equil_frac = ti_params.get('equil_frac')
        if equil_frac < 0 or equil_frac >= 1:
            errors.append("equil_frac must be between 0 and 1, "
                          f"got {equil_frac}")
        else:
            sim_params['equil_frac'] = equil_frac

        lambda_values = ti_params.get('lambda_values')
        if isinstance(lambda_values, int):
            if lambda_values <= 1:
                errors.append(
                    f"lambda_values as int must be >= 2, got {lambda_values}")
            else:
                lambda_values = list(
                    np.round(np.linspace(0.00, 1.00, num=lambda_values), 5))
        elif isinstance(lambda_values, list):
            if not all(isinstance(x, (int, float)) for x in lambda_values):
                errors.append("lambda_values list must contain only numbers")
            else:
                lambda_values = [float(x) for x in lambda_values]
                if not any(abs(x - 0.0) < 1e-9 for x in lambda_values):
                    errors.append("lambda_values list must include 0.0")
                if not any(abs(x - 1.0) < 1e-9 for x in lambda_values):
                    errors.append("lambda_values list must include 1.0")
        else:
            errors.append(
                "lambda_values must be a list of floats or an int count")

        if not errors:
            ti_params['lambda_values'] = lambda_values
            self.logger.info(
                f"\tlambda windows: {[f'{lam:.5f}' for lam in lambda_values]}")

        lambda_diff = ti_params.get('lambda_diff')
        if lambda_diff is None or lambda_diff <= 0:
            errors.append(f"lambda_diff must be > 0, got {lambda_diff!r}")

        atom_style = sim_params.get('atom_style')
        if atom_style not in ("full", "molecular"):
            errors.append(
                f"atom_style must be 'full' or 'molecular', got {atom_style!r}"
            )

        system_mode = system_params.get('system_mode')
        if system_mode.lower() not in ("pack", "prepared"):
            errors.append(f"Unknown system_mode {system_mode},"
                          "expected ``pack`` or ``prepared``")
        else:
            self.system_mode = system_mode.lower()

        if self.system_mode == "pack":
            pack_exe = executables.get('packmol')
            use_moltemplate = False
            if pack_exe is None:
                errors.append("Packmol executable missing from executables")
            else:
                if shutil.which(pack_exe) is None:
                    errors.append(f"Packmol execuatable {pack_exe} "
                                  "is not in path")

            required_fields = {
                "class", "ff_mode", "forcefield", "structure", "number"
            }
            solute = system_params.get('solute')
            if not isinstance(solute, dict):
                errors.append("Expect solute with fields ``class``, "
                              "``ff_mode``, ``forcefield``, ``structure``")
            else:
                system_params['solute']['number'] = 1
                missing = required_fields - solute.keys()
                if missing:
                    errors.append("Expect solute with fields ``class``, "
                                  "``ff_mode``, ``forcefield``, ``structure``"
                                  f", solute missing fields: {missing}")
                else:
                    ff_mode = solute.get('ff_mode')
                    if ff_mode.lower() not in ('radonpy', 'moltemplate'):
                        errors.append("Unknown ff_mode for solute, "
                                      "expected ``radonpy`` or ``moltemplate``"
                                      f", got {ff_mode}")
                    elif ff_mode.lower() == "moltemplate":
                        use_moltemplate = True

            solvent = system_params.get('solvent')
            if isinstance(solvent, dict):
                solvent = [solvent]
            if isinstance(solvent, list):
                for i, s in enumerate(solvent):
                    missing = required_fields - s.keys()
                    if missing:
                        errors.append(
                            "Expect solvent with fields ``class``, "
                            "``ff_mode``, ``forcefield``, ``structure``"
                            f", and ``number``, solvent {i} missing"
                            f" fields: {missing}")
                    else:
                        ff_mode = s.get('ff_mode')
                        if ff_mode.lower() not in ('radonpy', 'moltemplate'):
                            errors.append(
                                f"Unknown ff_mode for solvent {i},"
                                "expected ``radonpy`` or ``moltemplate``"
                                f", got {ff_mode}")
                        elif ff_mode.lower() == "moltemplate":
                            use_moltemplate = True
                        number = s.get('number')
                        if not isinstance(number, int):
                            errors.append(
                                f"Non-integer ``number`` for solute {i},"
                                f" got {number}")
            else:
                errors.append(f"Solvent should be a list, got {type(solvent)}")

            if use_moltemplate is True:
                moltemp_exe = executables.get('moltemp')
                if moltemp_exe is None:
                    errors.append("Moltemplate executable missing.")
                else:
                    if shutil.which(moltemp_exe) is None:
                        errors.append(f"Moltemplate executable {moltemp_exe}"
                                      " not in path.")
                moltemp_cleanup_exe = executables.get('moltemp_cleanup')
                if moltemp_cleanup_exe is None:
                    errors.append("Moltemplate cleanup executable missing.")
                else:
                    if shutil.which(moltemp_cleanup_exe) is None:
                        errors.append("Moltemplate cleanup executable "
                                      f"{moltemp_cleanup_exe}, not in path.")

        elif self.system_mode == "prepared":
            required_fields = {
                "data_file", "style_file", "param_file", "solute_molecule_id"
            }
            missing = required_fields - system_params.keys()
            if missing:
                errors.append(
                    "System_params requires ``data_file``, ``style_file``, "
                    "``param_file``, ``solute_molecule_id`` in "
                    "'prepared' system_mode.")
            data_file = system_params.get('data_file')
            style_file = system_params.get('style_file')
            param_file = system_params.get('param_file')
            sol_id = system_params.get('solute_molecule_id')
            if not isinstance(style_file, str) or not isfile(data_file):
                errors.append(f"Data_file {data_file} not found.")
            if not isinstance(style_file, str) or not isfile(style_file):
                errors.append(f"Style_file {style_file} not found.")
            if not isinstance(param_file, str) or not isfile(param_file):
                errors.append(f"Param_file {param_file} not found.")
            if not isinstance(sol_id, int):
                errors.append(
                    f"Solute_molecule_id is not an int, got {sol_id}")

        if errors:
            raise ValueError("Invalid solvation free energy inputs:\n  - "
                             + "\n  - ".join(errors))
        else:
            self.system_mode = system_mode.lower()
            return sim_params, system_params, ti_params

    def _write_results_file(self, lambda_values, lens, dudl, dudl_std,
                            dudl_err, filepath):
        """
        Write lambda-window dU/dL results to a results file.

        Existing results are read from ``filepath`` and merged with the newly
        supplied values. When the same lambda value occurs in both sets, the
        two results are combined into a pooled mean, standard deviation, and
        standard error weighted by sample count. Results are written in
        ascending lambda order.

        :param lambda_values: Lambda values corresponding to each dU/dL result.
        :type lambda_values: list of float
        :param lens: Number of samples used to compute each dU/dL result,
            corresponding to each lambda window.
        :type lens: list of int
        :param dudl: Mean dU/dL values corresponding to each lambda window.
        :type dudl: list of float
        :param dudl_std: Standard deviations of the per-step or per-block
            dU/dL samples corresponding to each lambda window.
        :type dudl_std: list of float
        :param dudl_err: Standard errors of the mean dU/dL values, accounting
            for autocorrelation, corresponding to each lambda window.
        :type dudl_err: list of float
        :param filepath: Path to the results file to read and update.
        :type filepath: str

        :returns:
            A tuple ``(lambda_values, lens, dudl, dudl_std, dudl_err)``
            containing the merged results in ascending lambda order.
        :rtype: tuple
        """

        # read prior results, if any
        lambda_values_old, lens_old, dudl_old, dudl_std_old, dudl_err_old = \
            self._read_results_file(filepath)

        # merge old and new, letting new values overwrite old ones
        combined = {}
        for lam, n, val, std, err in zip(lambda_values_old, lens_old, dudl_old,
                                         dudl_std_old, dudl_err_old):
            combined[lam] = (n, val, std, err)
        for lam, n, val, std, err in zip(lambda_values, lens, dudl, dudl_std,
                                         dudl_err):
            if lam in combined:
                n_a, val_a, std_a, _ = combined[lam]
                n_b, val_b, std_b, _ = n, val, std, err
                n_total = n_a + n_b
                val_c = (n_a * val_a + n_b * val_b) / n_total
                weighted_var = (n_a * std_a**2 + n_b * std_b**2) / n_total
                cross_var = (n_a * n_b * (val_a - val_b)**2) / n_total**2
                pooled_var = weighted_var + cross_var
                std_c = np.sqrt(pooled_var)
                err_c = std_c / np.sqrt(n_total)
                combined[lam] = (n_total, val_c, std_c, err_c)
            else:
                combined[lam] = (n, val, std, err)
        sorted_lambdas = sorted(combined.keys())

        with open(filepath, 'w') as f:
            f.write(f'# {"lambda":>10s}'
                    f' {"sample_num":>15s}'
                    f'{"dudl":>15s}'
                    f'{"stddev":>15s}'
                    f'{"stderr":>15s}\n')
            for lam in sorted_lambdas:
                n, val, std, err = combined[lam]
                f.write(f'{lam:10.5f}'
                        f'{n:15d} '
                        f'{val:15.6f} '
                        f'{std:15.6f} '
                        f'{err:15.6f}\n')

        # build the merged, sorted output lists
        merged_lambda_values = sorted_lambdas
        merged_lens = [combined[lam][0] for lam in sorted_lambdas]
        merged_dudl = [combined[lam][1] for lam in sorted_lambdas]
        merged_dudl_std = [combined[lam][2] for lam in sorted_lambdas]
        merged_dudl_err = [combined[lam][3] for lam in sorted_lambdas]

        return (merged_lambda_values, merged_lens, merged_dudl,
                merged_dudl_std, merged_dudl_err)

    def _read_results_file(self, filepath):
        """
        Read lambda-window dU/dL results from a results file.

        Blank lines and comment lines beginning with ``#`` are ignored. Each
        remaining line is expected to contain a lambda value, sample count,
        dU/dL value, standard deviation, and standard error, in that order.

        :param filepath: Path to the results file to read.
        :type filepath: str

        :returns:
            A tuple ``(lambda_values, lens, dudl, dudl_std, dudl_err)``
            containing the values read from the results file.
        :rtype: tuple

        If the results file does not exist, five empty lists are returned.
        """
        lambda_values, lens, dudl, dudl_std, dudl_err = [], [], [], [], []

        if not os.path.exists(filepath):
            return lambda_values, lens, dudl, dudl_std, dudl_err

        with open(filepath, 'r') as r:
            for line in r:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                lambda_values.append(round(float(parts[0]), 5))
                lens.append(int(parts[1]))
                dudl.append(float(parts[2]))
                dudl_std.append(float(parts[3]))
                dudl_err.append(float(parts[4]))

        return lambda_values, lens, dudl, dudl_std, dudl_err

    def _get_production_stats(self, logpath, quantity, equil_frac=0.25):
        """
        Extract production statistics for a quantity from a LAMMPS log file.

        The requested property is extracted from the LAMMPS log and the final
        ``1 - equil_frac`` fraction of the time series is treated as production
        data. The production mean, standard deviation, and standard error of
        the mean are calculated from this portion of the trajectory.

        :param logpath: Path to the LAMMPS log file containing the requested
            property.
        :type logpath: str
        :param quantity: Name of the LAMMPS log quantity to extract.
        :type quantity: str
        :param equil_frac: fraction of the time series to discard as
            equilibration. The default discards the first 25 percent.
        :type equil_frac: float, optional

        :returns:
            A tuple ``(production_data, prod_len, mean, std, stderr)``
            containing the production time series, its length, mean,
            standard deviation, and standard error of the mean.
        :rtype: tuple

        If the log file cannot be parsed or the requested quantity cannot be
        extracted, ``(0.0, 0, 0.0, 0.0, 0.0)`` is returned.
        """
        try:
            _, timeseries, avg, std = \
                AnalyzeLammpsLog.extract_property([logpath, quantity])
            full_len = len(timeseries)
            prod_len = int((1 - equil_frac) * full_len)
            equil_data = timeseries[-prod_len:]
            return (equil_data, prod_len, np.mean(equil_data),
                    np.std(equil_data), np.std(equil_data) / np.sqrt(prod_len))
        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            self.logger.warning(
                f'Could not extract "{quantity}" from {logpath}: {e}')
            return (0.0, 0, 0.0, 0.0, 0.0)

    def write_moltemplate_file(self, lt_file, class_name, mt_dir, name="mol"):
        """
        Copy a single molecule's .lt file into mt_dir, rename its molecule
        class to `name`, and write a small combined .lt file that imports it
        and instantiates one copy.

        :param lt_file: Path to the source .lt file for the molecule.
        :type lt_file: str
        :param class_name: Name of the molecule class defined in ``lt_file``,
            to be renamed to ``name``.
        :type class_name: str
        :param mt_dir: Directory in which to place the copied and combined
            .lt files.
        :type mt_dir: str
        :param name: Name to assign to the molecule class and the resulting
            combined .lt file.
        :type name: str, optional

        :returns: Path to the written combined .lt file.
        :rtype: str
        """
        if lt_file is None or not os.path.isfile(lt_file):
            raise ValueError(
                f".lt file {lt_file} does not exist or was not provided.")
        if not class_name or class_name == "unknown":
            raise ValueError(f"unknown class in .lt file for file {lt_file}")

        lt_name = f'{name}.{os.path.basename(lt_file)}'
        dest_path = os.path.join(mt_dir, lt_name)

        shutil.copy(lt_file, dest_path)
        subprocess.run(["sed", "-i", f"s/{class_name}/{name}/g", dest_path])

        new_lt_file = f'{name}.lt'
        lt_file_path = os.path.join(mt_dir, new_lt_file)
        with open(lt_file_path, "w") as f:
            f.write(f'import "{lt_name}"\n')
            f.write(f'{name} = new {name}[1]\n')

        return lt_file_path

    def run_moltemplate(self,
                        mt_dir,
                        lt_file,
                        structure_file,
                        structure_type="pdb",
                        atom_style="full",
                        moltemp_exe="moltemplate.sh",
                        cleanup_exe="cleanup_moltemplate.sh"):
        """
        Run moltemplate on a single-molecule .lt file, run the moltemplate
        cleanup script, then apply any per-atom charges moltemplate wrote to
        a separate .in.charges file into the resulting .data file.

        :param mt_dir: Directory in which to run moltemplate; also where
            ``structure_file`` is copied to and outputs are written.
        :type mt_dir: str
        :param lt_file: Path to the .lt file to pass to moltemplate.
        :type lt_file: str
        :param structure_file: Path to the input structure file to copy into
            ``mt_dir`` and pass to moltemplate.
        :type structure_file: str
        :param structure_type: Structure file format flag passed to
            moltemplate (e.g. "pdb").
        :type structure_type: str, optional
        :param atom_style: LAMMPS atom style used for the generated data file
            and for charge assignment.
        :type atom_style: str, optional
        :param moltemp_exe: Name or path of the moltemplate executable.
        :type moltemp_exe: str, optional
        :param cleanup_exe: Name or path of the moltemplate cleanup
            executable.
        :type cleanup_exe: str, optional

        :returns: None
        :rtype: None
        """
        if shutil.which(moltemp_exe) is None:
            raise ValueError("moltemp executable not found "
                             f"or not executable: {moltemp_exe}")
        if shutil.which(cleanup_exe) is None:
            raise ValueError("moltemp_cleanup executable not found "
                             f"or not executable: {cleanup_exe}")

        lt_basename = os.path.basename(lt_file)
        base_name = os.path.splitext(lt_basename)[0]
        shutil.copy(structure_file, mt_dir)
        structure_filename = os.path.basename(structure_file)

        cmd = [
            moltemp_exe, f"-{structure_type}", structure_filename,
            "-atomstyle", atom_style, lt_basename
        ]

        result = subprocess.run(cmd,
                                cwd=mt_dir,
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Moltemplate failed "
                f"(return code {result.returncode}): {result.stderr}")

        result = subprocess.run([cleanup_exe, "-base", base_name],
                                cwd=mt_dir,
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Moltemplate cleanup failed "
                f"(return code {result.returncode}): {result.stderr}")

        self.apply_charges_file(mt_dir=mt_dir,
                                base_name=base_name,
                                atom_style=atom_style)

    def apply_charges_file(self, mt_dir, base_name, atom_style="full"):
        """
        If moltemplate wrote a system_{name}.in.charges file (containing lines
        like 'set type 1 charge -0.834'), parse the atom-type -> charge
        mapping and apply it to every atom of that type in the charge column
        of system_{name}.data's Atoms section.

        :param mt_dir: Directory containing the charges file and data file.
        :type mt_dir: str
        :param base_name: Base name shared by the ``.in.charges`` and
            ``.data`` files (e.g. ``"system_mol"``).
        :type base_name: str
        :param atom_style: LAMMPS atom style, used to determine which
            columns in the Atoms section hold the atom type and charge.
            Supported values are "full" and "charge".
        :type atom_style: str, optional

        :returns: True if a charges file was found and applied, False if no
            charges file exists (nothing to do -- not an error, since
            moltemplate only emits this file when the .lt file sets
            per-type charges outside of the pair_coeff/force-field
            defaults).
        :rtype: bool
        """
        charges_file = os.path.join(mt_dir, f'{base_name}.in.charges')
        data_file = os.path.join(mt_dir, f'{base_name}.data')

        if not os.path.isfile(charges_file):
            return False
        if not os.path.isfile(data_file):
            raise ValueError(
                f"Found {charges_file} but matching data file {data_file} "
                "does not exist.")

        if atom_style.lower() == 'full':
            atom_type_column = 2
            charge_column = 3
        elif atom_style.lower() == 'charge':
            atom_type_column = 1
            charge_column = 1
        else:
            raise ValueError(
                f"Unsupported atom style for charge assignment: {atom_style}")

        # Parse "set type <type_id> charge <value>" lines
        charge_map = {}
        with open(charges_file) as f:
            for line in f:
                clean = line.split('#', 1)[0].strip()
                if not clean:
                    continue
                parts = clean.split()
                if len(parts) != 5 or parts[0].lower() != 'set' or parts[
                        1].lower() != 'type' or parts[3].lower() != 'charge':
                    raise ValueError(
                        f"Could not parse charges line in {charges_file}: "
                        f"{line!r}")
                type_id = self._try_int(parts[2])
                charge = self._try_float(parts[4])
                if type_id is None or charge is None:
                    raise ValueError(
                        f"Could not parse charges line in {charges_file}: "
                        f"{line}")
                charge_map[type_id] = charge

        if not charge_map:
            return False

        with open(data_file) as f:
            lines = f.readlines()

        section = None
        section_names = (
            'Masses',
            'Atoms',
            'Velocities',
            'Bonds',
            'Angles',
            'Dihedrals',
            'Impropers',
            'Pair Coeffs',
            'Bond Coeffs',
            'Angle Coeffs',
            'Dihedral Coeffs',
            'Improper Coeffs',
        )

        new_lines = []
        applied_types = set()
        for line in lines:
            clean = line.split('#', 1)[0].strip()

            if clean in section_names:
                section = clean
                new_lines.append(line)
                continue

            if section == 'Atoms' and clean:
                parts = clean.split()
                atom_type = int(parts[atom_type_column])
                if atom_type in charge_map:
                    parts[charge_column] = str(charge_map[atom_type])
                    applied_types.add(atom_type)
                    new_lines.append(''.join(f'{x:<6}' for x in parts[:3])
                                     + ''.join(f'{x:<25}'
                                               for x in parts[3:]) + '\n')
                    continue

            new_lines.append(line)

        missing = set(charge_map) - applied_types
        if missing:
            raise ValueError(
                f"Charges specified for atom type(s) {sorted(missing)} in "
                f"{charges_file}, but these types were not found in the "
                f"Atoms section of {data_file}.")

        with open(data_file, 'w') as f:
            f.writelines(new_lines)

        return True

    def run_single_moltemplate(self,
                               mt_dir,
                               lt_file,
                               class_name,
                               structure_file,
                               structure_type="pdb",
                               atom_style="full",
                               name="mol",
                               moltemp_exe="moltemplate.sh",
                               cleanup_exe="cleanup_moltemplate.sh"):
        """
        Full single-molecule moltemplate run: copy/rename the .lt file, write
        the tiny import+instantiate .lt file, then run moltemplate + cleanup
        (which also applies any charges file).

        :param mt_dir: Directory in which to run moltemplate and write
            outputs.
        :type mt_dir: str
        :param lt_file: Path to the source .lt file for the molecule.
        :type lt_file: str
        :param class_name: Name of the molecule class defined in ``lt_file``.
        :type class_name: str
        :param structure_file: Path to the input structure file for the
            molecule.
        :type structure_file: str
        :param structure_type: Structure file format flag passed to
            moltemplate (e.g. "pdb").
        :type structure_type: str, optional
        :param atom_style: LAMMPS atom style used for the generated data file.
        :type atom_style: str, optional
        :param name: Name to assign to the molecule class and output files.
        :type name: str, optional
        :param moltemp_exe: Name or path of the moltemplate executable.
        :type moltemp_exe: str, optional
        :param cleanup_exe: Name or path of the moltemplate cleanup
            executable.
        :type cleanup_exe: str, optional

        :returns: Path to the resulting LAMMPS ``.data`` file.
        :rtype: str
        """

        # Write input moltemplate file for single-molecule
        new_lt = self.write_moltemplate_file(
            lt_file=lt_file,
            class_name=class_name,
            mt_dir=mt_dir,
            name=name,
        )

        # Run moltemplate
        self.run_moltemplate(
            mt_dir=mt_dir,
            lt_file=new_lt,
            structure_file=structure_file,
            structure_type=structure_type,
            atom_style=atom_style,
            moltemp_exe=moltemp_exe,
            cleanup_exe=cleanup_exe,
        )

        # Return resulting files
        data_file = f"{mt_dir}/{name}.data"
        return data_file

    def trapezoidal_weights(self, x):
        """
        Compute global weights for trapezoidal rule integration estimate,
        such that: I = sum_i w_i * y_i

        :param x: a list of discrete x-values where a corresponding
            y-value is measured
        :type x: array of floats
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        w = np.zeros(n)
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        for i in range(1, n - 1):
            w[i] = 0.5 * (x[i + 1] - x[i - 1])
        return w

    def simpson_weights(self, x):
        """
        Compute global weights for composite Simpson's 1/3 rule
        on a potentially nonuniform grid.

        If an even number of points are passed, Simpson's rule is
        applied through the second-to-last point and the final interval
        is handled with the trapezoidal rule.

        :param x: a list of discrete x-values where a corresponding
            y-value is measured
        :type x: array of floats

        :returns: weights as an ndarray the same length as x
        :rtype: ndarray
        """
        n = len(x)
        if n < 3:
            raise ValueError("Simpson's rule requires at least 3 points.")

        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")

        weights = np.zeros(n)
        h = np.diff(x)
        simpson_n = n if n % 2 == 1 else n - 1

        # Apply Simpson's 1/3 rule to consecutive triplets.
        for j in range(1, simpson_n - 1, 2):
            h1 = h[j - 1]
            h2 = h[j]
            weights[j - 1] += (2 * h1**2 + h1 * h2 - h2**2) / (6 * h1)
            weights[j] += (h1 + h2)**3 / (6 * h1 * h2)
            weights[j + 1] += (2 * h2**2 + h1 * h2 - h1**2) / (6 * h2)

        # If n is even, use trapz for last interval
        if n % 2 == 0:
            h_last = x[n - 1] - x[n - 2]
            weights[n - 2] += 0.5 * h_last
            weights[n - 1] += 0.5 * h_last

        return weights

    def compute_integral(self, x, y, y_err, method='simpson'):
        """
        Compute an integral using either trapezoidal rule or Simpson's
        1/3 rule.

        :param x: a list of discrete x-values where a corresponding
            y-value is measured
        :type x: array of floats
        :param y: a list of y-values, each sampled as the corresponding
            x-value
        :type y: array of floats
        :param y_err: the sampling uncertainty of the corresponding y-value
        :type y_err: array of floats
        :param method: integration method to use, either "simpson" for
            composite Simpson's 1/3 rule or "trapz" for the trapezoidal rule.
        :type method: str, optional

        :returns: A tuple ``(I, error)`` containing the estimated integral
            value and its propagated uncertainty.
        :rtype: tuple
        """
        if method.lower() == 'simpson':
            w = self.simpson_weights(x)
        elif method.lower() == 'trapz':
            w = self.trapezoidal_weights(x)
        else:
            raise ValueError(f"Unknown method {method}")
        integral = np.dot(w, y)
        error = np.sqrt(np.sum((w * y_err)**2))
        return integral, error

    def _write_finished_jobs(self):
        """
        Write self.finished_jobs to a JSON manifest in the analysis directory,
        so job paths survive even if the checkpoint gets corrupted.
        """
        manifest_path = os.path.join(self.analysis_dir, "finished_jobs.json")
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.finished_jobs, f, indent=2)
        os.replace(tmp_path, manifest_path)
