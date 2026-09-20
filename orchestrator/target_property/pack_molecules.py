import os
import subprocess
import shutil
from random import randint


def _try_int(tok):
    """
    Try taking int() of input string without
    erroring out

    :param tok: An input to attempt int()
    :type tok: str
    """
    try:
        return int(tok)
    except ValueError:
        return None


def _try_float(tok):
    """
    Try taking float() of input string without
    erroring out

    :param tok: An input to attempt float()
    :type tok: str
    """
    try:
        return float(tok)
    except ValueError:
        return None


def _detect_filetype(filepath: str) -> str:
    """
    Infer the filetype from a file's extension.

    :param filepath: Path of file to extract extension from.
    :type filepath: str
    """
    ext = os.path.splitext(filepath)[1].lstrip(".").lower()
    return ext


def read_lammps_data(filepath: str, atom_style: str) -> (dict, dict, tuple):
    """
    Parse a LAMMPS data file and return (construction, topology).

    :param datafile: Path to a lammps .data file
    :type datafile: str
    :param atom_style: Atom style of corresponding .data file
    :type atom_style: str (supports ``atomic``, ``charge``,
    ``molecular``, or ``full``)

    Returns construction and topology.
    :param construction: Contains the number of atom types, bond types,
    angle types, dihedral types, improper types, atomic masses, and any
    coeffs in the lammps .data file. The primary role of constructor is
    to define how the lammps .data file was created.
    :type construction: dict
    :param topology: Contains all the topology information contained in
    the passed lammps .data file. This includes the atom, bond, angle,
    dihedral, and improper coeffs. The primary role of topology is to
    preserve the location and connectivity of the passed lammps .data
    file.
    :type topology: dict

    Examples of the resulting construction and topology are shown
    below:

    construction = {
      "atom types": int, "bond types": int, "angle types": int,
      "dihedral types": int, "improper types": int,
      "masses": {type_id: mass, ...},
      "pair_coeffs": {type_id: [coeff strings], ...},
      "bond_coeffs": {...}, "angle_coeffs": {...},
      "dihedral_coeffs": {...}, "improper_coeffs": {...},
    }
    topology = {
      "atoms": [(atom_id, mol_id, type, x, y, z, q), ...],
      "bonds": [(bond_id, type, atom1, atom2), ...],
      "angles": [(angle_id, type, atom1, atom2, atom3), ...],
      "dihedrals": [(dihedral_id, type, a1, a2, a3, a4), ...],
      "impropers": [(improper_id, type, a1, a2, a3, a4), ...],
    }
    """
    type_count_keys = [
        "atom types", "bond types", "angle types", "dihedral types",
        "improper types"
    ]
    coeff_sections = {
        "Pair Coeffs": "pair_coeffs",
        "PairIJ Coeffs": "pair_coeffs",
        "Bond Coeffs": "bond_coeffs",
        "Angle Coeffs": "angle_coeffs",
        "Dihedral Coeffs": "dihedral_coeffs",
        "Improper Coeffs": "improper_coeffs",
    }
    topology_sections = {"Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"}
    ignored_sections = {
        "Velocities", "Ellipsoids", "Lines", "Triangles", "Bodies"
    }
    all_sections = {
        "Masses"
    } | set(coeff_sections) | topology_sections | ignored_sections
    atom_style_columns = {
        "atomic": ["type", "x", "y", "z"],
        "charge": ["type", "q", "x", "y", "z"],
        "molecular": ["mol", "type", "x", "y", "z"],
        "full": ["mol", "type", "q", "x", "y", "z"],
    }
    box_bound_keys = {
        ("xlo", "xhi"): "x",
        ("ylo", "yhi"): "y",
        ("zlo", "zhi"): "z",
    }

    construction = {
        "atom types": 0,
        "bond types": 0,
        "angle types": 0,
        "dihedral types": 0,
        "improper types": 0,
        "masses": {},
        "pair_coeffs": {},
        "bond_coeffs": {},
        "angle_coeffs": {},
        "dihedral_coeffs": {},
        "improper_coeffs": {},
    }

    topology = {
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
    }

    box_bounds = {}

    with open(filepath, "r") as f:
        lines = f.readlines()[1:]  # skip the title/comment line

    n = len(lines)
    i = 0

    # Pull out the type counts, stop at the first section header
    while i < n:
        stripped = lines[i].split("#", 1)[0].rstrip().strip()
        i += 1
        if not stripped:
            continue

        if stripped in all_sections:
            i -= 1
            break

        tokens = stripped.split()

        if len(tokens) == 4 and (tokens[2], tokens[3]) in box_bound_keys:
            axis = box_bound_keys[(tokens[2], tokens[3])]
            box_bounds[axis] = (float(tokens[0]), float(tokens[1]))
            continue

        value = _try_int(tokens[0]) if tokens else None
        if value is None:
            continue
        key = " ".join(tokens[1:])
        if key in type_count_keys:
            construction[key] = value

    # --- sections: single pass, route each section to the right dict ---
    atoms_body_cache = []

    while i < n:
        stripped = lines[i].split("#", 1)[0].rstrip().strip()
        i += 1
        if not stripped or stripped not in all_sections:
            continue

        section = stripped
        while i < n and not lines[i].split("#", 1)[0].rstrip().strip():
            i += 1

        body = []
        while i < n:
            line = lines[i].split("#", 1)[0].rstrip().strip()
            if not line:
                i += 1
                if i >= n or lines[i].split(
                        "#", 1)[0].rstrip().strip() in all_sections:
                    break
                continue
            if line in all_sections:
                break
            body.append(line)
            i += 1

        if section == "Masses":
            for line in body:
                p = line.split()
                construction["masses"][int(p[0])] = float(p[1])
        elif section in coeff_sections:
            key = coeff_sections[section]
            for line in body:
                p = line.split()
                construction[key][int(p[0])] = p[1:]
        elif section == "Atoms":
            atoms_body_cache = body
        elif section == "Bonds":
            for line in body:
                topology["bonds"].append(
                    tuple(int(x) for x in line.split()[1:]))
        elif section == "Angles":
            for line in body:
                topology["angles"].append(
                    tuple(int(x) for x in line.split()[1:]))
        elif section == "Dihedrals":
            for line in body:
                topology["dihedrals"].append(
                    tuple(int(x) for x in line.split()[1:]))
        elif section == "Impropers":
            for line in body:
                topology["impropers"].append(
                    tuple(int(x) for x in line.split()[1:]))
        elif section in ignored_sections:
            pass

    # reduce Atoms lines to (mol_id, type, x, y, z, q, extra)
    if atoms_body_cache:

        col_names = atom_style_columns.get(atom_style)
        parsed_atoms = []  # (mol_id, type, x, y, z, q, extra)
        for line in atoms_body_cache:
            p = line.split()
            atom_id = int(p[0])
            rest = p[1:]

            if col_names is not None and len(rest) >= len(col_names):
                cols = dict(zip(col_names, rest))
                atom_type = int(cols["type"])
                x, y, z = float(cols["x"]), float(cols["y"]), float(cols["z"])
                q = float(cols["q"]) if atom_style == "full" else 0.0
                mol_id = int(cols["mol"]) if "mol" in cols else 1
                extra = " ".join(rest[len(col_names):])
            else:
                raise ValueError(
                    f"Unknown atom style: expected {atom_style_columns.keys()}"
                    f", got {atom_style}")

            parsed_atoms.append(
                (atom_id, mol_id, atom_type, x, y, z, q, extra))

        atom_id_to_index = {a[0]: idx for idx, a in enumerate(parsed_atoms)}
        topology["atoms"] = parsed_atoms
        topology["atom_id_to_index"] = atom_id_to_index

    # Assemble box tuple, defaulting to LAMMPs default: (-0.5, 0.5)
    xlo, xhi = box_bounds.get("x", (-0.5, 0.5))
    ylo, yhi = box_bounds.get("y", (-0.5, 0.5))
    zlo, zhi = box_bounds.get("z", (-0.5, 0.5))
    box = (xlo, xhi, ylo, yhi, zlo, zhi)

    return construction, topology, box


def read_xyz(filepath):
    """
    Read a .xyz file, return list of (label, x, y, z) in file order.

    :param filepath: A path to a .xyz file
    :filepath type: str
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    natoms = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + natoms]:
        p = line.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def write_xyz(atoms, filepath, comment="", element_map=None):
    """
    Write atoms out to a .xyz file.

    :param atoms: Atom data as a list of (mol_id, type, x, y, z, q, extra)
        tuples.
    :type atoms: list of tuple
    :param filepath: Path where the .xyz file will be written.
    :type filepath: str
    :param comment: Optional comment line to write as the second line
        of the .xyz file.
    :type comment: str
    :param element_map: Optional mapping from LAMMPS atom type IDs to
        element symbols. If omitted, the atom type number is used as
        the element label.
    :type element_map: dict, optional

    The resulting .xyz file contains one atom per line in the format
    ``label x y z``.
    """

    with open(filepath, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom_id, mol_id, atom_type, x, y, z, q, extra in atoms:
            label = element_map[atom_type] if element_map else str(atom_type)
            f.write(f"{label} {x:.6f} {y:.6f} {z:.6f}\n")


def write_packmol_script(mol_specs,
                         box,
                         filetype="xyz",
                         output_file="packed.xyz",
                         tolerance=2.0,
                         script_path="packmol.inp",
                         seed=None):
    """
    Write a Packmol input script for packing one or more molecule files
    into a simulation box.

    :param mol_specs: Molecule specifications containing the structure
        filepath and number of copies to pack. The order must match the
        corresponding molecule definitions used later when reconstructing
        the combined system.
    :type mol_specs: list of dict
    :param box: Simulation box bounds in the order
        ``(xlo, xhi, ylo, yhi, zlo, zhi)``.
    :type box: tuple
    :param filetype: Structure file format understood by Packmol.
    :type filetype: str
    :param output_file: Name of the packed structure file that Packmol
        should create.
    :type output_file: str
    :param tolerance: Minimum distance between atoms enforced by Packmol.
    :type tolerance: float
    :param script_path: Path where the Packmol input script will be written.
    :type script_path: str
    :param seed: Optional random seed for Packmol. If omitted, a random
        seed is generated.
    :type seed: int, optional

    Returns the path to the generated Packmol input script.
    :rtype: str

    Each entry in ``mol_specs`` should have the form::

        {"structure": path, "number": int}
    """
    seed = seed or randint(1, 10000)
    xlo, xhi, ylo, yhi, zlo, zhi = box

    with open(script_path, "w") as f:
        f.write(f"tolerance {tolerance}\n")
        f.write(f"filetype {filetype}\n")
        f.write(f"output {output_file}\n")
        f.write(f"seed {seed}\n\n")
        for spec in mol_specs:
            f.write(f"structure {spec['structure']}\n")
            f.write(f"  number {spec['number']}\n")
            f.write(f"  inside box {xlo} {ylo} {zlo} {xhi} {yhi} {zhi}\n")
            f.write("end structure\n\n")
    return script_path


def run_packmol(exe, script_path):
    """
    Run Packmol using a previously generated input script.

    :param exe: Packmol executable or command used to launch Packmol.
    :type exe: str
    :param script_path: Path to the Packmol input script.
    :type script_path: str

    Returns the completed subprocess result if Packmol succeeds.
    Raises a RuntimeError containing Packmol's stdout and stderr if
    the process exits with a non-zero return code.
    """
    with open(script_path) as stdin_file:
        result = subprocess.run(
            exe,
            stdin=stdin_file,
            cwd=os.path.dirname(script_path) or ".",
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Packmol failed (return code {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")
    return result


def pack_system(molecules,
                box,
                tolerance=2.0,
                pack_dir="./",
                packmol_exe="packmol",
                atom_style="full",
                seed=None,
                lammps_file=True):
    """
    Pack one or more molecule structures into a simulation box using
    Packmol and optionally reconstruct the result as a LAMMPS data file.

    :param molecules: Molecule specifications. Each dictionary must contain
        ``"structure"`` and ``"number"``. LAMMPS data files are converted
        to temporary .xyz structures before packing. For LAMMPS inputs,
        each molecule dictionary is also populated with its parsed
        ``"topology"``.
    :type molecules: list of dict
    :param box: Simulation box bounds in the order
        ``(xlo, xhi, ylo, yhi, zlo, zhi)``.
    :type box: tuple
    :param tolerance: Minimum distance between atoms enforced by Packmol.
    :type tolerance: float
    :param pack_dir: Directory in which Packmol input, copied structures,
        and output files will be stored.
    :type pack_dir: str
    :param packmol_exe: Packmol executable or command used to run Packmol.
    :type packmol_exe: str
    :param atom_style: LAMMPS atom style used when reading and reconstructing
        LAMMPS data files. Supports ``atomic``, ``charge``, ``molecular``,
        or ``full``.
    :type atom_style: str
    :param seed: Optional random seed passed to Packmol.
    :type seed: int, optional

    Returns a dictionary containing the packed output path, filetype,
    and packed atom coordinates. When the input structures are LAMMPS
    data files, the dictionary also contains the reconstructed LAMMPS
    data path, construction, and topology.

    :returns:
        A dictionary with keys ``output_path``, ``filetype``,
        ``packed_atoms``, and, for LAMMPS inputs, ``lammps_data_path``,
        ``construction``, and ``topology``.
    :rtype: dict
    """

    if not molecules:
        raise ValueError("molecules must be a non-empty list")

    # detect raw extensions first (before any lammps->xyz conversion)
    exts = {
        os.path.splitext(m["structure"])[1].lstrip(".").lower()
        for m in molecules
    }
    if len(exts) > 1:
        raise ValueError(
            f"All molecule files must share one extension, got: {sorted(exts)}"
        )
    ext = exts.pop()

    is_lammps_file = ext in ('data', 'lmp') or lammps_file

    os.makedirs(pack_dir, exist_ok=True)
    pack_dir_abs = os.path.abspath(pack_dir)

    if is_lammps_file:
        for m in molecules:
            old_file = m["structure"]
            shutil.copy(old_file,
                        os.path.join(pack_dir_abs, os.path.basename(old_file)))

            new_file = old_file.rsplit(".", 1)[0] + ".xyz"
            construction, topology, _ = read_lammps_data(old_file,
                                                         atom_style=atom_style)

            write_xyz(topology["atoms"], new_file)

            m["structure"] = new_file
            m["topology"] = (construction, topology)
        filetype = "xyz"
    else:
        filetype = _detect_filetype(
            molecules[0]["structure"])  # validates against supported set

    copied_files = {}
    for m in molecules:
        src = m["structure"]
        dst_name = os.path.basename(src)
        shutil.copy(src, os.path.join(pack_dir_abs, dst_name))
        copied_files[m["structure"]] = dst_name

    mol_specs = [{
        "structure": copied_files[m["structure"]],
        "number": m["number"]
    } for m in molecules]

    output_filename = f"system_packed.{filetype}"
    output_path = os.path.join(pack_dir_abs, output_filename)
    script_path = os.path.join(pack_dir, "packmol.inp")

    write_packmol_script(mol_specs,
                         box,
                         filetype=filetype,
                         output_file=output_filename,
                         tolerance=tolerance,
                         script_path=script_path,
                         seed=seed)
    run_packmol(packmol_exe, script_path)

    packed_atoms = read_xyz(output_path) if filetype == "xyz" else None

    result = {
        "output_path": output_path,
        "filetype": filetype,
        "packed_atoms": packed_atoms,
    }

    # In the future, rewrite the lammps file
    if is_lammps_file:
        expected_natoms = sum(m["number"] * len(m["topology"][1]["atoms"])
                              for m in molecules)
        assert expected_natoms == len(packed_atoms), (
            f"Atom count mismatch: expected {expected_natoms} from molecule"
            f"specs, got {len(packed_atoms)} from packed file -- check "
            "molecule ordering")

        lammps_out = os.path.join(pack_dir, "system_packed.data")
        combined_construction, combined_topology = write_lammps_data_from_xyz(
            molecules, packed_atoms, box, lammps_out, atom_style=atom_style)

        result["lammps_data_path"] = lammps_out
        result["construction"] = combined_construction
        result["topology"] = combined_topology

    return result


def compare_system_construction(construction1, construction2):
    """
    Check whether two molecule construction dictionaries describe the
    same LAMMPS system types.

    :param construction1: Construction dictionary for the first molecule.
    :type construction1: dict
    :param construction2: Construction dictionary for the second molecule.
    :type construction2: dict

    Returns True when both constructions have identical type counts,
    masses, and force-field coefficients. Matching constructions can
    safely share type IDs when combined into a larger system.
    :rtype: bool
    """
    keys_to_compare = [
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
        "masses",
        "pair_coeffs",
        "bond_coeffs",
        "angle_coeffs",
        "dihedral_coeffs",
        "improper_coeffs",
    ]
    return all(construction1[k] == construction2[k] for k in keys_to_compare)


def write_lammps_data(construction,
                      topology,
                      filepath,
                      box,
                      atom_style="full",
                      comment="LAMMPS data file"):
    """
    Write construction and topology dictionaries to a LAMMPS data file.

    :param construction: Dictionary containing LAMMPS type counts, masses,
        and force-field coefficients.
    :type construction: dict
    :param topology: Dictionary containing atom, bond, angle, dihedral,
        and improper topology information.
    :type topology: dict
    :param filepath: Path where the LAMMPS data file will be written.
    :type filepath: str
    :param box: Simulation box bounds in the order
        ``(xlo, xhi, ylo, yhi, zlo, zhi)``.
    :type box: tuple
    :param atom_style: LAMMPS atom style determining the column order
        of the Atoms section. Supports ``atomic``, ``charge``,
        ``molecular``, or ``full``.
    :type atom_style: str
    :param comment: Comment written as the first line of the data file.
    :type comment: str

    The atom, bond, angle, dihedral, and improper IDs are generated
    sequentially when the file is written.
    """
    atom_style_columns = {
        "atomic": ["type", "x", "y", "z"],
        "charge": ["type", "q", "x", "y", "z"],
        "molecular": ["mol", "type", "x", "y", "z"],
        "full": ["mol", "type", "q", "x", "y", "z"],
    }
    if atom_style not in atom_style_columns:
        raise ValueError(f"Unknown atom style: {atom_style}")

    xlo, xhi, ylo, yhi, zlo, zhi = box
    atoms = topology["atoms"]  # (mol_id, type, x, y, z, q, extra)
    bonds = topology["bonds"]
    angles = topology["angles"]
    dihedrals = topology["dihedrals"]
    impropers = topology["impropers"]

    with open(filepath, "w") as f:
        f.write(f"{comment}\n\n")

        f.write(f"{len(atoms)} atoms\n")
        if bonds:
            f.write(f"{len(bonds)} bonds\n")
        if angles:
            f.write(f"{len(angles)} angles\n")
        if dihedrals:
            f.write(f"{len(dihedrals)} dihedrals\n")
        if impropers:
            f.write(f"{len(impropers)} impropers\n")
        f.write("\n")

        f.write(f"{construction['atom types']} atom types\n")
        f.write(f"{construction['bond types']} bond types\n")
        f.write(f"{construction['angle types']} angle types\n")
        f.write(f"{construction['dihedral types']} dihedral types\n")
        f.write(f"{construction['improper types']} improper types\n")
        f.write("\n")

        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        if construction["masses"]:
            f.write("Masses\n\n")
            for tid, mass in sorted(construction["masses"].items()):
                f.write(f"{tid} {mass}\n")
            f.write("\n")

        coeff_sections = [
            ("pair_coeffs", "Pair Coeffs"),
            ("bond_coeffs", "Bond Coeffs"),
            ("angle_coeffs", "Angle Coeffs"),
            ("dihedral_coeffs", "Dihedral Coeffs"),
            ("improper_coeffs", "Improper Coeffs"),
        ]
        for key, header in coeff_sections:
            coeffs = construction[key]
            if coeffs:
                f.write(f"{header}\n\n")
                for tid, vals in sorted(coeffs.items()):
                    f.write(f"{tid} {' '.join(str(v) for v in vals)}\n")
                f.write("\n")

        f.write("Atoms\n\n")
        col_names = atom_style_columns[atom_style]
        for idx, mol_id, atype, x, y, z, q, extra in atoms:
            row = {
                "mol": mol_id,
                "type": atype,
                "q": q,
                "x": x,
                "y": y,
                "z": z,
                "extra": extra
            }
            f.write(f"{idx} " + " ".join(str(row[c]) for c in col_names))
            f.write(f" {extra}\n")
        f.write("\n")

        for key, label in [
            ("bonds", "Bonds"),
            ("angles", "Angles"),
            ("dihedrals", "Dihedrals"),
            ("impropers", "Impropers"),
        ]:
            entries = topology[key]
            if entries:
                f.write(f"{label}\n\n")
                for idx, entry in enumerate(entries, start=1):
                    etype, *atom_ids = entry
                    f.write(f"{idx} {etype} "
                            + " ".join(str(a) for a in atom_ids) + "\n")
                f.write("\n")


def build_combined_system(molecules, atom_style="full"):
    """
    Combine multiple molecule topologies and constructions into one
    LAMMPS system.

    :param molecules: Molecule specifications containing ``"number"`` and
        ``"topology"`` entries. Each topology contains a construction and
        topology dictionary for a single molecule.
    :type molecules: list of dict
    :param atom_style: LAMMPS atom style associated with the molecule data.
        This parameter is currently retained for interface consistency.
    :type atom_style: str

    Returns the combined construction and topology. Molecules with
    identical construction dictionaries share their type IDs; molecules
    with different constructions are assigned non-overlapping type ID
    ranges. Each copy of a molecule receives a unique molecule ID, and
    atom IDs in bonds, angles, dihedrals, and impropers are offset to
    reference the correct copy.

    :returns:
        A tuple ``(combined_construction, combined_topology)``.
    :rtype: tuple
    """

    combined_topology = {
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": []
    }
    combined_construction = {
        "atom types": 0,
        "bond types": 0,
        "angle types": 0,
        "dihedral types": 0,
        "improper types": 0,
        "masses": {},
        "pair_coeffs": {},
        "bond_coeffs": {},
        "angle_coeffs": {},
        "dihedral_coeffs": {},
        "improper_coeffs": {},
    }
    type_key_map = {
        "atom": "atom types",
        "bond": "bond types",
        "angle": "angle types",
        "dihedral": "dihedral types",
        "improper": "improper types",
    }
    atom_id_offset = 0
    mol_id = 0
    seen_constructions = []

    for m in molecules:
        mol_construction, mol_topology = m["topology"]
        n_copies = m["number"]
        n_atoms_per_copy = len(mol_topology["atoms"])
        id_to_index = mol_topology["atom_id_to_index"]

        match = next((offsets for c, offsets in seen_constructions
                      if compare_system_construction(c, mol_construction)),
                     None)

        if match is not None:
            type_offsets = match
        else:
            type_offsets = {
                "atom": combined_construction["atom types"],
                "bond": combined_construction["bond types"],
                "angle": combined_construction["angle types"],
                "dihedral": combined_construction["dihedral types"],
                "improper": combined_construction["improper types"],
            }
            for src_key, atype in [
                ("masses", "atom"),
                ("pair_coeffs", "atom"),
                ("bond_coeffs", "bond"),
                ("angle_coeffs", "angle"),
                ("dihedral_coeffs", "dihedral"),
                ("improper_coeffs", "improper"),
            ]:
                offset = type_offsets[atype]
                for tid, val in mol_construction[src_key].items():
                    combined_construction[src_key][tid + offset] = val
            for atype, key in type_key_map.items():
                combined_construction[key] += mol_construction[key]
            seen_constructions.append((mol_construction, type_offsets))

        for _ in range(n_copies):
            mol_id += 1

            for local_idx, (atom_id, _, atype, x, y, z, q,
                            extra) in enumerate(mol_topology["atoms"]):
                new_atom_id = atom_id_offset + local_idx + 1
                combined_topology["atoms"].append(
                    (new_atom_id, mol_id, atype + type_offsets["atom"], x, y,
                     z, q, extra))

            for (btype, a1, a2) in mol_topology["bonds"]:
                combined_topology["bonds"].append((
                    btype + type_offsets["bond"],
                    id_to_index[a1] + atom_id_offset + 1,
                    id_to_index[a2] + atom_id_offset + 1,
                ))
            for (antype, a1, a2, a3) in mol_topology["angles"]:
                combined_topology["angles"].append((
                    antype + type_offsets["angle"],
                    id_to_index[a1] + atom_id_offset + 1,
                    id_to_index[a2] + atom_id_offset + 1,
                    id_to_index[a3] + atom_id_offset + 1,
                ))
            for (dtype, a1, a2, a3, a4) in mol_topology["dihedrals"]:
                combined_topology["dihedrals"].append((
                    dtype + type_offsets["dihedral"],
                    id_to_index[a1] + atom_id_offset + 1,
                    id_to_index[a2] + atom_id_offset + 1,
                    id_to_index[a3] + atom_id_offset + 1,
                    id_to_index[a4] + atom_id_offset + 1,
                ))
            for (itype, a1, a2, a3, a4) in mol_topology["impropers"]:
                combined_topology["impropers"].append((
                    itype + type_offsets["improper"],
                    id_to_index[a1] + atom_id_offset + 1,
                    id_to_index[a2] + atom_id_offset + 1,
                    id_to_index[a3] + atom_id_offset + 1,
                    id_to_index[a4] + atom_id_offset + 1,
                ))
            atom_id_offset += n_atoms_per_copy

    return combined_construction, combined_topology


def substitute_packed_coordinates(topology, packed_atoms):
    """
    Replace the coordinates in a topology with coordinates generated by
    Packmol while preserving atom types, molecule IDs, charges, and topology.

    :param topology: Topology dictionary whose atom coordinates should be
        replaced.
    :type topology: dict
    :param packed_atoms: Atom coordinates returned by ``read_xyz`` after
        Packmol has generated the packed structure.
    :type packed_atoms: list of tuple

    The atoms are matched by their position in the two lists, so the
    ordering of ``packed_atoms`` must correspond to the ordering of atoms
    in ``topology["atoms"]``.

    Returns the updated topology dictionary.
    :rtype: dict

    Raises ValueError if the two atom lists contain different numbers
    of atoms.
    """
    if len(packed_atoms) != len(topology["atoms"]):
        raise ValueError(f"Atom count mismatch: {len(packed_atoms)} packed vs "
                         f"{len(topology['atoms'])} in combined topology")
    new_atoms = []
    for (atom_id, mol_id, atype, _, _, _, q,
         extra), (_, x, y, z) in zip(topology["atoms"], packed_atoms):
        new_atoms.append((atom_id, mol_id, atype, x, y, z, q, extra))
    topology["atoms"] = new_atoms
    return topology


def write_lammps_data_from_xyz(molecules,
                               packed_atoms,
                               box,
                               filepath,
                               atom_style="full"):
    """
    Reconstruct and write a combined LAMMPS data file using coordinates
    generated by Packmol.

    :param molecules: Molecule specifications containing ``"number"`` and
        ``"topology"`` entries. Each topology contains the construction
        and topology data for a single molecule.
    :type molecules: list of dict
    :param packed_atoms: Packed coordinates in the order returned by
        ``read_xyz`` on Packmol's output.
    :type packed_atoms: list of tuple
    :param box: Simulation box bounds in the order
        ``(xlo, xhi, ylo, yhi, zlo, zhi)``.
    :type box: tuple
    :param filepath: Path where the reconstructed LAMMPS data file will
        be written.
    :type filepath: str
    :param atom_style: LAMMPS atom style to use when writing the data file.
    :type atom_style: str

    Returns the combined construction and topology used to generate
    the LAMMPS data file.

    :returns:
        A tuple ``(combined_construction, combined_topology)``.
    :rtype: tuple
    """
    combined_construction, combined_topology = build_combined_system(
        molecules, atom_style=atom_style)
    combined_topology = substitute_packed_coordinates(combined_topology,
                                                      packed_atoms)
    write_lammps_data(combined_construction,
                      combined_topology,
                      filepath,
                      box,
                      atom_style=atom_style)
    return combined_construction, combined_topology


def make_molecule_unique(construction,
                         topology,
                         target_mol_id,
                         filepath,
                         box,
                         atom_style="full"):
    """
    Give every atom in a target molecule a unique atom type when that type
    is shared with atoms outside the target molecule.

    :param construction: Construction dictionary containing the system's
        atom types, masses, and pair coefficients.
    :type construction: dict
    :param topology: Topology dictionary containing the system's atoms
        and connectivity.
    :type topology: dict
    :param target_mol_id: Molecule ID whose atom types should be made unique.
    :type target_mol_id: int
    :param filepath: Path where the modified LAMMPS data file will be written.
    :type filepath: str
    :param box: Simulation box bounds in the order
        ``(xlo, xhi, ylo, yhi, zlo, zhi)``.
    :type box: tuple
    :param atom_style: LAMMPS atom style to use when writing the data file.
    :type atom_style: str

    Returns the modified construction, topology, and a mapping from each
    newly created atom type to the original type it was copied from.

    :returns:
        A tuple ``(construction, topology, new_to_old_type_map)``.
    :rtype: tuple

    Only atom types shared between the target molecule and other molecules
    are duplicated. The new types inherit the original masses and pair
    coefficients, so the physical parameters remain unchanged while the
    target molecule can be addressed independently.
    """

    atoms = topology["atoms"]  # (atom_id, mol_id, type, x, y, z, q)

    target_indices = [i for i, a in enumerate(atoms) if a[1] == target_mol_id]
    if not target_indices:
        raise ValueError(f"No atoms found with mol_id == {target_mol_id}")

    # types used anywhere OUTSIDE the target molecule
    other_types = {a[2] for a in atoms if a[1] != target_mol_id}

    old_to_new = {}
    next_type = construction["atom types"] + 1
    for i in target_indices:
        old_type = atoms[i][2]
        if (old_type in other_types) and (old_type not in old_to_new):
            old_to_new[old_type] = next_type
            next_type += 1

    new_to_old = {new_t: old_t for old_t, new_t in old_to_new.items()}

    for new_t, old_t in new_to_old.items():
        if old_t in construction["masses"]:
            construction["masses"][new_t] = construction["masses"][old_t]
        if old_t in construction["pair_coeffs"]:
            construction["pair_coeffs"][new_t] = list(
                construction["pair_coeffs"][old_t])

    construction["atom types"] = next_type - 1

    new_atoms = list(atoms)
    for i in target_indices:
        atom_id, mol_id, old_type, x, y, z, q, extra = new_atoms[i]
        if old_type in old_to_new:
            new_atoms[i] = (atom_id, mol_id, old_to_new[old_type], x, y, z, q,
                            extra)
    topology["atoms"] = new_atoms

    write_lammps_data(construction,
                      topology,
                      filepath,
                      box,
                      atom_style=atom_style)

    return construction, topology, new_to_old


def make_charge_exclusive_atom_types(construction,
                                     topology,
                                     filepath,
                                     box,
                                     atom_style="full",
                                     target_mol_id=None,
                                     ndigits=8):
    """
    Split atom types so that every type carries exactly one charge
    (and, if target_mol_id is given, so that the target molecule's types
    are not shared with any other molecule).

    Masses and pair coefficients are copied from the parent type.

    Returns (construction, topology, new_to_old, type_charges) where
    new_to_old maps each newly created type to its parent type and
    type_charges maps every type used by the target atoms to its charge.
    """
    if atom_style not in ("charge", "full"):
        write_lammps_data(construction,
                          topology,
                          filepath,
                          box,
                          atom_style=atom_style)
        return construction, topology, {}, {}, {}

    atoms = topology["atoms"]  # (atom_id, mol_id, type, x, y, z, q, extra)

    def in_target(mol_id):
        return target_mol_id is None or mol_id == target_mol_id

    # types that appear on atoms we are NOT modifying
    outside_types = {a[2] for a in atoms if not in_target(a[1])}

    key_to_type = {}  # (old_type, rounded q) -> assigned type
    claimed_originals = set()  # old types already kept by some charge
    new_to_old = {}
    next_type = construction["atom types"] + 1
    new_atoms = []

    for atom in atoms:
        atom_id, mol_id, atype, x, y, z, q, extra = atom
        if not in_target(mol_id):
            new_atoms.append(atom)
            continue

        key = (atype, round(q, ndigits))
        if key not in key_to_type:
            if atype not in outside_types and atype not in claimed_originals:
                key_to_type[key] = atype  # first charge keeps the type
                claimed_originals.add(atype)
            else:
                key_to_type[key] = next_type
                new_to_old[next_type] = atype
                next_type += 1
        new_atoms.append(
            (atom_id, mol_id, key_to_type[key], x, y, z, q, extra))

    # copy per-type parameters from parent to child
    for new_t, old_t in new_to_old.items():
        if old_t in construction["masses"]:
            construction["masses"][new_t] = construction["masses"][old_t]
        if old_t in construction["pair_coeffs"]:
            construction["pair_coeffs"][new_t] = list(
                construction["pair_coeffs"][old_t])

    construction["atom types"] = next_type - 1
    topology["atoms"] = new_atoms

    # type -> charge for JUST the target molecule's types
    target_type_charges = dict(
        sorted((t, q) for (_, q), t in key_to_type.items()))

    # type -> charge for EVERY type in the system (target and everything else)
    charges_by_type = {}
    for _, _, atype, _, _, _, q, _ in new_atoms:
        charges_by_type.setdefault(atype, set()).add(round(q, ndigits))

    all_type_charges = {}
    for t in sorted(charges_by_type):
        qs = sorted(charges_by_type[t])
        all_type_charges[t] = qs[0]
        if len(qs) > 1:
            print(f"Warning: type {t} carries multiple charges: {qs}")

    # sanity check: net charge of the target should still be ~0
    qtot = sum(a[6] for a in new_atoms if in_target(a[1]))
    if target_mol_id is not None and abs(qtot) > 1e-4:
        print(f"Warning: target molecule net charge is {qtot:.6f}")

    write_lammps_data(construction,
                      topology,
                      filepath,
                      box,
                      atom_style=atom_style)

    return (construction, topology, new_to_old, target_type_charges,
            all_type_charges)


def compute_total_mass(construction, topology):
    """
    Simple function for computing total mass of a system.
    """
    masses = construction["masses"]
    atoms = topology["atoms"]
    total_mass = 0
    for atom in atoms:
        atype = atom[2]
        total_mass += masses[atype]
    return total_mass
