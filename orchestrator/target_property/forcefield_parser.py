def _try_float(val):
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


def _try_int(val):
    """
    Try taking ``int()`` of an input value without raising an error.

    :param val: Value to attempt conversion to an int.
    :type val: str

    Returns the converted int when conversion succeeds, or ``None``
    when the value cannot be interpreted as an int.

    :rtype: int or None
    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _tokens_equal(a, b):
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


def _strip_accelerator_suffix(style):
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


def split_hybrid_style(style_str):
    """
    Split a LAMMPS hybrid *_style string into individual sub-styles.

    If ``style_str`` does not declare a hybrid style (i.e. its first token
    does not contain "hybrid"), it is returned unchanged as a single-item
    list. Otherwise, the remaining tokens are grouped into sub-styles by
    treating any non-numeric token as the start of a new sub-style name,
    with subsequent numeric tokens collected as that sub-style's
    arguments.

    Example: ``'hybrid/overlay lj/cut 1.0 lj/cut/soft 1 0.5 1.0'`` ->
    ``['lj/cut 1.0', 'lj/cut/soft 1 0.5 1.0']``

    :param style_str: A LAMMPS ``*_style`` value, possibly declaring a
        hybrid style (e.g. the text following ``pair_style`` in a LAMMPS
        input file).
    :type style_str: str

    :returns: The individual sub-style strings (name plus arguments) that
        make up the hybrid style, or a single-element list containing
        ``style_str`` unchanged if it is not a hybrid style.
    :rtype: list of str
    """

    parts = style_str.strip().split()

    # Not a hybrid style
    if not parts or "hybrid" not in parts[0]:
        return [style_str.strip()]

    styles = []
    current = []
    for part in parts[1:]:
        # A non-numeric token marks the beginning of a new pair style
        if _try_float(part) is None:
            if current:
                styles.append(" ".join(current))
            current = [part]
        else:
            current.append(part)

    # Don't forget the final style
    if current:
        styles.append(" ".join(current))

    return styles


def parse_styles(init_path):
    """
    Parse pair/bond/angle/dihedral/improper style declarations from a
    LAMMPS init-style file.

    Lines are stripped of comments (text after ``#``); lines with fewer
    than two tokens, or whose first token is not one of ``pair_style``,
    ``bond_style``, ``angle_style``, ``dihedral_style``, or
    ``improper_style``, are ignored. For each recognized style keyword,
    the declared style value is checked for a ``hybrid`` designation: if
    hybrid, the value is split into its individual sub-styles with
    :func:`split_hybrid_style` and each unique sub-style is recorded; if
    not, the single style value is recorded as-is. Duplicate style
    entries (compared token-wise) are not added twice. Only the most
    recently parsed line's hybrid designation is kept for a given
    keyword.

    :param init_path: Path to the LAMMPS init-style file to parse (e.g.
        a ``*.in.init`` file).
    :type init_path: str

    :returns: A dictionary keyed by style category (``"pair"``,
        ``"bond"``, ``"angle"``, ``"dihedral"``, ``"improper"``), each
        mapping to a dict with keys ``"hybrid"`` (the hybrid style
        designation string, e.g. ``"hybrid/overlay"``, or False if not
        hybrid) and ``"styles"`` (a list of the unique declared or
        sub-style strings for that category).
    :rtype: dict
    """
    styles = {}
    with open(init_path) as f:
        for line in f:
            clean = line.split("#", 1)[0].strip().split()
            if not clean or len(clean) < 2:
                continue
            keyword = clean[0]
            value = " ".join(clean[1:])
            if keyword not in ('pair_style', 'bond_style', 'angle_style',
                               'dihedral_style', 'improper_style'):
                continue
            keyword = keyword[:-len('_style')]
            if keyword not in styles:
                styles[keyword] = {
                    "hybrid": False,
                    "styles": [],
                }

            # Check if declared style is hybrid
            if "hybrid" in clean[1]:
                styles[keyword]["hybrid"] = clean[1]

                # If it is, split the styles by non-numeric entries
                hybrid_styles = split_hybrid_style(" ".join(clean[1:]))
                for style in hybrid_styles:
                    if not any(
                            _tokens_equal(style, existing)
                            for existing in styles[keyword]["styles"]):
                        styles[keyword]["styles"].append(style)
            else:
                styles[keyword]["hybrid"] = False
                if not any(
                        _tokens_equal(value, existing)
                        for existing in styles[keyword]["styles"]):
                    styles[keyword]["styles"].append(value)
    return styles


def parse_coeffs(parameter_path, styles, logger=None):
    """
    Parse pair/bond/angle/dihedral/improper coefficient lines from a LAMMPS
    parameter file.

    Lines are stripped of comments (text after ``#``); blank lines are
    skipped. Each remaining line's keyword (e.g. ``pair_coeff``) is
    matched against the corresponding entry of ``styles`` to identify
    which declared sub-style the coefficients belong to, using the first
    token after the keyword that matches a known (optionally
    accelerator-suffixed) sub-style name; if no such token is found, the
    first style declared for that category is assumed. Wildcard
    (``*``-containing) pair coefficient lines are skipped with a warning,
    since they must be added manually. For ``lj/*`` pair coefficients,
    epsilon and sigma are validated as numeric, and sigma is corrected to
    1.0 if both epsilon and sigma are zero, to avoid LAMMPS errors. Pair
    type indices are reordered so that ``i <= j``.

    :param parameter_path: Path to the LAMMPS parameter file to parse
        (e.g. a ``*.in.settings`` file).
    :type parameter_path: str
    :param styles: Parsed style dictionary, as returned by
        :func:`parse_styles`, used to identify valid sub-style names for
        each coefficient category. Every coefficient keyword found in the
        file must have a corresponding category in ``styles``.
    :type styles: dict
    :param logger: Optional logger. The wildcard warning is sent to
        ``logger.info`` if provided, otherwise it is printed.
    :type logger: logging.Logger, optional

    :returns: A dictionary keyed by coefficient category (``"pair"``,
        ``"bond"``, ``"angle"``, ``"dihedral"``, ``"improper"``). Pair
        coefficients are stored as ``coeffs["pair"][i][j] = (substyle,
        parts)`` for each type pair ``(i, j)`` with ``i <= j``; all other
        categories are stored as ``coeffs[category][type_index] =
        (substyle, parts)``. ``parts`` is the list of the line's tokens
        with the sub-style token removed. It still begins with the command
        keyword (e.g. ``"pair_coeff"``), so the type indices are at
        ``parts[1]`` (and ``parts[2]`` for pair coefficients).
    :rtype: dict

    :raises ValueError: If a ``lj/*`` pair coefficient line does not
        contain numeric epsilon and sigma values (expected at
        ``parts[3]`` and ``parts[4]``).
    """
    coeffs = {
        "pair": {},
        "bond": {},
        "angle": {},
        "dihedral": {},
        "improper": {}
    }

    # Get all valid styles for each key in styles
    valid_styles, is_hybrid = {}, {}
    substyle_names, stripped_substyle_names = {}, {}

    for k, v in styles.items():
        valid_styles[k] = styles.get(k, {
            "hybrid": False,
            "styles": []
        }).get("styles")
        is_hybrid[k] = len(valid_styles[k])
        substyle_names[k] = {entry.split()[0] for entry in valid_styles[k]}
        stripped_substyle_names[k] = {
            _strip_accelerator_suffix(entry.split()[0])[0]
            for entry in valid_styles[k]
        }

    with open(parameter_path) as f:
        for line in f:
            clean = line.split("#")[0].strip()
            if not clean:
                continue
            parts = clean.split()
            keyword = parts[0]
            keyword = keyword[:-len('_coeff')]

            if keyword == "pair":
                i, j = parts[1], parts[2]
                if "*" in i or "*" in j:
                    msg = "WARNING: Found wildcard coefficient.\n"
                    msg += " These should be added mannually"
                    if logger is None:
                        print(msg)
                    else:
                        logger.info(msg)
                    continue
                i, j = int(i), int(j)
                i, j = min(i, j), max(i, j)
                parts[1], parts[2] = str(i), str(j)
            elif keyword in ("bond", "angle", "dihedral", "improper"):
                type_index = int(parts[1])
            else:
                continue

            # Detect substyle token by matching against known styles,
            # defaulting to first pairstyle in styles
            substyle = valid_styles[keyword][0].split()[0]
            for idx, value in enumerate(parts[1:], start=1):
                base_value = value
                stripped_value = _strip_accelerator_suffix(value)
                check1 = base_value in substyle_names[keyword]
                check2 = base_value in stripped_substyle_names[keyword]
                check3 = stripped_value in stripped_substyle_names[keyword]
                if check1 or check2 or check3:
                    substyle = value
                    parts.pop(idx)
                    break

            if is_hybrid[keyword] and not substyle:
                raise ValueError(f"{parameter_path}: pair_style is hybrid but "
                                 f"pair_coeff line is missing a recognized "
                                 f"substyle token: {clean}")

            if keyword == "pair" and "lj/" in substyle:
                # Check that epsilon/sigma are numeric
                try:
                    epsilon = float(parts[3])
                    sigma = float(parts[4])
                    # Make sure that sigma is non-zero
                    # to avoid LAMMPs errors
                    if epsilon == 0.0 and sigma == 0.0:
                        parts[4] = str(max(sigma, 1.0))
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Expected numeric epsilon/sigma in: {clean}")
                coeffs[keyword].setdefault(i, {})[j] = (substyle, parts)
            elif keyword == "pair":
                coeffs[keyword].setdefault(i, {})[j] = (substyle, parts)
            else:
                coeffs[keyword][type_index] = (substyle, parts)

    return coeffs


def write_coeff_line(coeff, coeff_type, is_hybrid):
    """
    Format a single pair_coeff, bond_coeff, angle_coeff, dihedral_coeff,
    or improper_coeff line for writing to a LAMMPS parameter file.

    If ``is_hybrid`` is True, the sub-style name is reinserted into the
    coefficient line's tokens at the position LAMMPS expects it (after
    the command keyword and the two type indices for pair coefficients,
    or after the keyword and the single type index for
    bond/angle/dihedral/improper coefficients).

    :param coeff: A ``(substyle, parts)`` tuple as stored in a
        coefficients dictionary (e.g. one entry from the dict returned by
        :func:`parse_coeffs`), where ``parts`` are the coefficient line's
        tokens with the sub-style token already removed.
    :type coeff: tuple
    :param coeff_type: The coefficient category being written. Must be
        one of ``"pair"``, ``"bond"``, ``"angle"``, ``"dihedral"``, or
        ``"improper"``.
    :type coeff_type: str
    :param is_hybrid: Whether the corresponding style is a hybrid style,
        requiring the sub-style name to be written back into the line.
    :type is_hybrid: bool

    :returns: The formatted coefficient line, as a single space-joined
        string (without a trailing newline).
    :rtype: str

    :raises ValueError: If ``coeff_type`` is not a recognized coefficient
        category.
    """
    substyle, parts = coeff

    if coeff_type not in ("pair", "bond", "angle", "dihedral", "improper"):
        raise ValueError(f"Unknown coeff_type {coeff_type}")

    if is_hybrid:
        if coeff_type == "pair":
            substyle_location = 3
        elif coeff_type in ("bond", "angle", "dihedral", "improper"):
            substyle_location = 2
        parts = parts[:substyle_location] + [substyle
                                             ] + parts[substyle_location:]

    return " ".join(parts)


def write_coeff_lines(styles, coeffs, coeff_type, outfile):
    """
    Write all coefficient lines of one category to an already-open file.

    If no styles are declared for ``coeff_type``, nothing is written. The
    style is treated as hybrid (and sub-style names are reinserted into
    each line) if more than one style is declared for that category. Pair
    coefficients are iterated in ascending order of both type indices;
    other categories are iterated in ascending order of their single type
    index.

    :param styles: Parsed style dictionary, as returned by
        :func:`parse_styles`, used to determine whether ``coeff_type`` is
        a hybrid style.
    :type styles: dict
    :param coeffs: Parsed coefficients dictionary, as returned by
        :func:`parse_coeffs`.
    :type coeffs: dict
    :param coeff_type: The coefficient category to write. Must be one of
        ``"pair"``, ``"bond"``, ``"angle"``, ``"dihedral"``, or
        ``"improper"``.
    :type coeff_type: str
    :param outfile: An already-open, writable file object to which
        formatted coefficient lines are written, one per line.
    :type outfile: file object

    :returns: None. Coefficient lines are written directly to ``outfile``.
    :rtype: None
    """
    present_styles = styles.get(coeff_type, {}).get("styles")
    if present_styles:
        is_hybrid = len(present_styles) > 1
    else:
        return None
    style_coeffs = coeffs[coeff_type]
    for i in sorted(style_coeffs):
        if coeff_type == "pair":
            for j in sorted(style_coeffs[i]):
                line = write_coeff_line(style_coeffs[i][j], coeff_type,
                                        is_hybrid)
                outfile.write(line + "\n")
        else:
            line = write_coeff_line(style_coeffs[i], coeff_type, is_hybrid)
            outfile.write(line + "\n")


def write_parameter_file(styles, coeffs, extra_coeff_lines=None, outfile=None):
    """
    Write coefficient lines to `outfile` (a path).

    Any ``extra_coeff_lines`` are written first, verbatim, followed by a
    blank line. Then, for each coefficient category in the order
    ``('pair', 'bond', 'angle', 'dihedral', 'improper')``, all coefficient
    lines for that category are written via :func:`write_coeff_lines`,
    followed by a blank line; categories with no coefficients are
    skipped entirely.

    :param styles: Parsed style dictionary, as returned by
        :func:`parse_styles`.
    :type styles: dict
    :param coeffs: Parsed coefficients dictionary, as returned by
        :func:`parse_coeffs`.
    :type coeffs: dict
    :param extra_coeff_lines: Additional raw coefficient lines to write
        at the top of the file before the parsed coefficients, if any.
    :type extra_coeff_lines: list of str, optional
    :param outfile: Path to the file to create/overwrite with the
        formatted coefficient lines. Required in practice, since the
        file is opened unconditionally.
    :type outfile: str

    :returns: None. The formatted coefficients are written to ``outfile``.
    :rtype: None
    """
    with open(outfile, "w") as f:
        if extra_coeff_lines is not None:
            for line in extra_coeff_lines:
                f.write(line + "\n")
            f.write("\n")
        for style_type in ('pair', 'bond', 'angle', 'dihedral', 'improper'):
            if not coeffs[style_type]:
                continue
            write_coeff_lines(styles, coeffs, style_type, f)
            f.write("\n")


def parse_ff(style_file, params_file, logger=None):
    """
    Parse one LAMMPS force-field style/parameter file pair.

    The style file is parsed with :func:`parse_styles` to determine the
    declared pair/bond/angle/dihedral/improper styles, and the parameter
    file is then parsed with :func:`parse_coeffs` against those styles to
    extract the corresponding coefficients.

    :param style_file: Path to the LAMMPS style file to parse (e.g. a
        ``*.in.init`` file).
    :type style_file: str
    :param params_file: Path to the LAMMPS parameter file to parse (e.g.
        a ``*.in.settings`` file).
    :type params_file: str
    :param logger: Optional logger. The stacking/mass-comparison message
        is sent to ``logger.info`` if provided, otherwise printed.
    :type logger: logging.Logger, optional

    :returns:
        A tuple ``(styles, coeffs)`` containing the parsed style
        dictionary (as returned by :func:`parse_styles`) and the parsed
        coefficients dictionary (as returned by :func:`parse_coeffs`).
    :rtype: tuple
    """

    styles = parse_styles(style_file)
    coeffs = parse_coeffs(params_file, styles, logger=logger)

    return styles, coeffs


def merge_styles(styles, new_styles):
    """
    Merge force-field style dictionaries by appending new styles
    to the list of styles and retaining hybrid if it is defined.

    For each style category present in ``new_styles``, unique styles
    (compared token-wise) are appended to the corresponding category's
    style list in ``styles``. Categories present only in ``new_styles``
    are added as-is. For categories present in both, the ``hybrid``
    designation is taken from ``new_styles`` if the merged entry doesn't
    already have one. If the merged entry's ``hybrid`` is still False
    and ``new_styles`` declares any styles for that category (even if
    they were all duplicates), it is set to ``"hybrid/overlay"``, so a
    category may temporarily be flagged hybrid with only one style (see
    :func:`reduce_styles`, which resets this).

    :param styles: Base style dictionary to merge into, in the format
        produced by a style-parsing function: keyed by category (e.g.
        ``"pair"``, ``"bond"``), each mapping to a dict with ``"styles"``
        (list of style strings) and ``"hybrid"`` (hybrid designation
        string or False).
    :type styles: dict
    :param new_styles: Style dictionary to merge into ``styles``, in the
        same format.
    :type new_styles: dict

    :returns: A new merged style dictionary in the same format as
        ``styles``/``new_styles``. Neither input dictionary is modified.
    :rtype: dict
    """
    merged = {
        key: {
            "styles": value.get("styles", []).copy(),
            "hybrid": value.get("hybrid"),
        }
        for key, value in styles.items()
    }

    for key, value in new_styles.items():
        if key not in merged:
            merged[key] = {
                "styles": value.get("styles", []).copy(),
                "hybrid": value.get("hybrid"),
            }
            continue

        for val in value.get("styles", []):
            if not any(
                    _tokens_equal(val, existing)
                    for existing in merged[key]["styles"]):
                merged[key]["styles"].append(val)

        if not merged[key].get("hybrid") and value.get("hybrid"):
            merged[key]["hybrid"] = value["hybrid"]
        if merged[key].get("hybrid") is False and len(value.get("styles")):
            merged[key]["hybrid"] = "hybrid/overlay"

    return merged


def merge_coeffs(coeffs, new_coeffs):
    """
    Merge new_coeffs into coeffs, combining per-category dictionaries.

    For the ``"pair"`` category, coefficients are merged as a nested
    ``{i: {j: (...)}}`` mapping; for all other categories, coefficients
    are merged as a flat ``{type_id: (...)}`` mapping. Raises a
    ValueError on key collisions, since two params files defining the
    same type for the same category almost certainly indicates
    overlapping/conflicting forcefield definitions.

    :param coeffs: Base coefficients dictionary to merge into, keyed by
        category. Mutated in place.
    :type coeffs: dict
    :param new_coeffs: Coefficients dictionary to merge into ``coeffs``,
        in the same format.
    :type new_coeffs: dict

    :returns: The merged coefficients dictionary (the same object as
        ``coeffs``, mutated and returned for convenience).
    :rtype: dict

    :raises ValueError: If the same type (or type pair, for ``"pair"``)
        is defined in both ``coeffs`` and ``new_coeffs`` for the same
        category.
    """
    for category in new_coeffs:
        coeffs.setdefault(category, {})

        if category == "pair":
            # pair coeffs are nested: {i: {j: (...)}}
            for i, row in new_coeffs[category].items():
                coeffs[category].setdefault(i, {})
                for j, coeff in row.items():
                    if j in coeffs[category][i]:
                        raise ValueError(
                            f"Duplicate pair coeff for types ({i}, {j}) "
                            f"found while merging parameter files.")
                    coeffs[category][i][j] = coeff
        else:
            for type_id, coeff in new_coeffs[category].items():
                if type_id in coeffs[category]:
                    raise ValueError(
                        f"Duplicate {category} coeff for type {type_id} "
                        f"found while merging parameter files.")
                coeffs[category][type_id] = coeff

    return coeffs


def read_used_types(datafile, atom_style='full'):
    """
    Determine which pair/bond/angle/dihedral/improper types are actually
    used by the atoms/bonds/angles/dihedrals/impropers in a LAMMPS data
    file, and read the per-type masses.

    The data file is scanned section by section (Masses, Atoms, Bonds,
    Angles, Dihedrals, Impropers). The header counts (e.g. "N atoms",
    "N bonds") are read first so that each section reads exactly the
    right number of data lines; once that many lines have been consumed,
    the section is closed, so any following section (e.g. Velocities,
    Pair Coeffs) can never be misread as more data for the previous
    section. Lines are stripped of comments (text after ``#``), and
    blank lines are skipped. The atom type column used to determine used
    pair types depends on ``atom_style``.

    :param datafile: Path to the LAMMPS data file to scan.
    :type datafile: str
    :param atom_style: LAMMPS atom style, used to determine which column
        of the Atoms section holds the atom type. Supported values are
        "full" (type in column 2) and "charge"/"atomic" (type in
        column 1).
    :type atom_style: str, optional

    :returns:
        A tuple ``(types_used, masses)``: ``types_used`` is a dictionary
        keyed by category (``"pair"``, ``"bond"``, ``"angle"``,
        ``"dihedral"``, ``"improper"``), each mapping to a sorted list of
        the unique type IDs used in that category; ``masses`` is a
        dictionary mapping atom type ID to its mass, as declared in the
        Masses section.
    :rtype: tuple

    :raises ValueError: If ``atom_style`` is not "full", "charge", or
        "atomic".
    """

    if atom_style.lower() == 'full':
        atom_type_column = 2
    elif atom_style.lower() in ('charge', 'atomic'):
        atom_type_column = 1
    else:
        raise ValueError(f"Unsupported atom style: {atom_style}")

    # Section header -> (internal name, header-count keyword)
    known_sections = {
        "Masses": ("mass", None),
        "Atoms": ("atom", "atoms"),
        "Bonds": ("bond", "bonds"),
        "Angles": ("angle", "angles"),
        "Dihedrals": ("dihedral", "dihedrals"),
        "Impropers": ("improper", "impropers"),
    }
    count_keywords = {"atoms", "bonds", "angles", "dihedrals", "impropers"}

    types_used = {
        "pair": set(),
        "bond": set(),
        "angle": set(),
        "dihedral": set(),
        "improper": set()
    }
    masses = {}

    # --- First pass: read header counts (e.g. "1000 atoms", "50 bonds") ---
    counts = {}
    with open(datafile, 'r') as f:
        for line in f:
            clean = line.split('#', 1)[0].strip()
            if not clean:
                continue
            # Header-count section ends once we hit the first real section
            if clean in known_sections:
                break
            parts = clean.split()
            if len(parts) == 2 and parts[0].isdigit(
            ) and parts[1] in count_keywords:
                counts[parts[1]] = int(parts[0])

    # --- Second pass: read section data, respecting counts ---
    section = None  # internal section name ("atom", "bond", ...), or None
    remaining = 0  # how many data lines are still expected in this section

    with open(datafile, 'r') as f:
        for line in f:
            clean = line.split('#', 1)[0].strip()

            if not clean:
                continue

            # Section header line? (always resets state, even mid-count)
            if clean in known_sections:
                section, count_key = known_sections[clean]
                remaining = counts.get(
                    count_key, float('inf')) if count_key else float('inf')
                continue

            # If we've already consumed all expected lines for this
            # section, stop treating further lines as its data (this is
            # what protects Atoms from bleeding into a following
            # Velocities section, etc.)
            if section is not None and remaining <= 0:
                section = None

            if section is None:
                continue

            parts = clean.split()

            # Skip non-data lines
            if not parts or not parts[0].isdigit():
                continue

            if section == "mass":
                masses[int(parts[0])] = float(parts[1])
            elif section == "atom":
                atom_type = int(parts[atom_type_column])
                types_used["pair"].add(atom_type)
            elif section == "bond":
                types_used["bond"].add(int(parts[1]))
            elif section == "angle":
                types_used["angle"].add(int(parts[1]))
            elif section == "dihedral":
                types_used["dihedral"].add(int(parts[1]))
            elif section == "improper":
                types_used["improper"].add(int(parts[1]))

            remaining -= 1

    # Convert sets to sorted lists
    for key in types_used:
        types_used[key] = sorted(types_used[key])

    return types_used, masses


def reduce_types(styles, coeffs, types=None):
    """
    Renumber pair/bond/angle/dihedral/improper types to a contiguous
    range, keeping only the types actually in use.

    If ``types`` is None, no reduction is performed: a type mapping is
    built that maps every existing type in ``coeffs`` to itself, and
    ``styles``/``coeffs`` are returned unchanged. Otherwise, for each
    category present in ``types``, the listed types are renumbered
    starting from 1 in ascending order; pair coefficients are re-keyed
    under their new type pair (with the smaller new type first) and other
    coefficient categories are re-keyed under their new type ID, with the
    type index within each coefficient's argument list updated to match.
    Only types listed in ``types`` are retained in the reduced
    coefficients; the original ``coeffs`` dictionary is mutated in the
    process for the ``"pair"`` category (coefficient entries are updated
    in place before being re-keyed).

    :param styles: Style dictionary, as produced by a style-parsing
        function. Returned unchanged.
    :type styles: dict
    :param coeffs: Coefficients dictionary keyed by category (``"pair"``
        as a nested ``{i: {j: (substyle, parts)}}`` mapping, other
        categories as a flat ``{type_id: (substyle, parts)}`` mapping).
    :type coeffs: dict
    :param types: Mapping of category name to the list of type IDs, in
        that category, that should be retained and renumbered. If None,
        all existing types are kept and mapped to themselves.
    :type types: dict, optional

    :returns:
        A tuple ``(styles, coeffs, type_mapping)``: ``styles`` is the
        (unmodified) input style dictionary; ``coeffs`` is the
        coefficients dictionary with types renumbered (or, if ``types``
        is None, the original ``coeffs`` reference); ``type_mapping`` is
        a dictionary keyed by category, each mapping old type ID to new
        type ID.
    :rtype: tuple
    """
    if types is None:
        type_mapping = {}
        for category in coeffs:
            if category == "pair":
                # pair coeffs are keyed {i: {j: ...}} with string keys
                type_mapping[category] = {
                    int(i): int(i)
                    for i in coeffs[category]
                }
            else:
                type_mapping[category] = {t: t for t in coeffs[category]}
        return styles, coeffs, type_mapping

    old_coeffs = coeffs
    old_pair_coeffs = old_coeffs["pair"]

    type_mapping = {}
    for coeff_type in types:
        type_mapping[coeff_type] = {}
        for new_type, old_type in enumerate(sorted(types.get(coeff_type, [])),
                                            start=1):
            type_mapping[coeff_type][old_type] = new_type

    # Build reduced pair coefficients
    new_pair_coeffs = {}
    for k1 in old_pair_coeffs:
        if k1 not in types["pair"]:
            continue
        new_k1 = type_mapping["pair"][k1]
        for k2, old_coeff in old_pair_coeffs[k1].items():
            if k2 not in types["pair"]:
                continue
            new_k2 = type_mapping["pair"][k2]
            new_coeff = old_coeff
            i, j = min(new_k1, new_k2), max(new_k1, new_k2)
            new_coeff[1][1] = str(i)
            new_coeff[1][2] = str(j)
            if new_pair_coeffs.get(i) is None:
                new_pair_coeffs[i] = {}
            new_pair_coeffs[i][j] = new_coeff

    new_other_coeffs = {}
    for coeff_type, coeff in old_coeffs.items():
        if coeff_type == "pair":
            continue
        new_other_coeffs[coeff_type] = {}
        required = types.get(coeff_type, [])
        for old_type in old_coeffs[coeff_type]:
            if old_type not in required:
                continue
            new_type = type_mapping[coeff_type][old_type]
            new_other_coeffs[coeff_type][new_type] = coeff[old_type]
            new_other_coeffs[coeff_type][new_type][1][1] = str(new_type)

    coeffs = {**new_other_coeffs}
    coeffs["pair"] = new_pair_coeffs

    return styles, coeffs, type_mapping


def reduce_styles(styles, coeffs, extra_coeffs):
    """
    Remove declared styles that are no longer referenced by any
    coefficient, mutating ``styles`` in place.

    For each style category, the set of substyle names actually used is
    collected from ``coeffs``, plus the first token of each entry in
    ``extra_coeffs`` if provided. The ``extra_coeffs`` tokens are
    added to the used set for EVERY category, in addition to (not instead
    of) those found in ``coeffs``. Any declared style in ``styles``
    whose leading token is not in the used set is dropped. After
    filtering, a category's ``hybrid`` flag is reset to False if fewer
    than two styles remain, or set to ``"hybrid/overlay"`` if two or more
    styles remain but ``hybrid`` was previously False. ``coeffs`` must
    contain an entry for every category in ``styles``.

    :param styles: Style dictionary keyed by category (e.g. ``"pair"``,
        ``"bond"``), each mapping to a dict with ``"styles"`` (list of
        style strings) and ``"hybrid"``. Mutated in place.
    :type styles: dict
    :param coeffs: Coefficients dictionary keyed by category, used to
        determine which substyles are actually in use. ``"pair"``
        coefficients are a nested ``{i: {j: (substyle, parts)}}``
        mapping; other categories are a flat ``{type_id: (substyle,
        parts)}`` mapping.
    :type coeffs: dict
    :param extra_coeffs: Optional list of raw extra coefficient lines
        (e.g. manually-added lines); if provided, the first token of each
        line is used, for every style category, as the used-style list
        instead of styles derived from ``coeffs``.
    :type extra_coeffs: list of str, optional
    :param extra_coeffs: Optional list of raw extra lines (e.g. extra
        pair styles); the first token of each line is treated as an
        in-use style name. This only prevents matching styles from being
        removed; it does not add styles that are not already in
        ``styles``.
    :type extra_coeffs: list of str, optional

    :returns: A tuple ``(styles, coeffs)`` -- the same objects passed in,
        with ``styles`` filtered down to only the styles still in use.
    :rtype: tuple
    """

    used_styles = {}
    for style_type in styles:
        used_styles[style_type] = []

    for style_type in styles:
        if extra_coeffs:
            for ec in extra_coeffs:
                used_styles[style_type].append(ec.split()[0])
        for i, coeff in coeffs[style_type].items():
            if style_type == "pair":
                for j in coeff:
                    used_styles[style_type].append(coeff[j][0].split()[0])
            else:
                used_styles[style_type].append(coeff[0])

    # Remove styles that are no longer being used
    for style_type in styles:
        required = used_styles[style_type]
        styles[style_type]["styles"] = [
            s for s in styles[style_type]["styles"]
            if s.strip().split()[0] in required
        ]

        # Adjust hybrid if reverted
        if len(styles[style_type]["styles"]) < 2:
            styles[style_type]["hybrid"] = False
        elif styles[style_type]["hybrid"] is False:
            styles[style_type]["hybrid"] = "hybrid/overlay"

    return styles, coeffs


def reduce_data_file(datafile,
                     type_mapping,
                     masses,
                     atom_style,
                     outfile=None,
                     reduced_counts=None):
    """
    Rewrite a LAMMPS .data file using reduced type mappings.

    The following are modified:
        - header type counts (from ``reduced_counts``)
        - Masses (replaced entirely by ``masses``)
        - atom/bond/angle/dihedral/improper type IDs
        - Pair Coeffs are removed

    Atoms, Bonds, Angles, Dihedrals, and Impropers lines are rewritten
    with normalized whitespace and any trailing comments removed. All
    other content is preserved as written.

    :param datafile: Path to the input LAMMPS data file.
    :type datafile: str
    :param type_mapping: Mapping from old type IDs to new type IDs, keyed
        by category (``"pair"``, ``"bond"``, ``"angle"``, ``"dihedral"``,
        ``"improper"``).
    :type type_mapping: dict
    :param masses: Final atom-type -> mass mapping, written into the
        rewritten file's Masses section.
    :type masses: dict
    :param atom_style: LAMMPS atom style, used to determine which column
        of the Atoms section holds the atom type. Currently supports
        'full', 'charge', and 'atomic'.
    :type atom_style: str
    :param outfile: Path to write the rewritten data file to. Defaults to
        ``f"{datafile}.reduced"`` if not provided.
    :type outfile: str, optional
    :param reduced_counts: Mapping of header keys (e.g. ``"atom types"``,
        ``"bond types"``) to their new counts, substituted into the
        rewritten file's header lines. Required in practice; passing None
        raises a TypeError.
    :type reduced_counts: dict

    :raises ValueError: If ``atom_style`` is not "full", "charge", or
        "atomic", if a type ID cannot be parsed, or if a type ID in the
        data file has no entry in ``type_mapping``.
    :param reduced_counts: Mapping of header keys (e.g. ``"atom types"``,
        ``"bond types"``) to their new counts, substituted into the
        rewritten file's header lines.
    :type reduced_counts: dict, optional

    :returns: Path to the rewritten data file.
    :rtype: str

    :raises ValueError: If ``atom_style`` is not "full", "charge", or
        "atomic", or if a type ID in the data file has no corresponding
        entry in ``type_mapping``.
    """

    if outfile is None:
        outfile = f"{datafile}.reduced"

    if atom_style.lower() == 'full':
        atom_type_column = 2
    elif atom_style.lower() in ('charge', 'atomic'):
        atom_type_column = 1
    else:
        raise ValueError(f"Unsupported atom style: {atom_style}")

    with open(datafile, 'r') as f:
        lines = f.readlines()

    old_counts = {
        'atoms': 0,
        'bonds': 0,
        'angles': 0,
        'dihedrals': 0,
        'impropers': 0,
    }

    for line in lines:
        clean = line.split('#', 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue
        number = _try_int(parts[0])

        if number is None:
            continue

        if parts[1].lower() in old_counts:
            old_counts[parts[1].lower()] = number

    # Location of ID
    section_mapping = {
        'Atoms': ('pair', atom_type_column),
        'Bonds': ('bond', 1),
        'Angles': ('angle', 1),
        'Dihedrals': ('dihedral', 1),
        'Impropers': ('improper', 1),
    }

    section_names = (
        'Masses',
        'Atoms',
        'Bonds',
        'Angles',
        'Dihedrals',
        'Impropers',
        'Pair Coeffs',
    )

    section = None
    masses_written = False

    section_counters = {
        'Atoms': 0,
        'Bonds': 0,
        'Angles': 0,
        'Dihedrals': 0,
        'Impropers': 0,
    }

    new_lines = []
    for line in lines:
        clean = line.split('#', 1)[0].strip()

        if not clean:
            if section != 'Pair Coeffs':
                new_lines.append(line)
            continue

        if clean in section_names:
            section = clean

            if section == 'Pair Coeffs':
                continue

            if section == 'Masses':
                new_lines.append(line)
                if not masses_written:
                    for atom_type, mass in sorted(masses.items()):
                        new_lines.append(f'{atom_type:<10}{mass:<10}\n')
                    masses_written = True
                continue

            new_lines.append(line)
            section_counters[section] = 0
            continue

        if section == 'Pair Coeffs':
            continue

        if section == 'Masses':
            continue

        parts = clean.split()
        if len(parts) >= 2:
            if len(parts) == 2:
                header_key = parts[1].lower()
                if header_key in (
                        'atoms',
                        'bonds',
                        'angles',
                        'dihedrals',
                        'impropers',
                ):
                    new_lines.append(line)
                    continue

            if len(parts) >= 3:
                header_key = (f'{parts[1].lower()} {parts[2].lower()}')

                if header_key in reduced_counts:
                    leading_space = (line[:len(line) - len(line.lstrip())])
                    new_value = reduced_counts[header_key]
                    new_lines.append(f'{leading_space}{new_value} '
                                     f'{" ".join(parts[1:])}\n')
                    continue

        if section in section_mapping:
            mapping_key, type_column = section_mapping[section]
            old_type = _try_int(parts[type_column])

            if old_type is None:
                raise ValueError(f'Could not determine {mapping_key} type '
                                 f'in line:\n{line}')

            mapping = type_mapping.get(mapping_key, {})
            if old_type not in mapping:
                raise ValueError(
                    f'Old {mapping_key} type {old_type} exists in '
                    f'{datafile}, but has no mapping.')

            # Replace the type ID
            parts[type_column] = str(mapping[old_type])

            if section == 'Atoms':
                new_lines.append(''.join(f'{x:<6}' for x in parts[:3])
                                 + ''.join(f'{x:<25}'
                                           for x in parts[3:]) + '\n')
            else:
                new_lines.append('    '.join(parts) + '\n')
            section_counters[section] += 1

            # Once all entries in the original section have
            # been processed, return to the neutral state.
            if (section_counters[section] >= old_counts[section.lower()]):
                section = None
            continue

        new_lines.append(line)

    with open(outfile, 'w') as f:
        f.writelines(new_lines)

    return outfile


def _normalize_groups(files, n_expected, name, allow_none=False):
    """
    Normalize str/tuple/list input into a list of groups (tuples), one per
    forcefield entry. A bare str or tuple broadcasts across all entries.

    :param files: The value to normalize: a single string or tuple
        (broadcast to every group), a list with one entry per group (each
        entry a string, tuple, or, if ``allow_none`` is True, None), or
        None (only valid if ``allow_none`` is True).
    :type files: str, tuple, list, or None
    :param n_expected: The number of forcefield groups expected.
    :type n_expected: int
    :param name: Name of the parameter being normalized, used in error
        messages.
    :type name: str
    :param allow_none: Whether None is an acceptable value or list entry,
        producing None for that group instead of a tuple.
    :type allow_none: bool, optional

    :returns: A list of length ``n_expected``, where each entry is a
        tuple of strings for that group, or None if ``allow_none`` and no
        value was given for that group.
    :rtype: list

    :raises ValueError: If ``files`` is a list whose length does not
        match ``n_expected``.
    :raises TypeError: If ``files``, or an entry within it, is not a
        recognized type (str, tuple, list of str/tuple, or None when
        allowed).
    """
    if allow_none and files is None:
        return [None] * n_expected

    if isinstance(files, (str, tuple)):
        group = (files, ) if isinstance(files, str) else files
        return [group] * n_expected

    if isinstance(files, list):
        if len(files) != n_expected:
            raise ValueError(
                f"{name} is a list of length {len(files)} but there are "
                f"{n_expected} forcefield group(s). Pass a str or tuple "
                f"instead if this should be shared across all groups.")
        groups = []
        for f in files:
            if allow_none and f is None:
                groups.append(None)
            elif isinstance(f, str):
                groups.append((f, ))
            elif isinstance(f, tuple):
                groups.append(f)
            else:
                raise TypeError(f"Invalid entry in {name}: {f!r}")
        return groups

    raise TypeError(f"{name} must be a str, tuple, or list of str/tuple"
                    f"{'/None' if allow_none else ''}.")


def merge_used_types(used_types, new_used_types):
    """
    Union two 'used types' dicts together, category by category.

    :param used_types: Base 'used types' dictionary, e.g.
        ``{"pair": [...], "bond": [...], ...}``. Mutated in place.
    :type used_types: dict
    :param new_used_types: 'Used types' dictionary to union into
        ``used_types``, in the same format.
    :type new_used_types: dict

    :returns: The merged 'used types' dictionary (the same object as
        ``used_types``, mutated and returned for convenience), with each
        category's list containing the union of both inputs' type IDs,
        de-duplicated but not sorted.
    :rtype: dict
    """
    for category in new_used_types:
        used_types.setdefault(category, [])
        for t in new_used_types[category]:
            if t not in used_types[category]:
                used_types[category].append(t)
    return used_types


def forcefield_merger(style_files,
                      params_files,
                      data_files=None,
                      atom_style='full',
                      extra_pair_styles=None,
                      extra_coeff_lines=None,
                      stack_ff=True,
                      mixing_rule=None,
                      cross_pairstyle=None,
                      override_cross_ps=None,
                      outparams=None,
                      outstyle=None,
                      reduce=True,
                      logger=None):
    """
    Parse, merge, and type-reduce one or more LAMMPS forcefields, then
    optionally compute cross interactions and write the merged result to
    disk.

    Style and parameter files are grouped into one or more "forcefield
    groups" (one group per entry in ``style_files``/``params_files``,
    after normalization); each group's styles and coefficients are parsed
    and merged with :func:`parse_ff`/:func:`merge_styles`/
    :func:`merge_coeffs`. If ``data_files`` is given, the atom types
    actually used by each group's data files are determined via
    :func:`read_used_types` and used to reduce that group's types to a
    contiguous range with :func:`reduce_types`. If the groups' data-file
    masses differ, ``stack_ff`` is forced to True and each group's types
    are kept distinct by offsetting them. If the masses match and
    ``stack_ff=False``, every group is reduced against the shared union
    of used types, so matching types are combined rather than offset. If
    any group has no data files, that group is not reduced and, with
    ``stack_ff=False``, no groups are offset (types are assumed to
    already be consistent). Data files for each group are rewritten with
    the final type mapping via :func:`reduce_data_file`. Unused styles are
    then dropped via :func:`reduce_styles`, cross interactions are
    computed via :func:`compute_cross_interactions` if both
    ``mixing_rule`` and ``cross_pairstyle`` are given, and the merged
    coefficients/styles are optionally written to
    ``outparams``/``outstyle``.

    :param style_files: Style file path(s), one per forcefield group, or
        a single string to use for every group (see
        :func:`_normalize_groups`).
    :type style_files: str or list of str
    :param params_files: Parameter file path(s) to merge into each
        forcefield group. May be a single string/tuple (broadcast to all
        groups) or a list with one entry (string or tuple of strings) per
        group.
    :type params_files: str, tuple, or list
    :param data_files: Data file path(s) associated with each forcefield
        group, used to determine used types for type reduction. May be a
        single string/tuple (broadcast to all groups), a list with one
        entry per group, or None/contain None for groups with no
        associated data files (which are left unreduced and not
        rewritten).
    :type data_files: str, tuple, list, or None, optional
    :param atom_style: LAMMPS atom style used when rewriting data files
        with :func:`reduce_data_file`. (Note: it is currently not passed
        to :func:`read_used_types`, which always assumes "full".)
    :type atom_style: str, optional
    :param extra_pair_styles: Extra lines (e.g. manually specified pair
        styles) whose leading tokens are treated as in-use styles by
        :func:`reduce_styles`, so matching declared styles are not
        dropped. They are not added to the style declarations by this
        function.
    :type extra_pair_styles: list of str, optional
    :param extra_coeff_lines: Additional raw coefficient lines written
        verbatim at the top of ``outparams``, if given.
    :type extra_coeff_lines: list of str, optional
    :param stack_ff: Whether to keep each forcefield group's types
        distinct (offsetting them) rather than combining matching types
        across groups. Overridden to True automatically if the groups'
        masses differ.
    :type stack_ff: bool, optional
    :param mixing_rule: Mixing rule ("arithmetic" or "geometric") used to
        compute missing cross pair interactions, if given together with
        ``cross_pairstyle``.
    :type mixing_rule: str, optional
    :param cross_pairstyle: Pair substyle to use for computed cross
        interactions when the two interacting types don't already share a
        substyle (or when ``override_cross_ps`` is set).
    :type cross_pairstyle: str, optional
    :param override_cross_ps: If True, always use ``cross_pairstyle`` for
        computed cross interactions, even when both types already share
        the same substyle.
    :type override_cross_ps: bool, optional
    :param outparams: Path to write the final merged coefficients to, via
        :func:`write_parameter_file`. If not given, no parameter file is
        written.
    :type outparams: str, optional
    :param outstyle: Path to write the final merged style declarations
        to, via :func:`write_style_file`. If not given, no style file is
        written.
    :type outstyle: str, optional
    :param reduce: If True, pair types are reduced to only those used by
        atoms in the data files. If not True, all atom types declared in
        each data file's Masses section are kept (bond, angle, dihedral,
        and improper types are still reduced to those in use).
    :type reduce: bool, optional
    :param logger: Optional logger. The stacking/mass-comparison message
        is sent to ``logger.info`` if provided, otherwise printed.
    :type logger: logging.Logger, optional

    :returns:
        A tuple ``(final_styles, final_coeffs, reduced_names)``: the
        merged and reduced style dictionary, the merged and reduced
        coefficients dictionary, and the list of paths to the rewritten
        data files, named ``<data_file>.<group index>.reduced``.
    :rtype: tuple

    :raises TypeError: If ``style_files`` or an entry in
        ``params_files``/``data_files`` has an unsupported type.
    :raises ValueError: If ``style_files`` and the normalized
        ``params_files`` groups have mismatched lengths, or if a required
        type mapping is missing during reduction.
    """

    if isinstance(style_files, str):
        style_files = [style_files]
    if not isinstance(style_files, list) or not all(
            isinstance(s, str) for s in style_files):
        raise TypeError("style_files must be a str or a list of str.")

    params_groups = _normalize_groups(params_files, len(style_files),
                                      "params_files")

    if len(style_files) == 1 and len(params_groups) > 1:
        style_files = style_files * len(params_groups)
    if len(style_files) != len(params_groups):
        raise ValueError(
            f"style_files has {len(style_files)} entr(y/ies) but params_files "
            f"resolved to {len(params_groups)} group(s). These must match "
            f"(or style_files may be a single str to broadcast).")

    n_ff = len(style_files)
    data_groups = _normalize_groups(data_files,
                                    n_ff,
                                    "data_files",
                                    allow_none=True)

    styles = {}  # merged globally across all forcefield groups
    coeffs = []  # one merged coeffs dict per forcefield group
    for style_file, params_group in zip(style_files, params_groups):
        ff_coeffs = {}
        for params_file in params_group:
            new_styles, new_coeffs = parse_ff(style_file, params_file)
            styles = merge_styles(styles, new_styles)
            ff_coeffs = merge_coeffs(ff_coeffs, new_coeffs)
        coeffs.append(ff_coeffs)

    used_types = []
    masses = []
    for data_group in data_groups:
        if data_group is None:
            used_types.append(None)
            masses.append(None)
            continue
        group_used_types = {}
        group_masses = {}
        for data_file in data_group:
            file_used_types, file_masses = read_used_types(data_file)
            if reduce is not True:
                file_used_types["pair"] = list(file_masses.keys())
            group_used_types = merge_used_types(group_used_types,
                                                file_used_types)
            group_masses.update(file_masses)
        used_types.append(group_used_types)
        masses.append(group_masses)

    # Check if masses give the same dictionaries
    if all(mass == masses[0] for mass in masses):
        msg = "All mass dictionaries are the same"
        if stack_ff is True:
            print("Offsetting types for consecutive forcefields")
    elif stack_ff is False:
        msg = "Mass dictionaries differ, but stack_ff=False passed...\n"
        msg += "Reverting to stack_ff=True."
        stack_ff = True
    else:
        msg = "Mass dictionaries differ, stacking forcefields"

    if logger is None:
        print(msg)
    else:
        logger.info(msg)

    # Need to modify used_types to include ALL used types so reduction
    # is consistent across files
    if not stack_ff:
        if any(u is None for u in used_types):
            master_used_types = None
        else:
            master_used_types = {}
            for used_types_i in used_types:
                master_used_types = merge_used_types(master_used_types,
                                                     used_types_i)
        used_types = [master_used_types] * len(used_types)

    # Reduce the types to 1 .. N within each ff
    all_maps = []
    reduced_coeffs = []
    for coeffs_i, types_i in zip(coeffs, used_types):
        _, reduced_coeffs_i, type_mapping_i = reduce_types(
            styles, coeffs_i, types_i)
        reduced_coeffs.append(reduced_coeffs_i)
        all_maps.append(type_mapping_i)

    # Offset types and rebuild master masses dictionary
    offsets = {"pair": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}
    offset_maps = []
    final_masses = {}
    final_coeffs = {}
    for type_map, mass, group_coeffs in zip(all_maps, masses, reduced_coeffs):
        offset_map = {}
        for category, mapping in type_map.items():
            final_coeffs.setdefault(category, {})
            offset_map[category] = {}

            # Build globally offset type mapping
            for old_type, reduced_type in mapping.items():
                offset_map[category][old_type] = (int(reduced_type)
                                                  + offsets[category])

            if category != "pair":
                for reduced_type, coeff in group_coeffs[category].items():
                    new_type = int(reduced_type) + offsets[category]
                    substyle, parts = coeff
                    parts = list(parts)
                    parts[1] = str(new_type)
                    final_coeffs[category][new_type] = (substyle, parts)
            else:
                for reduced_i, row in group_coeffs[category].items():
                    new_i = int(reduced_i) + offsets[category]
                    for reduced_j, coeff in row.items():
                        new_j = int(reduced_j) + offsets[category]
                        i, j = min(new_i, new_j), max(new_i, new_j)
                        substyle, parts = coeff
                        parts = list(parts)
                        parts[1] = str(i)
                        parts[2] = str(j)
                        final_coeffs[category].setdefault(i,
                                                          {})[j] = (substyle,
                                                                    parts)

            if stack_ff and mapping:
                offsets[category] = max(mapping.values())

        offset_maps.append(offset_map)

        # Final masses: new_type -> mass
        if mass is not None:
            for old_type, new_type in offset_map["pair"].items():
                final_masses[new_type] = mass[old_type]

    # Compute global reduced counts across ALL groups
    global_reduced_counts = {
        'atom types':
        len(
            set(new_type for om in offset_maps
                for new_type in om["pair"].values())),
        'bond types':
        len(
            set(new_type for om in offset_maps
                for new_type in om["bond"].values())),
        'angle types':
        len(
            set(new_type for om in offset_maps
                for new_type in om["angle"].values())),
        'dihedral types':
        len(
            set(new_type for om in offset_maps
                for new_type in om["dihedral"].values())),
        'improper types':
        len(
            set(new_type for om in offset_maps
                for new_type in om["improper"].values())),
    }

    # Re-write files with .reduced format
    reduced_names = []
    for ff_idx, (data_group,
                 type_map) in enumerate(zip(data_groups, offset_maps)):
        if data_group is None:
            continue  # nothing to rewrite for group with no data files
        for data_file in data_group:
            reduced_name = f"{data_file}.{ff_idx}.reduced"
            reduce_data_file(data_file,
                             type_map,
                             final_masses,
                             atom_style,
                             reduced_name,
                             reduced_counts=global_reduced_counts)
            reduced_names.append(reduced_name)

    # Remove unused styles
    final_styles, final_coeffs = reduce_styles(styles, final_coeffs,
                                               extra_pair_styles)

    # Define cross-terms, if requested
    if mixing_rule and cross_pairstyle:
        final_styles, final_coeffs = compute_cross_interactions(
            final_styles,
            final_coeffs,
            mixing_rule=mixing_rule,
            cross_pairstyle=cross_pairstyle,
            override_ps=override_cross_ps)

    # Print coefficients
    if outparams:
        write_parameter_file(final_styles,
                             final_coeffs,
                             extra_coeff_lines,
                             outfile=outparams)
    if outstyle:
        write_style_file(final_styles, outfile=outstyle)

    return final_styles, final_coeffs, reduced_names


def compute_cross_interactions(styles,
                               coeffs,
                               cross_pairstyle="lj/cut",
                               mixing_rule='arithmetic',
                               override_ps=False):
    """
    Compute missing pair-pair cross interactions using a combining rule.

    This function will compute cross-interactions and should be run only
    after all interactions are specified and reduced. For every pair of
    types ``(i, j)`` with ``i <= j`` that does not already have a defined
    pair coefficient, epsilon and sigma are combined from the existing
    ``(i, i)`` and ``(j, j)`` coefficients according to ``mixing_rule``.
    The cross term's substyle is taken from the shared substyle of ``i``
    and ``j`` if they match and ``override_ps`` is False, otherwise
    ``cross_pairstyle`` is used. Newly used substyles are added to the
    declared pair styles, and the ``hybrid`` flag is updated accordingly.
    Mutates ``styles`` and ``coeffs`` in place.

    :param styles: Style dictionary, as produced by a style-parsing
        function, containing the declared pair styles under
        ``styles["pair"]``. Mutated in place.
    :type styles: dict
    :param coeffs: Coefficients dictionary containing existing pair
        coefficients under ``coeffs["pair"]``, keyed as
        ``{i: {j: (substyle, parts)}}`` with self-interactions
        (``i == j``) already present for every type. Mutated in place.
    :type coeffs: dict
    :param cross_pairstyle: Pair substyle to use for a cross interaction
        when the two types don't share a substyle (or when
        ``override_ps`` is True).
    :type cross_pairstyle: str, optional
    :param mixing_rule: Combining rule for epsilon/sigma. Must be
        "arithmetic" (sigma averaged, epsilon geometric mean) or
        "geometric" (both geometric mean).
    :type mixing_rule: str, optional
    :param override_ps: If True, always use ``cross_pairstyle`` for cross
        interactions, even when both types already share the same
        substyle.
    :type override_ps: bool, optional

    :returns: A tuple ``(styles, coeffs)`` -- the same objects passed in,
        updated with the newly computed cross interactions.
    :rtype: tuple

    :raises ValueError: If ``mixing_rule`` is not "arithmetic" or
        "geometric".
    """
    pair_coeffs = coeffs["pair"]
    used_pairstyles = set()

    for i in sorted(pair_coeffs):
        for j in sorted(pair_coeffs):
            if j < i:
                continue

            ij_coeff = coeffs["pair"][i].get(j)
            if ij_coeff:
                continue

            ii_coeff = coeffs["pair"][i][i]
            ii_epsilon, ii_sigma = float(ii_coeff[1][3]), float(ii_coeff[1][4])

            jj_coeff = coeffs["pair"][j][j]
            jj_epsilon, jj_sigma = float(jj_coeff[1][3]), float(jj_coeff[1][4])

            if ii_coeff[0] == jj_coeff[0] and not override_ps:
                ij_substyle = ii_coeff[0]
            else:
                ij_substyle = cross_pairstyle

            ij_parts = list(ii_coeff[1])
            ij_parts[1] = str(i)
            ij_parts[2] = str(j)

            if mixing_rule == "arithmetic":
                ij_sigma = (ii_sigma + jj_sigma) / 2
                ij_epsilon = (ii_epsilon * jj_epsilon)**0.5
            elif mixing_rule == "geometric":
                ij_sigma = (ii_sigma * jj_sigma)**0.5
                ij_epsilon = (ii_epsilon * jj_epsilon)**0.5
            else:
                raise ValueError(f"Unknown mixing rule: {mixing_rule}")

            ij_parts[3] = str(round(ij_epsilon, 5))
            ij_parts[4] = str(round(ij_sigma, 5))

            cross_coeff = (ij_substyle, ij_parts)
            coeffs["pair"][i][j] = cross_coeff
            used_pairstyles.add(ij_substyle)

    for ps in used_pairstyles:
        if ps not in [s.split()[0] for s in styles["pair"]["styles"]]:
            styles["pair"]["styles"].append(ps)

    n_styles = len(styles["pair"]["styles"])
    if n_styles < 2:
        styles["pair"]["hybrid"] = False
    elif styles["pair"].get("hybrid") is False:
        styles["pair"]["hybrid"] = "hybrid/overlay"

    return styles, coeffs


def build_style_args(styles, style_type):
    """
    Build the LAMMPS ``*_style`` argument string for one style category.

    :param styles: Style dictionary keyed by category, each mapping to a
        dict with ``"styles"`` (list of style strings) and ``"hybrid"``
        (hybrid designation string or False).
    :type styles: dict
    :param style_type: The style category to build the argument string
        for (e.g. ``"pair"``, ``"bond"``).
    :type style_type: str

    :returns: The style argument string: ``"none"`` if ``style_type`` is
        not present in ``styles``; the single declared style if not
        hybrid; or the hybrid designation followed by all declared
        substyles, space-separated, if hybrid.
    :rtype: str
    """
    style = styles.get(style_type)
    if style is None:
        string = "none"
    elif style.get("hybrid") is False:
        string = " ".join(style.get("styles"))
    elif style.get("hybrid") is not False:
        string = style.get("hybrid") + " "
        string += " ".join(style.get("styles"))
    return string


def build_style_strings(styles):
    """
    Build LAMMPS ``*_style`` argument strings for every style category.

    :param styles: Style dictionary keyed by category, as used by
        :func:`build_style_args`.
    :type styles: dict

    :returns: A dictionary mapping each category's ``"{category}_style"``
        key (e.g. ``"pair_style"``) to its built argument string, for
        every category in ``styles`` whose argument string is non-empty.
    :rtype: dict
    """
    style_dict = {}
    for style in styles:
        string = build_style_args(styles, style)
        if string:
            style_dict[f"{style}_style"] = string
    return style_dict


def write_style_file(styles, outfile):
    """
    Write LAMMPS ``*_style`` declaration lines to a file.

    Style argument strings are built via :func:`build_style_strings`, and
    a line is written for each of ``pair``, ``bond``, ``angle``,
    ``dihedral``, and ``improper`` (in that order) that has a non-empty
    argument string; categories with no declared style are skipped.

    :param styles: Style dictionary keyed by category, as used by
        :func:`build_style_strings`.
    :type styles: dict
    :param outfile: Path to the file to create/overwrite with the style
        declaration lines.
    :type outfile: str

    :returns: None. The style declarations are written to ``outfile``.
    :rtype: None
    """
    style_strings = build_style_strings(styles)
    with open(outfile, 'w') as f:
        for style in ('pair', 'bond', 'angle', 'dihedral', 'improper'):
            args = style_strings.get(f'{style}_style')
            if not args:
                continue
            f.write(f'{style}_style {args}\n')
