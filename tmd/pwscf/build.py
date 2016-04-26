import os
import tmd.pwscf.cell as cell

def build_pw2wan(material):
    pw2wan = [" &inputpp"]
    pw2wan.append("   outdir='./',")
    pw2wan.append("   prefix='{}',".format(material["prefix"]))
    pw2wan.append("   write_mmn=.true.,")
    pw2wan.append("   write_amn=.true.,")

    if material["soc"]:
        pw2wan.append("   write_spn=.true.,")
    else:
        pw2wan.append("   write_spn=.false.,")

    # TODO - support collinear spin-polarized case

    pw2wan.append("   write_unk=.false.,")
    pw2wan.append("   seedname='{}'".format(material["prefix"]))
    pw2wan.append(" /\n")

    return "\n".join(pw2wan)

def build_bands(material):
    '''Construct a string which gives the QE input file for bands postprocessing
    for the given material.

    material: a dict in the format of the entries in materials.yaml.
    '''
    bands = [" &bands"]
    bands.append("   prefix='{}',".format(material["prefix"]))
    bands.append("   outdir='./',")
    bands.append("   filband='{}_bands.dat'".format(material["prefix"]))
    bands.append(" /\n")

    return "\n".join(bands)

def build_qe(material, calc_type):
    '''Construct a string which gives the QE input file for the specified
    calculation.

    material: a dict in the format of the entries in materials.yaml.
    calc_type: one of 'relax', 'scf', 'nscf', 'bands'.
    '''
    calc_name = material["prefix"]
    calc_vals = ["relax", "scf", "nscf", "bands"]
    if calc_type not in calc_vals:
        raise ValueError("Unsupported calc_type " + calc_type)

    pseudo_dir = os.path.expandvars(material["pseudo_dir"])

    axes, latpos = cell.build(material["latconst"], material["latvecs"],
            material["cartpos"], material["vacuum_dist"])

    conv_thr = material[calc_type + "_conv_thr"]

    if calc_type == 'relax':
        etot_conv_thr = material["relax_etot_conv_thr"]
        forc_conv_thr = material["relax_forc_conv_thr"]
    else:
        etot_conv_thr, forc_conv_thr = None, None

    # Make strings representing each section of the PW input.
    # These may be None.
    control = _control(calc_type, pseudo_dir, etot_conv_thr, forc_conv_thr, calc_name)
    system = _system(calc_type, material)
    electrons = _electrons(conv_thr)
    ions = _ions(calc_type)
    atomic_species = _atomic_species(material["pseudo"], material["weight"])
    cell_parameters = _cell_parameters(axes)
    atomic_positions = _atomic_positions(latpos)
    k_points = _k_points(calc_type, material)

    # Join the sections with newlines, ignoring any None values.
    pw_input = _join([control, system, electrons, ions, atomic_species, cell_parameters,
        atomic_positions, k_points])

    return pw_input

def _join(xs):
    ret = None
    for x in xs:
        if ret == None and x != None:
            ret = x
        elif x != None:
            ret = "\n".join([ret, x])
    return ret + "\n"

def wannier_num_bands(valence):
    '''Number of bands to use in nscf/bands calculation for use with Wannier90.

    Originally used:
    Returns the total number of valence electrons times 28/18, the ratio used
    in the Wannier90 AHC paper Phys. Rev. B 74, 195118 (2006) (see section IV.A.)

    Changed to factor of 2 to include additional bands above TMD conduction
    band group (28/18 seems to just barely include full TMD conduction bands group;
    for some d values, this group overlaps with group above it).

    Since valence is reduced when soc is False, do not need to change the
    calculation here based on SOC.
    '''
    return int(valence["total"] * 2)

def _control(calc_type, pseudo_dir, etot_conv_thr, forc_conv_thr, calc_name):
    nl = [" &control"]
    nl.append("    calculation='{}',".format(calc_type))
    nl.append("    restart_mode='from_scratch',")
    nl.append("    disk_io='low',")
    nl.append("    wf_collect=.true.,")
    nl.append("    pseudo_dir='{}',".format(pseudo_dir))
    nl.append("    outdir='./',")

    if calc_type == 'relax':
        nl.append("    etot_conv_thr={},".format(str(etot_conv_thr)))
        nl.append("    forc_conv_thr={},".format(str(forc_conv_thr)))

    nl.append("    prefix='{}'".format(calc_name))
    nl.append(" /")
    return "\n".join(nl)

def _system(calc_type, material):
    bohr_in_A = 1.889726164
    latconst_bohr = bohr_in_A * material["latconst"]

    num_atoms, num_atom_types = _atom_types(material["cartpos"])
    num_bands = wannier_num_bands(material["valence"])

    nl = [" &system"]
    nl.append("    ibrav=0,celldm(1)={},nat={},ntyp={},".format(str(latconst_bohr),
        str(num_atoms), str(num_atom_types)))
    nl.append("    ecutwfc={},".format(str(material["ecutwfc"])))
    nl.append("    ecutrho={},".format(str(material["ecutrho"])))

    if material["soc"]:
        nl.append("    noncolin=.true.,")
        nl.append("    lspinorb=.true.,")

    if calc_type in ['scf', 'relax']:
        nl.append("    occupations='tetrahedra'")
    else:
        nl.append("    nosym=.true.,")
        nl.append("    nbnd={},".format(str(num_bands)))
        nl.append("    occupations='smearing',smearing='cold',degauss={}".format(str(material["degauss"])))

    nl.append(" /")    
    return "\n".join(nl)

def _atom_types(cartpos):
    num_atoms = len(cartpos)
    atom_types = set()
    for at, pos in cartpos:
        atom_types.add(at)

    num_atom_types = len(atom_types)
    return num_atoms, num_atom_types

def _electrons(conv_thr):
    nl = [" &electrons"]
    nl.append("    startingwfc='random',")
    nl.append("    diagonalization='david',")
    nl.append("    conv_thr={}".format(str(conv_thr)))
    nl.append(" /")
    return "\n".join(nl)

def _ions(calc_type):
    if calc_type != 'relax':
        return None

    nl = [" &ions"]
    nl.append("    ion_dynamics='bfgs'")
    nl.append(" /")
    return "\n".join(nl)

def _atomic_species(pseudo, weight):
    card = ["ATOMIC_SPECIES"]
    for k, v in pseudo.items():
        card.append(" {} {} {}".format(k, weight[k], v))
    return "\n".join(card)

def _cell_parameters(axes):
    card = ["CELL_PARAMETERS alat"]
    for ax in range(3):
        ax, ay, az = str(axes[ax][0]), str(axes[ax][1]), str(axes[ax][2])
        card.append(" {}    {}    {}".format(ax, ay, az))
    return "\n".join(card)

def _atomic_positions(pos):
    card = ["ATOMIC_POSITIONS crystal"]
    for atom, p in pos:
        pa, pb, pc = str(p[0]), str(p[1]), str(p[2])
        card.append(" {} {} {} {}".format(atom, pa, pb, pc))
    return "\n".join(card)

def _k_points(calc_type, material):
    if calc_type in ['scf', 'relax']:
        Nk1 = material["{}_Nk1".format(calc_type)]
        Nk2 = material["{}_Nk2".format(calc_type)]

        card = ["K_POINTS automatic"]
        card.append("{} {} 1 0 0 0".format(Nk1, Nk2))
        return "\n".join(card)
    elif calc_type == 'nscf':
        Nk1 = material["nscf_Nk1"]
        Nk2 = material["nscf_Nk2"]
        Nks = Nk1*Nk2
        weight = 1.0/Nks
        nscf_klist = nscf_ks(Nk1, Nk2)

        card = ["K_POINTS crystal"]
        card.append("{}".format(str(Nk1 * Nk2)))
        for k in nscf_klist:
            card.append("    {} {} {} {}".format(str(k[0]), str(k[1]), str(k[2]), str(weight)))

        return "\n".join(card)
    else:
        # calc_type == 'bands'
        num_points = len(material["band_path"])

        card = ["K_POINTS crystal_b"]
        card.append("{}".format(str(num_points)))
        for label, k in material["band_path"]:
            card.append(" {} {} 0.0 {}".format(str(k[0]), str(k[1]), str(material["Nkband"])))

        return "\n".join(card)

def nscf_ks(Nk1, Nk2):
    '''Returns a list of [k1, k2, 0.0] values giving the ks to be included
    in a nscf/Wannier run.
    '''
    ks = []
    for i in range(Nk1):
        for j in range(Nk2):
            ks.append([float(i)/float(Nk1), float(j)/float(Nk2), 0.0])
    return ks
