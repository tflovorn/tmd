from copy import deepcopy
from tmd.pwscf.build import nscf_ks, wannier_num_bands
import tmd.pwscf.cell as cell

def Header(nbnd, num_wann, soc):
    lines = ["num_bands = {}".format(str(nbnd))]
    lines.append("num_wann = {}".format(str(num_wann)))
    lines.append("num_iter = 0")
    
    # TODO - why is this conditional in SKfit?
    # Ambiguity in defining up/down basis?
    #if not soc:
    #    lines.append("hr_plot = true")

    lines.append("hr_plot = true")
    lines.append("")
    return lines

def Disentanglement():
    # TODO - is it possible to set reasonable defaults for:
    # dis_win_min, dis_win_max, dis_froz_min, dis_froz_max?
    # Leave with Wannier90 defaults for now: no inner window;
    # outer window covers all evs.
    lines = ["#TODO - set disentanglement window."]
    lines.append("dis_win_min = 0.0d0")
    lines.append("dis_win_max = 10.0d0")
    lines.append("dis_froz_min = 0.0d0")
    lines.append("dis_froz_max = 0.0d0")

    lines.append("dis_num_iter = 10000")
    lines.append("dis_mix_ratio = 0.5")
    lines.append("")
    return lines

def Update_Disentanglement(win_path, E_Fermi, outer, inner):
    vals = {"dis_win_min": E_Fermi + outer[0],
            "dis_win_max": E_Fermi + outer[1],
            "dis_froz_min": E_Fermi + inner[0],
            "dis_froz_max": E_Fermi + inner[1]}

    with open(win_path, 'r') as fp:
        lines = fp.readlines()

    updated = []
    for line in lines:
        if line.startswith("#TODO - set disentanglement window."):
            continue

        new_line = line
        for k, v in vals.items():
            if line.startswith(k):
                new_line = "{} = {}\n".format(k, v)
                break

        updated.append(new_line)

    with open(win_path, 'w') as fp:
        fp.write("".join(updated))

def Projections(latpos, soc, valence):
    atoms = []
    for at, pos in latpos:
        atoms.append(at)

    distinct_at = _distinct_atoms(atoms)

    lines = []
    if soc:
        lines.append("spinors = T")

    lines.append("begin projections")
    for at in distinct_at:
        at_proj = _proj_line(at, valence[at])
        lines.append(at_proj)
    lines.append("end projections")
    lines.append("")
    return lines

def _proj_line(at, valence):
    line = "{}: ".format(at)
    for val_index, this_val in enumerate(valence):
        if val_index != 0:
            line += ";"

        if this_val == "s":
            line += "l=0"
        elif this_val == "p":
            line += "l=1"
        elif this_val == "d":
            line += "l=2"
        else:
            raise ValueError("unrecognized valence component")

    return line

def _distinct_atoms(atoms):
    distinct_at = []
    for at_index, at in enumerate(atoms):
        if at not in distinct_at:
            distinct_at.append(at)

    return distinct_at

def Spin(spin_polarized):
    if not spin_polarized:
        return [], None
    else:
        lines_up = ["spin = up", ""]
        lines_down = ["spin = down", ""]
        return lines_up, lines_down

def UnitCell(axes, alat, abohr):
    lines = ["begin unit_cell_cart"]
    lines.append("bohr")

    for i in range(3):
        ix = (alat / abohr) * axes[i][0]
        iy = (alat / abohr) * axes[i][1]
        iz = (alat / abohr) * axes[i][2]
        lines.append("  {}  {}  {}".format(str(ix), str(iy), str(iz)))

    lines.append("end unit_cell_cart")
    lines.append("")
    return lines

def AtomPos(latpos):
    lines = ["begin atoms_frac"]
    for atom, pos in latpos:
        lines.append(" {} {} {} {}".format(atom, str(pos[0]), str(pos[1]), str(pos[2])))

    lines.append("end atoms_frac")
    lines.append("")
    return lines

def Kpoints(Nk1, Nk2):
    lines = ["mp_grid = {} {} 1".format(Nk1, Nk2)]
    lines.append("")
    lines.append("begin kpoints")
    lines.extend(ks_strs(Nk1, Nk2))
    lines.append("end kpoints")
    lines.append("")
    return lines

def ks_strs(Nk1, Nk2):
    ks_lists = nscf_ks(Nk1, Nk2)
    
    ret = []
    for ks in ks_lists:
        ret.append("{} {} {}".format(ks[0], ks[1], ks[2]))

    return ret

def _get_num_wann(maxl):
    num = 0
    for c in maxl:
        if c == "0":
            num += 1
        elif c == "1":
            num += 4
        elif c == "2":
            num += 9
        else:
            raise ValueError("unexpected value in maxl")
    return num

def Winfile(material):
    spin_polarized = False # TODO - handle case with spin polarization on and soc off
    soc = material["soc"]
    if soc and spin_polarized:
        raise ValueError("Cannot specify both spin_polarized and soc.")

    nbnd = wannier_num_bands(material["valence"])
    num_wann = material["valence"]["total"]

    lines = Header(nbnd, num_wann, soc)
    lines.extend(Disentanglement())
    
    axes, latpos = cell.build(material["latconst"], material["latvecs"],
            material["cartpos"], material["vacuum_dist"])
    alat = material["latconst"]
    abohr = 0.52917721

    lines.extend(Projections(latpos, soc, material["valence"]))
    lines.extend(UnitCell(axes, alat, abohr))
    lines.extend(AtomPos(latpos))

    Nk1, Nk2 = material["nscf_Nk1"], material["nscf_Nk2"]
    lines.extend(Kpoints(Nk1, Nk2))

    spin_up, spin_down = Spin(spin_polarized)
    if not spin_polarized:
        lines.extend(spin_up)

        lines_str = "\n".join(lines) + "\n"
        return lines_str, None
    else:
        lines_down = deepcopy(lines)

        lines.extend(spin_up)
        lines_down.extend(spin_down)

        lines_str = "\n".join(lines) + "\n"
        lines_down_str = "\n".join(lines_down) + "\n"
        return lines_str, lines_down_str
