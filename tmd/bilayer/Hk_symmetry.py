import os
from tmd.wannier.bands import Hk_recip
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.plot_ds import ds_from_prefixes, sorted_d_group, wrap_cell, get_atom_order, orbital_index
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.wannier import get_Hr

def find_d_val(dps, d_val):
    eps = 2e-3 # chosen based on precision of d in folder name

    for d, prefix in dps:
        if abs(d[0] - d_val[0]) < eps and abs(d[1] - d_val[1]) < eps:
            return prefix

    # Only get here if no matching d is found in dps.
    raise ValueError("d_val not found")

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    
    global_prefix = "MoS2_WS2"
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)

    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    ordering = "2H"

    if ordering == "2H":
        d0_prefix = find_d_val(dps, (2/3, 1/3))
        d1_prefix = find_d_val(dps, (0.0, 0.0))
        d2_prefix = find_d_val(dps, (1/3, 2/3))
    elif ordering == "2H_top":
        d0_prefix = find_d_val(dps, (0.0, 0.0))
        d1_prefix = find_d_val(dps, (1/3, 2/3))
        d2_prefix = find_d_val(dps, (2/3, 1/3))
    else:
        raise ValueError("unrecognized ordering")

    K = (1/3, 1/3, 0.0)

    for i_d, prefix in enumerate((d0_prefix, d1_prefix, d2_prefix)):
        atom_Hr_order = get_atom_order(work, prefix)
        dxy_M_i = orbital_index(atom_Hr_order, "M", "dxy", "up", soc=False) # spin ignored for no soc
        dxy_Mp_i = orbital_index(atom_Hr_order, "Mp", "dxy", "up", soc=False)
        dx2y2_M_i = orbital_index(atom_Hr_order, "M", "dx2-y2", "up", soc=False)
        dx2y2_Mp_i = orbital_index(atom_Hr_order, "Mp", "dx2-y2", "up", soc=False)
        dz2_M_i = orbital_index(atom_Hr_order, "M", "dz2", "up", soc=False)
        dz2_Mp_i = orbital_index(atom_Hr_order, "Mp", "dz2", "up", soc=False)

        Hr = get_Hr(work, prefix)
        HK = Hk_recip(K, Hr)

        dp2_M_dp2_Mp = (HK[dx2y2_M_i, dx2y2_Mp_i] + HK[dxy_M_i, dxy_Mp_i]
                + 1j*HK[dx2y2_M_i, dxy_Mp_i] - 1j*HK[dxy_M_i, dx2y2_Mp_i])/2

        dm2_M_dm2_Mp = (HK[dx2y2_M_i, dx2y2_Mp_i] + HK[dxy_M_i, dxy_Mp_i]
                - 1j*HK[dx2y2_M_i, dxy_Mp_i] + 1j*HK[dxy_M_i, dx2y2_Mp_i])/2

        dp2_M_dm2_Mp = (HK[dx2y2_M_i, dx2y2_Mp_i] - HK[dxy_M_i, dxy_Mp_i]
                - 1j*HK[dx2y2_M_i, dxy_Mp_i] - 1j*HK[dxy_M_i, dx2y2_Mp_i])/2

        dm2_M_dp2_Mp = (HK[dx2y2_M_i, dx2y2_Mp_i] - HK[dxy_M_i, dxy_Mp_i]
                + 1j*HK[dx2y2_M_i, dxy_Mp_i] + 1j*HK[dxy_M_i, dx2y2_Mp_i])/2

        dz2_M_dz2_Mp = HK[dz2_M_i, dz2_Mp_i]

        print("i_d = {}, prefix = {}".format(str(i_d), prefix))
        print("<d_+2^M|H(K)|d_+2^M'> = {}".format(str(dp2_M_dp2_Mp)))
        print("<d_-2^M|H(K)|d_-2^M'> = {}".format(str(dm2_M_dm2_Mp)))
        print("<d_+2^M|H(K)|d_-2^M'> = {}".format(str(dp2_M_dm2_Mp)))
        print("<d_-2^M|H(K)|d_+2^M'> = {}".format(str(dm2_M_dp2_Mp)))
        print("<d_z2^M|H(K)|d_z2^M'> = {}".format(str(dz2_M_dz2_Mp)))

if __name__ == "__main__":
    _main()
