import os
import argparse
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

def get_H_orbital_vals(H, work, prefix):
    atom_Hr_order = get_atom_order(work, prefix)
    dxy_M_i = orbital_index(atom_Hr_order, "M", "dxy", "up", soc=False) # spin ignored for no soc
    dxy_Mp_i = orbital_index(atom_Hr_order, "Mp", "dxy", "up", soc=False)
    dx2y2_M_i = orbital_index(atom_Hr_order, "M", "dx2-y2", "up", soc=False)
    dx2y2_Mp_i = orbital_index(atom_Hr_order, "Mp", "dx2-y2", "up", soc=False)
    dz2_M_i = orbital_index(atom_Hr_order, "M", "dz2", "up", soc=False)
    dz2_Mp_i = orbital_index(atom_Hr_order, "Mp", "dz2", "up", soc=False)

    vals = {}
    vals["dp2_M_dp2_Mp"] = (H[dx2y2_M_i, dx2y2_Mp_i] + H[dxy_M_i, dxy_Mp_i]
            + 1j*H[dx2y2_M_i, dxy_Mp_i] - 1j*H[dxy_M_i, dx2y2_Mp_i])/2

    vals["dm2_M_dm2_Mp"] = (H[dx2y2_M_i, dx2y2_Mp_i] + H[dxy_M_i, dxy_Mp_i]
            - 1j*H[dx2y2_M_i, dxy_Mp_i] + 1j*H[dxy_M_i, dx2y2_Mp_i])/2

    vals["dp2_M_dm2_Mp"] = (H[dx2y2_M_i, dx2y2_Mp_i] - H[dxy_M_i, dxy_Mp_i]
            - 1j*H[dx2y2_M_i, dxy_Mp_i] - 1j*H[dxy_M_i, dx2y2_Mp_i])/2

    vals["dm2_M_dp2_Mp"] = (H[dx2y2_M_i, dx2y2_Mp_i] - H[dxy_M_i, dxy_Mp_i]
            + 1j*H[dx2y2_M_i, dxy_Mp_i] + 1j*H[dxy_M_i, dx2y2_Mp_i])/2

    vals["dz2_M_dz2_Mp"] = H[dz2_M_i, dz2_Mp_i]

    return vals

def _main():
    parser = argparse.ArgumentParser("Analysis of H(k) symmetry")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--global_prefix", type=str, default="MoS2_WS2",
            help="Prefix for calculation")
    args = parser.parse_args()

    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)
    
    prefixes = get_prefixes(work, args.global_prefix)
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

    Ks = (("K0", (1/3, 1/3, 0.0)), ("K1", (-2/3, 1/3, 0.0)), ("K2", (1/3, -2/3, 0.0)))

    for i_d, prefix in enumerate((d0_prefix, d1_prefix, d2_prefix)):
        Hr = get_Hr(work, prefix)

        # H(R = 0)
        H0 = Hr[(0, 0, 0)][0]/Hr[(0, 0, 0)][1]
        H0_vals = get_H_orbital_vals(H0, work, prefix)

        # H(K)
        Hk_vals = {}
        for K_label, K in Ks:
            HK = Hk_recip(K, Hr)
            Hk_vals[K_label] = get_H_orbital_vals(HK, work, prefix)

        print("i_d = {}, prefix = {}".format(str(i_d), prefix))
        print("H(r = 0, 0, 0)[+,+] = {}".format(str(H0_vals["dp2_M_dp2_Mp"])))
        print("H(r = 0, 0, 0)[-,-] = {}".format(str(H0_vals["dm2_M_dm2_Mp"])))
        print("H(r = 0, 0, 0)[+,-] = {}".format(str(H0_vals["dp2_M_dm2_Mp"])))
        print("H(r = 0, 0, 0)[-,+] = {}".format(str(H0_vals["dm2_M_dp2_Mp"])))
        print("H(r = 0, 0, 0)[z2,z2] = {}".format(str(H0_vals["dz2_M_dz2_Mp"])))

        for K_label, K in Ks:
            print("------------")
            print("<d_+2^M|H({})|d_+2^M'> = {}".format(K_label, str(Hk_vals[K_label]["dp2_M_dp2_Mp"])))
            print("<d_-2^M|H({})|d_-2^M'> = {}".format(K_label, str(Hk_vals[K_label]["dm2_M_dm2_Mp"])))
            print("<d_+2^M|H({})|d_-2^M'> = {}".format(K_label, str(Hk_vals[K_label]["dp2_M_dm2_Mp"])))
            print("<d_-2^M|H({})|d_+2^M'> = {}".format(K_label, str(Hk_vals[K_label]["dm2_M_dp2_Mp"])))
            print("<d_z2^M|H({})|d_z2^M'> = {}".format(K_label, str(Hk_vals[K_label]["dz2_M_dz2_Mp"])))

        print("===================")

if __name__ == "__main__":
    _main()
