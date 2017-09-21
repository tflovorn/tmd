from __future__ import division
import argparse
import os
import json
from multiprocessing import Pool
import numpy as np
import yaml
from tmd.pwscf.parseScf import fermi_from_scf, latVecs_from_scf
from tmd.wannier.bands import Hk as Hk_Cart
from tmd.wannier.bands import dHk_dk
from tmd.bilayer.wannier import get_Hr
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.plot_ds import ds_from_prefixes, wrap_cell, sorted_d_group, plot_d_vals

_Angstrom_per_bohr = 0.5291772

def bottom_conduction_index(E_F, Es):
    '''Return the index of the bottom conduction band.
    '''
    first_above_E_F = None
    for i, E in enumerate(Es):
        if E > E_F:
            first_above_E_F = i
            break

    if first_above_E_F is None:
        raise ValueError("top of valence band not found")

    return first_above_E_F

def get_optical_data(work, prefix):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)
    latVecs = _Angstrom_per_bohr * latVecs_from_scf(scf_path)
    D = latVecs.T
    R = 2.0 * np.pi * np.linalg.inv(D)

    K_lat = np.array([1/3, 1/3, 0.0])
    K_Cart = np.dot(K_lat, R)

    Hr = get_Hr(work, prefix)
    Hk = Hk_Cart(K_Cart, Hr, latVecs)

    Es, U = np.linalg.eigh(Hk)

    cond_0 = bottom_conduction_index(E_F, Es)
    cond_1 = cond_0 + 1
    valence_0 = cond_0 - 1

    cond_0_state, cond_1_state, valence_0_state = [U[:, m] for m in [cond_0, cond_1, valence_0]]

    H_deriv = dHk_dk(K_Cart, Hr, latVecs)

    opt = {"v0_c0_re": [], "v0_c0_im": [], "v0_c1_re": [], "v0_c1_im": [], "v0_c0_norm": [], "v0_c1_norm": []}
    for c in range(2):
        v0_c0_c = np.dot(valence_0_state.conjugate().T, np.dot(H_deriv[c], cond_0_state))[0, 0]
        v0_c1_c = np.dot(valence_0_state.conjugate().T, np.dot(H_deriv[c], cond_1_state))[0, 0]

        opt["v0_c0_re"].append(v0_c0_c.real)
        opt["v0_c0_im"].append(v0_c0_c.imag)
        opt["v0_c0_norm"].append(abs(v0_c0_c))

        opt["v0_c1_re"].append(v0_c1_c.real)
        opt["v0_c1_im"].append(v0_c1_c.imag)
        opt["v0_c1_norm"].append(abs(v0_c1_c))

    return opt

def write_optical_data(work, dps):
    get_optical_data_args = [[work, prefix] for _, prefix in dps]

    with Pool() as p:
        all_optical_data = p.starmap(get_optical_data, get_optical_data_args)

    json_optical_data = {"_ds": []}

    for (d, _), opt in zip(dps, all_optical_data):
        json_optical_data["_ds"].append(d)
        for k, v in opt.items():
            if k not in json_optical_data:
                json_optical_data[k] = []

            json_optical_data[k].append(v)

    with open("K_optical_data.json", 'w') as fp:
        json.dump(json_optical_data, fp)

    # Highest valence band -> lowest conduction band; (x, y) components.
    opt_v0_c0_re = [[opt["v0_c0_re"][c] for opt in all_optical_data] for c in range(2)]
    opt_v0_c0_im = [[opt["v0_c0_im"][c] for opt in all_optical_data] for c in range(2)]
    opt_v0_c0_norm = [[opt["v0_c0_norm"][c] for opt in all_optical_data] for c in range(2)]
    # Highest valence band -> second-lowest conduction band; (x, y) components.
    opt_v0_c1_re = [[opt["v0_c1_re"][c] for opt in all_optical_data] for c in range(2)]
    opt_v0_c1_im = [[opt["v0_c1_im"][c] for opt in all_optical_data] for c in range(2)]
    opt_v0_c1_norm = [[opt["v0_c1_norm"][c] for opt in all_optical_data] for c in range(2)]

    plot_d_vals("K_v0_c0_x_re", "Re <K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, bottom conduction> [eV A]", dps, opt_v0_c0_re[0])
    plot_d_vals("K_v0_c0_y_re", "Re <K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, bottom conduction> [eV A]", dps, opt_v0_c0_re[1])
    plot_d_vals("K_v0_c1_x_re", "Re <K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, second conduction> [eV A]", dps, opt_v0_c1_re[0])
    plot_d_vals("K_v0_c1_y_re", "Re <K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, second conduction> [eV A]", dps, opt_v0_c1_re[1])

    plot_d_vals("K_v0_c0_x_im", "Im <K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, bottom conduction> [eV A]", dps, opt_v0_c0_im[0])
    plot_d_vals("K_v0_c0_y_im", "Im <K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, bottom conduction> [eV A]", dps, opt_v0_c0_im[1])
    plot_d_vals("K_v0_c1_x_im", "Im <K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, second conduction> [eV A]", dps, opt_v0_c1_im[0])
    plot_d_vals("K_v0_c1_y_im", "Im <K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, second conduction> [eV A]", dps, opt_v0_c1_im[1])

    plot_d_vals("K_v0_c0_x_norm", "|<K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, bottom conduction>| [eV A]", dps, opt_v0_c0_norm[0])
    plot_d_vals("K_v0_c0_y_norm", "|<K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, bottom conduction>| [eV A]", dps, opt_v0_c0_norm[1])
    plot_d_vals("K_v0_c1_x_norm", "|<K, top valence| $\\left.\\frac{dH}{dk_x}\\right|_K$ |K, second conduction>| [eV A]", dps, opt_v0_c1_norm[0])
    plot_d_vals("K_v0_c1_y_norm", "|<K, top valence| $\\left.\\frac{dH}{dk_y}\\right|_K$ |K, second conduction>| [eV A]", dps, opt_v0_c1_norm[1])

def _main():
    parser = argparse.ArgumentParser("Calculate optical matrix elements",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument('global_prefix', type=str,
            help="Calculation name")
    args = parser.parse_args()

    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)

    prefixes = get_prefixes(work, args.global_prefix)
    ds = ds_from_prefixes(prefixes)
    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    write_optical_data(work, dps)

if __name__ == "__main__":
    _main()
