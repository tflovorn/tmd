import argparse
import os
import numpy as np
import yaml
from tmd.pwscf.parseScf import fermi_from_scf
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.plot_ds import ds_from_prefixes, wrap_cell, sorted_d_group, plot_d_vals
from tmd.bilayer.gap import bracket_indices, _close

def get_gap(work, prefix, k_cart):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)

    bands_dir = os.path.join(work, prefix, "bands")
    evals_path = os.path.join(bands_dir, "{}_bands.dat".format(prefix))
    nbnd, nks, bands = extractQEBands(evals_path)

    bands_at_k = None
    eps = 1e-6
    for (k, w) in bands:
        if _close(k, k_cart, eps):
            bands_at_k = w
            break

    if bands_at_k is None:
        raise ValueError("k_cart not found")

    below_fermi, above_fermi = bracket_indices(bands_at_k, E_F)

    return bands_at_k[above_fermi] - bands_at_k[below_fermi]

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps")
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

    K_cart = (2/3, 0.0, 0.0)
    all_gaps = []
    for d, prefix in dps:
        gap = get_gap(work, prefix, K_cart)
        all_gaps.append(gap)

    gap_data = []
    for (d, prefix), gap in zip(dps, all_gaps):
        gap_data.append([list(d), all_gaps])

    fp = open("{}_gap_data".format(args.global_prefix), 'w')
    fp.write(yaml.dump(gap_data))
    fp.close()

    plot_d_vals("{}_gaps".format(args.global_prefix), "Total gap", dps, all_gaps)

if __name__ == "__main__":
    _main()
