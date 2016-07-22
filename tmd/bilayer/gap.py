import argparse
import os
from multiprocessing import Pool
import numpy as np
import yaml
from tmd.pwscf.parseScf import fermi_from_scf
from tmd.wannier.bands import Hk_recip
from tmd.bilayer.wannier import get_Hr
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.plot_ds import get_atom_order, orbital_index, ds_from_prefixes, wrap_cell, sorted_d_group, plot_d_vals

def get_layer_indices(work, prefix):
    atom_order = get_atom_order(work, prefix)

    syms = [["X1", "M", "X2"], ["X1p", "Mp", "X2p"]]
    orbitals = {"X1": ["pz", "px", "py"], "M": ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]}
    orbitals["X2"] = orbitals["X1"]
    orbitals["X1p"] = orbitals["X1"]
    orbitals["X2p"] = orbitals["X1"]
    orbitals["Mp"] = orbitals["M"]

    spins = ["up", "down"]

    layer_indices = []
    for layer_syms in syms:
        layer_indices.append([])
        for sym in layer_syms:
            for orb in orbitals[sym]:
                for spin in spins:
                    index = orbital_index(atom_order, sym, orb, spin, soc=True)
                    layer_indices[-1].append(index)

    return layer_indices

def get_layer_contribs(layer_indices, U):
    layer_contribs = [[], []]
    num_states = U.shape[0]
    for n in range(num_states):
        for l, l_indices in enumerate(layer_indices):
            contrib = 0.0
            for index in l_indices:
                contrib += abs(U[index, n])**2

            layer_contribs[l].append(contrib)

    return layer_contribs

def bracket_indices(w, E_F):
    for i, val in enumerate(w):
        if i == len(w) - 1:
            return None

        if val <= E_F and w[i+1] > E_F:
            return i, i+1

def get_gaps(work, prefix, layer_threshold):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    layer_indices = get_layer_indices(work, prefix)

    E_F = fermi_from_scf(scf_path)

    Hr = get_Hr(work, prefix)
    K = (1/3, 1/3, 0.0)
    HK = Hk_recip(K, Hr)

    w, U = np.linalg.eigh(HK)
    #print(E_F, w)
    layer_contribs = get_layer_contribs(layer_indices, U)
    #for contrib, val in zip(layer_contribs[0], w):
    #    print(contrib, val)
    #for contrib, val in zip(layer_contribs[1], w):
    #    print(contrib, val)

    below_fermi, above_fermi = bracket_indices(w, E_F)
    #print(w[below_fermi], w[above_fermi])

    conduction = [None, None]
    valence = [None, None]

    n = below_fermi
    while n >= 0:
        for l in [0, 1]:
            contrib = layer_contribs[l][n]
            if contrib > layer_threshold and valence[l] is None:
                valence[l] = n
        n -= 1

    n = above_fermi
    while n < len(w):
        for l in [0, 1]:
            contrib = layer_contribs[l][n]
            if contrib > layer_threshold and conduction[l] is None:
                conduction[l] = n
        n += 1

    #print(w[valence[0]], w[conduction[0]], w[valence[1]], w[conduction[1]])
    gaps = {}
    gaps["0/0"] = float(w[conduction[0]] - w[valence[0]])
    gaps["1/1"] = float(w[conduction[1]] - w[valence[1]])
    gaps["0/1"] = float(w[conduction[0]] - w[valence[1]])
    gaps["1/0"] = float(w[conduction[1]] - w[valence[0]])

    return gaps

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps")
    parser.add_argument("--threshold", type=float, default=0.9,
            help="Threshold for deciding if a state is dominated by one layer")
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

    layer0_gap_vals, layer1_gap_vals, interlayer_01_gap_vals, interlayer_10_gap_vals = [], [], [], []
    get_gaps_args = []
    for d, prefix in dps:
        get_gaps_args.append([work, prefix, args.threshold])

    with Pool() as p:
        all_gaps = p.starmap(get_gaps, get_gaps_args)

    gap_data = []
    for (d, prefix), gaps in zip(dps, all_gaps):
        gap_data.append([list(d), gaps])

    fp = open("gap_data", 'w')
    fp.write(yaml.dump(gap_data))
    fp.close()

    for d, gaps in gap_data:
        layer0_gap_vals.append(gaps["0/0"])
        layer1_gap_vals.append(gaps["1/1"])
        interlayer_01_gap_vals.append(gaps["0/1"])
        interlayer_10_gap_vals.append(gaps["1/0"])

    plot_d_vals("layer0_gaps", "MoS2 gap", dps, layer0_gap_vals)
    plot_d_vals("layer1_gaps", "WS2 gap", dps, layer1_gap_vals)
    plot_d_vals("interlayer_01_gaps", "MoS2->WS2 gap", dps, interlayer_01_gap_vals)
    plot_d_vals("interlayer_10_gaps", "WS2->MoS2 gap", dps, interlayer_10_gap_vals)

if __name__ == "__main__":
    _main()
