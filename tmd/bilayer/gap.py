import argparse
import os
from multiprocessing import Pool
import numpy as np
import yaml
from tmd.pwscf.parseScf import fermi_from_scf, D_from_scf
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.wannier.bands import Hk_recip
from tmd.wannier.parseWin import parse_inner_window
from tmd.wannier.fitError import dft_wan_correspondence
from tmd.bilayer.wannier import get_Hr
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.plot_ds import get_atom_order, orbital_index, ds_from_prefixes, wrap_cell, sorted_d_group, plot_d_vals

def _close(k, q, eps):
    for i in range(len(k)):
        if abs(k[i] - q[i]) > eps:
            return False

    return True

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

def get_gaps(work, prefix, layer_threshold, use_QE_evs=True, ev_width=8):
    K = (1/3, 1/3, 0.0)

    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)
    if use_QE_evs:
        D = D_from_scf(scf_path)
        R = 2*np.pi*np.linalg.inv(D)
        # ks in QE bands output are in units of 2pi/a;
        # D is in units of a
        K_cart_2pi = np.dot(np.array(K), R) / (2*np.pi)

        bands_dir = os.path.join(work, prefix, "bands")
        evals_path = os.path.join(bands_dir, "{}_bands.dat".format(prefix))
        nbnd, nks, QE_bands = extractQEBands(evals_path, ev_width=ev_width)
        eps = 1e-6
        QE_bands_K = None
        for qe_k_cart, qe_k_evals in QE_bands:
            if _close(K_cart_2pi, qe_k_cart, eps):
                QE_bands_K = qe_k_evals
                break

        if QE_bands_K is None:
            raise ValueError("could not find QE k = K")

        win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
        inner_win = parse_inner_window(win_path)

    layer_indices = get_layer_indices(work, prefix)

    Hr = get_Hr(work, prefix)
    # rotated 2pi/3: K_R2 = (-2/3, 1/3, 0.0)
    # rotated 4pi/3: K_R4 = (1/3, -2/3, 0.0)
    HK = Hk_recip(K, Hr)
    # TODO - check ws for QE bands.
    # Wannier functions may not preserve symmetry.
    # Possible that symmetry is exact in QE bands.

    w, U = np.linalg.eigh(HK)

    if use_QE_evs:
        dft_start_index, wan_start_index, num_states = dft_wan_correspondence(QE_bands_K,
                w, inner_win)
        offset = dft_start_index - wan_start_index

    layer_contribs = get_layer_contribs(layer_indices, U)

    below_fermi, above_fermi = bracket_indices(w, E_F)

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

    gaps = {}
    if use_QE_evs:
        ev = QE_bands_K
        gaps["0/0"] = float(ev[conduction[0]+offset] - ev[valence[0]+offset])
        gaps["1/1"] = float(ev[conduction[1]+offset] - ev[valence[1]+offset])
        gaps["0/1"] = float(ev[conduction[0]+offset] - ev[valence[1]+offset])
        gaps["1/0"] = float(ev[conduction[1]+offset] - ev[valence[0]+offset])
    else:
        gaps["0/0"] = float(w[conduction[0]] - w[valence[0]])
        gaps["1/1"] = float(w[conduction[1]] - w[valence[1]])
        gaps["0/1"] = float(w[conduction[0]] - w[valence[1]])
        gaps["1/0"] = float(w[conduction[1]] - w[valence[0]])

    return gaps

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use_QE_evs", action='store_true',
            help="Use eigenvalues from QE instead of Wannier H(k)")
    parser.add_argument("--ev_width", type=int, default=8,
            help="Number of characters per eigenvalue in QE bands.dat")
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
        get_gaps_args.append([work, prefix, args.threshold, args.use_QE_evs, args.ev_width])

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

    plot_d_vals("layer0_gaps", "MoS2 gap [eV]", dps, layer0_gap_vals)
    plot_d_vals("layer1_gaps", "WS2 gap [eV]", dps, layer1_gap_vals)
    plot_d_vals("interlayer_01_gaps", "MoS2->WS2 gap [eV]", dps, interlayer_01_gap_vals)
    plot_d_vals("interlayer_10_gaps", "WS2->MoS2 gap [eV]", dps, interlayer_10_gap_vals)

if __name__ == "__main__":
    _main()
