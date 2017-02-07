from __future__ import division
import argparse
import os
import json
from multiprocessing import Pool
import numpy as np
import yaml
import numdifftools
from tmd.pwscf.parseScf import fermi_from_scf, D_from_scf, alat_from_scf
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.wannier.bands import Hk_recip
from tmd.wannier.bands import Hk as Hk_Cart
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

def get_layer_indices(work, prefix, fixed_spin):
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
                    if spin != fixed_spin:
                        continue

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

def select_layer_contrib(layer_contribs_up, layer_contribs_down, spin, l, n):
    contrib_up = layer_contribs_up[l][n]
    contrib_down = layer_contribs_down[l][n]

    if spin is None:
        contrib = contrib_up + contrib_down
    elif spin == 'up':
        contrib = contrib_up
    elif spin == 'down':
        contrib = contrib_down
    else:
        raise ValueError("unrecognized spin value")

    return contrib

def get_curvature(D, Hr, k, n):
    '''Calculate d^2 E / d k^2 along kx and ky directions at band n.
    Assumes there are no band crossings in the region sampled, so that
    the single index n can be used for all sampled ks.
    '''
    curvature = []
    for d in range(2):
        def Er_d(kd):
            kr = []
            for dp in range(3):
                if dp == d:
                    kr.append(kd)
                else:
                    kr.append(k[dp])

            H_kr = Hk_Cart(kr, Hr, D.T)
            w, U = np.linalg.eigh(H_kr)
            Er = w[n]
            return Er

        fd = numdifftools.Derivative(Er_d, n=2)
        curvature_d = fd(k[d])

        curvature.append(curvature_d)

    return curvature

def get_gaps(work, prefix, layer_threshold, k, spin_valence=None, spin_conduction=None, use_QE_evs=False, ev_width=8, do_get_curvature=False):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)
    alat_Bohr = alat_from_scf(scf_path)
    D = D_from_scf(scf_path)
    R = 2*np.pi*np.linalg.inv(D)

    if use_QE_evs:
        # ks in QE bands output are in units of 2pi/a;
        # D is in units of a
        k_cart_2pi = np.dot(np.array(k), R) / (2*np.pi)

        bands_dir = os.path.join(work, prefix, "bands")
        evals_path = os.path.join(bands_dir, "{}_bands.dat".format(prefix))
        nbnd, nks, QE_bands = extractQEBands(evals_path, ev_width=ev_width)
        eps = 1e-6
        QE_bands_k = None
        for qe_k_cart, qe_k_evals in QE_bands:
            if _close(k_cart_2pi, qe_k_cart, eps):
                QE_bands_k = qe_k_evals
                break

        if QE_bands_k is None:
            raise ValueError("could not find QE k")

        win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
        inner_win = parse_inner_window(win_path)

    layer_indices_up = get_layer_indices(work, prefix, 'up')
    layer_indices_down = get_layer_indices(work, prefix, 'down')

    Hr = get_Hr(work, prefix)
    # note:
    # rotated K 2pi/3: K_R2 = (-2/3, 1/3, 0.0)
    # rotated K 4pi/3: K_R4 = (1/3, -2/3, 0.0)
    Hk = Hk_recip(k, Hr)
    # TODO - check ws for QE bands.
    # Wannier functions may not preserve symmetry.
    # Possible that symmetry is exact in QE bands.

    w, U = np.linalg.eigh(Hk)

    if use_QE_evs:
        dft_start_index, wan_start_index, num_states = dft_wan_correspondence(QE_bands_k,
                w, inner_win)
        offset = dft_start_index - wan_start_index

    layer_contribs_up = get_layer_contribs(layer_indices_up, U)
    layer_contribs_down = get_layer_contribs(layer_indices_down, U)

    below_fermi, above_fermi = bracket_indices(w, E_F)

    conduction = [None, None]
    valence = [None, None]
    conduction_curvature = [None, None]
    valence_curvature = [None, None]

    n = below_fermi
    while n >= 0:
        for l in [0, 1]:
            contrib = select_layer_contrib(layer_contribs_up, layer_contribs_down, spin_valence, l, n)

            if contrib > layer_threshold and valence[l] is None:
                valence[l] = n
                if do_get_curvature:
                    k_Cart = np.dot(np.array(k), R)
                    valence_curvature[l] = get_curvature(D, Hr, k_Cart, n)

        n -= 1

    n = above_fermi
    while n < len(w):
        for l in [0, 1]:
            contrib = select_layer_contrib(layer_contribs_up, layer_contribs_down, spin_conduction, l, n)

            if contrib > layer_threshold and conduction[l] is None:
                conduction[l] = n
                if do_get_curvature:
                    k_Cart = np.dot(np.array(k), R)
                    conduction_curvature[l] = get_curvature(D, Hr, k_Cart, n)

        n += 1

    gaps = {}
    if use_QE_evs:
        ev = QE_bands_k
        gaps["0/0"] = float(ev[conduction[0]+offset] - ev[valence[0]+offset])
        gaps["1/1"] = float(ev[conduction[1]+offset] - ev[valence[1]+offset])
        gaps["0/1"] = float(ev[conduction[0]+offset] - ev[valence[1]+offset])
        gaps["1/0"] = float(ev[conduction[1]+offset] - ev[valence[0]+offset])
        gaps["0_valence"] = float(ev[valence[0]+offset])
        gaps["1_valence"] = float(ev[valence[1]+offset])
        gaps["0_conduction"] = float(ev[conduction[0]+offset])
        gaps["1_conduction"] = float(ev[conduction[1]+offset])
        if do_get_curvature:
            add_curvature(gaps, valence_curvature, conduction_curvature, alat_Bohr)
    else:
        gaps["0/0"] = float(w[conduction[0]] - w[valence[0]])
        gaps["1/1"] = float(w[conduction[1]] - w[valence[1]])
        gaps["0/1"] = float(w[conduction[0]] - w[valence[1]])
        gaps["1/0"] = float(w[conduction[1]] - w[valence[0]])
        gaps["0_valence"] = float(w[valence[0]])
        gaps["1_valence"] = float(w[valence[1]])
        gaps["0_conduction"] = float(w[conduction[0]])
        gaps["1_conduction"] = float(w[conduction[1]])
        if do_get_curvature:
            add_curvature(gaps, valence_curvature, conduction_curvature, alat_Bohr)

    return gaps

def add_curvature(gaps, valence_curvature, conduction_curvature, alat_Bohr):
    hbar_eV_s = 6.582119514e-16
    me_eV_per_c2 = 0.5109989461e6
    c_m_per_s = 2.99792458e8
    Bohr_m = 0.52917721067e-10
    fac = hbar_eV_s**2 / (me_eV_per_c2 * (c_m_per_s)**(-2) * (Bohr_m)**2 * alat_Bohr**2)

    gaps["0_valence_effmass_kx"] = float(-fac/valence_curvature[0][0])
    gaps["1_valence_effmass_kx"] = float(-fac/valence_curvature[1][0])
    gaps["0_valence_effmass_ky"] = float(-fac/valence_curvature[0][1])
    gaps["1_valence_effmass_ky"] = float(-fac/valence_curvature[1][1])
    gaps["0_conduction_effmass_kx"] = float(fac/conduction_curvature[0][0])
    gaps["1_conduction_effmass_kx"] = float(fac/conduction_curvature[1][0])
    gaps["0_conduction_effmass_ky"] = float(fac/conduction_curvature[0][1])
    gaps["1_conduction_effmass_ky"] = float(fac/conduction_curvature[1][1])

def write_gap_data(work, dps, threshold, spin_valence, spin_conduction, use_QE_evs, ev_width, k, gap_label, gap_label_tex, do_get_curvature):
    get_gaps_args = []
    for d, prefix in dps:
        get_gaps_args.append([work, prefix, threshold, k, spin_valence, spin_conduction, use_QE_evs, ev_width, do_get_curvature])

    with Pool() as p:
        all_gaps = p.starmap(get_gaps, get_gaps_args)

    gap_data = []
    # For JSON output, use same format as plot_ds.
    json_gap_data = {"_ds": []}

    for (d, prefix), gaps in zip(dps, all_gaps):
        gap_data.append([list(d), gaps])

        json_gap_data["_ds"].append(d)
        for k, v in gaps.items():
            if k not in json_gap_data:
                json_gap_data[k] = []

            json_gap_data[k].append(v)

    with open("{}_gap_data.yaml".format(gap_label), 'w') as fp:
        fp.write(yaml.dump(gap_data))

    with open("{}_gap_data.json".format(gap_label), 'w') as fp:
        json.dump(json_gap_data, fp)

    layer0_gap_vals, layer1_gap_vals, interlayer_01_gap_vals, interlayer_10_gap_vals = [], [], [], []
    layer0_valence, layer1_valence, layer0_conduction, layer1_conduction = [], [], [], []
    layer0_valence_effmass_kx, layer1_valence_effmass_kx, layer0_valence_effmass_ky, layer1_valence_effmass_ky = [], [], [], []
    layer0_conduction_effmass_kx, layer1_conduction_effmass_kx, layer0_conduction_effmass_ky, layer1_conduction_effmass_ky = [], [], [], []
    for d, gaps in gap_data:
        layer0_gap_vals.append(gaps["0/0"])
        layer1_gap_vals.append(gaps["1/1"])
        interlayer_01_gap_vals.append(gaps["0/1"])
        interlayer_10_gap_vals.append(gaps["1/0"])
        layer0_valence.append(gaps["0_valence"])
        layer1_valence.append(gaps["1_valence"])
        layer0_conduction.append(gaps["0_conduction"])
        layer1_conduction.append(gaps["1_conduction"])

    plot_d_vals("{}_layer0_gaps".format(gap_label), "{} MoS$_2$ gap [eV]".format(gap_label_tex), dps, layer0_gap_vals)
    plot_d_vals("{}_layer1_gaps".format(gap_label), "{} WS$_2$ gap [eV]".format(gap_label_tex), dps, layer1_gap_vals)
    plot_d_vals("{}_interlayer_01_gaps".format(gap_label), "{} MoS$_2$ - WS$_2$ gap [eV]".format(gap_label_tex), dps, interlayer_01_gap_vals)
    plot_d_vals("{}_interlayer_10_gaps".format(gap_label), "{} WS$_2$ - MoS$_2$ gap [eV]".format(gap_label_tex), dps, interlayer_10_gap_vals)
    plot_d_vals("{}_layer0_valence".format(gap_label), "{} MoS$_2$ valence maximum [eV]".format(gap_label_tex), dps, layer0_valence)
    plot_d_vals("{}_layer1_valence".format(gap_label), "{} WS$_2$ valence maximum [eV]".format(gap_label_tex), dps, layer1_valence)
    plot_d_vals("{}_layer0_conduction".format(gap_label), "{} MoS$_2$ conduction minimum [eV]".format(gap_label_tex), dps, layer0_conduction)
    plot_d_vals("{}_layer1_conduction".format(gap_label), "{} WS$_2$ conduction minimum [eV]".format(gap_label_tex), dps, layer1_conduction)

    if do_get_curvature:
        for d, gaps in gap_data:
            layer0_valence_effmass_kx.append(gaps["0_valence_effmass_kx"])
            layer1_valence_effmass_kx.append(gaps["1_valence_effmass_kx"])
            layer0_valence_effmass_ky.append(gaps["0_valence_effmass_ky"])
            layer1_valence_effmass_ky.append(gaps["1_valence_effmass_ky"])
            layer0_conduction_effmass_kx.append(gaps["0_conduction_effmass_kx"])
            layer1_conduction_effmass_kx.append(gaps["1_conduction_effmass_kx"])
            layer0_conduction_effmass_ky.append(gaps["0_conduction_effmass_ky"])
            layer1_conduction_effmass_ky.append(gaps["1_conduction_effmass_ky"])

        plot_d_vals("{}_layer0_valence_effmass_kx".format(gap_label), "{} MoS$_2$ valence $m^*_x/m_e$".format(gap_label_tex), dps, layer0_valence_effmass_kx)
        plot_d_vals("{}_layer1_valence_effmass_kx".format(gap_label), "{} WS$_2$ valence $m^*_x/m_e$".format(gap_label_tex), dps, layer1_valence_effmass_kx)
        plot_d_vals("{}_layer0_valence_effmass_ky".format(gap_label), "{} MoS$_2$ valence $m^*_y/m_e$".format(gap_label_tex), dps, layer0_valence_effmass_ky)
        plot_d_vals("{}_layer1_valence_effmass_ky".format(gap_label), "{} WS$_2$ valence $m^*_y/m_e$".format(gap_label_tex), dps, layer1_valence_effmass_ky)
        plot_d_vals("{}_layer0_conduction_effmass_kx".format(gap_label), "{} MoS$_2$ conduction $m^*_x/m_e$".format(gap_label_tex), dps, layer0_conduction_effmass_kx)
        plot_d_vals("{}_layer1_conduction_effmass_kx".format(gap_label), "{} WS$_2$ conduction $m^*_x/m_e$".format(gap_label_tex), dps, layer1_conduction_effmass_kx)
        plot_d_vals("{}_layer0_conduction_effmass_ky".format(gap_label), "{} MoS$_2$ conduction $m^*_y/m_e$".format(gap_label_tex), dps, layer0_conduction_effmass_ky)
        plot_d_vals("{}_layer1_conduction_effmass_ky".format(gap_label), "{} WS$_2$ conduction $m^*_y/m_e$".format(gap_label_tex), dps, layer1_conduction_effmass_ky)

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--threshold", type=float, default=0.9,
            help="Threshold for deciding if a state is dominated by one layer")
    parser.add_argument("--spin_valence", type=str, default=None,
            help="Set 'up' or 'down' to choose valence band spin type; closest to E_F is used if not set")
    parser.add_argument("--spin_conduction", type=str, default=None,
            help="Set 'up' or 'down' to choose conduction band spin type; closest to E_F is used if not set")
    parser.add_argument("--use_QE_evs", action='store_true',
            help="Use eigenvalues from QE instead of Wannier H(k); if set, spin_valence and spin_conduction act as if not specified.")
    parser.add_argument("--ev_width", type=int, default=8,
            help="Number of characters per eigenvalue in QE bands.dat")
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

    K = (1/3, 1/3, 0.0)
    Gamma = (0.0, 0.0, 0.0)
    do_get_curvature_K, do_get_curvature_Gamma = True, False

    write_gap_data(work, dps, args.threshold, args.spin_valence, args.spin_conduction, args.use_QE_evs, args.ev_width, K, "K", "$K$", do_get_curvature_K)
    write_gap_data(work, dps, args.threshold, args.spin_valence, args.spin_conduction, args.use_QE_evs, args.ev_width, Gamma, "Gamma", "$\\Gamma$", do_get_curvature_Gamma)

if __name__ == "__main__":
    _main()
