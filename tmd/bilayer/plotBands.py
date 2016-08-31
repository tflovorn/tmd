import argparse
import os
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.pwscf.parseScf import alat_from_scf, latVecs_from_scf, fermi_from_scf
from tmd.wannier.extractHr import extractHr
from tmd.wannier.plotBands import plotBands
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.plot_ds import get_atom_order, orbital_index

def get_orbital_labels(work, prefix):
    atom_order = get_atom_order(work, prefix)

    syms = ["X1", "M", "X2", "X1p", "Mp", "X2p"]
    orbitals = {"X1": ["pz", "px", "py"], "M": ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]}
    orbitals["X2"] = orbitals["X1"]
    orbitals["X1p"] = orbitals["X1"]
    orbitals["X2p"] = orbitals["X1"]
    orbitals["Mp"] = orbitals["M"]

    spins = ["up", "down"]

    label_comps = {}
    for sym in syms:
        for orb in orbitals[sym]:
            for spin in spins:
                index = orbital_index(atom_order, sym, orb, spin, soc=True)
                label = "{}_{}_{}".format(sym, orb, spin)
                label_comps[index] = label

    label_comp_list = [None]*(len(label_comps.keys()))
    for k, v in label_comps.items():
        label_comp_list[k] = v

    return label_comp_list

def make_plot(work, prefix, plot_evecs, minE, maxE):
    qe_bands_path = os.path.join(work, prefix, "bands",  "{}_bands.dat".format(prefix))
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))

    nbnd, nks, evalsQE = extractQEBands(qe_bands_path)
    alat = alat_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    E_F = fermi_from_scf(scf_path)

    if minE is None:
        minE = E_F - 9.0
    else:
        minE = E_F + minE

    if maxE is None:
        maxE = E_F + 6.0
    else:
        maxE = E_F + maxE

    Hr = extractHr(Hr_path)
    outpath = prefix
    symList = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]
    component_labels = get_orbital_labels(work, prefix)
    plotBands(evalsQE, Hr, alat, latVecs, minE, maxE, outpath, symList=symList,
            fermi_energy=E_F, plot_evecs=plot_evecs, component_labels=component_labels)

def _main():
    parser = argparse.ArgumentParser(description="Plot TMD band structure result",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--global_prefix", type=str, default="MoS2_WS2",
            help="Calculation global prefix")
    parser.add_argument("--prefix", type=str, default=None,
            help="If specified, plot bands for one prefix only")
    parser.add_argument("--plot_evecs", action='store_true',
            help="Plot eigenvector components")
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot (not relative to E_F)")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot (not relative to E_F)")
    args = parser.parse_args()

    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)

    if args.prefix is None:
        prefixes = get_prefixes(work, args.global_prefix)

        for prefix in prefixes:
            make_plot(work, prefix, args.plot_evecs, args.minE, args.maxE)
    else:
        make_plot(work, args.prefix, args.plot_evecs, args.minE, args.maxE)

if __name__ == "__main__":
    _main()
