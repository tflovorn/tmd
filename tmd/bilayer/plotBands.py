import argparse
import os
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.pwscf.parseScf import alat_from_scf, latVecs_from_scf, fermi_from_scf
from tmd.wannier.extractHr import extractHr
from tmd.wannier.plotBands import plotBands
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes

def _main():
    parser = argparse.ArgumentParser(description="Plot TMD band structure result")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--global_prefix", type=str, default="MoS2_WS2",
            help="Calculation global prefix")
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot (not relative to E_F)")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot (not relative to E_F)")
    args = parser.parse_args()

    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)

    prefixes = get_prefixes(work, args.global_prefix)

    for prefix in prefixes:
        qe_bands_path = os.path.join(work, prefix, "bands",  "{}_bands.dat".format(prefix))
        wannier_dir = os.path.join(work, prefix, "wannier")
        scf_path = os.path.join(wannier_dir, "scf.out")
        Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))

        nbnd, nks, evalsQE = extractQEBands(qe_bands_path)
        alat = alat_from_scf(scf_path)
        latVecs = latVecs_from_scf(scf_path)
        E_F = fermi_from_scf(scf_path)

        if args.minE is None:
            minE = E_F - 9.0
        else:
            minE = args.minE

        if args.maxE is None:
            maxE = E_F + 6.0
        else:
            maxE = args.maxE

        Hr = extractHr(Hr_path)
        outpath = prefix
        symList = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]
        plotBands(evalsQE, Hr, alat, latVecs, minE, maxE, outpath, symList=symList)

if __name__ == "__main__":
    _main()
