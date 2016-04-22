import argparse
import os
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.pwscf.parseScf import alat_from_scf, latVecs_from_scf, fermi_from_scf
from tmd.wannier.extractHr import extractHr
from tmd.wannier.plotBands import plotBands
from tmd.bilayer.bilayer_util import global_config

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])

    parser = argparse.ArgumentParser(description="Plot TMD result")
    parser.add_argument('prefix', type=str, help="Calculation prefix")
    args = parser.parse_args()

    qe_bands_path = os.path.join(work, args.prefix, "bands",  "{}_bands.dat".format(args.prefix))
    wannier_dir = os.path.join(work, args.prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))

    nbnd, nks, evalsQE = extractQEBands(qe_bands_path)
    alat = alat_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    E_F = fermi_from_scf(scf_path)

    minE = E_F - 9.0
    maxE = E_F + 6.0

    Hr = extractHr(Hr_path)
    outpath = args.prefix
    plotBands(evalsQE, Hr, alat, latVecs, minE, maxE, outpath)

if __name__ == "__main__":
    _main()
