import argparse
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from tmd.wannier.bands import Hk, Hk_recip
from tmd.wannier.extractHr import extractHr
from tmd.pwscf.extractQEBands import extractQEBands
from tmd.pwscf.parseScf import alat_from_scf, latVecs_from_scf

# Switch to size eigenvalue intensity markers based on component values
# (instead of only using color to denote component value).
# Avoids problem of overlapping markers.
eval_dot_size = True

def plotBands(evalsDFT, Hr, alat, latVecs, minE, maxE, outpath, show=False, symList=None, fermi_energy=None, plot_evecs=False, plot_DFT_evals=True):
    '''Create a plot of the eigenvalues given by evalsDFT (which has the form
    returned by extractQEBands()). Additionally, plot the eigenvalues of the
    system described by the Wannier Hamiltonian Hr (which has the form
    returned by extractHr()), assuming periodicity in all directions.

    alat is the lattice vector units used by QE.
    latVecs contains a list of lattice vectors.
    If Hr == None, alat and latVecs are not used.

    The range of energies plotted is restricted to fall in [minE, maxE].
    If minE and/or maxE are None, the range of energies covers all bands.

    k-points used to plot the eigenvalues of Hr are linearly interpolated
    between the k-points listed in evalsDFT.
    '''
    # Get list of all k-points in evalsDFT.
    # While we're iterating through evalsDFT, also construct eigenvalue
    # sequences for plotting. Instead of a list of eigenvalues for each
    # k-point, we need to plot a sequence of lists ranging over all k-points,
    # where each list has one eigenvalue for every k-point.
    DFT_ks = []
    DFT_ys = []
    for k, evs in evalsDFT:
        DFT_ks.append(k)
        # Set up DFT_ys to be a length-len(evs) list of empty lists.
        if len(DFT_ys) == 0:
            for i in range(len(evs)):
                DFT_ys.append([])
        # Add each of this k-point's eigenvalues to the corresponding list.
        for i, ev in enumerate(evs):
            DFT_ys[i].append(ev)
    # Construct list of k-points to evaluate Wannier Hamiltonian at by
    # interpolating between k-points in evalsDFT.
    Hr_ks, Hr_xs, Hr_ys, DFT_xs, Hr_evecs = None, None, None, None, None
    Hr_ks_per_DFT_k = 1
    if Hr is not None:
        Hr_ks_per_DFT_k = 10
        Hr_ks = _interpolateKs(DFT_ks, Hr_ks_per_DFT_k)
        Hr_xs = range(len(Hr_ks))
        if not plot_evecs:
            Hr_ys = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False)[0]
        else:
            Hr_ys, Hr_evecs = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs)
        DFT_xs = range(0, len(Hr_ks), Hr_ks_per_DFT_k)
    else:
        DFT_xs = range(0, len(DFT_ks))

    Hr_ys_eval = None
    if plot_evecs and eval_dot_size:
        # Also need Hr_ys formatted for plot() in this case.
        Hr_ys_eval = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False)[0]

    # Make plot.
    if not plot_evecs:
        if Hr is not None:
            for Hr_evs in Hr_ys:
                plt.plot(Hr_xs, Hr_evs, 'r')

        if plot_DFT_evals:
            for DFT_evs in DFT_ys:
                if show:
                    plt.plot(DFT_xs, DFT_evs, 'ko')
                else:
                    plt.plot(DFT_xs, DFT_evs, 'ko', markersize=2)

        _set_fermi_energy_line(fermi_energy)
        _set_sympoints_ticks(symList, DFT_ks, Hr, Hr_ks_per_DFT_k)
        _set_plot_boundaries(DFT_xs, minE, maxE)
        _save_plot(show, outpath)
    else:
        # Eigenvectors are columns of each entry in Hr_evecs.
        # --> The number of eigenvector components = the number of rows.
        # Make plot with eigenvector weight = |eigenvector component|^2.
        evec_components = Hr_evecs[0].shape[0]
        for comp in range(evec_components):
            plt_xs, plt_ys, plt_cs = [], [], []
            for x in Hr_xs:
                ys = Hr_ys[x]
                for eval_index, y in enumerate(ys):
                    comp_val = abs(Hr_evecs[x][comp, eval_index])**2
                    plt_xs.append(x)
                    plt_ys.append(y)
                    plt_cs.append(comp_val)
            # Plotting eigenvalues colored by eigenvector weight.
            if not eval_dot_size:
                plt.scatter(plt_xs, plt_ys, c=plt_cs, cmap='gnuplot', s=5, edgecolors="none")
            # Or: plotting eigenvalues sized by eigenvector weight.
            else:
                for Hr_evs in Hr_ys_eval:
                    plt.plot(Hr_xs, Hr_evs, 'k', linewidth=0.5)
                s_weights = []
                scale = 10.0
                for val in plt_cs:
                    s_weights.append(scale*val)
                plt.scatter(plt_xs, plt_ys, c=plt_cs, cmap='gnuplot', s=s_weights, facecolors="none")
            plt.colorbar()

            _set_fermi_energy_line(fermi_energy)
            _set_sympoints_ticks(symList, DFT_ks, Hr, Hr_ks_per_DFT_k)
            _set_plot_boundaries(DFT_xs, minE, maxE)
            _save_plot(show, outpath + "_{}".format(str(comp)))

def _set_fermi_energy_line(fermi_energy):
    # Line to show Fermi energy.
    if fermi_energy != None:
        plt.axhline(fermi_energy, color='k')

def _set_sympoints_ticks(symList, DFT_ks, Hr, Hr_ks_per_DFT_k):
    # Lines and labels for symmetry points.
    if symList is not None:
        nk_per_sym = (len(DFT_ks) - 1) / (len(symList) - 1)
        sym_xs = None
        if Hr != None:
            sym_xs = [i*nk_per_sym*Hr_ks_per_DFT_k for i in range(len(symList)+1)]
        else:
            sym_xs = [i*nk_per_sym for i in range(len(symList))]
        for x in sym_xs:
            plt.axvline(x, color='k')
        plt.xticks(sym_xs, symList)

def _set_plot_boundaries(DFT_xs, minE, maxE):
    plt.xlim(0, DFT_xs[-1])
    plt.ylabel("$E$ [eV]")
    if minE != None and maxE != None:
        plt.ylim(minE, maxE)

def _save_plot(show, outpath):
    if show:
        plt.show()
    else:
        plt.savefig(outpath + '.png', bbox_inches='tight', dpi=500)
    plt.clf()

def _remove_duplicate_k_pairs(evalsDFT):
    result = []
    eps = 1e-9
    for k_index in range(len(evalsDFT)):
        if k_index == 0:
            result.append(evalsDFT[k_index])
        else:
            k = evalsDFT[k_index][0]
            prev_k = evalsDFT[k_index-1][0]
            if not _vec_equal_upto(k, prev_k, eps):
                result.append(evalsDFT[k_index])
    return result

def _vec_equal_upto(u, v, eps):
    if len(u) != len(v):
        return False
    for i in range(len(u)):
        if abs(u[i] - v[i]) > eps:
            return False
    return True

def plotDFTBands(dft_bands_filepath, outpath, minE=None, maxE=None, show=False, spin=None):
    nb, nks, qe_bands = extractQEBands(dft_bands_filepath)
    plotBands(qe_bands, None, None, None, minE, maxE, outpath, show)

def _interpolateKs(klist, fineness):
    '''Return a list of k-points which linearly interpolates between the
    values given in klist, with the number of points in the returned list
    equal to (fineness) * (the number of points in klist).
    '''
    interpolated = []
    for ik, k in enumerate(klist[:-1]):
        step = np.subtract(klist[ik+1], k) / float(fineness)
        for j in range(fineness):
            interpolated.append(k + j * step)
    interpolated.append(klist[-1])
    return interpolated

def _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False):
    '''Iterate through Hr_ks and return a sequence of lists ranging
    over all k-points, where each list has one eigenvalue for every k-point.

    Assume QE format: Hr_ks is in Cartesian basis in units
    of 2pi/alat. latVecs is then required to contain a list of lattice vectors.
    '''
    Hk_ys, Hk_evecs = [], []
    for k in Hr_ks:
        this_Hk = None
        # k in QE's bands.dat is given in Cartesian basis inunits 2pi/alat;
        # convert to distance units.
        kDist = []
        for i in range(3):
            kDist.append(k[i] * 2.0 * np.pi / alat)
        # Get eigenvalues for this k.
        this_Hk = Hk(kDist, Hr, latVecs)

        if not plot_evecs:
            evals = sorted(linalg.eigvalsh(this_Hk))
            # Set up Hk_ys to be a length-len(evs) list of empty lists.
            if len(Hk_ys) == 0:
                for i in range(len(evals)):
                    Hk_ys.append([])
            # Add each of this k-point's eigenvalues to the corresponding list.
            for i, ev in enumerate(evals):
                Hk_ys[i].append(ev)
        else:
            evals, evecs = linalg.eigh(this_Hk)
            Hk_ys.append(evals)
            Hk_evecs.append(evecs)

    return Hk_ys, Hk_evecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QE and Wannier bands.")
    parser.add_argument('QE_path', type=str,
            help="Path to QE eigenvalue file.")
    parser.add_argument('outPath', type=str, help="Path for output file.")
    parser.add_argument('--Hr_path', type=str,
            help="Path to Wannier Hamiltonian file.", default=None)
    parser.add_argument('--scf_path', type=str,
            help="Path to QE scf output file.", default=None)
    parser.add_argument('--minE', type=float, help="Minimal energy to plot.", default=None)
    parser.add_argument('--maxE', type=float, help="Maximum energy to plot.", default=None)
    parser.add_argument('--show', help="Show plot before writing file.",
                        action='store_true')
    args = parser.parse_args()

    if args.Hr_path is not None:
        nbnd, nks, evalsQE = extractQEBands(args.QE_path)
        print("QE eigenvalues loaded.")
        Hr = extractHr(args.Hr_path)
        print("Wannier Hamiltonian loaded.")
        alat = alat_from_scf(args.scf_path)
        latVecs = latVecs_from_scf(args.scf_path)
        plotBands(evalsQE, Hr, alat, latVecs, args.minE, args.maxE, args.outPath, args.show)
    else:
        plotDFTBands(args.QE_path, args.outPath, args.minE, args.maxE, args.show)
