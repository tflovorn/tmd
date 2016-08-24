import numpy as np
from tmd.wannier.bands import Hk_recip
#from tmd.wannier.extractHr import extractHr
#from tmd.wannier.parseWin import parse_inner_window
#from SKfit.wannier.scfUtil import _get_D_from_scf, _get_alat_from_scf

def FindFitError(Hr, inner_win, DFT_evals):
    '''Return the RMS error and sum of absolute errors for the Wannier fit
    with given inner_win = (inner_window_min, inner_window_max).
    
    Hr is the Wannier Hamiltonian; DFT_evals gives the DFT eigenvalues as a list
    of (k_recip, k_recip_evals) pairs.
    '''
    errors = [] # list of lists of errors at each k-point
    for k, this_DFT_evals in DFT_evals:
        this_DFT_evals = sorted(this_DFT_evals)
        Hk = Hk_recip(k, Hr)
        this_Wan_evals = sorted(np.linalg.eigvalsh(Hk))
        dft_start_index, wan_start_index, num_states = _dft_wan_correspondence(this_DFT_evals, this_Wan_evals, inner_win)
        #print(inner_win)
        #print(dft_start_index, wan_start_index, num_states)
        #print(this_DFT_evals[dft_start_index:dft_start_index+num_states])
        #print(this_Wan_evals[wan_start_index:wan_start_index+num_states])
        this_errors = [abs(this_DFT_evals[i+dft_start_index] - this_Wan_evals[i+wan_start_index]) for i in range(num_states)]
        errors.append(this_errors)
        #print(this_errors)

    rms_error = _rms_error(errors)
    abs_error = _abs_error(errors)
    return rms_error, abs_error

def dft_wan_correspondence(DFT_evals, Wan_evals, inner_win):
    num_dft_in_inner, num_wan_in_inner = _count_inner(DFT_evals, Wan_evals, inner_win)
    if num_dft_in_inner == num_wan_in_inner:
        #print("eq")
        # Equal number of states in inner --> take all of them.
        low_dft_index = _lowest_in_inner(DFT_evals, inner_win)
        low_wan_index = _lowest_in_inner(Wan_evals, inner_win)
        return low_dft_index, low_wan_index, num_dft_in_inner
    elif num_dft_in_inner < num_wan_in_inner:
        #print("dft less")
        # Fewer DFT states --> add DFT states closest to edges of inner.
        count = num_wan_in_inner - num_dft_in_inner
        closest_indices, aboves = _closest_to_edge(DFT_evals, inner_win, count)
        dft_start_index = min(_lowest_in_inner(DFT_evals, inner_win), min(closest_indices))
        wan_start_index = _lowest_in_inner(Wan_evals, inner_win)
        return dft_start_index, wan_start_index, num_wan_in_inner
    else:
        #print("wan less")
        # Fewer Wannier states --> add Wannier states closest to edges of inner.
        count = num_dft_in_inner - num_wan_in_inner
        closest_indices, aboves = _closest_to_edge(Wan_evals, inner_win, count)
        dft_start_index = _lowest_in_inner(DFT_evals, inner_win)
        wan_start_index = min(_lowest_in_inner(Wan_evals, inner_win), min(closest_indices))
        return dft_start_index, wan_start_index, num_dft_in_inner

def _count_inner(DFT_evals, Wan_evals, inner_win):
    inner_min, inner_max = inner_win[0], inner_win[1]
    DFT_count, Wan_count = 0, 0
    for val in DFT_evals:
        if val >= inner_min and val <= inner_max:
            DFT_count += 1

    for val in Wan_evals:
        if val >= inner_min and val <= inner_max:
            Wan_count += 1

    return DFT_count, Wan_count

def _lowest_in_inner(evals, inner_win):
    inner_min, inner_max = inner_win[0], inner_win[1]
    minimum, min_index = None, None
    for val_index, val in enumerate(evals):
        if val >= inner_min and val <= inner_max:
            if minimum == None or val < minimum:
                minimum = val
                min_index = val_index

    return min_index

def _closest_to_edge(evals, inner_win, count):
    closest_indices, aboves = [], []
    window = [inner_win[0], inner_win[1]]
    for i in range(count):
        closest, closest_index = _closest_to_win(evals, window)
        if closest < window[0]:
            aboves.append(False)
            window[0] = closest
        elif closest > window[1]:
            aboves.append(True)
            window[1] = closest
        closest_indices.append(closest_index)

    return closest_indices, aboves

def _closest_to_win(evals, window):
    win_min, win_max = window[0], window[1]
    closest, closest_index = None, None
    for val_index, val in enumerate(evals):
        if val < win_min:
            diff = win_min - val
            if closest == None or diff < closest:
                closest = diff
                closest_index = val_index
        elif val > win_max:
            diff = val - win_max
            if closest == None or diff < closest:
                closest = diff
                closest_index = val_index

    return closest, closest_index

def _rms_error(errors):
    tot = 0.0
    num = 0
    for this_errors in errors:
        this_sq = map(lambda x: x**2, this_errors)
        tot += sum(this_sq)
        num += len(this_errors)

    return np.sqrt(tot / num)

def _abs_error(errors):
    tot = 0.0
    num = 0
    for this_errors in errors:
        tot += sum(this_errors)
        num += len(this_errors)

    return tot / num

def convert_QE_k_to_recip(QE_ev, alat, R):
    # QE bands file gives k values in Cartesian basis in 2pi/alat units.
    # Convert to reciprocal lattice basis.
    Rinv = np.linalg.inv(R)
    DFT_ev_recip = []
    for k, evals in QE_ev:
        kDist = np.array([k[i] * 2.0 * np.pi for i in range(3)])
        kRecip = tuple(np.dot(kDist, Rinv))
        DFT_ev_recip.append((kRecip, evals))

    return DFT_ev_recip

if __name__ == "__main__":
    _main()
