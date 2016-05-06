import os
import itertools
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tmd.wannier.bands import Hk_recip
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.wannier import get_Hr
from tmd.bilayer.plot_ds import ds_from_prefixes, wrap_cell, sorted_d_group, orbital_index, get_atom_order

def H_klat_Glat(dps, kGs):
    '''Integrate by trapezoid method.
    Lazy way to compute: for each region, compute each Hk at boundaries.
    Avoid point storage scheme and possible high memory use
    at cost of 4x runtime.

    Trapezoid rule in 2D:
    \int_{x1, x2} dx \int_{y1, y2} dy f(x,y) = (1/4)*(x2-x1)(y2-y1)
      * (f(x1, y1) + f(x1, y2) + f(x2, y1) + f(x2, y2))

    To avoid repeated unecessary loads of Hrs, and to avoid keeping all
    Hrs in memory, compute integral values for each (k, G) pair
    simultaneously.
    '''
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    ds = []
    for d, prefix in dps:
        ds.append(d)

    d_boundary_indices, delta_a, delta_b = trapezoid_d_regions(ds)

    # Calculate integral over each region individually.
    # region_integral_vals is a list (with indices corresponding to regions)
    # whose elements are dicts {(k, G): region_integral_val, ...}
    rint_args = []
    for region_indices in d_boundary_indices:
        rint_args.append([region_indices, delta_a, delta_b, kGs, work, dps])

    with Pool() as p:
        region_integral_vals = p.starmap(region_integral, rint_args)

    # Collect region integrals into totals.
    integral_totals = []
    for kG_i in range(len(kGs)):
        integral_totals.append(None)

    for region_list in region_integral_vals:
        for kG_index, kG_val in enumerate(region_list):
            if integral_totals[kG_index] is None:
                integral_totals[kG_index] = kG_val
            else:
                integral_totals[kG_index] += kG_val

    return integral_totals

def region_integral(region_indices, delta_a, delta_b, kGs, work, dps):
    print(region_indices)
    region_sums = []
    for kG_index in range(len(kGs)):
        region_sums.append(None)

    # Sum up the integrand values for each boundary point of the region.
    for d_index in region_indices:
        d, prefix = dps[d_index]
        Hr = get_Hr(work, prefix)

        for kG_index in range(len(kGs)):
            k, G = kGs[kG_index]
            Hk = Hk_recip(k, Hr)
            Gdotd = 2*np.pi*(G[0]*d[0] + G[1]*d[1])
            efac = np.exp(1j*Gdotd)
            val = Hk * efac
            if region_sums[kG_index] is None:
                region_sums[kG_index] = val
            else:
                region_sums[kG_index] += val

    # Calculate this region's contribution to the integral.
    integrals = []
    for kG_index in range(len(kGs)):
        integral_val = delta_a * delta_b * region_sums[kG_index] / 4
        integrals.append(integral_val)

    return integrals

def trapezoid_d_regions(ds):
    '''Determine the points bounding each region in the d-grid given
    by ds. Return a list with elements equal to a list of four integers
    which give the indices in ds for each point bounding a region.
    Also return the spacing between points along da and db.

    Assume ds form a regular grid (possibly with different deltas for 
    da and db) and are listed in order, with db (d[1]) changing faster
    than da (d[0]).
    Assume that d's cover a region [da in [0, 1], db in [0, 1]] and that there
    are always points at [0,0], [0,1], [1, 0], and [1, 1].
    '''
    delta_b = ds[1][1] - ds[0][1]
    num_db = 0
    for d in ds:
        num_db += 1
        db = d[1]
        if db == 1.0:
            break

    num_da = int(len(ds) / num_db)
    delta_a = ds[num_db][0] - ds[0][0]

    d_boundary_indices = []
    for ia in range(num_da - 1):
        for ib in range(num_db - 1):
            # lower left, lower right, upper left, upper right
            # (where da increases to the right and db increases upward)
            ll = num_db*ia + ib
            lr = num_db*ia + ib + 1
            ul = num_db*(ia + 1) + ib
            ur = num_db*(ia + 1) + ib + 1
            
            d_boundary_indices.append([ll, lr, ul, ur])

    return d_boundary_indices, delta_a, delta_b

def verify_Hr_orders_identical(work, prefixes):
    order = None
    for prefix in prefixes:
        if order is None:
            order = get_atom_order(work, prefix)
        else:
            this_order = get_atom_order(work, prefix)
            if this_order != order:
                raise ValueError("atom orders vary between ds")

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    
    global_prefix = "MoS2_WS2"
    soc = False
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)
    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    # Verify all Hr_orders are the same;
    # after this, assume order is the same for all ds.
    verify_Hr_orders_identical(work, prefixes)
    atom_Hr_order = get_atom_order(work, prefixes[0])

    orb_type = ("X2", "pz", "up", "X1p", "pz", "up")
    i_sym, i_orbital, i_spin = orb_type[0], orb_type[1], orb_type[2]
    j_sym, j_orbital, j_spin = orb_type[3], orb_type[4], orb_type[5]
    i_index = orbital_index(atom_Hr_order, i_sym, i_orbital, i_spin, soc)
    j_index = orbital_index(atom_Hr_order, j_sym, j_orbital, j_spin, soc)

    Gs = []
    num_Ga, num_Gb = 5, 5
    for Ga in range(num_Ga):
        for Gb in range(num_Gb):
            G = (Ga, Gb)
            Gs.append(G)

    ks = [(1/3, 1/3, 0)]
    kGs = list(itertools.product(ks, Gs))

    all_H_vals = H_klat_Glat(dps, kGs)

    Gas, Gbs = [], []
    H_K_re_vals, H_K_im_vals = [], []
    for kG_index, val in enumerate(all_H_vals):
        k, G = kGs[kG_index]
        # Only one k used.
        # TODO - check k?
        H_K_re_vals.append(val[i_index, j_index].real)
        H_K_im_vals.append(val[i_index, j_index].imag)
        Gas.append(float(G[0]))
        Gbs.append(float(G[1]))

    plt.scatter(Gas, Gbs, c=H_K_re_vals, cmap='viridis', s=50, edgecolors="none")
    plt.colorbar()

    plt.savefig("G_K_re.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.scatter(Gas, Gbs, c=H_K_im_vals, cmap='viridis', s=50, edgecolors="none")
    plt.colorbar()

    plt.savefig("G_K_im.png", bbox_inches='tight', dpi=500)
    plt.clf()

if __name__ == "__main__":
    _main()
