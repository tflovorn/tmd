import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tmd.wannier.bands import Hk_recip
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes
from tmd.bilayer.wannier import get_Hr
from tmd.bilayer.plot_ds import ds_from_prefixes, wrap_cell, sorted_d_group, orbital_index, get_atom_order

def H_klat_Glat(klat, Gs, dps, all_orb_types, soc):
    '''Integrate by trapezoid method.
    Lazy way to compute: for each region, compute each Hk at boundaries.
    Avoid point storage scheme and possible high memory use
    at cost of 4x runtime.

    Trapezoid rule in 2D:
    \int_{x1, x2} dx \int_{y1, y2} dy f(x,y) = (1/4)*(x2-x1)(y2-y1)
      * (f(x1, y1) + f(x1, y2) + f(x2, y1) + f(x2, y2))

    To avoid repeated unecessary loads of Hrs, and to avoid keeping all
    Hrs in memory, compute integral values for each (G, orb_types) pair
    simultaneously.
    '''
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    ds = []
    for d, prefix in dps:
        ds.append(d)

    d_boundary_indices, delta_a, delta_b = trapezoid_d_regions(ds)

    integrals = {}
    for region_indices in d_boundary_indices:
        region_sums = {}
        for d_index in region_indices:
            d, prefix = dps[d_index]
            atom_Hr_order = get_atom_order(work, prefix)
            Hr = get_Hr(work, prefix)
            Hk = Hk_recip(klat, Hr)

            for G, orb_types in itertools.product(Gs, all_orb_types):
                i_sym, i_orbital, i_spin = orb_types[0], orb_types[1], orb_types[2]
                j_sym, j_orbital, j_spin = orb_types[3], orb_types[4], orb_types[5]

                i_index = orbital_index(atom_Hr_order, i_sym, i_orbital, i_spin, soc)
                j_index = orbital_index(atom_Hr_order, j_sym, j_orbital, j_spin, soc)

                efac = np.exp(2*np.pi*1j*np.dot(np.array(G), np.array(d)))
                val = Hk[i_index, j_index] * efac
                if (G, orb_types) not in region_sums:
                    region_sums[(G, orb_types)] = val
                else:
                    region_sums[(G, orb_types)] += val

        for G, orb_types in itertools.product(Gs, all_orb_types):
            region_integral = delta_a * delta_b * region_sums[(G, orb_types)] / 4
            if (G, orb_types) not in integrals:
                integrals[(G, orb_types)] = region_integral
            else:
                integrals[(G, orb_types)] += region_integral

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

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    
    global_prefix = "MoS2_WS2"
    soc = False
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)
    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    Gs = []
    num_Ga, num_Gb = 5, 5
    for Ga in range(num_Ga):
        for Gb in range(num_Gb):
            G = (Ga, Gb)
            Gs.append(G)

    K = np.array([1/3, 1/3, 0.0])
    all_orb_types = [("X2", "pz", "up", "X1p", "pz", "up")]
    all_H_vals = H_klat_Glat(K, Gs, dps, all_orb_types, soc)

    Gas, Gbs = [], []
    H_K_re_vals, H_K_im_vals = [], []
    for (G, orb_type), val in all_H_vals.items():
        if orb_type == all_orb_types[0]:
            H_K_re_vals.append(val.real)
            H_K_im_vals.append(val.imag)
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
