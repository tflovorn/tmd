import os
import numpy as np
from tmd.wannier.bands import Hk_recip
from tmd.wannier.extractHr import extractHr
from tmd.bilayer.bilayer_util import global_config

def get_Hr(work, prefix):
    wannier_dir = os.path.join(work, prefix, "wannier")
    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))

    Hr = extractHr(Hr_path)
    return Hr

def H_klat_Glat(klat, Glat, Hr_paths, dps):
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    # Integrate by trapezoid method.
    # Lazy way to compute: for each region, compute each Hk at boundaries.
    # Avoid point storage scheme and possible high memory use
    # at cost of 4x runtime.
    # Trapezoid rule in 2D:
    # \int_{x1, x2} dx \int_{y1, y2} dy f(x,y) = (1/4)*(x2-x1)(y2-y1)
    #   * (f(x1, y1) + f(x1, y2) + f(x2, y1) + f(x2, y2))
    ds = []
    for d, prefix in dps:
        ds.append(d)

    d_boundary_indices, delta_a, delta_b = trapezoid_d_regions(ds)

    integral = None
    for region_indices in d_boundary_indices:
        region_sum = None
        for d_index in region_indices:
            d, prefix = dps[d_index]
            Hr = get_Hr(work, prefix)
            Hk = Hk_recip(klat, Hr)

            efac = np.exp(2*np.pi*1j*np.dot(Glat, d))
            val = Hk * efac
            if this_sum is None:
                region_sum = val
            else:
                region_sum += val

        region_integral = delta_a * delta_b * region_sum / 4
        if integral is None:
            integral = region_integral
        else:
            integral += region_integral

    return integral

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
