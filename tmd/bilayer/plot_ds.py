import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tmd.pwscf.parseScf import total_energy_eV_from_scf
from tmd.bilayer.bilayer_util import global_config
from tmd.bilayer.dgrid import get_prefixes

def ds_from_prefixes(prefixes):
    ds = []
    for prefix in prefixes:
        sp = prefix.split("_")
        da = float(sp[-3])
        db = float(sp[-1])
        ds.append((da, db))

    return ds

def wrap_cell(ds, prefixes):
    wrapped_ds, wrapped_prefixes = [], []
    for d, prefix in zip(ds, prefixes):
        da, db = d[0], d[1]
        wrapped_ds.append(d)
        wrapped_prefixes.append(prefix)

        if da == 0.0 and db == 0.0:
            wrapped_ds.append((1.0, 1.0))
            wrapped_prefixes.append(prefix)
            wrapped_ds.append((0.0, 1.0))
            wrapped_prefixes.append(prefix)
            wrapped_ds.append((1.0, 0.0))
            wrapped_prefixes.append(prefix)
        elif da == 0.0:
            wrapped_ds.append((1.0, db))
            wrapped_prefixes.append(prefix)
        elif db == 0.0:
            wrapped_ds.append((da, 1.0))
            wrapped_prefixes.append(prefix)

    return wrapped_ds, wrapped_prefixes

def _main():
    gconf = global_config()
    work = gconf["work_base"]
    
    global_prefix = "MoS2_WS2"
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)

    ds, prefixes = wrap_cell(ds, prefixes)
    print(ds)
    print(prefixes)

    dps = list(zip(ds, prefixes))
    dps = sorted(dps, key=lambda dp: dp[0][1])
    dps = sorted(dps, key=lambda dp: dp[0][0])

    base = None
    for d_i, (d, prefix) in enumerate(dps):
        if d == (0.0, 0.0):
            base = d_i

    if base is None:
        raise ValueError("d = (0, 0) not found")

    energies = []
    for d, prefix in dps:
        print(d, prefix)
        wannier_dir = os.path.join(work, prefix, "wannier")
        scf_path = os.path.join(wannier_dir, "scf.out")
        energy = total_energy_eV_from_scf(scf_path)
        energies.append(energy)

    dps, energies = wrap_cell(dps, energies)

    base_energy = energies[base]
    energies_rel_meV = []
    for E in energies:
        E_rel = (E - base_energy) * 1000
        energies_rel_meV.append(E_rel)
        print(E_rel)

    xs, ys = [], []
    xs_set, ys_set = set(), set()
    for d, prefix in dps:
        xs.append(d[0])
        ys.append(d[1])
        xs_set.add(d[0])
        ys_set.add(d[1])

    num_xs = len(xs_set)
    num_ys = len(ys_set)

    X = np.array(xs).reshape((num_xs, num_ys))
    Y = np.array(ys).reshape((num_xs, num_ys))
    C_E = np.array(energies_rel_meV).reshape((num_xs, num_ys))

    plt.xlabel("$d_a$")
    plt.ylabel("$d_b$")

    num_ticks_xs, num_ticks_ys = 5, 5
    d_ticks_xs = []
    for x in np.linspace(0.0, 1.0, num_ticks_xs, endpoint=True):
        d_ticks_xs.append("{:.2f}".format(x))
    d_ticks_ys = []
    for y in np.linspace(0.0, 1.0, num_ticks_ys, endpoint=True):
        d_ticks_ys.append("{:.2f}".format(y))

    plt.xticks(np.linspace(0.0, num_xs-1, num_ticks_xs, endpoint=True), d_ticks_xs)
    plt.yticks(np.linspace(0.0, num_ys-1, num_ticks_ys, endpoint=True), d_ticks_ys)

    #plt.pcolormesh(X, Y, C_E.T, cmap=cm.viridis)
    #plt.pcolormesh(xs, ys, C_E.T, cmap=cm.viridis)
    plt.imshow(C_E.T, origin='lower', interpolation='none', cmap=cm.viridis)
    plt.colorbar()
    plt.title("$\\Delta E$ [meV]")
    #plt.pcolor(C_E.T, cmap=cm.viridis)
    #plt.scatter(xs, ys, c=energies_rel_meV)
    plt.savefig("{}.png".format(global_prefix), bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
