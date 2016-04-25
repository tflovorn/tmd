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

def wrap_cell(ds, values):
    wrapped_ds, wrapped_values = [], []
    for d, v in zip(ds, values):
        da, db = d[0], d[1]
        wrapped_ds.append(d)
        wrapped_values.append(v)

        if da == 0.0 and db == 0.0:
            wrapped_ds.append((1.0, 1.0))
            wrapped_values.append(v)
            wrapped_ds.append((0.0, 1.0))
            wrapped_values.append(v)
            wrapped_ds.append((1.0, 0.0))
            wrapped_values.append(v)
        elif da == 0.0:
            wrapped_ds.append((1.0, db))
            wrapped_values.append(v)
        elif db == 0.0:
            wrapped_ds.append((da, 1.0))
            wrapped_values.append(v)

    return wrapped_ds, wrapped_values

def sorted_d_group(ds, values):
    dvs = list(zip(ds, values))
    dvs = sorted(dvs, key=lambda dp: dp[0][1])
    dvs = sorted(dvs, key=lambda dp: dp[0][0])
    return dvs

def get_energies(work, dps):
    energies = []
    for d, prefix in dps:
        wannier_dir = os.path.join(work, prefix, "wannier")
        scf_path = os.path.join(wannier_dir, "scf.out")
        energy = total_energy_eV_from_scf(scf_path)
        energies.append(energy)

    return energies

def energies_relative_to(energies, dps, base_d):
    base_d_index = None
    for d_i, (d, prefix) in enumerate(dps):
        if d == (0.0, 0.0):
            base_d_index = d_i

    if base_d_index is None:
        raise ValueError("d = (0, 0) not found")

    base_energy = energies[base_d_index]
    energies_rel_meV = []
    for E in energies:
        E_rel = (E - base_energy) * 1000
        energies_rel_meV.append(E_rel)

    return energies_rel_meV

def plot_d_vals(plot_name, title, dps, values):
    xs, ys = [], []
    xs_set, ys_set = set(), set()
    for d, prefix in dps:
        xs.append(d[0])
        ys.append(d[1])
        xs_set.add(d[0])
        ys_set.add(d[1])

    num_xs = len(xs_set)
    num_ys = len(ys_set)

    C_E = np.array(values).reshape((num_xs, num_ys))

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

    plt.imshow(C_E.T, origin='lower', interpolation='none', cmap=cm.viridis)
    plt.colorbar()
    plt.title(title)
    plt.savefig("{}.png".format(plot_name), bbox_inches='tight', dpi=500)

def _main():
    gconf = global_config()
    work = os.path.expandvars(gconf["work_base"])
    
    global_prefix = "MoSe2_WSe2"
    prefixes = get_prefixes(work, global_prefix)
    ds = ds_from_prefixes(prefixes)

    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    energies = get_energies(work, dps)
    energies_rel_meV = energies_relative_to(energies, dps, (0.0, 0.0))

    E_title = "$\\Delta E$ [meV]"
    E_plot_name = "{}_energies".format(global_prefix)
    plot_d_vals(E_plot_name, E_title, dps, energies_rel_meV)

if __name__ == "__main__":
    _main()
