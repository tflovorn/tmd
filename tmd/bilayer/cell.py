import os
import numpy as np
import ase.db
from tmd.bilayer.bilayer_util import _base_dir, global_config

def get_atoms(db, formula, phase):
    system_list = list(db.select('formula={},xc=PBE,phase={}'.format(formula, phase)))
    if len(system_list) > 1:
        # TODO - handle this better
        raise ValueError("found multiple matches for {}, PBE, H phase".format(formula))

    atoms = system_list[0]
    #if atoms["hform"] > 0.0:
    #    print("Warning: hform {} > 0".format(formula))

    return atoms

def min_energy(db, formula):
    H = get_atoms(db, formula, "H")
    T = get_atoms(db, formula, "T")

    if H.hform < T.hform:
        return "H"
    else:
        return "T"

# Consistent ordering of lattice vectors and atoms for (Mo,W)(S,Se,Te)2 family.
def a_from_2H(atoms):
    cell = atoms.get_cell()
    a = cell[0][0]

    return a

def h_from_2H(atoms):
    pos = atoms.get_positions()
    c_S2 = pos[2][2]
    c_S1 = pos[1][2]
    
    return c_S2 - c_S1

def symbols_from_2H(atoms):
    syms = atoms.get_chemical_symbols()
    # Consistent M, X, X order.
    return syms[0], syms[1]

def bilayer_setup(atoms_A, atoms_B=None, c_sep=None, d_a=None, d_b=None):
    # Choose lattice constant from A.
    a = a_from_2H(atoms_A)

    h_A = h_from_2H(atoms_A)

    # Setyawan and Curtarolo 2010 basis
    avec = [1/2, -float(np.sqrt(3)/2)]
    bvec = [1/2, float(np.sqrt(3)/2)]
    latvecs = [avec, bvec]

    # Monolayer special case.
    if atoms_B == None:
        X2_A = h_A/2.0
        M_A  = 0.0
        X1_A = -h_A/2.0

        lat_pos = [[0.0, 0.0],
                   [1/3, 2/3],
                   [1/3, 2/3]]

        cartpos_2D = []
        for pos in lat_pos:
            cartpos_2D.append(np.dot(pos, np.array(latvecs)))

        cartpos = []
        zvals = [M_A, X1_A, X2_A]

        sym_M_A, sym_X_A = symbols_from_2H(atoms_A)
        symbols = [sym_M_A, sym_X_A, sym_X_A]

        for pos, z, sym in zip(cartpos_2D, zvals, symbols):
            cartpos_3D = [float(pos[0]), float(pos[1]), float(z) / float(a)]
            cartpos.append([sym, cartpos_3D])

        return latvecs, cartpos, float(a)

    # Bilayer.
    h_B = h_from_2H(atoms_B)

    X2_B = c_sep/2.0 + h_B
    M_B  = c_sep/2.0 + h_B/2.0
    X1_B = c_sep/2.0

    X2_A = -c_sep/2.0
    M_A  = -c_sep/2.0 - h_A/2.0
    X1_A = -c_sep/2.0 - h_A

    # M_A, X1_A, X2_A, M_B, X1_B, X2_B
    lat_pos = [[0.0, 0.0],
               [1/3, 2/3],
               [1/3, 2/3],
               [(0.0+d_a) % 1, (0.0+d_b) % 1],
               [(1/3+d_a) % 1, (2/3+d_b) % 1],
               [(1/3+d_a) % 1, (2/3+d_b) % 1]]

    cartpos_2D = []
    for pos in lat_pos:
        cartpos_2D.append(np.dot(pos, np.array(latvecs)))

    cartpos = []
    zvals = [M_A, X1_A, X2_A, M_B, X1_B, X2_B]

    sym_M_A, sym_X_A = symbols_from_2H(atoms_A)
    sym_M_B, sym_X_B = symbols_from_2H(atoms_B)
    symbols = [sym_M_A, sym_X_A, sym_X_A, sym_M_B, sym_X_B, sym_X_B]

    for pos, z, sym in zip(cartpos_2D, zvals, symbols):
        cartpos_3D = [float(pos[0]), float(pos[1]), float(z) / float(a)]
        cartpos.append([sym, cartpos_3D])

    return latvecs, cartpos, float(a)

def _emit_data(atoms):
    # For Atoms type docs, see https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    print(atoms.get_chemical_symbols())
    print(atoms.get_masses())
    print("Forces")
    print(atoms.get_forces())
    print("Cart-position")
    print(atoms.get_positions())
    print("Cell")
    print(atoms.get_cell())
    print("Lat-position")
    print(atoms.get_scaled_positions()) # positions in direct lattice coordinates

def _main():
    base = _base_dir()
    db_path = os.path.join(base, "c2dm.db")
    db = ase.db.connect(db_path)
    #ws2 = get_atoms(db, "WS2", "H").toatoms()
    #_emit_data(ws2)
    #compounds = ["MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"]
    #for f in compounds:
    #    print(f, min_energy(db, f))
    #    atoms = get_atoms(db, f, "H").toatoms()
    #    _emit_data(atoms)
    #    print("h = {} Ang".format(str(h_from_2H(atoms))))
    MoS2 = get_atoms(db, "MoS2", "H").toatoms()
    WS2 = get_atoms(db, "WS2", "H").toatoms()
    latvecs, cartpos, eq_latconst = bilayer_setup(MoS2, WS2, 3.0, 0.1, 0.1)
    print(latvecs)
    print(cartpos)
    print(eq_latconst)

if __name__ == "__main__":
    _main()
