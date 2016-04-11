import ase.db
import tmd.bilayer.cell

def base_material():
    # The following are values which are independent of the bilayer material.
    # To complete the dataset, must fix:
    #    eq_latconst, latvecs, cartpos, pseudo, weight, valence
    material = {}
    material["pseudo_dir"] = "$HOME/tmd/pseudo"

    material["band_path"] = [["K", [2/3, 1/3]], ["\\Gamma", [0.0, 0.0]],
            ["M", [0.5, 0.0]], ["K", [2/3, 1/3]]]

    material["vacuum_dist"] = 20.0

    material["ecutwfc_min"] = 60.0
    material["ecutrho_min"] = 200.0
    material["degauss_max"] = 0.02

    material["scf_conv_thr"] = 1e-8
    material["nscf_conv_thr"] = 1e-10
    material["bands_conv_thr"] = 1e-10

    material["relax_conv_thr"] = 1e-8
    material["relax_etot_conv_thr"] = 2e-5
    material["relax_forc_conv_thr"] = 2e-4

    material["scf_Nk1_min"] = 9
    material["scf_Nk2_min"] = 9
    material["relax_Nk1_min"] = 9
    material["relax_Nk2_min"] = 9
    material["nscf_Nk1_min"] = 9
    material["nscf_Nk2_min"] = 9
    material["Nkband_default"] = 20

    return material

def get_weights(atoms_A, atoms_B):
    weights = {}
    for sym, w in zip(atoms_A.get_chemical_symbols(), atoms_A.get_masses()):
        weights[sym] = w

    for sym, w in zip(atoms_B.get_chemical_symbols(), atoms_B.get_masses()):
        weights[sym] = w

    return weights

def get_pseudo(atoms_A, atoms_B):
    pseudo = {}
    for sym in atoms_A.get_chemical_symbols():
        pseudo_name = "{}_r.oncvpsp.upf".format(sym)
        pseudo[sym] = pseudo_name

    for sym in atoms_B.get_chemical_symbols():
        pseudo_name = "{}_r.oncvpsp.upf".format(sym)
        pseudo[sym] = pseudo_name

    return pseudo

def get_valence(atoms_A, atoms_B):
    Ms = ["Mo", "W"]
    Xs = ["S", "Se", "Te"]

    # total = nspin*(2*9 + 4*6)
    valence = {"total": 2*(2*9 + 4*6)} # or 4*9?

    syms = atoms_A.get_chemical_symbols()
    syms.extend(atoms_B.get_chemical_symbols())

    for sym in syms:
        if sym in Ms:
            valence[sym] = ["s", "p", "d"]
        else:
            valence[sym] = ["s", "p"]

    return valence

def _main():
    db = ase.db.connect("c2dm.db")
    MoS2 = tmd.bilayer.cell.get_atoms(db, "MoS2", "H").toatoms()
    WS2 = tmd.bilayer.cell.get_atoms(db, "WS2", "H").toatoms()
    latvecs, cartpos, eq_latconst = tmd.bilayer.cell.bilayer_setup(MoS2, WS2, 3.0, 0.1, 0.1)

    material = base_material()
    material["eq_latconst"] = eq_latconst
    material["latvecs"] = latvecs
    material["cartpos"] = cartpos

    material["weight"] = get_weights(MoS2, WS2)
    material["valence"] = get_valence(MoS2, WS2)

    for k, v in material.items():
        print(k, v)

if __name__ == "__main__":
    _main()
