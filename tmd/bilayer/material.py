import os
import inspect
import ase.db
import tmd.bilayer.cell
from tmd.bilayer.bilayer_util import _base_dir, global_config

def _base_dir():
    frame = inspect.getfile(inspect.currentframe())
    this_dir = os.path.dirname(os.path.abspath(frame))
    base_dir = os.path.join(this_dir, "..", "..")
    return os.path.normpath(base_dir)

def base_material(soc, xc, pp='nc'):
    # The following are values which are independent of the bilayer material.
    # To complete the dataset, must fix:
    #    eq_latconst, latvecs, cartpos, pseudo, weight, valence
    material = {}
    if pp == 'nc':
        if soc:
            if xc == 'lda':
                material["pseudo_dir"] = os.path.join(_base_dir(), "pseudo", "lda_soc")
            else:
                material["pseudo_dir"] = os.path.join(_base_dir(), "pseudo", "soc")
        else:
            if xc == 'lda':
                material["pseudo_dir"] = os.path.join(_base_dir(), "pseudo", "lda_no_soc")
            else:
                material["pseudo_dir"] = os.path.join(_base_dir(), "pseudo", "no_soc")
    elif pp == 'paw':
        if soc and xc == 'lda':
            material["pseudo_dir"] = os.path.join(_base_dir(), "pseudo", "pslibrary_lda_soc_paw")
        else:
            raise ValueError("paw/non-soc not supported")

    material["soc"] = soc

    material["band_path"] = [["\\Gamma", [0.0, 0.0]], ["M", [1/2, 0.0]], 
            ["K", [1/3, 1/3]], ["\\Gamma", [0.0, 0.0]]]

    material["vacuum_dist"] = 20.0

    if pp == 'nc':
        material["ecutwfc"] = 60.0
        material["ecutrho"] = 240.0
    elif pp == 'paw':
        material["ecutwfc"] = 85.0
        material["ecutrho"] = 395.0
    else:
        raise ValueError("unsupported pp value")

    material["degauss"] = 0.02

    material["scf_conv_thr"] = 1e-8
    material["nscf_conv_thr"] = 1e-10
    material["nscf_dos_conv_thr"] = 1e-10
    material["bands_conv_thr"] = 1e-10

    material["relax_conv_thr"] = 1e-8
    material["relax_etot_conv_thr"] = 2e-5
    material["relax_forc_conv_thr"] = 2e-4

    material["scf_Nk1"] = 9
    material["scf_Nk2"] = 9
    material["relax_Nk1"] = 9
    material["relax_Nk2"] = 9
    material["nscf_Nk1"] = 9
    material["nscf_Nk2"] = 9
    material["nscf_dos_Nk1"] = 9
    material["nscf_dos_Nk2"] = 9
    material["Nkband"] = 20

    return material

def get_weights(atoms_A, atoms_B=None):
    weights = {}
    syms, masses = list(atoms_A.get_chemical_symbols()), list(atoms_A.get_masses())
    if atoms_B is not None:
        syms.extend(list(atoms_B.get_chemical_symbols()))
        masses.extend(list(atoms_B.get_masses()))

    for sym, w in zip(syms, masses):
        weights[sym] = float(w)

    return weights

def get_pseudo(atoms_A, atoms_B=None, soc=True, pp='nc'):
    pseudo = {}
    syms = list(atoms_A.get_chemical_symbols())
    if atoms_B is not None:
        syms.extend(list(atoms_B.get_chemical_symbols()))

    for sym in syms:
        if pp == 'nc':
            if soc:
                pseudo_name = "{}_r.oncvpsp.upf".format(sym)
            else:
                pseudo_name = "{}.oncvpsp.upf".format(sym)
        elif pp == 'paw': # NOTE/TODO - this breaks for non-lda paw
            if soc:
                pseudo_name = "{}.rel-pz_high_psl.1.0.0.UPF".format(sym)
            else:
                raise ValueError("paw/non-soc not implemented")
        else:
            raise ValueError("unsupported pp value")

        pseudo[sym] = pseudo_name

    return pseudo

def get_valence(atoms_A, atoms_B=None, soc=True):
    Ms = ["Mo", "W"]
    Xs = ["S", "Se", "Te"]

    # total = nspin*(nlayers*9 + 2*nlayers*6)
    total = 0

    syms = list(atoms_A.get_chemical_symbols())
    if atoms_B is not None:
        syms.extend(list(atoms_B.get_chemical_symbols()))

    valence = {}
    for sym in syms:
        if sym in Ms:
            if soc:
                total += 10
            else:
                total += 5

            valence[sym] = ["d"]
        else:
            if soc:
                total += 6
            else:
                total += 3

            valence[sym] = ["p"]

    valence["total"] = total

    return valence

def get_material(db_path, sym_A, sym_B=None, c_bulk=None, d_a=None, d_b=None, c_sep=None, soc=True, xc='lda', atoms_A=None, atoms_B=None, ordering=None, pp='nc'):
    db = ase.db.connect(db_path)
    if atoms_A is None:
        atoms_A = tmd.bilayer.cell.get_atoms(db, sym_A, "H").toatoms()

    if atoms_B is None and sym_B is not None:
        atoms_B = tmd.bilayer.cell.get_atoms(db, sym_B, "H").toatoms()

    latvecs, cartpos, eq_latconst = tmd.bilayer.cell.bilayer_setup(atoms_A,
            atoms_B, c_bulk, d_a, d_b, c_sep_input=c_sep, ordering=ordering)

    material = base_material(soc, xc, pp)

    if sym_B is None:
        material["prefix"] = "{}".format(sym_A)
    else:
        material["prefix"] = "{}_{}_da_{:.3f}_db_{:.3f}".format(sym_A, sym_B, d_a, d_b)

    material["latconst"] = eq_latconst
    material["latvecs"] = latvecs
    material["cartpos"] = cartpos

    material["pseudo"] = get_pseudo(atoms_A, atoms_B, soc, pp)
    material["weight"] = get_weights(atoms_A, atoms_B)
    material["valence"] = get_valence(atoms_A, atoms_B, soc)

    # TODO - additional W90 parameters? (energy windows - or dist of window from E_F)

    return material, atoms_A, atoms_B

def _main():
    base = _base_dir()
    db_path = os.path.join(base, "c2dm.db")
    c_bulk = 12.296
    d_a, d_b = 0.1, 0.1

    material = get_material(db_path, "MoS2", "WS2", c_bulk, d_a, d_b)

    for k, v in material.items():
        print(k, v)

if __name__ == "__main__":
    _main()
