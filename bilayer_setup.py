import ase.db

def get_atoms(db, formula):
    system_list = list(db.select('formula={},xc=PBE,phase=H'.format(formula)))
    if len(system_list) > 1:
        # TODO - handle this better
        raise ValueError("found multiple matches for {}, PBE, H phase".format(formula))

    atoms = system_list[0]
    if atoms["hform"] > 0.0:
        print("Warning: hform {} > 0".format(formula))

    return atoms

def _main():
    db = ase.db.connect("c2dm.db")
    ws2 = get_atoms(db, "WS2").toatoms()
    # For Atoms type docs, see https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    print(ws2.get_chemical_symbols())
    print(ws2.get_masses())
    print(ws2.get_positions())
    print(ws2.get_cell())

    print(ws2.get_scaled_positions()) # positions in direct lattice coordinates

if __name__ == "__main__":
    _main()
