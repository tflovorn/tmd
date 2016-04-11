import numpy as np

def build(latconst, latvecs, cartpos, vacuum_dist):
    '''Construct the 3D cell and atomic basis.

    Returns (axes, latpos) where:
        axes: a list of the lattice vectors (in units of latconst);
        latpos: a list of (atom, position) pairs where atom is the atom's
        label ("C", "W", etc.) and position is the atom's position in direct
        lattice coordinates.

    latconst: the lattice constant in Angstroms;
    latvecs: a list of two 2D lattice vectors in units of latconst;
    cartpos: a list of (atom label, 3D atomic position) pairs, where the
        atomic positions are in Cartesian coordinates in units of latconst;
    vacuum_dist: minimum distance in Angstroms between monolayers in the
        periodic cell.
    '''
    min_z, max_z = _zbounds(cartpos)
    vac_lat = vacuum_dist / latconst
    z_length = vac_lat + max_z - min_z

    a2d, b2d = latvecs[0], latvecs[1]
    a = [a2d[0], a2d[1], 0.0]
    b = [b2d[0], b2d[1], 0.0]
    c = [0.0, 0.0, z_length]
    axes = [a, b, c]

    # D is the matrix with columns given by the lattice vectors.
    # (x, y, z)^T = D * (a, b, c)^T
    DTinv = np.linalg.inv(np.array(axes))
    latpos = []
    for at, pos in cartpos:
        # (a, b, c) = (x, y, z) * (D^T)^(-1)
        latval = np.dot(np.array(pos), DTinv)
        # Shift c values to be centered on the center of the cell.
        latval[2] = latval[2] + 0.5

        latpos.append([at, latval])

    return axes, latpos

def _zbounds(cartpos):
    '''Return min, max where min and max are the lowest and highest value
    of the z coordinate in atompos.
    '''
    minval, maxval = None, None
    for at, pos in cartpos:
        z = pos[2]
        if minval is None or z < minval:
            minval = z

        if maxval is None or z > maxval:
            maxval = z

    return minval, maxval
