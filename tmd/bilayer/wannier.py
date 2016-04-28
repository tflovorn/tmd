import os
from tmd.wannier.extractHr import extractHr

def get_Hr(work, prefix):
    wannier_dir = os.path.join(work, prefix, "wannier")
    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))

    Hr = extractHr(Hr_path)
    return Hr
