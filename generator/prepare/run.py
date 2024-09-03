"""
HTMD version used: 1.16
Note: CATH files don't have HETATM
wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S20.pdb.tgz
tar -zxvf cath-dataset-nonredundant-S20.pdb.tgz
"""

import os
import ray
import glob
from htmd.ui import *


def getBBsize(m):
    Xmin = np.amin(m.coords[:, :, 0], axis=0)
    Xmax = np.amax(m.coords[:, :, 0], axis=0)
    return Xmax - Xmin

@ray.remote
def cbuild(pdb):
    # Get a cube fitting the bounding box of the given molecule + at least R per side
    def getCubicSize(m, R=9.0):
        box = getBBsize(m)
        rmax = np.max(box)
        rmax_h = rmax / 2.0 + R
        cminmax = np.array([[-rmax_h, -rmax_h, -rmax_h], [rmax_h, rmax_h, rmax_h]])
        return cminmax

    try:
        m = Molecule(f"dompdb/{pdb}", type="pdb")
        m.center()
        nRes = len(np.unique(m.resid))
        if nRes < 50 or nRes > 500:
            raise ValueError(f"Domain has {nRes} residues, out of range.")
        cminmax = getCubicSize(m)
        if -2.0 * cminmax[0, 0] > 100:
            raise ValueError(f"Cubic box {-2*cminmax[0,0]} too large, out of range.")
        mp = proteinPrepare(m, pH=7.0)
        ms = autoSegment(mp)
        mw = solvate(ms, minmax=cminmax)
        charmm.build(
            mw,
            topo=["top/top_all22star_prot.rtf", "top/top_water_ions.rtf"],
            param=["par/par_all22star_prot.prm", "par/par_water_ions.prm"],
            outdir=f"build/{pdb}",
            saltconc=0.150,
        )
        return f"{pdb}: OK"
    except Exception as e:
        return f"{pdb}: {e}"


if __name__ == "__main__":
    # https://colab.research.google.com/github/ray-project/tutorial/blob/master/exercises/colab01-03.ipynb#scrollTo=IlrIrAyldfu4
    ray.init()

    pdblist = glob.glob("dompdb/???????")
    pdblist = [os.path.basename(p) for p in pdblist]
    out = [cbuild.remote(pdb) for pdb in pdblist]
    outf = ray.get(out)

    with open("build_failures.log", "w") as f:
        f.write("\n".join(outf))
