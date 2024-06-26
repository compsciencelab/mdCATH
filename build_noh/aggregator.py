import os
import numpy as np
from moleculekit.molecule import Molecule
import logging
from moleculekit.periodictable import periodictable
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('aggregator')

nohcgmap = {
    "CH0": 1,
    "CH1": 2,
    "CH2": 3,
    "CH3": 4,
    "NH0": 5,
    "NH1": 6,
    "NH2": 7,
    "NH3": 8,
    "OH0": 9,
    "OH1": 10,
    "SH0": 11,
    "SH1": 12,
}

class AggForce:
    def __init__(self, pdb, top, z):
        """Starting from mdCATH dataset entry, return embeddings, forces and coords in the form
        of noh-mdCATH dataset.
        Parameters
        ----------
        pdb : str
            Path to the pdb file.
        top : str
            Path to the psf file.
        z : np.array
            Atomic numbers of the protein atoms.
        """
        self.pdb = pdb
        self.top = top
        self.mol = Molecule(pdb)
        self.mol.read(top)
        self.mol.filter('protein')
        if z is not None:
            self.z = z
            assert len(self.z) == self.mol.numAtoms
            assert (self.z == [periodictable[i].number for i in self.mol.get("element")]).all()
        else:
            self.z = [periodictable[i].number for i in self.mol.get("element")]

        self.noh_idxs = self.mol.get("index", sel="noh")
        self.constraints = self.getConstraints(self.mol)
        logger.info(f"Number of noh atoms: {len(self.noh_idxs)}")
        self.noh_mol = self.mol.copy()
        self.noh_mol.atomtype = self.noh_map(self.noh_mol)
        self.noh_mol.filter("noh")
        self.emb = np.array([nohcgmap[at] for at in self.noh_mol.atomtype])

    def noh_map(self, mol):
        """ Map the atom types to the noh atom types, return the atom types in the form of noh-mdCATH dataset."""
        atomtype = []
        for i in range(mol.numAtoms):
            if mol.element[i] == "H":
                atomtype.append("H")
                continue
            num_hydr = 0
            for j in mol.bonds:
                if i not in j:
                    continue
                elements = [mol.element[k] for k in j if k != i]
                num_hydr += elements.count("H")
            atomtype.append(mol.element[i] + "H" + str(num_hydr))

        return np.array(atomtype)

    def getConstraints(self, mol):
        """Get the constraints from the molecule object. The constraints are the hydrogen bonds."""
        elements = mol.element
        hydr_bonds = []
        for bond in mol.bonds:
            i = elements[bond[0]]
            j = elements[bond[1]]
            if i == "H" or j == "H":
                hydr_bonds.append(bond)
        return set(frozenset(v) for v in hydr_bonds)
    
    def process_coords(self, coords):
        "Return the processed coordinates, i.e. without the hydrogen atoms."
        tmp_coords = coords.copy()
        return tmp_coords[:, self.noh_idxs, :]
    
    def process_forces(self, forces):
        """Return the processed forces, i.e. with the hydrogen atom forces summed on the heavy atom 
        which they are bonded to, instead of using the aggforce matrix from the force-aggragation paper
        we are simply summing the forces on the heavy atom."""
        tmp_forces = forces.copy()
        for bond in self.constraints:
            i, j = bond
            if i in self.noh_idxs and j not in self.noh_idxs:
                tmp_forces[:, i, :] += tmp_forces[:, j, :]
            elif j in self.noh_idxs and i not in self.noh_idxs:
                tmp_forces[:, j, :] += tmp_forces[:, i, :]
            else:
                logger.error(f"Hydrogen bond between two hydrogen atoms: {i}, {j}")
        return tmp_forces[:, self.noh_idxs, :]