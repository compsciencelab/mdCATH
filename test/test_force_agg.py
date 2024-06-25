from aggregator import AggForce
from pytest import mark
from os.path import join as opj
from moleculekit.molecule import Molecule
import numpy as np

def generate_coords(num_atoms, noh_idxs):
    """ Genrate a numpy array of shape (1, num_atoms, 3) with all the coordinates set to 1 for the noh atoms, 
    and 0 for the heavy atoms. This is used to test if Aggforce is working correctly: no ones should be present
    in the final coordinates."""
    coords = np.zeros((1,num_atoms, 3))
    coords[:,noh_idxs,:] = 1
    return coords


def generate_forces(num_atoms, noh_idxs):
    """ Genrate a numpy array of shape (1, num_atoms, 3) with all the forces set to 1 for the noh atoms, 
    and 0 for the heavy atoms. This is used to test if Aggforce is working correctly: based on the num of hydrogens bonded, 
    the final force value at the heavy atom idx i should be the sum of the forces of the hydrogens bonded to it (i.e. num_hydr * 1 * 3) 
    where the 3 is constant for the x, y, z components of the force.
    """
    forces = np.zeros((1,num_atoms, 3))
    forces[:,noh_idxs,:] = 1
    return forces

@mark.parametrize("pdbids", ["1bl0A02", "2dl0A01", "3ossC00"])
def test_AggForce(pdbids):
    pdb_test_dir = 'pdb_test_dir'
    pdb = opj(pdb_test_dir, pdbids + ".pdb")
    psf = opj(pdb_test_dir, pdbids + ".psf")
    tempmol = Molecule(pdb)
    h_idxs = tempmol.get("index", sel="hydrogen")
    numAtoms = tempmol.numAtoms
    aggregator = AggForce(pdb, psf, None)
    coords = generate_coords(numAtoms, h_idxs)
    forces = generate_forces(numAtoms, h_idxs)
    process_coords = aggregator.process_coords(coords)
    process_forces = aggregator.process_forces(forces)
    atomtype = aggregator.noh_mol.atomtype
    assert np.all(process_coords == 0), "The coordinates of the hydrogen atoms should be set to 0."
    for i, at in enumerate(atomtype):
        if at == "H":
            continue
        num_hydr = int(at[-1])
        # The processed forces should be the sum of the forces of the hydrogens bonded to the atom.
        # This mean that the sum of the forces should be num_hydr * 1 * 3
        # For example 2 hydrogens bonded to a carbon atom, the sum of the forces should be 2 * 1 * 3 = 6
        fp = np.sum(process_forces[0, i, :])
        assert fp == num_hydr * 3, f"Expected force: {num_hydr * 3}, got {fp}"