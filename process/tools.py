import numpy as np

def readPDBs(pdbList):
    if isinstance(pdbList, list):
        return pdbList
    pdblist = []
    with open(pdbList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

def get_secondary_structure_compositions(dssp):
    '''This funtcion returns the percentage composition of alpha, beta and coil in the protein.
    A special "NA" code will be assigned to each "residue" in the topology which isn"t actually 
    a protein residue (does not contain atoms with the names "CA", "N", "C", "O")
    '''
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2, 'NA': 3} 
    
    decoded_dssp = [el.decode() for el in dssp[0]]
    float_dssp = np.array([floatMap[el] for el in decoded_dssp])
    unique, counts = np.unique(float_dssp, return_counts=True)
    numResAlpha, numResBeta, numResCoil = 0, 0, 0
    for u, c in zip(unique, counts):
        if u == 0:
            numResAlpha += c
        elif u == 1:
            numResBeta += c
        else:
            # NA or Coil
            numResCoil += c
    # percentage composition in alpha, beta and coil
    alpha_comp = (numResAlpha / np.sum(counts)) * 100
    beta_comp = (numResBeta / np.sum(counts)) * 100
    coil_comp = (numResCoil / np.sum(counts)) * 100
    
    return alpha_comp, beta_comp, coil_comp

def get_max_neighbors(coords, distance):
    """This function computes the maximum number of neighbors for all the conformations in a replica using a distance threshold,
    Parameters:
    coords: np.array, shape=(num_frames, num_atoms, 3)
    distance: float, the distance threshold to consider two atoms as neighbors
    Returns:
    max_neighbors: int, the maximum number of neighbors found in the replica
    """
    from scipy.spatial import cKDTree
    
    max_neighbors = 0
    for i in range(coords.shape[0]):
        tree = cKDTree(coords[i])
        # Query the tree to find neighbors within the specified distance
        num_neighbors = tree.query_ball_tree(tree, distance)
        # Get the maximum number of neighbors for this conformation
        max_neighbors = max(max_neighbors, max(len(n) for n in num_neighbors))
    return max_neighbors

def get_solid_secondary_structure(dssp):
    """ This function returns the percentage of solid secondary structure in the protein, computed as 
    the sum of alpha and beta residues over the total number of residues."""
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2, 'NA': 3}     
    decoded_dssp = [el.decode() for el in dssp]
    float_dssp = np.array([floatMap[el] for el in decoded_dssp])
    unique, counts = np.unique(float_dssp, return_counts=True)
    numResAlpha, numResBeta, numResCoil = 0, 0, 0
    for u, c in zip(unique, counts):
        if u == 0:
            numResAlpha += c
        elif u == 1:
            numResBeta += c
        else:
            # NA or Coil
            numResCoil += c
    
    solid_secondary_structure = (numResAlpha+numResBeta)/np.sum(counts)
    return solid_secondary_structure