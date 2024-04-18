import os 
import logging
import numpy as np 
import mdtraj as md
import MDAnalysis
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable

NM_TO_ANGSTROM = 10
RMSD_CUTOFF = 20 # nm 

class molAnalyzer:
    def __init__(self, pdbFile, filter=None, file_handler=None, processed_path="."):
        """ MolAnalyzer class take care of the analysis of the molecule, it builds the molecule object and compute all a serires of properties
        this will be then used to generate a series othe h5dataset.
        Parameters
        ----------
        pdbFile : str
            The path to the pdb file
        filter : str
            VMD filter to be used to select the atoms to be considered (default is None)
        file_handler : logging.FileHandler
            The file handler to be used to write the log file
        processed_path : str
            The path where the processed files will be saved, in this case the filtered pdb file
        """
        self.molLogger = logging.getLogger("MolAnalyzer")
        if file_handler is not None:
            self.molLogger.addHandler(file_handler)        
        logging.getLogger("moleculekit").handlers = [self.molLogger]
        
        self.pdbFile = pdbFile
        self.pdbName = os.path.basename(pdbFile).split(".")[0]
        self.mol = Molecule(pdbFile)
        self.mol.read(pdbFile.replace(".pdb", ".psf"))
        self.mol.filter("protein")
        self.pdb_filtered_name = f"{processed_path}/{self.pdbName}_protein_filter.pdb"
        if os.path.exists(self.pdb_filtered_name):
            self.molLogger.warning(f"Filtered pdb file {self.pdb_filtered_name} already exists, it will be overwritten")
            os.remove(self.pdb_filtered_name)
        self.mol.write(self.pdb_filtered_name)
        if filter is not None:
            try:
                self.mol.filter(filter)
            except:
                self.molLogger.warning(f"Filter {filter} not applied, all protein atoms will be considered")
                           
    def computeProperties(self,):
        """Compute the properties of the molecule"""
        self.tmpmol = self.mol.copy()
        # dataset
        self.molData = {}
        self.molData["chain"] = self.tmpmol.chain
        self.molData["resname"] = self.tmpmol.resname
        self.molData["resid"] = self.tmpmol.resid
        self.molData["element"] = self.tmpmol.element
        self.molData["z"] = np.array([periodictable[x].number for x in self.tmpmol.element])
        ## attrs
        self.molAttrs = {}
        self.molAttrs["numProteinAtoms"] = self.tmpmol.numAtoms
        self.molAttrs["numResidues"] = self.tmpmol.numResidues
        self.molAttrs["numChains"] = len(set(list(self.molData["chain"])))
        self.molAttrs["numBonds"] = self.tmpmol.numBonds
        self.proteinIdxs = self.tmpmol.get("index", sel="protein")
        
    def trajAnalysis(self, trajFiles, batch_idx):
        """Perform the analysis of the trajectory file after concatenation
        Parameters
        ----------
        trajFiles : list
            The list of the trajectory files (.xtc format)
        """
        self.trajAttrs = {}
        self.metricAnalysis = {}
        try:
            self.traj = md.load(trajFiles, top=self.pdbFile, atom_indices=self.proteinIdxs)
            self.coords = self.traj.xyz.copy() * NM_TO_ANGSTROM # convert to Angstrom
            
        except (RuntimeError, ValueError, OSError) as e:
            self.molLogger.error(f"TRAJECTORY LOADING ERROR ON BATCH:{batch_idx} | SIM: {os.path.basename(trajFiles[0]).split('-')[0]}")
            self.molLogger.error(e)
            return None           
        
        # rmsd analysis will consider the heavy atoms only
        idxsHeavyAtoms = self.traj.top.select("not element H")
        self.traj = self.traj.atom_slice(idxsHeavyAtoms)
                    
        self.metricAnalysis["rmsd"] = md.rmsd(self.traj, self.traj, 0)
        # the cutoff is used to filter the frames where a PBC jump occurred or other artifacts
        self.metricAnalysis["gyrationRadius"] = md.compute_rg(self.traj)
        self.last_idx_by_rmsd= np.where(self.metricAnalysis["rmsd"] < RMSD_CUTOFF)[0][-1]
        self.molLogger.info(f"Number of frames accepted after rmsd cutoff: {len(self.rmsd_accepted_frames)} of {self.traj.n_frames}")
        # the rmsf is computed for the CA atoms only, superposed to the first frame and respect to the average position
        idxsCA = self.traj.top.select("name CA")
        rmsf_traj_superpose = self.traj.copy()
        # cut the trajectory to the last frame accepted by the rmsd, otherwise the average position will be influenced by the frames with PBC jumps
        rmsf_traj_superpose = rmsf_traj_superpose.xyz[:self.last_idx_by_rmsd, :, :]
        rmsf_traj_superpose.superpose(rmsf_traj_superpose, 0, atom_indices=idxsCA)
        avg_xyz = np.mean(rmsf_traj_superpose.xyz[:, idxsCA, :], axis=0)
        self.metricAnalysis["rmsf"] = np.sqrt(3*np.mean((rmsf_traj_superpose.xyz[:, rmsf_traj_superpose, :] - avg_xyz)**2, axis=(0,2)))
        self.metricAnalysis["dssp"] = encodeDSSP(md.compute_dssp(self.traj, simplified=False))
        self.box = self.traj.unitcell_vectors[0] # the box is the same for all frames, so we take the first one

        # traj attributes
        self.trajAttrs["numFrames"] = self.traj.n_frames
        self.trajAttrs["trajLength"] = self.traj.n_frames * 1 # ns, gpugrid computation with reportInterval every 1 ns
        self.trajAttrs["box"] = self.traj.unitcell_vectors[0] # the box is the same for all frames, so we take the first one
        
        self.molLogger.info(f"Trajectory length: {self.trajAttrs['trajLength']} ns")         
    
    def readDCD(self, dcdFiles, batch_idx):
        try:
            u = MDAnalysis.Universe(self.pdbFile, dcdFiles)
            self.forces = np.zeros(shape=(u.trajectory.n_frames, len(self.proteinIdxs), 3), dtype=np.float32)
            for i, ts in enumerate(u.trajectory):
                self.forces[i] = u.atoms.positions[self.proteinIdxs,:] # already in kcal/mol/Angstrom
        except (RuntimeError, ValueError, OSError) as e:
            self.molLogger.error(f"FORCE LOADING ERROR ON BATCH:{batch_idx} | SIM: {os.path.basename(dcdFiles[0]).split('-')[0]}")
            self.molLogger.error(e)
            return None
        
        if self.forces.shape != self.coords.shape:
                self.molLogger.warning(f"Forces {self.forces.shape} and Coords {self.coords.shape} shapes do not match")
                last_idx = min(self.forces.shape[0], self.coords.shape[0])
                self.forces = self.forces[:last_idx, :, :]
                self.coords = self.coords[:last_idx, :, :]
                self.molLogger.warning(f"Shapes have been adjusted to {self.forces.shape} and {self.coords.shape}")
    
    def write_toH5(self, molGroup, replicaGroup, attrs, datasets):
        """Write the data to the h5 file, according to the properties defined in the input for the dataset
        Parameters
        ----------
        molGroup : h5py.Group
            The group of the molecule, this will be the parent group of the replicas
            so the properties of the molecule will be written here, they are shared among all replicas
        replicaGroup : h5py.Group
            The group of the replica, this will be the parent group of the properties of the replica, each replica has its own properties
            defined by trajectory analysis
        attrs: 
            list of attributes to be written in the h5 group
        datasets: 
            list of datasets to be written in the h5 group
        """
        if molGroup is not None and replicaGroup is None:
            # write the pdb file to the h5 file 
            write_toH5(self.pdbFile, molGroup, dataset_name="pdb")
            # write the filtered pdb file to the h5 file
            write_toH5(self.pdb_filtered_name, molGroup, dataset_name="pdbProteinAtoms")
            # mol attributes
            for key, value in self.molAttrs.items():
                if key in attrs:
                    molGroup.attrs[key] = value
            # mol datasets
            for key, value in self.molData.items():
                if key in datasets:
                    molGroup.create_dataset(key, data=value)
                    
        elif molGroup is None and replicaGroup is not None:
            # the number of frames to be written is the minimum between the last frame accepted by the rmsd and the total number of frames from the traj
            acceppted_frames = min(self.last_idx_by_rmsd, self.coords.shape[0])
            
            # replica attributes
            for key, value in self.trajAttrs.items():
                if key in attrs:
                    replicaGroup.attrs[key] = value
            # replica datasets
            for key, value in self.metricAnalysis.items():
                if key in datasets:
                    if key in ["rmsd", "gyrationRadius", "dssp"]:
                        replicaGroup.create_dataset(key, data=value[:acceppted_frames])
                    else:
                        replicaGroup.create_dataset(key, data=value)
                    # add attr for the units 
                    replicaGroup[key].attrs["unit"] = "nm" 
            
            replicaGroup.create_dataset("box", data=self.box)
  
            # coords and forces are written here using mdtraj function
            replicaGroup.create_dataset("coords", data=self.coords[:acceppted_frames])
            replicaGroup.create_dataset("forces", data=self.forces[:acceppted_frames])
            # add units attributes 
            replicaGroup["coords"].attrs["unit"] = "Angstrom"
            replicaGroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"
            
        else:
            self.molLogger.error("Only one of the two groups could be None")
            return
        

def write_toH5(txtfile, h5group, dataset_name="pdb"):
    """ Write the content of the txt file to the h5 group as a dataset. 
    Parameters
    ----------
    txtfile : str
        The path to the txt file to be written in the h5 group (pdb or psf file)
    h5group : h5py.Group
        The group of the h5 file where the dataset will be written
    dataset_name : str
        The name of the dataset to be written in the h5 group, used just for pdb extension. It can be either "pdb" or "pdbProteinAtoms"
    """
    if txtfile.endswith(".pdb"):
        with open(txtfile, "r") as pdb_file:
            if dataset_name == "pdb":
                pdbcontent = pdb_file.read()
            elif dataset_name == "pdbProteinAtoms":
                pdb_lines = [line for line in pdb_file.readlines() if line.startswith("ATOM") or line.startswith("MODEL")]
                pdbcontent = "".join(pdb_lines)
            h5group.create_dataset(dataset_name, data=pdbcontent.encode('utf-8'))
            
    elif txtfile.endswith(".psf"):
        with open(txtfile, "r") as psf_file:
            psfcontent = psf_file.read()
            h5group.create_dataset("psf", data=psfcontent.encode('utf-8'))
    else:
        raise ValueError(f"Unknown file type: {txtfile}")
    
def encodeDSSP(dssp):
    encodeDSSP = []
    for i in range(len(dssp)):
        encodeDSSP.append([x.encode('utf-8') for x in dssp[i]])
    return encodeDSSP