import os 
import logging
import numpy as np 
import mdtraj as md
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable

class molAnalyzer:
    def __init__(self, pdbFile, filter=None):
        """ MolAnalyzer class take care of the analysis of the molecule, it builds the molecule object and compute all a serires of properties
        this will be then used to generate a series othe h5dataset.
        Parameters
        ----------
        pdbFile : str
            The path to the pdb file
        filter : str
            VMD filter to be used to select the atoms to be considered (default is None)
        """
        self.pdbFile = pdbFile
        self.pdbName = os.path.basename(pdbFile).split(".")[0]
        self.mol = Molecule(pdbFile)
        self.mol.filter("protein")
        self.molLogger = logging.getLogger("MolAnalyzer")
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
        
    def trajAnalysis(self, trajFiles):
        """Perform the analysis of the trajectory file after concatenation
        Parameters
        ----------
        trajFiles : list
            The list of the trajectory files (could be .xtc or .dcd files)
        """
        # check if the extension is .dcd or .xtc
        if trajFiles[0].endswith(".dcd"):
            self.molLogger.info(f"mol: {self.pdbName} | Trajectory is in .dcd format, processing as forces")
            try:
                self.forces = md.load(trajFiles, top=self.pdbFile, atom_indices=self.proteinIdxs)
            except RuntimeError as e:
                self.molLogger.error(f"Error while loading the trajectory {os.path.basename(trajFiles[0])}")
                self.molLogger.error(e)
                return None
        elif trajFiles[0].endswith(".xtc"):
            self.trajAttrs = {}
            self.metricAnalysis = {}
            refMol = md.load(self.pdbFile, atom_indices=self.proteinIdxs)
            self.molLogger.info(f"mol: {self.pdbName} | Trajectory is in .xtc format, processing as coordinates")
            try:
                self.traj = md.load(trajFiles, top=self.pdbFile, atom_indices=self.proteinIdxs)
            except RuntimeError as e:
                self.molLogger.error(f"Error while loading the trajectory {os.path.basename(trajFiles[0])}")
                self.molLogger.error(e)
                return None
            # md analysis
            self.metricAnalysis["rmsd"] = md.rmsd(self.traj, refMol)
            self.metricAnalysis["gyrationRadius"] = md.compute_rg(self.traj)
            self.metricAnalysis["rmsf"] = md.rmsf(self.traj, None)
            self.metricAnalysis["dssp"] = encodeDSSP(md.compute_dssp(self.traj, simplified=False))
            
            # traj attributes
            self.trajAttrs["numFrames"] = self.traj.n_frames
            self.trajAttrs["trajLength"] = self.traj.n_frames * 1 # ns, gpugrid computation with reportInterval every 1 ns           
        else:
            self.molLogger.error(f"Trajectory format not recognized, {trajFiles[0]}")
            return
        
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
            write_toH5(self.pdbFile, molGroup)
            # mol attributes
            for key, value in self.molAttrs.items():
                if key in attrs:
                    molGroup.attrs[key] = value
            # mol datasets
            for key, value in self.molData.items():
                if key in datasets:
                    molGroup.create_dataset(key, data=value)
                    
        elif molGroup is None and replicaGroup is not None:
            # replica attributes
            for key, value in self.trajAttrs.items():
                if key in attrs:
                    replicaGroup.attrs[key] = value
            # replica datasets
            for key, value in self.metricAnalysis.items():
                if key in datasets:
                    replicaGroup.create_dataset(key, data=value)
            # coords and forces are written here using mdtraj function
            replicaGroup.create_dataset("coords", data=self.traj.xyz)
            replicaGroup.create_dataset("forces", data=self.forces.xyz)
            # add units attributes 
            replicaGroup["coords"].attrs["unit"] = "Angstrom"
            replicaGroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"
            
        else:
            self.molLogger.error("Only one of the two groups could be None")
            return
        

def write_toH5(txtfile, h5group):
    if txtfile.endswith(".pdb"):
        with open(txtfile, "r") as pdb_file:
            pdbcontent = pdb_file.read()
            h5group.create_dataset("pdb", data=pdbcontent.encode('utf-8'))
            
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