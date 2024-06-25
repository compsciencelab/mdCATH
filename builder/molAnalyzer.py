import os
import logging
import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable
from moleculekit.projections.metricrmsd import MetricRmsd
from moleculekit.projections.metricgyration import MetricGyration
from moleculekit.projections.metricfluctuation import MetricFluctuation
from moleculekit.projections.metricsecondarystructure import MetricSecondaryStructure

ANGSTROM_TO_NM = 0.1
RMSD_CUTOFF = 40  # nm

def encodeDSSP(dssp):
    encodedDSSP = []
    for i in range(len(dssp)):
        encodedDSSP.append([x.encode("utf-8") for x in dssp[i]])
    return encodedDSSP

def txt_toH5(txtfile, h5group, dataset_name="pdb"):
    """Write the content of the txt file to the h5 group as a dataset.
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
                pdb_lines = [
                    line
                    for line in pdb_file.readlines()
                    if line.startswith("ATOM") or line.startswith("MODEL")
                ]
                pdbcontent = "".join(pdb_lines)
            h5group.create_dataset(dataset_name, data=pdbcontent.encode("utf-8"))

    elif txtfile.endswith(".psf"):
        with open(txtfile, "r") as psf_file:
            psfcontent = psf_file.read()
            h5group.create_dataset("psf", data=psfcontent.encode("utf-8"))
    else:
        raise ValueError(f"Unknown file type: {txtfile}")

class molAnalyzer:
    def __init__(self, pdbFile, file_handler=None, processed_path="."):
        """MolAnalyzer class take care of the analysis of the molecule, it builds the molecule object and compute all a serires of properties
        this will be then used to generate a series othe h5dataset.
        Parameters
        ----------
        pdbFile : str
            The path to the pdb file
        file_handler : logging.FileHandler
            The file handler to be used to write the log file
        processed_path : str
            The path where the processed files will be saved, in this case the filtered pdb file
        """
        self.processed_path = processed_path
        self.molLogger = logging.getLogger("MolAnalyzer")
        if file_handler is not None:
            self.molLogger.addHandler(file_handler)
        logging.getLogger("moleculekit").handlers = [self.molLogger]

        self.pdbFile = pdbFile
        self.pdbName = os.path.basename(pdbFile).split(".")[0]

        # the mol object is created from structure.pdb and structure.psf files which were used to start the simulation
        # all atom and solvent atoms are considered
        self.mol = Molecule(pdbFile)
        self.mol.read(pdbFile.replace(".pdb", ".psf"))
        self.proteinIdxs = self.mol.get("index", sel="protein")

        # to have retrocompatibility with initial version of mdcath the chain of the protein atoms is set to 0
        # TODO: change in future version of mdcath
        self.mol.chain[self.proteinIdxs] = "0"
        
        self.protein_mol = self.mol.copy()
        self.protein_mol.filter("protein")
        self.pdb_filtered_name = (
            f"{self.processed_path}/{self.pdbName}_protein_filter.pdb"
        )
        self.protein_mol.write(self.pdb_filtered_name)

    def computeProperties(
        self,
    ):
        """Compute the properties of the molecule"""
        tmpmol = self.protein_mol.copy()
        
        # dataset
        self.molData = {}
        self.molData["chain"] = tmpmol.chain
        self.molData["resname"] = tmpmol.resname
        self.molData["resid"] = tmpmol.resid
        self.molData["element"] = tmpmol.element
        self.molData["z"] = np.array([periodictable[x].number for x in tmpmol.element])
        ## attrs
        self.molAttrs = {}
        self.molAttrs["numProteinAtoms"] = tmpmol.numAtoms
        self.molAttrs["numResidues"] = tmpmol.numResidues
        self.molAttrs["numChains"] = len(set(list(self.molData["chain"])))
        self.molAttrs["numBonds"] = tmpmol.numBonds

    def trajAnalysis(self, traj_files, batch_idx):
        """Perform the analysis of the trajectory file after concatenation
        Parameters
        ----------
        trajFiles : list
            The list of the trajectory files (.xtc format)
        """
        self.trajAttrs = {}
        self.metricAnalysis = {}
        trajmol = self.mol.copy()
        try:
            trajmol.read(traj_files)
            trajmol.filter("protein")

        except (RuntimeError, ValueError, OSError) as e:
            self.molLogger.error(
                f"TRAJECTORY LOADING ERROR ON BATCH:{batch_idx} | SIM: {os.path.basename(traj_files[0]).split('-')[0]}"
            )
            self.molLogger.error(e)
            return None

        # first frame is used as reference for the rmsd
        # TODO: compute rmsd wrt to the input structure of the md-simulation (it's not the first frame of the trajectory)
        refmol = self.protein_mol.copy()
        refmol.coords = trajmol.coords[:, :, 0].copy()[:, :, np.newaxis]
        
        # RMSD
        # the rmsd is computed for the heavy atoms only wrt the first frame
        rmsd_metric = MetricRmsd(
            refmol=refmol,
            trajrmsdstr="protein and not element H",
            trajalnstr="protein and name CA",
            pbc=True,
        )
        rmsd = rmsd_metric.project(trajmol) * ANGSTROM_TO_NM  # shape (numFrames) [nm]
        rmsd_accepted_frames = np.where(rmsd < RMSD_CUTOFF)[0]
        self.metricAnalysis["rmsd"] = rmsd[rmsd_accepted_frames]

        trajmol.dropFrames(keep=rmsd_accepted_frames)
        
        # GYRATION RADIUS
        # gyration radius computed for the heay atoms only
        gyr_metric = MetricGyration(atomsel="not element H", refmol=refmol, 
                                    trajalnsel='name CA', refalnsel='name CA', centersel='protein', pbc=True)
        
        # the gyr_metric projection output rg, rg_x, rg_y, rg_z. We take only the first column which is the radius of gyration average over the three dimensions
        # the dtype is set to float64 to have retrocompatibility with initial version of mdcath
        # TODO: make everything float32 in future version of mdcath
        self.metricAnalysis["gyrationRadius"] = (gyr_metric.project(trajmol)[:, 0] * ANGSTROM_TO_NM).astype(np.float64)  # nm

        # RMSF
        # compute rmsf wrt their mean positions
        rmsf_metric = MetricFluctuation(atomsel="name CA")
        self.metricAnalysis["rmsf"] = (np.sqrt(np.mean(rmsf_metric.project(trajmol), axis=0)) * ANGSTROM_TO_NM).astype(np.float32)  # nm

        # DSSP
        dssp_metric = MetricSecondaryStructure(sel="protein", simplified=False, integer=False)
        dssp = dssp_metric.project(trajmol)
        self.metricAnalysis["dssp"] = np.array(encodeDSSP(dssp)).astype(object)
        
        # COORDS 
        self.coords = trajmol.coords.copy()  # Angstrom (numAtoms, 3, numFrames)
        
        # BOX
        # the box has shape (3, numFrames), we take the first frame only
        box = trajmol.box.copy()[:, 0] * ANGSTROM_TO_NM  # nm, shape (3,)
        self.box = np.diag(box) # shape (3, 3) 
        
    def readDCD(self, dcdFiles, batch_idx):
        dcdmol = self.mol.copy()
        try:
            dcdmol.read(dcdFiles)
            dcdmol.filter("protein")
            self.forces = dcdmol.coords.copy()  # kcal/mol/Angstrom

        except (RuntimeError, ValueError, OSError) as e:
            self.molLogger.error(
                f"FORCE LOADING ERROR ON BATCH:{batch_idx} | SIM: {os.path.basename(dcdFiles[0]).split('-')[0]}"
            )
            self.molLogger.error(e)
            return None

        if self.forces.shape != self.coords.shape:
            self.molLogger.warning(
                f"Forces {self.forces.shape} and Coords {self.coords.shape} shapes do not match"
            )
            last_idx = min(self.forces.shape[2], self.coords.shape[2])
            self.forces = self.forces[:, :, :last_idx]
            self.coords = self.coords[:, :, :last_idx]

            # Update the metricAnalysis to match the new shapes
            for metric, values in self.metricAnalysis.items():
                # rmsf skipped since it has shape num
                if metric == "rmsf":
                    continue
                self.metricAnalysis[metric] = values[:last_idx]
                self.molLogger.warning(
                    f"Shapes of {metric} have been adjusted to {self.metricAnalysis[metric].shape}"
                )

            self.molLogger.warning(
                f"Shapes have been adjusted to: coords {self.coords.shape}, forces {self.forces.shape} "
            )

    def sanityCheck(self):
        """Sanity check on the shapes of the arrays"""       
        numAtoms = self.protein_mol.numAtoms
        numResidues = self.protein_mol.numResidues
               
        # Fix coords and forces shapes to (numFrames, numAtoms, 3)
        if self.coords.shape[0] == numAtoms:
            self.coords = np.moveaxis(self.coords, -1, 0)
        if self.forces.shape[0] == numAtoms:
            self.forces = np.moveaxis(self.forces, -1, 0)       
        
        numFrames = self.coords.shape[0]
        assert self.coords.shape == (numFrames, numAtoms, 3), f"Coords shape {self.coords.shape} does not match (numFrames, numAtoms, 3)"
        assert self.coords.shape == self.forces.shape, f"Shapes of coords {self.coords.shape} and forces {self.forces.shape} do not match"
        assert self.metricAnalysis["rmsd"].shape[0] == numFrames, f'rmsd shape {self.metricAnalysis["rmsd"].shape[0]} and numFrames {numFrames} do not match'
        assert self.metricAnalysis["gyrationRadius"].shape[0] == numFrames,  f'gyrationRadius shape {self.metricAnalysis["gyrationRadius"].shape[0]} and numFrames {numFrames} do not match'
        assert self.metricAnalysis["rmsf"].shape[0] == numResidues, f'rmsf shape {self.metricAnalysis["rmsf"].shape[0]} and numResidues {numResidues} do not match'
        assert self.metricAnalysis["dssp"].shape[0] == numFrames, f'dssp shape {self.metricAnalysis["dssp"].shape[0]} and numFrames {numFrames} do not match'

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
            txt_toH5(self.pdbFile, molGroup, dataset_name="pdb")
            # write the filtered pdb file to the h5 file
            txt_toH5(self.pdb_filtered_name, molGroup, dataset_name="pdbProteinAtoms")
            # write the psf file to the h5 file
            txt_toH5(self.pdbFile.replace(".pdb", ".psf"), molGroup)
            # mol attributes
            for key, value in self.molAttrs.items():
                if key in attrs:
                    molGroup.attrs[key] = value
            # mol datasets
            for key, value in self.molData.items():
                if key in datasets:
                    molGroup.create_dataset(key, data=value)

        elif molGroup is None and replicaGroup is not None:
            self.sanityCheck()
            # replica attributes
            replicaGroup.attrs["numFrames"] = self.coords.shape[0]
            # replica datasets
            for key, value in self.metricAnalysis.items():
                if key in datasets:
                    replicaGroup.create_dataset(key, data=value)
                    replicaGroup[key].attrs["unit"] = "nm"

            replicaGroup.create_dataset("box", data=self.box)

            # coords and forces are written here using mdtraj function
            replicaGroup.create_dataset("coords", data=self.coords)
            replicaGroup.create_dataset("forces", data=self.forces)
            # add units attributes
            replicaGroup["coords"].attrs["unit"] = "Angstrom"
            replicaGroup["forces"].attrs["unit"] = "kcal/mol/Angstrom"

        else:
            self.molLogger.error("Only one of the two groups could be None")
            return