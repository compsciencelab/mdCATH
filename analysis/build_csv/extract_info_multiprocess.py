# write a csv file containing the information of the mdCATH database
# Storing the following columns: Replica, Temperature, Time, Residue, Residue Name, Residue ID, DSSP State, Domain
import os 
import sys 
sys.path.append("/shared/antoniom/buildCATHDataset/builder/")
import h5py 
from os.path import join as opj
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import math 
import time
import concurrent.futures
from scheduler import ComputationScheduler
import tempfile

logger = logging.getLogger('reader')
logger.setLevel(logging.INFO)
# all the error messages will be written in the error.log file
fh = logging.FileHandler('write_csv_error.log')
rh = logging.FileHandler('write_csv_info.log')
fh.setLevel(logging.ERROR)
rh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(rh)

def readPDBs(pdbFileList):
    pdblist = []
    with open(pdbFileList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

class Payload:
    def __init__(self, scheduler, data_dir, output_dir='.'):
        self.scheduler = scheduler
        self.data_dir = data_dir
        self.output_dir = output_dir

    def runComputation(self, batch_idx):
        #logger.info(f"Batch {batch_idx} started")
        #logger.info(f"OMP_NUM_THREADS= {os.environ.get('OMP_NUM_THREADS')}")
        run(self.scheduler, batch_idx, self.data_dir, self.output_dir)

def get_secondary_structure_compositions(dssp):
    '''A special "NA" code will be assigned to each "residue" in the topology which isn"t 
    actually a protein residue (does not contain atoms with the names "CA", "N", "C", "O")
    '''
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2, 'NA': 3} 
    dssp_decoded = np.zeros((dssp.shape[0], dssp.shape[1]), dtype=object)
    for i in range(dssp.shape[0]):
        dssp_decoded[i] = [floatMap[el.decode()] for el in dssp[i]]
    
    dssp_decoded_flat = dssp_decoded.flatten()
    unique, counts = np.unique(dssp_decoded_flat, return_counts=True)
    total_residues = np.sum(counts)
    composition_percentage = {}
    inverse_floatMap = {0: "Helix", 1: "Sheet", 2: "Coil", 3: "NA"}
    for u, c in zip(unique, counts):
        composition_percentage[inverse_floatMap[u]] = (c / total_residues) * 100
    
    fixed_composition = {}
    for k in inverse_floatMap.values():
        if k in composition_percentage:
            fixed_composition[k] = composition_percentage[k]
        else:
            fixed_composition[k] = 0.0
    return fixed_composition['Helix'], fixed_composition['Sheet'], fixed_composition['Coil']

def read_data(h5file, pdbname):
    """ Function that iterates over the h5 file and extracts the information of the mdCATH dataset.
    This should be done for each replica,temperature, frame and residue. The information extracted is:
    - Replica
    - Temperature
    - Time
    - Residue
    - DSSP State (pandas category)
    - Domain
    it should return a pandas dataframe with the information extracted of shape numResidues, numTemps, NumReplicas
    
    """
    sim_names = ["sims320K", "sims348K", "sims379K", "sims413K", "sims450K"]
    all_df = []
    with h5py.File(h5file, "r") as h5:
        for sim in (sim_names):
            try:
                temperature = int(sim.strip("simsK"))
                for j, replica in enumerate(h5[pdbname][sim].keys()):
                    replica = int(replica)
                    assert replica == j, f"Replica {replica} is not equal to {j}"
                    dssp = h5[f"{pdbname}/{sim}/{replica}/dssp"] # shape (numFrames, numResidues)
                    for frame in range(dssp.shape[0]):
                        for residue in range(dssp.shape[1]):
                            dssp_label = dssp[frame, residue].decode()
                            new_df = pd.DataFrame({'replica': [replica], 
                                                    'temperature': [temperature], 
                                                    'time': [frame], 
                                                    'residue': [residue], 
                                                    'dssp_label': [dssp_label], 
                                                    'domain': [pdbname]},
                                                    ) 
                            # convert dssp_label to category
                            new_df['dssp_label'] = new_df['dssp_label'].astype('category')
                            all_df.append(new_df)
            except Exception as e:
                logger.error(f"{e} processing {pdbname} {sim} {replica}")
                continue
    tmp_df = pd.concat(all_df, axis=0, ignore_index=True)
    return tmp_df
  
def run(scheduler, batch_idx, data_dir, output_dir='.'):
    """Extract information from the mdCATH dataset."""
    pbbIndices = scheduler.process(batch_idx)
    for i, pdb in tqdm(enumerate(pbbIndices), total=len(pbbIndices), desc="processing"):                
        resfile = opj(output_dir, f"mdCATH_info_{pdb}.csv") #{batch_idx}.csv")
        h5_file = opj(data_dir, pdb, f"cath_dataset_{pdb}.h5")
        if not os.path.exists(h5_file):
            logger.error(f"File {h5_file} does not exist")
            continue
        domain_df = read_data(h5_file, pdb) 
        if i == 0:
            df = domain_df.copy()
        else:
            df = pd.concat([df, domain_df], axis=0) 
    df.to_csv(resfile)
    logger.info(f"{pdb} done!! Successfully written {resfile} for batch {batch_idx}")
            
def launch():
    data_dir = "/workspace7/antoniom/mdCATH/"
    output_dir = "/workspace7/antoniom/csv_info_mdCATH"
    pdb_list_file = '/shared/antoniom/buildCATHDataset/accetptedPDBs.txt'
    pdb_list = readPDBs(pdb_list_file)
    batch_size = 1
    toRunBatches = None
    startBatch = None   
    max_workers = 24
    # Get a number of batches
    numBatches = int(math.ceil(len(pdb_list) / batch_size))
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of total batches: {numBatches}")

    if toRunBatches is not None and startBatch is not None:
        numBatches = toRunBatches + startBatch
    elif toRunBatches is not None:
        numBatches = toRunBatches
    elif startBatch is not None:
        pass
    
    # Initialize the parallelization system
    scheduler = ComputationScheduler(batch_size, startBatch, numBatches, pdb_list)
    toRunBatches = scheduler.getBatches()
    logger.info(f"numBatches to run: {len(toRunBatches)}")

    payload = Payload(scheduler, data_dir, output_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        try:
            results = list(
                tqdm(
                    executor.map(payload.runComputation, toRunBatches),
                    total=len(toRunBatches),
                )
            )
        except Exception as e:
            print(e)
            raise e
    # this return it's needed for the tqdm progress bar
    return results



if __name__ == "__main__":
    launch()
