# This script reads the mdCATH dataset (h5 files) and writes the information to a new h5 file (source or analysis).
# It includes multiprocessing to speed up the process (batch processing).

import os 
import sys 
import h5py 
import math 
import shutil
import logging
import tempfile
import numpy as np
from tqdm import tqdm
import concurrent.futures
from os.path import join as opj
from tools import get_secondary_structure_compositions, get_max_neighbors, get_solid_secondary_structure, readPDBs
sys.path.append("/shared/antoniom/buildCATHDataset/builder/")
from scheduler import ComputationScheduler


logger = logging.getLogger('writer')
logger.setLevel(logging.INFO)
# all the error messages will be written in the error.log file
fh = logging.FileHandler('error.log')
fh.setLevel(logging.ERROR)
logger.addHandler(fh)

class Payload:
    def __init__(self, scheduler, data_dir, output_dir='.', file_type='source', noh=False):
        self.scheduler = scheduler
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.file_type = file_type
        self.noh = noh

    def runComputation(self, batch_idx):
        logger.info(f"Batch {batch_idx} started")
        run(self.scheduler, batch_idx, self.data_dir, self.output_dir, self.file_type, self.noh)
    
def run(scheduler, batch_idx, data_dir, output_dir='.', file_type='source', noh=False):
    """Extract information from the mdCATH dataset and write them to a h5 file per batch
    Parameters:
    scheduler: ComputationScheduler 
        the scheduler object that will process the batch
    batch_idx: int
        the index of the batch to process
    data_dir: str  
        the path to the directory containing the mdCATH dataset
    output_dir: str
        the path to the directory where the h5 files will be written
    file_type: str
        the type of file to be written: source or analysis
    noh: bool
        if True, the information will be extracted from the noh dataset
    """
    pdb_idxs = scheduler.process(batch_idx)
    basename = 'mdcath_noh' if noh else 'mdcath'
    file_name = f"{basename}_{file_type}_{batch_idx}.h5"
    resfile = opj(output_dir, file_name)
    
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with h5py.File(tmp_file, "w") as h5:
            for i, pdb in tqdm(enumerate(pdb_idxs), total=len(pdb_idxs), desc=f"processing batch {batch_idx}"):                
                h5_file = opj(data_dir, f"{basename}_dataset_{pdb}.h5")
                if not os.path.exists(h5_file):
                    logger.error(f"File {h5_file} does not exist")
                    continue
                                
                group = h5.create_group(pdb)
                with h5py.File(h5_file, "r") as origin:
                    group.attrs['numResidues'] = origin[pdb].attrs['numResidues']
                    group.attrs['numProteinAtoms'] = origin[pdb].attrs['numProteinAtoms']
                    group.attrs['numChains'] = origin[pdb].attrs['numChains']
                    group.attrs['numNoHAtoms'] = len([el for el in origin[pdb]['z'][:] if el != 1])
                    availample_temps = [t for t in ['320', '348', '379', '413', '450'] if t in origin[pdb].keys()]
                    for temp in availample_temps:
                        temp_group = group.create_group(temp)
                        for replica in origin[pdb][temp]:
                            repl_group = temp_group.create_group(replica)
                            if 'numFrames' not in origin[pdb][temp][replica].attrs.keys():
                                logger.error(f"numFrames not found in {pdb} {temp} {replica}")
                                continue
                                
                            repl_group.attrs['numFrames'] = origin[pdb][temp][replica].attrs['numFrames']
                            
                            if file_type == 'analysis':
                                assert noh == False, "Analysis file cannot be created for noh dataset"
                                repl_group.create_dataset('gyration_radius', data = origin[pdb][temp][replica]['gyrationRadius'][:])
                                repl_group.create_dataset('rmsd', data = origin[pdb][temp][replica]['rmsd'][:])
                                repl_group.create_dataset('rmsf', data = origin[pdb][temp][replica]['rmsf'][:])
                                repl_group.create_dataset('box', data = origin[pdb][temp][replica]['box'][:])
                                solid_secondary_structure = np.zeros(origin[pdb][temp][replica]['dssp'].shape[0])
                                for i in range(origin[pdb][temp][replica]['dssp'].shape[0]):
                                    solid_secondary_structure[i] = get_solid_secondary_structure(origin[pdb][temp][replica]['dssp'][i])
                                
                                repl_group.create_dataset('solid_secondary_structure', data=solid_secondary_structure)
                            
                            elif file_type == 'source':
                                if noh:
                                    repl_group.attrs['max_num_neighbors_5A'] = get_max_neighbors(origin[pdb][temp][replica]['coords'][:], 5.5) # use 5.5 for confidence on the 5A
                                    repl_group.attrs['max_num_neighbors_9A'] = get_max_neighbors(origin[pdb][temp][replica]['coords'][:], 9.5) # use 9.5 for confidence on the 9A
                                    
                                    # The noh dataset does not have the dssp information, to store it in the source file we need to read the dssp from the original dataset                             
                                    with h5py.File(opj('/workspace3/mdcath', f"mdcath_dataset_{pdb}.h5"), "r") as ref_h5:
                                        repl_group.attrs['min_gyration_radius'] = np.min(ref_h5[pdb][temp][replica]['gyrationRadius'][:])
                                        repl_group.attrs['max_gyration_radius'] = np.max(ref_h5[pdb][temp][replica]['gyrationRadius'][:])
                                        
                                        alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(ref_h5[pdb][temp][replica]['dssp'])

                                        repl_group.attrs['alpha'] = alpha_comp
                                        repl_group.attrs['beta'] = beta_comp
                                        repl_group.attrs['coil'] = coil_comp
                                else:
                                    repl_group.attrs['min_gyration_radius'] = np.min(origin[pdb][temp][replica]['gyrationRadius'][:])
                                    repl_group.attrs['max_gyration_radius'] = np.max(origin[pdb][temp][replica]['gyrationRadius'][:])
                                    
                                    alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(origin[pdb][temp][replica]['dssp'])

                                    repl_group.attrs['alpha'] = alpha_comp
                                    repl_group.attrs['beta'] = beta_comp
                                    repl_group.attrs['coil'] = coil_comp
                                                                
                                                       
        shutil.copyfile(tmp_file, resfile) 
                
def launch():
    data_dir = "PATH/TO/MDCATH/DATASET/DIR"
    output_dir = "batch_files"
    pdb_list_file = '../accetptedPDBs.txt'
    # Define the type of file to be written, source or analysis
    # Based on this different attributes will be written
    file_type = 'source' 
    noh_mode = True
    pdb_list = readPDBs(pdb_list_file)
    batch_size = 250
    toRunBatches = None
    startBatch = None   
    max_workers = 24
    
    os.makedirs(output_dir, exist_ok=True)
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

    payload = Payload(scheduler, data_dir, output_dir, file_type, noh_mode)

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
