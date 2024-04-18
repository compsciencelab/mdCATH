# write a csv file containing the information of the mdCATH database
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
import concurrent.futures
from scheduler import ComputationScheduler
import tempfile
import shutil

logger = logging.getLogger('writer')
logger.setLevel(logging.INFO)
# all the error messages will be written in the error.log file
fh = logging.FileHandler('error.log')
fh.setLevel(logging.ERROR)
logger.addHandler(fh)

def readPDBs(pdbFileList):
    pdblist = []
    with open(pdbFileList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

class Payload:
    def __init__(self, scheduler, data_dir, output_dir='.', file_type='source'):
        self.scheduler = scheduler
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.file_type = file_type

    def runComputation(self, batch_idx):
        logger.info(f"Batch {batch_idx} started")
        logger.info(f"OMP_NUM_THREADS= {os.environ.get('OMP_NUM_THREADS')}")
        run(self.scheduler, batch_idx, self.data_dir, self.output_dir, self.file_type)

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
    
def run(scheduler, batch_idx, data_dir, output_dir='.', file_type='source'):
    """Extract information from the mdCATH dataset and write them to a h5 file per batch"""
    pbbIndices = scheduler.process(batch_idx)
    file_name = f"mdcath_source_{batch_idx}.h5" if file_type == 'source' else f"mdcath_analysis_{batch_idx}.h5"
    resfile = opj(output_dir, file_name)
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with h5py.File(tmp_file, "w") as h5:
            for i, pdb in tqdm(enumerate(pbbIndices), total=len(pbbIndices), desc=f"processing batch {batch_idx}"):                
                h5_file = opj(data_dir, f"mdcath_dataset_{pdb}.h5")
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
                                del repl_group
                                continue
                                
                            repl_group.attrs['numFrames'] = origin[pdb][temp][replica].attrs['numFrames']
                            if file_type == 'analysis':
                                repl_group.create_dataset('gyration_radius', data = origin[pdb][temp][replica]['gyrationRadius'][:])
                                repl_group.create_dataset('rmsd', data = origin[pdb][temp][replica]['rmsd'][:])
                                repl_group.create_dataset('rmsf', data = origin[pdb][temp][replica]['rmsf'][:])
                                repl_group.create_dataset('box', data = origin[pdb][temp][replica]['box'][:])
                                solid_secondary_structure = np.zeros(origin[pdb][temp][replica]['dssp'].shape[0])
                                for i in range(origin[pdb][temp][replica]['dssp'].shape[0]):
                                    solid_secondary_structure[i] = get_solid_secondary_structure(origin[pdb][temp][replica]['dssp'][i])
                                
                                repl_group.create_dataset('solid_secondary_structure', data=solid_secondary_structure)
                            elif file_type == 'source':
                                repl_group.attrs['min_gyration_radius'] = np.min(origin[pdb][temp][replica]['gyrationRadius'][:])
                                repl_group.attrs['max_gyration_radius'] = np.max(origin[pdb][temp][replica]['gyrationRadius'][:])
                            
                            alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(origin[pdb][temp][replica]['dssp'])

                            repl_group.attrs['alpha'] = alpha_comp
                            repl_group.attrs['beta'] = beta_comp
                            repl_group.attrs['coil'] = coil_comp
                            
        shutil.copyfile(tmp_file, resfile) 
                
def launch():
    data_dir = "/workspace3/mdCATH_final"
    output_dir = "batch_files"
    pdb_list_file = '/shared/antoniom/buildCATHDataset/accetptedPDBs.txt'
    # Define the type of file to be written, source or analysis
    # Based on this different attributes will be written
    file_type = 'analysis' 
    pdb_list = readPDBs(pdb_list_file)
    batch_size = 200
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

    payload = Payload(scheduler, data_dir, output_dir, file_type)

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