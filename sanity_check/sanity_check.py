# Check the units and shapes of the data/attributes in the h5 files of the mdcath dataset

import h5py
from tqdm import tqdm
from os.path import join as opj
import concurrent.futures

import logging
# add handlers to logger
logger = logging.getLogger('sanity_check')
fh = logging.FileHandler('sanity_check.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

class Payload:
    def __init__(self, pdblist):
        self.pdblist = pdblist
        
    def runComputation(self, batch_idx):
        sanity_check(self.pdblist, batch_idx)
        
def units_check(name, repl_group):
    assert repl_group['coords'].attrs['unit'] == 'Angstrom', f"{name}: Found unit {repl_group['coords'].attrs['unit']} for coords"
    assert repl_group['forces'].attrs['unit'] == 'kcal/mol/Angstrom', f"{name}: Found unit {repl_group['forces'].attrs['unit']} for forces"
    for prop in ['rmsd', 'gyrationRadius', 'rmsf', 'box']:
        assert repl_group[prop].attrs['unit'] == 'nm', f"{name}: Found unit {repl_group[prop].attrs['unit']} for {prop}"
    
    assert 'unit' not in repl_group['dssp'].attrs.keys(), f"{name}: Found unit in dssp"
        
def shapes_check(name, repl_group, numResidues, numAtoms, numFrames):
    assert repl_group['coords'].shape == (numFrames, numAtoms, 3), f"{name}: Found shape {repl_group['coords'].shape} for coords"
    assert repl_group['forces'].shape == (numFrames, numAtoms, 3), f" {name}: Found shape {repl_group['forces'].shape} for forces"
    assert repl_group["rmsd"].shape[0] == numFrames, f'{name}: rmsd shape {repl_group["rmsd"].shape[0]} and numFrames {numFrames} do not match'
    assert repl_group["gyrationRadius"].shape[0] == numFrames,  f'{name}: gyrationRadius shape {repl_group["gyrationRadius"].shape[0]} and numFrames {numFrames} do not match'
    assert repl_group["rmsf"].shape[0] == numResidues, f'{name}: rmsf shape {repl_group["rmsf"].shape[0]} and numResidues {numResidues} do not match'
    assert repl_group["dssp"].shape[0] == numFrames, f'{name}: dssp shape {repl_group["dssp"].shape[0]} and numFrames {numFrames} do not match'
    assert repl_group["box"].shape == (3, 3), f'{name}: box shape {repl_group["box"].shape} does not match (3, 3)'
        

def sanity_check(pdblist, batch_idx):
    mdcath_dir = '/workspace8/antoniom/mdcath_htmd'
    dom = pdblist[batch_idx]
 
    with  h5py.File(opj(mdcath_dir, f'{dom}/mdcath_dataset_{dom}.h5'), 'r') as f:
        numAtoms = f[dom].attrs['numProteinAtoms']
        numResidues = f[dom].attrs['numResidues']
        
        for temp in ["320", "348", "379", "413", "450"]:
            for replica in ['0', '1', '2', '3', '4']:
                path_to_repl = f'{dom}/{temp}/{replica}'
                repl_group = f[path_to_repl]
                try:
                    numFrames = repl_group.attrs['numFrames']
                except Exception as e:
                    logger.error(f"{e}: {path_to_repl}")
                    continue
                    
                shapes_check(path_to_repl, repl_group, numResidues, numAtoms, numFrames)
                units_check(path_to_repl, repl_group)                
                
def readPDBs(pdbList):
    if isinstance(pdbList, list):
        return pdbList
    pdblist = []
    with open(pdbList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

def launch():
    pdblist_file = '/shared/antoniom/buildCATHDataset/accepted_pdbs_sorted.txt'
    pdblist = readPDBs(pdblist_file)
    payload = Payload(pdblist)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        future_to_batch = {executor.submit(payload.runComputation, batch): batch for batch in range(len(pdblist))}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(pdblist)):
            batch = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # report the assertion error if any
                logger.error(f"Error in batch {batch}: {e}")
                
    return results

if __name__ == '__main__':
    launch()