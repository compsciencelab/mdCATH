import h5py 
from os.path import join as opj
from glob import glob
from tqdm import tqdm
import os
if __name__ == "__main__":
    origin = '/workspace3/mdCATH_final/'
    all_h5 = glob(opj(origin, 'mdcath_dataset_*.h5'))
    report = open('sanity_check.log', 'w')
    report.write('Sanity check for mdCATH dataset\n')
    report.write('Checking numFrames and shape of coords and forces\n')
    report.write(f'Origin Dir: {origin}\n')
    for i, file in tqdm(enumerate(all_h5), total=len(all_h5)):
        pdb_id = os.path.basename(file).split('_')[-1].split('.')[0]
        with h5py.File(file, 'r') as f:
            pdbgroup = f[pdb_id]
            for temp in ['320', '348', '379', '413', '450']:
                tempgroup = pdbgroup[temp]
                for repl in tempgroup.keys():
                    replicagroup = tempgroup[repl]
                    numframes = replicagroup.attrs['numFrames']    
                    if replicagroup['coords'].shape[0] != numframes:
                        report.write(f'Error in {file} {temp} {repl}, numFrames is wrong\n')
                    if replicagroup['forces'].shape[0] != replicagroup['coords'].shape[0]:
                        report.write(f'Error in {file} {temp} {repl}, mismatch in forces and coords\n') 
    report.close()                           