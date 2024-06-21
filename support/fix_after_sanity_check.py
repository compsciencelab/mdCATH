import h5py 
from os.path import join as opj
from glob import glob
from tqdm import tqdm
import os

def recover_file_path(sanity_check_log_path):
    import re
    tofix = []
    pattern = r'([\/\w\.-]+\.h5) (\d+) (\d+)'
    with open(sanity_check_log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Error in' in line:
                matches = re.findall(pattern, line)
                filepath = [match[0] for match in matches][0]
                temperature = [int(match[1]) for match in matches][0]
                replica = [int(match[2]) for match in matches][0]
                tofix.append((filepath, temperature, replica))
    return tofix
    
    
if __name__ == "__main__":
    sanity_check_log_path = 'sanity_check.log'
    tofix_list = recover_file_path(sanity_check_log_path)
    report = open('fix_after_sanity_check.log', 'w')
    report.write('Fixing numFrames and shape of coords and forces\n')
    """ for i, fixgroup in tqdm(enumerate(tofix_list), total=len(tofix_list)):
        file = fixgroup[0]
        pdb_id = os.path.basename(file).split('_')[-1].split('.')[0]
        with h5py.File(file, 'a') as f:
            report.write(f'Fixing {file} {fixgroup[1]} {fixgroup[2]}\n')
            replicagroup = f[pdb_id][str(fixgroup[1])][str(fixgroup[2])]
            report.write(f'Found {replicagroup.attrs["numFrames"]} frames\n')
            assert replicagroup['coords'].shape[0] == replicagroup['forces'].shape[0], f'Error in {file} {fixgroup[1]} {fixgroup[2]}, mismatch in forces and coords\n'
            replicagroup.attrs['numFrames'] = replicagroup['coords'].shape[0]
            report.write(f'Fixed numFrames to {replicagroup.attrs["numFrames"]}\n')
            report.write(f'-----------------------------------------------------------------\n')
    report.close() """
    source_file = '/shared/antoniom/buildCATHDataset/all_data_unique_h5/mdcath_analysis.h5'
    with h5py.File(source_file, 'a') as f:
        for i, fixgroup in tqdm(enumerate(tofix_list), total=len(tofix_list)):
            h5file = fixgroup[0]
            pdb_id = os.path.basename(h5file).split('_')[-1].split('.')[0]
            with h5py.File(h5file, 'r') as h5f:
                fixednumFrames = h5f[pdb_id][str(fixgroup[1])][str(fixgroup[2])]['coords'].shape[0]
                assert fixednumFrames == h5f[pdb_id][str(fixgroup[1])][str(fixgroup[2])].attrs['numFrames']
            replicagroup = f[pdb_id][str(fixgroup[1])][str(fixgroup[2])]
            report.write(f'Found {replicagroup.attrs["numFrames"]} frames\n')
            replicagroup.attrs['numFrames'] = fixednumFrames
            report.write(f'Fixed numFrames to {replicagroup.attrs["numFrames"]}\n')
            report.write(f'-----------------------------------------------------------------\n')
    report.close() 