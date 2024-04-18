from tqdm import tqdm
from os.path import join as opj
from glob import glob
import h5py 
import tempfile
import shutil

def sorter(filelist):
    """ returns a sorted list of files based on the last number (batch number) in the filename"""
    sortdict = {}
    for file in filelist:
        sortdict[file.split('_')[-1].split('.')[0]] = file
    return [sortdict[key] for key in sorted(sortdict.keys())]

if __name__ == '__main__':
    batches_dir = 'batch_files/'
    base_filename = 'mdcath_analysis' # or 'mdcath_source
    h5_list = sorter(glob(opj(batches_dir, f"{base_filename}_*.h5")))
    with tempfile.TemporaryDirectory() as temp:
        with h5py.File(opj(temp, 'merged.h5'), 'w') as merged:
            for h5_file in tqdm(h5_list, total=len(h5_list), desc="Merging"):
                with h5py.File(h5_file, 'r') as h5:
                    for key in h5.keys():
                        if key in merged.keys():
                            print(f"Key {key} already in merged file")
                            continue
                        merged.copy(h5[key], key)
                        
        shutil.copyfile(opj(temp, 'merged.h5'), f'{base_filename}.h5')
        print(f"Merged file saved to {base_filename}.h5')")