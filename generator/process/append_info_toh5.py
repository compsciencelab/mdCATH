# Use this script if you want to append information to an existing source/analysis file.
# The script will append or modify the information in the source/analysis file. It works per domain-ID, 
# so all the temperatures and replicas will be update. 
# Use this if a small number of domains need to be updated, otherwise use the write_info_toh5.py script (multiprocessing supported).

import sys 
import h5py 
import logging
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from tools import get_secondary_structure_compositions, get_max_neighbors, get_solid_secondary_structure, readPDBs
sys.path.append("/../builder/")

# Create a custom logger
logger = logging.getLogger('append_info_toh5')
logger.setLevel(logging.INFO)  # Set the logger level to the lowest level you want to log

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('append_to_mdcath_analysis.log')

# Set levels for handlers
console_handler.setLevel(logging.ERROR)  # Only log errors to the console
file_handler.setLevel(logging.INFO)  # Log info and higher level messages to the file

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':
    # Define the h5 file for which the information will be modified
    origin_file = 'mdcath_analysis.h5'
    data_dir = "PATH/TO/MDCATH/DATASET/DIR"
    pdb_list = ['1cqzB02', '4i69A00', '3qdkA02'] 
    
    # Define the type of file to be written, source or analysis
    # Based on this different attributes will be written
    file_type = 'analysis' 
    noh_mode = False
    pdb_list = readPDBs(pdb_list)
    if file_type == 'analysis':
        to_recheck = open('log_doms_torecheck_mdcath_analysis_update.txt', 'a')
    basename = 'mdcath_noh' if noh_mode else 'mdcath' 
    
    with h5py.File(opj('h5files', origin_file), mode='a', libver='latest') as dest:
        for dom in tqdm(pdb_list, total=len(pdb_list)):
            source_file = f"{basename}_dataset_{dom}.h5"
            if dom in dest:
                del dest[dom]
            
            dom_group = dest.create_group(dom)
            with h5py.File(opj(data_dir, source_file), 'r') as source:
                    dom_group.attrs['numResidues'] = source[dom].attrs['numResidues']
                    dom_group.attrs['numProteinAtoms'] = source[dom].attrs['numProteinAtoms']
                    dom_group.attrs['numChains'] = source[dom].attrs['numChains']
                    dom_group.attrs['numNoHAtoms'] = len([el for el in source[dom]['z'][:] if el != 1])
                    availample_temps = [t for t in ['320', '348', '379', '413', '450'] if t in source[dom].keys()]
                    for temp in availample_temps:
                        temp_group = dom_group.create_group(temp)
                        for replica in source[dom][temp]:
                            repl_group = temp_group.create_group(replica)
                            if 'numFrames' not in source[dom][temp][replica].attrs.keys():
                                logger.error(f"numFrames not found in {dom} {temp} {replica}")
                                continue
                                
                            repl_group.attrs['numFrames'] = source[dom][temp][replica].attrs['numFrames']
                            
                            if file_type == 'analysis':
                                assert noh_mode == False, "Analysis file cannot be created for noh dataset"
                                repl_group.create_dataset('gyration_radius', data = source[dom][temp][replica]['gyrationRadius'][:])
                                repl_group.create_dataset('rmsd', data = source[dom][temp][replica]['rmsd'][:])
                                repl_group.create_dataset('rmsf', data = source[dom][temp][replica]['rmsf'][:])
                                repl_group.create_dataset('box', data = source[dom][temp][replica]['box'][:])
                                
                                try: 
                                    solid_secondary_structure = np.zeros(source[dom][temp][replica]['dssp'].shape[0])
                                    for i in range(source[dom][temp][replica]['dssp'].shape[0]):
                                        solid_secondary_structure[i] = get_solid_secondary_structure(source[dom][temp][replica]['dssp'][i])
                                    
                                    repl_group.create_dataset('solid_secondary_structure', data=solid_secondary_structure)
                                except Exception as e:
                                    logger.error(f"Error in {dom} {temp} {replica}")
                                    logger.error(e)
                                    to_recheck.write(f"{dom} {temp} {replica}\n")
                                    continue
                                    
                            
                            elif file_type == 'source':
                                if noh_mode:
                                    repl_group.attrs['max_num_neighbors_5A'] = get_max_neighbors(source[dom][temp][replica]['coords'][:], 5.5) # use 5.5 for confidence on the 5A
                                    repl_group.attrs['max_num_neighbors_9A'] = get_max_neighbors(source[dom][temp][replica]['coords'][:], 9.5) # use 9.5 for confidence on the 9A
                                    
                                    # The noh dataset does not have the dssp information, to store it in the source file we need to read the dssp from the original dataset                             
                                    with h5py.File(opj('/workspace8/antoniom/mdcath_htmd', dom, f"mdcath_dataset_{dom}.h5"), "r") as ref_h5:
                                        repl_group.attrs['min_gyration_radius'] = np.min(ref_h5[dom][temp][replica]['gyrationRadius'][:])
                                        repl_group.attrs['max_gyration_radius'] = np.max(ref_h5[dom][temp][replica]['gyrationRadius'][:])
                                        
                                        alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(ref_h5[dom][temp][replica]['dssp'])

                                        repl_group.attrs['alpha'] = alpha_comp
                                        repl_group.attrs['beta'] = beta_comp
                                        repl_group.attrs['coil'] = coil_comp
                                else:
                                    repl_group.attrs['min_gyration_radius'] = np.min(source[dom][temp][replica]['gyrationRadius'][:])
                                    repl_group.attrs['max_gyration_radius'] = np.max(source[dom][temp][replica]['gyrationRadius'][:])
                                    
                                    alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(source[dom][temp][replica]['dssp'])

                                    repl_group.attrs['alpha'] = alpha_comp
                                    repl_group.attrs['beta'] = beta_comp
                                    repl_group.attrs['coil'] = coil_comp

            logger.info(f"Successfully updated information for {dom}")