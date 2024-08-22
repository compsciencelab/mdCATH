import h5py 
from tqdm import tqdm 
from tools import readPDBs

if __name__ == "__main__":
    # Set the directory and base filename (mdcath_analysis or mdcath_source)
    source = '/PATH/TO/SOURCE/FILE/FROM/WHICH/TO/COPY.h5'
    dest = 'h5files/mdcath_noh_source.h5'
    pdb_list = '/PATH/TO/PDB/LIST/FILE.txt/OR/LIST'
    
    doms_list = readPDBs(pdb_list)
    
    with h5py.File(source, mode='r') as source_h5:
        with h5py.File(dest, mode='a') as dest_h5:
            for dom in tqdm(doms_list, total=len(doms_list)):
                # del the group from dest_h5 if it exists
                if dom in dest_h5:
                    del dest_h5[dom]
                # copy the group from source_h5 to dest_h5
                source_h5.copy(dom, dest_h5)