import h5py 
from tqdm import tqdm 

def readPDBs(pdbList):
    if isinstance(pdbList, list):
        return pdbList
    pdblist = []
    with open(pdbList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

if __name__ == "__main__":
    # Set the directory and base filename (mdcath_analysis or mdcath_source)
    source = '/shared/antoniom/buildCATHDataset/process/h5files/recomputed_mdcath_htmd_noh_source/mdcath_noh_source_merged_htmd_fixed.h5'
    dest = 'h5files/mdcath_noh_source.h5'
    pdb_list_file = '/shared/antoniom/buildCATHDataset/process/all_doms_to_append.txt'
    
    doms_list = readPDBs(pdb_list_file)
    
    with h5py.File(source, mode='r') as source_h5:
        with h5py.File(dest, mode='a') as dest_h5:
            for dom in tqdm(doms_list, total=len(doms_list)):
                # del the group from dest_h5 if it exists
                if dom in dest_h5:
                    del dest_h5[dom]
                # copy the group from source_h5 to dest_h5
                source_h5.copy(dom, dest_h5)