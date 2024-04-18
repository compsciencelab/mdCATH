# concatenate all multiple csv files
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def sorter(file_list):
    files_dict = {}
    for file in file_list:
        name = os.path.basename(file).split(".")[0]
        num = int(name.split("_")[-1])
        files_dict[num] = file
    
    return [files_dict[key] for key in sorted(files_dict.keys())]

csv_files = glob("csv_files/*.csv")

all_replicas = []
all_times = []
all_temperatures = []
all_residues = []
all_dssp = []
all_domains = []

for file in tqdm(csv_files, total=len(csv_files), desc="Concatenating"):
    tmpdf = pd.read_csv(file)
    tmpdf["DSSP State"] = tmpdf["DSSP State"].astype('category')
    tmpdf["Domain"] = tmpdf["Domain"].astype('category')
    all_replicas.extend(tmpdf["Replica"].values)
    all_times.extend(tmpdf["Time"].values)
    all_temperatures.extend(tmpdf["Temperature"].values)
    all_residues.extend(tmpdf["Residue"].values)
    all_dssp.extend(tmpdf["DSSP State"].values)
    all_domains.extend(tmpdf["Domain"].values)

df = pd.DataFrame({'replica': all_replicas,
                     'time': all_times,
                     'temperature': all_temperatures,
                     'residues': all_residues,
                     'dssp state': all_dssp,
                     'domain': all_domains})
    
df.to_csv("mdcath_info.csv", index=False)