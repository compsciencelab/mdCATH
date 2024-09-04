# Kolmogorov-Smirnov test for the difference between distributions of 
# secondary structure contents at the end of the trajectories between the 
# four CATH superfamilies (reviewer 4 request).

import scipy
import pandas as pd

d = pd.read_csv("HeatMap_RSF_vs_TIME_50Samples_4Superfamilies.csv.gz")

T = 450
timepoint = 400

print(f"Comparing at {T} K and {timepoint} ns")

df = d.loc[(d.temp==T) & (d.time_points==timepoint), ]

for sf1 in range(4):
    for sf2 in range(sf1+1,4):
        p=scipy.stats.ks_2samp(df.loc[df.sf==sf1+1,"all_alpha_beta"], 
                               df.loc[df.sf==sf2+1,"all_alpha_beta"]).pvalue
        print(f"KS test between superfamily {sf1+1} and {sf2+1}: p = {p}")
        
