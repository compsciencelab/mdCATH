# Kolmogorov-Smirnov test for the difference between distributions of 
# secondary structure contents at the end of the trajectories between the 
# four CATH superfamilies (reviewer 4 request).

import scipy
import pandas as pd

def ks_pairwise_tests(df):
    for sf1 in range(4):
        for sf2 in range(sf1+1,4):
            p=scipy.stats.ks_2samp(df.loc[df.sf==sf1+1,"all_alpha_beta"], 
                                   df.loc[df.sf==sf2+1,"all_alpha_beta"]).pvalue
            print(f"KS test between superfamily {sf1+1} and {sf2+1}: p = {p}")

def mw_pairwise_tests(df):
    for sf1 in range(4):
        for sf2 in range(sf1+1,4):
            p=scipy.stats.mannwhitneyu(df.loc[df.sf==sf1+1,"all_alpha_beta"], 
                                   df.loc[df.sf==sf2+1,"all_alpha_beta"]).pvalue
            print(f"MW test between superfamily {sf1+1} and {sf2+1}: p = {p}")


T = 450
timepoint = 400

print(f"Comparing at {T} K and {timepoint} ns")

print("Dataset 50...")
d = pd.read_csv("HeatMap_RSF_vs_TIME_50Samples_4Superfamilies.csv.gz")
df = d.loc[(d.temp==T) & (d.time_points==timepoint), ]
ks_pairwise_tests(df)
mw_pairwise_tests(df)

print("Dataset ALL...")
d = pd.read_csv("HeatMap_RSF_vs_TIME_NoneSamples_4Superfamilies.csv.gz")
df = d.loc[(d.temp==T) & (d.time_points==timepoint), ]
ks_pairwise_tests(df)
mw_pairwise_tests(df)



