# mdCATH Dataset Repository

Welcome to the mdCATH dataset repository! This repository houses all the scripts and notebooks utilized for generating, analyzing, and validating the mdCATH dataset.

## Directory Structure

- **builder**: This directory is dedicated to generating the mdCATH dataset in H5 format. For each domain, an H5 file is created within a specific folder. This folder also contains a filtered PDB file and a corresponding logger file.

- **analysis**: In this directory, you will find the tools necessary to analyze the dataset. It includes notebooks used to generate the plots featured in our paper and perform comprehensive dataset analysis.

- **support**: This directory contains scripts designed to interface with the CATH API, retrieving critical information such as superfamilies, architectures, and topologies.

## Highlights
- **Batching and Multiprocessing**: Leverage the power of parallel processing to generate the dataset faster and more efficiently.
  
- **Comprehensive Analysis**: A set of pre-configured scripts and notebooks are available in analysis dir to replicate the plots and analyses presented in the paper.