# mdCATH Dataset Repository

Welcome to the mdCATH dataset repository! This repository houses all the scripts and notebooks utilized for generating, analyzing, and validating the mdCATH dataset. The dataset is available on the Hugging Face platform. All mdCATH trajectories can be directly visualized on PlayMolecule without needing to download, or alternatively download them in XTC format from PlayMolecule if needed.

## Useful Links
- Playmolecule: https://open.playmolecule.org/mdcath </br>
- Hugging Face: https://huggingface.co/datasets/compsciencelab/mdCATH

## Repository Structure

- #### `user`
    - Provides tutorials and example scripts to help new users familiarize themselves with the dataset.
    - Step-by-step tutorials to guide users through common tasks and procedures using the dataset.
    - Example scripts that demonstrate practical applications of the dataset in research scenarios.

- #### `utils`
    - TCL code to load mdCATH's HDF5 files in VMD (for end-users)
    - Python code to convert files to XTC format (for end-users)

- #### `generator`
    - Directory with the scripts used to generate the dataset.
    - `builder/generator.py`: is the main script responsible for dataset creation. It processes a list of CATH domains and their molecular dynamics outputs to produce H5 files for the mdCATH dataset. It features multiprocessing to accelerate the dataset generation process. For each domain, an H5 file is created accompanied by a log file that records the progress.

- #### `analysis`
    - Houses tools required for analyzing the dataset.
    - This directory includes various scripts and functions used to perform the analyses and generate the plots presented in the paper.


## Citation

> Antonio Mirarchi, Toni Giorgino and Gianni De Fabritiis. *mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics*. https://arxiv.org/abs/2407.14794 

```
@misc{mirarchi2024mdcathlargescalemddataset,
      title={mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics}, 
      author={Antonio Mirarchi and Toni Giorgino and Gianni De Fabritiis},
      year={2024},
      eprint={2407.14794},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2407.14794}, 
}
```
