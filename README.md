# mdCATH Dataset Repository

Welcome to the mdCATH dataset repository! This repository houses all the scripts and notebooks utilized for generating, analyzing, and validating the mdCATH dataset. The dataset is available on the Hugging Face platform. All mdCATH trajectories can be directly visualized on PlayMolecule without needing to download, or alternatively download them in XTC format from PlayMolecule if needed.

## Useful Links
- Playmolecule: https://open.playmolecule.org/mdcath </br>
- Hugging Face: https://huggingface.co/datasets/compsciencelab/mdCATH

## Repository Structure

- #### `generator`
    - **Purpose**: Contains scripts used to generate the dataset.
    - **Key Scripts**:
        - `builder/generator.py`: This is the main script responsible for dataset creation. It accepts a list of CATH domains as input and produces mdcath dataset H5 files.
    - **Features**:
        - **Multiprocessing**: Utilizes multiprocessing to expedite the generation of the dataset.
        - **Output**: For each domain, an H5 file is created in a designated folder. Additionally, a log file is generated to track the progress of dataset creation.

- #### `analysis`
    - **Purpose**: Houses tools required for analyzing the dataset.
    - **Components**:
        - This directory includes various scripts and functions used to perform the analyses and generate the plots presented in the paper.

- #### `user`
    - **Purpose**: Provides tutorials and example scripts to help new users familiarize themselves with the dataset.
    - **Contents**:
        - Step-by-step tutorials to guide users through common tasks and procedures using the dataset.
        - Example scripts that demonstrate practical applications of the dataset in research scenarios. [TODO]

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
