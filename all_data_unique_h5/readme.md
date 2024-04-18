This directory contains the script to write essential info to a unique h5 file. There are 2 possible file_type:
- mdcath_source.h5: used by the mdcath_dataloader in torchmd-net to select the data and setup the idxs.
- mdcath_analysis.h5: used to store the data for the analysis and visualization by analysis/plot_metrics_from_h5.py

Script usage:
- write_info_to_h5.py: generate the selected file type using batching
- read_info.ipynb: to inspect the content of the h5 file
- join_multiple_h5.py: to join multiple h5 files into one
- error.log: store the error message during the write_info_process
