This directory contains the script to write essential info to a unique h5 file. There are 2 possible file_type:
- mdcath_source.h5: used by the mdcath_dataloader in torchmd-net to select the data and setup the idxs.
- mdcath_analysis.h5: used to store the data for the analysis and visualization by analysis/plot_metrics_from_h5.py

Script usage:
- append_info_from_joined.py: append info to mdcath_ source/analysis h5 file, using another source/analysis h5 file (copy)
- append_info_to_h5.py: append/modify source/analysis h5 file retrieving the info from the mdcath dataset h5 files
- write_info_to_h5.py: generate the selected file type (source/analysis) using batching
- read_info.ipynb: to inspect the content of the mdcath source/analysis h5 file
- join_multiple_h5.py: to join multiple h5 files (batches), output of write_info_to_h5.py, into a single h5 file