import h5py
import math
import numpy as np
from tqdm import tqdm
import seaborn as sns
import json
from os.path import join as opj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import *



if __name__ == "__main__":
    output_dir = "figures/"
    h5metrics = h5py.File("../generator/process/h5files/mdcath_analysis.h5", "r")

    plot_len_trajs(h5metrics, output_dir)
    plot_numAtoms(h5metrics, output_dir)
    plot_numResidues(h5metrics, output_dir)
    plot_RMSD(h5metrics, output_dir, rmsdcutoff=10, yscale="linear")
    plot_RMSF(h5metrics, output_dir, yscale="linear", temp_oi=None)
    plot_numRes_trajLength(h5metrics, output_dir)
    plot_GyrRad_SecondaryStruc(h5metrics, output_dir, numSamples=6, shared_axes=False, plot_type=['A', 'B'])
    plot_solidFraction_RMSF(h5metrics, output_dir, numSamples=3, simplified=True, repl='1')
    plot_solidFraction_vs_numResidues(h5metrics, output_dir, mean_across='all', temps=None, simplified=True)
    plot_heatmap_ss_time_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdbs=None, simplified=True)
    plot_ternary_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdbs=None, cbar=True) 
    plot_combine_metrics(h5metrics, output_dir)
    plot_maxNumNeighbors(h5metrics, output_dir, cutoff=['5A'])
    scatterplot_maxNumNeighbors_numNoHAtoms(h5metrics, output_dir, cutoff=['5A', '9A'])
    plot_numNoHAtoms(h5metrics, output_dir)
