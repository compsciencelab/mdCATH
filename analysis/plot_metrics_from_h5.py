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

def print_stats(data, metric=""):
    # print, mean, std, min, max, median
    print("--------------------------")
    print(f"Stats for {metric}")
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Median: {np.median(data)}")
    if metric in ["Trajectory length"]:
        print(f'Total time of simulation: {np.sum(data)*1e-6} ms')
        print(f'Total number of trajectories: {len(data)}')
    if metric in ["Number of atoms", "Number of residues"]:
        print(f"Total {metric}: {np.sum(data)}")
    if metric in ["RMSD", "Trajectory length"]:
        return
    print(" ")
    
def plot_len_traj(h5metrics, output_dir):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Trajectory length"):
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                data.append(h5metrics[pdb][temp][repl].attrs['numFrames'])
    print_stats(data, metric="Trajectory length")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("Trajectory length (ns)")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "traj_len.png"), dpi=600)

def plot_numAtoms(h5metrics, output_dir):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Number of atoms"):
        data.append(h5metrics[pdb].attrs['numProteinAtoms'])
    print_stats(data, metric="Number of atoms")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("Number of atoms")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "num_atoms.png"), dpi=600)

def plot_numResidues(h5metrics, output_dir):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Number of residues"):
        data.append(h5metrics[pdb].attrs['numResidues'])
    print_stats(data, metric="Number of residues")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("Number of residues")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "num_residues.png"), dpi=600)

def plot_rmsd_dist(h5metrics, output_dir, rmsdcutoff=6, yscale="linear"):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="RMSD"):
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                data.append(h5metrics[pdb][temp][repl]['rmsd'][-1])
    print_stats(data, metric="RMSD")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("RMSD (nm)")
    plt.ylabel("Counts")
    if yscale == "log":
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opj(output_dir, f"rmsd{'_log' if yscale == 'log' else ''}.png"), dpi=600)

def plot_rmsf_dist(h5metrics, output_dir, yscale="linear", temp_oi=None):
    data = []
    temperatures = ['320', '348', '379', '413', '450'] if temp_oi is None else [temp_oi]
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="RMSF"):
        for temp in temperatures:
            for repl in h5metrics[pdb][temp].keys():
                data.extend(h5metrics[pdb][temp][repl]['rmsf'][:])
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("RMSF (nm)")
    plt.ylabel("Counts")
    if yscale == "log":
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opj(output_dir, f"rmsf{'_log' if yscale == 'log' else ''}.png"), dpi=200)
    
def plot_numRes_lenTraj(h5metrics, output_dir):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Number of residues vs Trajectory length"):
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                data.append([h5metrics[pdb][temp][repl].attrs['numResidues'], h5metrics[pdb][temp][repl].attrs['numFrames']])
    
    data = np.array(data)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Number of residues")
    plt.ylabel("Trajectory length (ns)")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "numRes_trajLen.png"), dpi=600)

def plot_GyrRad_SecondaryStruc(h5data, output_dir, numSamples=6, shared_axes=False):
    ''' Select 6 random keys from the h5 file and plot the gyration radius and secondary structure
    plot1: 6 different pdbs, same temperature and replica
    plot2: 6 different temperatures, same pdb and replica
    '''
    np.random.seed(42)
    numFrames = 450 
    deltaFrames = 50 # it's an arbitrary number, in order to not have too different lengths of the trajectories
    domain_figures = '/shared/antoniom/buildCATHDataset/analysis/figures/domain_figure3'
    ## cbar common settings ##
    cbar_kws = {"orientation":"vertical", "shrink":0.8, "aspect":40}
    cbar_label = "Simulation time (ns)"
    cbar_ticks = [0, 250, 500]
    cbar_ticklabels = [0, 250, 500]
    
    # HERE WE PLOT A GRID OF SAMPLES, LOWEST TEMP AND ONE REPLICA (SAME FOR ALL SAMPLES)
    samples = np.random.choice(list(h5data.keys()), numSamples, replace=False)
    ncols = 3
    nrows = math.ceil(numSamples / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.3, wspace=0.2)

    for i, sample in tqdm(enumerate(samples), total=numSamples, desc="GyrRad_solidSS"):
        if nrows == 1 or ncols == 1:  # Single row or column case
            ax = axs.flatten()[i] if numSamples != 1 else axs
        else:
            ax = axs[i // ncols, i % ncols]

        temp = '320'
        repl = '1'
        temp_repl_group = h5data[sample][temp][repl]
        
        # be sure that the trajectory has at least numFrames inside the range numFrames-deltaFrames,numFrames+deltaFrames 
        if  temp_repl_group.attrs['numFrames'] >= numFrames-deltaFrames and temp_repl_group.attrs['numFrames'] <= numFrames+deltaFrames:
            pass
        else:
            print(f"Sample {sample} has not the right number of frames({temp_repl_group.attrs['numFrames']})")
            wrong_sample = sample
            while not (numFrames - deltaFrames < temp_repl_group.attrs['numFrames'] < numFrames + deltaFrames):
                sample = np.random.choice(list(h5data.keys()), 1, replace=False)[0]
                temp_repl_group = h5data[sample][temp][repl]
            print(f"Sample {wrong_sample} has been replaced by {sample}")
            # add the sample to the list of samples and replace the one that was not good
            samples[i] = sample
            
        ss = temp_repl_group['solid_secondary_structure'][:]
        gr = temp_repl_group['gyration_radius'][:]
        pngpath = opj(domain_figures, f"{sample}.png")
        # Normalized color mapping
        norm = mcolors.Normalize(vmin=0, vmax=numFrames+deltaFrames)
        cmap = plt.get_cmap('viridis')
        
        # Scatter plot
        scatter = ax.scatter(ss, gr, c=range(len(ss)), cmap=cmap, norm=norm, s=5, zorder=2)
        if i != len(samples)-1:
            loc = 'upper right'
        else:
            loc = 'lower left'
        # Add the image of the domain as an inset
        axins = inset_axes(ax, width="40%", height="40%", loc=loc)
        img = plt.imread(pngpath)
        axins.imshow(img)
        axins.axis('off')  # Hide the axis of the inset        
        
        ax.set_title(f"{sample}")
    
    xmin = min([ax.get_xlim()[0] for ax in axs.flatten()])
    xmax = max([ax.get_xlim()[1] for ax in axs.flatten()])
    ymin = min([ax.get_ylim()[0] for ax in axs.flatten()])
    ymax = max([ax.get_ylim()[1] for ax in axs.flatten()])
    if shared_axes:
        for axi, ax in enumerate(axs.flatten()):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)
            if axi % ncols == 0:
                ax.set_ylabel("Gyration radius (nm)")
                ax.set_ylim(ymin-0.1, ymax+0.1)   
            if axi // ncols == nrows-1:
                ax.set_xlabel("Fraction of α+β structure")
                ax.set_xlim(xmin-0.1, xmax+0.1)

    else:
        for ax in axs.flatten():
            ax.set_xlim(xmin-0.1, xmax+0.1)
            ax.set_ylim(ymin-0.1, ymax+0.1)   
            ax.set_xlabel("Fraction of α+β structure")
            ax.set_ylabel("Gyration radius (nm)")          
            
    # Colorbar with dedicated space
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])  # x, y, width, height
    cbar = fig.colorbar(scatter, cax=cbar_ax, **cbar_kws)
    cbar.set_label(cbar_label)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    plt.savefig(opj(output_dir, "GyrRad_solidSS_A_domainImages_ShareAxs.png"), dpi=600)
    return
    
    ## HERE WE PLOT A GRID FOR THE SAME SAMPLE BUT DIFFERENT TEMPERATURES (SAME REPLICA) ## 
    sample_i = '5sicI00' #np.random.choice(samples, 1, replace=False)[0]
    ncols = 3
    nrows = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axs = axs.flatten()
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.3, wspace=0.35)
    for i, temp in tqdm(enumerate(list(h5data[sample_i].keys())), total=len(h5data[sample_i].keys()), desc="GyrRad_solidSS"):
        ax = axs[i]
        temp_repl_group = h5data[sample_i][temp][repl]
                        
        ss = temp_repl_group['solid_secondary_structure'][:]
        gr = temp_repl_group['gyration_radius'][:]
        
        # Normalized color mapping
        norm = mcolors.Normalize(vmin=0, vmax=numFrames+deltaFrames)
        cmap = plt.get_cmap('viridis')
        scatter = ax.scatter(ss, gr, c=range(len(ss)), cmap=cmap, norm=norm, s=5)
        ax.set_title(f"{temp}K")
        ax.set_xlabel("Fraction of α+β structure")
        ax.set_ylabel("Gyration radius (nm)")
    
    
    xmin = 0.28
    xmax = 0.62 
    ymin = 1.28
    ymax = 1.52
    
    # last axis instead of be empty, report the last temperature with a zoom out
    # the zoom out is created getting the xmin, xmax, ymin, ymax as the min and the max values of the other plots
    ax = axs[-1]
    ax.scatter(ss, gr, c=range(len(ss)), cmap=cmap, norm=norm, s=5)
    # add a rectangle to show the zoom out
    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel("Fraction of α+β structure")
    ax.set_ylabel("Gyration radius (nm)")
    ax.set_title(f"{temp}K (zoom out)")
    
    """ xmin = min([ax.get_xlim()[0] for ax in axs.flatten()])
    xmax = max([ax.get_xlim()[1] for ax in axs.flatten()])
    ymin = min([ax.get_ylim()[0] for ax in axs.flatten()])
    ymax = max([ax.get_ylim()[1] for ax in axs.flatten()]) """
    
    for ax in axs[:-1]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6])
        ax.set_yticks([1.3, 1.4, 1.5])   
    
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])  # x, y, width, height
    cbar = fig.colorbar(scatter, cax=cbar_ax, **cbar_kws)
    cbar.set_label(cbar_label)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    plt.savefig(opj(output_dir, f"GyrRad_solidSS_B_{sample_i}_replica{repl}.png"), dpi=600)
        
def plot_solidFraction_RMSF(h5metrics, output_dir):
    np.random.seed(2)
    samples = np.random.choice(list(h5metrics.keys()), 3, replace=False)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.35, wspace=0.3)
    #axs = axs.flatten()
    repl = '1'
    # floatMap = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, " ": 7} # 8 type differentiation
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # 3 type differentiation
    for ax_i, sample in tqdm(enumerate(samples), total=len(samples), desc="Solid fraction vs RMSF"):
        h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{sample}.h5", "r")
        for j, temp in enumerate(['320', '450']):
            encoded_dssp = h5file[sample][temp][repl]['dssp']
            dssp_decoded_float = np.zeros((encoded_dssp.shape[0], encoded_dssp.shape[1]), dtype=np.float32)
            for i in range(encoded_dssp.shape[0]):
                dssp_decoded_float[i] = [floatMap[el.decode()] for el in encoded_dssp[i]]
            solid_fraction_time = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1).mean(axis=0)
            rmsf = h5file[sample][temp][repl]['rmsf'][:]
            ax = axs[ax_i, j]
            ax.scatter(rmsf, solid_fraction_time, c=np.arange(len(rmsf)), cmap='rainbow')
            ax.set_title(f"{sample} {temp}K")
            ax.set_xlabel("RMSF (nm)")
            ax.set_ylabel("Fraction of α+β structure")
            # column zero set xlim 
            if j == 0:
                ax.set_xlim(0, 1)
            else:
                ax.set_xlim(0, 2.6)
                ax.axvline(x=1, color='grey', linestyle='--')
            ax.set_ylim(-0.1, 1.1)
        h5file.close()
    plt.tight_layout()
    plt.savefig(opj(output_dir, "solidFraction_RMSF.png"), dpi=600)

def recover_trajecoryNames_rmsd_based(h5metrics, output_dir, rmsd_oi=5.0):
    with open(opj(output_dir, f"outliers_{rmsd_oi}.txt"), "w") as f:
        for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Recover trajectories"):
            for temp in h5metrics[pdb].keys():
                for repl in h5metrics[pdb][temp].keys():
                    if h5metrics[pdb][temp][repl]['rmsd'][-1] > rmsd_oi:
                        # add to the file the pdb, temp, repl
                        f.write(f"{pdb} {temp} {repl}\n")

def get_replicas(mean_across):
    if mean_across == 'all':
        return ['0', '1', '2', '3', '4']
    
    elif isinstance(mean_across, list) and len(mean_across) == 1:
        return mean_across

    else:
        raise ValueError("The mean_across should be 'all' or a list of one element")

def plot_alpha_beta_fraction_vs_numResidues(h5metrics, output_dir, mean_across='all', temps=None):
    """ Plot the fraction of alpha+beta structure vs the number of residues in the protein,
    this is done for all the proteins in the dataset. One value of the fraction of alpha+beta per domain
    and one value of the number of residues per domain. The mean of secondary structure could be computed
    across all the replicas (all) or just a replica of interest (replica id). If temps is None, one plot for 
    each temperature is generated, otherwise, the plot is generated for the temperatures in the list temps."""
           
    # floatMap = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, " ": 7} # 8 type differentiation
    
    # helices are defined as 8 because when the mean is computed across all the replicas, different numFrames are considered 
    # the max num of frames is used to build the array of decoded dssp, and then the 0 need to be removed
    floatMap = {"H": 8, "B": 1, "E": 1, "G": 8, "I": 8, "T": 2, "S": 2, " ": 2} # 3 type differentiation
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    replicas = get_replicas(mean_across)
    
    nPlots = len(temps)
    nCols = nPlots if nPlots < 3 else 3
    nRows = math.ceil(nPlots / nCols)
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 5, nRows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.35, wspace=0.3)
    axs = axs.flatten() if nPlots > 1 else [axs]
    for temp_i, temp in enumerate(temps):
        print(f"Temperature: {temp}")
        all_alpha_beta_mean = []
        all_numResidues = []
        for pdb_idx, pdb in tqdm(enumerate(h5metrics.keys()), total=len(h5metrics.keys()), desc="Solid fraction vs numResidues"):
            h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{pdb}.h5", "r")
            max_num_frames = max([h5metrics[pdb][temp][repl].attrs['numFrames'] for repl in replicas]) # used to build the array of decoded dssp
            # dssp_decoded_float shape (len(replicas), numFrames, numResidues)
            dssp_decoded_float = np.zeros((len(replicas), max_num_frames, encoded_dssp.shape[1]), dtype=np.float32)
            for repl_i, repl in enumerate(replicas):
                encoded_dssp = h5file[pdb][temp][repl]['dssp']
                for frame_i in range(encoded_dssp.shape[0]):
                    # we use the axis 0 to store the value of the fraction of alpha+beta structure per replica, 
                    dssp_decoded_float[repl_i, frame_i, :] = [floatMap[el.decode()] for el in encoded_dssp[frame_i]]
                    
            # drop zeros from the array, they are the frames that are not present in all the replicas
            dssp_decoded_float = dssp_decoded_float.flatten()
            dssp_decoded_float = dssp_decoded_float[dssp_decoded_float != 0]
            solid_fraction_time = np.logical_or(dssp_decoded_float == 8, dssp_decoded_float == 1).mean() # 8 is the value of helices, 1 is the value of beta strands
            all_alpha_beta_mean.append(solid_fraction_time)
            all_numResidues.append(h5metrics[pdb].attrs['numResidues'])
            h5file.close()
                
        # plot the data
        ax = axs[temp_i]
        ax.scatter(all_numResidues, all_alpha_beta_mean, s=2.5)
        ax.set_title(f"{temp}K")
        ax.set_xlabel("Number of residues")
        ax.set_ylabel("Fraction of α+β structure")
        # save also the single plot
        plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_numResidues_{temp}K.png"), dpi=600)
        
    axs[-1].axis('off')
    plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_numResidues_replica_{mean_across}_{len(temps)}Temps.png"), dpi=600)

def plot_time_vs_ssFraction_respectToStart(h5metrics, output_dir, mean_across='all', temps=None, skipFrames=1):
    """ Plot on x_axis the time in ns and on y_axis the fraction of alpha+beta structure respect to the start of the simulation.
    One plot for each temperature, the mean across all the replicas is computed if the mean_across is set to 'all', otherwise 
    only one replica is considered. 
    Each trajectory will be plotted with a different color, the color is assigned based on the superfamily of the protein.
    Params:
    - h5metrics: h5 file with the metrics of the dataset
    - output_dir: directory where to save the plots
    - mean_across: 'all' or ['replica id'] # the replica id is a string, and len of the list should be 1
    - temps: list of temperatures to consider, if None, all the temperatures are considered """
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    floatMap = {"H": 8, "B": 1, "E": 1, "G": 8, "I": 8, "T": 2, "S": 2, " ": 2} # 3 type differentiation
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    replicas = get_replicas(mean_across)  
    nPlots = len(temps)
    nCols = nPlots if nPlots < 3 else 3
    nRows = math.ceil(nPlots / nCols)
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 5, nRows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.35, wspace=0.35)
    axs = axs.flatten() if nPlots > 1 else [axs]
    
    # one color for each superfamily
    colors = {1: 'green', 2: 'blue', 3: 'red', 4: 'yellow'}
    superfamiliy_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structures'}
    for temp_i, temp in enumerate(temps):
        print(f"Temperature: {temp}")
        all_alpha_beta_mean = []
        super_families = []
        for pdb_idx, pdb in tqdm(enumerate(h5metrics.keys()), total=len(h5metrics.keys()), desc="Solid fraction vs numResidues"):
            h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{pdb}.h5", "r")
            numResidues = h5metrics[pdb].attrs['numResidues']
            max_num_frames = max([h5metrics[pdb][temp][repl].attrs['numFrames'] for repl in replicas]) # used to build the array of decoded dssp
            # dssp_decoded_float shape (len(replicas), numFrames, numResidues)
            dssp_decoded_float = np.zeros((len(replicas), max_num_frames, numResidues), dtype=np.float32)
            for repl_i, repl in enumerate(replicas):
                encoded_dssp = h5file[pdb][temp][repl]['dssp']
                for frame_i in range(encoded_dssp.shape[0]):
                    # we use the axis 0 to store the value of the fraction of alpha+beta structure per replica, 
                    dssp_decoded_float[repl_i, frame_i, :] = [floatMap[el.decode()] for el in encoded_dssp[frame_i]]
            
            dssp_decoded_float[dssp_decoded_float == 0] = np.nan     # replace the zeros with Nan
            solid_fraction = np.logical_or(dssp_decoded_float == 8, dssp_decoded_float == 1) # return a boolean array
            mean_across_time = np.nanmean(solid_fraction, axis=2).mean(axis=0) # mean across the residues and the replicas, shape (numFrames,)
            assert not np.isnan(mean_across_time).any(), f"NaN values in the mean_across_time for {pdb} {temp}K"
            
            # normalize the fraction of alpha+beta respect to the start of the simulation
            normalized_ss_time = np.divide(mean_across_time[1:], mean_across_time[0])
            all_alpha_beta_mean.append(normalized_ss_time)
            super_families.append(int(super_family_json[pdb]['superfamily_id'].split(".")[0]))   
            h5file.close()
            if pdb_idx == 200:
                break        
        for i in range(len(all_alpha_beta_mean)):
            axs[temp_i].plot(np.arange(0, len(all_alpha_beta_mean[i]), 1)[::skipFrames], all_alpha_beta_mean[i][::skipFrames], alpha=0.5, color=colors[super_families[i]])
        axs[temp_i].set_title(f"{temp}K")
        axs[temp_i].set_xlabel("Time (ns)")
        axs[temp_i].set_ylabel("Fraction of α+β structure")
        axs[temp_i].set_xlim(0, 500)
        #plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_time_{mean_across}_{len(temps)}Temps.png"), dpi=600)
    
    # the upper ylim is the max value across all the plots
    ymax = max([ax.get_ylim()[1] for ax in axs]) + 0.1
    for ax in axs:
        ax.set_ylim(-0.1, ymax)
        
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=superfamiliy_labels[k], markerfacecolor=v, markersize=10) for k, v in colors.items()]
    if nPlots > 3:
        axs[-1].axis('off')
        axs[-1].legend(handles=handles, loc='upper right', title='Superfamily')
    elif nPlots ==1:
        axs[-1].legend(handles=handles, loc='upper right', title='Superfamily')
    else:
        for ax in axs:
            ax.legend(handles=handles, loc='upper right', title='Superfamily')
    
    plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_time_{mean_across}_{len(temps)}Temps.png"), dpi=600)

def check_values(ssfraction_in_time, threshold=0.5):
    """ Check if the values are changing in the ssfraction_in_time array, if the difference between the values is less than the threshold
    the function returns False, otherwise True and the index of the first value that is different from the previous one."""
    for i in range(1, len(ssfraction_in_time)):
        if abs(ssfraction_in_time[i] - ssfraction_in_time[i-1]) > threshold:
            return True, i
    return False, None
      
def plot_time_vs_ssFraction_respectToSuperfamily(h5metrics, output_dir, mean_across='all', temps=None, skipFrames=1, num_pdb=None):
    """" Plot the fraction of alpha+beta structure vs the time in ns, the fraction is computed respect to the start of the simulation.
    Params:
    - h5metrics: h5 file with the metrics of the dataset
    - output_dir: directory where to save the plots
    - mean_across: 'all' or ['replica id'] # the replica id is a string, and len of the list should be 1
    - temps: list of temperatures to consider, if None, all the temperatures are considered
    - skipFrames: number of frames to skip in the plot
    - num_pdb: number of pdb to consider per superfamily, if None all the pdb are considered"""
    
    np.random.seed(42)
    # Load superfamily information from JSON
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # DSSP to float map for alpha and beta structures

    # Default temperatures if not provided
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    
    # Get the replica settings from function argument
    replicas = get_replicas(mean_across)
    
    # Determine number of plots based on the number of superfamilies
    superfamilies = sorted({int(super_family_json[pdb]['superfamily_id'].split(".")[0]) for pdb in h5metrics.keys() if pdb in super_family_json.keys()})
    nRows = len(temps)
    nCols = len(superfamilies)

    # Setup figure and axes
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 6, nRows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.4, wspace=0.4)

    # Color map and labels for superfamilies
    colors = {1: 'green', 2: 'blue', 3: 'red', 4: 'purple'}
    superfamiliy_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structures'}

    # shuffle the pdb list if num_pdb is not None
    pdb_list = list(h5metrics.keys()) if num_pdb is None else np.random.choice(list(h5metrics.keys()), len(h5metrics.keys()), replace=False)
    # Iterate over temperatures and superfamilies
    for row, temp in enumerate(temps):
        print(f"Temperature: {temp}")
        for col, sf in enumerate(superfamilies):
            ax = axs[row, col]
            accepted_superfamilies_domains = 0
            for pdb in tqdm(pdb_list, total=len(pdb_list), desc=f"Solid Fraction vs Time {superfamiliy_labels[sf]}"):
                if num_pdb is not None and accepted_superfamilies_domains >= num_pdb:
                    break  # Exit loop if reached the specified number of PDBs for the superfamily
                super_family_id = int(super_family_json[pdb]['superfamily_id'].split(".")[0])
                if super_family_id != sf:
                    continue  # Skip non-matching superfamilies
                
                accepted_superfamilies_domains += 1

                # Process H5 files to get the dssp dataset
                h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{pdb}.h5", "r")
                numResidues = h5metrics[pdb].attrs['numResidues']
                for repl in replicas:
                    encoded_dssp = h5file[pdb][temp][repl]['dssp']
                    assert encoded_dssp.shape[1] == numResidues, f"Number of residues mismatch for {pdb} {temp}K"
                    dssp_decoded_float = np.zeros((encoded_dssp.shape[0], encoded_dssp.shape[1]), dtype=np.float32)
                    for frame_i in range(encoded_dssp.shape[0]):
                        dssp_decoded_float[frame_i, :] = [floatMap[el.decode()] for el in encoded_dssp[frame_i]]
                    
                    solid_fraction = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1) # return a boolean array
                    mean_across_time = np.nanmean(solid_fraction, axis=1) # mean across the residues, shape (numFrames,)
                    assert not np.isnan(mean_across_time).any(), f"NaN values in the mean_across_time for {pdb} {temp}K"
                    # If the first value is zero, the division will give a NaN, so we need to check this
                    if mean_across_time[0] == 0:
                        Warning(f"First value of the solid fraction is zero for {pdb} {temp}K replica {repl}, the trajectory will be skipped!")
                        continue
                    normalized_ss_time = mean_across_time / mean_across_time[0]
                    ax.plot(np.arange(0, len(normalized_ss_time), skipFrames), normalized_ss_time[::skipFrames], alpha=0.2, color=colors[sf])              
                
            # Axis labels and title
            if col == 0:
                ax.set_ylabel(f"{temp}K\n Relative fraction of α+β structure") 
            else:
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
            if row == nRows - 1:
                ax.set_xlabel("Time (ns)")
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

            ax.set_title(superfamiliy_labels[sf] if row == 0 else "")
            ax.set_xlim(0, 501)
    
    axs = axs.flatten()
    # set the ylim to be the same across all the plots
    #ymax = max([ax.get_ylim()[1] for ax in axs]) + 0.1
    for ax in axs:
        ax.set_ylim(-0.01, 2.1)
        
    
    plt.tight_layout()
    # Save the plot
    plt.savefig(opj(output_dir, f"Relative_solidFraction_vs_time_{num_pdb}Samples_4Superfamilies.png"), dpi=300)
    plt.close()

def plot_Grid_time_vs_ssFraction_respectToSuperfamily(h5metrics, output_dir, mean_across='all', temps=None, num_pdb=None):
    """ Plot on x_axis the time in ns and on y_axis the fraction of alpha+beta structure respect to the start of the simulation, 
    one plot for each temperature, the mean across all the replicas is computed if the mean_across is set to 'all', otherwise only one replica is considered.
    Each trajectory will be plotted with a different color, the color is assigned based on the superfamily of the protein.
    Params:
    - h5metrics: h5 file with the metrics of the dataset
    - output_dir: directory where to save the plots
    - mean_across: 'all' or ['replica id'] # the replica id is a string, and len of the list should be 1
    - temps: list of temperatures to consider, if None, all the temperatures are considered
    - num_pdb: number of pdb to consider, if None all the pdb are considered otherwise the the h5metrics.keys() are shuffled and for 
        the each superfamily num_pdb are considered.
    """
    
    np.random.seed(7)
    # Load superfamily information from JSON
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # DSSP to float map for alpha and beta structures

    # Default temperatures if not provided
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    
    # Get the replica settings from function argument
    replicas = get_replicas(mean_across)
    
    # Determine number of plots based on the number of superfamilies
    superfamilies = sorted({int(super_family_json[pdb]['superfamily_id'].split(".")[0]) for pdb in h5metrics.keys() if pdb in super_family_json.keys()})
    nRows = len(temps)
    nCols = len(superfamilies)

    # Setup figure and axes
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 6, nRows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.4, wspace=0.4)

    superfamiliy_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structure'}

    pdb_list = list(h5metrics.keys()) if num_pdb is None else np.random.choice(list(h5metrics.keys()), len(h5metrics.keys()), replace=False)
    
    # Iterate over temperatures and superfamilies
    for row, temp in enumerate(temps):
        print(f"Temperature: {temp}")
        for col, sf in enumerate(superfamilies):
            ax = axs[row, col] 
            # initialize the arrays to store the data for the 2D histogram
            time_points = []
            all_alpha_beta = []
            # Iterate over PDB entries
            accepted_superfamilies_domains = 0
            for pdb in tqdm(pdb_list, total=len(pdb_list), desc=f"Solid Fraction vs Time {superfamiliy_labels[sf]}"):
                if num_pdb is not None and accepted_superfamilies_domains >= num_pdb:
                    break  # Exit loop if reached the specified number of PDBs for the superfamily
                super_family_id = int(super_family_json[pdb]['superfamily_id'].split(".")[0])
                if super_family_id != sf:
                    continue  # Skip non-matching superfamilies
                
                accepted_superfamilies_domains += 1
                # Process H5 files to get the dssp dataset
                h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{pdb}.h5", "r")
                numResidues = h5metrics[pdb].attrs['numResidues']
                for repl in replicas:
                    encoded_dssp = h5file[pdb][temp][repl]['dssp']
                    assert encoded_dssp.shape[1] == numResidues, f"Number of residues mismatch for {pdb} {temp}K"
                    dssp_decoded_float = np.zeros((encoded_dssp.shape[0], encoded_dssp.shape[1]), dtype=np.float32)
                    for frame_i in range(encoded_dssp.shape[0]):
                        # we use the axis 0 to store the value of the fraction of alpha+beta structure per replica, 
                        dssp_decoded_float[frame_i, :] = [floatMap[el.decode()] for el in encoded_dssp[frame_i]]
                
                    solid_fraction = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1) # return a boolean array
                    mean_across_time = np.mean(solid_fraction, axis=1) # mean across the residues, shape (numFrames,)
                    assert not np.isnan(mean_across_time).any(), f"NaN values in the mean_across_time for {pdb} {temp}K"
                    
                    if mean_across_time[0] == 0:
                        Warning(f"First value of the solid fraction is zero for {pdb} {temp}K replica {repl}, the trajectory will be skipped!")
                        continue
                    
                    # Compute the mean across time and normalize
                    normalized_ss_time = mean_across_time / mean_across_time[0] # shape (numFrames,)
                    
                    # Extend the time_points and all_alpha_beta arrays to store the data for the 2D histogram
                    time_points.extend(np.arange(0, len(normalized_ss_time), 1))
                    all_alpha_beta.extend(normalized_ss_time)

                h5file.close()
            
            print(f"Number of proteins in {superfamiliy_labels[sf]} superfamily : {accepted_superfamilies_domains}")         
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(time_points, all_alpha_beta, bins=50, range=[[0, 450], [0, 1.5]], density=True)
            ax.imshow(hist.T, origin='lower', aspect='auto', extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap='viridis')

            # Axis labels and title
            if col == 0:
                ax.set_ylabel(f"{temp}K\nRel. frac. of α+β structure") 
            else:
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
            if row == nRows - 1:
                ax.set_xlabel("Time (ns)")
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

            ax.set_title(superfamiliy_labels[sf] if row == 0 else "")
            ax.set_xlim(0, 450)
            ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    # Save the plot
    plt.savefig(opj(output_dir, f"Grid_RelativeSolidFraction_vs_time_{num_pdb}Samples_4Superfamilies.png"), dpi=300)
    plt.close()

def get_secondary_structure_compositions(dssp):
    '''This funtcion returns the percentage composition of alpha, beta and coil in the protein.
    A special "NA" code will be assigned to each "residue" in the topology which isn"t actually 
    a protein residue (does not contain atoms with the names "CA", "N", "C", "O")
    '''
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2, 'NA': 3} 
    
    decoded_dssp = [el.decode() for el in dssp[-1]] # consider only the last frame
    float_dssp = np.array([floatMap[el] for el in decoded_dssp])
    unique, counts = np.unique(float_dssp, return_counts=True)
    numResAlpha, numResBeta, numResCoil = 0, 0, 0
    for u, c in zip(unique, counts):
        if u == 0:
            numResAlpha += c
        elif u == 1:
            numResBeta += c
        else:
            # NA or Coil
            numResCoil += c
    # percentage composition in alpha, beta and coil
    alpha_comp = (numResAlpha / np.sum(counts)) 
    beta_comp = (numResBeta / np.sum(counts)) 
    coil_comp = (numResCoil / np.sum(counts))
    
    return alpha_comp, beta_comp, coil_comp 

def plot_ternary_for_temp_and_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdb=None):
    import mpltern
    ''' Use mpltern to plot alpha, beta and coil fractions for each superfamily at each temperature considering only the last
    frame of the trajectory in a ternary plot.
    Params:
    - h5metrics: h5 file with the metrics of the dataset
    - output_dir: directory where to save the plots
    - mean_across: 'all' or ['replica id'] # the replica id is a string, and len of the list should be 1
    - temps: list of temperatures to consider, if None, all the temperatures are considered
    - num_pdb: number of pdb to consider, if None all the pdb are considered otherwise the the h5metrics.keys() are shuffled and for
        the each superfamily num_pdb are considered. '''
        
    np.random.seed(7)
    # Load superfamily information from JSON
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # DSSP to float map for alpha and beta structures

    # Default temperatures if not provided
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    
    # Get the replica settings from function argument
    replicas = get_replicas(mean_across)
    
    # Determine number of plots based on the number of superfamilies
    superfamilies = sorted({int(super_family_json[pdb]['superfamily_id'].split(".")[0]) for pdb in h5metrics.keys() if pdb in super_family_json.keys()})
    nRows = len(temps)
    nCols = len(superfamilies)

    # Setup figure and axes
    fig = plt.figure(figsize=(nCols * 5, nRows * 5))

    superfamiliy_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structure'}

    pdb_list = list(h5metrics.keys()) if num_pdb is None else np.random.choice(list(h5metrics.keys()), len(h5metrics.keys()), replace=False)
    
    # Iterate over temperatures and superfamilies
    for row, temp in enumerate(temps):
        #print(f"Temperature: {temp}")
        for col, sf in enumerate(superfamilies):
            accepted_superfamilies_domains = 0
            all_alpha = []
            all_beta = []
            all_coil = []
            
            for pdb in tqdm(pdb_list, total=len(pdb_list), desc=f"Ternary Plot for temp {temp} {superfamiliy_labels[sf]}", ):
                if num_pdb is not None and accepted_superfamilies_domains >= num_pdb:
                    break  # Exit loop if reached the specified number of PDBs for the superfamily
                super_family_id = int(super_family_json[pdb]['superfamily_id'].split(".")[0])
                if super_family_id != sf:
                    continue
                accepted_superfamilies_domains += 1
                
                # Process H5 files to get the dssp dataset
                h5file = h5py.File(f"/workspace3/mdCATH_final/mdcath_dataset_{pdb}.h5", "r")
                for repl in replicas:
                    alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(h5file[pdb][temp][repl]['dssp'])
                    all_alpha.append(alpha_comp)
                    all_beta.append(beta_comp)
                    all_coil.append(coil_comp)
                h5file.close()
                       
            
            # plot the ternary plot for the specific superfamily and temperature
            ax = plt.subplot(nRows, nCols, row * nCols + col + 1, projection='ternary')
            ax.set_tlabel('Alpha', fontsize=12)
            ax.set_llabel('Beta', fontsize=12)
            ax.set_rlabel('Coil/turn', fontsize=12)
            
            t, l, r = np.array(all_alpha), np.array(all_beta), np.array(all_coil)
            if sf == 4:
                ax.hexbin(t, l, r, bins='log', edgecolors='face', cmap='viridis', gridsize=30, linewidths=0) # few secondary structure less points
            else:                 
                ax.hexbin(t, l, r, bins='log', edgecolors='face', cmap='viridis', gridsize=50, linewidths=0)
            
            if col == 0:
                ax.annotate(f"{temp}K", xy=(0.5, 0.5), 
                            xytext=(-0.45, 0.5), 
                            fontsize=18, 
                            ha='center', 
                            va='center', 
                            xycoords='axes fraction', 
                            textcoords='axes fraction',
                            #fontweight='bold',
                            )
            if row == 0:
                ax.annotate(superfamiliy_labels[sf], 
                            xy=(0.5, 0.5), 
                            xytext=(0.5, 1.45), 
                            fontsize=18, 
                            ha='center', 
                            va='center', 
                            xycoords='axes fraction', 
                            textcoords='axes fraction', 
                            #fontweight='bold',
                            )
    plt.tight_layout()
    # Save the plot
    plt.savefig(opj(output_dir, f"ternary_plot_{(str(num_pdb) + 'Samples_') if num_pdb is not None else ''}4Superfamilies.png"), dpi=600)
    plt.close()        

def plot_maxNumNeighbors(h5metrics, output_dir, cutoff=['5A']):
    ''' Plot the maximum number of neighbors for each protein in the dataset. '''
    data_dict = {}
    for c in cutoff:
        if c not in data_dict:
            data_dict[c] = []
        for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc=f"MaxNumNeighbors {c}"):
            for temp in h5metrics[pdb].keys():
                for repl in h5metrics[pdb][temp].keys():
                    data_dict[c].append(h5metrics[pdb][temp][repl].attrs[f'max_num_neighbors_{c}'])
    
    f, axs = plt.subplots(1, len(cutoff), figsize=(len(cutoff) * 6, 5))
    for i, c in enumerate(cutoff):
        axs[i].hist(data_dict[c], bins=50, color='skyblue', edgecolor='black', linewidth=1.2)
        axs[i].set_xlabel("Max number of neighbors per replica")
        axs[i].set_title(f"Cutoff {c}")
        axs[i].set_ylabel("Counts")
        axs[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(opj(output_dir, "maxNumNeighbors.png"), dpi=600)

def plot_maxNumNeighbors_vs_numNoHAtoms(h5metrics, output_dir, cutoff=['5A', '9A']):
    """ Plot the maximum number of neighbors distribution and color the points based on the number of heavy atoms in the protein. """

    cutoff_neighbors_results = {}
    num_heavy_atoms = []
    min_heavy_atoms = 0
    max_heavy_atoms = 0
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="MaxNumNeighbors vs numNoHAtoms"):
        counter_ = 0
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                numNoHAtoms = h5metrics[pdb].attrs['numNoHAtoms']
                max_heavy_atoms = max(max_heavy_atoms, numNoHAtoms)
                min_heavy_atoms = min(min_heavy_atoms, numNoHAtoms)
                counter_ += 1
                for c in cutoff:
                    if c not in cutoff_neighbors_results:
                        cutoff_neighbors_results[c] = []
                    cutoff_neighbors_results[c].append(h5metrics[pdb][temp][repl].attrs[f'max_num_neighbors_{c}'])
                    
        num_heavy_atoms.extend([numNoHAtoms]*counter_)
    
    f, axs = plt.subplots(1, len(cutoff), figsize=(len(cutoff) * 6, 5), sharey=False)
    # create a dataframe in which the pdb num of heavy atoms and the max_num_neighbors are stored in expanded form
    
    color_map = plt.cm.viridis
    norm = plt.Normalize(vmin=min_heavy_atoms, vmax=max_heavy_atoms)        
    for i, c in enumerate(cutoff):
        #axs[i].scatter(num_heavy_atoms, cutoff_neighbors_results[c], c=num_heavy_atoms, cmap=color_map, norm=norm, s=0.8)
        axs[i].scatter(num_heavy_atoms, cutoff_neighbors_results[c], color='dodgerblue', s=0.8)
        axs[i].set_ylabel("Max number of neighbors per replica")
        axs[i].set_title(f"Cutoff {c}")
        axs[i].set_xlabel("Number of heavy atoms")
        if i == len(cutoff) - 1:
            pass
            # keep the ticks but not the labels
            #axs[i].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        
    # add axis for colorbar
    #sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    #f.colorbar(sm, ax=axs, orientation='vertical', label='Number of heavy atoms')
    
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.savefig(opj(output_dir, "maxNumNeighbors_vs_numNoHAtoms_nocmap.png"), dpi=600)
        
def plot_numNoHAtoms(h5metrics, output_dir):
    ''' Plot the number of heavy atoms in the protein for each protein in the dataset. '''
    numNoHAtoms = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="numNoHAtoms"):
        numNoHAtoms.append(h5metrics[pdb].attrs['numNoHAtoms'])
    
    plt.figure(figsize=(6, 5))
    plt.hist(numNoHAtoms, bins=50)
    plt.xlabel("Number of heavy atoms")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "numNoHAtoms.png"), dpi=300)

def plot_figure2_paper(h5metrics, output_dir):
    labels = ['Number of Atoms', 'Number of Residues', 'Trajectory length (ns)', 'RMSD (nm)']
    data = {label: [] for label in labels}
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Figure 2"):
        data['Number of Atoms'].append(h5metrics[pdb].attrs['numProteinAtoms'])
        data['Number of Residues'].append(h5metrics[pdb].attrs['numResidues'])
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                data['Trajectory length (ns)'].append(h5metrics[pdb][temp][repl].attrs['numFrames'])
                data['RMSD (nm)'].append(h5metrics[pdb][temp][repl]['rmsd'][-1])
                
                
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    letters = ['a', 'b', 'c', 'd']
    for i, label in enumerate(labels):
        axs[i].set_title(letters[i], loc='left', fontweight='bold')
        axs[i].hist(data[label], linewidth=1.2, bins=40, color='cornflowerblue', edgecolor='white')
        axs[i].set_xlabel(label)
        axs[i].set_ylabel("Counts")

    plt.tight_layout()
    plt.savefig(opj(output_dir, "figure2_paper.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    output_dir = "figures/"
    h5metrics = h5py.File("../all_data_unique_h5/mdcath_analysis.h5", "r")
    plt.rcParams.update({'font.size': 16,
                        'axes.labelsize': 16,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 14,
                        'ytick.labelsize': 14,
                        'legend.fontsize': 14,
                        })
    #sns.set(context="paper", style="white", font="sans-serif", font_scale=1.5, color_codes=True, rc=None, palette="muted")
    #plot_len_traj(h5metrics, output_dir)
    #plot_numAtoms(h5metrics, output_dir)
    #plot_numResidues(h5metrics, output_dir)
    #plot_rmsd_dist(h5metrics, output_dir, rmsdcutoff=6, yscale="linear")
    #plot_numRes_lenTraj(h5metrics, output_dir)
    #plot_GyrRad_SecondaryStruc(h5metrics, output_dir, numSamples=6, shared_axes=False)
    #plot_solidFraction_RMSF(h5metrics, output_dir)
    #recover_trajecoryNames_rmsd_based(h5metrics, output_dir, rmsd_oi=5.0)
    #plot_alpha_beta_fraction_vs_numResidues(h5metrics, output_dir, mean_across='all', temps=None)
    #plot_time_vs_ssFraction_respectToStart(h5metrics, output_dir, mean_across=['1'], temps=None, skipFrames=5)
    #plot_time_vs_ssFraction_respectToSuperfamily(h5metrics, output_dir, mean_across=['1'], temps=None, skipFrames=1, num_pdb=2)
    #plot_Grid_time_vs_ssFraction_respectToSuperfamily(h5metrics, output_dir, mean_across='all', temps=None, num_pdb=50)
    #plot_rmsf_dist(h5metrics, output_dir, yscale="log", temp_oi='450')
    #plot_ternary_for_temp_and_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdb=None)
    #plot_maxNumNeighbors(h5metrics, output_dir, cutoff=['5A', '9A'])
    #plot_maxNumNeighbors_vs_numNoHAtoms(h5metrics, output_dir, cutoff=['5A', '9A'])
    #plot_numNoHAtoms(h5metrics, output_dir)
    plot_figure2_paper(h5metrics, output_dir)