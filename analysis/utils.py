import os
import h5py
import math
import json
import numpy as np
from tqdm import tqdm
from os.path import join as opj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set global plotting parameters
plt.rcParams.update({'font.size': 18,
                    'axes.labelsize': 18,
                    'axes.titlesize': 20,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16,
                    'legend.fontsize': 16,
                    })

def get_stats(data, metric=""):
    # print, mean, std, min, max, median for a specific metric
    print(f"Stats for mdCATH: {metric}")
    print("--------------------------")
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

def plot_len_trajs(h5metrics, output_dir):
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Trajectory length"):
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                data.append(h5metrics[pdb][temp][repl].attrs['numFrames'])
    get_stats(data, metric="Trajectory length")
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
    get_stats(data, metric="Number of atoms")
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
    get_stats(data, metric="Number of residues")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("Number of residues")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(opj(output_dir, "num_residues.png"), dpi=600)

def plot_RMSD(h5metrics, output_dir, rmsdcutoff=10, yscale="linear"):
    # Compute RMSD distribution considering only the last frame of each trajectory
    
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="RMSD"):
        for temp in h5metrics[pdb].keys():
            for repl in h5metrics[pdb][temp].keys():
                rmsd = h5metrics[pdb][temp][repl]['rmsd'][-1]
                if rmsd > rmsdcutoff:
                    print(f"RMSD above cutoff {rmsdcutoff}: {rmsd} nm for {pdb} at {temp} K and replica {repl}")
                    continue
                data.append(h5metrics[pdb][temp][repl]['rmsd'][-1])
    get_stats(data, metric="RMSD")
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel("RMSD (nm)")
    plt.ylabel("Counts")
    if yscale == "log":
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opj(output_dir, f"rmsd{'_log' if yscale == 'log' else ''}.png"), dpi=600)

def plot_RMSF(h5metrics, output_dir, yscale="linear", temp_oi=None):
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

def plot_numRes_trajLength(h5metrics, output_dir):
    # Number of residues vs Trajectory length 
    data = []
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="numResidues vs trajLength"):
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

def plot_GyrRad_SecondaryStruc(h5data, output_dir, numSamples=6, shared_axes=False, plot_type=['A']):
    ''' Select numSamples random keys from the h5 file and plot the gyration radius and secondary structure
        plot1: numSamples different pdbs, same temperature and replica (A)
        plot2: numSamples different temperatures, same pdb and replica (B)
    '''
    np.random.seed(42)
    numFrames = 450 
    deltaFrames = 50 # it's an arbitrary number, in order to not have too different lengths of the trajectories
    # domain figures is the directory where the images of the domains are stored, these are going to be overlapped to the scatter plot
    domain_figures = '/shared/antoniom/buildCATHDataset/analysis/figures/domains_figure4'
    
    ## cbar common settings ##
    cbar_kws = {"orientation":"vertical", "shrink":0.8, "aspect":40}
    cbar_label = "Simulation time (ns)"
    cbar_ticks = [0, 250, 500]
    cbar_ticklabels = [0, 250, 500]
    
    if 'A' in plot_type:
        # HERE WE PLOT A GRID OF SAMPLES, LOWEST TEMP AND ONE REPLICA (SAME FOR ALL SAMPLES)
        temp = '320'
        repl = '1'
        samples = np.random.choice(list(h5data.keys()), numSamples, replace=False)
        
        ncols = 3
        nrows = math.ceil(numSamples / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), sharex=shared_axes, sharey=shared_axes)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.3, wspace=0.3)

        for i, sample in tqdm(enumerate(samples), total=numSamples, desc="GyrRad_solidSS (A)"):
            if nrows == 1 or ncols == 1:  # Single row or column case
                ax = axs.flatten()[i] if numSamples != 1 else axs
            else:
                ax = axs[i // ncols, i % ncols]
                
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
            pngpath = opj(domain_figures, f"{sample}.png")
            if not os.path.exists(pngpath):
                print(f"Image {pngpath} not found")
                continue
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
                ax.set_yticks([round(el,1) for el in np.linspace(ymin+0.1, ymax-0.1, 4)])
                
        # Colorbar with dedicated space
        cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])  # x, y, width, height
        cbar = fig.colorbar(scatter, cax=cbar_ax, **cbar_kws)
        cbar.set_label(cbar_label)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        plt.savefig(opj(output_dir, f"GyrRad_solidSS_A_domainImages_{'ShareAxs' if shared_axes else ''}.png"), dpi=600)
    
    
    ## HERE WE PLOT A GRID FOR THE SAME SAMPLE BUT DIFFERENT TEMPERATURES (SAME REPLICA) ## 
    if 'B' in plot_type:
        sample_i = '5sicI00' 
        ncols = 3
        nrows = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
        axs = axs.flatten()
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.3, wspace=0.35)
        for i, temp in tqdm(enumerate(list(h5data[sample_i].keys())), total=len(h5data[sample_i].keys()), desc="GyrRad_solidSS (B)"):
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
        
        for ax in axs[:-1]:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([0.3, 0.4, 0.5, 0.6])
            ax.set_yticks([1.3, 1.4, 1.5])   
            
        axs[-1].set_yticks([round(el,1) for el in np.linspace(np.min(gr)+0.1, np.max(gr)-0.1, 3)])
        
        cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])  # x, y, width, height
        cbar = fig.colorbar(scatter, cax=cbar_ax, **cbar_kws)
        cbar.set_label(cbar_label)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        plt.savefig(opj(output_dir, f"GyrRad_solidSS_B_{sample_i}_replica{repl}.png"), dpi=600)

def get_solid_fraction(dssp, simplified=False):
    # Compute the solid fraction of α+β structure in the secondary structure wrt to the time.
    if simplified:
        floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # 3 type differentiation
    else: 
        floatMap = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, " ": 7}

    dssp_decoded_float = np.zeros((dssp.shape[0], dssp.shape[1]), dtype=np.float32) # shape (numFrames, numResidues)
    for i in range(dssp.shape[0]):
                dssp_decoded_float[i] = [floatMap[el.decode()] for el in dssp[i]]
    solid_fraction_time = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1) # shape (numFrames, numResidues)
    return solid_fraction_time
    
def plot_solidFraction_RMSF(h5metrics, output_dir, numSamples=3, simplified=False, repl='1'):
    """ Solid fraction vs RMSF for N random samples at 320K and 450K.
    Solid fraction is defined as the fraction of α+β structure in the secondary structure.
    simplified: if True, the solid fraction is simplified to 3 types: α, β, and other (dssp based)
    """    
    np.random.seed(2)
    samples = np.random.choice(list(h5metrics.keys()), numSamples, replace=False)
    mdcath_dir = "/workspace8/antoniom/mdcath_htmd"
    ncols = 2 # 2 temperatures
    nrows = numSamples
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.35, wspace=0.3)
    
    for ax_i, sample in tqdm(enumerate(samples), total=len(samples), desc="Solid fraction vs RMSF"):
        with h5py.File(opj(mdcath_dir, sample, f"mdcath_dataset_{sample}.h5"), "r") as h5file: 
            for j, temp in enumerate(['320', '450']):
                solid_fraction_time = get_solid_fraction(h5file[sample][temp][repl]['dssp'], simplified=simplified) # shape (numFrames, numResidues)
                solid_fraction_time = solid_fraction_time.mean(axis=0) # mean across the frames
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

    plt.tight_layout()
    plt.savefig(opj(output_dir, f"solidFraction_RMSF_{'simplified' if simplified else ''}.png"), dpi=600)

def get_replicas(mean_across):
    if mean_across == 'all':
        return ['0', '1', '2', '3', '4']
    elif isinstance(mean_across, list):
        return mean_across
    elif isinstance(mean_across, int) or isinstance(mean_across, str):
        return [str(mean_across)]
    else:
        raise ValueError("The mean_across should be 'all' or a list of one element")

def get_solid_fraction_extended(h5group, replicas, numResidues, simplified=False):
    # Compute solid fraction of α+β structure in the secondary structure wrt to the time, across multiple replicas.
    # h5group is the group of a specific pdb at a specific temperature
    if simplified:
        floatMap = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2} # 3 type differentiation
    else:
        floatMap = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, " ": 7} # 8 type differentiation

        max_num_frames = max([h5group[repl].attrs['numFrames'] for repl in replicas]) # used to build the array of decoded dssp
        # dssp_decoded_float shape (len(replicas), numFrames, numResidues)
        dssp_decoded_float = np.zeros((len(replicas), max_num_frames, numResidues), dtype=np.float32)
        for repl_i, repl in enumerate(replicas):
            encoded_dssp = h5group[repl]['dssp']
            for frame_i in range(encoded_dssp.shape[0]):
                # we use the axis 0 to store the value of the fraction of alpha+beta structure per replica, 
                dssp_decoded_float[repl_i, frame_i, :] = [floatMap[el.decode()] for el in encoded_dssp[frame_i]]
        
        return dssp_decoded_float

def plot_solidFraction_vs_numResidues(h5metrics, output_dir, mean_across='all', temps=None, simplified=False):
    """ Plot the fraction of alpha+beta structure vs the number of residues in the protein,
    this is done for all the proteins in the dataset. One value of the fraction of alpha+beta per domain
    and one value of the number of residues per domain. The mean of secondary structure could be computed
    across all the replicas (all) or just a replica of interest (replica id). If temps is None, one plot for 
    each temperature is generated, otherwise, the plot is generated for the temperatures in the list temps."""
    
    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    replicas = get_replicas(mean_across)
    mdcath_dir = "/workspace8/antoniom/mdcath_htmd"

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
        for temp_i, temp in enumerate(temps):
            for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Solid fraction vs numResidues"):
                numResidues = h5metrics[pdb].attrs['numResidues']
                with h5py.File(opj(mdcath_dir, f"mdcath_dataset_{pdb}.h5"), "r") as h5file:
                    # dssp_decoded_float of shape (len(replicas), maxNumFrames, numResidues)
                    dssp_decoded_float = get_solid_fraction_extended(h5file[pdb][temp], replicas, numResidues, simplified=simplified)
                    # drop zeros from the array, they are the frames that are not present in all the replicas
                    dssp_decoded_float = dssp_decoded_float.flatten()
                    dssp_decoded_float = dssp_decoded_float[dssp_decoded_float != 0]
                            
                    solid_fraction_time = np.logical_or(dssp_decoded_float == 8, dssp_decoded_float == 1).mean() # 8 is the value of helices, 1 is the value of beta strands
                    all_alpha_beta_mean.append(solid_fraction_time)
                    all_numResidues.append(numResidues)
                
            # plot the scatter plot for the specific temperature
            ax = axs[temp_i]
            ax.scatter(all_numResidues, all_alpha_beta_mean, s=2.5)
            ax.set_title(f"{temp}K")
            ax.set_xlabel("Number of residues")
            ax.set_ylabel("Fraction of α+β structure")
            # save also the single plot
            plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_numResidues_{temp}K.png"), dpi=600)
        
    axs[-1].axis('off')
    plt.savefig(opj(output_dir, f"all_dataset_plots_studycase/solidFraction_vs_numResidues_replica_{mean_across}_{len(temps)}Temps.png"), dpi=600)

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

def plot_heatmap_ss_time_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdbs=None, simplified=False):
    """ Plot on x_axis the time in ns and on y_axis the fraction of alpha+beta structure respect to the start of the simulation.
    Rows are the temperatures and columns are the superfamilies. The relative solid fraction (RSF) is computed as the fraction of 
    α+β structure in the secondary structure. [figure 8 of the paper]
    
    Params:
    ------------
    - h5metrics: 
        h5 file with the metrics of the dataset
    - output_dir: 
        directory where to save the plots
    - mean_across: 
        replica to consider, if 'all' all the replicas are considered
    - temps: 
        temperatures to consider, if None all the temperatures are considered
    - num_pdbs:
        number of pdbs to consider per superfamily   
    - simplified:
        if True, the solid fraction is simplified to 3 types: α, β, and other (dssp based)
    """
    np.random.seed(7)
    superfamily_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structures'}
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    mdcath_dir = "/workspace8/antoniom/mdcath_htmd"

    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    replicas = get_replicas(mean_across)
    # Determine number of columns based on the number of superfamilies considered 
    superfamilies = sorted({int(super_family_json[pdb]['superfamily_id'].split(".")[0]) for pdb in h5metrics.keys() if pdb in super_family_json.keys()})
    
    # In order to avoid bias, we shuffle the list of pdbs if a subset is requested
    pdb_list = list(h5metrics.keys()) if num_pdbs is None else np.random.choice(list(h5metrics.keys()), len(h5metrics.keys()), replace=False)
    
    nRows = len(temps)
    nCols = len(superfamilies)
   
    # Setup figure and axes
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 6, nRows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.4, wspace=0.4)

    
    # Iterate over temperatures and superfamilies
    for row, temp in enumerate(temps):
        print(f"Temperature: {temp}")
        for col, sf in enumerate(superfamilies):
            ax = axs[row, col] 
            
            # initialize the arrays to store the data for the 2D histogram (heat-map)
            time_points = []
            all_alpha_beta = []
            accepted_superfamilies_domains = 0
            
            for pdb in tqdm(pdb_list, total=len(pdb_list), desc=f"Solid Fraction vs Time {superfamily_labels[sf]}"):
                if num_pdbs is not None and accepted_superfamilies_domains >= num_pdbs:
                    break  # Exit loop if reached the specified number of PDBs for the superfamily
                super_family_id = int(super_family_json[pdb]['superfamily_id'].split(".")[0])
                if super_family_id != sf:
                    continue  # Skip non-matching superfamilies
                
                accepted_superfamilies_domains += 1
                
                with h5py.File(opj(mdcath_dir, pdb, f"mdcath_dataset_{pdb}.h5"), "r") as h5file:
                    numResidues = h5metrics[pdb].attrs['numResidues']
                    for repl in replicas:
                        dssp = h5file[pdb][temp][repl]['dssp'] # shape (numFrames, numResidues)
                        assert dssp.shape[1] == numResidues, f"Number of residues mismatch for {pdb} {temp}K {repl}"
                        
                        solid_fraction = get_solid_fraction(h5file[pdb][temp][repl]['dssp'], simplified=simplified)
                        mean_across_time = np.mean(solid_fraction, axis=1) # mean across the residues, shape (numFrames,)
                        assert not np.isnan(mean_across_time).any(), f"NaN values in the mean_across_time for {pdb} {temp}K {repl}"
                        
                        if mean_across_time[0] == 0:
                            Warning(f"First value of the solid fraction is zero for {pdb} {temp}K replica {repl}, the trajectory will be skipped!")
                            continue
                        
                        normalized_ss_time = mean_across_time / mean_across_time[0] # shape (numFrames,)
                        time_points.extend(np.arange(0, len(normalized_ss_time), 1))
                        all_alpha_beta.extend(normalized_ss_time)
            
            print(f"Number of domains in {superfamily_labels[sf]} superfamily : {accepted_superfamilies_domains}")         
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(time_points, all_alpha_beta, bins=50, range=[[0, 450], [0, 1.5]], density=True)
            ax.imshow(hist.T, origin='lower', aspect='auto', extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap='viridis')

            # Axis labels and title
            if col == 0:
                ax.set_ylabel(f"{temp}K\nRel. frac. of α+β structure", fontsize=20) 
            else:
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
            if row == nRows - 1:
                ax.set_xlabel("Time (ns)", fontsize=20)
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

            ax.set_title(superfamily_labels[sf] if row == 0 else "", fontsize=21)
            ax.set_xlim(0, 450)
            ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig(opj(output_dir, f"HeatMap_RSF_vs_TIME_{num_pdbs}Samples_4Superfamilies.png"), dpi=300)

def plot_ternary_superfamilies(h5metrics, output_dir, mean_across='all', temps=None, num_pdbs=None):
    import mpltern
    
    ''' Use mpltern to plot alpha, beta and coil fractions for each superfamily at each temperature considering only the last
    frame of the trajectory in a ternary plot.
    Params:
    - h5metrics: 
        h5 file with the metrics of the dataset
    - output_dir: 
        directory where to save the plots
    - mean_across: 
        replica to consider, if 'all' all the replicas are considered
    - temps: 
        temperatures to consider, if None all the temperatures are considered
    - num_pdbs:
        number of pdbs to consider per superfamily   
    - simplified:
        if True, the solid fraction is simplified to 3 types: α, β, and other (dssp based)
    '''
    
    np.random.seed(7)
    superfamily_labels = {1:'Mainly Alpha', 2:'Mainly Beta', 3:'Mixed Alpha-Beta', 4:'Few Secondary Structures'}
    super_family_json = json.load(open("/shared/antoniom/buildCATHDataset/support/cath_info.json", "r"))
    mdcath_dir = "/workspace8/antoniom/mdcath_htmd"

    temps = ['320', '348', '379', '413', '450'] if temps is None else temps
    replicas = get_replicas(mean_across)
    superfamilies = sorted({int(super_family_json[pdb]['superfamily_id'].split(".")[0]) for pdb in h5metrics.keys() if pdb in super_family_json.keys()})
    
    # In order to avoid bias, we shuffle the list of pdbs if a subset is requested
    pdb_list = list(h5metrics.keys()) if num_pdbs is None else np.random.choice(list(h5metrics.keys()), len(h5metrics.keys()), replace=False)
    
    nRows = len(temps)
    nCols = len(superfamilies)

    # Setup figure and axes
    fig = plt.figure(figsize=(nCols * 5, nRows * 5))
    
    for row, temp in enumerate(temps):
        for col, sf in enumerate(superfamilies):
            accepted_superfamilies_domains = 0
            all_alpha = []
            all_beta = []
            all_coil = []
            
            for pdb in tqdm(pdb_list, total=len(pdb_list), desc=f"Ternary Plot for temp {temp} {superfamily_labels[sf]}", ):
                if num_pdbs is not None and accepted_superfamilies_domains >= num_pdbs:
                    break  # Exit loop if reached the specified number of PDBs for the superfamily
                super_family_id = int(super_family_json[pdb]['superfamily_id'].split(".")[0])
                if super_family_id != sf:
                    continue
                accepted_superfamilies_domains += 1
                
                with h5py.File(opj(mdcath_dir, pdb, f"mdcath_dataset_{pdb}.h5"), "r") as h5file:
                    for repl in replicas:
                        alpha_comp, beta_comp, coil_comp = get_secondary_structure_compositions(h5file[pdb][temp][repl]['dssp'])
                        all_alpha.append(alpha_comp)
                        all_beta.append(beta_comp)
                        all_coil.append(coil_comp)
                        
            # ternary plot for the specific superfamily and temperature
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
                ax.annotate(superfamily_labels[sf], 
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
    plt.savefig(opj(output_dir, f"ternary_plot_{(str(num_pdbs) + 'Samples_') if num_pdbs is not None else ''}4Superfamilies.png"), dpi=600)
    plt.close()        

def plot_combine_metrics(h5metrics, output_dir):
    # Figure 3 of the paper
    labels = ['Number of Atoms', 'Number of Residues', 'Trajectory length (ns)', 'RMSD (nm)']
    data = {label: [] for label in labels}
    for pdb in tqdm(h5metrics.keys(), total=len(h5metrics.keys()), desc="Figure 3"):
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
        axs[i].hist(data[label], linewidth=1.2, bins=40, color='cornflowerblue', edgecolor='black')
        axs[i].set_xlabel(label)
        axs[i].set_ylabel("Counts")

    plt.tight_layout()
    plt.savefig(opj(output_dir, "combined_metrics.png"), dpi=300)
    plt.close()

def plot_maxNumNeighbors(h5metrics, output_dir, cutoff=['5A']):
    ''' Plot the maximum number of neighbors for each domain in the dataset. '''
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

def scatterplot_maxNumNeighbors_numNoHAtoms(h5metrics, output_dir, cutoff=['5A', '9A']):
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

    for i, c in enumerate(cutoff):
        axs[i].scatter(num_heavy_atoms, cutoff_neighbors_results[c], color='dodgerblue', s=0.8)
        axs[i].set_ylabel("Max number of neighbors per replica")
        axs[i].set_title(f"Cutoff {c}")
        axs[i].set_xlabel("Number of heavy atoms")

    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.savefig(opj(output_dir, "maxNumNeighbors_numNoHAtoms.png"), dpi=600)
        
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