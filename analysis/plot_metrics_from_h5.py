import h5py
import math
import numpy as np
from tqdm import tqdm
import seaborn as sns
from os.path import join as opj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    
    ## cbar common settings ##
    cbar_kws = {"orientation":"vertical", "shrink":0.8, "aspect":40}
    cbar_label = "Simulation time (ns)"
    cbar_ticks = [0, 250, 500]
    cbar_ticklabels = [0, 250, 500]
    
    # HERE WE PLOT A GRID OF SAMPLES, LOWEST TEMP AND ONE REPLICA (SAME FOR ALL SAMPLES)
    samples = np.random.choice(list(h5data.keys()), numSamples, replace=False)
    ncols = 3
    nrows = math.ceil(numSamples / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.3, wspace=0.35)

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

        # Normalized color mapping
        norm = mcolors.Normalize(vmin=0, vmax=numFrames+deltaFrames)
        cmap = plt.get_cmap('viridis')
        scatter = ax.scatter(ss, gr, c=range(len(ss)), cmap=cmap, norm=norm, s=5)
        ax.set_title(f"{sample}")
    
    xmin = min([ax.get_xlim()[0] for ax in axs.flatten()])
    xmax = max([ax.get_xlim()[1] for ax in axs.flatten()])
    ymin = min([ax.get_ylim()[0] for ax in axs.flatten()])
    ymax = max([ax.get_ylim()[1] for ax in axs.flatten()])
    
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
    plt.savefig(opj(output_dir, "GyrRad_solidSS_A.png"), dpi=600)

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
        h5file = h5py.File(f"/workspace7/antoniom/mdCATH/{sample}/cath_dataset_{sample}.h5", "r")
        for j, temp in enumerate(['320', '450']):
            encoded_dssp = h5file[sample][f'sims{temp}K'][repl]['dssp']
            dssp_decoded = np.zeros((encoded_dssp.shape[0], encoded_dssp.shape[1]), dtype=object)
            dssp_decoded_float = np.zeros((encoded_dssp.shape[0], encoded_dssp.shape[1]), dtype=np.float32)
            for i in range(encoded_dssp.shape[0]):
                dssp_decoded_float[i] = [floatMap[el.decode()] for el in encoded_dssp[i]]
                dssp_decoded[i] = [el.decode() for el in encoded_dssp[i]]
            solid_fraction_time = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1).mean(axis=0)
            rmsf = h5file[sample][f'sims{temp}K'][repl]['rmsf'][:]
            ax = axs[ax_i, j]
            ax.scatter(rmsf, solid_fraction_time, c=np.arange(len(rmsf)), cmap='rainbow')
            ax.set_title(f"{sample} {temp}K")
            ax.set_xlabel("RMSF (nm)")
            ax.set_ylabel("Fraction of α+β structure")
            ax.set_xlim(0, 3.5)
            ax.set_ylim(-0.1, 1.1)
        h5file.close()
    plt.tight_layout()
    plt.savefig(opj(output_dir, "solidFraction_RMSF.png"), dpi=600)
            
if __name__ == "__main__":
    output_dir = "figures/"
    h5metrics = h5py.File("/shared/antoniom/buildCATHDataset/dataloader_h5/mdcath_analysis.h5", "r")
    sns.set(context="paper", style="white", font="sans-serif", font_scale=1.5, color_codes=True, rc=None)
    #plot_len_traj(h5metrics, output_dir)
    plot_numAtoms(h5metrics, output_dir)
    plot_numResidues(h5metrics, output_dir)
    #plot_rmsd_dist(h5metrics, output_dir, rmsdcutoff=6, yscale="linear")
    #plot_numRes_lenTraj(h5metrics, output_dir)
    #plot_GyrRad_SecondaryStruc(h5metrics, output_dir, numSamples=6, shared_axes=False)
    #plot_solidFraction_RMSF(h5metrics, output_dir)
    #recover_trajecoryNames_rmsd_based(metrics, output_dir, rmsd_oi=5.0)