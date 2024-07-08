import os 
os.environ["NUMEXPR_MAX_THREADS"] = "24"
os.environ["OMP_NUM_THREADS"] = "24"
import math
import sys
import h5py
import shutil
import argparse
import logging
import tempfile
from tqdm import tqdm
import concurrent.futures
from os.path import join as opj
from glob import glob
from utils import readPDBs, save_argparse, LoadFromFile
from trajManager import TrajectoryFileManager
from molAnalyzer import molAnalyzer
from scheduler import ComputationScheduler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("builder")

def get_argparse():
    parser = argparse.ArgumentParser(description="mdCATH dataset builder", prefix_chars="--")
    
    # Add arguments with default values from the configuration
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    
    parser.add_argument('--pdblist', help='Path to the list of accepted PDBs or a list of PDBs')
    parser.add_argument('--gpugridResultsPath', type=str, help='Path to GPU grid results')
    parser.add_argument('--gpugridInputsPath', type=str, help='Path to GPU grid inputs')
    parser.add_argument('--concatTrajPath', type=str, default=None, help='Path to concatenated trajectory')
    parser.add_argument('--finaldatasetPath', type=str, default='mdcath', help='Path to the final dataset')
    parser.add_argument('--temperatures', type=list, default=["320", "348", "379", "413", "450"], help='The simulation temperatures to consider')
    parser.add_argument('--numReplicas', type=int, default=1, help='Number of replicas, available for each temperature', choices=range(0,4,1))
    parser.add_argument('--trajAttrs', type=list, default=['numFrames'], help='Trajectory attributes for each replica')
    parser.add_argument('--trajDatasets', type=list, default=['rmsd', 'gyrationRadius', 'rmsf', 'dssp'], help='Trajectory datasets for each replica')
    parser.add_argument('--pdbAttrs', type=list, default=['numProteinAtoms', 'numResidues', 'numChains'], help='PDB attributes, shared by temperatures and replicas')
    parser.add_argument('--pdbDatasets', type=list, default=['element', 'z', 'resname', 'resid', 'chain'], help='PDB datasets, shared by temperatures and replicas')
    parser.add_argument('--batchSize', type=int, default=1, help='batch size to use in the computation')
    parser.add_argument('--toRunBatches', type=int, default=None, help='Number of batches to run, if None all the batches will be run')
    parser.add_argument('--startBatch', type=int, defaul=None, help='Start batch, if None the first batch will be run')
    parser.add_argument('--endBatch', type=int, default=None, help='End batch, if None the last batch will be run')
    parser.add_argument('--maxWorkers', type=int, default=24, help='Number of workers to use in the multiprocessing')
    
    return parser

def get_args():
    parser = get_argparse()
    args = parser.parse_args()
    os.makedirs(args.finaldatasetPath, exist_ok=True)
    save_argparse(args, os.path.join(args.finaldatasetPath, "input.yaml"), exclude=["conf"])
    return args

class Payload:
    def __init__(self, scheduler, args):
        self.scheduler = scheduler
        self.args = args

    def runComputation(self, batch_idx):
        logger.info(f"Batch {batch_idx} started")
        logger.info(f"OMP_NUM_THREADS= {os.environ.get('OMP_NUM_THREADS')}")
        run(self.scheduler, self.args, batch_idx)

def run(scheduler, args, batch_idx):
    """Run the dataset generation for a specific batch.
     Parameters
    ----------
    scheduler : Scheduler object
        The scheduler object is used to get the indices of the molecules to be processed in the batch, 
        and to get the name of the file to be generated
    args: argparse.Namespace
        The arguments from the command line
    batch_idx: int
        The index of the batch to be processed
    """
    pbbIndices = scheduler.process(batch_idx)
    trajFileManager = TrajectoryFileManager(args.gpugridResultsPath, args.concatTrajPath)
    for pdb in tqdm(pbbIndices, total=len(pbbIndices), desc="reading PDBs"):    
        with tempfile.TemporaryDirectory() as temp:
            
            tmpFile = opj(temp, f"mdcath_dataset_{pdb}.h5")
            tmplogfile = tmpFile.replace(".h5", ".txt")
            
            resFile = opj(args.finaldatasetPath, pdb, f"mdcath_dataset_{pdb}.h5")
            logFile = opj(args.finaldatasetPath, pdb, f"log_{pdb}_batch{batch_idx}.txt")
            
            pdbLogger = logging.getLogger(f"builder_{pdb}")
            file_handler = logging.FileHandler(tmplogfile)
            file_handler.setLevel(logging.INFO)
            pdbLogger.addHandler(file_handler)
            pdbLogger.setLevel(logging.INFO)
                    
            pdbLogger.info(f"Starting the dataset generation for {pdb} and batch {batch_idx}")
            
            if os.path.exists(resFile):
                pdbLogger.info(f"File {resFile} already exists, removing it due to previous errors on the pdb idxs")
                os.remove(resFile)
            pdbFilePath = glob(opj(args.gpugridInputsPath, pdb, "*/*.pdb"))[0] # get structure.pdb from input folder (same for all replicas and temps)
            if not os.path.exists(pdbFilePath):
                logger.warning(f"{pdb} does not exist")
                continue
            
            with h5py.File(tmpFile, "w", libver='latest') as h5:
                
                h5.attrs["layout"] = "mdcath-only-protein-v1.0"
                pdbGroup = h5.create_group(pdb)
                os.makedirs(os.path.dirname(resFile), exist_ok=True)
                Analyzer = molAnalyzer(pdbFilePath, file_handler, os.path.dirname(resFile))
                Analyzer.computeProperties()
                for temp in args.temperatures:
                    pdbTempGroup = pdbGroup.create_group(temp)
                    pdbLogger.info(f"Starting the analysis for {pdb} at {temp}K")
                    pdbLogger.info(f"---------------------------------------------------")
                    for repl in range(args.numReplicas):
                        pdbLogger.info(f"## REPLICA {repl} ##")
                        pdbTempReplGroup = pdbTempGroup.create_group(str(repl))
                        try:
                            trajFiles = trajFileManager.getTrajFiles(pdb, temp, repl)
                            dcdFiles = [f.replace("9.xtc", "8.vel.dcd") for f in trajFiles]
                        except AssertionError as e:
                            pdbLogger.error(e)
                            continue

                        Analyzer.readXTC(trajFiles, batch_idx)
                        Analyzer.readDCD(dcdFiles, batch_idx)
                        Analyzer.trajAnalysis()
                        
                        if not hasattr(Analyzer, "forces") or not hasattr(Analyzer, "coords"):
                            pdbLogger.error(f"forces or traj not found for {pdb} {temp} {repl}")
                            continue
                        
                        # write the data to the h5 file for the replica
                        Analyzer.write_toH5(molGroup=None, replicaGroup=pdbTempReplGroup, attrs=args.trajAttrs, datasets=args.trajDatasets)
                        
                # If no replica was found, skip the molecule. The molecule will be written to the h5 file only if it has at least one replica at one temperature
                if not hasattr(Analyzer, "molAttrs"):
                    pdbLogger.error(f"molAttrs not found for {pdb} and batch {batch_idx}")
                    continue
                
                # write the data to the h5 file for the molecule 
                Analyzer.write_toH5(molGroup=pdbGroup, replicaGroup=None, attrs=args.pdbAttrs, datasets=args.pdbDatasets)  
            
            os.makedirs(opj(args.finaldatasetPath, pdb), exist_ok=True)
            shutil.move(tmpFile, resFile)
            pdbLogger.info(f"\n{pdb} batch {batch_idx} completed successfully added to mdCATH dataset: {args.finaldatasetPath}")
            shutil.move(tmplogfile, logFile)

def launch():
    args = get_args()
   
    acceptedPDBs = readPDBs(args.pdblist) if args.pdblist else None
    if acceptedPDBs is None:
        logger.error("Please provide a list of accepted PDBs which will be used to generate the dataset.")
        sys.exit(1) 
    
    logger.info(f"numAccepetedPDBs: {len(acceptedPDBs)}")
    
    # Get a number of batches
    numBatches = int(math.ceil(len(acceptedPDBs) / args.batchSize))
    logger.info(f"Batch size: {args.batchSize}")
    logger.info(f"Number of total batches: {numBatches}")
    
    if args.toRunBatches is not None and args.startBatch is not None:
        numBatches = args.toRunBatches + args.startBatch
    elif args.toRunBatches is not None:
        numBatches = args.toRunBatches
    elif args.startBatch is not None:
        pass

    # Initialize the parallelization system
    scheduler = ComputationScheduler(
        args.batchSize, args.startBatch, numBatches, acceptedPDBs
    )
    toRunBatches = scheduler.getBatches()
    logger.info(f"numBatches to run: {len(toRunBatches)}")
    logger.info(f"starting from batch: {args.startBatch}")

    payload = Payload(scheduler, args)

    with concurrent.futures.ProcessPoolExecutor(args.maxWorkers) as executor:
        try:
            results = list(
                tqdm(
                    executor.map(payload.runComputation, toRunBatches),
                    total=len(toRunBatches),
                )
            )
        except Exception as e:
            print(e)
            raise e
    # this return it's needed for the tqdm progress bar
    return results

if __name__ == "__main__":
    launch()
    logger.info("CATH-DATASET BUILDING COMPLETED!")