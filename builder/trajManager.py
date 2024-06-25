import os
from os.path import join as opj
from glob import glob
import logging

# The exception cases are the trajectories that not have a dir in gpugrid_extend_results, so gpugrid_run_results is used
exception_cases = {"2k88A00_413_0", "3vsmA01_320_4", "3qdkA02_320_2", "3qdkA02_413_0", "4qxdB02_450_2"}

class TrajectoryFileManager:
    def __init__(self, gpugridResultsPath, concatTrajPath):
        """Initialize the TrajectoryFileManager object, which is responsible for managing the trajectory files.
        Parameters
        ----------
        gpugridResultsPath: str
            The path to the gpugrid results directory.
        concatTrajPath: str or None
            The path to the concatenated trajectory files.
        """
        self.logger = logging.getLogger("TrajectoryFileManager")
        self.gpugridResultsPath = gpugridResultsPath
        self.gpugridRunResults = "/workspace7/toni_cath/gpugrid_run_results/"
        self.concatTrajPath = concatTrajPath

    def getTrajFiles(self, pdbname, temp, repl):
        """Get the trajectory files from the input directory, and if concatTrajPath is
            not None, look for the concatenated trajectory file directly. Also in concaTrajPath the file are for replica
        Parameters
        ----------
        pdbname: str
            The PDB name.
        temp: str
            The temperature.
        repl: int
            The replica number, to retrieve the corresponding trajectory file.

        Returns
        -------
        list
            The list of trajectory files (xtc files).
        """
        basename = f"{pdbname}_{temp}_{repl}"
        if self.concatTrajPath:
            trajFiles = sorted(
                glob(opj(self.concatTrajPath, pdbname, f"{basename}.xtc"))
            )
            if len(trajFiles) > 0:
                return trajFiles
            self.logger.info(
                f"No concatenated trajectory files found for {pdbname} at {temp}K"
            )

        alltrajs = []
        if basename not in exception_cases:
            trajs = sorted(
                glob(opj(self.gpugridResultsPath, basename, f"{basename}*.xtc"))
            )
        else:
            trajs = sorted(
                glob(opj(self.gpugridRunResults, basename, f"{basename}*.xtc"))
            )
        alltrajs.extend(trajs)

        assert len(alltrajs) > 0, "No trajectory files found"
        alltrajs = self.orderTrajFiles(alltrajs)
        return alltrajs

    def orderTrajFiles(self, trajFiles):
        """Order the trajectory files by the traj index.
        Parameters
        ----------
        trajFiles: list
            The list of trajectory files.
        """
        sortDict = {}
        for traj in trajFiles:
            filename = os.path.basename(traj)
            trajid = int(filename.split("-")[-3])
            sortDict[trajid] = traj
        return [sortDict[trajid] for trajid in sorted(sortDict.keys())]
