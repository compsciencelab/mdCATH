import os
from os.path import join as opj
from glob import glob
import logging 


class TrajectoryFileManager:
    def __init__(self, gpugridResultsPath, concatTrajPath):
        self.logger = logging.getLogger("TrajectoryFileManager")
        self.gpugridResultsPath = gpugridResultsPath
        self.concatTrajPath = concatTrajPath

    def getTrajFiles(self, pdbname, temp, repl):
        """ Get the trajectory files from the input directory, and if concatTrajPath is 
            not None, look for the concatenated trajectory file directly. Also in concaTrajPath the file are for replica
        Parameters
        ----------
        trajdir: list 
            The input directory where the trajectories are stored.
        pdbname: str
            The PDB name.
        temp: str
            The temperature.
        repl: int
            The replica number, to retrieve the corresponding trajectory file.
        concatTrajPath: str or None
            The path to the concatenated trajectory files.
        """
        basename = f"{pdbname}_{temp}_{repl}"
        if self.concatTrajPath:
            trajFiles = sorted(glob(opj(self.concatTrajPath, pdbname, f"{basename}.xtc")))
            if len(trajFiles) > 0:
                return trajFiles
        # self.logger.info(f"No concatenated trajectory files found for {pdbname} at {temp}K")

        alltrajs = []
        for trajdir_ in self.gpugridResultsPath:
            trajs = sorted(glob(opj(trajdir_, basename, f"{basename}*.xtc")))
            alltrajs.extend(trajs)

        assert len(alltrajs) > 0, "No trajectory files found"
        alltrajs = self.orderTrajFiles(alltrajs)
        return alltrajs

    def orderTrajFiles(self, trajFiles):
        """ Order the trajectory files by the traj index.
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