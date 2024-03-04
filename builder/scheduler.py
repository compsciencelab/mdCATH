import os
import logging
import numpy as np


class ComputationScheduler:
    """Class for the parallelization of the dataset generation and computation using batches."""

    def __init__(self, batchSize, startBatch, numBatches, molecules):
        self.batchSize = batchSize
        self.numBatches = numBatches
        self.molecules = molecules
        self.startBatch = startBatch if startBatch != None else 0

    def getBatches(self):
        self.allBatches = np.arange(self.startBatch, self.numBatches)
        return self.allBatches

    def process(self, idBatch):
        assert idBatch >= 0
        assert idBatch in self.allBatches
        self.idStart = self.batchSize * idBatch
        self.idEnd = min(self.batchSize * (idBatch + 1), len(self.molecules))
        indices = self.molecules[self.idStart : self.idEnd]
        return indices

    def getFileName(self, outPath, idBatch, fileName=None):
        name = fileName if fileName != None else "cath_dataset"
        resFile = os.path.join(
            outPath,
            f"{name}_{idBatch:06d}_{self.idStart:09d}_{self.idEnd-1:09d}.h5",
        )
        return resFile
