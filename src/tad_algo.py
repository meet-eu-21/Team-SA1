import numpy as np
import pandas as pd
import os, time, logging

from abc import ABC, abstractmethod

class TADsDetector(ABC):
    @abstractmethod
    def runAlgo(self):
        pass

class TopDom(TADsDetector):
    def runAlgo(self):
        pass

    def TopDomStep1(self, matrix, window):
        nbins = matrix.shape[0]
        binsignal = np.zeros(nbins)
        for i in range(nbins):
            # R code
            # lowerbound = max( 1, i-size+1 )
            # upperbound = min( i+size, n_bins)
            # mat.data[lowerbound:i, (i+1):upperbound]

            lowerbound = np.maximum(0, i-window)
            upperbound = np.minimum(nbins-1, i+window-1) # TODO: Check if this is correct
            diamond = matrix[lowerbound:i, i:upperbound]
            binsignal[i] = np.mean(diamond)
        return binsignal

    def TopDomStep2(self, binsignal, window):
        # TODO: Review the step 2 to fit curve before capturing local minimas
        binextremums = np.zeros(len(binsignal))
        for i in range(len(binsignal)):
            if binsignal[i] == np.minimum(binsignal[i-window:i+window]):
                binextremums[i] = -1
            elif binsignal[i] == np.maximum(binsignal[i-window:i+window]):
                binextremums[i] = 1
        # np.where(np.array(binextremums) == -1) to find index of local minima
        return binextremums

    def TopDomStep3(self, binextremums, window):