import numpy as np
import pandas as pd
import os, time, logging
from scipy import stats
from scipy.stats import ranksums
from pytadbit.tadbit import tadbit # Only on Linux

from abc import ABC, abstractmethod

class TADsDetector(ABC):
    @abstractmethod
    def getTADs(self, hic_obj):
        pass

class TopDom(TADsDetector):
    def getTADs(self, hic_obj, window=5):
        # Run different steps of the algorithm
        binsignal = self.TopDomStep1(hic_obj.original_matrix, window)
        binextremums = self.TopDomStep2(hic_obj.regions, binsignal, window)
        binextremums = self.TopDomStep3(hic_obj.original_matrix, hic_obj.regions, binextremums, window)
        print('TopDom : Exporting TADs')
        tads = []
        for start,end in hic_obj.regions:
            # Within a region, find the TADs (from a local minimum to another local minimum)
            idx = np.where(binextremums[start:end] == -1)[0]
            nb_minimas = (binextremums[start:end] == -1).sum()
            if nb_minimas == 1:
                pass
            else:
                for i in range(len(idx)-1):
                    tads.append(((start+idx[i])*hic_obj.resolution, (start+idx[i+1])*hic_obj.resolution))
        return tads # Exporting TADs as tuple (from, to)

    def TopDomStep1(self, matrix, window):
        print("TopDom Step 1 : Generating binSignals by computing bin-level contact frequencies")
        nbins = matrix.shape[0]
        binsignal = np.zeros(nbins)
        for i in range(nbins):
            # R code
            # lowerbound = max( 1, i-size+1 )
            # upperbound = min( i+size, n_bins)
            # mat.data[lowerbound:i, (i+1):upperbound]

            if i == 0 or i == nbins-1:
                # Extremums - no binSignal
                binsignal[i] = 0
            else: 
                lowerbound = np.maximum(0, i-window)
                upperbound = np.minimum(nbins-1, i+window-1)
                # Build diamond corresponding to the index
                diamond = matrix[lowerbound:i, i:upperbound]
                binsignal[i] = np.mean(diamond)
        return binsignal

    def findLocalExtremums(self, binsignal, window):
        binextremums = np.zeros(len(binsignal))
        for i in range(len(binsignal)):
            # On a window, look if the binSignal is a local minimum or maximum
            lowerbound = np.maximum(0, i-window)
            upperbound = np.minimum(len(binsignal), i+window)
            if binsignal[i] == np.amin(binsignal[lowerbound:upperbound]):
                binextremums[i] = -1 # Local minimum
            elif binsignal[i] == np.amax(binsignal[lowerbound:upperbound]):
                binextremums[i] = 1 # Local maximum
        # np.where(np.array(binextremums) == -1) to find index of local minima
        return binextremums

    def TopDomStep2(self, regions, binsignal, window):
        # TODO: Review the step 2 to fit curve before capturing local minimas
        print("TopDom Step 2 : Detect TD boundaries based on binSignals")
        binextremums = np.zeros(len(binsignal))
        for start,end in regions:
            binextremums[start:end] = self.findLocalExtremums(binsignal[start:end], window)
        return binextremums
        

    def getFlattenDiamond(self, matrix, i, window):
        n_bins = matrix.shape[0]
        new_matrix = np.zeros((window, window))
        # Get diamond corresponding to the index - by creating new_matrix
        for k in range(window):
            if i-(k-1) >= 1 and i < n_bins:
                lower = min(i, n_bins-1)
                upper = min(i+window, n_bins-1)
                new_matrix[window-k-1, 0:(upper-lower)] = matrix[i-k-1, lower:upper]
        return new_matrix.flatten() # Flatten the matrix

    def getUpstream(self, matrix, i, window):
        # Get the triangle upstream to the index
        n_bins = matrix.shape[0]
        lower = max(0, i-window)
        new_matrix = matrix[lower:i, lower:i]
        return new_matrix[np.triu_indices(new_matrix.shape[0], k=1)]

    def getDownstream(self, matrix, i, window):
        # get the triangle downstream to the index
        n_bins = matrix.shape[0]
        upper = min(n_bins, i+window)
        new_matrix = matrix[i:upper, i:upper]
        return new_matrix[np.triu_indices(new_matrix.shape[0], k=1)]

    # Pass the submatrix as argument
    def getPValue(self, matrix, window, scale=1):
        pvalues = np.ones(matrix.shape[0])
        n_bins = matrix.shape[0]
        for i in range(n_bins):
            # Get the diamond, as well as the upstream and downstream triangles
            diamond = self.getFlattenDiamond(matrix, i, window)
            up = self.getUpstream(matrix, i, window)
            down = self.getDownstream(matrix, i, window)
            # Compare by Wilcox Ranksum Test
            # If diamond interactions are significantly lower than upstream and downstream triangles, then we are at a TAD boundary!
            pvalues[i] = ranksums(diamond*scale, np.concatenate((up, down)), alternative = 'less').pvalue
        pvalues[np.isnan(pvalues)] = 1
        return pvalues

    def TopDomStep3(self, matrix, regions, binextremums, window):
        print("TopDom Step 3 : Statistical Filtering of false positive TD boundaries")
        # TODO: Scale matrix before?
        pvalues = np.ones(len(binextremums))
        for start,end in regions:
            pvalues[start:end] = self.getPValue(matrix[start:end, start:end], window, scale=1)
        # Filter false positive boundaries - i.e. with a non-significant difference between the boundary region and the upstream/downstream regions
        false_pos = (binextremums == -1) & (pvalues >= 0.05)
        binextremums[false_pos] = -2
        return binextremums


class TADtree(TADsDetector):
    def getTADs(self, hic_obj, path_to_TADtree='exe/TADtree.py', S=30, M=10, p=3, q=12, gamma=500, N=500):
        if not os.path.isfile(path_to_TADtree):
            raise Exception("TADtree.py not found")
        if hic_obj.resolution != 100000:
            raise Exception('TADtree is mean to be called with 100kb data only')
        # TODO: Check
        folder_path = hic_obj.get_folder()
        chrom_data_filename = hic_obj.get_name().replace(".npy",".txt")
        output_folder = 'TADtree_outputs_p{}_S{}_M{}_q{}_gamma{}'.format(p, S, M, q, gamma)
        result_path = os.path.join(folder_path, output_folder, chrom_data_filename.split('_')[0], 'N{}.txt'.format(int(N-1)))
        if not os.path.isfile(result_path):
            self.runSingleTADtree(path_to_TADtree, folder_path, chrom_data_filename, S, M, p, q, gamma, N)

        tads_by_tadtree = pd.read_csv(result_path, delimiter='\t')
        tads_by_tadtree = tads_by_tadtree.iloc[:, [1,2]]
        tads = []
        for i in range(len(tads_by_tadtree['start'])):
            tads.append((int(tads_by_tadtree['start'][i]*hic_obj.resolution), int(tads_by_tadtree['end'][i]*hic_obj.resolution)))
        return tads
    
    def runMultipleTADtree(self, path_to_TADtree, folder_path, chrom_data_filenames, S=30, M=10, p=3, q=12, gamma=500, N=400):
        chrom_names = [chrom.split('_')[0] for chrom in chrom_data_filenames]
        output_folder = 'TADtree_outputs_p{}_S{}_M{}_q{}_gamma{}'.format(p, S, M, q, gamma)
        if not os.path.isdir(os.path.join(folder_path, output_folder)):
            os.mkdir(os.path.join(folder_path, output_folder))
        # construct the controle file (conatining parameters of TADtree)
        controle_file = open(folder_path+os.path.sep+'control_file.txt', 'w')
        controle_file.write('S = '+str(S)+'\nM = '+str(M)+'\np = '+str(p)+'\nq = '+str(q)+'\ngamma = '+str(gamma)+'\n\ncontact_map_path = '+folder_path+os.path.sep+chrom_data_filenames[0])
        for i in range(1, len(chrom_data_filenames)):
            controle_file.write(','+folder_path+os.path.sep+chrom_data_filenames[i])
        controle_file.write('\ncontact_map_name = '+chrom_names[0])
        for i in range(0, len(chrom_names)):
            if not os.path.isdir(os.path.join(folder_path, output_folder, chrom_names[i])):
                os.mkdir(os.path.join(folder_path, output_folder, chrom_names[i]))
            if i!=0:
                controle_file.write(','+chrom_names[i])
        controle_file.write('\nN = '+str(N)+'\n\noutput_directory = '+folder_path+os.path.sep+output_folder)
        controle_file.close()
        # apply the command line
        os.system('python '+path_to_TADtree+' '+os.path.join(folder_path, 'control_file.txt'))
        os.rename(os.path.join(folder_path, 'control_file.txt'), os.path.join(folder_path, output_folder, 'control_file.txt'))
        return chrom_names

    def runSingleTADtree(self, path_to_TADtree, folder_path, chrom_data_filename, S=30, M=10, p=3, q=12, gamma=500, N=400):
        chrom_name = chrom_data_filename.split('_')[0]
        output_folder = 'TADtree_outputs_p{}_S{}_M{}_q{}_gamma{}'.format(p, S, M, q, gamma)
        if not os.path.isdir(os.path.join(folder_path, output_folder)):
            os.mkdir(os.path.join(folder_path, output_folder))
        # construct the controle file (conatining parameters of TADtree)
        controle_file = open(folder_path+os.path.sep+'control_file.txt', 'w')
        controle_file.write('S = '+str(S)+'\nM = '+str(M)+'\np = '+str(p)+'\nq = '+str(q)+'\ngamma = '+str(gamma)+'\n\ncontact_map_path = '+folder_path+os.path.sep+chrom_data_filename)
        controle_file.write('\ncontact_map_name = '+chrom_name)
        if not os.path.isdir(os.path.join(folder_path, output_folder, chrom_name)):
            os.mkdir(os.path.join(folder_path, output_folder, chrom_name))
        controle_file.write('\nN = '+str(N)+'\n\noutput_directory = '+folder_path+os.path.sep+output_folder)
        controle_file.close()
        # apply the command line
        os.system('python '+path_to_TADtree+' '+os.path.join(folder_path, 'control_file.txt'))
        os.rename(os.path.join(folder_path, 'control_file.txt'), os.path.join(folder_path, output_folder, 'control_file.txt'))
        return chrom_name

class OnTAD(TADsDetector):
    def getTADs(self, hic_obj, min_size=100000, max_size=3000000, penalty=0.1, ldiff=1.96, wsize=100000, log2=False):
        folder_path = hic_obj.get_folder()
        chrom_data_filename = hic_obj.get_name().replace(".npy",".txt")
        lsize = int(wsize/hic_obj.resolution)
        maxsz = int(max_size/hic_obj.resolution)
        minsz = int(min_size/hic_obj.resolution)
        if log2:
            chrom_tad_output = os.path.join(folder_path, 'OnTAD', chrom_data_filename.replace(".txt", "_p{}_log2.ontad".format(penalty)))
        else:
            chrom_tad_output = os.path.join(folder_path, 'OnTAD', chrom_data_filename.replace(".txt", "_p{}.ontad".format(penalty)))
        if not os.path.isfile(chrom_tad_output):
            data_file = os.path.join(folder_path, chrom_data_filename)
            self.runSingleTAD(data_file, chrom_tad_output, maxsz=maxsz, minsz=minsz, penalty=penalty, ldiff=ldiff, lsize=lsize, log2=log2)
        tads = []
        # startpos  endpos  TADlevel  TADmean  TADscore
        with open(chrom_tad_output, 'r') as f:
            tad_lines = f.readlines()    
            for line in tad_lines:
                start, end, level, mean, score = line.strip().split('\t')
                tads.append((int(start*hic_obj.resolution), int(end*hic_obj.resolution)))
        return tads

    def runSingleTAD(self, in_file, out_file, maxsz, minsz, penalty, ldiff, lsize, log2):
        # TODO: Check parameters
        if log2:
            os.system('./exe/OnTAD {} -maxsz {} -minsz {} -penalty {} -ldiff {} -lsize {} -log2 -o {}'.format(in_file, maxsz, minsz, penalty, ldiff, lsize, out_file))
        else:
            os.system('./exe/OnTAD {} -maxsz {} -minsz {} -penalty {} -ldiff {} -lsize {} -o {}'.format(in_file, maxsz, minsz, penalty, ldiff, lsize, out_file))


