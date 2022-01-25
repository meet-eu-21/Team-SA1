from abc import ABC
import logging, os, tqdm, io, contextlib
import numpy as np


from src.data import load_hic_groundtruth
from src.utils import chrom_name_to_variables
from src.ctcf import bedPicks, checkCTCFcorrespondance
from src.tad_algo import TopDom, TADtree, OnTAD, TADbit

reference_scores = {
        'tadtree':83.23,
        'topdom':71.26,
        'arrowhead':78.58
    }
reference_resolution = 25000

def get_all_boundaries(TADs, gap):
    score = {
        'tadtree':83.23,
        'topdom':71.26,
        'arrowhead':78.58
    }
    dict_pos_score = {}
    for key,tads in TADs.items():
        for tad in tads:
            for i in range(-gap, gap+1):
                if tad[0]+i in dict_pos_score:
                    dict_pos_score[tad[0]+i]+=score[key]*(1/(abs(i)+1))
                else:
                    dict_pos_score[tad[0]+i]=score[key]*(1/(abs(i)+1))
    return dict(sorted(dict_pos_score.items(), key=lambda x:x[0]))

def construct_tads(dict_pos_score, lim, threshold):
    dict_pos_score = {pos:score for pos,score in dict_pos_score.items() if score>threshold}
    pos = list(dict_pos_score.keys())
    score = list(dict_pos_score.values())
    output = {}
    for i in range(len(pos)-1):
        if pos[i+1]-pos[i]>lim:
            continue
        output[(pos[i], pos[i+1])]=score[i]+score[i+1]
    return output

def consensus(all_tads, resolution, threshold, gap=200000, lim=3000000):
    lim = int(lim/resolution)
    extended_lists = []
    for method,list_i in all_tads.items():
        all_tads[method] = sorted(set(list_i))
    gap = int(gap/resolution)
    dico = get_all_boundaries(all_tads, gap)
    output = construct_tads(dico, lim, threshold)
    return output

def compare_TADs(obs, trues, gap):
    counter=0
    in_trues=False
    for tad in obs:
        for i in range(-gap, gap+1):
            for j in range(-gap, gap+1):
                if (tad[0]+i, tad[1]+j) in trues:
                    in_trues=True
                    counter+=1
                    break
            if in_trues:
                in_trues=False
                break
    return counter/len(obs)

class ConsensusMethod(ABC):
    def get_consensus(self, TADs):
        pass

class BordersConsensus(ConsensusMethod):
    def __init__(self) -> None:
        self.ctcf = {
                'GM12878':'data/CTCF/GM12878/ENCFF796WRU.bed',
                'HMEC':'data/CTCF/HMEC/ENCFF059YXD.bed',
                'HUVEC':'data/CTCF/HUVEC/ENCFF949KVG.bed',
                'IMR90':'data/CTCF/IMR90/ENCFF203SRF.bed',
                'NHEK':'data/CTCF/NHEK/ENCFF351YOQ.bed'
            }
        # TODO: Use algos for some resolution only

        # Implement score as CTCF*(M_1 + M2)
        self.algo_scores = {
                TADtree: {25000: np.NaN, 100000: np.NaN},
                TopDom: {25000: np.NaN, 100000: np.NaN},
                'arrowhead': np.NaN, # Check Ground Truth
                OnTAD: {25000: np.NaN, 100000: np.NaN}, # TODO: Check OnTAD issue on 100kb
                TADbit: {25000: np.NaN, 100000: np.NaN} # TODO: Check TADbit performance
        }

        self.algo_usage = {25000: [TopDom, OnTAD], 100000: [TopDom, TADtree, TADbit]}

    def get_all_boundaries(self, all_algo_TADs, resolution, ctcf_width_region=4):
        dict_pos_score = {}
        for algo, tads in all_algo_TADs.items():
            for tad in tads:
                for i in range(-ctcf_width_region, ctcf_width_region+1):
                    idx_tad = int( (tad[0]+i) / resolution)
                    if idx_tad in dict_pos_score:
                        dict_pos_score[idx_tad] += self.algo_scores[algo] * (1/pow(2, abs(i))) # TODO: Find which law to use (Normal? Log?)
                    else:
                        dict_pos_score[idx_tad] = self.algo_scores[algo] * (1/pow(2, abs(i)))
        return dict(sorted(dict_pos_score.items(), key=lambda x:x[0]))

    def construct_tads(self, dict_pos_score, resolution, lim, threshold): # TODO: Tune threshold
        lim = int(lim/resolution)

        dict_pos_score = {pos:score for pos,score in dict_pos_score.items() if score >= threshold}

        positions = list(dict_pos_score.keys())
        output_tads = {}
        for i in range(len(positions)-1):
            if positions[i+1]-positions[i] > lim:
                continue
            output_tads[(positions[i], positions[i+1])] = dict_pos_score[positions[i]] + dict_pos_score[positions[i+1]]
        return output_tads
    
    def get_consensus(self, hic_mat, threshold, ctcf_width_region=4, lim=3000000):
        all_tads = {}
        for algo in self.algo_scores.keys():
            if self.algo_scores[algo] == np.NaN:
                raise ValueError('ScoreConsensus not trained')
            elif algo in self.algo_usage[hic_mat.resolution]:
                tad_caller = algo()
                all_tads[algo] = tad_caller.getTADs(hic_mat)
        return self.build_consensus(all_tads, threshold, ctcf_width_region, lim)

    def build_consensus(self, all_tads, resolution, threshold, ctcf_width_region, lim):
        extended_lists = []
        for method,list_i in all_tads.items():
            all_tads[method] = sorted(set(list_i))
        scores_dic = self.get_boundaries(all_tads, resolution, ctcf_width_region)
        return construct_tads(scores_dic, lim, threshold)

    def evaluateAlgorithmScore(self, development_set):
        # TODO: Save scores in a file and reload them next time
        logging.info('Tuning TopDom on intrachromosomal HiC data')
        set_25kb = []
        set_100kb = []
        for f in development_set:
            if '25kb' in f:
                set_25kb.append(f)
            elif '100kb' in f:
                set_100kb.append(f)
            else:
                raise ValueError('File name {} was unexpected'.format(f))

        ctcf_scores_25kb = {algo: [] for algo in self.algo_scores.keys()}
        ctcf_scores_100kb = {algo: [] for algo in self.algo_scores.keys()}
        with contextlib.redirect_stdout(io.StringIO()) as f:
            for algo in self.algo_scores.keys():
                if algo in self.algo_usage[25000]:
                    for j, f_25kb in enumerate(set_25kb):                    
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                        tad_caller = algo()
                        tads = tad_caller.getTADs(hic_mat)
                        chrom, cell_type = chrom_name_to_variables(f_25kb)
                        ctcf_peaks = bedPicks(self.ctcf[cell_type], chrom)
                        ctcf_scores_25kb[algo].append(checkCTCFcorrespondance(ctcf_peaks, tads))
                if algo in self.algo_usage[100000]:
                    for j, f_100kb in enumerate(set_100kb):
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                        tad_caller = algo()
                        tads = tad_caller.getTADs(hic_mat)
                        chrom, cell_type = chrom_name_to_variables(f_100kb)
                        ctcf_peaks = bedPicks(self.ctcf[cell_type], chrom)
                        ctcf_scores_100kb[algo].append(checkCTCFcorrespondance(ctcf_peaks, tads))
        print(ctcf_scores_25kb)
        print(ctcf_scores_100kb)
        for algo in self.algo_scores.keys():
            if algo in self.algo_usage[25000]:
                self.algo_scores[algo][25000] = np.mean(ctcf_scores_25kb[algo])
            if algo in self.algo_usage[100000]:
                self.algo_scores[algo][100000] = np.mean(ctcf_scores_100kb[algo])
