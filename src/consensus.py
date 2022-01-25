from abc import ABC
import logging, os, tqdm, io, contextlib
import numpy as np


from src.data import load_hic_groundtruth
from src.utils import chrom_name_to_variables
from src.ctcf import bedPicks, checkCTCFcorrespondance
from src.tad_algo import TopDom, TADtree, OnTAD, TADbit
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
    return counter/len(obs)    def get_consensus(self, TADs):
        pass

class ScoreConsensus(ConsensusMethod):
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

