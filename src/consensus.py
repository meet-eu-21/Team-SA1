from abc import ABC
import logging, os, tqdm, io, contextlib, json, logging
import numpy as np

from src.metrics import compare_to_groundtruth
from src.data import load_hic_groundtruth
from src.utils import chrom_name_to_variables
from src.ctcf import bedPicks, checkCTCFcorrespondance
from src.tad_algo import TopDom, TADtree, OnTAD, TADbit, TAD_class_to_str, str_to_TAD_class

# Reference score for three methods (first scores)
reference_scores = {
        'tadtree':83.23,
        'topdom':71.26,
        'arrowhead':78.58
    }
reference_resolution = 25000

# get the TADs boundaries of each method and give them a score
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

# construct the TADs from a list of scores boundaries that we filter to keep the best boundaries
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

# Get the consensus from a dictionary associating methods to their TADs founds on a chromosome
def consensus(all_tads, resolution, threshold, gap=200000, lim=3000000):
    lim = int(lim/resolution)
    extended_lists = []
    for method,list_i in all_tads.items():
        all_tads[method] = sorted(set(list_i))
    gap = int(gap/resolution)
    dico = get_all_boundaries(all_tads, gap)
    output = construct_tads(dico, lim, threshold)
    return output

# Compare consensus TADs with ground_truth (first function)
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
    def __init__(self, ctcf_coeff=1, metrics_coeff=1, init=False) -> None:
        # path to file containing CTCF positions
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
                'TADtree': {'25000': np.NaN, '100000': np.NaN},
                'TopDom': {'25000': np.NaN, '100000': np.NaN},
                'arrowhead': np.NaN, # Check Ground Truth
                'OnTAD': {'25000': np.NaN, '100000': np.NaN}, # TODO: Check OnTAD issue on 100kb
                'TADbit': {'25000': np.NaN, '100000': np.NaN} # TODO: Check TADbit performance
        }

        # used methods according to the resolution
        self.algo_usage = {'25000': ['TopDom', 'OnTAD'], '100000': ['TopDom', 'TADtree', 'TADbit']}

        # weight of the CTCF metric and the both metrics which compare a method with the groundtruth
        self.ctcf_coeff = ctcf_coeff / ((ctcf_coeff + metrics_coeff) / 2)
        self.metrics_coeff = metrics_coeff / ((ctcf_coeff + metrics_coeff) / 2)

        if init:
            self.set_scores()
    
    # get the TADs boundaries of each method and give them a score
    def get_all_boundaries(self, all_algo_TADs, resolution, ctcf_width_region=4):
        dict_pos_score = {}
        for algo, tads in all_algo_TADs.items():
            for tad in tads:
                for i in range(-ctcf_width_region, ctcf_width_region+1):
                    idx_tad = int( (tad[0]+i) / resolution)
                    if idx_tad in dict_pos_score:
                        dict_pos_score[idx_tad] += self.algo_scores[algo]['{}'.format(resolution)] * (1/pow(2, abs(i))) # TODO: Find which law to use (Normal? Log?)
                    else:
                        dict_pos_score[idx_tad] = self.algo_scores[algo]['{}'.format(resolution)] * (1/pow(2, abs(i)))
        return dict(sorted(dict_pos_score.items(), key=lambda x:x[0]))

    # construct the TADs from a list of scores boundaries that we filter to keep the best boundaries
    def construct_tads(self, hic_mat, dict_pos_score, lim, min_size, threshold): # TODO: Tune threshold
        lim = round(lim/hic_mat.resolution)
        minsz = round(min_size/hic_mat.resolution)
        assert lim>minsz

        dict_pos_score = {pos:score for pos,score in dict_pos_score.items() if score*100 >= threshold}

        positions = list(dict_pos_score.keys())
        output_tads = {}
        for i in range(len(positions)-1):
            # control if tad is too large
            if positions[i+1]-positions[i] > lim:
                continue

            # control if tad is too small
            next = 1
            for j in range(i+1, len(positions)):
                assert positions[j] - positions[i] >= 0
                if positions[j] - positions[i] > minsz:
                    break
                next += 1
            if i+next < len(positions):
                output_tads[(int(positions[i]*resolution), int(positions[i+next]*resolution))] = dict_pos_score[positions[i]] + dict_pos_score[positions[i+next]]
        return output_tads

    def get_consensus_tads(self, hic_mat, threshold=10, ctcf_width_region=4, min_tad_size=50000, max_tad_size=3000000):
        #TODO: Implement min_tad_size
        return [k for k in self.get_consensus(hic_mat, threshold, ctcf_width_region, min_tad_size, max_tad_size).keys()]
    
    # Get the consensus from a dictionary associating methods to their TADs founds on a chromosome    
    def get_consensus(self, hic_mat, threshold, ctcf_width_region=4, min_size=50000, lim=3000000):
        all_tads = {}
        for algo in self.algo_scores.keys():
            if self.algo_scores[algo] == np.NaN:
                raise ValueError('ScoreConsensus not trained')
            elif algo in self.algo_usage['{}'.format(hic_mat.resolution)]:
                tad_caller = str_to_TAD_class(algo)()
                all_tads[algo] = tad_caller.getTADs(hic_mat)
        return self.build_consensus(hic_mat, all_tads, threshold, ctcf_width_region, min_size, lim)

    # Get the consensus from a dictionary associating methods to their TADs founds on a chromosome 
    def build_consensus(self, hic_mat, all_tads, threshold, ctcf_width_region, min_size, lim):
        extended_lists = []
        for method,list_i in all_tads.items():
            all_tads[method] = sorted(set(list_i))
        scores_dic = self.get_all_boundaries(all_tads, hic_mat.resolution, ctcf_width_region)
        return self.construct_tads(hic_mat, scores_dic, lim, min_size, threshold)

    def evaluate_algorithm_score(self, development_set):
        logging.info('Evaluating algorithm scores')
        score_save_path = os.path.join('saves', 'algo_scores_consensus.json')
        ctcf_scores_path = os.path.join('saves', 'ctcf_scores.json')
        metrics_scores_path = os.path.join('saves', 'metrics_scores.json')
        
        if os.path.isfile(ctcf_scores_path):
            ctcf_scores = json.load(open(ctcf_scores_path, "r"))
        else:
            ctcf_scores = self.compute_ctcf_scores(development_set)
            with open(ctcf_scores_path, "w") as f:
                json.dumps(ctcf_scores, f)

        if os.path.isfile(metrics_scores_path):
            metrics_scores = json.load(open(metrics_scores_path, "r"))
        else:
            metrics_scores = self.compute_metrics_scores(development_set)
            with open(metrics_scores_path, "w") as f:
                json.dump(metrics_scores, f)
        
        for algo in self.algo_scores.keys():
            if algo in self.algo_usage['25000']:
                self.algo_scores[algo]['25000'] = (self.ctcf_coeff * ctcf_scores[algo]['25000']) * (self.metrics_coeff * sum(metrics_scores[algo]['25000']))
            if algo in self.algo_usage['100000']:
                self.algo_scores[algo]['100000'] = (self.ctcf_coeff * ctcf_scores[algo]['100000']) * (self.metrics_coeff * sum(metrics_scores[algo]['100000']))
        with open(score_save_path, "w") as f:
            json.dump(self.algo_scores, f)
        logging.info('Algorithm scores computed and saved')

    def set_scores(self):
        logging.info('Setting algorithm scores')
        score_save_path = os.path.join('saves', 'algo_scores_consensus.json')
        ctcf_scores_path = os.path.join('saves', 'ctcf_scores.json')
        metrics_scores_path = os.path.join('saves', 'metrics_scores.json')
        
        if not os.path.isfile(ctcf_scores_path) or not os.path.isfile(metrics_scores_path):
            raise ValueError('BordersConsensus: Scores not computed - please run evaluate_algorithm_score()')
        else:
            ctcf_scores = json.load(open(ctcf_scores_path, "r"))
            metrics_scores = json.load(open(metrics_scores_path, "r"))
        
        for algo in self.algo_scores.keys():
            if algo in self.algo_usage['25000']:
                self.algo_scores[algo]['25000'] = (self.ctcf_coeff * ctcf_scores[algo]['25000']) * (self.metrics_coeff * sum(metrics_scores[algo]['25000']))
            if algo in self.algo_usage['100000']:
                self.algo_scores[algo]['100000'] = (self.ctcf_coeff * ctcf_scores[algo]['100000']) * (self.metrics_coeff * sum(metrics_scores[algo]['100000']))
        with open(score_save_path, "w") as f:
            json.dump(self.algo_scores, f)



    # def compute_ctcf_algo_score(self, algo, set, resolution):
    #     if algo in self.algo_usage[resolution]:
    #         ctcf_scores = []
    #         for j, f in enumerate(set):
    #             hic_mat, arrowhead_tads = load_hic_groundtruth(f, resolution)
    #             tad_caller = algo()
    #             tads = tad_caller.getTADs(hic_mat)
    #             chrom, cell_type = chrom_name_to_variables(f)
    #             ctcf_peaks = bedPicks(self.ctcf[cell_type], chrom)
    #             ctcf_scores.append(checkCTCFcorrespondance(ctcf_peaks, tads))
    #         return np.mean(ctcf_scores)
    #     else:
    #         return np.NaN
            
    def compute_ctcf_scores(self, development_set):
        logging.info('Computing CTCF scores on intrachromosomal HiC data')
        set_25kb = []
        set_100kb = []
        for f in development_set:
            if '25kb' in f:
                set_25kb.append(f)
            elif '100kb' in f:
                set_100kb.append(f)
            else:
                raise ValueError('File name {} was unexpected'.format(f))

        algo_ctcf_scores = {algo: {'25000': np.NaN, '100000': np.NaN} for algo in self.algo_scores.keys()}
        with contextlib.redirect_stdout(io.StringIO()) as f:
            for algo in self.algo_scores.keys():
                if algo in self.algo_usage['25000']:
                    ctcf_scores_25kb = []
                    for j, f_25kb in enumerate(set_25kb):                    
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, '25000')
                        tad_caller = str_to_TAD_class(algo)()
                        tads = tad_caller.getTADs(hic_mat)
                        chrom, cell_type = chrom_name_to_variables(f_25kb)
                        ctcf_peaks = bedPicks(self.ctcf[cell_type], chrom)
                        ctcf_scores_25kb.append(checkCTCFcorrespondance(ctcf_peaks, tads))
                    algo_ctcf_scores[algo]['25000'] = np.mean(ctcf_scores_25kb)
                
                if algo in self.algo_usage['100000']:
                    ctcf_scores_100kb = []
                    for j, f_100kb in enumerate(set_100kb):
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, '100000')
                        tad_caller = str_to_TAD_class(algo)()
                        tads = tad_caller.getTADs(hic_mat)
                        chrom, cell_type = chrom_name_to_variables(f_100kb)
                        ctcf_peaks = bedPicks(self.ctcf[cell_type], chrom)
                        ctcf_scores_100kb.append(checkCTCFcorrespondance(ctcf_peaks, tads))
                    algo_ctcf_scores[algo]['100000'] = np.mean(ctcf_scores_100kb)
        
        return algo_ctcf_scores

    def compute_metrics_scores(self, development_set):
        logging.info('Computing Metrics on intrachromosomal HiC data')
        set_25kb = []
        set_100kb = []
        for f in development_set:
            if '25kb' in f:
                set_25kb.append(f)
            elif '100kb' in f:
                set_100kb.append(f)
            else:
                raise ValueError('File name {} was unexpected'.format(f))
        
        algo_performances = {algo: {'25000': (np.NaN, np.NaN), '100000': (np.NaN, np.NaN)} for algo in self.algo_scores.keys()}
        for algo in self.algo_scores.keys():
            gt_rates_25kb, pred_rates_25kb = np.zeros(len(set_25kb)), np.zeros(len(set_25kb))
            gt_rates_100kb, pred_rates_100kb = np.zeros(len(set_100kb)), np.zeros(len(set_100kb))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                if algo in self.algo_usage['25000']:
                    for i, f_25kb in enumerate(set_25kb):
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, '25000')
                        method = str_to_TAD_class(algo)()
                        tads = method.getTADs(hic_mat)
                        _, _, gt_rate_tadbit, pred_rate_tadbit = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tads)
                        gt_rates_25kb[i] = gt_rate_tadbit
                        pred_rates_25kb[i] = pred_rate_tadbit
                    pred_rates_25kb = pred_rates_25kb.mean(axis=0)
                    gt_rates_25kb = gt_rates_25kb.mean(axis=0)
                    algo_performances[algo]['25000'] = (gt_rates_25kb, pred_rates_25kb)

                if algo in self.algo_usage['100000']:
                    for i, f_100kb in enumerate(set_100kb):
                        hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, '100000')
                        method = str_to_TAD_class(algo)()
                        tads = method.getTADs(hic_mat)
                        _, _, gt_rate_tadbit, pred_rate_tadbit = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tads)
                        gt_rates_100kb[i] = gt_rate_tadbit
                        pred_rates_100kb[i] = pred_rate_tadbit
                    pred_rates_100kb = pred_rates_100kb.mean(axis=0)
                    gt_rates_100kb = gt_rates_100kb.mean(axis=0)
                    algo_performances[algo]['100000'] = (gt_rates_100kb, pred_rates_100kb)
            
        return algo_performances

