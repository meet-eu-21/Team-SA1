import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, logging, io
from tqdm import tqdm

from src.tad_algo import TopDom, TADtree
from src.metrics import compare_to_groundtruth
from src.data import load_hic_groundtruth

def tune_topdom(development_set, param_ranges={'window': (2,15)}):
    logging.info('Tuning TopDom on GM12878 intrachromosomal HiC data')
    set_25kb = []
    set_100kb = []
    for f in development_set:
        if '25kb' in f:
            set_25kb.append(f)
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    
    if 'window' in param_ranges:
        window_range = range(param_ranges['window'][0], param_ranges['window'][1])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(window_range), len(set_25kb))), np.zeros((len(window_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(window_range), len(set_100kb))), np.zeros((len(window_range), len(set_100kb)))
        for i, window in enumerate(tqdm(window_range)):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    topdom = TopDom()
                    topdom_tads = topdom.getTADs(hic_mat, window=window)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=topdom_tads, gap=200000)
                    gt_rates_25kb[i,j] = gt_rate_topdom
                    pred_rates_25kb[i,j] = pred_rate_topdom
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    topdom = TopDom()
                    topdom_tads = topdom.getTADs(hic_mat, window=window)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=topdom_tads, gap=200000)
                    gt_rates_100kb[i,j] = gt_rate_topdom
                    pred_rates_100kb[i,j] = pred_rate_topdom
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(window_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by TopDom')
        ax1.plot(window_range, pred_rates_25kb, label='Rate of Predicted TADs by TopDom present in Ground Truth')
        ax1.set_xlabel('Window size')
        ax1.set_ylabel('Rate')
        ax1.set_title('TopDom on GM12878 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(window_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TopDom')
        ax2.plot(window_range, pred_rates_100kb, label='Rate of Predicted TADs by TopDom present in Ground Truth')
        ax2.set_xlabel('Window size')
        ax2.set_ylabel('Rate')
        ax2.set_title('TopDom on GM12878 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_topdom_window{}-{}.png'.format(param_ranges['window'][0], param_ranges['window'][1]))


def tune_tadtree(development_set, param_ranges={'p': (2,5), 'N': (100,700)}):
    set_25kb = []
    set_100kb = []
    for f in development_set:
        if '25kb' in f:
            set_25kb.append(f)
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    
    logging.info('Tuning TADTree on GM12878 intrachromosomal HiC data')
    if 'N' in param_ranges:
        N_range_rev = range(param_ranges['N'][1]-1, param_ranges['N'][0]-1, -1) # Reverse range
        N_range = range(param_ranges['N'][0], param_ranges['N'][1])
        assert len(N_range) == len(N_range_rev)
        len_range = len(N_range)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(N_range), len(set_25kb))), np.zeros((len(N_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(N_range), len(set_100kb))), np.zeros((len(N_range), len(set_100kb)))
        for i, n in enumerate(tqdm(N_range_rev)):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, N=n)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_25kb[len_range-i-1,j] = gt_rate_topdom
                    pred_rates_25kb[len_range-i-1,j] = pred_rate_topdom
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, N=n)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_100kb[len_range-i-1,j] = gt_rate_topdom
                    pred_rates_100kb[len_range-i-1,j] = pred_rate_topdom
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(N_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax1.plot(N_range, pred_rates_25kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Rate')
        ax1.set_title('TADtree on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(N_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax2.plot(N_range, pred_rates_100kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax2.set_xlabel('N')
        ax2.set_ylabel('Rate')
        ax2.set_title('TADtree on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_tadtree_N{}-{}.png'.format(param_ranges['N'][0], param_ranges['N'][1]))
    
    if 'p' in param_ranges:
        p_range = range(param_ranges['p'][0], param_ranges['p'][1])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(p_range), len(set_25kb))), np.zeros((len(p_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(p_range), len(set_100kb))), np.zeros((len(p_range), len(set_100kb)))
        for i, p in enumerate(tqdm(p_range)):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, p=p)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_25kb[i,j] = gt_rate_topdom
                    pred_rates_25kb[i,j] = pred_rate_topdom
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, p=p)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_100kb[i,j] = gt_rate_topdom
                    pred_rates_100kb[i,j] = pred_rate_topdom
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(p_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax1.plot(p_range, pred_rates_25kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax1.set_xlabel('p')
        ax1.set_ylabel('Rate')
        ax1.set_title('TADtree on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(p_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax2.plot(p_range, pred_rates_100kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax2.set_xlabel('p')
        ax2.set_ylabel('Rate')
        ax2.set_title('TADtree on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_tadtree_p{}-{}.png'.format(param_ranges['p'][0], param_ranges['p'][1]))
        