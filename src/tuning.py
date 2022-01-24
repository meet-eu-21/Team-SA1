import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, contextlib, logging, io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.tad_algo import TopDom, TADtree, OnTAD, TADbit
from src.metrics import compare_to_groundtruth
from src.data import load_hic_groundtruth

def tune_topdom(development_set, param_ranges={'window': (2,15)}):
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
    
    if 'window' in param_ranges:
        window_range = range(param_ranges['window'][0], param_ranges['window'][1])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(window_range), len(set_25kb))), np.zeros((len(window_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(window_range), len(set_100kb))), np.zeros((len(window_range), len(set_100kb)))
        for i, window in enumerate(tqdm(window_range)):
            print('TopDom tuning - Window size: {}'.format(window))
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
        print('\tTopDom tuning - plotting')
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(window_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by TopDom')
        ax1.plot(window_range, pred_rates_25kb, label='Rate of Predicted TADs by TopDom present in Ground Truth')
        ax1.set_xlabel('Window size')
        ax1.set_ylabel('Rate')
        ax1.set_title('TopDom on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(window_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TopDom')
        ax2.plot(window_range, pred_rates_100kb, label='Rate of Predicted TADs by TopDom present in Ground Truth')
        ax2.set_xlabel('Window size')
        ax2.set_ylabel('Rate')
        ax2.set_title('TopDom on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_topdom_window{}-{}.png'.format(param_ranges['window'][0], param_ranges['window'][1]))


def tune_tadtree(development_set, param_ranges={'p': (2,5), 'N': (100,600)}):
    set_100kb = []
    for f in development_set:
        if '25kb' in f:
            pass
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    
    logging.info('Tuning TADTree on intrachromosomal HiC data')
    if 'N' in param_ranges:
        # Preprocess TAD results using parallelization
        data_100kb = []
        with contextlib.redirect_stdout(io.StringIO()) as f:
            for j, f_100kb in enumerate(set_100kb):
                hic_mat, _ = load_hic_groundtruth(f_100kb, 100000)
                data_100kb.append(hic_mat)
        print('Run TADtree on 100kb data')
        with ThreadPoolExecutor(max_workers=16) as executor:
            tadtree = TADtree()
            fct = partial(tadtree.getTADs, N=param_ranges['N'][1])
            executor.map(fct, data_100kb)

        N_range_rev = range(param_ranges['N'][1]-1, param_ranges['N'][0]-1, -1) # Reverse range
        N_range = range(param_ranges['N'][0], param_ranges['N'][1])
        assert len(N_range) == len(N_range_rev)
        len_range = len(N_range)
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(N_range), len(set_100kb))), np.zeros((len(N_range), len(set_100kb)))
        for i, n in enumerate(N_range_rev):
            print('TADTree tuning - N: {}'.format(n))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                print('\tTADTree tuning - starting 100kb')
                for j, f_100kb in enumerate(tqdm(set_100kb)):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, N=n)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_100kb[len_range-i-1,j] = gt_rate_topdom
                    pred_rates_100kb[len_range-i-1,j] = pred_rate_topdom
        print('\tTADTree tuning - plotting')
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax.plot(N_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax.plot(N_range, pred_rates_100kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax.set_xlabel('N')
        ax.set_ylabel('Rate')
        ax.set_title('TADtree on 100kb intrachromosomal HiC data')
        ax.legend()
        plt.savefig('figures/tune_tadtree_N{}-{}.png'.format(param_ranges['N'][0], param_ranges['N'][1]))
    
    if 'p' in param_ranges:
         # Preprocess TAD results using parallelization
        data_100kb = []
        with contextlib.redirect_stdout(io.StringIO()) as f:
            for j, f_100kb in enumerate(set_100kb):
                hic_mat, _ = load_hic_groundtruth(f_100kb, 100000)
                data_100kb.append(hic_mat)
        print('Run TADtree on 100kb data')
        with ThreadPoolExecutor(max_workers=16) as executor:
            tadtree = TADtree()
            fct = partial(tadtree.getTADs, p=param_ranges['p'][1])
            executor.map(fct, data_100kb)

        p_range = range(param_ranges['p'][0], param_ranges['p'][1])
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(p_range), len(set_100kb))), np.zeros((len(p_range), len(set_100kb)))
        for i, p in enumerate(tqdm(p_range)):
            print('TADTree tuning - p: {}'.format(p))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                print('\tTADTree tuning - starting 100kb')
                for j, f_100kb in enumerate(tqdm(set_100kb)):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    tadtree = TADtree()
                    tadtree_tads = tadtree.getTADs(hic_mat, p=p)
                    _, _, gt_rate_topdom, pred_rate_topdom = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadtree_tads, gap=200000)
                    gt_rates_100kb[i,j] = gt_rate_topdom
                    pred_rates_100kb[i,j] = pred_rate_topdom
        print('\tTADTree tuning - plotting')
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax.plot(p_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TADtree')
        ax.plot(p_range, pred_rates_100kb, label='Rate of Predicted TADs by TADtree present in Ground Truth')
        ax.set_xlabel('p')
        ax.set_ylabel('Rate')
        ax.set_title('TADtree on 100kb intrachromosomal HiC data')
        ax.legend()
        plt.savefig('figures/tune_tadtree_p{}-{}.png'.format(param_ranges['p'][0], param_ranges['p'][1]))


def tune_ontad(development_set, param_ranges={'penalty': (0.05,0.35), 'log2': (True,False)}):
    logging.info('Tuning OnTAD on intrachromosomal HiC data')
    set_25kb = []
    set_100kb = []
    for f in development_set:
        if '25kb' in f:
            set_25kb.append(f)
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    
    if 'penalty' in param_ranges:
        penalty_range = np.arange(param_ranges['penalty'][0], param_ranges['penalty'][1], 0.05) # Step of 0.05
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(penalty_range), len(set_25kb))), np.zeros((len(penalty_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(penalty_range), len(set_100kb))), np.zeros((len(penalty_range), len(set_100kb)))
        for i, penalty in enumerate(tqdm(penalty_range)):
            print('OnTAD tuning - penalty: {}'.format(penalty))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    ontad = OnTAD()
                    ontad_tads = ontad.getTADs(hic_mat, penalty=penalty)
                    _, _, gt_rate_ontad, pred_rate_ontad = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=ontad_tads, gap=200000)
                    gt_rates_25kb[i,j] = gt_rate_ontad
                    pred_rates_25kb[i,j] = pred_rate_ontad
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    ontad = OnTAD()
                    ontad_tads = ontad.getTADs(hic_mat, penalty=penalty)
                    _, _, gt_rate_ontad, pred_rate_ontad = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=ontad_tads, gap=200000)
                    gt_rates_100kb[i,j] = gt_rate_ontad
                    pred_rates_100kb[i,j] = pred_rate_ontad
        print('\OnTAD tuning - plotting')
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(penalty_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by OnTAD')
        ax1.plot(penalty_range, pred_rates_25kb, label='Rate of Predicted TADs by OnTAD present in Ground Truth')
        ax1.set_xlabel('Penalty')
        ax1.set_ylabel('Rate')
        ax1.set_title('OnTAD on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(penalty_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by OnTAD')
        ax2.plot(penalty_range, pred_rates_100kb, label='Rate of Predicted TADs by OnTAD present in Ground Truth')
        ax2.set_xlabel('Penalty')
        ax2.set_ylabel('Rate')
        ax2.set_title('OnTAD on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_ontad_penalty{}-{}.png'.format(param_ranges['penalty'][0], param_ranges['penalty'][1]))

    if 'log2' in param_ranges:
        log2_range = [True, False]
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        results_df = pd.DataFrame({'resolution':[25000, 25000, 100000, 100000], 'log2':[True, False, True, False], 'pred_rates':[0,0,0,0], 'gt_rates':[0,0,0,0]})
        for i, log2 in enumerate(tqdm(log2_range)):
            print('OnTAD tuning - log2: {}'.format(log2))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                gt_rates_25kb, pred_rates_25kb = np.zeros(len(set_25kb)), np.zeros(len(set_25kb))
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    ontad = OnTAD()
                    ontad_tads = ontad.getTADs(hic_mat, log2=log2)
                    _, _, gt_rate_ontad, pred_rate_ontad = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=ontad_tads, gap=200000)
                    gt_rates_25kb[j] = gt_rate_ontad
                    pred_rates_25kb[j] = pred_rate_ontad
                results_df.loc[((results_df.resolution==25000) & (results_df.log2==log2)), 'gt_rates'] = gt_rates_25kb.mean(axis=0)
                results_df.loc[((results_df.resolution==25000) & (results_df.log2==log2)), 'pred_rates'] = pred_rates_25kb.mean(axis=0)

                gt_rates_100kb, pred_rates_100kb = np.zeros(len(set_100kb)), np.zeros(len(set_100kb))
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    ontad = OnTAD()
                    ontad_tads = ontad.getTADs(hic_mat, log2=log2)
                    _, _, gt_rate_ontad, pred_rate_ontad = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=ontad_tads, gap=200000)
                    gt_rates_100kb[j] = gt_rate_ontad
                    pred_rates_100kb[j] = pred_rate_ontad
                results_df.loc[((results_df.resolution==100000) & (results_df.log2==log2)), 'gt_rates'] = gt_rates_100kb.mean(axis=0)
                results_df.loc[((results_df.resolution==100000) & (results_df.log2==log2)), 'pred_rates'] = pred_rates_100kb.mean(axis=0)
        print('\OnTAD tuning - plotting')
        width=0.4
        ax1.bar(results_df.loc[results_df.resolution==25000]['log2']-0.2, results_df.loc[results_df.resolution==25000]['pred_rates'], width, label='Rate of Predicted TADs by OnTAD present in Ground Truth')
        ax1.bar(results_df.loc[results_df.resolution==25000]['log2']+0.2, results_df.loc[results_df.resolution==25000]['gt_rates'], width, label='Rate of Ground Truth TADs correctly predicted by OnTAD')
        ax1.set_xlabel('log2')
        ax1.set_ylabel('Rate')
        ax1.set_xticks(results_df.loc[results_df.resolution==25000]['log2'])
        ax1.set_title('OnTAD on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.bar(results_df.loc[results_df.resolution==100000]['log2']-0.2, results_df.loc[results_df.resolution==100000]['pred_rates'], width, label='Rate of Predicted TADs by OnTAD present in Ground Truth')
        ax2.bar(results_df.loc[results_df.resolution==100000]['log2']+0.2, results_df.loc[results_df.resolution==100000]['gt_rates'], width, label='Rate of Ground Truth TADs correctly predicted by OnTAD')
        ax2.set_xlabel('log2')
        ax2.set_ylabel('Rate')
        ax2.set_xticks(results_df.loc[results_df.resolution==100000]['log2'])
        ax2.set_title('OnTAD on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_ontad_log2True-False.png')


def tune_tadbit(development_set, param_ranges={'score_threshold': (0.0,10.0)}):
    logging.info('Tuning TADbit on intrachromosomal HiC data')
    set_25kb = []
    set_100kb = []
    for f in development_set:
        if '25kb' in f:
            set_25kb.append(f)
        elif '100kb' in f:
            set_100kb.append(f)
        else:
            raise ValueError('File name {} was unexpected'.format(f))
    
    if 'score_threshold' in param_ranges:
        score_threshold_range = np.arange(param_ranges['score_threshold'][0], param_ranges['score_threshold'][1], 0.5) # Step of 0.5
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
        gt_rates_25kb, pred_rates_25kb = np.zeros((len(score_threshold_range), len(set_25kb))), np.zeros((len(score_threshold_range), len(set_25kb)))
        gt_rates_100kb, pred_rates_100kb = np.zeros((len(score_threshold_range), len(set_100kb))), np.zeros((len(score_threshold_range), len(set_100kb)))
        for i, score_threshold in enumerate(tqdm(score_threshold_range)):
            print('TADbit tuning - score_threshold: {}'.format(score_threshold))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                for j, f_25kb in enumerate(set_25kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_25kb, 25000)
                    tadbit = TADbit()
                    tadbit_tads = tadbit.getTADs(hic_mat, score_threshold=score_threshold)
                    _, _, gt_rate_tadbit, pred_rate_tadbit = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadbit_tads, gap=200000)
                    gt_rates_25kb[i,j] = gt_rate_tadbit
                    pred_rates_25kb[i,j] = pred_rate_tadbit
                for j, f_100kb in enumerate(set_100kb):
                    hic_mat, arrowhead_tads = load_hic_groundtruth(f_100kb, 100000)
                    tadbit = TADbit()
                    tadbit_tads = tadbit.getTADs(hic_mat, score_threshold=score_threshold)
                    _, _, gt_rate_tadbit, pred_rate_tadbit = compare_to_groundtruth(ground_truth=arrowhead_tads, predicted_tads=tadbit_tads, gap=200000)
                    gt_rates_100kb[i,j] = gt_rate_tadbit
                    pred_rates_100kb[i,j] = pred_rate_tadbit
        print('\TADbit tuning - plotting')
        pred_rates_25kb = pred_rates_25kb.mean(axis=1)
        pred_rates_100kb = pred_rates_100kb.mean(axis=1)
        gt_rates_25kb = gt_rates_25kb.mean(axis=1)
        gt_rates_100kb = gt_rates_100kb.mean(axis=1)
        ax1.plot(score_threshold_range, gt_rates_25kb, label='Rate of Ground Truth TADs correctly predicted by TADbit')
        ax1.plot(score_threshold_range, pred_rates_25kb, label='Rate of Predicted TADs by TADbit present in Ground Truth')
        ax1.set_xlabel('Score Threshold')
        ax1.set_ylabel('Rate')
        ax1.set_title('TADbit on 25kb intrachromosomal HiC data')
        ax1.legend()
        ax2.plot(score_threshold_range, gt_rates_100kb, label='Rate of Ground Truth TADs correctly predicted by TADbit')
        ax2.plot(score_threshold_range, pred_rates_100kb, label='Rate of Predicted TADs by TADbit present in Ground Truth')
        ax2.set_xlabel('Score Threshold')
        ax2.set_ylabel('Rate')
        ax2.set_title('TADbit on 100kb intrachromosomal HiC data')
        ax2.legend()
        plt.savefig('figures/tune_tadbit_score_threshold{}-{}.png'.format(param_ranges['score_threshold'][0], param_ranges['score_threshold'][1]))