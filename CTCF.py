import pandas as pd
import numpy as np

def bedPicks(file, chrom, resolution):
    
    df = pd.read_csv(file, sep='\t', comment = 't', header=None)
    header = ['chrom', 'chStart', 'chEnd', 'name', 'score', 'strand', 'sigValue', 'pValue', 'qValue', 'peak']
    df.columns = header[:len(df.columns)]
    l_peak = []                               #to store pairs (chromStart, chromEnd) for a specific chrom
    
    #delete non useful columns 
    if set(df['name'])=={'.'}:
        del df['name']
    if set(df['strand'])=={'.'}:
        del df['strand']
    if set(df['pValue'])=={-1.0}:
        del df['pValue']
        
    #we take into account data for a specific chromosome 
    df = df[df['chrom']==chrom]
    df = df.sort_values(by = 'chStart')
    
    #just some tests for a eventual filter 
    score = np.array(df['sigValue'])
    #print(score.mean(), score.min(), score.max()) 
    index = df.index.tolist()
    for ctcf in index:
        l_peak.append(int(round(df['chStart'][ctcf]/resolution, 0)))
        l_peak.append(int(round(df['chEnd'][ctcf]/resolution, 0)))
    return list(set(l_peak))