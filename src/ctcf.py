import pandas as pd


def checkCTCFcorrespondance(ctcf_df, tads, ctcf_allowed_distance=50000):
    """
    Get the rate of TADs boundaries which correspond to CTCF peaks
    ----------
    INPUT
    ctcf_df : list of int
        list containing the CTCF peaks
    tads : list of tuples
        list containing all the TADs in tuple (from, to)
    ctcf_allowed_distance : int
        maximum distance between a TAD boundary and a CTCF peaks to consider them like the same
    -----------
    OUTPUT
    rate of predicted TAD boundaries corresponding to CTCF peaks
    """
    tads_borders_ctcf = set()
    all_borders = set()
    for i, tad in enumerate(tads):
        for j, ctcf in ctcf_df.iterrows():
            all_borders.add(tad[0])
            all_borders.add(tad[1])
            if abs(ctcf['chStart'] - tad[0]) <= ctcf_allowed_distance:
                tads_borders_ctcf.add(tad[0])
            if abs(ctcf['chStart'] - tad[1]) <= ctcf_allowed_distance:
                tads_borders_ctcf.add(tad[1])
    return round(len(tads_borders_ctcf)/max(1,len(all_borders)), 4)


def bedPicks(file, chrom):
    """
    Get all CTCF peaks of a chromosome 
    ----------
    INPUT
    file : str
        path to the file containing all the CTCF peaks of the whole genome for a cell type
    chrom : str
        chromosome of interest
    -----------
    OUTPUT
    list of all the CTCF peaks
    """
    df = pd.read_csv(file, sep='\t', comment = 't', header=None)
    header = ['chrom', 'chStart', 'chEnd', 'name', 'score', 'strand', 'sigValue', 'pValue', 'qValue', 'peak']
    df.columns = header[:len(df.columns)]
    l_peak = []                               #to store pairs (chromStart, chromEnd) for a specific chrom
    
    #delete unuseful columns 
    if set(df['name'])=={'.'}:
        del df['name']
    if set(df['strand'])=={'.'}:
        del df['strand']
    if set(df['pValue'])=={-1.0}:
        del df['pValue']
        
    #we take into account data for a specific chromosome 
    df = df[df['chrom']==chrom]
    return df.sort_values(by = 'chStart') # sorted the peaks by chronological order

