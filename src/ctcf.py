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
    ctcf_allowed_distance :
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

# 
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
    
    # l_sigValue = []
    # for ctcf in df.index.tolist():
    #     locus = int(df['chStart'][ctcf])
    #     if locus not in l_peak:
    #         l_peak.append(locus)
    #         l_sigValue.append(df['sigValue'][ctcf])
    # return l_peak, l_sigValue

# TO REMOVE
def evaluate(TADs, ctcf):
    a=0
    b=[]
    c=set()
    for tad in TADs:
        if tad[0] in ctcf and tad[0] not in b:
            a+=1
            b.append(tad[0])
        if tad[1] in ctcf and tad[1] not in b:
            a+=1
            b.append(tad[1])
        c.add(tad[0])
        c.add(tad[1])
    return round(a/max(1,len(c)), 4)*100