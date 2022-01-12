import pandas as pd

def bedPicks(file, chrom, resolution):
    
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
    df = df.sort_values(by = 'chStart')
    
    l_sigValue = []
    for ctcf in df.index.tolist():
        locus = int(round(df['chStart'][ctcf]/resolution, 0))
        if locus not in l_peak:
            l_peak.append(locus)
            l_sigValue.append(df['sigValue'][ctcf])
    return l_peak, l_sigValue

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