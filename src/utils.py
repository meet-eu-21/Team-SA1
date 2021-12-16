import pandas as pd

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome, resolution):
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t')
    df = df[df['chr1']==chromosome]
    index_chr = df.index.tolist()
    list_tads = []
    for i in index_chr:
        list_tads.append((int(round(df['x1'][i]/resolution), 0), int(round(df['x2'][i]/resolution), 0)))
    return list_tads
