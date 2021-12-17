import pandas as pd

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome, resolution):
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t')
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in df.index.tolist():
        list_tads.append((int(round(df['x1'][i]/resolution, 0)), int(round(df['x2'][i]/resolution, 0))))
    return list_tads
