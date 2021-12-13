import pandas as pd

# load the available ArrowHead results
def read_arrowhead_result(path, chromosome, resolution):
    # TODO: Refactor
    df = pd.read_csv(path, sep='\t')
    df = df[df['chr1']==chromosome]
    list_tads = []
    for i in range(len(df)):
        list_tads.append((int(df['x1'][i]/resolution), int(df['x2'][i]/resolution)))
    return list_tads