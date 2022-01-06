#Make sure you have run the Simon's preprocess algorithm in the folder you're using


data_path="C:/Users/Alexis Trang/Documents/Cours_UPMC_M2/MEET-U/Data/HiC/GM12878/100kb_resolution_intrachromosomal/chr2_100kb.npy.txt.npy"       #Change the nb of the chromosom
hic_mat = Hicmat(data_path, resolution)
hic_mat.filter(threshold = 1)
topdom_tads = topdom.getTADs(hic_mat, window=5)
plot_bar_TD(topdom_tads)