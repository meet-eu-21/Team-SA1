import os

#Make sure you have run the Simon's preprocess algorithm in the folder you're using
path="C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)"
os.chdir(path)

from lmfit.models import GaussianModel
from math import *
from src.Analyse_longueur_TADs import *

#Just a function that returns y=f(x) if f() is a Gaussian function
def Gaussian_function(x,A,mu,sigma):
    return (A/(sigma*sqrt(2*pi)))*exp(-((x-mu)**2)/(2*sigma)**2)


#fitting_curve_signle_file is a function that takes 2 arguments: 1. file_path = the path to the file you want to analyse (str)
#                                                                2. TADs_methd = the method you used to get your TADs (str)

#The output is None, and a graph will be ploted: the plot shows the experimental datas (in blue) which are the number of TADs for each size detected; it also shows the fitted curve that could be used as prediction (in green).

def fitting_curve_single_file(file_path,TADs_method): #TADs_method should be either "TD" (if your file is to be analysed by TopDom) or "AH" (of analysed by ArrowHead).

    if TADs_method=="TD":                             #This block extracts
        topdom_tads=prod_topdom_tads(file_path)       #the sizes and number
        data=extraction_data_TD(topdom_tads)          #of TADs for
    elif TADs_method=="AH":                           #each sizes detected
        data=extraction_data_AH(file_path)            #
    x,y=np.array(data[0]),np.array(data[1])           #

    #Do the method with the model = Gaussian
    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp,Center,Sigma=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for i in x:
        courbe.append(Gaussian_function(i,Amp,Center,Sigma))

    #plot of the comparison between empirical and predicted datas
    plt.plot(x, y, 'b.-')
    plt.plot(x,courbe,'g-')
    plt.show()
    return None


