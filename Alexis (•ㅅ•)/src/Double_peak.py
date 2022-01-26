import glob
import os
os.chdir('C:\\Users\\Alexis Trang\\Documents\\Cours_UPMC_M2\\MEET-U\\Alexis (•ㅅ•)')

from math import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ExponentialModel, GaussianModel
from src.Analyse_longueur_TADs import *


folder_path="C:/Users/Alexis Trang/Documents/Cours_UPMC_M2/MEET-U/Alexis (•ㅅ•)/Data/TAD"

#Just a function that returns y=f(x) if f() is a Gaussian function
def Gaussian_function(x,A,mu,sigma):
    return (A/(sigma*sqrt(2*pi)))*exp(-((x-mu)**2)/(2*sigma)**2)


#double_peak does exactly what the fitting_curve_whole_folder does but more precisely since it will take into account the little peak after the initial drop.
#double peak is only usable on ArrowHead datas for the moment.
#double peak takes as input a path to the folder you try to analyse (folder_path=str).
#it only returns the fitted curve.

def double_peak(folder_path):
    count=0
    dico_long={}

    Title="Gaussian fitting curve for ArrowHead TADs, 5kb resolution."

    for i in glob.glob(folder_path+"/*.txt"):
        count+=1 #nmbr of files
        data_step=extraction_data_AH(i)
        dico_step=data_step[2]
        for j in list(dico_step.keys()):
            if j in dico_long.keys():
                dico_long[j]+=dico_step[j]
            else:
                dico_long[j]=dico_step[j]

    l_long=list(dico_long.keys())
    l_long.sort()
    l_count=[]
    for j in l_long:
        l_count.append(dico_long[j]/count)


    dat=l_long,l_count

    x = np.array(dat[0])
    y = np.array(dat[1][:38]+[0]*(len(x)-38))

    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp1,Center1,Sigma1=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Gaussian_function(k,Amp1,Center1,Sigma1))



    #####
    x = np.array(dat[0])
    y = np.array([0]*45+dat[1][45:70]+[0]*(len(x)-70))

    mod = GaussianModel()       #Says what method we want to use
    pars = mod.guess(y, x=x)    #Initial parametres to be used
    out = mod.fit(y, pars, x=x) #Fitting with Least-Square Minimization method

    #Print the nutshell of the fitting
    print(out.fit_report(min_correl=0.25))

    #Gets the best values for Amplitude, Center, and Sigma, to get the closest Gaussian model to our experimental datas.
    Amp2,Center2,Sigma2=out.best_values['amplitude'],out.best_values['center'],out.best_values['sigma']

    #Values of our model
    courbe=[]
    for k in x:
        courbe.append(Gaussian_function(k,Amp2,Center2,Sigma2))

    ######
    x = np.array(dat[0])
    y = np.array(dat[1])

    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    pars['g1_center'].set(value=Center1)  #Giving the values that we found earlier
    pars['g1_sigma'].set(value=Sigma1)
    pars['g1_amplitude'].set(value=Amp1)

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())

    pars['g2_center'].set(value=Center2, min=0.5*1e6, max= 0.6*1e6)
    pars['g2_sigma'].set(value=Sigma2)
    pars['g2_amplitude'].set(value=Amp2)

    mod = gauss1 + gauss2  #Merging the models

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    print(out.fit_report(min_correl=0.5))

    plt.plot(x, y,'b.-')
    plt.plot(x, out.best_fit, 'r-', label='best fit')
    plt.ylabel("Effectives",fontsize=25)
    plt.xlabel("TAD sizes",fontsize=25)
    plt.title(Title, fontsize=30)
    plt.legend()

    plt.show()