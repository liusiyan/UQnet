import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.rcParams["axes.grid"] = False


''' Plotting function for Figure 2 
To reproduce the plots, you can directly run this file to utilize the pre-generated data, or re-run the code:
--- enter the 'QD_sensitivity_tests' folder and run the 'main_QD_sensitivity.py'
--- Each run will produce a .txt result file for given data set ('boston by default')
--- To reproduce the results, simply assign the data set name to the "data_name" variable at the beginning of this code before running the main file.
--- Accepted data names are: 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD'
--- After running the 'main_QD_sensitivity.py' for all 10 data sets, move all the results file (for example 'boston_QD_sensitivity_results_seed_1_10.txt') from the ./QD_sensitivity_tests/Results to upper level folder
--- Run the 'Figure_2_plots.py' to reproduce the plots.

Have fun! 
'''



############################################################################
#### plot boxplot of Concrete example form UCI dataset for NeurIPs2021
############################################################################
files =[]
for file in os.listdir("./"):
    if file.endswith(".txt"):
        files.append(file)
files.sort()
MSD= files[0]
files.remove(MSD)
files.append(MSD)


plotvarnames =['PICP_test','MPIW_test','RMSE']
color = dict(boxes='blue', whiskers='black', medians='red', caps='black')
for ivar in np.arange(len(plotvarnames)):
    # ------ PLOT ------
    plt.figure(dpi=400, figsize=(15, 5))
    isub=0
    for file in files:
          ix= file.index('_')
          case = file[:ix]
          data = pd.read_csv(file, sep=" ", usecols=[3+ivar])
          isub += 1
          plt.subplot(2,5,isub)
          g1=data.boxplot(fontsize=18, widths=(0.15), boxprops=dict(linewidth=2.0),
                  whiskerprops=dict(linestyle='-',linewidth=2.0),
                  medianprops = dict(linewidth=2), color=color,
                  positions = [1.0], whis=(0,100))
          plt.xlim(0.4,1.4)
          #plt.ylim(0.6,1.3)
          #plt.yticks(np.arange(0.6,1.4,0.1))
          if case=='MSD':
              plt.title(case, fontsize=18)
          else:
              plt.title(case.capitalize(), fontsize=18)

          plt.xlabel('')
          g1.set(xticklabels=[])
          g1.set(xlabel=None)
          if isub==1 or isub==6:
              if '_' in plotvarnames[ivar]:
                  ix = plotvarnames[ivar].index('_')
              else:
                  ix=-1

              plt.ylabel(plotvarnames[ivar][:ix],fontsize=18)
          plt.grid(None)
          ymin = np.amin(data[plotvarnames[ivar]])
          ymax = np.amax(data[plotvarnames[ivar]])
          plt.text(0.425, ymax-0.35*(ymax-ymin), 'Max:{:.2f}'.format(ymax)+'\nMin:{:.2f}'.format(ymin),fontsize=14)
    plt.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.9, wspace=0.4, hspace=0.25)
    plt.savefig(plotvarnames[ivar]+'.png',pad_inches=0.2)



