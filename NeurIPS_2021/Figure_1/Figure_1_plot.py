import numpy as np
import matplotlib.pyplot as plt

''' Plotting function for Figure 1 
To reproduce the plots, you can directly run this file to utilize the pre-generated data, or re-run the code:
--- For PI3NN, enter the 'PI3NN_cubic' folder and run the 'PI3NN_cubic.py' ---> produce 'TrainData.dat' and 'OurMethod_cubic.dat'
--- For QD, enter the 'QD_cubic' folder and run the 'QD_cubic.py' ---> produce ---> produce 'PI_cubic.dat'
--- For DER, enter the 'DER_cubic' folder and run the 'run_cubic_evidentialDL.py' ---> produce 'EvidentialDL_cubic.dat'

After running all three codes, you can move the generated results files 'TrainData.dat', 'OurMethod_cubic.dat', 'PI_cubic.dat', and 'EvidentialDL_cubic.dat' to upper level folder and run this file to plot the results.

Have fun! 
'''

#---- load the training data from our method
train = np.loadtxt('TrainData.dat')
xtrain = train[:, 0]
ytrain = train[:, 1]

#----- Prediction intervl method results
PI1 = np.loadtxt('PI_cubic.dat')
x1 = PI1[0, :]
upper1 = PI1[1, :]
lower1 = PI1[2, :]
print('size of PI1 ', PI1.shape)


#----- Evidential regression method results
evd = np.loadtxt('EvidentialDL_cubic.dat')
x3 = evd[0, :]
evd_mean = evd[1, :]
evd_std = evd[2, :]
print('size of ens ', evd.shape)


#---- our method results
PI2 = np.loadtxt('OurMethod_cubic.dat')
x2 = PI2[:, 0]
upper2 = PI2[:, 1]
lower2 = PI2[:, 2]
pred = PI2[:, 3]
print('size of PI2 ', PI2.shape)


#------ PLOT ------
lw = 1.5 # linewidth

plt.figure(figsize=(11, 3.5), dpi=200)

#---- quality-driven PI ---
plt.subplot(1, 3, 1)
plt.plot(xtrain, ytrain, 'k.', markersize=2, alpha=0.2, zorder=0, label='Data')
plt.plot(x1, x1**3, 'r-', linewidth=lw, label='Ground Truth')
plt.fill_between(x1, lower1, upper1,
            alpha=0.4,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Uncertainty")
plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.fill_between([-7, -4], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0, label='No data')
plt.fill_between([4, 7], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0)
plt.ylim(-150, 150)
plt.xlim(-7, 7)
plt.xticks(np.arange(-6, 8, 2))
plt.title('QD', fontsize=14)

#--- evidential regression method ----
plt.subplot(1, 3, 2)
plt.plot(xtrain, ytrain, 'k.', markersize=2, alpha=0.2, zorder=0, label='Data')
plt.plot(x3, x3**3, 'r-', linewidth=lw, label='Ground Truth')
plt.plot(x3, evd_mean, 'b--', linewidth=lw, label='Prediction')
plt.fill_between(x3, evd_mean-1.96*evd_std, evd_mean+1.96*evd_std,
            alpha=0.4,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="95% PI")
plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.fill_between([-7, -4], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0, label='No data')
plt.fill_between([4, 7], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0)
plt.ylim(-150, 150)
plt.xlim(-7, 7)
plt.xticks(np.arange(-6, 8, 2))
plt.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.3), fancybox=False, shadow=False, fontsize=14)
plt.title('DER', fontsize=14)

#--- our method ----
plt.subplot(1,3,3)
plt.plot(xtrain, ytrain, 'k.', markersize=2, alpha=0.2, zorder=0, label='Data')
plt.plot(x2, x2**3, 'r-', linewidth=lw, label='Ground Truth')
plt.plot(x2, pred, 'b--', linewidth=lw, label='Prediction')
plt.fill_between(x2, lower2, upper2,
            alpha=0.4,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="PI")
plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.fill_between([-7, -4], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0, label='No data')
plt.fill_between([4, 7], [-150, -150], [150, 150], edgecolor=None, facecolor='k', alpha=0.1, zorder=0)
plt.ylim(-150, 150)
plt.xlim(-7, 7)
plt.xticks(np.arange(-6, 8, 2))
plt.title('PI3NN', fontsize=14)
plt.subplots_adjust(bottom=0.2, wspace=0.3)

# plt.tight_layout()
plt.savefig('Demo.png', pad_inches=0.1, dpi=400)
plt.show()
