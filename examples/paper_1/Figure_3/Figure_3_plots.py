
''' Plotting function for Figure 3 
To reproduce the plots, you can directly run this file to utilize the pre-generated data, or re-run the code:
----- (1) QD method: './QD_OOD/main_QD_FUN_OOD.py'
----- (2) DER method: './DER_OOD/DER_FUN_OOD.py'
----- (3) PI3NN method: ./PI3NN_OOD/PI3NN_OOD_Pytorch.py
----- Move all the generated results .txt files to upper level folder, and run this file to generate the plots.

Have fun! 
'''


import numpy as np
import matplotlib.pyplot as plt


''' For PI3NN (bias 10.0 and 0.0)'''
PI3NN_OOD_train_np_bias_10 = np.loadtxt('PI3NN_OOD_train_np_bias_10.0.txt', delimiter=',')
PI3NN_OOD_test_np_bias_10 = np.loadtxt('PI3NN_OOD_test_np_bias_10.0.txt', delimiter=',')
PI3NN_OOD_train_np_bias_0 = np.loadtxt('PI3NN_OOD_train_np_bias_0.0.txt', delimiter=',')
PI3NN_OOD_test_np_bias_0 = np.loadtxt('PI3NN_OOD_test_np_bias_0.0.txt', delimiter=',')

plt.plot(PI3NN_OOD_train_np_bias_10[:, 0], PI3NN_OOD_train_np_bias_10[:, 1], 'r.', label='Training data (in distribution)')
plt.plot(PI3NN_OOD_test_np_bias_10[:, 0], PI3NN_OOD_test_np_bias_10[:, 1],  'b.', label='testing data (out of distribution')
plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
plt.ylabel('The Confidence Score')
plt.legend(loc='lower left')
plt.title('PI3NN with OOD detection')
plt.ylim(0, 1.2)
plt.savefig('PI3NN_OOD_plot_1.png')
plt.clf()

plt.plot(PI3NN_OOD_train_np_bias_0[:, 0], PI3NN_OOD_train_np_bias_0[:, 1], 'r.', label='Training data (in distribution)')
plt.plot(PI3NN_OOD_test_np_bias_0[:, 0], PI3NN_OOD_test_np_bias_0[:, 1],  'b.', label='testing data (out of distribution')
plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
plt.ylabel('The Confidence Score')
plt.legend(loc='lower left')
plt.title('PI3NN without OOD detection')
plt.ylim(0, 1.2)
plt.savefig('PI3NN_OOD_plot_2.png')
plt.clf()


''' For QD '''
QD_OOD_train_np = np.loadtxt('QD_OOD_train_np.txt', delimiter=',')
QD_OOD_test_np = np.loadtxt('QD_OOD_test_np.txt', delimiter=',')

# print(QD_OOD_train_np)
plt.plot(QD_OOD_train_np[:, 0], QD_OOD_train_np[:, 1], 'r.', label='Training data (in distribution)')
plt.plot(QD_OOD_test_np[:, 0], QD_OOD_test_np[:, 1],  'b.', label='testing data (out of distribution')
plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
plt.ylabel('The Confidence Score')
plt.legend(loc='lower left')
plt.title('QD')
plt.ylim(0, 1.2)
plt.savefig('QD_OOD_plot.png')
plt.clf()
# plt.show()


''' For DER  '''
DER_OOD_train_np = np.loadtxt('DER_OOD_train_np.txt', delimiter=',')
DER_OOD_test_np = np.loadtxt('DER_OOD_test_np.txt', delimiter=',')

plt.plot(DER_OOD_train_np[:, 0], DER_OOD_train_np[:, 1], 'r.', label='Training data (in distribution)')
plt.plot(DER_OOD_test_np[:, 0], DER_OOD_test_np[:, 1],  'b.', label='testing data (out of distribution')
plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
plt.ylabel('The Confidence Score')
plt.legend(loc='lower left')
plt.title('DER')
plt.ylim(0, 1.2)
plt.savefig('DER_OOD_plot.png')
plt.clf()
# plt.show()


