# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import importlib
import os
import pandas as pd


# import random
# seed = 1 # 2
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)



"""
this file contains object to create data sets for regression

synthetic datasets: 
drunk_bow_tie - as in paper, with gaussian noise
drunk_bow_tie_exp - as in paper with exp noise
x_cubed_gap - as in paper to show model uncertainty

real datasets:
~boston - standard boston housing dataset

"""

''' Add the Train/Test/Valid splitting instead of only Train/Valid splitting 
	Add UCI splitted data loading functions
'''

class DataGenerator:
	def __init__(self, type_in, n_feat=1):
		# select type of data to produce
		# not really using no. feat anymore

		self.n_feat = n_feat
		self.type_in = type_in

		return


	def CreateData(self, n_samples, seed_in=5, 
		train_prop=0.9, bound_limit=6., n_std_devs=1.96,**kwargs):

		# np.random.seed(seed_in)
		scale_c=1.0 # default
		shift_c=1.0

		# for ideal boundary
		X_ideal = np.linspace(start=-bound_limit,stop=bound_limit, num=500)
		y_ideal_U = np.ones_like(X_ideal)+1. # default
		y_ideal_L = np.ones_like(X_ideal)-1.
		y_ideal_mean = np.ones_like(X_ideal)+0.5

		if self.type_in=="drunk_bow_tie":
			"""
			similar to bow tie but less linear
			"""	

			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.
			X_train = X
			y_train = y	

			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.		
			X_val = X
			y_val = y

			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + n_std_devs*np.power(X_ideal,2)
			y_ideal_U = y_ideal_U/5.
			y_ideal_L = 1.5*np.sin(np.pi*X_ideal) - n_std_devs*np.power(X_ideal,2)
			y_ideal_L = y_ideal_L/5.
			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_mean = y_ideal_mean/5.	

			# overwrite for convenience!
			X_val = X_train
			y_val = y_train

		elif self.type_in=="drunk_bow_tie_exp":
			"""
			similar to bow tie but less linear, now with non-gaussian noise
			"""	

			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.
			X_train = X
			y_train = y	

			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.		
			X_val = X
			y_val = y

			# for exponential quantile = ln(1/quantile) /lambda
			# note that np inputs beta = 1/lambda
			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + np.log(1/(1-0.95))*np.power(X_ideal,2)
			y_ideal_U = y_ideal_U/5.
			y_ideal_L = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_L = y_ideal_L/5.
			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_mean = y_ideal_mean/5.	

			X_val = X_train
			y_val = y_train

		elif self.type_in=="periodic_1":
			"""
			creates a bow tie shape with changing variance
			"""
			X = np.random.uniform(low=-5.,high=5.,size=(n_samples,self.n_feat))
			y = 2.1*np.cos(0.2*X[:,0]) + 0.7*np.cos(20.1*X[:,0]) + 0.2*np.cos(10.4*X[:,0]) + np.random.normal(loc=0.,scale=0.1*np.ones_like(X[:,0]))
			y = y.reshape([-1,1])/1.
			X_train = X
			y_train = y	
			X_val = X_train
			y_val = y_train
			# y_ideal_U = X_ideal/5. + n_std_devs * np.abs(X_ideal)/5.
			# y_ideal_L = X_ideal/5. - n_std_devs * np.abs(X_ideal)/5.

		elif self.type_in=="x_cubed_gap":
			"""
			toy data problem from Probabilistic Backprop (Lobato) & 
			deep ensembles (Blundell)
			but added gap here

			"""
			scale_c = 50.
			half_samp = int(round(n_samples/2))
			X_1 = np.random.uniform(low=-4.,high=-1.,size=(half_samp,1))
			X_2 = np.random.uniform(low=1.,high=4.,size=(n_samples - half_samp,1))
			X = np.concatenate((X_1, X_2))
			y = X[:,0]**3 + np.random.normal(loc=0.,scale=3., size=X[:,0].shape[0])
			y = y.reshape([-1,1])/scale_c
			X_train = X
			y_train = y			
			X_val = X_train
			y_val = y_train

			y_ideal_U = X_ideal**3 + n_std_devs*3.
			y_ideal_U = y_ideal_U/scale_c
			y_ideal_L = X_ideal**3 - n_std_devs*3.
			y_ideal_L = y_ideal_L/scale_c
			y_ideal_mean = X_ideal**3
			y_ideal_mean = y_ideal_mean/scale_c

		# use single char '~' at start to identify real data sets
		elif self.type_in[:1] == '~':

			if self.type_in=="~boston":
				path = 'boston_housing_data.csv'
				data = np.loadtxt(path,skiprows=0)
			elif self.type_in=="~concrete":
				path = 'Concrete_Data.csv'
				data = np.loadtxt(path, delimiter=',',skiprows=1)

		# Add by Dan Lu for 1000 ELM data 		
			elif self.type_in=="~ELM":  
				path = 'ELM_out15.dat'
				data = np.loadtxt(path)[:,:9]
				print(data.shape)

			# Add for UCI datasets (in the original QD loading scheme)
			# The train/test/valid data splitted data loading is implemented in another function
			# LoadData_Splitted_UCI
			elif self.type_in=="~energy":
				data_df = pd.read_excel('UCI_datasets/energy-efficiency/ENB2012_data.xlsx', engine='openpyxl')
				data_df = data_df.dropna(how='all', axis='columns')
				data_df = data_df.dropna(how='all', axis='rows')
				data = data_df.values
				data = data_df.values

			elif self.type_in=="~kin8nm":
				data_df = pd.read_csv('UCI_datasets/kin8nm/dataset_2175_kin8nm.csv', sep=',')
				data = data_df.values

			elif self.type_in=="~naval":
				data = np.loadtxt('UCI_datasets/naval/data.txt')

			elif self.type_in=="~power":
				data_df = pd.read_excel('UCI_datasets/power-plant/Folds5x2_pp.xlsx', engine='openpyxl')
				data = data_df.values

			elif self.type_in=="~protein":
				data_df = pd.read_csv('UCI_datasets/protein/CASP.csv', sep=',')
				data = data_df.values

			elif self.type_in=="~wine":
				data_df = pd.read_csv('UCI_datasets/wine-quality/winequality-red.csv', sep=';')
				data = data_df.values

			elif self.type_in=="~yacht":
				data = np.loadtxt('UCI_datasets/yacht/yacht_hydrodynamics.data')

			elif self.type_in=="~MSD":
				with open('UCI_datasets/song/YearPredictionMSD.npy', 'rb') as f:
					data = np.load(f)


			# work out normalisation constants (need when unnormalising later)
			scale_c = np.std(data[:,-1])
			shift_c = np.mean(data[:,-1])

			# normalise data
			for i in range(0,data.shape[1]):
				# avoid zero variance features (exist one or two)
				sdev_norm = np.std(data[:,i])
				sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
				data[:,i] = (data[:,i] - np.mean(data[:,i]) )/sdev_norm

			# split into train/test
			perm = np.random.permutation(data.shape[0])
			train_size = int(round(train_prop*data.shape[0]))
			train = data[perm[:train_size],:]
			test = data[perm[train_size:],:]


			## modified x,y choosing to adjust the 'energy' and 'naval' datasets
			if self.type_in=="~energy" or self.type_in=="~naval":
				y_train = train[:,-2].reshape(-1,1)
				X_train = train[:,:-2]
				y_val = test[:,-2].reshape(-1,1)
				X_val = test[:,:-2]
			else:
				y_train = train[:,-1].reshape(-1,1)
				X_train = train[:,:-1]
				y_val = test[:,-1].reshape(-1,1)
				X_val = test[:,:-1]

			### original
			# y_train = train[:,-1].reshape(-1,1)
			# X_train = train[:,:-1]
			# y_val = test[:,-1].reshape(-1,1)
			# X_val = test[:,:-1]


		# save important stuff
		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_ideal = X_ideal
		self.y_ideal_U = y_ideal_U
		self.y_ideal_L = y_ideal_L
		self.y_ideal_mean = y_ideal_mean
		self.scale_c = scale_c
		self.shift_c = shift_c

		return X_train, y_train, X_val, y_val

	def LoadData_Splitted_UCI(self, loadCSVName, original_data_path, splitted_data_path, split_seed, **kwargs):

		## (1) Load the original data for the normalization purpose
		# current_dir = os.path.dirname(__file__)
		# uci_dir = os.path.join(current_dir, 'UCI_datasets')
		uci_dir = original_data_path
		if loadCSVName == 'boston':
			data = np.loadtxt(os.path.join(uci_dir, 'boston-housing/boston_housing.txt'))

		if loadCSVName == 'concrete':
			data_df = pd.read_excel(os.path.join(uci_dir, 'concrete/Concrete_Data.xls'))
			data = data_df.values

		if loadCSVName == 'energy':
			data_df = pd.read_excel(os.path.join(uci_dir, 'energy-efficiency/ENB2012_data.xlsx'), engine='openpyxl')
			data_df = data_df.dropna(how='all', axis='columns')
			data_df = data_df.dropna(how='all', axis='rows')
			data = data_df.values


		if loadCSVName == 'kin8nm':
			data_df = pd.read_csv(os.path.join(uci_dir, 'kin8nm/dataset_2175_kin8nm.csv'), sep=',')
			data = data_df.values

		if loadCSVName == 'naval':
			data = np.loadtxt(os.path.join(uci_dir, 'naval/data.txt'))

		if loadCSVName == 'power':
			data_df = pd.read_excel(os.path.join(uci_dir, 'power-plant/Folds5x2_pp.xlsx'), engine='openpyxl')
			data = data_df.values

		if loadCSVName == 'protein':
			data_df = pd.read_csv(os.path.join(uci_dir, 'protein/CASP.csv'), sep=',')
			# print(data_df)
			'''Move the Y data (originally located at the first column) to last column in order to keep consistency
			with the normalization process'''
			col_names = data_df.columns.tolist()
			col_names.append(col_names[0])
			del col_names[col_names.index(col_names[0])]
			# print(col_names)
			data_df = data_df[col_names]
			# print(data_df)
			data = data_df.values

		if loadCSVName == 'wine':
			data_df = pd.read_csv(os.path.join(uci_dir, 'wine-quality/winequality-red.csv'), sep=';')
			data = data_df.values

		if loadCSVName == 'yacht':
			data = np.loadtxt(os.path.join(uci_dir, 'yacht/yacht_hydrodynamics.data'))

		if loadCSVName == 'MSD':
			with open(os.path.join(uci_dir, 'song/YearPredictionMSD.npy'), 'rb') as f:
				data = np.load(f)


		## (2) Load the pre-splitted train/test data
		## 
		xyTrain_load = np.loadtxt(splitted_data_path+'xyTrain_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
		xyTest_load = np.loadtxt(splitted_data_path+'xyTest_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
		xyTrain_load = xyTrain_load.astype(np.float32)
		# xyValid_load = xyValid_load.astype(np.float32)
		xyTest_load = xyTest_load.astype(np.float32)

		# original normalization functions 
		# work out normalisation constants (need when unnormalising later)
		scale_c = np.std(data[:, -1])
		shift_c = np.mean(data[:, -1])

		# normalise data
		num_cols = xyTrain_load.shape[1]
		print('num cols: {}'.format(num_cols))

		for i in range(0, num_cols):
			# get the sdev_norm from original data
			sdev_norm = np.std(data[:, i])
			sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
			# apply on the pre-splitted data
			xyTrain_load[:, i] = (xyTrain_load[:, i] - np.mean(data[:, i]) )/sdev_norm
			xyTest_load[:, i]  = (xyTest_load[:, i] - np.mean(data[:, i]) )/sdev_norm
			# xyValid_load[:, i] = (xyValid_load[:, i] - np.mean(data[:, i]) )/sdev_norm

		if loadCSVName == 'energy' or loadCSVName == 'naval':
			xTrain = xyTrain_load[:, :-2]  ## all columns except last two columns as inputs
			yTrain = xyTrain_load[:, -1]  ## last column as output
			xTest = xyTest_load[:, :-2]
			yTest = xyTest_load[:, -1]
		else:
			xTrain = xyTrain_load[:, :-1]
			yTrain = xyTrain_load[:, -1]
			xTest = xyTest_load[:, :-1]
			yTest = xyTest_load[:, -1]

		self.scale_c = scale_c
		self.shift_c = shift_c

		return xTrain, yTrain, xTest, yTest




	def ViewData(self, n_rows=5, hist=False, plot=False, print_=True):
		"""
		print first few rows of data
		option to view histogram of x and y
		option to view scatter plot of x vs y
		"""
		if print_:
			print("\nX_train\n",self.X_train[:n_rows], 
				"\ny_train\n", self.y_train[:n_rows], 
				"\nX_val\n", self.X_val[:n_rows], 
				"\ny_val\n", self.y_val[:n_rows])

		if hist:
			fig, ax = plt.subplots(1, 2)
			ax[0].hist(self.X_train)
			ax[1].hist(self.y_train)
			ax[0].set_title("X_train")
			ax[1].set_title("y_train")
			fig.show()

		if plot:
			n_feat = self.X_train.shape[1]
			fig, ax = plt.subplots(n_feat, 1) # create an extra
			if n_feat == 1:	ax = [ax] # make into list
			for i in range(0,n_feat):
				ax[i].scatter(self.X_train[:,i],self.y_train,
					alpha=0.5,s=2.0)
				ax[i].set_xlabel('x_'+str(i))
				ax[i].set_ylabel('y')

			fig.show()

		return

		