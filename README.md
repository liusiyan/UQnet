### PI3NN
We utilize a novel prediction interval method to learn the prediction mean values, upper and lower bounds of the prediction intervals from three independently trained neural networks using only standard mean squared error (MSE) loss, for uncertainty quantification in regression tasks. Our method does not contain unusual hyperparameters either in the neural networks or in the loss function. The distributional assumption on data is not required. In addtion, out method can effectively identify out-of-distribution samples and reasonably quantify their uncertainty.

### Prerequisite
To run the code, make sure these packages are installed in addition to the commonly used Numpy, Pandas, Matplotlib, sklearn, etc. 
--- python (>=3.8, version 3.8.3 is used in this study)
--- TensorFlow (>=2.0, version 2.4.1 is used in this study)
--- PyTorch (>=1.7, version 1.7.1 is used in this study)

The detailed environment configurations can be found in the 'environment.yml' file.

### Download the data sets
The default example used in our experiments are based on UCI and pre-split UCI data sets, which can be obtained by:
--- (1) Simply run the scirpt sh download_data.sh in the datasets folder ('./PI3NN/datasets') to download and unzip the data
--- (2) Manual download the zipped data through the links below:
----- (2.1) UCI datasets:            https://figshare.com/s/53cdf79be5ba5d216ba8
----- (2.2) Pre-split UCI datasets:  https://figshare.com/s/e470b7131b55df1b074e
----- (2.3) Unzip the two files and place them in the datasets folder './PI3NN/datasets'

### User input data sets can be loaded by using user customized code integrated in the 'main_PI3NN.py' 


### Run the code
To run the code, simply run the main file:
python main_PI3NN.py
By default, the program will load the UCI data sets and conduct the training on the randomly splitted train/test data. Firstly, you can make some modifications on the initial parameters/hyperparameters:
```python
configs = {'data_name':'yacht',    
           'original_data_path': '/datasets/UCI_datasets/',
           'splitted_data_path': '/datasets/UCI_TrainTest_Split/',
           'split_seed': 1,                 # random seed for splitting train/test data
           'split_test_ratio': 0.1,         # ratio of the testing data during random split
           'seed': 10,                      # general random seed
           'num_neurons_mean_net': 100,     # number of neurons in hidden layer for the 'MEAN' network
           'num_neurons_up_down_net': 100,  # number of neurons in hidden layer for the 'UP'and 'DOWN' network
           'quantile': 0.95,                # target percentile for optimization step# target percentile for optimization step
           'Max_iter': 5000,
           'lr': [0.1, 0.1, 0.1],           # learning rate
           'optimizers': ['Adam', 'Adam', 'Adam'],
           'learning_decay': True,
           'exponential_decay': True,
           'decay_steps': 3000,
           'decay_rate': 0.8,
           'saveWeights': True,
           'loadWeights_test': False,
           'early_stop': True,
           'early_stop_start_iter': 500,
           'wait_patience': 300,
           'restore_best_weights': True,
           'save_loss_history': True,
           'save_loss_history_path': './Results_PI3NN/loss_history/',
           'plot_loss_history' : True,
           'plot_loss_history_path':'./Results_PI3NN/loss_curves/',
           'verbose': 1,
           'experiment_id': 9
          }
```
Then in this case, we initialize the data loader specifically for UCI data sets:
```python
dataLoader = CL_dataLoader(os.getcwd()+configs['original_data_path'])
X_data_load, Y_data_load = dataLoader.load_single_dataset(configs['data_name'])
```

And conduct the random train/test data splitting:
```python
xTrain, xTest, yTrain, yTest = train_test_split(X_data_load, Y_data_load, 
                                                test_size=configs['split_test_ratio'],
                                                random_state=configs['split_seed'],
                                                shuffle=True)
```
Normalize the data before training:
```python
''' Standardize inputs '''
xTrain, xTrain_mean, xTrain_std = dataLoader.standardizer(xTrain)
xTest = (xTest - xTrain_mean) / xTrain_std

yTrain, yTrain_mean, yTrain_std = dataLoader.standardizer(yTrain)
yTest = (yTest - yTrain_mean) / yTrain_std
```

The networks for mean values, upper and lower bounds are initialized independently:
```python
''' Create network instances'''
net_mean = UQ_Net_mean_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_mean_net'])
net_std_up = UQ_Net_std_TF2(num_inputs, num_outputs,  num_neurons=configs['num_neurons_up_down_net'])
net_std_down = UQ_Net_std_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_up_down_net'])
```

Initialize the trainer instance, and take the networks and normalized data as well as the configurations as inputs:
```python
trainer = CL_trainer(configs, net_mean, net_std_up, net_std_down, xTrain, yTrain, xTest, yTest)
```

Followed by the main training and optimization process:
Train three networks:
```python
trainer.train()           
```
Optimize the upper and lower boundary using bisection method:
```python
trainer.boundaryOptimization()  
```
Evaluate the trained networks on testing data set
```python
trainer.testDataPrediction()  
```
Metrics calculations, including the prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and MSE, RMSE, etc.
```python
trainer.capsCalculation()
```
The final results can be saved to txt file (/Results_PI3NN/):
```python
trainer.saveResultsToTxt()
```



## References

[1] Siyan Liu, Pei Zhang, Dan Lu, and Guannan Zhang. "PI3NN: Prediction intervals from three independently trained neural networks." arXiv preprint arXiv:2108.02327 (2021). https://arxiv.org/abs/2108.02327

[2] Pei Zhang, Siyan Liu, Dan Lu, Guannan Zhang, and Ramanan Sankaran. "A prediction interval method for uncertainty quantification of regression models". ICLR 2021 SimDL Workshop. https://simdl.github.io/files/52.pdf


Have fun!

