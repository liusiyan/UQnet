### Prerequisite
To run the code, make sure these packages are installed in addition to the commonly used Numpy, Pandas, Matplotlib, sklearn, etc. 
--- python (>=3.8, version 3.8.3 is used in this study)
--- TensorFlow (>=2.0, version 2.4.1 is used in this study)
--- PyTorch (>=1.7, version 1.7.1 is used in this study)

The detailed environment configurations can be found in the 'environment.yml' file.

### Download the data sets
The UCI and pre-split UCI data sets can be obtained by:
--- (1) Simply run the scirpt sh download_data.sh in the datasets folder ('./PI3NN/datasets') to download and unzip the data
--- (2) Manual download the zipped data through the links below:
----- (2.1) UCI datasets:            https://figshare.com/s/53cdf79be5ba5d216ba8
----- (2.2) Pre-split UCI datasets:  https://figshare.com/s/e470b7131b55df1b074e
----- (2.3) Unzip the two files and place them in the datasets folder './PI3NN/datasets'

### User input data sets can be loaded by using user customized code integrated in the 'main_PI3NN.py' 


### Run the code
python main_PI3NN.py 
We are working on more detailed descriptions.

## References

[1] Siyan Liu, Pei Zhang, Dan Lu, and Guannan Zhang. "PI3NN: Prediction intervals from three independently trained neural networks." arXiv preprint arXiv:2108.02327 (2021). https://arxiv.org/abs/2108.02327

[2] Pei Zhang, Siyan Liu, Dan Lu, Guannan Zhang, and Ramanan Sankaran. "A prediction interval method for uncertainty quantification of regression models". ICLR 2021 SimDL Workshop. https://simdl.github.io/files/52.pdf


Have fun!

