### Prerequisite
To run the code, make sure these packages are installed in addition to the commonly used Numpy, Pandas, Matplotlib, sklearn, etc. 
--- python (>=3.8, version 3.8.3 is used in this study)
--- TensorFlow (>=2.0, version 2.4.1 is used in this study)
--- PyTorch (>=1.7, version 1.7.1 is used in this study)

The detailed environment configurations can be found in the 'environment.yml' file.

### Download the data sets
Due to the supplemental material size limit, we prepared the download URL for obtain the UCI data sets and our pre-split of the UCI data sets.

There are three approaches to get the data:

--- (1) Simply run the scirpt sh download_data.sh in the root folder ('./PI3NN_Supplemental_Material/') to download and unzip the data
--- (2) Manual download the zipped data through the links below:
----- (2.1) UCI datasets:            https://figshare.com/s/53cdf79be5ba5d216ba8
----- (2.2) Pre-split UCI datasets:  https://figshare.com/s/e470b7131b55df1b074e
----- (2.3) Unzip the two files and place them in the root folder './PI3NN_Supplemental_Material/'
--- (3) Have the UCI data sets in hand (should be in UCI\_datasets folder, either prepared yourself or download through using the link from (2.1)), and run the 'UCI\_data\_splitting.py

### Run the code
The following folders contain the code and pre-generated data to reproduce the results:
--- (1) Figure_1: tests on cubic function from PI3NN, DER and QD method.
--- (2) Figure_2: tests on sensitivity of hyper-parameters for QD method.
--- (3) Figure_3: OOD identification comparison between PI3NN, QD and DER method.
--- (4) Table_1: performance evaluation of the PI3NN, QD and DER method.

You can find detailed instructions (README files) on running the code and reproduce the plots within those folders, some of them are also available in the form of comments within the code.


For the results in the Appendix:
--- The code for generating the MPIW, RMSE sensitivity analysis for QD method is in ./Figure\_2/QD\_sensitivity\_tests/


Have fun!

