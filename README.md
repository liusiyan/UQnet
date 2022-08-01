# PI3NN-LSTM: Out-of-distribution-aware prediction intervals from three neural networks --- for time-series data
<br/>

This is the development branch for PI3NN-LSTM. We use the vanilla LSTM network as the encoder to extract the information from time-series data, then we train PI3NN-MLP networks for UQ based on the intermediate predictions from the LSTM encoder. 

### Run the examples

First run the LSTM-encoder (use 'ph' data as an example):
```python
python main_PI3NN.py --data ph --mode lstm_encoder --project ./examples/streamflow_proj/ --exp ph --configs ./examples/streamflow_proj/ph/configs_encoder.json
```
And run the PI3NN-MLP
```python
python main_PI3NN.py --data ph --mode PI3NN_MLP --project ./examples/streamflow_proj/ --exp ph --configs ./examples/streamflow_proj/ph/configs_PI3NN.json
```

Similarly, you can run the encoder and PI3NN-MLP for 'Qui' and 'Rock' data
```python
python main_PI3NN.py --data Qui --mode lstm_encoder --project ./examples/streamflow_proj/ --exp Qui --configs ./examples/streamflow_proj/Qui/configs_encoder.json
python main_PI3NN.py --data Qui --mode PI3NN_MLP --project ./examples/streamflow_proj/ --exp Qui --configs ./examples/streamflow_proj/Qui/configs_PI3NN.json 
```

```python
python main_PI3NN.py --data Rock --mode lstm_encoder --project ./examples/streamflow_proj/ --exp Rock --configs ./examples/streamflow_proj/Rock/configs_encoder.json
python main_PI3NN.py --data Rock --mode PI3NN_MLP --project ./examples/streamflow_proj/ --exp Rock --configs ./examples/streamflow_proj/Rock/configs_PI3NN.json
```

Note: You will be loading the pre-trained models to generate results and plots, you can re-train the networks by setting: <br/>
(1)"save_encoder": true and "load_encoder": false in 'configs_encoder.json' for LSTM-encoder <br/>
(2)"save_PI3NN_MLP": true and "load_PI3NN_MLP": false in 'configs_PI3NN.json' for PI3NN-MLP <br/>
You will need to train (or load) encoder first to generate intermediate results before running the PI3NN-MLP. <br/>

More descriptions and documentations in progress...

## References

## References

[1] Siyan Liu, Pei Zhang, Dan Lu, and Guannan Zhang. "PI3NN: Out-of-distribution-aware Prediction Intervals from Three Neural Networks." International Conference on Learning Representations, 2022. https://openreview.net/forum?id=NoB8YgRuoFU

[2] Siyan Liu, Pei Zhang, Dan Lu, and Guannan Zhang. "PI3NN: Out-of-distribution-aware prediction intervals from three neural networks." arXiv preprint arXiv:2108.02327 (2021). https://arxiv.org/abs/2108.02327

[3] Pei Zhang, Siyan Liu, Dan Lu, Guannan Zhang, and Ramanan Sankaran. "A prediction interval method for uncertainty quantification of regression models". ICLR 2021 SimDL Workshop. https://simdl.github.io/files/52.pdf

[4] Pei Zhang, Siyan Liu, Dan Lu, Ramanan Sankaran, and Guannan Zhang. "An out-of-distribution-aware autoencoder model for reduced chemical kinetics." Discrete Continuous Dynamical Systems - S 15(4)(2022)913-930 https://www.aimsciences.org/article/doi/10.3934/dcdss.2021138 


Have fun!

