# electric-drive-assembly-temperature-prediction
Transient temperature field prediction model of electric drive assembly based on graph convolutional neural network
This is a TensorFlow implementation of OLS-RGCN: spatio-temporal relational graph convolutional neural network combined with ordinary least square method

## Requirements:
* tensorflow
* scipy
* numpy
* pandas
* math
*os

## Run the demo
Python main.py

Our baselines included: <br>
(1) Temporal Graph Convolutional Network model based on Ordinary Least Square Method(OLS-TGCN)<br>
(2) Gated Recurrent Unit model based on Ordinary Least Square Method(OLS-GRU)<br>

The OLS-TGCN and OLS-GRU models were in Main_TGCN.py and Main_GRU.py respective.


## Implement
In this paper, we set time interval as 10 seconds, 15 seconds, 20 seconds, 25 seconds and 30 seconds.

In the water-cooled EDA and oil-cooled EDA dataset, we set the parameters seq_len to 5 and pre_len to 20, 30, 40, 50, 60, respectively.

## Data Description
There are one datasets in the data fold, the other oil-cooled EDA dataset. Another oil-cooled dataset is being supplemented for further testing and can be contacted by email if required.<br>
(1)  water-cooled EDA. The data set was obtained by testing a water-cooled single motor, which can be accessed by accessing " https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature".<br>
(2) oil-cooled EDA. The dataset was obtained by testing the EDA of dual motors on the EDA temperature test bench of Zhao Zhiguo's research group in Tongji University, including temperature node information, motor current, voltage, speed and torque information

In order to use the model, we need
* adjacency matrix "TNT_adj", which describes the spatial relationship between each nodes,  and it can be derived from a specific EDA or PMSM lumped parameter thermal network model.

# Citation
This repository is published in order to support reproducability of experiments from the published journal article [Short-Term Prediction Method of Transient Temperature Field Variation for PMSM in Electric Drive Gearbox Using Spatial-Temporal Relational Graph Convolutional Thermal Neural Network](https://ieeexplore.ieee.org/document/10232897) and Conference article [Transient Temperature Field Prediction of PMSM Based on Electromagnetic-Heat-Flow Multi-Physics Coupling and Data-Driven Fusion Modeling](https://saemobilus.sae.org/content/2023-01-7031/)
If you are using this code please cite as follows.
```
@ARTICLE{
  author={Peng Tang, Zhiguo zhao, and Haodi Li},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Short-Term Prediction Method of Transient Temperature Field Variation for PMSM in Electric Drive Gearbox Using Spatial-Temporal Relational Graph Convolutional Thermal Neural Network}, 
  year={2023},
  doi={10.1109/TIE.2023.3303650}
@Conference{
  author={Peng Tang, Zhiguo zhao, and Haodi Li},
  Conference={SAE 2023 Vehicle Powertrain Diversification Technology Forum}, 
  title={Transient Temperature Field Prediction of PMSM Based on Electromagnetic-Heat-Flow Multi-Physics Coupling and Data-Driven Fusion Modeling}, 
  year={2023},
  doi={https://doi.org/10.4271/2023-01-7031}
