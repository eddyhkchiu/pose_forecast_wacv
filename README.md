# Action-Agnostic Human Pose Forecasting
WACV 2019

Hsu-kuang Chiu, Ehsan Adeli, Borui Wang, De-An Huang, Juan Carlos Niebles

https://arxiv.org/abs/1810.09676

### Dependencies
The code is written in the following environment:
* tensorflow 1.5.0
* keras 1.2.2
* h5py 2.8.0

### Human 3.6M Dataset: Download the code, the datasets, and the model checkpoints
```bash
git clone https://github.com/eddyhkchiu/pose_forecast_wacv
cd pose_forecast_wacv

mkdir data
cd data

wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip

wget https://eddyhkchiu.github.io/share/Penn_Action_Pose.tar.gz
tar xvzf Penn_Action_Pose.tar.gz 
rm Penn_Action_Pose.tar.gz

cd ..
```

### Human 3.6M Dataset: Evaluation using the checkpoints to reproduce the performance numbers of the paper
* TP-RNN basic (M=2, K=2): python src/tprnn_train_human.py --dataset human --learning_rate 0.01 --dropout_keep 1.0 --iterations 100006 --model basic --tprnn_scale 2 --tprnn_layers 2 --seq_length_out 25 --sample --load 92000 


### Human 3.6M Dataset: Training new models from scratch:
* TP-RNN basic (M=2, K=2): python src/tprnn_train_human.py --dataset human --learning_rate 0.01 --dropout_keep 1.0 --iterations 100000 --model basic --tprnn_scale 2 --tprnn_layers 2 --seq_length_out 25
* TP-RNN generic (M=3, K=2): python src/tprnn_train_human.py --dataset human --learning_rate 7e-05 --dropout_keep 0.5 --iterations 100000 --model generic --tprnn_scale 2 --tprnn_layers 3 --seq_length_out 25 

### Penn Action Dataset: Evaluation using the checkpoints to reproduce the performance numbers of the paper
* TP-RNN basic (M=2, K=2) (with zero init velocity): python src/tprnn_train_penn.py --dataset penn --learning_rate 0.01 --dropout_keep 1.0 --iterations 100087 --model basic --tprnn_scale 2 --tprnn_layers 2 --seq_length_in 2 --seq_length_out 16 --sample --load 63000

### Penn Action Dataset: Training new models from scratch:
* TP-RNN basic (M=2, K=2) (with zero init velocity): python src/tprnn_train_penn.py --dataset penn --learning_rate 0.01 --dropout_keep 1.0 --iterations 100000 --model basic --tprnn_scale 2 --tprnn_layers 2 --seq_length_in 2 --seq_length_out 16 

