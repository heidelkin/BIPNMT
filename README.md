# BIPNMT

## This repository is still under construction. Further update is coming

Code for our paper "A Reinforcement Learning Approach to Interactive-Predictive Neural Machine Translation"

The code is written based on Nguyen et.al, 2017's code:
https://github.com/khanhptnk/bandit-nmt/

Requirements:  
* Python 3.5
* PyTorch 0.3
------
1. Download and create the vocabulary
* Go to the scripts folder and run:
  * download_data.sh
  * make_data.sh fr en
2. To train the model; you can follow a sample script in scripts folder
* run_train.sh
3. To evaluate the trained models
* use the -eval flag (please refer to the run_eval.sh)

## Notes: For BIP-NMT, it takes about 25 mins for training 1000 sentences under a Nvidia Tesla P40 - 24GB RAM GPU.
