# Convolutional Social Pooling

Code for model proposed in [1] Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018

This code is used as a baseline and starting point for further code evolutions and experiments with seq2seq, attention and transformer models.

### Dependencies

We recommend using python3. You may find convenient to use a virtual env.

```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with ```deactivate```.


# Dataset Preprocessing

### NGSIM dataset
  
From NGSIM website:  
* Register at https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj  
* Download US-101-LosAngeles-CA.zip and I-80-Emeryville-CA.zip  
* Unzip and extract vehicle-trajectory-data into raw/us-101 and raw/i-80  
  
From googledrive:  
* Download i-80: https://drive.google.com/open?id=19ovxiJLCnS1ar1dYvUIrfdz_UnGejK-k  
* Download us-101: https://drive.google.com/open?id=14dMKew22_5evfOoSGEBYqHP3L92e7nyJ  
  
Dataset fields:  
* doc/trajectory-data-dictionary.htm  

### Reference .mat files
Obtained with preprocess_data.m (legacy) or preprocess_data_faster.m (much faster) applied to above NGSIM dataset    
https://drive.google.com/open?id=1xxAmnsn_sROUjvJiNWetCQ7odLNFA_Zt  

# Running experiments

### Training  

```bash
python train.py
```
Using a GPU is highly recommended due to the huge speedup.

### Evaluating 

```bash
python evaluate.py
```

### Experiments results

In https://github.com/PhilippeW83440/conv-social-pooling/tree/master/experiments  
