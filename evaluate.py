from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time

import pdb
import os
import utils_nn as utils
import logging

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='baseline', help="baseline, baselineX, seq2seq, seq2seqX, attention or transformer")

cmd_args = parser.parse_args()


## Network Arguments
## cf params.json

#args['encoder_size'] = 64
#args['decoder_size'] = 128
#args['in_length'] = 16
#args['out_length'] = 25
#args['grid_size'] = (13,3)
#args['soc_conv_depth'] = 64
#args['conv_3x1_depth'] = 16
#args['dyn_embedding_size'] = 32
#args['input_embedding_size'] = 32
#args['num_lat_classes'] = 3
#args['num_lon_classes'] = 2
#args['use_maneuvers'] = True

args = {}
args['train_flag'] = False
args['model_dir'] = 'experiments/' + cmd_args.experiment

utils.set_logger(os.path.join(args['model_dir'], 'evaluate.log'))

json_path = os.path.join(args['model_dir'], 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
params.grid_size = (params.grid_size_lon, params.grid_size_lat)

# use GPU if available
params.use_cuda = torch.cuda.is_available()
params.train_flag = args['train_flag']
params.model_dir = args['model_dir']


# Evaluation metric:
metric = 'nll'	#or rmse
metric = 'rmse'	#or rmse


# Initialize network
batch_size=128
batch_size=1024
net = highwayNet(params)

net_path = os.path.join(args['model_dir'], 'best.pth.tar')
assert os.path.isfile(net_path), "No net file found at {}".format(net_path)
#net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
utils.load_checkpoint(net_path, net)

if params.use_cuda:
	net = net.cuda()

# This corrects for the differences in dropout, batch normalization during training and testing.
# No dropout, batch norm so far; but it is a good default practice anyways
net.eval()

logging.info("Loading the datasets...")
if 'X' in cmd_args.experiment:
	tsSet = ngsimDataset('data/TestSetX.mat')
else:
	tsSet = ngsimDataset('data/TestSet.mat')

tsDataloader = DataLoader(tsSet,batch_size=batch_size,shuffle=False,num_workers=8,collate_fn=tsSet.collate_fn)

if params.use_cuda:
	lossVals = torch.zeros(25).cuda()
	counts = torch.zeros(25).cuda()
else:
	lossVals = torch.zeros(25)
	counts = torch.zeros(25)

total = len(tsDataloader)
with torch.no_grad():
	for i, data in tqdm(enumerate(tsDataloader), total=total):
		st_time = time.time()
		hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_grid = data
	
		# Initialize Variables
		if params.use_cuda:
			hist = hist.cuda()
			nbrs = nbrs.cuda()
			mask = mask.cuda()
			lat_enc = lat_enc.cuda()
			lon_enc = lon_enc.cuda()
			fut = fut.cuda()
			op_mask = op_mask.cuda()
			hist_grid = hist_grid.cuda()

		if metric == 'nll':
			# Forward pass
			if params.use_maneuvers:
				fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid)
				l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
			else:
				fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid)
				l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
		else:
			# Forward pass
			if params.use_maneuvers:
				fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid)
				fut_pred_max = torch.zeros_like(fut_pred[0])
				for k in range(lat_pred.shape[0]):
					lat_man = torch.argmax(lat_pred[k, :]).detach()
					lon_man = torch.argmax(lon_pred[k, :]).detach()
					indx = lon_man*3 + lat_man
					fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
				l, c = maskedMSETest(fut_pred_max, fut, op_mask)
			else:
				fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
				l, c = maskedMSETest(fut_pred, fut, op_mask)
	
		#logging.info("Batch {}: l {} c {}".format(i, l, c))
		lossVals +=l.detach()
		counts += c.detach()
		batch_time = time.time()-st_time
		#print("eval batch_time:", batch_time)

if metric == 'nll':
	logging.info("NLL: {}".format(lossVals / counts))
else:
	logging.info("RMSE: {}".format(torch.pow(lossVals / counts,0.5)*0.3048))	 # Calculate RMSE and convert from feet to meters


