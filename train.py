from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
import math

import pdb
import utils_nn as utils
import logging
import os
import random

#from torchsummary import summary

import pdb


## Network Arguments
## cf params.json
## Baseline params

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
# Full features: soc/grid + maneuvers
args['model_dir'] = 'experiments/baseline'
args['model_dir'] = 'experiments/baseline_bidir'
args['model_dir'] = 'experiments/seq2seq'
args['model_dir'] = 'experiments/seq2seq_bidir'
args['model_dir'] = 'experiments/attention'

# Limited features: 1 at a time
args['model_dir'] = 'experiments/transformer_simple'
args['model_dir'] = 'experiments/transformer_grid'
args['model_dir'] = 'experiments/transformer_maneuver'

# Full features: soc/grid + maneuvers
args['model_dir'] = 'experiments/transformer'
args['model_dir'] = 'experiments/baseline'
args['model_dir'] = 'experiments/transformer_simple'

args['train_flag'] = True
args['restore_file'] = None # or 'last' or 'best'


utils.set_logger(os.path.join(args['model_dir'], 'train.log'))

json_path = os.path.join(args['model_dir'], 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
params.grid_size = (params.grid_size_lon, params.grid_size_lat)

# use GPU if available
params.use_cuda = torch.cuda.is_available()
params.train_flag = args['train_flag']
params.model_dir = args['model_dir']
params.model_dir = args['restore_file']

print("\nEXPERIMENT:", args['model_dir'], "\n")


# Initialize network
batch_size = 128
# just when using pdb.set_trace() to discriminate batch_size more easily in tensor shape dumps
#batch_size = 127
net = highwayNet(params, batch_size)
print(net)
if params.use_cuda:
	net = net.cuda()

# Set the random seed for reproducible experiments
random.seed(30)
torch.manual_seed(230)
if params.use_cuda: torch.cuda.manual_seed(230)

# This corrects for the differences in dropout, batch normalization during training and testing.
# No dropout, batch norm so far; but it is a good default practice anyways
net.train()

## Initialize optimizer
pretrainEpochs = 5
trainEpochs = 3
optimizer = torch.optim.Adam(net.parameters())
crossEnt = torch.nn.BCELoss()


## Initialize data loaders
logging.info("Loading the datasets...")
trSet = ngsimDataset('data/TrainSet.mat')
valSet = ngsimDataset('data/ValSet.mat')
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)


# reload weights from restore_file if specified
if args['restore_file'] is not None:
	restore_path = os.path.join(args['model_dir'], args['restore_file'] + '.pth.tar')
	logging.info("Restoring parameters from {}".format(restore_path))
	utils.load_checkpoint(restore_path, net, optimizer)


## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf
best_val_loss = math.inf

for epoch_num in range(pretrainEpochs+trainEpochs):
	logging.info("Epoch {}/{}".format(epoch_num + 1, pretrainEpochs+trainEpochs))
	if epoch_num == 0:
		logging.info('Pre-training with MSE loss')
	elif epoch_num == pretrainEpochs:
		logging.info('Training with NLL loss')


	## Train:_____________________________________________________________________
	net.train_flag = True
	net.train() # This is super important for Transformer ... as it uses dropouts...

	# Variables to track training performance:
	avg_tr_loss = 0
	avg_tr_time = 0
	avg_lat_acc = 0
	avg_lon_acc = 0


	for i, data in enumerate(trDataloader):

		st_time = time.time()
		hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_grid = data

		if params.use_cuda:
			hist = hist.cuda()
			nbrs = nbrs.cuda()
			mask = mask.cuda()
			lat_enc = lat_enc.cuda()
			lon_enc = lon_enc.cuda()
			fut = fut.cuda()
			op_mask = op_mask.cuda()
			hist_grid = hist_grid.cuda()

		# Forward pass
		if params.use_maneuvers:
			fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid, fut)
			# Pre-train with MSE loss to speed up training
			if epoch_num < pretrainEpochs:
				l = maskedMSE(fut_pred, fut, op_mask)
			else:
			# Train with NLL loss
				l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
				avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
				avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
		else:
			fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid, fut)
			if epoch_num < pretrainEpochs:
				l = maskedMSE(fut_pred, fut, op_mask)
			else:
				l = maskedNLL(fut_pred, fut, op_mask)

		#print("LOSS:", l)

		# Backprop and update weights
		optimizer.zero_grad()
		l.backward()
		a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
		optimizer.step()

		# Track average train loss and average train time:
		batch_time = time.time()-st_time
		avg_tr_loss += l.item()
		avg_tr_time += batch_time

		#print("batch_time:", batch_time)

		if i%100 == 99:
			eta = avg_tr_time/100*(len(trSet)/batch_size-i)

			logging.info("Epoch no: {} | Epoch progress(%): {:0.2f} | Avg train loss: {:0.4f} | Acc: {:0.4f} {:0.4f} | Validation loss prev epoch {:0.4f} | ETA(s): {}".format(epoch_num+1, i/(len(trSet)/batch_size)*100, avg_tr_loss/100, avg_lat_acc, avg_lon_acc, prev_val_loss, int(eta)))

			train_loss.append(avg_tr_loss/100)
			avg_tr_loss = 0
			avg_lat_acc = 0
			avg_lon_acc = 0
			avg_tr_time = 0
	# ______________________________________________________________________________



	## Validate:____________________________________________________________________
	net.train_flag = False
	net.eval() # This is super important for Transformer ... as it uses dropouts...

	logging.info("Epoch {} complete. Calculating validation loss...".format(epoch_num+1))

	avg_val_loss = 0
	avg_val_lat_acc = 0
	avg_val_lon_acc = 0
	val_batch_count = 0
	total_points = 0

	for i, data  in enumerate(valDataloader):
		st_time = time.time()
		hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_grid = data


		if params.use_cuda:
			hist = hist.cuda()
			nbrs = nbrs.cuda()
			mask = mask.cuda()
			lat_enc = lat_enc.cuda()
			lon_enc = lon_enc.cuda()
			fut = fut.cuda()
			op_mask = op_mask.cuda()
			hist_grid = hist_grid.cuda()

		# Forward pass
		if params.use_maneuvers:
			if epoch_num < pretrainEpochs:
				# During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
				net.train_flag = True
				fut_pred, _ , _ = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid, fut)
				l = maskedMSE(fut_pred, fut, op_mask)
			else:
				# During training with NLL loss, validate with NLL over multi-modal distribution
				fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
				l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
				avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
				avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
		else:
			fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc, hist_grid, fut)
			if epoch_num < pretrainEpochs:
				l = maskedMSE(fut_pred, fut, op_mask)
			else:
				l = maskedNLL(fut_pred, fut, op_mask)

		avg_val_loss += l.item()
		val_batch_count += 1

		#batch_time = time.time()-st_time
		#print("val batch_time:", batch_time)

	logging.info("Validation loss : {:0.4f} | Val Acc: {:0.4f} {:0.4f}".format(avg_val_loss/val_batch_count, avg_val_lat_acc/val_batch_count*100, avg_val_lon_acc/val_batch_count*100))
	val_loss.append(avg_val_loss/val_batch_count)
	prev_val_loss = avg_val_loss/val_batch_count

	# Save weights
	nn_val_loss = avg_val_loss/val_batch_count
	is_best = nn_val_loss < best_val_loss
	utils.save_checkpoint({'epoch': epoch_num + 1,
							'state_dict': net.state_dict(),
							'optim_dict' : optimizer.state_dict(), 
							'val_loss': avg_val_loss/val_batch_count,
							'val_lat_acc': avg_val_lat_acc/val_batch_count*100,
							'val_lon_acc': avg_val_lon_acc/val_batch_count*100 },
							is_best=is_best,
							checkpoint = args['model_dir'])

	# If best_eval, best_save_path        
	if is_best:
		logging.info("- Found new best val_loss")
		best_val_loss = nn_val_loss


	#_____________________________________________________________________________

torch.save(net.state_dict(), 'trained_models/cslstm_m.tar')



