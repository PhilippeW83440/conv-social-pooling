from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pdb

import copy

#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):


	def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3), newFeats=0):
		self.newFeats = newFeats
		self.D = scp.loadmat(mat_file)['traj']
		self.T = scp.loadmat(mat_file)['tracks']
		self.t_h = t_h	# length of track history
		self.t_f = t_f	# length of predicted trajectory
		self.d_s = d_s	# down sampling rate of all sequences
		self.enc_size = enc_size # size of encoder LSTM
		self.grid_size = grid_size # size of social context grid

		self.Tx = self.t_h//self.d_s + 1

	def __len__(self):
		return len(self.D)



	def __getitem__(self, idx):

		dsId = self.D[idx, 0].astype(int)
		vehId = self.D[idx, 1].astype(int)
		t = self.D[idx, 2]
		grid = self.D[idx, self.newFeats + 8:]
		neighbors = []

		# Get track history 'hist' = ndarray, and future track 'fut' = ndarray
		hist = self.getHistory(vehId,t,vehId,dsId)

		# ACHTUNG: this may be slow (check later)
		hist_grid = np.zeros((self.Tx, self.grid_size[0], self.grid_size[1]))
		i = idx; count = 1
		while i >= 0 and i > idx - self.Tx:
			past_grid = self.D[i, self.newFeats + 8:] # First one is actually current grid
			past_grid = np.reshape(past_grid, (self.grid_size[1], self.grid_size[0])).transpose()
			past_grid = (past_grid > 0).astype(int) # just an occupancy grid
			hist_grid[-count, :] = past_grid
			i -= 1
			count += 1

		fut = self.getFuture(vehId,t,dsId)

		# Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
		for i in grid:
			neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

		# Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
		lon_enc = np.zeros([2])
		lon_enc[int(self.D[idx, self.newFeats + 7] - 1)] = 1
		lat_enc = np.zeros([3])
		lat_enc[int(self.D[idx, self.newFeats + 6] - 1)] = 1

		return hist,fut,neighbors,lat_enc,lon_enc,hist_grid



	## Helper function to get track history
	def getHistory(self,vehId,t,refVehId,dsId):
		if vehId == 0:
			return np.empty([0,2 + self.newFeats ])
		else:
			if self.T.shape[1]<=vehId-1:
				return np.empty([0,2 + self.newFeats])
			refTrack = self.T[dsId-1][refVehId-1].transpose()
			vehTrack = self.T[dsId-1][vehId-1].transpose()
			refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

			if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
				return np.empty([0,2 + self.newFeats])
			else:
				stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
				enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
				# ACHTUNG copy is mandatory !
				hist = copy.copy(vehTrack[stpt:enpt:self.d_s,1:3 + self.newFeats]) # 0:Time, 1:X, 2:Y, 3:V or A
				hist[:,0:2] = hist[:,0:2] - refPos
				#hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

			if len(hist) < self.t_h//self.d_s + 1:
				return np.empty([0,2 + self.newFeats])
			return hist



	## Helper function to get track future
	def getFuture(self, vehId, t,dsId):
		vehTrack = self.T[dsId-1][vehId-1].transpose()
		refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
		stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
		enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
		fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
		return fut



	## Collate function for dataloader
	def collate_fn(self, samples):

		# Initialize neighbors and neighbors length batches:
		nbr_batch_size = 0
		for _,_,nbrs,_,_,_ in samples:
			nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])

		maxlen = self.t_h//self.d_s + 1
		nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2 + self.newFeats)

		# Initialize social mask batch:
		pos = [0, 0]
		mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
		mask_batch = mask_batch.byte()


		# Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
		hist_batch = torch.zeros(maxlen,len(samples),2 + self.newFeats)
		fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
		op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
		lat_enc_batch = torch.zeros(len(samples),3)
		lon_enc_batch = torch.zeros(len(samples), 2)
		hist_grid_batch = torch.zeros(len(samples), self.Tx, self.grid_size[0], self.grid_size[1])


		count = 0
		for sampleId,(hist, fut, nbrs, lat_enc, lon_enc, hist_grid) in enumerate(samples):

			# Set up history, future, lateral maneuver and longitudinal maneuver batches:
			#hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
			#hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
			for feat in range(2 + self.newFeats):
				hist_batch[0:len(hist),sampleId, feat] = torch.from_numpy(hist[:, feat])

			fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
			fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
			op_mask_batch[0:len(fut),sampleId,:] = 1

			lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
			lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

			hist_grid_batch[sampleId, :, :, :] = torch.from_numpy(hist_grid)

			# Set up neighbor, neighbor sequence length, and mask batches:
			for id,nbr in enumerate(nbrs):
				if len(nbr)!=0:
					#nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
					#nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
					for feat in range(2 + self.newFeats):
						nbrs_batch[0:len(nbr),count, feat] = torch.from_numpy(nbr[:, feat])

					pos[0] = id % self.grid_size[0]
					pos[1] = id // self.grid_size[0]
					mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
					count+=1

		#print("DBG:", count, ", ", nbr_batch_size)

		# Example with batch_size=128 grid=(13, 3) and encoding of every traj history of 3 secs of (X,Y)rel into 64 features
		# out of RNN-LSTM h_n output
		# ------------------------------------------------------------------------------------
		# Usage of masks [128, 3, 13, 64] and nbrs_enc [850, 64] in model.py soc_enc is tricky
		# ------------------------------------------------------------------------------------
		# We must tag e.g. 850 times 64 ones in [128, 3, 13, 64] to retrieve the LSTM encoding of 850 neighbors history of 3 sec ...
		# with 850 == count == nbr_batch_size
		# 850/128 => here in this batch, on average 6.6 cars in the (13,3) grid
		assert count==nbr_batch_size, "Otherwise in model.py soc_enc.masked_scatter_(masks, nbrs_enc) WILL NOT MATCH !!!" 

		return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, hist_grid_batch

#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
	muX = x[:,:,0:1]
	muY = x[:,:,1:2]
	sigX = x[:,:,2:3]
	sigY = x[:,:,3:4]
	rho = x[:,:,4:5]
	sigX = torch.exp(sigX)
	sigY = torch.exp(sigY)
	rho = torch.tanh(rho)
	out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
	return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
	acc = torch.zeros_like(mask)
	muX = y_pred[:,:,0]
	muY = y_pred[:,:,1]
	sigX = y_pred[:,:,2]
	sigY = y_pred[:,:,3]
	rho = y_pred[:,:,4]
	ohr = torch.pow(1-torch.pow(rho,2),-0.5)
	x = y_gt[:,:, 0]
	y = y_gt[:,:, 1]
	out = torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr)
	acc[:,:,0] = out
	acc[:,:,1] = out
	acc = acc*mask
	lossVal = torch.sum(acc)/torch.sum(mask)
	return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
	if use_maneuvers:
		acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
		count = 0
		for k in range(num_lon_classes):
			for l in range(num_lat_classes):
				wts = lat_pred[:,l]*lon_pred[:,k]
				wts = wts.repeat(len(fut_pred[0]),1)
				y_pred = fut_pred[k*num_lat_classes + l]
				y_gt = fut
				muX = y_pred[:, :, 0]
				muY = y_pred[:, :, 1]
				sigX = y_pred[:, :, 2]
				sigY = y_pred[:, :, 3]
				rho = y_pred[:, :, 4]
				ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
				x = y_gt[:, :, 0]
				y = y_gt[:, :, 1]
				out = -(torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr))
				acc[:, :, count] =	out + torch.log(wts)
				count+=1
		acc = -logsumexp(acc,dim = 2)
		acc = acc * op_mask[:,:,0]
		if avg_along_time:
			lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
			return lossVal
		else:
			lossVal = torch.sum(acc,dim=1)
			counts = torch.sum(op_mask[:,:,0],dim=1)
			return lossVal,counts
	else:
		acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
		y_pred = fut_pred
		y_gt = fut
		muX = y_pred[:, :, 0]
		muY = y_pred[:, :, 1]
		sigX = y_pred[:, :, 2]
		sigY = y_pred[:, :, 3]
		rho = y_pred[:, :, 4]
		ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
		x = y_gt[:, :, 0]
		y = y_gt[:, :, 1]
		out = torch.pow(ohr, 2) * (
		torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(
			sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
		acc[:, :, 0] = out
		acc = acc * op_mask[:, :, 0:1]
		if avg_along_time:
			lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
			return lossVal
		else:
			lossVal = torch.sum(acc[:,:,0], dim=1)
			counts = torch.sum(op_mask[:, :, 0], dim=1)
			return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
	acc = torch.zeros_like(mask)
	muX = y_pred[:,:,0]
	muY = y_pred[:,:,1]
	x = y_gt[:,:, 0]
	y = y_gt[:,:, 1]
	out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
	acc[:,:,0] = out
	acc[:,:,1] = out
	acc = acc*mask
	lossVal = torch.sum(acc)/torch.sum(mask)
	return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
	acc = torch.zeros_like(mask)
	muX = y_pred[:, :, 0]
	muY = y_pred[:, :, 1]
	x = y_gt[:, :, 0]
	y = y_gt[:, :, 1]
	out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
	acc[:, :, 0] = out
	acc[:, :, 1] = out
	acc = acc * mask
	lossVal = torch.sum(acc[:,:,0],dim=1)
	counts = torch.sum(mask[:,:,0],dim=1)
	return lossVal, counts

## Quantify and catch out Big Errors (+ or - 3 stddev)
def maskedBIGERRTest(y_pred, y_gt, mask):
	acc = torch.zeros_like(mask)
	muX = y_pred[:,:, 0]
	muY = y_pred[:,:, 1]
	# ACHTUNG !!! what is called sigX/sigY is the inverse in the code
	sigX = 1/y_pred[:,:, 2]
	sigY = 1/y_pred[:,:, 3]

	# sigX typically between 0 (at 0+ sec) and 2  feets (at 5 sec)
	# sigY typically between 0 (at 0+ sec) and 15 feets (at 5 sec)

	# n=1 in between 20% and 33%
	# n=2 arround 6%
	# n=3 arround 2%

	n=3
	minX = muX - n*sigX
	maxX = muX + n*sigX
	minY = muY - n*sigY
	maxY = muY + n*sigY

	x = y_gt[:,:, 0]
	y = y_gt[:,:, 1]

	errX = (x < minX) + (x > maxX)
	errY = (y < minY) + (y > maxY)
	out  = ((errX + errY) > 0) # [Ty, Batch] array of 0 or 1

	acc[:, :, 0] = out
	acc[:, :, 1] = out
	acc = acc * mask
	lossVal = torch.sum(acc[:,:,0],dim=1)
	counts = torch.sum(mask[:,:,0],dim=1)
	return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
	if dim is None:
		inputs = inputs.view(-1)
		dim = 0
	s, _ = torch.max(inputs, dim=dim, keepdim=True)
	outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
	if not keepdim:
		outputs = outputs.squeeze(dim)
	return outputs
