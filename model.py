from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import pdb
import transformer as tsf

class highwayNet(nn.Module):

	## Initialization
	def __init__(self,params):
		super(highwayNet, self).__init__()

		## Unpack arguments
		self.params = params

		## Use gpu flag
		self.use_cuda = params.use_cuda

		# Flag for maneuver based (True) vs uni-modal decoder (False)
		self.use_maneuvers = params.use_maneuvers
		self.use_transformer = params.use_transformer
		self.use_grid = params.use_grid

		# Flag for train mode (True) vs test-mode (False)
		self.train_flag = params.train_flag

		## Sizes of network layers
		self.encoder_size = params.encoder_size
		self.decoder_size = params.decoder_size
		self.in_length = params.in_length
		self.out_length = params.out_length
		self.grid_size = params.grid_size
		self.soc_conv_depth = params.soc_conv_depth
		self.conv_3x1_depth = params.conv_3x1_depth
		self.dyn_embedding_size = params.dyn_embedding_size
		self.input_embedding_size = params.input_embedding_size
		self.num_lat_classes = params.num_lat_classes
		self.num_lon_classes = params.num_lon_classes
		self.soc_embedding_size = (((params.grid_size[0]-4)+1)//2)*self.conv_3x1_depth

		## Define network weights

		# Input embedding layer
		self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

		# Encoder LSTM
		self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

		# Vehicle dynamics embedding
		self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

		# Convolutional social pooling layer and social embedding layer
		self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
		self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
		self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

		# FC social pooling layer (for comparison):
		# self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((params.grid_size[0]-4)+1)//2)*self.conv_3x1_depth)

		# Decoder LSTM
		if self.use_maneuvers:
			self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
		else:
			self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

		# Output layers:
		self.op = torch.nn.Linear(self.decoder_size,5)
		self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
		self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

		# Activations:
		self.leaky_relu = torch.nn.LeakyReLU(0.1)
		self.relu = torch.nn.ReLU()
		self.softmax = torch.nn.Softmax(dim=1)

		# TRANSFORMER
		if self.use_transformer:
			src_feats = tgt_feats = 2 # (X,Y) point
			tgt_params = 5 # 5 params for bivariate Gaussian distrib

			if self.use_grid:
				src_ngrid = self.in_length # with soc
			else:
				src_ngrid = 0 # without soc

			if self.use_maneuvers:
				tgt_classes = self.num_lat_classes + self.num_lon_classes
				src_lon = self.num_lon_classes; src_lat = self.num_lat_classes
			else:
				tgt_classes = 0
				src_lon = 0; src_lat = 0

			self.transformer = tsf.make_model(src_feats, tgt_feats, N=2, 
												src_ngrid=src_ngrid, src_lon=src_lon, src_lat=src_lat, 
												tgt_params=tgt_params, tgt_classes=tgt_classes)

			self.batch = tsf.Batch()
			print("TRANSFORMER:", self.transformer)


	## Forward Pass
	def forward(self,hist,nbrs,masks,lat_enc,lon_enc, hist_grid, fut=None):

		# TRANSFORMER
		if self.use_transformer:
			print("HIST_GRID", hist_grid.shape) # [128, 16, 13, 3]
			assert fut is not None

			if self.use_grid:
				source_grid = copy.copy(hist_grid)
			else:
				source_grid = None

			if self.use_maneuvers:

				if self.train_flag:
					source_lon = lon_enc; source_lat = lat_enc
					self.batch.transfo(hist, fut, source_grid=source_grid, source_lon=source_lon, source_lat=source_lat)
					out = self.transformer.forward(self.batch.src, self.batch.trg, self.batch.src_mask, self.batch.trg_mask, 
													src_grid=self.batch.src_grid, src_lon=self.batch.src_lon, src_lat=self.batch.src_lat)
					fut_pred = self.transformer.generator(out)
					print("OUT:", out.shape); print("FUT_PRED:", fut_pred.shape)
				else:
					fut_pred = []
					## Predict trajectory distributions for each maneuver class
					for k in range(self.num_lon_classes):
						for l in range(self.num_lat_classes):
							lat_enc_tmp = torch.zeros_like(lat_enc)
							lon_enc_tmp = torch.zeros_like(lon_enc)
							lat_enc_tmp[:, l] = 1
							lon_enc_tmp[:, k] = 1

							source_lon = lon_enc_tmp; source_lat = lat_enc_tmp
							self.batch.transfo(hist, fut, source_grid=source_grid, source_lon=source_lon, source_lat=source_lat)
							out = self.transformer.forward(self.batch.src, self.batch.trg, self.batch.src_mask, self.batch.trg_mask, 
															src_grid=self.batch.src_grid, src_lon=self.batch.src_lon, src_lat=self.batch.src_lat)
							fut_pred_tmp = self.transformer.generator(out)

							fut_pred.append(fut_pred_tmp)

				# TODO: this is not good; use 2 Transformers or at least 2 different models, 1 for Regression and 1 for Classif
				# We can not use the lon/lat features for the lon/lat classifier ...
				# First without lon/lat features, predict lon/lat classes; THEN predict Traj with lon/lat features
				lat_pred = self.transformer.generator_lat(out)
				lon_pred = self.transformer.generator_lon(out)
				print("LAT_PRED:", lat_pred.shape); print("LON_PRED:", lon_pred.shape)
				return fut_pred, lat_pred, lon_pred
			else:
				self.batch.transfo(hist, fut, source_grid=source_grid)
				out = self.transformer.forward(self.batch.src, self.batch.trg, self.batch.src_mask, self.batch.trg_mask, src_grid=self.batch.src_grid)
				fut_pred = self.transformer.generator(out)

				print("OUT:", out.shape)
				print("FUT_PRED:", fut_pred.shape)
				return fut_pred

		## Forward pass hist:
		# hist:				 [3sec 16, batch 128,  xy 2]
		# self.ip_emb(hist): [16, 128, 32] via nn.Linear(2, 32)
		# an, (hn, cn)	   : via nn.LSTM(in 32, hid 64)
		# we retrieve hn   : [ 1, 128, 64]	 NB: note that an [16, 128, 64] is not used (will be used for ATTENTION)
		# hist_enc =  hn   : [ layers 1, batch 128, hid 64]
		# hist_enc		   : [ 128, 64] via reshaping (cf hist_enc.view(...))
		# hist_enc		   : [ 128, 32] via nn.Linear(hid 64, dyn_emb_size 32)
		_,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
		hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

		## Forward pass nbrs
		# nbrs:				 [3sec 16, #nbrs_hist as much as we found in 128x13x3 eg 850, xy 2]
		# self.ip_emb(nbrs): [16, 850, 32] vi nn.Linear(2, 32)
		# an, (hn, cn)	   : via nn.LSTM(in 32, hid64)
		# we retrieve hn   : [1, 850, 64]	NB: note that an [16, 850, 64] is not used
		# nbrs_enc		   : [850, 64] via reshaping
		# the traj hist of 3 secs of (X,Y)rel coords is tranformed into 64 features
		_, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
		nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

		## Masked scatter
		soc_enc = torch.zeros_like(masks).float() # [128, 3, 13, 64] tensor
		soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc) # [128, 3, 13, 64] = masked_scatter_([128, 3, 13, 64], [eg 850, 64])
		soc_enc = soc_enc.permute(0,3,2,1) # we end up with a [128, 64, 13, 3] tensor

		# soc_enc: [batch 128, hid 64, grid_lon 13, grid lat 3]
		# NB: VALID with PyTorch (not SAME default like with TF)
		# self.soc_conv(soc_enc):  [128, 64, 11, 1] via Conv2d(in 64, out 64, f 3)
		# self.conv_3x1(...):	   [128, 16,  9, 1] via Conv2d(in 64, out 16, f 3)
		# self.soc_maxpool(...):   [128, 16,  5, 1] via MaxPool2d(2, 1)
		# soc_enc: [128, 80] via reshaping NB: self.soc_embedding_size = 80

		## Apply convolutional social pooling:
		soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
		soc_enc = soc_enc.view(-1,self.soc_embedding_size)

		## Apply fc soc pooling
		# soc_enc = soc_enc.contiguous()
		# soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
		# soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

		## Concatenate encodings:
		# enc: [128, 112] via cat [128, 32] with [128, 80]
		enc = torch.cat((soc_enc,hist_enc),1)

		if self.use_maneuvers:
			## Maneuver recognition:
			# self.op_lat(enc): [128, 3] via nn.Linear(112, 3)
			# lat_pred		  : [128, 3] via softmax
			lat_pred = self.softmax(self.op_lat(enc))
			# self.op_lon(enc): [128, 2] via nn.Linear(112, 2)
			# lat_pred		  : [128, 2] via softmax
			lon_pred = self.softmax(self.op_lon(enc))

			if self.train_flag:
				## Concatenate maneuver encoding of the true maneuver
				# enc: [128, 117] via cat [128, 112],[128,3],[128,2]
				# 117 features: 32 dync, 80 soc, 5 maneuver
				enc = torch.cat((enc, lat_enc, lon_enc), 1)
				fut_pred = self.decode(enc) # enc: [batch 128, feats 117]
				return fut_pred, lat_pred, lon_pred
			else:
				fut_pred = []
				## Predict trajectory distributions for each maneuver class
				for k in range(self.num_lon_classes):
					for l in range(self.num_lat_classes):
						lat_enc_tmp = torch.zeros_like(lat_enc)
						lon_enc_tmp = torch.zeros_like(lon_enc)
						lat_enc_tmp[:, l] = 1
						lon_enc_tmp[:, k] = 1
						enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
						fut_pred.append(self.decode(enc_tmp))
				return fut_pred, lat_pred, lon_pred
		else:
			fut_pred = self.decode(enc)
			return fut_pred


	def decode(self,enc):
		# enc: [batch 128, feats 117]
		# we just repeat hn output, not using a_1 up to a_Tx (TODO via ATTENTION)
		# enc: [5sec 25, batch 128, feats 117] after repeat
		enc = enc.repeat(self.out_length, 1, 1)
		# And now we retrieve the T_y=25 outputs and discard (hn, cn)
		# h_dec: [25, 128, 128] via nn.LSTM(feats 117, hid 128)
		h_dec, _ = self.dec_lstm(enc)
		# h_dec: [batch 128, Ty 25, hid 128] via permute(1, 0, 2)
		h_dec = h_dec.permute(1, 0, 2)
		# fut_pred: [batch 128, Ty 25, bivariate gaussian params 5] via self.op=nn.Linear(128, 5)
		fut_pred = self.op(h_dec)
		# fut_pred: [Ty 25, batch 128, 5] via permute
		fut_pred = fut_pred.permute(1, 0, 2)
		fut_pred = outputActivation(fut_pred)
		# fut_pred: [Ty 25, batch 128, bivariate gaussian params 5] via outputActivation which enforces pred constraints
		return fut_pred

# TODO, ideas for improvement:
# 1) Use Attention mechanism
# 2) Use a real Seq2Seq decoder
# 3) Transformer
