from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import outputActivation
import pdb
import transformer as tsf
import copy
import random

class highwayNet(nn.Module):

	## Initialization
	def __init__(self,params, batch_size):
		super(highwayNet, self).__init__()

		## Unpack arguments
		self.params = params

		## Use gpu flag
		self.use_cuda = params.use_cuda

		# Flag for maneuver based (True) vs uni-modal decoder (False)
		self.use_maneuvers = params.use_maneuvers
		self.use_grid = params.use_grid

		# Transformer architecture related
		self.use_transformer = params.use_transformer
		self.teacher_forcing_ratio = 0.0 # TODO ultimately set it at > 0.9

		# RNN-LSTM Seq2seq architecture related
		self.use_seq2seq = params.use_seq2seq
		self.use_bidir = params.use_bidir

		# RNN-LSTM with Attention architecture related
		self.use_attention = params.use_attention
		if self.use_attention:
			self.use_bidir = True

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
		# TRANSFORMER
		if self.use_transformer:
			src_feats = tgt_feats = 2 # (X,Y) point
			tgt_params = 5 # 5 params for bivariate Gaussian distrib

			if self.use_grid:
				src_ngrid = self.in_length # with soc
			else:
				src_ngrid = 0 # without soc

			if self.use_maneuvers:
				d_lon = self.num_lon_classes
				d_lat = self.num_lat_classes

				self.transformer_cls = tsf.make_model_cls(src_feats, tgt_feats, 
                                                          tgt_lon_classes = d_lon, 
                                                          tgt_lat_classes = d_lat, 
                                                          N=2, src_ngrid = src_ngrid)
				print("TRANSFORMER_CLS:", self.transformer_cls)
			else:
				d_lon = 0
				d_lat = 0

			self.transformer_reg = tsf.make_model_reg(src_feats, tgt_feats, 
                                                      tgt_params=tgt_params, N=2,
                                                      src_ngrid=src_ngrid, 
                                                      src_lon=d_lon, src_lat=d_lat)
			print("TRANSFORMER_REG:", self.transformer_reg)

			self.batch = tsf.Batch()
			self.transfo_infer = tsf.Infer(self.out_length, batch_size)
			return

		# Input embedding layer
		self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

		# Encoder LSTM
		if self.use_bidir:
			self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1, bidirectional=True)
			self.encoder_ndir = 2
		else:
			self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
			self.encoder_ndir = 1

		# Vehicle dynamics embedding
		self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

		# Convolutional social pooling layer and social embedding layer
		self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
		self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
		self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

		# FC social pooling layer (for comparison):
		# self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((params.grid_size[0]-4)+1)//2)*self.conv_3x1_depth)

		if self.use_seq2seq or self.use_attention:# Decoder seq2seq LSTM (Attention buils on top of seq2seq)
			if self.use_maneuvers:
				self.proj_seq2seq = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
			else:
				self.proj_seq2seq = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

			if self.use_cuda:
				self.h0 = torch.zeros(1, batch_size, self.decoder_size).cuda() # [SeqLen, Batch, decoder size]
				self.c0 = torch.zeros(1, batch_size, self.decoder_size).cuda() # [1, 128, 128]
			else:
				self.h0 = torch.zeros(1, batch_size, self.decoder_size) # [1, 128, 128]
				self.c0 = torch.zeros(1, batch_size, self.decoder_size) # [1, 128, 128]

			self.dec_seq2seq = torch.nn.LSTM(self.decoder_size, self.decoder_size)
		else: # Legacy Decoder LSTM
			if self.use_maneuvers:
				self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
			else:
				self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

		if self.use_attention:
			self.attn_densor1 = torch.nn.Linear(self.encoder_ndir * self.encoder_size + self.decoder_size, 10)
			self.attn_densor2 = torch.nn.Linear(10, 1)


		# Output layers:
		self.op = torch.nn.Linear(self.decoder_size,5)
		self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
		self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

		# Activations:
		self.leaky_relu = torch.nn.LeakyReLU(0.1)
		self.relu = torch.nn.ReLU()
		self.softmax = torch.nn.Softmax(dim=1)


	## Forward Pass
	def forward(self,hist,nbrs,masks,lat_enc,lon_enc, hist_grid=None, fut=None):

		# TRANSFORMER
		if self.use_transformer:
			#print("HIST_GRID", hist_grid.shape) # [128, 16, 13, 3]
			assert fut is not None

			# Determine if we are using teacher forcing this iteration
			use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

			if self.use_grid:
				source_grid = copy.copy(hist_grid)
			else:
				source_grid = None

			if self.use_maneuvers:
				self.batch.transfo(hist, fut, source_grid=source_grid, source_lon=lon_enc, source_lat=lat_enc)
				#self.batch.transfo(hist, fut, source_grid=source_grid) # TODO do it once
				out = self.transformer_cls.forward(self.batch.src, self.batch.trg, 
				                                   self.batch.src_mask, self.batch.trg_mask, 
												   src_grid=self.batch.src_grid)
				lat_pred = self.transformer_cls.generator_lat(out)
				lon_pred = self.transformer_cls.generator_lon(out)
				#print("LAT_PRED:", lat_pred.shape); print("LON_PRED:", lon_pred.shape)

				if self.train_flag:
					#self.batch.transfo(hist, fut, source_grid=source_grid, source_lon=lon_enc, source_lat=lat_enc)
					out = self.transformer_reg.forward(self.batch.src, self.batch.trg, 
					                                   self.batch.src_mask, self.batch.trg_mask, 
													   src_grid=self.batch.src_grid, 
					                                   src_lon=self.batch.src_lon, src_lat=self.batch.src_lat)
					fut_pred = self.transformer_reg.generator(out)
					#print("OUT:", out.shape); print("FUT_PRED:", fut_pred.shape)
				else:
					fut_pred = []
					## Predict trajectory distributions for each maneuver class
					for k in range(self.num_lon_classes):
						for l in range(self.num_lat_classes):
							lat_enc_tmp = torch.zeros_like(lat_enc)
							lon_enc_tmp = torch.zeros_like(lon_enc)
							lat_enc_tmp[:, l] = 1
							lon_enc_tmp[:, k] = 1

							self.batch.transfo(hist, fut, source_grid=source_grid, 
							                   source_lon=lon_enc_tmp, source_lat=lat_enc_tmp)

							out_tmp = self.transformer_reg.forward(self.batch.src, self.batch.trg, 
							                                       self.batch.src_mask, self.batch.trg_mask, 
															       src_grid=self.batch.src_grid, 
																   src_lon=self.batch.src_lon, src_lat=self.batch.src_lat)
							fut_pred_tmp = self.transformer_reg.generator(out_tmp)
							fut_pred.append(fut_pred_tmp)
				return fut_pred, lat_pred, lon_pred
			else:
				self.batch.transfo(hist, fut, source_grid=source_grid)

				if use_teacher_forcing:
					out = self.transformer_reg.forward(self.batch.src, self.batch.trg, 
					                                   self.batch.src_mask, self.batch.trg_mask, 
													   src_grid=self.batch.src_grid)
					fut_pred = self.transformer_reg.generator(out)
				else:
					fut_pred = self.transfo_infer(self.transformer_reg, 
					                              self.batch.src, self.batch.src_mask, 
												  self.out_length)

				return fut_pred

		## Forward pass hist:
		# hist:				 [3sec 16, batch 128,  xy 2]
		# self.ip_emb(hist): [16, 128, 32] via nn.Linear(2, 32)

		# an, (hn, cn)	   : via nn.LSTM(in 32, hid 64)
		# we retrieve hn   : [ 1, 128, 64]	 NB: note that an [16, 128, 64] is not used (will be used for ATTENTION)
		# hist_a: [Tx 16, Batch 128, Dir * enc_size]
		# hist_enc =  hn   : [ layers 1, batch 128, hid 64]

		# hist_enc		   : [ 128, 64] via reshaping (cf hist_enc.view(...))
		# hist_enc		   : [ 128, 32] via nn.Linear(hid 64, dyn_emb_size 32)
		hist_a, (hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
		self.hist_a = hist_a # [Tx 16, Batch, dir * encoder_size]
		self.hist_enc = hist_enc  # [dir, Batch, encoder_size]
		if self.use_bidir: # sum bidir outputs
			# Somewhat similar to https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
			# TODO Maybe there is something more clever to do ... cat and proj ? Check papers
			hist_enc = hist_enc[0, :, :] + hist_enc[1, :, :]
			hist_enc = hist_enc.unsqueeze(0)
		hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

		## Forward pass nbrs
		# nbrs:				 [3sec 16, #nbrs_hist as much as we found in 128x13x3 eg 850, xy 2]
		# self.ip_emb(nbrs): [16, 850, 32] vi nn.Linear(2, 32)

		# an, (hn, cn)	   : via nn.LSTM(in 32, hid64)
		# we retrieve hn   : [1, 850, 64]	NB: note that an [16, 850, 64] is not used
		# nbrs_a: [Tx 16, 850, Dir * enc_size] TODO to be used by ATTENTION

		# nbrs_enc		   : [850, 64] via reshaping
		# the traj hist of 3 secs of (X,Y)rel coords is tranformed into 64 features
		nbrs_a, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
		self.nbrs_a = nbrs_a # [Tx, eg 850, Batch]
		self.nbrs_enc = nbrs_enc # [dir, eg 850, encoder_size]
		if self.use_bidir: # sum bidir outputs
			# Somewhat similar to https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
			# TODO Maybe there is something more clever to do ... cat and proj ? Check papers
			nbrs_enc = nbrs_enc[0, :, :] + nbrs_enc[1, :, :]
			nbrs_enc = nbrs_enc.unsqueeze(0)
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


	def decode(self, enc):

		if self.use_attention: # Attention builds on top of seq2seq
			enc = self.proj_seq2seq(enc) # proj from [Batch, 117] to [Batch, 128]
			yn = enc.unsqueeze(0) # [1, Batch 128, decoder size 128]
			context = self.one_step_attention(self.hist_a, self.h0).unsqueeze(0)
			yn, (hn, cn) = self.dec_seq2seq(context, (self.h0, self.c0))
			h_dec = yn
			for t in range(self.out_length - 1):
				context = self.one_step_attention(self.hist_a, hn).unsqueeze(0)
				yn, (hn, cn) = self.dec_seq2seq(context, (hn, cn))
				h_dec = torch.cat((h_dec, yn), dim=0)
		elif self.use_seq2seq:
			enc = self.proj_seq2seq(enc) # proj from [Batch, 117] to [Batch, 128]
			yn = enc.unsqueeze(0) # [1, Batch 128, decoder size 128]
			yn, (hn, cn) = self.dec_seq2seq(yn, (self.h0, self.c0))
			h_dec = yn
			for t in range(self.out_length - 1):
				yn, (hn, cn) = self.dec_seq2seq(yn, (hn, cn))
				h_dec = torch.cat((h_dec, yn), dim=0)
		else:
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

	def one_step_attention(self, a, h_prev):
		"""
		Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
		"alphas" and the states "a" of the Bi-LSTM.
		
		Arguments:
		a -- state output of the Bi-LSTM, numpy-array of shape (Tx, m, 2*n_a)
		h_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (1, m, n_s)
		
		Returns:
		context -- context vector, input of the next (post-attetion) LSTM cell
		"""
		Tx, m, _ = a.shape
		s_prev = torch.repeat_interleave(h_prev, Tx, dim=0) # => [Tx, m, n_s]
		concat = torch.cat((a, s_prev), dim=2) # => [Tx, m, 2*n_a + n_s]
		e = (self.attn_densor1(concat)).tanh() # => [Tx, m, 10]
		energies = F.relu(self.attn_densor2(e)) # => [Tx, m, 1]
		alphas = F.softmax(energies, dim=0) # softmax along Tx dim: [Tx, m, 1]
		# alphas * a: [Tx, m, 1] * [Tx, m, 2*n_a] => [Tx, m, 2_na]
		context = torch.sum(alphas * a, dim=0) # => [m, 2*n_a]
		return context
