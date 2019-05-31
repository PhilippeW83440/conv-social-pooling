import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from utils import outputActivation

import pdb


# Customizations
# - DONE Embeddings: linear transform d_feats -> d_model features
# - DONE Generator
# - DONE Batching

# DONE: add social context
# DONE : use maneuvers
#			- GeneratorLat and GeneratorLon DONE
#			- Embeddings with traj/grid/lat/lon features DONE


# ---------- EMBEDDINGS ----------

class Embeddings(nn.Module):
	def __init__(self, d_model, src_feats, src_ngrid=0, src_grid=(13,3), src_lon=0, src_lat=0, soc_emb_size=0):
		super(Embeddings, self).__init__()
		#self.lut = nn.Embedding(vocab, d_model)
		self.d_model = copy.copy(d_model)

		self.traj_emb = None
		self.grid_emb = None
		self.lat_emb = None
		self.lon_emb = None
		self.soc_emb = None

		self.soc_emb_size = soc_emb_size

		# Baiscally out of the 512 features for d_model encoding we split as:
		#	256 features for ego traj inputs
		#	256 features for social context (occupancy grid) inputs
		# Additionaly we may reserve 20 features (3*4+2*4) for maneuveurs used as inputs

		# Or just 512 features for taj_emb (eg at the output)

		if src_ngrid > 0: # handle 2D input features with conv net
			assert src_grid == (13,3) # so far this is the current assumption
			d_model_grid = d_model//2
			d_model -= d_model_grid
			# We start with [Batch, src_ngrid, 13, 3]
			self.conv1 = torch.nn.Conv2d(src_ngrid, 64, 3) # => [64, 11, 1]
			self.conv2 = torch.nn.Conv2d(64, 16, (3,1))  # => [16,	9, 1]
			self.maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0)) # => [16, 5, 1]
			self.leaky_relu = torch.nn.LeakyReLU(0.1)
			self.grid_emb = torch.nn.Linear(5, d_model_grid) # 5 from [16, 5, 1]

			if soc_emb_size > 0:
				self.soc_emb = torch.nn.Linear(soc_emb_size, d_model_grid) # projection

		if src_lon > 0:
			d_model_lon = src_lon * 4
			d_model -= d_model_lon
			self.lon_emb = torch.nn.Linear(src_lon, d_model_lon)

		if src_lat > 0:
			d_model_lat = src_lat * 4
			d_model -= d_model_lat
			self.lat_emb = torch.nn.Linear(src_lat, d_model_lat)

		self.traj_emb = torch.nn.Linear(src_feats, d_model)

	def forward(self, x):
		# workaround to make nn.Sequential work with multiple inputs
		# cf https://discuss.pytorch.org/t/nn-sequential-layers-forward-with-multiple-inputs-error/35591/3
		#x, soc = x[0], x[1]
		traj, grid, lon, lat = x
		emb = self.traj_emb(traj) # * math.sqrt(self.d_model)

		if grid is not None:
			if len(grid.shape) == 3: # 1D input
				assert self.soc_emb is not None
				soc_emb = self.soc_emb(grid) # * math.sqrt(self.d_model)
				emb = torch.cat((emb, soc_emb), dim=-1)
			else: # 2D input
				assert self.grid_emb is not None
				## Apply convolutional social pooling: => [128, 16, 5, 1]
				grid_enc = self.maxpool(self.leaky_relu(self.conv2(self.leaky_relu(self.conv1(grid)))))
				grid_enc = torch.squeeze(grid_enc) # [128, 16, 5]
				grid_emb = self.grid_emb(grid_enc)
				emb = torch.cat((emb, grid_emb), dim=-1)

		if lon is not None:
			assert self.lon_emb is not None
			lon_emb = self.lon_emb(lon) # * math.sqrt(self.d_model)
			emb = torch.cat((emb, lon_emb), dim=-1)

		if lat is not None:
			assert self.lat_emb is not None
			lat_emb = self.lat_emb(lat) # * math.sqrt(self.d_model)
			emb = torch.cat((emb, lat_emb), dim=-1)

		#print("EMB:", emb.shape)
		return emb # * math.sqrt(self.d_model)
		#return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0., max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], 
						 requires_grad=False)
		return self.dropout(x)


# ---------- COMMON LAYERS for encoder/decoder ----------

def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)


class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ---------- ENCODER ----------

class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
	"Core encoder is a stack of N layers"
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, mask):
		"Pass the input (and mask) through each layer in turn."
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)


# ---------- DECODER ----------

class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)


# ---------- ENCODER/DECODER ----------

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator=None, generator_lat=None, generator_lon=None):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator
		self.generator_lat = generator_lat
		self.generator_lon = generator_lon
		
	def forward(self, src, tgt, src_mask, tgt_mask, src_grid=None, src_lon=None, src_lat=None):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask, src_grid, src_lon, src_lat), src_mask,
							tgt, tgt_mask)
	
	def encode(self, src, src_mask, src_grid=None, src_lon=None, src_lat=None):
		return self.encoder(self.src_embed((src, src_grid, src_lon, src_lat)), src_mask)
	
	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed((tgt, None, None, None)), memory, src_mask, tgt_mask)

	#def prepare_infer(self, Ty, batch_size):
	#	self.ys_masks = []
	#	self.Ty = Ty
	#	for i in range(Ty):
	#		ys_mask = np.ones( (i+1, i+1), dtype='uint8')
	#		ys_mask = np.tril(ys_mask, 0)
	#		ys_mask = np.repeat(ys_mask[np.newaxis, :, :], batch_size, axis=0)
	#		ys_mask = torch.from_numpy(ys_mask)
	#		if torch.cuda.is_available():
	#			ys_mask = ys_mask.cuda()
	#		self.ys_masks.append(ys_mask)

	def infer(self, model, src, src_mask, Ty, src_grid=None, src_lon=None, src_lat=None):
		m, Tx, nx = src.shape
		memory = model.encode(src, src_mask, src_grid, src_lon, src_lat) # [Batch 128, Tx 16, d_model 512]
		ys = src[:, -1, :].unsqueeze(1) # [Batch 128, ys.size(1) 1, X/Y 2]
	
		for i in range(Ty):
			ys_mask = np.ones( (ys.size(1), ys.size(1)), dtype='uint8')
			ys_mask = np.tril(ys_mask, 0)
			ys_mask = np.repeat(ys_mask[np.newaxis, :, :], m, axis=0)
			ys_mask = torch.from_numpy(ys_mask)
			if torch.cuda.is_available():
				ys_mask = ys_mask.cuda()

			#out = model.decode(memory, src_mask, ys, self.ys_masks[i]) # [Batch 128, ys.size(1), d_model 512]
			# Last batch is usually not of size batch_size ...
			out = model.decode(memory, src_mask, ys, ys_mask) # [Batch , ys.size(1), d_model 512]
			fut_pred = model.generator(out) # [ys.size(1), Batch 128, gaussian_params 5]
			fut_pred = fut_pred.permute(1, 0, 2) # [Batch 128, ys.size(1), gaussian_params 5]
			next_y = fut_pred[:, -1, 0:2].unsqueeze(1) # [Batch 128, 1, muX/muY 2]
			ys = torch.cat( (ys, next_y), dim=1) # [Batch 128, ys.size(1)+1, 2]
	
		fut_pred = fut_pred.permute(1, 0, 2) # [Ty 25, Batch 128, 5]
		return fut_pred


# ---------- GENERATOR: for final output ----------

class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, tgt_params):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, tgt_params)

	def forward(self, x):
		# params: [batch 128, Ty 25, bivariate gaussian params 5] 
		fut_pred = self.proj(x)
		# fut_pred: [Ty 25, batch 128, 5] via permute
		fut_pred = fut_pred.permute(1, 0, 2)
		fut_pred = outputActivation(fut_pred)
		# fut_pred: [Ty 25, batch 128, bivariate gaussian params 5] via outputActivation which enforces pred constraints
		return fut_pred
		#return F.log_softmax(self.proj(x), dim=-1)

class GeneratorLat(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, tgt_lat_classes):
		super(GeneratorLat, self).__init__()
		# 3 classes: right, left, none
		self.proj = nn.Linear(d_model, tgt_lat_classes)

	def forward(self, x):
		lat_pred = F.softmax(self.proj(x), dim=-1) # [Batch 128, Ty, 3]
		lat_pred = lat_pred[:, -1, :]
		lat_pred = torch.squeeze(lat_pred)
		return lat_pred # [Batch 128, 3]

class GeneratorLon(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, tgt_lon_classes):
		super(GeneratorLon, self).__init__()
		# 2 classes: braking or not
		self.proj = nn.Linear(d_model, 2, tgt_lon_classes)

	def forward(self, x):
		lon_pred = F.softmax(self.proj(x), dim=-1)
		lon_pred = lon_pred[:, -1, :]
		lon_pred = torch.squeeze(lon_pred)
		return lon_pred # [Batch 128, 2]



# ---------- FULL MODEL ----------

# This model does not use lon/lat features as inputs
# But predicts lon/lat maneuvers
# DEPRECATED
def make_model_cls(src_feats, tgt_feats, tgt_lon_classes=2, tgt_lat_classes=3, 
					N=6, d_model=512, d_ff=2048, h=8, dropout=0.1,
					src_ngrid=0, src_grid=(13,3), src_soc_emb_size=0):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)

	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_feats, src_ngrid, src_grid, src_soc_emb_size), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_feats), c(position)),
		generator_lat = GeneratorLat(d_model, tgt_lon_classes),
		generator_lon = GeneratorLon(d_model, tgt_lat_classes))
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model


# This model uses lon/lat features as inputs
# And predicts traj
#def make_model(src_feats, tgt_feats, tgt_params=5, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1,
#def make_model(src_feats, tgt_feats, tgt_params=5, N=1, d_model=256, d_ff=1024, h=1, dropout=0.1,
def make_model(src_feats, tgt_feats, tgt_params=5, N=1, d_model=256, d_ff=256, h=4, dropout=0.1,
			   src_ngrid=0, src_grid=(13,3), # for 2D image like input features
			   src_soc_emb_size = 0,
			   src_lon=0, src_lat=0): # additional input features (TODO: list for genericity)
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)

	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_feats, src_ngrid, src_grid, src_lon, src_lat, src_soc_emb_size), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_feats), c(position)),
		generator = Generator(d_model, tgt_params))
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model


# ---------- BATCH utility ----------

class Batch:
	"Object for holding a batch of data with mask during training."
	def __init__(self):
		self.src = None
		self.src_grid = None
		self.src_mask = None
		self.src_lon = None
		self.src_lat = None
		self.trg = None
		self.trg_mask = None
		self.trg_y = None

	def transfo(self, source, target=None, source_grid=None, source_lon=None, source_lat=None):
		# We want [Batch, Tx, Nx]
		src = copy.copy(source)
		src = src.permute(1, 0, 2)
		self.src = src

		m, Tx, _ = src.shape

		# [Batch, Tx, 13, 3] for grid or [Batch, Tx, 80] for grid_soc
		src_grid = copy.copy(source_grid)
		self.src_grid = src_grid

		# encoder has full visibility on all inputs
		src_mask = np.ones((1, Tx), dtype='uint8')
		#src_mask[:,0] = 0
		src_mask = np.repeat(src_mask[np.newaxis, :, :], m, axis=0)
		self.src_mask = torch.from_numpy(src_mask)

		if source_lon is not None:
			src_lon = copy.copy(source_lon)
			src_lon = torch.unsqueeze(src_lon, dim=1)
			src_lon = torch.repeat_interleave(src_lon, Tx, dim=1)
			self.src_lon = src_lon
		else:
			self.src_lon = None

		if source_lat is not None:
			src_lat = copy.copy(source_lat)
			src_lat = torch.unsqueeze(src_lat, dim=1)
			src_lat = torch.repeat_interleave(src_lat, Tx, dim=1)
			self.src_lat = src_lat
		else:
			self.src_lat = None

		self.ntokens  = torch.from_numpy(np.array([m*Tx]))

		# We want [Batch, Ty, Ny]
		if target is not None:
			trg = copy.copy(target)
			trg = trg.permute(1, 0, 2)

			# Create a fake Transformer "start symbol/step" by repeating "end of input" in beginning of trg
			# The "start symbol" is pretty common for NMT taks; do something similar here
			trg = torch.cat((src[:,-1,:].unsqueeze(1), trg), dim=1)

			my, Ty, ny = trg.shape
			assert m == my, "src and trg batch sizes do not match"

			# ensure sequentiality between input and output of decoder
			# y(n) depends on y(1)...y(n-1)
			self.trg = trg[:, :-1, :]	# input  of DECODER
			self.trg_y = trg[:, 1:, :] # expected output of DECODER
			# otherwise the decoder just "learns" to copy the input ...
			# with quickly a loss of 0 during training .....
			
			
			# decoder at step n, has visibility on y(1)..y(n-1)
			trg_mask = np.ones((Ty-1,Ty-1), dtype='uint8')
			trg_mask = np.tril(trg_mask, 0)
			trg_mask = np.repeat(trg_mask[np.newaxis, :, :], m, axis=0)
			self.trg_mask = torch.from_numpy(trg_mask)

			if torch.cuda.is_available():
				self.trg = self.trg.cuda()
				self.trg_y = self.trg_y.cuda()
				self.trg_mask = self.trg_mask.cuda()
		else:
			self.trg = None
			self.trg_y = None
			self.trg_mask = None

		#print("SRC:", self.src.shape)
		#if self.src_grid is not None:
		#	print("SRC_GRID:", self.src_grid.shape)
		#print("TRG:", self.trg.shape)
		#print("TRG_Y:", self.trg_y.shape)

		if torch.cuda.is_available():
			self.src = self.src.cuda()
			self.src_mask = self.src_mask.cuda()
			if self.src_lon is not None:
				self.src_lon = self.src_lon.cuda()
			if self.src_lat is not None:
				self.src_lat = self.src_lat.cuda()
