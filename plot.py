from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button


import os
import utils_nn as utils
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='baseline', help="baseline, baselineX, seq2seq, seq2seqX, attention or transformer")

cmd_args = parser.parse_args()

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

# Initialize network
batch_size=128
net = highwayNet(params)

net_path = os.path.join(args['model_dir'], 'best.pth.tar')
assert os.path.isfile(net_path), "No net file found at {}".format(net_path)
#net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
utils.load_checkpoint(net_path, net)

if params.use_cuda:
	net = net.cuda()

net.eval()




# https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
					 mux=0.0, muy=0.0, sigmaxy=0.0):
	"""
	Bivariate Gaussian distribution for equal shape *X*, *Y*.
	See `bivariate normal
	<http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
	at mathworld.
	"""
	Xmu = X-mux
	Ymu = Y-muy

	rho = sigmaxy/(sigmax*sigmay)
	z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
	denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
	return np.exp(-z/(2*(1-rho**2))) / denom

class DiscreteSlider(Slider):
	def __init__(self, *args, **kwargs):
		self.inc = kwargs.pop('increment', 1)
		Slider.__init__(self, *args, **kwargs)
		self.valfmt = '%d / %d'
		valinit = kwargs.get('valinit', 0)
		valmax = kwargs.get('valmax', 0)
		self.set_val(0, 0)
		self.set_val(valinit, valmax)

	def set_val(self, val, max_val=0):
		if self.val != val:
			discrete_val = int(int(val / self.inc) * self.inc)
			xy = self.poly.xy
			xy[2] = discrete_val, 1
			xy[3] = discrete_val, 0
			self.poly.xy = xy
			progress = self.valfmt % (int(discrete_val), int(max_val))
			self.valtext.set_text(progress)
			if self.drawon:
				self.ax.figure.canvas.draw()
			self.val = val
			if not self.eventson:
				return
			for cid, func in self.observers.items():
				func(discrete_val)

	def update_val_external(self, val, max_val):
		self.set_val(val, max_val)


data_set = ngsimDataset('data/TestSet.mat')

delta = 2.0
x_min = 2*-120
x_max = 2*120
range_y = 20
scale = 0.4
v_w = 5
v_l = 15

# Out of https://github.com/mljack
class VisualizationPlot(object):
	def __init__(self, dataset, fig=None):
		self.current_frame = 1
		self.changed_button = False
		self.plotted_objects = []
		self.data_set = dataset
		self.maximum_frames = len(dataset)

		# Create figure and axes
		if fig is None:
			self.fig, self.ax = plt.subplots(1, 1)
			self.fig.set_size_inches(32, 4)
			plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=1.0)
			#fig, ax = plt.subplots(figsize=((x_max-x_min)*scale,range_y*2*scale))
		else:
			self.fig = fig
			self.ax = self.fig.gca()

		# Initialize the plot with the bounding boxes of the first frame
		self.update_figure()

		ax_color = 'lightgoldenrodyellow'
		# Define axes for the widgets
		self.ax_slider =		   self.fig.add_axes([0.3, 0.11, 0.40, 0.03], facecolor=ax_color)  # Slider
		base = 0.35
		self.ax_button_previous2 = self.fig.add_axes([base+0.00, 0.01, 0.05, 0.06])  # Previous x50 button
		self.ax_button_previous =  self.fig.add_axes([base+0.06, 0.01, 0.05, 0.06])  # Previous button
		self.ax_button_next =	   self.fig.add_axes([base+0.12, 0.01, 0.05, 0.06])  # Next button
		self.ax_button_next2 =	   self.fig.add_axes([base+0.18, 0.01, 0.05, 0.06])  # Next x50 button
		self.ax_random =		   self.fig.add_axes([base+0.24, 0.01, 0.05, 0.06])  # Random button

		# Define the widgets
		self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', valmin=1, valmax=self.maximum_frames, valinit=self.current_frame, valfmt="%d")
		self.button_previous2 = Button(self.ax_button_previous2, 'Previous x50')
		self.button_previous = Button(self.ax_button_previous, 'Previous')
		self.button_next = Button(self.ax_button_next, 'Next')
		self.button_next2 = Button(self.ax_button_next2, 'Next x50')
		self.button_random = Button(self.ax_random, 'Random')

		# Define the callbacks for the widgets' actions
		self.frame_slider.on_changed(self.update_slider)
		self.button_previous.on_clicked(self.update_button_previous)
		self.button_next.on_clicked(self.update_button_next)
		self.button_previous2.on_clicked(self.update_button_previous2)
		self.button_next2.on_clicked(self.update_button_next2)
		self.button_random.on_clicked(self.update_button_random)
		self.scroll_event_handler = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

		self.ax.set_autoscale_on(False)

	def trigger_update(self):
		self.remove_patches()
		self.update_figure()
		self.frame_slider.update_val_external(self.current_frame, self.maximum_frames)
		self.fig.canvas.draw_idle()

	def update_figure(self):
		# Dictionaries for the style of the different objects that are visualized
		rect_style = dict(facecolor="r", fill=True, edgecolor="k", zorder=19)
		triangle_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19)
		text_style = dict(picker=True, size=8, color='k', zorder=10, ha="center")
		text_box_style = dict(boxstyle="round,pad=0.2", fc="yellow", alpha=.6, ec="black", lw=0.2)
		track_style = dict(color="r", linewidth=1, zorder=10)

		# XXX while True:
		# XXX 	data = data_set[self.current_frame - 1]
		# XXX 	batch = ((data),)
		# XXX 	#pdb.set_trace()
		# XXX 	#hist, nbrs, nbrs_fur, mask, lat_enc, lon_enc, fut, op_mask = data_set.collate_fn(batch)
		# XXX 	hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_grid = data_set.collate_fn(batch)
		# XXX 	if nbrs.nelement() > 0:
		# XXX 		break
		# XXX 	self.current_frame += 1

		data = data_set[self.current_frame - 1]
		batch = ((data),)
		hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, hist_grid = data_set.collate_fn(batch)
		self.current_frame += 1

		# Initialize Variables
		if params.use_cuda:
			hist = hist.cuda()
			nbrs = nbrs.cuda()
			mask = mask.cuda()
			lat_enc = lat_enc.cuda()
			lon_enc = lon_enc.cuda()
			fut = fut.cuda()
			op_mask = op_mask.cuda()

		# Forward pass
		fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
		nbr_count_all = 0
		Z = None
		k = 0
		plotted_objects = []

		#print(hist.shape)	 # [16, 128, 2]
		#print(nbrs.shape)	 # [16, 939, 2]
		#print(mask.shape)	 # [128, 3, 13, 64]
		#print(np.count_nonzero(mask)/nbrs.shape[1]) # 64
		#print(mask[0, :, :, 0])
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
		# [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
		nbr_count = np.count_nonzero(mask[k, :, :, 0].cpu())
		#print(nbr_count)	 # 7 neighbors
		local_nbrs = nbrs[:, nbr_count_all:nbr_count_all+nbr_count, :]
		nbr_count_all += nbr_count
		#print(local_nbrs.shape) # [16, 7, 2]	 found 7 neighbor in 3*13 grid
		#print(lat_enc.shape)	 # [128, 3]
		#print(lon_enc.shape)	 # [128, 2]
		#print(len(nbrs_fur))	# 128 grids with neighbor future

		#for nbr_fur in nbrs_fur[k]:
		#	if nbr_fur.size > 0:
		#		nbr_fur_x = nbr_fur[:,0]
		#		nbr_fur_y = nbr_fur[:,1]
		#		t = self.ax.plot(nbr_fur_y, nbr_fur_x, '|-')	# neighbor future
		#		plotted_objects.append(t)

		if local_nbrs.nelement() > 0:
			for j in range(local_nbrs.shape[1]):
				nbr_x = local_nbrs[:,j,0].cpu().numpy()
				nbr_y = local_nbrs[:,j,1].cpu().numpy()
				#t = self.ax.plot(nbr_y, nbr_x, 'b|-')	# neighbor history
				#plotted_objects.append(t)
				patch = matplotlib.patches.Rectangle((nbr_y[-1]-v_l*0.5, nbr_x[-1]-v_w*0.5), v_l, v_w, color="w", fill=False)
				self.ax.add_patch(patch) # neighbor vehicles
				plotted_objects.append(patch)

		hist_x = hist[:,k,0].cpu().numpy()
		hist_y = hist[:,k,1].cpu().numpy()
		t = self.ax.plot(hist_y, hist_x, 'w|-')
		plotted_objects.append(t)

		fur_x = fut[:,k,0].cpu().numpy()
		fur_y = fut[:,k,1].cpu().numpy()
		t = self.ax.plot(fur_y, fur_x, 'g|-')
		plotted_objects.append(t)

		lat_man = torch.argmax(lat_pred[k, :]).detach()
		lon_man = torch.argmax(lon_pred[k, :]).detach()
		selected_indx = lon_man*3 + lat_man
		man_count = 0
		man_Zs = []
		for j in range(lat_pred.shape[1]):	# lat
			for jj in range(lon_pred.shape[1]): # lon
				lat_man = j
				lon_man = jj
				indx = lon_man*3 + lat_man

				prob = lat_pred[k, lat_man].detach() * lon_pred[k, lon_man].detach()
				# XXX if prob < 0.2:
				# XXX 	continue

				#lon_man = lon_man.cpu().numpy()
				#lat_man = lat_man.cpu().numpy()
				#if lat_man == 0:
				#	 continue
				#print("\nlon: ", lon_man, "\nlat: ", lat_man)

				fut_pred_ = fut_pred[indx][:,k,:].cpu().detach().numpy()

				x = np.arange(x_min, x_max, delta)
				y = np.arange(-range_y, range_y, delta)
				X, Y = np.meshgrid(x, y)
				xx = []
				yy = []

				Zs = []
				for i in range(fut_pred_.shape[0]): # time
					#print(fut_pred_.shape)
					muX = fut_pred_[i,0]
					muY = fut_pred_[i,1]
					sigX = fut_pred_[i,2]
					sigY = fut_pred_[i,3]
					rho = fut_pred_[i,4]
					#print(muX, muY, sigX, sigY, rho)

					xx.append(muX)
					yy.append(muY)

					Z1 = bivariate_normal(X, Y, 1/sigY, 1/sigX, muY, muX, rho/sigX/sigY)
					#Z1 = bivariate_normal(X, Y, 1/sigX, 1/sigY, muX, muY, rho/sigX/sigY)

					factor = (Z1.max() - Z1.min())
					# XXX if factor < 0.00001:
					# XXX 	continue
					Z1 /= factor
					Z1 *= prob.cpu().numpy() # XXX

					Zs.append(Z1)

					#Z1 = np.log(Z1+1)
					#print("range:", np.min(Z1), np.max(Z1))
					if Z is None:
						Z = Z1.copy()
					else:
						Z = Z + Z1
					#if i >= 5:
					#	 break
				man_Zs.append(Zs)

				if indx == selected_indx:
					man_selected = man_count
					t = self.ax.plot(yy, xx, 'rx-') # maneuver with max prob
				else:
					t = self.ax.plot(yy, xx, 'y.-')	# other maneuvers
				plotted_objects.append(t)
				man_count += 1
		if Z is not None:
			patch = matplotlib.patches.Rectangle((-v_l*0.5, -v_w*0.5), v_l, v_w, color="r", fill=False)
			self.ax.add_patch(patch) # main vehicle
			plotted_objects.append(patch)
			#m = self.ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.inferno, extent=(x_min, x_max, -range_y, range_y))
			# XXX 10 => we visualize pdf for selected maneuver at 3 seconds
			m = self.ax.imshow(man_Zs[man_selected][15], interpolation='bilinear', origin='lower', cmap=cm.inferno, extent=(x_min, x_max, -range_y, range_y))
			#m = self.ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cm.viridis, extent=(x_min, x_max, -range_y, range_y))
			plotted_objects.append(m)
		self.plotted_objects = plotted_objects

	def update_slider(self, value):
		if not self.changed_button:
			self.current_frame = value
			self.remove_patches()
			self.update_figure()
			self.fig.canvas.draw_idle()
		self.changed_button = False
		
	def update_button_previous(self, _):
		if self.current_frame > 1:
			self.current_frame = self.current_frame - 1
			self.changed_button = True
			self.trigger_update()
		else:
			print("There are no frames available with an index lower than 1.")

	def update_button_previous2(self, _):
		if self.current_frame - 50 > 0:
			self.current_frame = self.current_frame - 50
			self.changed_button = True
			self.trigger_update()
		else:
			print("There are no frames available with an index lower than 1.")

	def update_button_next(self, _):
		if self.current_frame < self.maximum_frames:
			self.current_frame = self.current_frame + 1
			self.changed_button = True
			self.trigger_update()
		else:
			print("There are no frames available with an index higher than {}.".format(self.maximum_frames))  #

	def update_button_next2(self, _):
		if self.current_frame + 50 <= self.maximum_frames:
			self.current_frame = self.current_frame + 50
			self.changed_button = True
			self.trigger_update()
		else:
			print("There are no frames available with an index higher than {}.".format(self.maximum_frames))

	def on_scroll(self, event):
		if event.button == 'up':
			self.update_button_previous(event)
		else:
			self.update_button_next(event)


	def update_button_random(self, _):
		self.current_frame = np.random.randint(self.maximum_frames) + 1
		self.changed_button = True
		self.trigger_update()

	def get_figure(self):
		return self.fig

	def remove_patches(self, ):
		#self.fig.canvas.mpl_disconnect('pick_event')
		for figure_object in self.plotted_objects:
			if isinstance(figure_object, list):
				figure_object[0].remove()
			else:
				figure_object.remove()

	@staticmethod
	def show():
		plt.show()
		plt.close()

visualization_plot = VisualizationPlot(data_set)
visualization_plot.show()


