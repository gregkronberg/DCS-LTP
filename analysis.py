"""
analysis

data structure for each trial is organized as ['tree'][polarity][section][segment]
"""
from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as it
import os
import cPickle as pickle
import param
import math
import run_control


class Weights():
	"""
	measure weight change at group of synapses

	saves initial and final weights at each active synapse across all simulated neurons (dcs polarity x synapses)
	"""
	def __init__(self,p):
		self.n_pol = len(p['field'])

	def dw(self, p):
		self.group_dw(p)
		self.save_dw(p)
		self.plot_dw_all(p, self.w_end_all)
		self.plot_dw_mean(p, self.w_end_all)

	def group_dw(self,p):
		# arrays for storing all weight changes across neurons
		self.w_end_all = np.empty([self.n_pol,0])
		self.w_start_all = np.empty([self.n_pol,0])
		# loop over experiments (neurons)
		for data_file in os.listdir(p['data_folder']):
			# check for proper data file format
			if 'data' in data_file:

				with open(p['data_folder']+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)

					# load data file
					# pkl_file = open(p['data_folder']+data_file, 'rb')
					# data = pickle.load(pkl_file)

					self.p = data['p']
					
					self.n_act_seg = len(self.p['seg_list'])
					
					# measure weight changes for individual neuron
					self.measure_dw(data)
					
					# add individual neuron to the group
					self.w_end_all = np.append(self.w_end_all,self.w_end,axis=1)
					self.w_start_all = np.append(self.w_start_all,self.w_start,axis=1)

				# # close pickle file
				# pkl_file.close()

	def measure_dw(self, data):
		# set up arrays to record final and initial weights at each active synapse
		self.w_end = np.empty([self.n_pol,self.n_act_seg]) # polarity x segments
		self.w_start = np.empty([self.n_pol,self.n_act_seg]) # polarity x segments

		# find active synapses (all recorded segments were active)
		# measure weight change at each active synapse

		for tree_key,tree in data.iteritems():
			if (data['p']['tree'] in tree_key) and ('_w' in tree_key):
				for f_i,f in enumerate(data['p']['field']):
					cnt = -1
					for sec_i,sec in enumerate(data['p']['sec_idx']):
						for seg_i,seg in enumerate(data['p']['seg_idx'][sec_i]):
							cnt += 1
							self.w_end[f_i,cnt] = data[tree_key][f_i][sec][seg][-1]
							self.w_start[f_i,cnt] = data[tree_key][f_i][sec][seg][0]

	def save_dw(self,p):
		with open(p['data_folder']+'dw_all_'+p['experiment']+'.pkl', 'wb') as output:
			pickle.dump(self.w_end_all, output,protocol=pickle.HIGHEST_PROTOCOL)

	def plot_dw_all(self,p,dw):
		# create figure
		self.fig_dw_all = plt.figure()
		# loop over n_pol
		for field_i,field in enumerate(p['field']):
			# plot
			plt.plot(field_i*np.ones(len(dw[field_i,:])),dw[field_i,:],p['field_color'][field_i]+'.')
		# save figure
		self.fig_dw_all.savefig(p['data_folder']+'fig_dw_all'+'.png', dpi=250)
		plt.close(self.fig_dw_all)
	
	def plot_dw_mean(self,p,dw):
		# determine stats
		dw_mean = np.mean(dw,axis=1)
		dw_std = np.std(dw,axis=1)
		dw_sem = stats.sem(dw,axis=1)
		# create figure
		self.fig_dw_mean = plt.figure()
		# loop over n_pol
		for field_i,field in enumerate(p['field']):
			# plot
			plt.errorbar(field_i,dw_mean[field_i],yerr=dw_sem[field_i],color = p['field_color'][field_i],fmt='.')
		# save figure
		self.fig_dw_mean.savefig(p['data_folder']+'fig_dw_mean'+'.png', dpi=250)
		plt.close(self.fig_dw_mean)

class Spikes():
	"""
	detect spikes and determine where they originated
	"""
	def __init__(self):
		pass

	def analysis_function(self, p):
		self.initialize_vectors(p)
		self.group_spikes(p)
		self.spike_start(p)
		self.save_spikes(p)
		self.plot_spike_hist_soma(self.spiket_soma,p)
		self.plot_spike_hist_dend(self.spike_dend_init,p)

	def initialize_vectors(self,p):
		self.n_pol = len(p['field'])
		# initialize lists
		self.spiket_soma = [] # soma spike times [polarity list][spikes array]
		self.spiket_dend = [] # dendrite spike times [polarity list][spikes array]
		self.sec_list = [] # keep track of sections (same dimensions as spiket_dend)
		self.seg_list = [] # keep track of segments (same dimensions as spiket_dend)
		self.cell_list_soma = []
		self.cell_list_dend = []
		self.win_list_soma = []
		self.win_list_dend = []
		# loop over polarity
		for pol in range(self.n_pol):
			self.spiket_soma.append(np.empty([1,0]))
			self.spiket_dend.append(np.empty([1,0]))
			self.sec_list.append([])
			self.seg_list.append([])
			self.cell_list_soma.append([])
			self.cell_list_dend.append([])
			self.win_list_soma.append([])
			self.win_list_dend.append([])

	def group_spikes(self,p):
		cell_num = -1 	# track which cell number
		for data_file in os.listdir(p['data_folder']):
			# check for proper data file format
			if 'data' in data_file:
				cell_num+=1
				# load data file
				pkl_file = open(p['data_folder']+data_file, 'rb')
				data = pickle.load(pkl_file)
				# get parameters from specific experiment
				self.p = data['p']
				self.n_act_seg = len(self.p['seg_list'])
				self.measure_spikes(data,self.p,cell_num)

	def spike_window(self,p):
		# determine windows for spike time detection [window number][min max]
		bursts = range(p['bursts'])
		pulses = range(p['pulses'])
		burst_freq = p['burst_freq']
		pulse_freq = p['pulse_freq']
		nrn_fs = 1000.
		fs = nrn_fs/p['dt']
		warmup = p['warmup']
		# for each input pulse their is a list containing the window start and stop time [window number][start,stop]
		return [[warmup*fs+(burst)*fs/burst_freq+(pulse)*fs/pulse_freq,warmup*fs+(burst)*fs/burst_freq+(pulse+1)*fs/pulse_freq] for burst in bursts for pulse in pulses]

	def measure_spikes(self,data,p,cell_num=0):
		# nrn_fs = 1000. # conversion from seconds to miliseconds
		# fs = nrn_fs/p['dt'] # sampling rate in samples/second
		window  = self.spike_window(p) # list of spike windows [window #][start,stop]
		
		# detect spikes for individual neurons
		for pol in range(self.n_pol):
			
			# detect soma spikes
			self.soma_spikes = self.detect_spikes(data['soma_v'][pol][0][0])['times']
			if self.soma_spikes.size!=0:
				
				# add spike times to array
				self.spiket_soma[pol] = np.append(self.spiket_soma[pol],self.soma_spikes*p['dt'],axis=1)

				# track cell number
				for spike_i,spike in enumerate(self.soma_spikes[0,:]):
					# detect the window that the spike occurred in, indexed by the onset time of the window
					spike_soma_win = [win[0] for win in window if (spike >= win[0]) and (spike < win[1]) ]
					self.cell_list_soma[pol].append(cell_num)
					self.win_list_soma[pol].append(spike_soma_win)
			
			# detect dendritic spikes and track location
			cnt=-1
			for sec_i,sec in enumerate(data[p['tree']+'_v'][pol]): # loop over sections
				for seg_i,seg in enumerate(data[p['tree']+'_v'][pol][sec_i]): # loop over segemnts
					cnt+=1
					# detect spikes
					dend_spikes = self.detect_spikes(np.array(data[p['tree']+'_v'][pol][sec_i][seg_i]))['times']
					if dend_spikes.size!=0:
						# add spike times to array
						self.spiket_dend[pol] = np.append(self.spiket_dend[pol],dend_spikes,axis=1)
						# spiket_dend_track = np.append(spiket_dend_track,dend_spikes,axis=1)
						# for each spike store the section, segment, cell number in the appropriate list
						for spike in dend_spikes[0,:]:
							spike_dend_win = [win[0] for win in window if (spike >= win[0]) and (spike < win[1]) ]
							self.sec_list[pol].append(sec_i)
							self.seg_list[pol].append(seg_i)
							self.cell_list_dend[pol].append(cell_num)
							self.win_list_dend[pol].append(spike_dend_win)

	def spike_start(self,p):
		# determine windows for spike time detection [window number][min max]
		bursts = range(p['bursts'])
		pulses = range(p['pulses'])
		burst_freq = p['burst_freq']
		pulse_freq = p['pulse_freq']
		fs = 1./p['dt']
		warmup = p['warmup']
		# for each input pulse their is a list containing the window start and stop time [window number][start,stop]
		window =  [[warmup*fs+(burst)*1000*fs/burst_freq+(pulse)*1000*fs/pulse_freq,warmup*fs+(burst)*1000*fs/burst_freq+(pulse+1)*1000*fs/pulse_freq] for burst in bursts for pulse in pulses]

		# numpy array for storing minumum spike time for each cell 
		self.spike_dend_init = []	# minimum spike time for each cell
		self.spike_dend_init_sec = []	# section where first spike occured
		self.spike_dend_init_seg = []	# segment where first spike occured
		self.spike_dend_init_cell = [] # keep track of cell number
		self.spike_dend_init_win = [] # timing of presynaptic input 
		# loop over polarities
		for pol in range(self.n_pol):
			# list all cells with a dendritic spike
			cells = list(set(self.cell_list_dend[pol]))
			# numpy array for storing minumum spike time for each cell 
			self.spike_dend_init.append([])	# minimum spike time for each cell
			self.spike_dend_init_sec.append([])	# section where first spike occured
			self.spike_dend_init_seg.append([])	# segment where first spike occured
			self.spike_dend_init_cell.append([]) # keep track of cell number
			self.spike_dend_init_win.append([]) # keep track of cell number
			# loop over cells
			for cell_i,cell in enumerate(cells):
				# print len(self.spiket_dend[pol][0,:])
				# print len(self.cell_list_dend[pol])
				# for each cell list all dendritic spike times
				spiket_dend = [spikes for spike_i,spikes in enumerate(self.spiket_dend[pol][0,:]) if self.cell_list_dend[pol][spike_i]==cell]
				# keep track of the index for in the full list of spike times
				spikei_dend = [spike_i for spike_i,spikes in enumerate(self.spiket_dend[pol][0,:]) if self.cell_list_dend[pol][spike_i]==cell]
				# loop over spike windows
				for win in window:
					# return spikes for this cell that fit the window
					spiket_dend_win = [spike for spike in spiket_dend if spike >= win[0] and spike < win[1]]
					# keep track of indeces
					spikei_dend_win = [spike for spike_i,spike in enumerate(spikei_dend) if spiket_dend[spike_i] >= win[0] and spiket_dend[spike_i] < win[1]]
					# check if spike was found
					if spiket_dend_win:
						# print min(spiket_dend_win)/fs
						# index in full list for first spike in current time window in current cell
						spike_idx = spikei_dend_win[spiket_dend_win.index(min(spiket_dend_win))]
						# store minimum spike time and keep track of section, segment, and cell number
						self.spike_dend_init[pol].append(min(spiket_dend_win)/fs)
						self.spike_dend_init_sec[pol].append(self.sec_list[pol][spike_idx])
						self.spike_dend_init_seg[pol].append(self.seg_list[pol][spike_idx])
						self.spike_dend_init_cell[pol].append(cell)
						self.spike_dend_init_win[pol].append(win[0])
			
			self.spike_dend_init[pol] = np.array([self.spike_dend_init[pol]])

	def spike_start_compare(self,p):
		cell1 = cell.CellMigliore2005(p)
		spikes = {}
		for tree_i,tree in cell1.geo.iteritems():
			spikes[tree_key] = []
			for field_i,field in p['field']:
				spikes[tree_key].append([])
				for sec_i,sec in tree:
					spikes[tree_key][field_i].append([])
					for seg_i,seg in sec:
						spikes[tree_key][field_i][sec_i].append({})
						spikes[tree_key][field_i][sec_i][seg_i]['times'] = []
						spikes[tree_key][field_i][sec_i][seg_i]['train'] = []
						spikes[tree_key][field_i][sec_i][seg_i]['p'] = []
						spikes[tree_key][field_i][sec_i][seg_i]['xcorr'] = []


		for data_file in os.listdir(p['data_folder']):
			# check for proper data file format
			if 'data' in data_file:
				with open(p['data_folder']+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)

				for tree_key,tree in spikes:
					for field_i,field in tree:
						for sec_i,sec in field:
							for seg_i,seg in sec:
								spike_temp = self.detect_spikes(data[tree_key+'_v'][field_i][sec_i][seg_i],thresshold=-20)
								if tree_key is 'soma':
									spike_temp_soma =spike_temp 

								seg['times'].append(spike_temp['times'])
								seg['train'].append(spike_temp['train'])
								seg['p'].append(data['p'])
								if tree_key is not 'soma':
									xcorr_temp = scipy.signal.correlate(spike_temp['train'],spikes['soma'][field_i])

								seg['xcorr'].append(spike_temp['xcorr'])
								


								spikes = self.detect_spikes(seg,threshold = -20)


		# loop over polarities
		# loop over cells
		# loop over spike windows
		# compare first dendritic spike to first somatic spike
		# get cross correlation for all time delays
		# determine whether these features influence field effects 

	def save_spikes(self,p):
		with open(p['data_folder']+'spiket_soma_all_'+p['experiment']+'.pkl', 'wb') as output:
			pickle.dump(self.spiket_soma, output,protocol=pickle.HIGHEST_PROTOCOL)

		with open(p['data_folder']+'spiket_dend_all_'+p['experiment']+'.pkl', 'wb') as output:
			pickle.dump(self.spiket_dend, output,protocol=pickle.HIGHEST_PROTOCOL)

	def plot_spike_hist_soma(self,data,p):
		warmup = p['warmup']
		finish = p['tstop']
		bins = np.linspace(warmup,finish ,(finish-warmup)/p['dt'])
		self.fig_spike_hist_soma = plt.figure()
		for pol in range(self.n_pol):
			plt.hist(data[pol][0,:],bins=bins,color = p['field_color'][pol])
		plt.title('Somatic spike time histogram')
		plt.xlabel('spike onset (ms)')
		plt.ylabel('count')
		# save figure
		self.fig_spike_hist_soma.savefig(p['data_folder']+'fig_spike_hist_soma'+'.png', dpi=250)
		plt.close(self.fig_spike_hist_soma)

	def plot_spike_hist_dend(self,data,p):
		warmup = p['warmup']
		finish = p['tstop']
		bins = np.linspace(warmup,finish ,(finish-warmup)/p['dt'])
		self.fig_spike_hist_dend = plt.figure()
		for pol in range(self.n_pol):
			plt.hist(data[pol][0,:],bins=bins,color = p['field_color'][pol])
		plt.title('Dendritic spike time histogram')
		plt.xlabel('spike onset (ms)')
		plt.ylabel('count')
		self.fig_spike_hist_dend.savefig(p['data_folder']+'fig_spike_hist_dend'+'.png', dpi=250)
		plt.close(self.fig_spike_hist_dend)

	def spikes_xcorr(self,data1,data2,p):
		pass
		# 


	def detect_spikes(self,data,threshold=-20):
		spike_times = np.asarray(np.where(np.diff(np.sign(data-threshold))>0))
		spike_train = np.zeros([1,len(data)])
		for time in spike_times:
			spike_train[0,time] = 1 
		# detect indeces where vector crosses threshold in the positive direction
		return {'times':spike_times,'train':spike_train}
		
class Voltage():
	""" plot voltage in specific sections 
	"""
	def __init__(self):
		pass

	def plot_all(self, p):
		"""
		"""
		# iterate over all data files in folder
		for data_file in os.listdir(p['data_folder']):
			# check for proper data file format
			if 'data' in data_file:
				print data_file
				# open data file
				with open(p['data_folder']+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
					print 'file loaded'
				# update specific experiment parameters
				p_data = data['p']
				# plot voltage traces in specified segments (automatically saved to same folder)
				self.plot_trace(data=data, 
					tree=p_data['tree'], 
					sec_idx=p_data['sec_idx'], 
					seg_idx=p_data['seg_idx'])


	def plot_trace(self, data, tree, sec_idx, seg_idx, soma=True):
		"""
		"""
		# load parameters
		p = data['p']
		# number field intensities/polarities
		n_pol = len(p['field'])
		# number of segments to plot
		nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for sec in seg_idx])+1

		if soma:
			nseg+=1
		cols = int(math.ceil(math.sqrt(nseg)))
		rows = int(math.ceil(math.sqrt(nseg)))
		# create plot array, axes is a list of figures in the array
		# fig, axes = plt.subplots(rows, cols )

		fig = plt.figure()
		
		# count segments
		cnt=0
		t = data['t'][0]
		# iterate over sections
		for sec_i,sec in enumerate(seg_idx):
			# iterate over segments
			for seg in sec:
				cnt+=1
				seg_dist = p['seg_dist'][p['tree']][sec_idx[sec_i]][seg]
				plt.subplot(rows, cols, cnt)
				plt.title(str(seg_dist))
				# plt.ylim([-70, -50])
				# iterate over stimulation polarity
				for pol in range(n_pol):
					
					# time vector
					# t = data['t'][pol]
					
					# voltage vector
					if soma and cnt<nseg:
						v = data[tree+'_v'][pol][sec_idx[sec_i]][seg]

					color = p['field_color'][pol]
					
					
					# add to corrsponding axis
					
					plt.plot(t, v, color=color)
					
		
		for pol in range(n_pol):
			# soma
			v = data['soma_v'][pol][0][0] 
			
			color = p['field_color'][pol]
			
			# add to corrsponding axis
			plt.subplot(rows, cols, nseg)
			plt.plot(t, v, color=color)
			plt.title('soma')
			# plt.ylim([-70, -50])
		
		# save and close figure
		fig.savefig(p['data_folder']+p['experiment']+'_'+p['tree']+'_trace_'+p['trial_id']+'.png', dpi=300)
		plt.close(fig)

class Shapeplot():
	""" create shape plot 
	"""
	pass

class Experiment:
	"""analyses for individual experiments
	"""
	def __init__(self, **kwargs):
		experiment = getattr(self, kwargs['experiment'])

		experiment(**kwargs) 

	def exp_1(self, **kwargs):
		""" 
		plot average weight change as a function of distance from soma 
		"""
		npol = 3 
		control_idx = 1
		data_folder = 'Data/'+kwargs['exp']+'/'
		files = os.listdir(data_folder)
		dw = np.zeros([npol, len(files)])
		syn_weight = np.zeros([1,len(files)])
		w = {'cell_list' : [],
		'weight_list' : [],
		'syn_frac_list' : [] }
		spikes = Spikes()
		fig3 = plt.figure(3)
		fig4 = plt.figure(4)
		for data_file_i, data_file in enumerate(files):
			if 'data' in data_file:

				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']
				cell_id = p['trial_id']
				w['cell_list'].append(cell_id)
				w['weight_list'].append(p['w_mean'])
				w['syn_frac_list'].append(p['syn_frac'])


				fig1 = plt.figure(1)
				fig2 = plt.figure(2)
				
				w[cell_id]=[]
				corr_win = int(p['warmup']/p['dt'])
				
				for field_i, field in enumerate(p['field']):
					w[cell_id].append([])
					spike_t_soma = spikes.detect_spikes(data['soma'+'_v'][field_i][0][0])['times']
					spike_train_soma = spikes.detect_spikes(data['soma'+'_v'][field_i][0][0])['train']
					for sec_i, sec in enumerate(p['sec_idx']):
						w[cell_id][field_i].append([])
						for seg_i,seg in enumerate(p['seg_idx'][sec_i]):
							w_end = data[p['tree']+'_w'][field_i][sec][seg][-1]
							w_start = data[p['tree']+'_w'][field_i][sec][seg][0]
							dw = w_end/w_start
							seg_dist = p['seg_dist'][p['tree']][sec][seg]
							spike_t = spikes.detect_spikes(data[p['tree']+'_v'][field_i][sec][seg])['times']
							spike_train = spikes.detect_spikes(data[p['tree']+'_v'][field_i][sec][seg])['train']
							w[cell_id][field_i][sec_i].append({})
							w[cell_id][field_i][sec_i][seg_i]['dw'] = dw
							w[cell_id][field_i][sec_i][seg_i]['seg_dist'] = seg_dist
							# measure spike times
							w[cell_id][field_i][sec_i][seg_i]['spike_t'] = spike_t
							w[cell_id][field_i][sec_i][seg_i]['spike_train'] = spike_train
							# spike cross correlation with soma
							# print spike_train_soma.shape
							# print spike_train.shape
							spike_xcorr = np.correlate(spike_train_soma[0,:], spike_train[0,:], mode='full')
							w[cell_id][field_i][sec_i][seg_i]['spike_xcorr'] = spike_xcorr
							plt.figure(1)
							plt.plot(seg_dist, dw, '.', color=p['field_color'][field_i])
							plt.figure(2)
							plt.plot(p['dt']*(np.arange(spike_xcorr.size)-spike_xcorr.size/2), spike_xcorr, color=p['field_color'][field_i])
							plt.figure(3)
							plt.plot(seg_dist, dw, '.', color=p['field_color'][field_i])

				for field_i, field in enumerate(p['field']):
					for sec_i, sec in enumerate(p['sec_idx']):
						for seg_i,seg in enumerate(p['seg_idx'][sec_i]):
							w[cell_id][field_i][sec_i][seg_i]['dw_effect'] = w[cell_id][field_i][sec_i][seg_i]['dw']/w[cell_id][control_idx][sec_i][seg_i]['dw']
							plt.figure(4)
							plt.plot(w[cell_id][field_i][sec_i][seg_i]['seg_dist'], w[cell_id][field_i][sec_i][seg_i]['dw_effect'], '.', color=p['field_color'][field_i])

				plt.figure(1)			
				plt.xlabel('distance from soma um')
				plt.ylabel('weight change')
				fig1.savefig(data_folder+'fig_dw_dist_'+cell_id+'.png', dpi=300)
				plt.close(fig1)
				plt.figure(2)			
				plt.xlabel('delay (ms)')
				plt.ylabel('correlation')
				fig2.savefig(data_folder+'fig_xcorr_'+cell_id+'.png', dpi=300)
				plt.close(fig2)
		print w['weight_list']
		# mean for across cells and synapses
		w_all = {}
		w_all_mean ={}
		w_all_sem = {}
		w_all_std = {}
		for weight_i, weight in enumerate(set(w['weight_list'])):
			weight_key = str(weight)
			w_all[weight_key]={}
			w_all_mean[weight_key]={}
			w_all_sem[weight_key]={}
			w_all_std[weight_key]={}
			for syn_frac_i, syn_frac in enumerate(set(w['syn_frac_list'])):
				syn_frac_key = str(syn_frac)
				w_all[weight_key][syn_frac_key]=[]
				w_all_mean[weight_key][syn_frac_key] = np.zeros([3,1])
				w_all_std[weight_key][syn_frac_key] = np.zeros([3,1])
				w_all_sem[weight_key][syn_frac_key] = np.zeros([3,1])
				for field_i in range(npol):
					w_all[weight_key][syn_frac_key].append([])
					for cell_key, cell in w.iteritems():
						if cell_key not in ['cell_list', 'weight_list', 'syn_frac_list']: 
							for sec_i, sec in enumerate(cell[field_i]):
								for seg_i,seg in enumerate(sec):
									cell_num = [cell_id_i for cell_id_i, cell_id in enumerate(w['cell_list']) if cell_id is cell_key][0]
									print cell_num
									if (w['weight_list'][cell_num] == weight) and (w['syn_frac_list'][cell_num]) == syn_frac:
										w_all[weight_key][syn_frac_key][field_i].append( seg['dw'])
					w_all[weight_key][syn_frac_key][field_i] = np.array(w_all[weight_key][syn_frac_key][field_i])
					print w_all[weight_key][syn_frac_key][field_i].shape
					w_all_mean[weight_key][syn_frac_key][field_i] = np.mean(w_all[weight_key][syn_frac_key][field_i])
					w_all_sem[weight_key][syn_frac_key][field_i] = stats.sem(w_all[weight_key][syn_frac_key][field_i])
					w_all_std[weight_key][syn_frac_key][field_i] = np.std(w_all[weight_key][syn_frac_key][field_i])
				fig5 = plt.figure(5)
				for field_i in range(npol):
					plt.plot(w_all_mean[weight_key][syn_frac_key][field_i], '.', color=p['field_color'][field_i])
			
				plt.ylabel('weight change')
				fig5.savefig(data_folder+'fig_dw_mean'+'_weight_'+weight_key+'syn_frac_'+syn_frac_key+'.png', dpi=300)
				plt.close(fig5)


		plt.figure(3)			
		plt.xlabel('distance from soma um')
		plt.ylabel('weight change')
		fig3.savefig(data_folder+'fig_dw_dist_all'+'.png', dpi=300)
		plt.close(fig3)

		plt.figure(4)			
		plt.xlabel('distance from soma um')
		plt.ylabel('dcs effect (dw field/control)')
		fig4.savefig(data_folder+'fig_dw_effect_dist_all'+'.png', dpi=300)
		plt.close(fig4)

				# plot weight cchange as a function of distance for all cells



		# iterate over data files
		# iterate over synapses
				# if active
						# store weight change
						# store spike time
						# store distance from soma
						# store in ['cell identifier'][tree][section][segment]['varaiable']
	
	def exp_2(self, **kwargs):
		npol = 3 
		data_folder = 'Data/'+kwargs['exp']+'/'
		files = os.listdir(data_folder)
		dw = np.zeros([npol, len(files)])
		syn_weight = np.zeros([1,len(files)])
		for data_file_i, data_file in enumerate(files):
			# check for proper data file format
			if 'data' in data_file:

				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']

				for field_i, field in enumerate(p['field']):
					for sec_i,sec in enumerate(p['sec_idx']):
						for seg_i,seg in enumerate(p['seg_idx'][sec_i]):
							w_end = data[p['tree']+'_w'][field_i][sec][seg][-1]
							w_start = data[p['tree']+'_w'][field_i][sec][seg][0]
							dw[field_i,data_file_i] = w_end/w_start
							syn_weight[0,data_file_i] = p['w_list'][sec_i][seg]


		fig = plt.figure()
		for pol in range(npol):
				plt.plot(syn_weight[0,:], dw[pol, :], '.', color=p['field_color'][pol])
		plt.xlabel('peak conductance (uS)')
		plt.ylabel('weight change')
		fig.savefig(data_folder+'fig_dw'+'.png', dpi=250)
		plt.close(fig)

	def exp_4(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances
		"""
		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# list files
		files = os.listdir(data_folder)

		# if group variable in folder, load variable
		if 'trial_id' in files:
			
			with open(data_folder+'trial_id', 'rb') as pkl_file:
					trial_id = pickle.load(pkl_file)
		# otherwise create variable
		else:
			trial_id = []

		npol = 3 
		
		g_range = run_control.Arguments('exp_4').kwargs['conductance_range']
		
		asymmetry = np.zeros([len(g_range), len(g_range)])
		cathodal = np.zeros([len(g_range), len(g_range)])
		anodal = np.zeros([len(g_range), len(g_range)])
		control = np.zeros([len(g_range), len(g_range)])
		g_h = g_range*0.00005 #np.zeros([1, len(files)])
		g_ka = g_range*0.03 #np.zeros([1, len(files)])

		syn_weight = np.zeros([1,len(files)])
		for data_file_i, data_file in enumerate(files):
			# check for proper data file format
			if 'data' in data_file:
				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']
				
				# check if file has already been processed
				if p['trial_id'] not in trial_id:
					trial_id.append(p['trial_id'])

					# retrieve conductance parameters

					# add to parameter vector

					# sort parameter vector

				# add polarization to matrix, based according to index in paramter vectors
				gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
				ka_i = [i for i, val in enumerate(g_ka) if p['KMULT']==val]
				

				control[gh_i, ka_i] = data['soma_v'][1][0][0][-1]
				cathodal[gh_i, ka_i] = data['soma_v'][0][0][0][-1] - control[gh_i, ka_i]
				anodal[gh_i, ka_i] = data['soma_v'][2][0][0][-1] - control[gh_i, ka_i]
				asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i] 

				# control[gh_i, ka_i] = data['apical_dist_v'][1][0][0][-1]
				# cathodal[gh_i, ka_i] = data['apical_dist_v'][0][0][0][-1] - control[gh_i, ka_i]
				# anodal[gh_i, ka_i] = data['apical_dist_v'][2][0][0][-1] - control[gh_i, ka_i]
				# asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i]

				# asym[0,data_file_i] = asymmetry
				# g_h[0,data_file_i] = p['ghd']
				# g_ka[0,data_file_i] = p['KMULT']
		print cathodal.shape
		fig1 = plt.figure(1)
		plt.imshow(cathodal)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Cathodal Membrane polarization (mV)')
		fig1.savefig(data_folder+'cathodal_conductance_parameters'+'.png', dpi=250)
		plt.close(fig1)

		fig2 = plt.figure(2)
		plt.imshow(anodal)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Anodal Membrane polarization (mV)')
		fig2.savefig(data_folder+'anodal_conductance_parameters'+'.png', dpi=250)
		plt.close(fig2)

		fig3 = plt.figure(3)
		plt.imshow(asymmetry)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Asymmetry, anodal + cathodal polarization (mV)')
		fig3.savefig(data_folder+'asymmetry_conductance_parameters'+'.png', dpi=250)
		plt.close(fig3)

		fig4 = plt.figure(4)
		plt.imshow(control)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Membrane potential (mV)')
		fig4.savefig(data_folder+'control_conductance_parameters'+'.png', dpi=250)
		plt.close(fig4)

	def exp_5(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances
		"""
		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# list files
		files = os.listdir(data_folder)

		# if group variable in folder, load variable
		if 'trial_id' in files:
			
			with open(data_folder+'trial_id', 'rb') as pkl_file:
					trial_id = pickle.load(pkl_file)
		# otherwise create variable
		else:
			trial_id = []
		
		# range of parameters
		grad_range = run_control.Arguments('exp_5').kwargs['grad_range']
		
		# preallocate 
		asymmetry = np.zeros([len(grad_range), len(grad_range)])
		cathodal = np.zeros([len(grad_range), len(grad_range)])
		anodal = np.zeros([len(grad_range), len(grad_range)])
		control = np.zeros([len(grad_range), len(grad_range)])
		
		# gradient parameter vectors
		grad_h = grad_range*3. #np.zeros([1, len(files)])
		grad_ka = grad_range*1. #np.zeros([1, len(files)])

		for data_file_i, data_file in enumerate(files):
			# check for proper data file format
			if 'data' in data_file:
				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']
				
				# check if file has already been processed
				if p['trial_id'] not in trial_id:
					trial_id.append(p['trial_id'])

					# retrieve conductance parameters

					# add to parameter vector

					# sort parameter vector

					# add polarization to matrix, based according to index in paramter vectors
				gh_i = [i for i, val in enumerate(grad_h) if p['ghd_grad']==val]
				ka_i = [i for i, val in enumerate(grad_ka) if p['ka_grad']==val]
				
				# soma
				# control[gh_i, ka_i] = data['soma_v'][1][0][0][-1]
				# cathodal[gh_i, ka_i] = data['soma_v'][0][0][0][-1] - control[gh_i, ka_i]
				# anodal[gh_i, ka_i] = data['soma_v'][2][0][0][-1] - control[gh_i, ka_i]
				# asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i] 

				# distal apical
				control[gh_i, ka_i] = data['apical_dist_v'][1][0][0][-1]
				cathodal[gh_i, ka_i] = data['apical_dist_v'][0][0][0][-1] - control[gh_i, ka_i]
				anodal[gh_i, ka_i] = data['apical_dist_v'][2][0][0][-1] - control[gh_i, ka_i]
				asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i] 

		print cathodal.shape
		fig1 = plt.figure(1)
		plt.imshow(cathodal)
		plt.colorbar()
		plt.ylabel('Ih gradient')
		plt.xlabel('Ka gradient')
		plt.yticks(range(len(grad_h)), grad_h)
		plt.xticks(range(len(grad_ka)), grad_ka)
		plt.title('Cathodal Membrane polarization (mV)')
		fig1.savefig(data_folder+'cathodal_gradient_parameters'+'.png', dpi=250)
		plt.close(fig1)

		fig2 = plt.figure(2)
		plt.imshow(anodal)
		plt.colorbar()
		plt.ylabel('Ih gradient')
		plt.xlabel('Ka gradient')
		plt.yticks(range(len(grad_h)), grad_h)
		plt.xticks(range(len(grad_ka)), grad_ka)
		plt.title('Anodal Membrane polarization (mV)')
		fig2.savefig(data_folder+'anodal_gradient_parameters'+'.png', dpi=250)
		plt.close(fig2)

		fig3 = plt.figure(3)
		plt.imshow(asymmetry)
		plt.colorbar()
		plt.ylabel('Ih gradient')
		plt.xlabel('Ka gradient')
		plt.yticks(range(len(grad_h)), grad_h)
		plt.xticks(range(len(grad_ka)), grad_ka)
		plt.title('Asymmetry, anodal + cathodal polarization (mV)')
		fig3.savefig(data_folder+'asymmetry_gradient_parameters'+'.png', dpi=250)
		plt.close(fig3)

		fig4 = plt.figure(4)
		plt.imshow(control)
		plt.colorbar()
		plt.ylabel('Ih gradient')
		plt.xlabel('Ka gradient')
		plt.yticks(range(len(grad_h)), grad_h)
		plt.xticks(range(len(grad_ka)), grad_ka)
		plt.title('Membrane potential (mV)')
		fig4.savefig(data_folder+'control_gradient_parameters'+'.png', dpi=250)
		plt.close(fig4)

	def exp_6(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances
		"""
		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# list files
		files = os.listdir(data_folder)

		# if group variable in folder, load variable
		if 'trial_id' in files:
			
			with open(data_folder+'trial_id', 'rb') as pkl_file:
					trial_id = pickle.load(pkl_file)
		# otherwise create variable
		else:
			trial_id = []

		npol = 3 
		
		g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		
		asymmetry = np.zeros([len(g_range), len(g_range)])
		cathodal = np.zeros([len(g_range), len(g_range)])
		anodal = np.zeros([len(g_range), len(g_range)])
		control = np.zeros([len(g_range), len(g_range)])
		g_h = g_range*0.00005 #np.zeros([1, len(files)])
		g_ka = g_range*0.03 #np.zeros([1, len(files)])

		syn_weight = np.zeros([1,len(files)])
		for data_file_i, data_file in enumerate(files):
			# check for proper data file format
			if 'data' in data_file:
				print data_file
				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']
				
				# check if file has already been processed
				if p['trial_id'] not in trial_id:
					trial_id.append(p['trial_id'])

					# retrieve conductance parameters

					# add to parameter vector

					# sort parameter vector

				# add polarization to matrix, based according to index in paramter vectors
				gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
				ka_i = [i for i, val in enumerate(g_ka) if p['KMULT']==val]
				

				control[gh_i, ka_i] = np.amax(data['soma_v'][1][0][0][(30*40):])
				print control[gh_i, ka_i]
				cathodal[gh_i, ka_i] = np.amax(data['soma_v'][0][0][0][(30*40):]) - control[gh_i, ka_i]
				anodal[gh_i, ka_i] = np.amax(data['soma_v'][2][0][0][(30*40):]) - control[gh_i, ka_i]
				asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i] 

				# control[gh_i, ka_i] = np.amax(data['apical_dist_v'][1][0][-1][(30*40):])
				# cathodal[gh_i, ka_i] = np.amax(data['apical_dist_v'][0][0][-1][(30*40):]) - control[gh_i, ka_i]
				# anodal[gh_i, ka_i] = np.amax(data['apical_dist_v'][2][0][-1][(30*40):]) - control[gh_i, ka_i]
				# asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i]

				# control[gh_i, ka_i] = np.amax(data['apical_prox_v'][1][0][-1][(30*40):])
				# cathodal[gh_i, ka_i] = np.amax(data['apical_prox_v'][0][0][-1][(30*40):]) - control[gh_i, ka_i]
				# anodal[gh_i, ka_i] = np.amax(data['apical_prox_v'][2][0][-1][(30*40):]) - control[gh_i, ka_i]
				# asymmetry[gh_i, ka_i] = anodal[gh_i, ka_i] + cathodal[gh_i, ka_i]

				# asym[0,data_file_i] = asymmetry
				# g_h[0,data_file_i] = p['ghd']
				# g_ka[0,data_file_i] = p['KMULT']
		print cathodal.shape
		fig1 = plt.figure(1)
		plt.imshow(cathodal)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Cathodal change in EPSP peak (mV)')
		fig1.savefig(data_folder+'cathodal_conductance_parameters'+'.png', dpi=250)
		plt.close(fig1)

		fig2 = plt.figure(2)
		plt.imshow(anodal)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Anodal change in EPSP peak (mV)')
		fig2.savefig(data_folder+'anodal_conductance_parameters'+'.png', dpi=250)
		plt.close(fig2)

		fig3 = plt.figure(3)
		plt.imshow(asymmetry)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Asymmetry, anodal + cathodal effect (mV)')
		fig3.savefig(data_folder+'asymmetry_conductance_parameters'+'.png', dpi=250)
		plt.close(fig3)

		fig4 = plt.figure(4)
		plt.imshow(control)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Control peak EPSP (mV)')
		fig4.savefig(data_folder+'control_conductance_parameters'+'.png', dpi=250)
		plt.close(fig4)








if __name__ =="__main__":
	# Weights(param.exp_3().p)
	# # Spikes(param.exp_3().p)
	kwargs = run_control.Arguments('exp_6').kwargs
	plots = Voltage()
	plots.plot_all(param.Experiment(**kwargs).p)
	Experiment(experiment='exp_6')




