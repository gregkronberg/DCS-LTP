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

class PlotRangeVar():
	""" plot different range variables over time 
	"""
	def __init__(self):
		pass

	def plot_all(self, p):
		""" Plot times series data for all data files in folder without replotting

		Arguments:

		p : parameter dictionary containing the following entries
			'data_folder' : string specifying the folder containing data and plots

			'tree' : neuron subtree containing the data to plot

			'sec_idx' : list of section indeces to plot

			'seg_idx' : list of segments for each section to plot [section_num][segment] 

			'plot_variables' : list of range variables to plot; e.g. ['v', 'i_hd']

		"""
		# all files in directory
		files = os.listdir(p['data_folder'])
		
		# files containing plot figures
		plot_files = [file for file in files if 'trace' in file]
		
		# data files
		data_files = [file for file in files if 'data' in file]
		
		# unique identifiers for each file type
		plot_files_id = [file[-39:-4] for file in files if 'trace' in file]
		data_files_id= [file[-39:-4] for file in files if 'data' in file]
		
		# iterate over all data files in folder
		for file_i, file in enumerate(data_files):
			
			# check if plot already exists matching uid's
			if data_files_id[file_i] not in plot_files_id:
				print data_files_id[file_i] 
				# open data file
				with open(p['data_folder']+file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)

				# update specific experiment parameters
				p_data = data['p']
				
				# plot range variables (automatically saved to same folder)
				self.plot_trace(data=data, 
					trees=p_data['trees'], 
					sec_idx=p_data['sec_idx'], 
					seg_idx=p_data['seg_idx'],
					variables=p_data['plot_variables'],
					x_variables=p_data['x_variables'],
					file_name=p_data['trial_id'],
					group_trees=p_data['group_trees'])

	def plot_trace(self, data, trees, sec_idx, seg_idx, soma=True, axon=True, group_trees=True, variables=['v'], x_variables='t', xlim=[], ylim=[], file_name=''):
		""" Create a figure containing a subplot for each segment specified in seg_idx

		variable is a list of range variable key words indicating which varaible to plot.  A new figure is created and saved for each variable
		"""
		# load parameters
		p = data['p']

		# number field intensities/polarities
		n_pol = len(p['field'])
	
		# dictionary to hold figures
		fig={}	

		# iterate over list of y variables to plot
		for var in variables:

			fig[var] = {}

			# iterate over list of x variables to plot
			for x_var in x_variables:


				# if data from all subtrees in one plot
				if group_trees:

					# create figure
					fig[var][x_var] = plt.figure()

					# number of segments to plot
					nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for tree_key, tree in seg_idx.iteritems() for sec in tree if tree_key in p['trees'] ])+1

					print nseg
					# plot soma trace?
					if soma:
						nseg+=1
					if axon:
						nseg+=1

					# columns and rows for subplot array	
					cols = int(math.ceil(math.sqrt(nseg)))
					rows = int(math.ceil(math.sqrt(nseg)))

				
				# if each subtree to get its own figure
				elif not group_trees:
					
					# create another dimension to store each subtree figure separately
					fig[var][x_var]={}
			
				# count segments
				cnt=0

				# iterate over trees
				for tree_key, tree in seg_idx.iteritems():

					if not group_trees:
						cnt=0

					# check that there are active sections in the tree
					if tree:

						# if each subtree gets its own figure
						if not group_trees:
							fig[var][x_var][tree_key] = plt.figure()

							# number of segments to plot
							nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for sec in tree])+1
							
							print'nseg:',nseg
							# plot soma trace?
							if soma:
								nseg+=1
							if axon:
								nseg+=1

							# columns and rows for subplot array	
							cols = int(math.ceil(math.sqrt(nseg)))
							rows = int(math.ceil(math.sqrt(nseg)))

							print 'rows,cols:',rows,cols

						# iterate over sections
						for sec_i,sec in enumerate(tree):

							# derive section number from index
							sec_num = sec_idx[tree_key][sec_i]
							
							# iterate over segments
							for seg_i, seg in enumerate(sec):
								
								# count subplots (segments)
								cnt+=1
								
								# create subplot
								plt.subplot(rows, cols, cnt)

								# get segment distance from soma
								seg_dist = p['seg_dist'][tree_key][sec_num][seg]
								
								# plot title
								plt.title(tree_key + ('%.2f'%seg_dist) )

								# adjust limits
								if var is 'v':
									if xlim:
										plt.xlim(xlim)
									if ylim:
										plt.ylim(ylim)
								
								# iterate over stimulation polarity
								for f_i, f in enumerate(p['field']):
								
									# check if variable exists in the current section
									if data[str(f)][tree_key+'_'+var][sec_num]:

										# if not plotting the soma trace
										if soma and cnt<nseg:

											# retrieve time series to plot
											v = data[str(f)][tree_key+'_'+var][sec_num][seg]

											# retrieve x variable
											if x_var =='t':
												# time vector
												xv = data[str(f)]['t']

											# other variable from arguments
											else:
												xv = data[str(f)][tree_key+'_'+x_var][sec_num][seg]


										# set plot color based on stimulation polarity
										color = p['field_color'][f_i]
										
										# add trace to corresponding subplot
										plt.plot(xv, v, color=color)
										plt.xlabel(x_var)
										plt.ylabel(var)

						if soma:
							cnt+=1

							# if plotting soma trace
							for f_i, f in enumerate(p['field']):

	
								# if variable exists in soma data
								if 'soma_'+var in data[str(f)].keys():
									if len(data[str(f)]['soma_'+var][0][0])>0:
									
										# subplot for soma trace
										plt.subplot(rows, cols, cnt)

										# adjust limits
										if var is 'v':
											if xlim:
												plt.xlim(xlim)
											if ylim:
												plt.ylim(ylim)

										# retrieve data to plot
										v = data[str(f)]['soma_'+var][0][0] 

										# determine x variable to plot
										if x_var =='t':
											# time vector
											xv = data[str(f)]['t']
										else:
											xv = data[str(f)]['soma_'+x_var][0][0]
										
										# set plot color
										color = p['field_color'][f_i]
										
										plt.plot(xv, v, color=color)
										plt.title('soma')
										plt.xlabel(x_var)
										plt.ylabel(var)

						if axon:
							cnt+=1

							# if plotting soma trace
							for f_i, f in enumerate(p['field']):

								# if variable exists in soma data
								if 'axon_'+var in data[str(f)].keys():
									if len(data[str(f)]['axon_'+var][0][0])>0:
									
										# subplot for soma trace
										plt.subplot(rows, cols, cnt)

										# adjust limits
										if var is 'v':
											if xlim:
												plt.xlim(xlim)
											if ylim:
												plt.ylim(ylim)

										# retrieve data to plot
										v = data[str(f)]['axon_'+var][0][0] 

										# determine x variable to plot
										if x_var =='t':
											# time vector
											xv = data[str(f)]['t']
										else:
											xv = data[str(f)]['axon_'+x_var][0][0]
										
										# set plot color
										color = p['field_color'][f_i]
										
										plt.plot(xv, v, color=color)
										plt.title('axon')
										plt.xlabel(x_var)
										plt.ylabel(var)
						
						# save figure
						# if each tree has separate figure
						if not group_trees:
							
							# info to add to file name
							file_name_add = tree_key+'_trace_'+x_var+'_x_'+var

							# save figure
							fig[var][x_var][tree_key].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

							# close figure
							plt.close(fig[var][x_var][tree_key])

					# if all trees are in the same figure
					if group_trees:
						all_trees =''
						for tree_key, tree in seg_idx.iteritems():
							all_trees = all_trees+tree_key+'_'

						# info to add to file name
						file_name_add = all_trees+'trace_'+x_var+'_x_'+var

						# save and close figure
						fig[var][x_var].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

						plt.close(fig[var][x_var])

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
		plots = PlotRangeVar()
		plots.plot_all(kwargs['p'])

	
	def exp_2(self, **kwargs):
		pass
		# npol = 3 
		# data_folder = 'Data/'+kwargs['exp']+'/'
		# files = os.listdir(data_folder)
		# dw = np.zeros([npol, len(files)])
		# syn_weight = np.zeros([1,len(files)])
		# for data_file_i, data_file in enumerate(files):
		# 	# check for proper data file format
		# 	if 'data' in data_file:

		# 		with open(data_folder+data_file, 'rb') as pkl_file:
		# 			data = pickle.load(pkl_file)
		# 		p = data['p']

		# 		for field_i, field in enumerate(p['field']):
		# 			for sec_i,sec in enumerate(p['sec_idx']):
		# 				for seg_i,seg in enumerate(p['seg_idx'][sec_i]):
		# 					w_end = data[p['tree']+'_w'][field_i][sec][seg][-1]
		# 					w_start = data[p['tree']+'_w'][field_i][sec][seg][0]
		# 					dw[field_i,data_file_i] = w_end/w_start
		# 					syn_weight[0,data_file_i] = p['w_list'][sec_i][seg]


		# fig = plt.figure()
		# for pol in range(npol):
		# 		plt.plot(syn_weight[0,:], dw[pol, :], '.', color=p['field_color'][pol])
		# plt.xlabel('peak conductance (uS)')
		# plt.ylabel('weight change')
		# fig.savefig(data_folder+'fig_dw'+'.png', dpi=250)
		# plt.close(fig)

	def exp_2b(self, **kwargs):
		""" activate groups of synapses at varying distance from soma until dendritic or somatic spikes are observed
		"""
		pass

	def exp_2c(self, **kwargs):
		""" activate synapse in each segment until a spike either occurs in the soma or dendrite.  Plot the number of synapses(weight) required at each segment as a function of distance from soma
		"""

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		plot_files = [file for file in files if 'trace' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]
		plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		spike_anal = Spikes()
		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					tree = p['trees'][0]
					sec = p['sec_idx'][tree][0]
					seg = p['seg_idx'][tree][sec][0]

					for f_i, f in enumerate(p['field']):

						# detect spikes
						spikes_dend[f_i] = spike_anal.detect_spikes(data[str(f)][tree+'_v'][sec][seg], theshold=-25)

						spikes_soma[f_i] = spike_anal.detect_spikes(data[str(f)]['soma_v'][sec][seg], theshold=-25)

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
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances and gradients
		"""
		# FIXME
		npol = 3 

		# things to measure
		measures = ['epsp', 'ss']

		locations = ['soma', 'apical_dist']

		conditions = ['cathodal', 'control', 'anodal', 'asymmetry']

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		# print id_list

		# kwargs = run_control.Arguments('exp_6').kwargs
		# g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		# grad_range = run_control.Arguments('exp_6').kwargs['grad_range']
		g_range = kwargs['conductance_range']
		grad_range = kwargs['grad_range']
		
		# N-dimensional array for storing data (each tested parameter is a dimension)
		mat = {}
		for measure_key, measure in enumerate(measures):
			mat[measure] = {}
			for condition_i, condition in enumerate(conditions):
				mat[measure][condition]= {}
				for location_i, location in enumerate(locations):
					mat[measure][condition][location] = []
					save_string = '_IhKaMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					if save_string in files:
						with open(data_folder+save_string, 'rb') as pkl_file:
							mat[measure][condition][location] = pickle.load(pkl_file)

					else:
						mat[measure][condition][location] = np.zeros([len(g_range), len(g_range), len(grad_range), len(grad_range)])

		# asymmetry = np.zeros([len(g_range), len(g_range), len(grad_range), len(grad_range)])
		# cathodal = np.zeros([len(g_range), len(g_range), len(grad_range), len(grad_range)])
		# anodal = np.zeros([len(g_range), len(g_range), len(grad_range), len(grad_range)])
		# control = np.zeros([len(g_range), len(g_range), len(grad_range), len(grad_range)])

		# vectors of parameter values
		g_h = g_range*kwargs['ghd'] 
		g_ka = g_range*kwargs['KMULT'] 
		grad_h = grad_range*kwargs['ghd_grad']
		grad_ka = grad_range*kwargs['ka_grad']

		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					# detect parameter combination for specific file
					gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
					ka_i = [i for i, val in enumerate(g_ka) if p['KMULT']==val]
					grad_h_i = [i for i, val in enumerate(grad_h) if p['ghd_grad']==val]
					grad_ka_i = [i for i, val in enumerate(grad_ka) if p['ka_grad']==val]
					
					# epsp onset
					onset = int(p['warmup']*1./p['dt'])
					print onset

					for measure_i, measure in enumerate(measures):
						for location_i, location in enumerate(locations):
							for condition_i, condition in enumerate(conditions):
								if 'epsp' in measure:
									window1 = onset
									window2 = len(data['t'][0])
								elif 'ss' in measure:
									window1 = onset-2
									window2 = onset-1

								if condition is not 'asymmetry':
									# measure peak epsp and store in data array
									mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i] = np.amax(data[location+'_v'][condition_i][0][-1][window1:window2])
									# print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]

								elif condition is 'asymmetry':
									mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i] = mat[measure]['anodal'][location][gh_i, ka_i, grad_h_i, grad_ka_i] + mat[measure]['cathodal'][location][gh_i, ka_i, grad_h_i, grad_ka_i] - 2.*mat[measure]['control'][location][gh_i, ka_i, grad_h_i, grad_ka_i]

									if mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i] > 1:
										print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]
		# save processed list
		with open(data_folder+'id_list'+'.pkl', 'wb') as output:pickle.dump(id_list, output,protocol=pickle.HIGHEST_PROTOCOL)

		# save asymmetry data matrix
		for measure_i, measure in enumerate(measures):
			for location_i, location in enumerate(locations):
				for condition_i, condition in enumerate(conditions):
					save_data = mat[measure][condition][location]
					save_string = data_folder+'_IhKaMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					# save matrix for each condition separately
					with open(save_string, 'wb') as output:
						pickle.dump(save_data, output,protocol=pickle.HIGHEST_PROTOCOL)

		# find maximum asymmetry
		best_val = np.amax(mat['ss']['asymmetry']['apical_dist'])
		best = np.argmax(mat['ss']['asymmetry']['apical_dist'])
		best_idx  = np.unravel_index(best, (len(g_range), len(g_range), len(grad_range), len(grad_range)))
		best_file = id_list[best]
		print best_idx
		print best_val
		print best_file

		#FIX MEEEEEEEE
		# choose parameters to plot
		fig={}
		fig['conductance'] = plt.figure(1)
		plot_mat = mat['ss']['asymmetry']['apical_dist'][:,:,best_idx[2],best_idx[3]]
		plt.imshow(plot_mat)
		plt.colorbar()
		plt.ylabel('Ih conductance')
		plt.xlabel('Ka conductance')
		plt.yticks(range(len(g_h)), g_h)
		plt.xticks(range(len(g_ka)), g_ka)
		plt.title('Asymmetry in apical_dist EPSP peak (mV)')
		fig['conductance'].savefig(data_folder+'_IhKa_conductance'+'.png', dpi=250)
		plt.close(fig['conductance'])

		# FIXMEEEEE
		# plot Ih as a function of membrane potential for each compartment

	def exp_7(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances and gradients
		"""
		# FIXME
		npol = 3 

		# things to measure
		measures = ['epsp', 'ss']

		locations = ['soma', 'apical_tuft']

		conditions = ['cathodal', 'control', 'anodal', 'asymmetry']

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		plot_files = [file for file in files if 'trace' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]
		plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		# print id_list

		# kwargs = run_control.Arguments('exp_6').kwargs
		# g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		# grad_range = run_control.Arguments('exp_6').kwargs['grad_range']
		g_range = kwargs['conductance_range']
		activation_range = kwargs['activation_range_h']
		slope_range = kwargs['slope_range']
		
		# N-dimensional array for storing data (each tested parameter is a dimension)
		
		# varaible to decide whether to create new matrix to store data
		new_mat=0
		# iterate over measures, conditions, locations
		mat={}
		for measure_key, measure in enumerate(measures):
			mat[measure] = {}
			for condition_i, condition in enumerate(conditions):
				mat[measure][condition]= {}
				for location_i, location in enumerate(locations):
					# string to save data
					save_string = '_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'
					# if data file already exists
					if save_string in files:
						# load data
						with open(data_folder+save_string, 'rb') as pkl_file:
							mat[measure][condition][location] = pickle.load(pkl_file)
					else:
						new_mat=1
						mat[measure][condition][location] = np.zeros([len(g_range), len(activation_range), len(slope_range)])

		# vectors of parameter values
		g_h = g_range*kwargs['ghd'] 
		activations = activation_range+kwargs['vhalfl_hd_prox']
		slopes = kwargs['kl_hd'] - slope_range
		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					# detect parameter combination for specific file
					gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
					activations_i = [i for i, val in enumerate(activations) if p['vhalfl_hd_prox']==val]
					slopes_i = [i for i, val in enumerate(slopes) if p['kl_hd']==val]
					
					# epsp onset
					onset = int(p['warmup']*1./p['dt'])
					print onset

					# iterate over things to measure
					for measure_i, measure in enumerate(measures):
						# iterate over locations to record from
						for location_i, location in enumerate(locations):
							# iterate over stimulation conditions
							for condition_i, condition in enumerate(conditions):
								# set window to take measurements
								if 'epsp' in measure:
									window1 = onset
									window2 = len(data['t'][0])
								elif 'ss' in measure:
									window1 = onset-2
									window2 = onset-1

								if condition is not 'asymmetry':
									# measure peak epsp and store in data array
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = np.amax(data[location+'_v'][condition_i][0][-1][window1:window2])
									# print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]

								elif condition is 'asymmetry':
									# measure asymmetry as anodal effect + cathodal effect
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = mat[measure]['anodal'][location][gh_i, activations_i, slopes_i] + mat[measure]['cathodal'][location][gh_i, activations_i, slopes_i] - 2.*mat[measure]['control'][location][gh_i, activations_i, slopes_i]

									if mat[measure][condition][location][gh_i, activations_i, slopes_i] > 1:
										print measure, location, condition, ':', mat[measure][condition][location][gh_i, activations_i, slopes_i]

					fig = plt.figure()
					h_trace = data
		# save processed list
		with open(data_folder+'id_list'+'.pkl', 'wb') as output:pickle.dump(id_list, output,protocol=pickle.HIGHEST_PROTOCOL)
		
		# save asymmetry data matrix 
		for measure_i, measure in enumerate(measures):
			for location_i, location in enumerate(locations):
				for condition_i, condition in enumerate(conditions):
					save_data = mat[measure][condition][location]
					save_string = data_folder+'_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					# save matrix for each condition separately
					with open(save_string, 'wb') as output:
						pickle.dump(save_data, output,protocol=pickle.HIGHEST_PROTOCOL)

		# find maximum asymmetry
		best_val = np.amax(mat['epsp']['asymmetry']['soma'])
		best = np.argmax(mat['epsp']['asymmetry']['soma'])
		best_idx  = np.unravel_index(best, (len(g_range), len(activation_range), len(slope_range)))
		best_file = id_list[best]
		print best_idx
		print best_val
		print best_file

		#FIX MEEEEEEEE
		# choose parameters to plot
		fig=[]
		for parameter in range(mat['epsp']['asymmetry']['soma'].shape[2]):
			fig.append(plt.figure(parameter))
			plot_mat = mat['epsp']['asymmetry']['soma'][:,:,parameter]#best_idx[2]]
			plt.imshow(plot_mat)
			plt.colorbar()
			plt.ylabel('Ih conductance')
			plt.xlabel('Ih activation gate half max voltage')
			plt.yticks(range(len(g_h)), g_h)
			plt.xticks(range(len(activations)), activations)
			plt.title('Asymmetry in soma EPSP peak (mV), kl='+ str(int(slopes[parameter])))
			fig[parameter].savefig(data_folder+'_Ih_conductance_kl_'+str(int(slopes[parameter]))+'.png', dpi=250)
			plt.close(fig[parameter])

		# FIXMEEEEE
		# plot Ih as a function of membrane potential for each compartment
		plot_files_hv = [file for file in plot_files if '_hv_' in file]
		plot_files_hv_id = [file[-36:-1] for file in plot_files if '_v_x_i_hd_' in file]
		for data_file_i, data_file in enumerate(data_files):
			if data_files_id[data_file_i] not in plot_files_hv_id:
				with open(data_folder+data_file, 'rb') as pkl_file:
					data = pickle.load(pkl_file)
				p = data['p']
				PlotRangeVar().plot_trace(
					data=data, 
					tree=p['tree'], 
					sec_idx=p['sec_idx'], 
					seg_idx=p['seg_idx'],
					variables=p['plot_variables'],
					x_var='v')

	def exp_8(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances and gradients
		"""
		# FIXME
		npol = 3 

		# things to measure
		measures = ['epsp', 'ss']

		locations = ['soma', 'basal']

		conditions = ['cathodal', 'control', 'anodal', 'asymmetry']

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		plot_files = [file for file in files if 'trace' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]
		plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		# print id_list

		# kwargs = run_control.Arguments('exp_6').kwargs
		# g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		# grad_range = run_control.Arguments('exp_6').kwargs['grad_range']
		g_range = kwargs['conductance_range']
		activation_range = kwargs['activation_range_h']
		slope_range = kwargs['slope_range']
		
		# N-dimensional array for storing data (each tested parameter is a dimension)
		
		# varaible to decide whether to create new matrix to store data
		new_mat=0
		# iterate over measures, conditions, locations
		mat={}
		for measure_key, measure in enumerate(measures):
			mat[measure] = {}
			for condition_i, condition in enumerate(conditions):
				mat[measure][condition]= {}
				for location_i, location in enumerate(locations):
					# string to save data
					save_string = '_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'
					# if data file already exists
					if save_string in files:
						# load data
						with open(data_folder+save_string, 'rb') as pkl_file:
							mat[measure][condition][location] = pickle.load(pkl_file)
					else:
						new_mat=1
						mat[measure][condition][location] = np.zeros([len(g_range), len(activation_range), len(slope_range)])

		# vectors of parameter values
		g_h = g_range*kwargs['ghd'] 
		activations = activation_range+kwargs['vhalfl_hd_prox']
		slopes = kwargs['kl_hd'] - slope_range
		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					# detect parameter combination for specific file
					gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
					activations_i = [i for i, val in enumerate(activations) if p['vhalfl_hd_prox']==val]
					slopes_i = [i for i, val in enumerate(slopes) if p['kl_hd']==val]
					
					# epsp onset
					onset = int(p['warmup']*1./p['dt'])
					print onset

					# iterate over things to measure
					for measure_i, measure in enumerate(measures):
						# iterate over locations to record from
						for location_i, location in enumerate(locations):
							# iterate over stimulation conditions
							for condition_i, condition in enumerate(conditions):
								# set window to take measurements
								if 'epsp' in measure:
									window1 = onset
									window2 = len(data['t'][0])
								elif 'ss' in measure:
									window1 = onset-2
									window2 = onset-1

								if condition is not 'asymmetry':
									# measure peak epsp and store in data array
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = np.amax(data[location+'_v'][condition_i][0][-1][window1:window2])
									# print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]

								elif condition is 'asymmetry':
									# measure asymmetry as anodal effect + cathodal effect
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = mat[measure]['anodal'][location][gh_i, activations_i, slopes_i] + mat[measure]['cathodal'][location][gh_i, activations_i, slopes_i] - 2.*mat[measure]['control'][location][gh_i, activations_i, slopes_i]

									if mat[measure][condition][location][gh_i, activations_i, slopes_i] > 1:
										print measure, location, condition, ':', mat[measure][condition][location][gh_i, activations_i, slopes_i]

					fig = plt.figure()
					h_trace = data
		# save processed list
		with open(data_folder+'id_list'+'.pkl', 'wb') as output:pickle.dump(id_list, output,protocol=pickle.HIGHEST_PROTOCOL)
		
		# save asymmetry data matrix 
		for measure_i, measure in enumerate(measures):
			for location_i, location in enumerate(locations):
				for condition_i, condition in enumerate(conditions):
					save_data = mat[measure][condition][location]
					save_string = data_folder+'_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					# save matrix for each condition separately
					with open(save_string, 'wb') as output:
						pickle.dump(save_data, output,protocol=pickle.HIGHEST_PROTOCOL)

		# find maximum asymmetry
		# best_val = np.amax(mat['epsp']['asymmetry']['soma'])
		# best = np.argmax(mat['epsp']['asymmetry']['soma'])
		# best_idx  = np.unravel_index(best, (len(g_range), len(activation_range), len(slope_range)))
		# best_file = id_list[best]
		# print best_idx
		# print best_val
		# print best_file

		# plot asymmetry as a function of parameters
		fig=[]
		for parameter in range(mat['epsp']['asymmetry']['soma'].shape[2]):
			fig.append(plt.figure(parameter))
			plot_mat = mat['epsp']['asymmetry']['soma'][:,:,parameter]#best_idx[2]]
			plt.imshow(plot_mat)
			plt.colorbar()
			plt.ylabel('Ih conductance')
			plt.xlabel('Ih activation gate half max voltage')
			plt.yticks(range(len(g_h)), g_h)
			plt.xticks(range(len(activations)), activations)
			plt.title('Asymmetry in soma EPSP peak (mV), kl='+ str(int(slopes[parameter])))
			fig[parameter].savefig(data_folder+'_Ih_conductance_kl_'+str(int(slopes[parameter]))+'.png', dpi=250)
			plt.close(fig[parameter])

	def exp_9(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances and gradients
		"""
		# FIXME
		npol = 3 

		# things to measure
		measures = ['epsp', 'ss']

		locations = ['soma', 'basal']

		conditions = ['cathodal', 'control', 'anodal', 'asymmetry']

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		plot_files = [file for file in files if 'trace' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]
		plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		# print id_list

		# kwargs = run_control.Arguments('exp_6').kwargs
		# g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		# grad_range = run_control.Arguments('exp_6').kwargs['grad_range']
		g_range = kwargs['conductance_range']
		activation_range = kwargs['activation_range_h']
		slope_range = kwargs['slope_range']
		
		# N-dimensional array for storing data (each tested parameter is a dimension)
		
		# varaible to decide whether to create new matrix to store data
		new_mat=0
		# iterate over measures, conditions, locations
		mat={}
		for measure_key, measure in enumerate(measures):
			mat[measure] = {}
			for condition_i, condition in enumerate(conditions):
				mat[measure][condition]= {}
				for location_i, location in enumerate(locations):
					# string to save data
					save_string = '_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'
					# if data file already exists
					if save_string in files:
						# load data
						with open(data_folder+save_string, 'rb') as pkl_file:
							mat[measure][condition][location] = pickle.load(pkl_file)
					else:
						new_mat=1
						mat[measure][condition][location] = np.zeros([len(g_range), len(activation_range), len(slope_range)])

		# vectors of parameter values
		g_h = g_range*kwargs['ghd'] 
		activations = activation_range+kwargs['vhalfl_hd_prox']
		slopes = kwargs['kl_hd'] - slope_range
		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					# detect parameter combination for specific file
					gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
					activations_i = [i for i, val in enumerate(activations) if p['vhalfl_hd_prox']==val]
					slopes_i = [i for i, val in enumerate(slopes) if p['kl_hd']==val]
					
					# epsp onset
					onset = int(p['warmup']*1./p['dt'])
					print onset

					# iterate over things to measure
					for measure_i, measure in enumerate(measures):
						# iterate over locations to record from
						for location_i, location in enumerate(locations):
							# iterate over stimulation conditions
							for condition_i, condition in enumerate(conditions):
								# set window to take measurements
								if 'epsp' in measure:
									window1 = onset
									window2 = len(data['t'][0])
								elif 'ss' in measure:
									window1 = onset-2
									window2 = onset-1

								if condition is not 'asymmetry':
									# measure peak epsp and store in data array
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = np.amax(data[location+'_v'][condition_i][0][-1][window1:window2])
									# print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]

								elif condition is 'asymmetry':
									# measure asymmetry as anodal effect + cathodal effect
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = mat[measure]['anodal'][location][gh_i, activations_i, slopes_i] + mat[measure]['cathodal'][location][gh_i, activations_i, slopes_i] - 2.*mat[measure]['control'][location][gh_i, activations_i, slopes_i]

									if mat[measure][condition][location][gh_i, activations_i, slopes_i] > 1:
										print measure, location, condition, ':', mat[measure][condition][location][gh_i, activations_i, slopes_i]

					fig = plt.figure()
					h_trace = data
		# save processed list
		with open(data_folder+'id_list'+'.pkl', 'wb') as output:pickle.dump(id_list, output,protocol=pickle.HIGHEST_PROTOCOL)
		
		# save asymmetry data matrix 
		for measure_i, measure in enumerate(measures):
			for location_i, location in enumerate(locations):
				for condition_i, condition in enumerate(conditions):
					save_data = mat[measure][condition][location]
					save_string = data_folder+'_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					# save matrix for each condition separately
					with open(save_string, 'wb') as output:
						pickle.dump(save_data, output,protocol=pickle.HIGHEST_PROTOCOL)

		# find maximum asymmetry
		# best_val = np.amax(mat['epsp']['asymmetry']['soma'])
		# best = np.argmax(mat['epsp']['asymmetry']['soma'])
		# best_idx  = np.unravel_index(best, (len(g_range), len(activation_range), len(slope_range)))
		# print best
		# print len(id_list)
		# best_file = id_list[best]
		# print best_idx
		# print best_val
		# print best_file

		# plot asymmetry as a function of parameters
		fig=[]
		for parameter in range(mat['epsp']['asymmetry']['soma'].shape[2]):
			fig.append(plt.figure(parameter))
			plot_mat = mat['epsp']['asymmetry']['soma'][:,:,parameter]#best_idx[2]]
			plt.imshow(plot_mat)
			plt.colorbar()
			plt.ylabel('Ih conductance')
			plt.xlabel('Ih activation gate half max voltage')
			plt.yticks(range(len(g_h)), g_h)
			plt.xticks(range(len(activations)), activations)
			plt.title('Asymmetry in soma EPSP peak (mV), kl='+ str(int(slopes[parameter])))
			fig[parameter].savefig(data_folder+'_Ih_conductance_kl_'+str(int(slopes[parameter]))+'.png', dpi=250)
			plt.close(fig[parameter])

	def exp_10(self, **kwargs):
		""" plot asymmetric voltage change at soma as a function of Ih and Ka conductances and gradients
		"""
		# FIXME
		npol = 3 

		# things to measure
		measures = ['epsp', 'ss']

		locations = ['soma', 'basal']

		conditions = ['cathodal', 'control', 'anodal', 'asymmetry']

		# identify data folder
		data_folder = 'Data/'+kwargs['experiment']+'/'

		# all files in directory
		files = os.listdir(data_folder)
		# data files
		data_files = [file for file in files if 'data' in file]
		plot_files = [file for file in files if 'trace' in file]
		# unique identifiers for each file type
		data_files_id= [file[-36:-1] for file in files if 'data' in file]
		plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

		# if group variable in folder, load variable
		if 'id_list.pkl' in files:
			print 'id_list found'
			
			with open(data_folder+'id_list'+'.pkl', 'rb') as pkl_file:
					id_list = pickle.load(pkl_file)
		
		# otherwise create variable
		else:
			id_list = []

		# print id_list

		# kwargs = run_control.Arguments('exp_6').kwargs
		# g_range = run_control.Arguments('exp_6').kwargs['conductance_range']
		# grad_range = run_control.Arguments('exp_6').kwargs['grad_range']
		g_range = kwargs['conductance_range']
		activation_range = kwargs['activation_range_h']
		slope_range = kwargs['slope_range']
		
		# N-dimensional array for storing data (each tested parameter is a dimension)
		
		# varaible to decide whether to create new matrix to store data
		new_mat=0
		# iterate over measures, conditions, locations
		mat={}
		for measure_key, measure in enumerate(measures):
			mat[measure] = {}
			for condition_i, condition in enumerate(conditions):
				mat[measure][condition]= {}
				for location_i, location in enumerate(locations):
					# string to save data
					save_string = '_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'
					# if data file already exists
					if save_string in files:
						# load data
						with open(data_folder+save_string, 'rb') as pkl_file:
							mat[measure][condition][location] = pickle.load(pkl_file)
					else:
						new_mat=1
						mat[measure][condition][location] = np.zeros([len(g_range), len(activation_range), len(slope_range)])

		# vectors of parameter values
		g_h = g_range*kwargs['ghd'] 
		activations = activation_range+kwargs['vhalfl_hd_prox']
		slopes = kwargs['kl_hd'] - slope_range
		# iterate over data files
		for data_file_i, data_file in enumerate(data_files):
			
			if data_file_i >=0:
				print 'data_file:', data_file_i

				# check if data has been processed already
				if data_file not in id_list:

					# open unprocessed data
					with open(data_folder+data_file, 'rb') as pkl_file:
						data = pickle.load(pkl_file)

						print 'file', data_file_i,  'opened'
					
					# add to processed list
					id_list.append(data_file)

					# parameter dictionary
					p = data['p']

					# detect parameter combination for specific file
					gh_i = [i for i, val in enumerate(g_h) if p['ghd']==val]
					activations_i = [i for i, val in enumerate(activations) if p['vhalfl_hd_prox']==val]
					slopes_i = [i for i, val in enumerate(slopes) if p['kl_hd']==val]
					
					# epsp onset
					onset = int(p['warmup']*1./p['dt'])
					print onset

					# iterate over things to measure
					for measure_i, measure in enumerate(measures):
						# iterate over locations to record from
						for location_i, location in enumerate(locations):
							# iterate over stimulation conditions
							for condition_i, condition in enumerate(conditions):
								# set window to take measurements
								if 'epsp' in measure:
									window1 = onset
									window2 = len(data['t'][0])
								elif 'ss' in measure:
									window1 = onset-2
									window2 = onset-1

								if condition is not 'asymmetry':
									# measure peak epsp and store in data array
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = np.amax(data[location+'_v'][condition_i][0][-1][window1:window2])
									# print measure, location, condition, ':', mat[measure][condition][location][gh_i, ka_i, grad_h_i, grad_ka_i]

								elif condition is 'asymmetry':
									# measure asymmetry as anodal effect + cathodal effect
									mat[measure][condition][location][gh_i, activations_i, slopes_i] = mat[measure]['anodal'][location][gh_i, activations_i, slopes_i] + mat[measure]['cathodal'][location][gh_i, activations_i, slopes_i] - 2.*mat[measure]['control'][location][gh_i, activations_i, slopes_i]

									if mat[measure][condition][location][gh_i, activations_i, slopes_i] > 1:
										print measure, location, condition, ':', mat[measure][condition][location][gh_i, activations_i, slopes_i]

					fig = plt.figure()
					h_trace = data
		# save processed list
		with open(data_folder+'id_list'+'.pkl', 'wb') as output:pickle.dump(id_list, output,protocol=pickle.HIGHEST_PROTOCOL)
		
		# save asymmetry data matrix 
		for measure_i, measure in enumerate(measures):
			for location_i, location in enumerate(locations):
				for condition_i, condition in enumerate(conditions):
					save_data = mat[measure][condition][location]
					save_string = data_folder+'_IhMat_'+'_'+measure+'_'+location+'_'+condition+'.pkl'

					# save matrix for each condition separately
					with open(save_string, 'wb') as output:
						pickle.dump(save_data, output,protocol=pickle.HIGHEST_PROTOCOL)

		# find maximum asymmetry
		# best_val = np.amax(mat['epsp']['asymmetry']['soma'])
		# best = np.argmax(mat['epsp']['asymmetry']['soma'])
		# best_idx  = np.unravel_index(best, (len(g_range), len(activation_range), len(slope_range)))
		# print best
		# print len(id_list)
		# best_file = id_list[best]
		# print best_idx
		# print best_val
		# print best_file

		# plot asymmetry as a function of parameters
		fig=[]
		for parameter in range(mat['epsp']['asymmetry']['soma'].shape[2]):
			fig.append(plt.figure(parameter))
			plot_mat = mat['epsp']['asymmetry']['soma'][:,:,parameter]#best_idx[2]]
			plt.imshow(plot_mat)
			plt.colorbar()
			plt.ylabel('Ih conductance')
			plt.xlabel('Ih activation gate half max voltage')
			plt.yticks(range(len(g_h)), g_h)
			plt.xticks(range(len(activations)), activations)
			plt.title('Asymmetry in soma EPSP peak (mV), kl='+ str(int(slopes[parameter])))
			fig[parameter].savefig(data_folder+'_Ih_conductance_kl_'+str(int(slopes[parameter]))+'.png', dpi=250)
			plt.close(fig[parameter])


if __name__ =="__main__":
	# Weights(param.exp_3().p)
	# # Spikes(param.exp_3().p)
	kwargs = run_control.Arguments('exp_8').kwargs
	# plots = Voltage()
	# plots.plot_all(param.Experiment(**kwargs).p)
	Experiment(**kwargs)




