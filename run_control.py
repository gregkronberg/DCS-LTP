

"""
run control
"""
# imports
from neuron import h
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import cell
# import itertools as it
# import stims
# import pickle
import param
import cell
import run
import time
import uuid
import analysis
import sys
import copy

h.load_file("stdrun.hoc")

# 
class Experiment:
	""" Impliment experimental procedures.  Paramters/arguments can be set using the Arguments class
	"""
	def __init__(self, **kwargs):
		""" choose experiment to run

		kwargs must be a dictionary with 'experiment' as a key, the corresponding value points to a given experiment, which is then fed the same kwarg dictionary and run
		"""
		# retrieve which experiment to run
		experiment = getattr(self, kwargs['experiment'])

		# run experiment
		experiment(**kwargs) 

	# random fraction of all synapses in a given tree
	def exp_1(self, **kwargs):
		""" randomly activate subset of synapses

		vary synaptic weights, fraction of activated synapses
		"""
		weights = [.001]
		self.kwargs = {
		'experiment' : 'exp_1', 
		'trees' : ['apical_tuft'],
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .1,
		'field':[-10.,0.,10.],
		'KMULT':.2*.03,
		'KMULTP':.2*.03,
		}

		# instantiate default parameter class
		self.p_class = param.Default()

		# reference to default parameters
		self.p = self.p_class.p

		# update parameters
		for key, val in self.kwargs.iteritems():		# update parameters
			self.p[key] = val

		# data and figure folder
		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell and store in parameter dictionary
		cell1 = cell.CellMigliore2005(self.p)
		cell1.geometry(self.p)
		cell1.mechanisms(self.p)
		# cell1 = cell.PyramidalCylinder(self.p)

		# measure distance of each segment from the soma and store in parameter dictionary
		self.p_class.seg_distance(cell1)

		# randomly choose active segments 
		self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=self.p['syn_frac'])
		
		

		# loop over trials
		for tri in range(self.p['trials']):
			# loop over weights
			for w_i,w in enumerate(self.p['w_mean']):
				# choose fraction of synapses to be activated
				# syn_frac = np.random.normal(loc=.1, scale=.1) # chosen from gaussian
				
				# load rest of parameters from parameter module
				# p = param.Experiment(exp=exp, tree=tree, w_mean=w, w_std=w_std, w_rand=w_rand, syn_frac=syn_frac).p
				
				# update weight parameter
				self.p['w_mean'] = w

				# set weights for active segments
				self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

				# store trial number
				self.p['trial']=tri
				
				# create unique identifier for each trial
				self.p['trial_id'] = str(uuid.uuid4())
				
				# start timer
				start = time.time() 
				
				# run simulation
				sim = run.Run(self.p)	

				# end timer
				end = time.time() 

				# print trial and simulation time
				print 'trial'+ str(tri) + ' duration:' + str(end -start) 
				
				# set file name to save data
				file_name = str(
				'data_'+
				self.p['experiment']+
				'_weight_'+str(self.p['w_mean'])+
				'_trial_'+str(self.p['trial'])+
				'_synfrac_'+str(self.p['syn_frac'])+
				'_id_'+self.p['trial_id']
				)

				# save data for eahc trial
				run.save_data(data=sim.data, file_name=file_name)

				# plot traces
				analysis.PlotRangeVar().plot_trace(data=sim.data, 
					trees=self.p['trees'], 
					sec_idx=self.p['sec_idx'], 
					seg_idx=self.p['seg_idx'],
					variables=self.p['plot_variables'],
					x_variables=self.p['x_variables'],
					file_name=self.p['trial_id'],
					group_trees=self.p['group_trees'])

	# choose specific synapses
	def exp_2(self, **kwargs):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		sequence_delay_list=[3.]
		n_syn=3.
		self.kwargs = {
		'experiment' : 'exp_2', 
		'trees' : ['apical_tuft'],
		'num_sec':1,
		'seg_L' : 4.,
		'seg_spacing':6,
		'max_seg':1,
		'branch_distance':250,
		'branch_seg_distance':[0, 90],
		'sequence_delay': 4,
		'sequence_direction':'in',
		'trials' : 1,
		'w_mean' : n_syn*.001,
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .1,
		'field':[-20.,0.,20.],
		'KMULT':.5*.03,
		'KMULTP':.5*.03,
		'pulses':1,
		'groupt_trees':False,
		'plot_variables':['v','gbar'],
		}

		# instantiate default parameter class
		self.p_class = param.Default()

		# reference to default parameters
		self.p = self.p_class.p

		# update parameters
		for key, val in self.kwargs.iteritems():		# update parameters
			self.p[key] = val

		# data and figure folder
		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell and store in parameter dictionary
		cell1 = cell.CellMigliore2005(self.p)
		cell1.geometry(self.p)

		# choose random branch to activate
		# self.p_class.choose_branch_rand(trees=self.p['trees'], geo=cell1.geo, num_sec=self.p['num_sec'], distance=self.p['branch_distance'])
		self.p_class.choose_branch_manual(geo=cell1.geo, trees=self.p['trees'], sec_list=[-1, -2, -3, -4])

		# update branch discretization
		cell1.set_branch_nseg(geo=cell1.geo, sec_idx=self.p['sec_idx'], seg_L=self.p['seg_L'])

		# measure distance of each segment from the soma and store in parameter dictionary
		self.p_class.seg_distance(cell1)

		# choose segments on branch to activate
		self.p_class.choose_seg_branch(geo=cell1.geo, sec_idx=self.p['sec_idx'], seg_dist=self.p['seg_dist'], spacing=self.p['seg_spacing'], distance=self.p['branch_seg_distance'], max_seg=self.p['max_seg'])

		# insert mechanisms
		cell1.mechanisms(self.p)

		# set weights for active segments
		self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		# loop over trials
		for tri in range(self.p['trials']):
			# loop over delays
			for seq_del_i, seq_del in enumerate(sequence_delay_list):

				self.p['sequence_delay']=seq_del

				# set branch input sequence
				self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

				print self.p['sec_idx']
				print self.p['seg_idx']
				print self.p['sequence_delays']

				# store trial number
				self.p['trial']=tri
				
				# create unique identifier for each trial
				self.p['trial_id'] = str(uuid.uuid4())
				
				# start timer
				start = time.time() 
				
				# run simulation
				sim = run.Run(self.p)	

				# end timer
				end = time.time() 

				# print trial and simulation time
				print 'trial'+ str(tri) + ' duration:' + str(end -start) 
				
				# set file name to save data
				file_name = str(
				'data_'+
				self.p['experiment']+
				'_weight_'+str(self.p['w_mean'])+
				'_trial_'+str(self.p['trial'])+
				'_synfrac_'+str(self.p['syn_frac'])+
				'_id_'+self.p['trial_id']
				)

				# save data for eahc trial
				run.save_data(data=sim.data, file_name=file_name)

				# plot traces
				analysis.PlotRangeVar().plot_trace(data=sim.data, 
					trees=self.p['trees'], 
					sec_idx=self.p['sec_idx'], 
					seg_idx=self.p['seg_idx'],
					variables=self.p['plot_variables'],
					x_variables=self.p['x_variables'],
					file_name=self.p['trial_id'],
					group_trees=self.p['group_trees'])

	# plot spike generation as a function of distance from soma
	def exp_2a(self, **kwargs):
		""" activate synapses at varying distnace from soma until a local spike is generated.  Determine whether spike was initiated in the soma or dendrite
		"""
		w_mean = .001
		self.kwargs = {
		'experiment' : 'exp_2a', 
		'trees' : ['apical_tuft'],
		'nsyns':range(2,16,2),
		'num_sec':1,
		'seg_L' : 4.,
		'seg_spacing':20,
		'max_seg':[],
		'branch':False,
		'full_path':False,
		'branch_distance':[],
		'branch_seg_distance':[],
		'sequence_delay': 0,
		'sequence_direction':'in',
		'trials' : 5,
		'w_mean' : [],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .1,
		'field':[-20.,0.,20.],
		'KMULT':.5*.03,
		'KMULTP':.5*.03,
		'pulses':4,
		'pulse_freq':200,
		'groupt_trees':False,
		'plot_variables':['v','gbar'],
		'cell':[]
		}

		# instantiate default parameter class
		self.p_class = param.Default()

		# reference to default parameters
		self.p = self.p_class.p

		# update parameters
		for key, val in self.kwargs.iteritems():		# update parameters
			self.p[key] = val

		# data and figure folder
		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# iterate over trials
		for trial_i, trial in enumerate(range(self.p['trials'])):
			# load cell and store in parameter dictionary
			cell1 = cell.CellMigliore2005(self.p)
			cell1.geometry(self.p)

			# choose section
			self.p_class.choose_branch_rand(trees=self.p['trees'], geo=cell1.geo, num_sec=self.p['num_sec'], distance=self.p['branch_distance'], branch=self.p['branch'], full_path=self.p['full_path'])

			# update branch discretization
			cell1.set_branch_nseg(geo=cell1.geo, sec_idx=self.p['sec_idx'], seg_L=self.p['seg_L'])

			# measure distance of each segment from the soma and store in parameter dictionary
			self.p_class.seg_distance(cell1)

			# choose segments on branch to activate
			self.p_class.choose_seg_branch(geo=cell1.geo, sec_idx=self.p['sec_idx'], seg_dist=self.p['seg_dist'], spacing=self.p['seg_spacing'], distance=self.p['branch_seg_distance'], max_seg=self.p['max_seg'])

			# insert mechanisms
			cell1.mechanisms(self.p)

			# update segment to stimulate
			seg_idx_copy = copy.copy(self.p['seg_idx'])

			for tree_key, tree in seg_idx_copy.iteritems():
				if tree_key in self.p['trees']:
					for sec_i, sec in enumerate(tree):
						for seg_i, seg in enumerate(sec):
							distance_from_soma = self.p['seg_dist'][tree_key][self.p['sec_idx'][tree_key][sec_i]][seg]
							self.p['seg_idx'][tree_key][sec_i] = [seg]

							# iterate over number of synapses
							for nsyn_i, nsyn in enumerate(self.p['nsyns']):
								self.p['w_mean'] = nsyn*w_mean
								self.p['nsyn']=nsyn

								# set weights for active segments
								self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

								# set branch input sequence
								self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

								print self.p['sec_idx']
								print self.p['seg_idx']
								print 'nsyn:',nsyn, 'w (nS):',self.p['w_mean'] 

								# store trial number
								self.p['trial']=trial
								
								# create unique identifier for each trial
								self.p['trial_id'] = str(uuid.uuid4())
								
								# start timer
								start = time.time() 
								
								# print cell1.syns
								# run simulation
								sim = run.Run(p=self.p, cell=cell1)	

								# end timer
								end = time.time() 

								# print trial and simulation time
								print 'trial'+ str(self.p['trial']) + ' duration:' + str(end -start) 
								
								# set file name to save data
								file_name = str(
								self.p['experiment']+
								'_weight_'+str(self.p['w_mean'])+
								'_trial_'+str(self.p['trial'])+
								'_dist_'+str(distance_from_soma)+
								'_id_'+self.p['trial_id']
								)

								# save data for eahc trial
								run.save_data(data=sim.data, file_name=file_name)

								# plot traces
								analysis.PlotRangeVar().plot_trace(data=sim.data, 
									trees=self.p['trees'], 
									sec_idx=self.p['sec_idx'], 
									seg_idx=self.p['seg_idx'],
									variables=self.p['plot_variables'],
									x_variables=self.p['x_variables'],
									file_name=file_name,
									group_trees=self.p['group_trees'],
									xlim=[self.p['warmup']-5,self.p['tstop']],
									ylim=[-70, -40])

	
	def exp_2b(self, **kwargs):
			""" active a random group of segments with varying mean distance from soma and varying weights (number of synapses). monitor spike initiation in soma/dendrite as a function of this mean distance
			"""
			w_mean = .001
			distances = [[0,100], [100,200], [200,300],[300,400],[400,500]]
			self.kwargs = {
			'experiment' : 'exp_2b', 
			'trees' : ['apical_tuft', 'apical_trunk'],
			'nsyns':range(1,16,1),
			'num_sec':1,
			'seg_L' : 4.,
			'seg_spacing':20,
			'max_seg':[],
			'branch':False,
			'full_path':False,
			'branch_distance':[],
			'branch_seg_distance':[],
			'sequence_delay': 0,
			'sequence_direction':'in',
			'trials' : 5,
			'w_mean' : [],
			'w_std' : [.002],
			'w_rand' : False, 
			'syn_frac' : .2,
			'field':[-20.,0.,20.],
			'KMULT':1.*.03,
			'KMULTP':1.*.03,
			'ka_grad':1.,
			'SOMAM': 1.,
			'AXONM': 40.,
			'dgna':1.*-.000025,
			'pulses':4,
			'pulse_freq':100,
			'group_trees':False,
			'plot_variables':['v','gbar'],
			'cell':[]
			}

			# instantiate default parameter class
			self.p_class = param.Default()

			# reference to default parameters
			self.p = self.p_class.p

			# update parameters
			for key, val in self.kwargs.iteritems():		# update parameters
				self.p[key] = val

			# data and figure folder
			self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
			self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

			# load cell and store in parameter dictionary
			cell1 = cell.CellMigliore2005(self.p)
			cell1.geometry(self.p)
			# insert mechanisms
			cell1.mechanisms(self.p)
			# measure distance of each segment from the soma and store in parameter dictionary
			self.p_class.seg_distance(cell1)

			# iterate over distance
			for distance_i, distance in enumerate(distances):
				for trial_i, trial in enumerate(range(self.p['trials'])):
					
					print distance
					if distance:
						distance_from_soma = distance[0]
					else:
						distance_from_soma='None'

					self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=self.p['syn_frac'], seg_dist=self.p['seg_dist'], distance=distance)

					for nsyn_i, nsyn in enumerate(self.p['nsyns']):

						self.p['w_mean'] = nsyn*w_mean

						self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

						self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])


						print self.p['sec_idx']
						print self.p['seg_idx']
						print 'nsyn:',nsyn, 'w (nS):',self.p['w_mean'] 

						# store trial number
						self.p['trial']=trial
						
						# create unique identifier for each trial
						self.p['trial_id'] = str(uuid.uuid4())
						
						# start timer
						start = time.time() 
						
						# print cell1.syns
						# run simulation
						sim = run.Run(p=self.p, cell=cell1)	

						# end timer
						end = time.time() 

						# print trial and simulation time
						print 'trial'+ str(self.p['trial']) + ' duration:' + str(end -start) 
						
						# set file name to save data
						file_name = str(
						self.p['experiment']+
						'_weight_'+str(self.p['w_mean'])+
						'_trial_'+str(self.p['trial'])+
						'_dist_'+str(distance_from_soma)+
						'_id_'+self.p['trial_id']
						)

						# save data for eahc trial
						run.save_data(data=sim.data, file_name=file_name)

						# plot traces
						analysis.PlotRangeVar().plot_trace(data=sim.data, 
							trees=self.p['trees'], 
							sec_idx=self.p['sec_idx'], 
							seg_idx=self.p['seg_idx'],
							variables=self.p['plot_variables'],
							x_variables=self.p['x_variables'],
							file_name=file_name,
							group_trees=self.p['group_trees'],
							xlim=[self.p['warmup']-5,self.p['tstop']],
							ylim=[])

	def exp_2c(self, **kwargs):
		""" active each segment with varying weights (number of synapses). monitor spike initiation in soma/dendrite as a function of distance from soma

		"""
		w_mean = .001 # weight of single synapse uS
		trees = ['apical_trunk', 'apical_tuft']
		self.kwargs = {
		'experiment' : 'exp_2c', 
		'trees' : ['apical_tuft', 'apical_trunk'],
		'nsyns':range(4,26,2),
		'num_sec':1,
		'seg_L' : 4.,
		'seg_spacing':20,
		'max_seg':[],
		'branch':False,
		'full_path':False,
		'branch_distance':[],
		'branch_seg_distance':[],
		'sequence_delay': 0,
		'sequence_direction':'in',
		'trials' : 1,
		'w_mean' : [],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .2,
		'field':[-20.,0.,20.],
		'KMULT':1.*.03,
		'KMULTP':1.*.03,
		'ka_grad':1.,
		'SOMAM': 1.5,
		'AXONM': 50.,
		'gna':.04,
		'dgna':1.*-.000025,
		'pulses':1,
		'pulse_freq':100,
		'group_trees':False,
		'plot_variables':['v','gbar'],
		'cell':[],
		'tstop':40
		}

		# instantiate default parameter class
		self.p_class = param.Default()

		# reference to default parameters
		self.p = self.p_class.p

		# update parameters
		for key, val in self.kwargs.iteritems():		# update parameters
			self.p[key] = val

		# data and figure folder
		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell and store in parameter dictionary
		cell1 = cell.CellMigliore2005(self.p)
		cell1.geometry(self.p)
		# insert mechanisms
		cell1.mechanisms(self.p)
		# measure distance of each segment from the soma and store in parameter dictionary
		self.p_class.seg_distance(cell1)

		spike_analysis = analysis.Spikes()
		threshold=-30

		# iterate over all segments in tree
		for trial_i, trial in enumerate(range(self.p['trials'])):
			for tree_key, tree in cell1.geo.iteritems():
				if tree_key in trees:
					self.p['trees'] = [tree_key]
					for sec_i, sec in enumerate(tree):
						for seg_i, seg in enumerate(sec):
							sec_list = {tree_key:[sec_i]}
							seg_list = {tree_key:[seg_i]}
							print sec_list, seg_list
							distance_from_soma = self.p['seg_dist'][tree_key][sec_i][seg_i]

							self.p_class.choose_seg_manual(trees=self.p['trees'], sec_list=sec_list, seg_list=seg_list)

							for nsyn_i, nsyn in enumerate(self.p['nsyns']):

								self.p['w_mean'] = nsyn*w_mean

								self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

								self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

								print self.p['sec_idx']
								print self.p['seg_idx']
								print 'nsyn:',nsyn, 'w (nS):',self.p['w_mean'] 
								print 'distance from soma:', distance_from_soma

								# store trial number
								self.p['trial']=trial
								
								# create unique identifier for each trial
								self.p['trial_id'] = str(uuid.uuid4())
								
								# start timer
								start = time.time() 
								
								# print cell1.syns
								# run simulation
								sim = run.Run(p=self.p, cell=cell1)	

								# end timer
								end = time.time() 

								# print trial and simulation time
								print 'trial'+ str(self.p['trial']) + ' duration:' + str(end -start) 
								
								# set file name to save data
								file_name = str(
								self.p['experiment']+
								'_weight_'+str(self.p['w_mean'])+
								'_trial_'+str(self.p['trial'])+
								'_dist_'+str(distance_from_soma)+
								'_id_'+self.p['trial_id']
								)

								# save data for eahc trial
								run.save_data(data=sim.data, file_name=file_name)

								# plot traces
								analysis.PlotRangeVar().plot_trace(data=sim.data, 
									trees=self.p['trees'], 
									sec_idx=self.p['sec_idx'], 
									seg_idx=self.p['seg_idx'],
									variables=self.p['plot_variables'],
									x_variables=self.p['x_variables'],
									file_name=file_name,
									group_trees=self.p['group_trees'],
									xlim=[self.p['warmup']-5,self.p['tstop']],
									ylim=[])
								
								spike_count=0
								for f_i, f in enumerate(self.p['field']):
									# if threshold crossed, continue
									dend_spike = spike_analysis.detect_spikes(sim.data[str(f)][tree_key+'_v'][0][0], threshold=threshold)['times'][0]

									soma_spike = spike_analysis.detect_spikes(sim.data[str(f)]['soma'+'_v'][0][0], threshold=threshold)['times'][0]

									if len(dend_spike)>0 or len(soma_spike)>0:
										spike_count+=1

								print 'spike count:', spike_count
								if spike_count ==len(self.p['field']):
									print 'spike detected for all polarities'
									break
				




	# random fraction of all synapses in a given tree
	def exp_3a(self, **kwargs):
		exp = 'exp_3'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		syn_frac = kwargs['syn_frac']

		# loop over trials
		for tri in range(trials):
			# loop over weights
			for w_i,w in enumerate(w_mean):
				# choose fraction of synapses to be activated
				# syn_frac = np.random.normal(loc=.1, scale=.1) # chosen from gaussian
				
				# load rest of parameters from parameter module
				p = param.Experiment(experiment=exp, tree=tree, w_mean=w, w_std=w_std, w_rand=w_rand, syn_frac=syn_frac).p
				
				# store trial number
				p['trial']=tri
				
				# create unique identifier for each trial
				p['trial_id'] = str(uuid.uuid4())
				
				# start timer
				start = time.time() 
				
				# run simulation
				sim = run.Run(p)	

				# end timer
				end = time.time() 

				# print trial and simulation time
				print 'trial'+ str(tri) + ' duration:' + str(end -start) 
				
				# save data for eahc trial
				run.save_data(sim.data)

		self.p = p

	def exp_4(self, **kwargs):
		exp = 'exp_4'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		syn_frac = kwargs['syn_frac']

		# loop over trials
		for tri in range(trials):
			for gh_i, gh in enumerate(kwargs['conductance_range']):
				for gka_i, gka in enumerate(kwargs['conductance_range']):
					
					# load rest of parameters from parameter module
					p = param.Experiment(**kwargs).p
					
					# set Ih and Ka conductance parameters
					p['ghd'] = gh*0.00005
					p['KMULT'] =  gka*0.03
					p['KMULTP'] =  gka*0.03

					print 'g_h:', p['ghd'], 'g_ka:', p['KMULT']

					# store trial number
					p['trial']=tri
					
					# create unique identifier for each trial
					p['trial_id'] = str(uuid.uuid4())
					
					# start timer
					start = time.time() 
					
					# run simulation
					sim = run.Run(p)	

					# end timer
					end = time.time() 

					# print trial and simulation time
					print 'trial'+ str(tri) + ' duration:' + str(end -start) 
					
					# save data for eahc trial
					run.save_data(sim.data)

		self.p = p

	def exp_5(self, **kwargs):
		exp = 'exp_5'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		syn_frac = kwargs['syn_frac']

		# loop over trials
		for tri in range(trials):
			for h_grad_i, h_grad in enumerate(kwargs['grad_range']):
				for ka_grad_i, ka_grad in enumerate(kwargs['grad_range']):
					
					# load rest of parameters from parameter module
					p = param.Experiment(**kwargs).p
					
					# set Ih and Ka conductance parameters
					p['ghd_grad'] = h_grad*3.
					p['ka_grad'] =  ka_grad*1

					# store trial number
					p['trial']=tri
					
					# create unique identifier for each trial
					p['trial_id'] = str(uuid.uuid4())
					
					# start timer
					start = time.time() 
					
					# run simulation
					sim = run.Run(p)	

					# end timer
					end = time.time() 

					# print trial and simulation time
					print 'trial'+ str(tri) + ' duration:' + str(end -start) 
					
					# save data for eahc trial
					run.save_data(sim.data)

		self.p = p

	def exp_6(self, **kwargs):
		""" vary Ih and Ka parameters and measure effects on peak EPSP
		"""
		exp = 'exp_6'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		syn_frac = kwargs['syn_frac']
		plots = analysis.Voltage()

		# loop over trials
		for tri in range(trials):
			for gh_i, gh in enumerate(kwargs['conductance_range']):
				for gka_i, gka in enumerate(kwargs['conductance_range']):
					for gh_grad_i, gh_grad in enumerate(kwargs['grad_range']):
						if gh==0.:
							continue
						for ka_grad_i, ka_grad in enumerate(kwargs['grad_range']):
							if gka==0.:
								continue 


					
							# load rest of parameters from parameter module
							p = param.Experiment(**kwargs).p
							
							# set Ih and Ka conductance parameters
							p['ghd'] = gh*kwargs['ghd']
							p['KMULT'] =  gka*kwargs['KMULT']
							p['KMULTP'] =  gka*kwargs['KMULTP']
							p['ghd_grad'] = gh_grad*kwargs['ghd_grad']
							p['ka_grad'] =ka_grad*kwargs['ka_grad']

							print 'g_h:', p['ghd'], 'g_ka:', p['KMULT'], 'h_grad:', p['ghd_grad'], 'ka_grad:', p['ka_grad']

							# store trial number
							p['trial']=tri
							
							# create unique identifier for each trial
							p['trial_id'] = str(uuid.uuid4())
							
							# start timer
							start = time.time() 
							
							# run simulation
							sim = run.Run(p)	


							# end timer
							end = time.time() 

							# print trial and simulation time
							print 'trial'+ str(tri) + ' duration:' + str(end -start) 
							
							# save data for eahc trial
							run.save_data(sim.data)

							plots.plot_trace(data=sim.data, 
							tree=p['tree'], 
							sec_idx=p['sec_idx'], 
							seg_idx=p['seg_idx'],
							variables=p['plot_variables'])

							



		self.p = p

	def exp_7(self, **kwargs):
		""" vary Ih parameters and measure effects on peak EPSP
		"""
		vhalfl_hd_prox=copy.copy(kwargs['vhalfl_hd_prox'])
		vhalfl_hd_dist=copy.copy(kwargs['vhalfl_hd_dist'])
		ghd = copy.copy(kwargs['ghd'])
		kl_hd = copy.copy(kwargs['kl_hd'])
		plots = analysis.PlotRangeVar()

		# loop over trials
		for tri in range(kwargs['trials']):
			for conductance_i, conductance  in enumerate(kwargs['conductance_range']):
				for vhalfl_h_i, vhalfl_h in enumerate(kwargs['activation_range_h']):
					for slope_i, slope in enumerate(kwargs['slope_range']):
						if conductance==0. and (vhalfl_h_i >0 or slope_i> 0):
							continue

						
						
						# set Ih and Ka conductance parameters
						kwargs['vhalfl_hd_prox'] = vhalfl_h+vhalfl_hd_prox
						kwargs['vhalfl_hd_dist'] = vhalfl_h+vhalfl_hd_dist
						kwargs['ghd'] = conductance*ghd
						kwargs['kl_hd'] = kl_hd-slope

						# load rest of parameters from parameter module
						p = param.Experiment(**kwargs).p

						print 'vhalfl_hd_prox:', p['vhalfl_hd_prox'] 
						print 'ghd:', p['ghd'] 
						print 'kl_hd:', p['kl_hd']


						# store trial number
						p['trial']=tri
						
						# create unique identifier for each trial
						p['trial_id'] = str(uuid.uuid4())
						
						# start timer
						start = time.time() 
						
						# run simulation
						sim = run.Run(p)	


						# end timer
						end = time.time() 

						# print trial and simulation time
						print 'trial'+ str(tri) + ' duration:' + str(end -start) 
						
						# save data for eahc trial
						run.save_data(sim.data)

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'])

		self.p = p
	
	def exp_8(self, **kwargs):
		""" vary Ih parameters and measure effects on peak EPSP
		"""
		vhalfl_hd_prox=copy.copy(kwargs['vhalfl_hd_prox'])
		vhalfl_hd_dist=copy.copy(kwargs['vhalfl_hd_dist'])
		ghd = copy.copy(kwargs['ghd'])
		kl_hd = copy.copy(kwargs['kl_hd'])

		plots = analysis.PlotRangeVar()

		# loop over trials
		for tri in range(kwargs['trials']):
			for conductance_i, conductance  in enumerate(kwargs['conductance_range']):
				for vhalfl_h_i, vhalfl_h in enumerate(kwargs['activation_range_h']):
					for slope_i, slope in enumerate(kwargs['slope_range']):
						if conductance==0. and (vhalfl_h_i >0 or slope_i> 0):
							continue

						# set Ih and Ka conductance parameters
						kwargs['vhalfl_hd_prox'] = vhalfl_h+vhalfl_hd_prox
						kwargs['vhalfl_hd_dist'] = vhalfl_h+vhalfl_hd_dist
						kwargs['ghd'] = conductance*ghd
						kwargs['kl_hd'] = kl_hd-slope

						# load rest of parameters from parameter module
						p = param.Experiment(**kwargs).p

						print 'vhalfl_hd_prox:', p['vhalfl_hd_prox'] 
						print 'ghd:', p['ghd'] 
						print 'kl_hd:', p['kl_hd']


						# store trial number
						p['trial']=tri
						
						# create unique identifier for each trial
						p['trial_id'] = str(uuid.uuid4())
						
						# start timer
						start = time.time() 
						
						# run simulation
						sim = run.Run(p)	

						# end timer
						end = time.time() 

						# print trial and simulation time
						print 'trial'+ str(tri) + ' duration:' + str(end -start) 
						
						# save data for eahc trial
						run.save_data(sim.data)

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'],
						x_var='t')

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'],
						x_var='v')

		self.p = p

	def exp_9(self, **kwargs):
		""" vary Ih parameters and measure effects on peak EPSP
		"""
		vhalfl_hd_prox=copy.copy(kwargs['vhalfl_hd_prox'])
		vhalfl_hd_dist=copy.copy(kwargs['vhalfl_hd_dist'])
		ghd = copy.copy(kwargs['ghd'])
		kl_hd = copy.copy(kwargs['kl_hd'])
		plots = analysis.PlotRangeVar()

		# loop over trials
		for tri in range(kwargs['trials']):
			for conductance_i, conductance  in enumerate(kwargs['conductance_range']):
				for vhalfl_h_i, vhalfl_h in enumerate(kwargs['activation_range_h']):
					for slope_i, slope in enumerate(kwargs['slope_range']):
						if conductance==0. and (vhalfl_h_i >0 or slope_i> 0):
							continue

						
						
						# set Ih and Ka conductance parameters
						kwargs['vhalfl_hd_prox'] = vhalfl_h+vhalfl_hd_prox
						kwargs['vhalfl_hd_dist'] = vhalfl_h+vhalfl_hd_dist
						kwargs['ghd'] = conductance*ghd
						kwargs['kl_hd'] = kl_hd-slope

						# load rest of parameters from parameter module
						p = param.Experiment(**kwargs).p

						print 'vhalfl_hd_prox:', p['vhalfl_hd_prox'] 
						print 'ghd:', p['ghd'] 
						print 'kl_hd:', p['kl_hd']


						# store trial number
						p['trial']=tri
						
						# create unique identifier for each trial
						p['trial_id'] = str(uuid.uuid4())
						
						# start timer
						start = time.time() 
						
						# run simulation
						sim = run.Run(p)	


						# end timer
						end = time.time() 

						# print trial and simulation time
						print 'trial'+ str(tri) + ' duration:' + str(end -start) 
						
						# save data for eahc trial
						run.save_data(sim.data)

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'])

		self.p = p
	
	def exp_10(self, **kwargs):
		""" vary Ih parameters and measure effects on peak EPSP
		"""
		vhalfl_hd_prox=copy.copy(kwargs['vhalfl_hd_prox'])
		vhalfl_hd_dist=copy.copy(kwargs['vhalfl_hd_dist'])
		ghd = copy.copy(kwargs['ghd'])
		kl_hd = copy.copy(kwargs['kl_hd'])
		plots = analysis.PlotRangeVar()

		# loop over trials
		for tri in range(kwargs['trials']):
			for conductance_i, conductance  in enumerate(kwargs['conductance_range']):
				for vhalfl_h_i, vhalfl_h in enumerate(kwargs['activation_range_h']):
					for slope_i, slope in enumerate(kwargs['slope_range']):
						if conductance==0. and (vhalfl_h_i >0 or slope_i> 0):
							continue

						
						
						# set Ih and Ka conductance parameters
						kwargs['vhalfl_hd_prox'] = vhalfl_h+vhalfl_hd_prox
						kwargs['vhalfl_hd_dist'] = vhalfl_h+vhalfl_hd_dist
						kwargs['ghd'] = conductance*ghd
						kwargs['kl_hd'] = kl_hd-slope

						# load rest of parameters from parameter module
						p = param.Experiment(**kwargs).p

						print 'vhalfl_hd_prox:', p['vhalfl_hd_prox'] 
						print 'ghd:', p['ghd'] 
						print 'kl_hd:', p['kl_hd']


						# store trial number
						p['trial']=tri
						
						# create unique identifier for each trial
						p['trial_id'] = str(uuid.uuid4())
						
						# start timer
						start = time.time() 
						
						# run simulation
						sim = run.Run(p)	


						# end timer
						end = time.time() 

						# print trial and simulation time
						print 'trial'+ str(tri) + ' duration:' + str(end -start) 
						
						# save data for eahc trial
						run.save_data(sim.data)

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'])

						plots.plot_trace(data=sim.data, 
						tree=p['tree'], 
						sec_idx=p['sec_idx'], 
						seg_idx=p['seg_idx'],
						variables=p['plot_variables'],
						x_var='v')


		self.p = p
	

class Arguments:
	"""
	"""
	def __init__(self, exp):
		experiment = getattr(self, exp)

		experiment() 

	def exp_1(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [.0015]
		self.kwargs = {
		'experiment' : 'exp_1', 
		'tree' : ['apical_tuft','apical_trunk'],
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .2
		}

	def exp_2(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [0.0005]#[.03]
		self.kwargs = {
		'experiment' : 'exp_2', 
		'tree' : 'apical_tuft' ,
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.0002],
		'w_rand' : False, 
		'sec_idx' : [-1, -2, -3, -4, -5, -6, -7, -8, -9,-10,-11,-12,-13,-14,-15,-16], 
		'seg_idx' : [[0], [0], [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
		'pulses' : 4,
		'gna' : 0.025,
		}

	def exp_3(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [0]
		self.kwargs = {
		'experiment' : 'exp_3', 
		'tree' : 'apical_dist',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .1
		}

	def exp_4(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		# weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [0]
		self.kwargs = {
		'conductance_range' : np.arange(.1, 3, .5),
		'experiment' : 'exp_4', 
		'tree' : 'apical_dist',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0
		}

	def exp_5(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		# weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [0]
		self.kwargs = {
		'grad_range' : np.arange(0, 1, .2),
		'KMULT' : 0.1*.03, # chosen based on experiment 4 results to bas towards depolarization
		'KMULTP' : 0.1*.03,
		'ghd' : 1*0.00005,
		'experiment' : 'exp_5', 
		'tree' : 'apical_dist',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0
		}

	def exp_6(self):
		""" vary Ih and Ka parameters and measure effects on peak EPSP
		"""
		
		weights = 0.03
		self.kwargs = {
		'conductance_range' : np.arange(0., 3., .5),
		'grad_range' :  np.arange(0., 3., .5),
		'ghd' : 0.00005,
		'KMULT' :  0.03,
		'KMULTP' :  0.03,
		'ghd_grad' : .75,
		'ka_grad' : .25,
		'experiment' : 'exp_6', 
		'tree' : 'apical_dist',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [0, -1],
		'sec_list' : [ 0, 0],
		'pulses':3,
		}

	def exp_7(self):
		""" vary Ih parameters and measure effects on peak EPSP
		"""
		
		weights = 0.002
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 4.),
		'conductance_range' : np.arange(0., 3., .5),
		'slope_range' : np.arange(0.,4., 1.),
		'gna' : 0.,
		'kl_hd' : -4,
		'ghd' : 0.00005,
		'KMULT' :  .5*0.03,
		'KMULTP' :  .5*0.03,
		'ghd_grad' : 5,
		'ka_grad' : 1,
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_7', 
		'tree' : 'apical_tuft',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [-1, -1,],
		'sec_list' : [-1, -2,],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		}

	def exp_8(self):
		""" repeat experiment 7 using simplified cylinder model

		vary Ih parameters and measure effects on peak EPSP
		"""
		
		weights = 0.001
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 5.),
		'conductance_range' : np.arange(0., 2., .5),
		'slope_range' : np.arange(0.,4., 2.),
		'gna' : 0.,
		'dgna':0,
		'Vrest':-75.,
		'gcalbar': 0.,
		'kl_hd' : -4.,
		'ghd' : 0.00005,
		'KMULT' :  .5*0.03,
		'KMULTP' :  .5*0.03,
		'ghd_grad' : 5.,
		'ka_grad' : 1.,
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_8', 
		'tree' : 'basal',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [-20,],
		'sec_list' : [0,],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		'field':[-30,0,30],
		'L_basal' : 1600./4.,
		'L_soma' : 7.5,
		'L_apical_prox' : 1000./4.,
		'L_apical_dist' : 1000./4.,
		'diam1_basal' : 1.*1.9/4.,
		'diam1_soma' : 7.5,
		'diam1_apical_prox' : 1.*2.75/4.,
		'diam1_apical_dist' : 1.*2.75/6.,
		'diam2_basal' : 1.9/10.,
		'diam2_soma' : 7.5,
		'diam2_apical_prox' : 2.75/6.,
		'diam2_apical_dist' : 2.75/10.,
		'RaAll' : 1.*150.,
		'fixnseg':True
		}

	def exp_9(self):
		""" vary Ih parameters and measure effects on peak EPSP in basal dendrites
		"""
		
		weights = 0.005
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 4.),
		'conductance_range' : np.arange(0., 3., .5),
		'slope_range' : np.arange(0.,4., 1.),
		'gna' : 0.,
		'dgna':0.,
		'kl_hd' : -4,
		'ghd' : 0.00005,
		'gcalbar': 0.,
		'KMULT' :  .5*0.03,
		'KMULTP' :  .5*0.03,
		'ghd_grad' : 5,
		'ka_grad' : 1,
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_9', 
		'tree' : 'basal',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [-1, -1,],
		'sec_list' : [-1, -2,],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		'field':[-30,0,30],
		}

	def exp_10(self):
		""" vary Ih parameters and measure effects on peak EPSP in basal dendrites but remove Ih gradient
		"""
		
		weights = 0.005
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 4.),
		'conductance_range' : np.arange(0., 3., .5),
		'slope_range' : np.arange(0.,4., 1.),
		'gna' : 0.,
		'dgna':0.,
		'kl_hd' : -4,
		'ghd' : 0.00005,
		'gcalbar': 0.,
		'KMULT' :  1.*0.03,
		'KMULTP' :  1.*0.03,
		'ghd_grad' : 0,#####
		'ka_grad' : 0,######
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_10', 
		'tree' : 'basal',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [-1, -1,],
		'sec_list' : [-1, -2,],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		'field':[-30,0,30],
		}

	def exp_11(self):
		""" repeat experiment 7 using simplified cylinder model

		vary Ih parameters and measure effects on peak EPSP
		"""
		
		weights = 0.005
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 5.),
		'conductance_range' : np.arange(0., 9., 2.),
		'slope_range' : np.arange(0.,4., 5.),
		'gna' : 0.,
		'dgna':0,
		'Vrest':-80.,
		'gcalbar': 0.,
		'kl_hd' : -4.,
		'ghd' : 0.0004,
		'KMULT' :  .5*0.03,
		'KMULTP' :  .5*0.03,
		'ghd_grad' : 5.,
		'ka_grad' : 1.,
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_8', 
		'tree' : 'basal',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [0,],
		'sec_list' : [0,],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		'field':[-30,0,30],
		'L_basal' : 1600./4.,
		'L_soma' : 7.5,
		'L_apical_prox' : 1000./4.,
		'L_apical_dist' : 1000./4.,
		'diam_basal' : 1.*1.9,
		'diam_soma' : 7.5,
		'diam_apical_prox' : 1.*2.75,
		'diam_apical_dist' : 1.*2.75,
		'RaAll' : 10.*150.,
		}



if __name__ =="__main__":
	kwargs = {'experiment':'exp_2c'}
	# kwargs = Arguments('exp_1').kwargs
	x = Experiment(**kwargs)
	# analysis.Experiment(**kwargs)
	# plots = analysis.PlotRangeVar()
	# plots.plot_all(x.p)
	# analysis.Experiment(exp='exp_3')
