

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
import run
import time
import uuid
import analysis
import sys
h.load_file("stdrun.hoc")

# 
class Experiment:
	""" Impliment experimental procedures.  Paramters/arguments can be set using the Arguments class
	"""
	def __init__(self, **kwargs):
		experiment = getattr(self, kwargs['experiment'])

		experiment(**kwargs) 

	# random fraction of all synapses in a given tree
	def exp_1(self, **kwargs):
		exp = 'exp_1'
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
				p = param.Experiment(exp=exp, tree=tree, w_mean=w, w_std=w_std, w_rand=w_rand, syn_frac=syn_frac).p
				
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

	# choose specific synapses
	def exp_2(self, **kwargs):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		exp = 'exp_2'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		sec_idx = kwargs['sec_idx']
		seg_idx = kwargs['seg_idx']


		# loop over trials
		for tri in range(trials):
			# loop over weights
			for w in w_mean:
				# choose fraction of synapses to be activated
				# syn_frac = np.random.normal(loc=.1, scale=.1) # chosen from gaussian

				kwargs['w_mean'] = w
				# load rest of parameters from parameter module
				p = param.Experiment(**kwargs).p
				
				# store trial number
				p['trial']=tri
				
				# create unique identifier for each trial
				p['trial_id'] = str(uuid.uuid4())
				
				# start timer
				start = time.time() 
				
				# run simulation
				sim = run.Run(p)	

				# create shape plot
				# sim.shape_plot(p)

				# end timer
				end = time.time() 

				# print trial and simulation time
				print 'trial'+ str(tri) + ' duration:' + str(end -start) 
				
				# save data for eahc trial
				run.save_data(sim.data)

		self.p = p

	# random fraction of all synapses in a given tree
	def exp_3(self, **kwargs):
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

		# loop over trials
		for tri in range(trials):
			for gh_i, gh in enumerate(kwargs['conductance_range']):
				for gka_i, gka in enumerate(kwargs['conductance_range']):
					
					# load rest of parameters from parameter module
					p = param.Experiment(**kwargs).p
					
					# set Ih and Ka conductance parameters
					p['ghd'] = gh*0.00005
					p['KMULT'] =  0.*gka*0.03
					p['KMULTP'] =  0.*gka*0.03

					print 'g_h:', p['ghd'], 'g_ka:', p['KMULT']

					# store trial number
					p['trial']=tri
					
					# create unique identifier for each trial
					p['trial_id'] = str(uuid.uuid4())
					
					# start timer
					start = time.time() 
					
					# run simulation
					sim = run.Run(p)	

					print 'g_h:', p['ghd'], 'g_ka:', p['KMULT']

					# end timer
					end = time.time() 

					# print trial and simulation time
					print 'trial'+ str(tri) + ' duration:' + str(end -start) 
					
					# save data for eahc trial
					run.save_data(sim.data)

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
		weights = [.0002]
		self.kwargs = {
		'exp' : 'exp_1', 
		'tree' : 'apical_trunk',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : .05
		}

	def exp_2(self):
		""" choose a specific set of synapses, iterate over increasing synaptic weights, measure resulting LTP and dendritic spike initiation
		"""
		weights = np.arange(.005, .03, .005)
		# weights = np.arange(.5, 1, .1)
		weights = [.015]
		self.kwargs = {
		'exp' : 'exp_2', 
		'tree' : 'apical_trunk',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.0002],
		'w_rand' : False, 
		'sec_idx' : [-1], 
		'seg_idx' : [[-1]]
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
		
		weights = .01
		self.kwargs = {
		'conductance_range' : np.arange(1, 1.5, 1),
		'experiment' : 'exp_6', 
		'tree' : 'apical_prox',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [-1],
		'sec_list' : [0],
		'pulses':3,
		}


if __name__ =="__main__":
	kwargs = Arguments('exp_6').kwargs
	x = Experiment(**kwargs)
	analysis.Experiment(experiment='exp_6')
	plots = analysis.Voltage()
	plots.plot_all(x.p)
	# analysis.Experiment(exp='exp_3')
