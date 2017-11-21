

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
import copy

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
		plots = analysis.PlotRangeVar()
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

				plots.plot_trace(data=sim.data, 
					tree=p['tree'], 
					sec_idx=p['sec_idx'], 
					seg_idx=p['seg_idx'],
					variables=p['plot_variables'],
					x_var='t')

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
		exp = 'exp_7'
		tree = kwargs['tree']
		trials = kwargs['trials']
		w_mean = kwargs['w_mean']
		w_std = kwargs['w_std']
		w_rand = kwargs['w_rand']
		syn_frac = kwargs['syn_frac']
		plots = analysis.PlotRangeVar()

		# loop over trials
		for tri in range(trials):
			for conductance_i, conductance  in enumerate(kwargs['conductance_range']):
				for vhalfl_h_i, vhalfl_h in enumerate(kwargs['activation_range_h']):
					for slope_i, slope in enumerate(kwargs['slope_range']):
						if conductance==0. and (vhalfl_h_i >0 or slope_i> 0):
							continue

						# load rest of parameters from parameter module
						p = param.Experiment(**kwargs).p
						
						# set Ih and Ka conductance parameters
						p['KMULT'] =  kwargs['KMULT']
						p['KMULTP'] =  kwargs['KMULTP']
						p['ghd_grad'] = kwargs['ghd_grad']
						p['ka_grad'] = kwargs['ka_grad']
						p['vhalfl_hd_prox'] = vhalfl_h+kwargs['vhalfl_hd_prox']
						p['vhalfl_hd_dist'] = vhalfl_h+kwargs['vhalfl_hd_dist']
						p['ghd'] = conductance*kwargs['ghd']
						p['kl_hd'] = kwargs['kl_hd']-slope

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
		weights = [.003]
		self.kwargs = {
		'experiment' : 'exp_1', 
		'tree' : 'apical',
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
		
		weights = 0.005
		self.kwargs = {
		'activation_range_h': np.arange(0., 25., 4.),
		'conductance_range' : np.arange(0., 3., .5),
		'slope_range' : np.arange(0.,4., 1.),
		'gna' : 0.,
		'dgna':0,
		'Vrest':-65.,
		'gcalbar': 0.,
		'kl_hd' : -4.,
		'ghd' : 0.00005,
		'KMULT' :  .5*0.03,
		'KMULTP' :  .5*0.03,
		'ghd_grad' : 5,
		'ka_grad' : 1,
		'vhalfl_hd_prox' : -95.,#-73.,			
		'vhalfl_hd_dist' : -95.,
		'experiment' : 'exp_8', 
		'tree' : 'apical_prox',
		'trials' : 1,
		'w_mean' : weights,#[.001],
		'w_std' : [.002],
		'w_rand' : False, 
		'syn_frac' : 0,
		'seg_list' : [5,10,20],
		'sec_list' : [0,0,0],
		'pulses':4,
		'tstop':70,
		'field_on':15,
		'field_off':80,
		}


if __name__ =="__main__":
	kwargs = Arguments('exp_8').kwargs
	x = Experiment(**kwargs)
	analysis.Experiment(**kwargs)
	# plots = analysis.Voltage()
	# plots.plot_all(x.p)
	# analysis.Experiment(exp='exp_3')
