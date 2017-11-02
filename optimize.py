"""
parameter optimization
"""
import param
import run
import analysis
import numpy as np
import scipy.optimize
import uuid
import shutil
import os
import pickle

class Optimize:
	""" parameter optimization
	"""
	def __init__(self):
		"""
		"""
		self.scale_syn_frac = 0.1
		self.scale_weights = .001
		self.scale_A_p = .001
		self.opt_evolution()

	def opt_evolution(self):
		""" differential evolution
		"""
		# initial parameter guess
		self.x0 = [1.02488,1.03302,.970386]
		# upper parameter bound 
		self.upper = [2,2,2]
		# lower bound
		self.lower = [.5,.5,.5]
		# bounds together
		self.bounds = zip(self.lower,self.upper)
		# number of trials to average over
		self.trials = 10
		# folder to store data
		self.exp = 'exp_opt'
		# experimental data to fit
		self.target=np.array([1.46,1.25,1.22])
		# list to store parameters after each iteration
		self.x_history = []
		# maximum number of iterations
		self.maxiter = 100
		# differential evolution parameters
		self.strategy='best1bin'
		self.popsize = 3
		self.tol = .01
		self.recombination=0.7
		self.mutation = (0.5,1)
		# run optimization
		self.opt = scipy.optimize.differential_evolution(
			self.error_func,
			self.bounds,
			args=(self.trials, self.exp, self.target), 
			strategy = self.strategy,
			maxiter = self.maxiter,
			popsize = self.popsize,
			tol = self.tol,
			mutation  =self.mutation,
			recombination = self.recombination,
			callback=self.store_params_evo,
			disp=True,
			)
		# save optimization result
		self.save_params_evo()

	def opt_fmin(self):
		# initial parameter guess
		self.x0 = [1.06727,1.04256,0.97902]
		# upper parameter bound 
		self.upper = [4,10,5]
		# lower bound
		self.lower = [.8,.1,.01]
		# bounds together
		self.bounds = zip(self.lower,self.upper)
		# number of trials to average over
		self.trials = 7
		# folder to store data
		self.exp = 'exp_opt'
		# experimental data to fit
		self.target=np.array([1.46,1.25,1.22])
		# list to store parameters after each iteration
		self.x_history = []
		# tolerance for parameter changes
		self.xtol=0.01
		# tolerance for error function
		self.ftol=0.01
		# maximum number of iterations
		self.maxiter = 100
		# run optimization
		self.opt = scipy.optimize.fmin(self.error_func,self.x0,
			args=(self.trials, self.exp, self.target), 
			xtol=self.xtol, 
			ftol = self.ftol, 
			maxiter=self.maxiter, 
			callback=self.store_params,
			retall=True)
		# save optimization result
		self.save_params()

	def error_func(self,x,*args):
		""" error function 
		"""
		# scaled parameters 
		syn_frac = self.scale_syn_frac*x[0]
		weights = self.scale_weights*x[1]
		A_p = self.scale_A_p*x[2]
		
		# number of trials
		trials = args[0]
		# name of experiment for storing data
		exp = args[1]
		# experimental data to be fit
		target = args[2]

		# load all simulation parameters
		p = param.exp_3(syn_frac=syn_frac, w_mean=weights, w_std=.2*weights, w_rand=True, exp=exp).p

		# clear existing data from directory
		# GOOGLE DRIVE MUST BE DISABLED, it interferes with file i/o
		directory = p['data_folder']
		# check if directory exists
		if os.path.isdir(directory) is True:
			# loop over files in directory
			for file in os.listdir(directory):
				# only delete raw data files
				if 'data' in file:
					try:
						# remove data
						os.remove(os.path.join(directory,file))
					# if failed, report it back to the user 
					except OSError, e:  
		        			print ("Error: %s - %s." % (e.filename,e.strerror))
		# loop over trials
		for trial in range(trials):
			# update parameters
			# clopath potentiation amplitude
			p['clopath_A_p'] = A_p
			#trial number
			p['trial'] = trial
			# unique identifier for each trial
			p['trial_id'] = str(uuid.uuid4())
			
			# run simulation
			sim = run.Run(p)
			
			# save data 
			run.save_data(sim.data)
			
			# print trial number
			print 'trial'+ str(trial) 

		# get weight changes
		W = analysis.Weights(p)
		# average overage overall all synapses, all trials
		dw_mean = np.divide(np.mean(W.w_end_all,axis=1),np.mean(W.w_start_all,axis=1))
		
		# sum of square error, add terms to bias towards asymmetry
		error = sum(np.square(dw_mean - target)) - (dw_mean[0]-dw_mean[1])**2 + (dw_mean[2]-dw_mean[1])**2	

		print x
		print dw_mean
		return error

	def store_params(self, xk):
		self.x_history.append(xk)
		directory = 'Data/'+self.exp+'/'
		with open(directory+'parameter_history'+'.pkl', 'wb') as output:
			pickle.dump(self.x_history, output,protocol=pickle.HIGHEST_PROTOCOL)

	def store_params_evo(self, xk, convergence):
		self.x_history.append([xk,convergence])
		directory = 'Data/'+self.exp+'/'
		with open(directory+'evo_parameter_history'+'.pkl', 'wb') as output:
			pickle.dump(self.x_history, output,protocol=pickle.HIGHEST_PROTOCOL)

	def save_params(self):
		directory = 'Data/'+self.exp+'/'
		with open(directory+'parameter_optimum'+'.pkl', 'wb') as output:
			pickle.dump(self.opt, output,protocol=pickle.HIGHEST_PROTOCOL)

	def read_params(self):
		directory = 'Data/'+self.exp+'/'
		with open(directory+'parameter_optimum'+'.pkl', 'rb') as output:
			data = pickle.load(pkl_file)

		return data

if __name__=='__main__':
	Optimize()

