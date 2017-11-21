# parameters 


from neuron import h
import numpy as np
import cell 
import stims

class Default(object):
	""" base class for experimental parameters
	"""
	def __init__(self):
		self.default_parameters()

	def default_parameters(self):
		exp='default'
		self.p = {
			'experiment' : exp,
			'cell' : [], 
			'data_folder' : 'Data/'+exp+'/',
			'fig_folder' : 'png figures/'+exp+'/',
			# equivalent cylinder parameters determined by cell.DendriteTransform of Migliore cell geo5038804.hoc
			'L_basal' : 1600.,
			'L_soma' : 7.5,
			'L_apical_prox' : 1000.,
			'L_apical_dist' : 1000.,
			'diam1_basal' : 1.9,
			'diam1_soma' : 7.5,
			'diam1_apical_prox' : 2.75,
			'diam1_apical_dist' : 2.75,
			'diam2_basal' : 1.9,
			'diam2_soma' : 7.5,
			'diam2_apical_prox' : 2.75,
			'diam2_apical_dist' : 2.75,
			'nsec_basal' : 1,
			'nsec_soma' : 1,
			'nsec_apical_prox' : 1,
			'nsec_apical_dist' : 1,
			'syn_types' : ['ampa', 'nmda', 'clopath'],
			'fixnseg':True,

			'record_variables' : 
			{
			'v' : {'range':'v'},
			'gbar' : {'syn':'clopath'},
			'i': {'syn':'nmda'},
			'i_hd' : {'range' : 'hd'},
			'ik_kad' : {'range': 'kad'},
			'ik_kap' : {'range': 'kap'},
			'ica_calH' : {'range':'calH'},
			'ina_na3' : {'range':'na3'},
			}, 
			'plot_variables' : ['v','i','ik_kad','i_hd', 'ica_calH', 'ina_na3'],


			'syn_frac':[],
			'trial':0,
			'trial_id':0,
			'w_rand':[],
			'w_std' : [],
			'w_mean': [], # mean synaptic weight (microsiemens or micro-ohms)
			'tree': [],
			'w_list':[],
			'sec_list':[],
			'seg_list':[],
			'sec_idx': [],
			'seg_idx':[],
			'seg_dist' : {},
			'field_angle': 0,#np.pi/2.0,
			'field':[-30,0,30],
			'field_color':['b','k','r'],
			'field_on':20,
			'field_off': 70,
			'dt' : .025,
			'warmup': 30,
			'tstop' : 70,#5*1000/5 + 30 + 5*1000/100 +30
			'bursts':1,
			'pulses':4,
			'pulse_freq':100,
			'burst_freq':5,
			'noise' : 0,

			# clopath synapse parameters
			'clopath_delay_steps': 1,
			'clopath_tau_0':6, # time constant (ms) for low passed membrane potential for depression
			'clopath_tau_r' : 30, # time constant (ms) for low pass filter presynaptic variable
			'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
			'clopath_A_m':.0001, # depression magnitude parameter (mV^-1)
			'clopath_A_p': .0005, # amplitude for potentiation (mV^-2)
			'clopath_tetam':-41, # depression threshold (mV)
			'clopath_tetap':-38, # potentiation threshold (mV)

			# ampa synapse parameters
			'tau1_ampa' : 0.2,
			'tau2_ampa' : 2,
			'i_ampa' : 0.18,

			# nmda synapse parameters
			'tau1_nmda' : 1,
			'tau2_nmda' : 50,

			# Parameters from Migliore 2005 (signal propogation in oblique dendrites)
			
			# conductances reported as (nS/um2) in paper, but need to be in (mho/cm2)
			# conversion 10,000*(pS/um2) = 10*(nS/um2) = (mho/cm2) = .001*(mS/cm2)
			# *** units in paper are a typo, values are already reported in (mho/cm2) ***
			'Vrest' : -65.,				# resting potential (mV)
			'gna' :  1.*0.025,#.025,				# peak sodium conductance (mho/cm2)
			'dgna' : -.000025,			# change in sodium conductance with distance (ohm/cm2/um) from Kim 2015
			'ena' : 55.,					# sodium reversal potential (mV)
			'AXONM' : 100.,				# multiplicative factor for axonal conductance
			'gkdr' : 1.*0.01,#0.01,				# delayed rectifier potassium peak conductance (mho/cm2)
			'ek' : -90.,					# potassium reversal potential
			'celsius' : 35.0,  				# temperature (degrees C)
			'KMULT' :  1.*0.03,#0.03,			# multiplicative factor for distal A-type potassium conductances
			'KMULTP' : 1.*.03,#0.03,				# multiplicative factor for proximal A-type potassium conductances
			'ghd' : 1.*0.75*0.0001,#0.0001,			# peak h-current conductance (mho/cm2)
			'gcalbar': 1.*.00125 ,			# L-type calcium conductance from Kim et al. 2015 (mho/cm2)
			'ehd' : -30.,					# h-current reversal potential (mV)
			'kl_hd' : -5.,#-8.,
			'vhalfl_hd_prox' : -83.,#-73,			# activation threshold for proximal h current (mV)
			'vhalfl_hd_dist' : -83.,#-81,			# activation threshold for distal h-current (mV)
			'vhalfl_kad' : -56.,#-56.,			# inactivation threshold for distal a-type current (mV)
			'vhalfl_kap' : -56.,#-56.,			# inactivation threshold for proximal a-type current (mV)
			'vhalfn_kad' : -1.,#-1.,			# activation threshold for distal a-type urrent (mV)
			'vhalfn_kap' : -1.,#-1.,			# activation threshold for proximal a-type current (mV)
			'RaAll' : 150.,				# axial resistance, all compartments (ohm*cm)
			'RaAx' : 50.,					# axial resistance, axon (ohm*cm)					
			'RmAll' : 28000.,			# specific membrane resistance (ohm/cm2)
			'Cm' : 1.,					# specific membrane capacitance (uf/cm2)
			'ka_grad' : 1.,#1.,#1.,				# slope of a-type potassium channel gradient with distance from soma 
			'ghd_grad' : 5.,#1.,#3.,				# slope of h channel gradient with distance from soma 
			}

	def choose_seg_rand(self, syn_list, syn_frac):
		""" choose random segments to activate

		arguments:
		syn_list = list of all synapses to be chosen from ogranized as [section number][segment number]

		syn_frac = fraction of synapses to be chosen, with equal probability for all synapses

		updates the parameter dictionary according to the chosen synapses
		"""

		# list all segments as [[section,segment]] 
		segs_all = [[sec_i,seg_i] for sec_i,sec in enumerate(syn_list) for seg_i,seg in enumerate(syn_list[sec_i])]

		# choose segments to activate
		segs_choose = np.random.choice(len(segs_all),int(syn_frac*len(segs_all)),replace=False)

		# list of active sections (contains duplicates)
		sec_list = [segs_all[a][0] for a in segs_choose]
		
		# list of active segments
		seg_list = [segs_all[a][1] for a in segs_choose]

		# uniqure list of active sections
		sec_idx  = list(set(sec_list))
		
		# list of active segments as [unique section][segments]
		seg_idx = []
		for sec in sec_idx:
			seg_idx.append([seg_list[sec_i] for sec_i,sec_num in enumerate(sec_list) if sec_num==sec])

		# update parameter dictionary
		self.p['sec_list'] = sec_list
		self.p['seg_list'] = seg_list
		self.p['sec_idx'] = sec_idx
		self.p['seg_idx'] = seg_idx
		self.p['syn_frac'] = syn_frac

	def choose_seg_manual(self, sec_list, seg_list):
		""" manually choose segments to activate
		"""
		# uniqure list of active sections
		sec_idx  = list(set(sec_list))
		
		# list of active segments as [unique section][segments]
		seg_idx = []
		for sec in sec_idx:
			seg_idx.append([seg_list[sec_i] for sec_i,sec_num in enumerate(sec_list) if sec_num==sec])

		# update parameter dictionary
		self.p['sec_list'] = sec_list
		self.p['seg_list'] = seg_list
		self.p['sec_idx'] = sec_idx
		self.p['seg_idx'] = seg_idx

	def set_weights(self, seg_idx, w_mean, w_std, w_rand):
		w_list = []
		# loop over sections
		for sec_i,sec in enumerate(seg_idx):
			w_list.append([])
			# loop over segments
			for seg_i,seg in enumerate(seg_idx[sec_i]):

				# if weights are randomized
				if w_rand:
					# choose from normal distribution
					w_list[sec_i].append(np.random.normal(w_mean,w_std))
				# otherwise set all weights to the same
				else:
					w_list[sec_i].append(w_mean)

		# update parameter dictionary
		self.p['w_list']=w_list

	def seg_distance(self, cell):
		""" calculate distance from soma of each segment and store in parameters
		"""
		# iterate over trees
		for tree_key,tree in cell.geo.iteritems():
			self.p['seg_dist'][tree_key]=[]
			# iterate over sections
			for sec_i,sec in enumerate(tree):
				self.p['seg_dist'][tree_key].append([])
				# iterate over segments
				for seg_i,seg in enumerate(sec):
					# calculate and store distance from soma
					distance =  h.distance(seg.x, sec=sec)
					self.p['seg_dist'][tree_key][sec_i].append(distance)

class Experiment(Default):
	"""
	"""
	def __init__(self, **kwargs):
		# initialize with default parameters
		super(Experiment, self).__init__()

		# retrieve experiment to run
		experiment = getattr(self, kwargs['experiment'])

		# set specific experimental parameters
		experiment(**kwargs) 

	def exp_1(self, **kwargs):
		""" randomly activate subset of synapses

		set parameters in dictionary p
		
		p cannot contain any hoc objects, as this will be pickled and stored with each experiment so that the parameters can be retrieved
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.cell = cell.CellMigliore2005(self.p)
		self.seg_distance(self.cell)
		# randomly choose active segments 
		self.choose_seg_rand(syn_list=self.cell.syns[tree]['ampa'], syn_frac=self.p['syn_frac'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

	def exp_2(self, **kwargs):
		""" choose specific section/segment and activate with varying weights
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		self.cell = cell.CellMigliore2005(self.p)
		self.seg_distance(self.cell)
		sec_idx = kwargs['sec_idx']
		seg_idx = kwargs['seg_idx']
		sec_list = [sec_idx[sec_i] for sec_i,sec in enumerate(seg_idx) for seg in sec]
		seg_list = [ seg for sec_i,sec in enumerate(seg_idx) for seg in sec]

		# update parameter dictionary
		self.p['sec_list'] = sec_list
		self.p['seg_list'] = seg_list
		self.p['sec_idx'] = sec_idx
		self.p['seg_idx'] = seg_idx

		# create weight list
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

	def exp_3(self, **kwargs):
		""" randomly activate subset of synapses for reduced 4 compartment model
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.cell = cell.PyramidalCell(self.p)
		self.seg_distance(self.cell)
		# randomly choose active segments 

		self.choose_seg_rand(syn_list=self.cell.syns[self.p['tree']]['ampa'], syn_frac=self.p['syn_frac'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		print self.p['seg_idx']

	def exp_4(self, **kwargs):
		""" randomly activate subset of synapses for reduced 4 compartment model
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.cell = cell.PyramidalCell(self.p)
		self.seg_distance(self.cell)
		# randomly choose active segments 

		self.choose_seg_rand(syn_list=self.cell.syns[self.p['tree']]['ampa'], syn_frac=self.p['syn_frac'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		print self.p['seg_idx']

	def exp_5(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure steady state membrane polarization in the absence of synaptic inputs
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.cell = cell.PyramidalCell(self.p)
		self.seg_distance(self.cell)
		# randomly choose active segments 

		self.choose_seg_rand(syn_list=self.cell.syns[self.p['tree']]['ampa'], syn_frac=self.p['syn_frac'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		print self.p['seg_idx']

	def exp_6(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure peak membrane depolarization in response to synaptic input
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.p['cell'] = cell.PyramidalCell(self.p)
		self.seg_distance(self.p['cell'])
		# randomly choose active segments 

		self.choose_seg_manual(sec_list=self.p['sec_list'], seg_list=self.p['seg_list'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		print self.p['seg_idx']
		print self.p['w_list']
		print self.p['pulses']

		# delete created cell
		# self.cell=None

	def exp_7(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure peak membrane depolarization in response to synaptic input
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.p['cell'] = cell.CellMigliore2005(self.p)
		# self.p['cell'] = cell.PyramidalCell(self.p)
		self.seg_distance(self.p['cell'])
		# randomly choose active segments 

		self.choose_seg_manual(sec_list=self.p['sec_list'], seg_list=self.p['seg_list'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		# delete created cell
		# self.cell=None

	def exp_8(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure peak membrane depolarization in response to synaptic input
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.p['cell'] = cell.PyramidalCylinder(self.p)
		# self.p['cell'] = cell.PyramidalCell(self.p)
		self.seg_distance(self.p['cell'])
		# randomly choose active segments 

		self.choose_seg_manual(sec_list=self.p['sec_list'], seg_list=self.p['seg_list'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		# delete created cell
		# self.cell=None

	def exp_9(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure peak membrane depolarization in response to synaptic input
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.p['cell'] = cell.CellMigliore2005(self.p)
		# self.p['cell'] = cell.PyramidalCell(self.p)
		self.seg_distance(self.p['cell'])
		# randomly choose active segments 

		self.choose_seg_manual(sec_list=self.p['sec_list'], seg_list=self.p['seg_list'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		# delete created cell
		# self.cell=None

	def exp_10(self, **kwargs):
		""" vary gradient of Ka and Ih

		measure peak membrane depolarization in response to synaptic input
	
		"""
		# update parameters
		for key, val in kwargs.iteritems():
			self.p[key] = val

		self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
		self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

		# load cell
		self.p['cell'] = cell.CellMigliore2005(self.p)
		# self.p['cell'] = cell.PyramidalCell(self.p)
		self.seg_distance(self.p['cell'])
		# randomly choose active segments 

		self.choose_seg_manual(sec_list=self.p['sec_list'], seg_list=self.p['seg_list'])
		
		# set weights for active segments
		self.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

		# delete created cell
		# self.cell=None

# set procedure if called as a script
if __name__ == "__main__":
	pass