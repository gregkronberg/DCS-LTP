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
			
			# equivalent cylinder parameters determined by cell.DendriteTransform() of Migliore cell geo5038804.hoc
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
			'fixnseg':False,		# determine number of segments in cylinder according to d_lambda rule

			# FIXME, must be set so that variable names are unique
			# set recording variables
				# organized a dictionary of dictionaries [attribute name: [variable type: mechanism]
				# note that if the attribute is part of a synapse object, it will accessed differently than a range variable
					# range variables can be simply accessed by dot notation directly from a given neuron section
					# synapse attributes need to be accesed from the synapse object stored in cell.syns
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

			# choose y variables to plot [varaibles]
			'plot_variables' : ['v','i','ik_kad','i_hd', 'ica_calH', 'ina_na3', 'gbar'],
			# FIXME, should be a list, where you can choose arbitrary combinations of variables 
			# x variables to plot 
			'x_variables':['t'],
			'group_trees':False,

			# synapse activation
			'syn_frac':[],		# fraction of synapses to activate with choose_seg_rand()
			'trial':0,			# count the current trial number
			'trial_id':0,		# a unique identifier for each trial using uuid64
			'w_rand':[],		# choose synapse weights from a random distribution (Bool)
			'w_std' : [],		# standard deviation of weights distribution, if w_rand is True
			'w_mean': [], # mean synaptic weight (microsiemens or micro-ohms)
			'trees': [],		# list of subtrees with active synapses [trees]
			'w_list':[],		# nested list of weights, determined by set_weights().  Weights correspond to segments indexed in seg_idx.  Organized as [tree][section][segment]
			'sec_list':[],		# list of active sections with repeats, each entry corresponds to the section for a given segment in seg_list.  [tree][section number]
			'seg_list':[],		# list of active segments, corresponding to sections in sec_list {tree}[segment number]
			'sec_idx': [],		# list of active sections, without repeats. Indeces in the list correspond to indeces in seg_idx {tree}[section number]
			'seg_idx':[],		# nested list of active segments {tree}[section index][segment number]
			'seg_dist' : {},	# distance of each segment from soma {tree}[section index][segment number]

			# extracellular field stimualation
			'field_angle': 0,	# angle relative to principle cell axis in radians 
			'field':[-20,0,20],	# list of stimulation intensities in V/m, negative = cathodal, postivie = anodal
			'field_color':['b','k','r'],	# plot colors correesponding to entries in field
			'field_on':20,		# stimulation onset time in (ms)
			'field_off': 70,	# stimulation offset time in (ms)
			'dt' : .025,		# integration timestep (ms)
			'warmup': 30,		# simulation warmup time (ms)
			'tstop' : 70,		# simulation duration (ms)

			# bipolar stimulation parameters
			'bursts':1,			# bipolar stimulus bursts
			'pulses':4,			# pulses per bursts 
			'pulse_freq':100,	# pulse frequency within burst (Hz)
			'burst_freq':5,		# burst frequency (Hz)
			'noise' : 0,		# noise in input arrival (see NetCon documentation)

			# clopath synapse parameters
			'clopath_delay_steps': 1,
			'clopath_tau_0':6, # time constant (ms) for low passed membrane potential for depression
			'clopath_tau_r' : 30, # time constant (ms) for low pass filter presynaptic variable
			'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
			'clopath_A_m':.0001, # depression magnitude parameter (mV^-1)
			'clopath_A_p': .0005, # amplitude for potentiation (mV^-2)
			'clopath_tetam':-60,#-41, # depression threshold (mV)
			'clopath_tetap':-60,#-38, # potentiation threshold (mV)

			# ampa synapse parameters
			'tau1_ampa' : 0.2,	# rise time constant (ms)
			'tau2_ampa' : 2,	# decay time constant	(ms)
			'i_ampa' : 0.18,	# default peak ampa current in uS

			# nmda synapse parameters
			'tau1_nmda' : 1,	# rise time constant (ms)
			'tau2_nmda' : 50,	# decay time constant (ms)

			
			# Parameters from Migliore 2005 (signal propogation in oblique dendrites)
			# conductances reported as (nS/um2) in paper, but need to be in (mho/cm2)
			# conversion 10,000*(pS/um2) = 10*(nS/um2) = (mho/cm2) = .001*(mS/cm2)
			# *** units in paper are a typo, values are already reported in (mho/cm2) ***
			'Vrest' : -65.,				# resting potential (mV)
			'gna' :  1.*0.025,#.025,				# peak sodium conductance (mho/cm2)
			'dgna' : -.000025,			# change in sodium conductance with distance (ohm/cm2/um) from Kim 2015
			'ena' : 55.,					# sodium reversal potential (mV)
			'AXONM' : 5.,				# multiplicative factor for axonal conductance
			'gkdr' : 1.*0.01,#0.01,				# delayed rectifier potassium peak conductance (mho/cm2)
			'ek' : -90.,					# potassium reversal potential
			'celsius' : 35.0,  				# temperature (degrees C)
			'KMULT' :  1.*0.03,#0.03,			# multiplicative factor for distal A-type potassium conductances
			'KMULTP' : 1.*.03,#0.03,				# multiplicative factor for proximal A-type potassium conductances
			'ghd' : 0.75*0.0001,#0.0001,			# peak h-current conductance (mho/cm2)
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

	def set_branch_sequence_ordered(self, seg_idx, delay, direction):
		"""
		"""
		delays = {}
		# iterate over trees
		for tree_key, tree in seg_idx.iteritems():
			delays[tree_key] = []
			# iterate over sections
			for sec_i, sec in enumerate(tree):
				
				delays[tree_key].append([])
				
				n_seg = len(sec)

				# reverse the order of segments and iterate
				for seg_i, seg in enumerate(sec):

					# if sequence is towards soma
					if direction is 'in':

						# calculate delay
						add_delay = float(n_seg-1-seg_i)*delay

					# otherwise if sequence is away from soma
					elif direction is 'out':

						# calculate delay
						add_delay = float(seg_i)*delay
					
					# store delay
					delays[tree_key][sec_i].append(add_delay)

		# store in parameter dictionary
		self.p['sequence_delays'] = delays

	# choose dendritic branch to activate synapses
	def choose_branch_rand(self, trees, geo, num_sec=1, distance=0, branch=True):
		""" choose random dendritic branch to activate synapses on and store in sec_idx

		arguments:
		"""
		# structure to store section list
		sec_idx = {}

		# iterate over all trees
		for tree_key, tree in geo.iteritems():

			# add list for chosen sections
			sec_idx[tree_key] =[]

			# if tree is in active list
			if tree_key in trees:

				# list all sections
				secs_all = [sec_i for sec_i, sec in enumerate(tree)] 
				
				# choose random index
				secs_choose = np.random.choice(len(secs_all), num_sec, replace=False)

				# list of randomly chosen sections
				sec_list = [secs_all[choose] for choose in secs_choose]

				# iterate over chosen sections
				for sec_i, sec in enumerate(sec_list):

					if branch:
						# get distance from soma of section start
						dist1 = h.distance(0, sec=tree[sec])

						# distance from soma of section end
						dist2 = h.distance(1, sec=tree[sec])

						# until chosen section is a branch terminal (no children sections) that hasn't already been chosen
						sref = h.SectionRef(sec=tree[sec])
						while (sref.nchild() is not 0) and (sec in sec_idx[tree_key]) and (dist2 > distance) :

							# replace with a new random branch
							sec = secs_all[np.random.choice(len(secs_all), 1, replace=False)]

							# get distance from soma of section start
							dist1 = h.distance(0, sec=tree[sec])

							# distance from soma of section end
							dist2 = h.distance(1, sec=tree[sec])

					# once current section is a branch terminal, store the section number in sec_idx
					sec_idx[tree_key].append(sec)

		# store section list in parameter dictionary
		self.p['sec_idx'] = sec_idx

	def choose_seg_branch(self, geo, sec_idx, seg_dist, max_seg=[], spacing=0, distance=[]):
		""" choose active segments on branch based on spacing between segments

		Arguments:

		distance is a list with 1 or 2 entries.  If 1 entry is given, this is considered a minimum distance requirement.  If two entries are given, they are treated as min and max requirements respectively
		"""
		# store segment list
		seg_idx = {}
		
		# iterate over trees
		for tree_key, tree in sec_idx.iteritems():
			
			# add dimension for sections
			seg_idx[tree_key]=[]
			
			# iterate over sections
			for sec_i, sec in enumerate(tree):
				
				# add dimension for segments
				seg_idx[tree_key].append([])
				
				# iterate over segments
				for seg_i, seg in enumerate(geo[tree_key][sec]):
					
					# check distance requirement
					dist_check=False

					# if there is a distance requirement
					if distance:
						
						# if there is a max distance requirement
						if len(distance)>1:
							
							# if segment meets maxdistance requirement
							if seg_dist[tree_key][sec][seg_i]>distance[0] and seg_dist[tree_key][sec][seg_i]<distance[1]:

								#update distance requireemnt
								dist_check=True

						# if there is only a minimum requirement
						elif len(distance)==1:

							# if segment meet min distance requirement
							if seg_dist[tree_key][sec][seg_i]>distance[0]:

								# update distance requirement
								dist_check=True

					else:
						dist_check=True

					# if segment has not been stored yet and min distance is met
					if not seg_idx[tree_key][sec_i] and dist_check:

						# add segment to list
						seg_idx[tree_key][sec_i].append(seg_i)

					# otherwise check spacing requirement between current segment and previously added segment
					elif seg_idx[tree_key][sec_i] and max_seg and len(seg_idx[tree_key][sec_i])< max_seg :
						
						# retrieve segment object of previously stored segment
						previous_seg= [segment for segment_i, segment in enumerate(geo[tree_key][sec]) if segment_i == seg_idx[tree_key][sec_i][-1]][0]

						# previous segment location in um
						previous_seg_loc = previous_seg.x*geo[tree_key][sec].L
						# current segment location in um
						current_seg_loc = seg.x*geo[tree_key][sec].L
						
						# distance between current and previous segment
						seg_space = current_seg_loc - previous_seg_loc

						# if spacing requirement is met
						if abs(seg_space) > spacing:

							# add segment to list
							seg_idx[tree_key][sec_i].append(seg_i)

		# update parameter dictionary
		self.p['seg_idx'] = seg_idx

	def choose_seg_rand(self, trees, syns, syn_frac):
		""" choose random segments to activate
		arguments:
		
		trees = list of subtrees with active synapses

		syn_list = list of all synapses to be chosen from ogranized as {trees}[section number][segment number][synapse type]

		syn_frac = fraction of synapses to be chosen, with equal probability for all synapses

		updates the parameter dictionary according to the chosen synapses
		"""
		self.p['sec_list']={}
		self.p['seg_list']={}
		self.p['sec_idx']={}
		self.p['seg_idx']={}

		# iterate over all trees
		for tree_key, tree in syns.iteritems():

			# if tree is in active list
			if tree_key in trees:

				# list all segments as [[section,segment]] 
				segs_all = [[sec_i,seg_i] for sec_i,sec in enumerate(tree) for seg_i,seg in enumerate(tree[sec_i])]

				# choose segments to activate
				segs_choose = np.random.choice(len(segs_all), int(syn_frac*len(segs_all)), replace=False)

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
				self.p['sec_list'][tree_key] = sec_list
				self.p['seg_list'][tree_key] = seg_list
				self.p['sec_idx'][tree_key] = sec_idx
				self.p['seg_idx'][tree_key] = seg_idx
				self.p['syn_frac'] = syn_frac

	def choose_seg_manual(self, trees, sec_list, seg_list):
		""" manually choose segments to activate

		arguments:
		trees = list of subtrees to be activated

		sec_list and seg_list should be organized as {trees}[list of sections/segments].  Indices in each list match, so that sec_list[i] and seg_list[i] output the section and segment number for an active segment
		"""
		sec_idx = {}
		seg_idx = {}
		for tree_key, tree in sec_list.iteritems():

			# uniqure list of active sections
			sec_idx[tree_key]  = list(set(tree))
			
			# list of active segments as [unique section][segments]
			seg_idx[tree_key] = []
			for sec in sec_idx:
				seg_idx[tree_key].append([seg_list[tree_key][sec_i] for sec_i,sec_num in enumerate(tree) if sec_num==sec])

		# update parameter dictionary
		self.p['sec_list'] = sec_list
		self.p['seg_list'] = seg_list
		self.p['sec_idx'] = sec_idx
		self.p['seg_idx'] = seg_idx

	def set_weights(self, seg_idx, w_mean, w_std, w_rand):
		"""
		sets weights using nested list with same structure as seg_idx: [tree][section index][segment index]

		arguments: 
		seg_idx = nested list of segment numbers to be activated {trees}[section index][segment number]

		w_mean = mean synaptic weight

		w_std = standard deviation of weights, if w_rand is True

		w_rand = choose synaptic weights from normal distibution? (bool)
		"""
		w_list = {}

		# iterate over trees in seg_idx
		for tree_key, tree in seg_idx.iteritems():

			# add dimension for sections
			w_list[tree_key]=[]
			
			# loop over sections
			for sec_i,sec in enumerate(tree):
				
				# add sections dimension for segments
				w_list[tree_key].append([])

				# loop over segments
				for seg_i,seg in enumerate(sec):

					# if weights are randomized
					if w_rand:
						# choose from normal distribution
						w_list[tree_key][sec_i].append(np.random.normal(w_mean,w_std))
					
					# otherwise set all weights to the same
					else:
						w_list[tree_key][sec_i].append(w_mean)

		# update parameter dictionary
		self.p['w_list']=w_list

	def seg_distance(self, cell):
		""" calculate distance from soma of each segment and store in parameter dictionary

		p['seg_dist'] organized as {trees}[section idx][segment number]
		"""

		self.p['seg_dist']={}
		
		# iterate over trees
		for tree_key,tree in cell.geo.iteritems():

			# add dimension for sections
			self.p['seg_dist'][tree_key]=[]
			
			# iterate over sections
			for sec_i,sec in enumerate(tree):
				
				# add dimension for segments
				self.p['seg_dist'][tree_key].append([])
				
				# iterate over segments
				for seg_i,seg in enumerate(sec):
					
					# calculate and store distance from soma and store 
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