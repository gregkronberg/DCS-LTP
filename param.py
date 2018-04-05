# parameters 


from neuron import h
import numpy as np
import cell 
import stims
import copy

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
            'fixnseg':False,        # determine number of segments in cylinder according to d_lambda rule

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
            'syn_frac':[],      # fraction of synapses to activate with choose_seg_rand()
            'trial':0,          # count the current trial number
            'trial_id':0,       # a unique identifier for each trial using uuid64
            'w_rand':[],        # choose synapse weights from a random distribution (Bool)
            'w_std' : [],       # standard deviation of weights distribution, if w_rand is True
            'w_mean': [], 		# mean synaptic weight (microsiemens or micro-ohms)
            'trees': [],        # list of subtrees with active synapses [trees]
            'w_list':[],        # nested list of weights, determined by set_weights().  Weights correspond to segments indexed in seg_idx.  Organized as [tree][section][segment]
            'sec_list':[],      # list of active sections with repeats, each entry corresponds to the section for a given segment in seg_list.  [tree][section number]
            'seg_list':[],      # list of active segments, corresponding to sections in sec_list {tree}[segment number]
            'sec_idx': [],      # list of active sections, without repeats. Indeces in the list correspond to indeces in seg_idx {tree}[section number]
            'seg_idx':[],       # nested list of active segments {tree}[section index][segment number]
            'seg_dist' : {},    # distance of each segment from soma {tree}[section index][segment number]

            # extracellular field stimualation
            'field_angle': 0,   # angle relative to principle cell axis in radians 
            'field':[-20,0,20], # list of stimulation intensities in V/m, negative = cathodal, postivie = anodal
            'field_color':['b','k','r'],    # plot colors correesponding to entries in field
            'field_on':20,      # stimulation onset time in (ms)
            'field_off': 70,    # stimulation offset time in (ms)
            'dt' : .025,        # integration timestep (ms)
            'warmup': 30,       # simulation warmup time (ms)
            'tstop' : 70,       # simulation duration (ms)

            # bipolar stimulation parameters
            'bursts':1,         # bipolar stimulus bursts
            'pulses':4,         # pulses per bursts 
            'pulse_freq':100,   # pulse frequency within burst (Hz)
            'burst_freq':5,     # burst frequency (Hz)
            'noise' : 0,        # noise in input arrival (see NetCon documentation)

            # clopath synapse parameters
            'clopath_delay_steps': 1,
            'clopath_tau_0':6, # time constant (ms) for low passed membrane potential for depression
            'clopath_tau_r' : 15, # time constant (ms) for low pass filter presynaptic variable
            'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
            'clopath_A_m':.0001, # depression magnitude parameter (mV^-1)
            'clopath_A_p': .0005, # amplitude for potentiation (mV^-2)
            'clopath_tetam':-60,#-41, # depression threshold (mV)
            'clopath_tetap':-40,#-38, # potentiation threshold (mV)

            # ampa synapse parameters
            'tau1_ampa' : 0.2,  # rise time constant (ms)
            'tau2_ampa' : 2,    # decay time constant   (ms)
            'i_ampa' : 0.18,    # default peak ampa current in uS

            # facilitation depression parameters for AMPA from Varela et al. 1997
            'f_ampa':5.,
            'tau_F_ampa':94.,
            'd1_ampa':.45,
            'tau_D1_ampa':540.,
            'd2_ampa':.12,
            'tau_D2_ampa':45.,
            'd3_ampa':.98,
            'tau_D3_ampa':120000.,

            # nmda synapse parameters
            'tau1_nmda' : 1,    # rise time constant (ms)
            'tau2_nmda' : 50,   # decay time constant (ms)

            
            # Parameters from Migliore 2005 (signal propogation in oblique dendrites)
            # conductances reported as (nS/um2) in paper, but need to be in (mho/cm2)
            # conversion 10,000*(pS/um2) = 10*(nS/um2) = (mho/cm2) = .001*(mS/cm2)
            # *** units in paper are a typo, values are already reported in (mho/cm2) ***
            'Vrest' : -65.,             # resting potential (mV)
            'gna' :  1.*0.04,#.025,                # peak sodium conductance (mho/cm2)
            'dgna' : -.000025,          # change in sodium conductance with distance (ohm/cm2/um) from Kim 2015
            'ena' : 55.,                    # sodium reversal potential (mV)
            'gna_inact': 0., # sodium slow inactivation factor (1=none, 0=max inactivation)
            'AXONM' : 50.,              # multiplicative factor for axonal conductance to generate axon potentials in AIS
            'SOMAM':1.5,
            'gkdr' : 1.*0.01,#0.01,             # delayed rectifier potassium peak conductance (mho/cm2)
            'ek' : -90.,                    # potassium reversal potential
            'celsius' : 35.0,               # temperature (degrees C)
            'KMULT' :  1.*0.03,#0.03,           # multiplicative factor for distal A-type potassium conductances
            'KMULTP' : 1.*.03,#0.03,                # multiplicative factor for proximal A-type potassium conductances
            'ghd' : 1.*0.0001,#0.0001,         # peak h-current conductance (mho/cm2)
            'gcalbar': 1.*.00125 ,          # L-type calcium conductance from Kim et al. 2015 (mho/cm2)
            'ehd' : -30.,                   # h-current reversal potential (mV)
            'kl_hd' : -6.,#-8.,
            'vhalfl_hd_prox' : -83.,#-73,           # activation threshold for proximal h current (mV)
            'vhalfl_hd_dist' : -83.,#-81,           # activation threshold for distal h-current (mV)
            'vhalfl_kad' : -56.,#-56.,          # inactivation threshold for distal a-type current (mV)
            'vhalfl_kap' : -56.,#-56.,          # inactivation threshold for proximal a-type current (mV)
            'vhalfn_kad' : -1.,#-1.,            # activation threshold for distal a-type urrent (mV)
            'vhalfn_kap' : -1.,#-1.,            # activation threshold for proximal a-type current (mV)
            'RaAll' : 150.,             # axial resistance, all compartments (ohm*cm)
            'RaAx' : 50.,                   # axial resistance, axon (ohm*cm)                   
            'RmAll' : 28000.,           # specific membrane resistance (ohm/cm2)
            'Cm' : 1.,                  # specific membrane capacitance (uf/cm2)
            'ka_grad' : 1.,#1.,#1.,             # slope of a-type potassium channel gradient with distance from soma 
            'ghd_grad' : 1.5,#1.,#3.,                # slope of h channel gradient with distance from soma 
            }

    def add_pathway(self, stim_param):
        """
        Add pathway to parameter dictionary p

        stim_param is a dictionary of parameters that are specific to the pathway being added

        pathways are organized as p{'p_path'}[path number]{'parameter'}
        """
        if 'p_path' not in self.p:
            self.p['p_path'] = {}
        
        self.p['p_path'].append({})
        for param_key, param in stim_param.iteritems():
            self.p['p_path'][-1][param_key]=param


    def set_branch_sequence_ordered(self, **kwargs):
        """ set delays for sequence of inputs on dendritic branch

		---
        Keyword Arguments
        ---
        seg_idx = segment index structure of active segments as ['tree'][section index][segment number]
        	where section index is the index of active sections listed in sec_idx
        delay = fixed delay between each segment active segment (ms)
        direction = 'in' (towards soma) or 'out' (away from soma)

        ---
        Return
        ---
        dic['sequence delays'] = time delays from start of simulation for each active segment.  Same organization as seg_idx: ['tree'][section index][segment number]
        """

        seg_idx = kwargs['seg_idx']
        delay = kwargs['delay']
        direction = kwargs['direction']

        dic={}
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
        dic['sequence_delays']=delays
        return dic

    def choose_branch_manual(self, **kwargs):
        """ manually choose a dendritic branch to activate based on section number

        ---
        Keyword Arguments
        ---
        geo = geometry structure containing hoc section and segment objects
        	['tree'][secction number][segment number]

    	trees = list of trees containing branches to activate

    	sec_list = list of sections to activate [section number]

    	full_path = True/False
    		if True: activate all sections from the given section to the soma
    		if False: only activate the given sections in sec_list

		---
		Return
		---
		dic['sec_idx'] = section index structure for indeces of active sections
			['tree'][section number]
        """

        geo = kwargs['geo']
        trees = kwargs['trees']
        sec_list = kwargs['sec_list']
        full_path = kwargs['full_path']
        dic = {}
        sec_idx = {}
        parent_idx={}
        for tree_key, tree in geo.iteritems():
            sec_idx[tree_key]=[]
            parent_idx[tree_key]=[]
            if tree_key in trees:
                for sec_i, sec in enumerate(sec_list):

                    sec_idx[tree_key].append(sec)
                    parent_idx[tree_key][sec_i].append([])

                    # if full path from branch terminal to soma is to be tracked
                    if full_path:

                        # current section ref starting branch terminal
                        sref_current = h.SectionRef(sec=tree[sec])

                        # while the current section has a parent
                        while sref_current.has_parent:
                            # retrieve parent section object
                            parent=sref_current.parent
                            # get index in geo structure
                            parent_sec_i = [section_i for section_i, section in enumerate(tree) if section is parent][0]
                            # add to parent list
                            parent_idx[tree_key][sec_i].append(parent_sec_i)
                            # update current section ref
                            sref_current = h.SectionRef(sec=parent)
        dic['sec_idx'] = sec_idx
        return dic

    # choose dendritic branch to activate synapses
    
    def choose_branch_rand(self, trees, geo, num_sec=1, distance=[], branch=True, full_path=True):
        """ choose random dendritic branch to activate synapses on and store in sec_idx.  option to store list of sections that lead from branch terminal to soma.

        ---
        Arguments
        ---
        geo = geometry structure containing hoc section and segment objects
        	['tree'][secction number][segment number]

    	trees = list of trees containing branches to activate

    	num_sec = number of sections to choose

    	distance = [minimum distance, maximum distance]
    		if empty list, no distance requirement will be enforced

		branch = True/False
			if True: sections must be branch terminals
			if False: sections can be any section

		full_path = True/False
			if True: keep track of all sections along path from soma to chosen section
			if False: only include chosen section

		---
		Return
		---
		dic['sec_idx'] = section numbers to activate ['tree'][section number]

		dic['parent_idx'] = parent sections along path from soma to section in sec_idx as ['tree'][section index from sec_idx][parent section numbers]
        """

        dic = {}
        # structure to store section list
        sec_idx = {}
        # store list of parent sections for chosen section
        # parent idx will be in reverse order(i.e. from branch terminal towards soma)
        parent_idx ={}

        # iterate over all trees
        for tree_key, tree in geo.iteritems():

            # add list for chosen sections
            sec_idx[tree_key] =[]
            parent_idx[tree_key] = []

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

                    # add dimension for each section
                    parent_idx[tree_key].append([])
                    
                    # check distance and branch requirements
                    dist_check=False
                    branch_check=False
                    iter_cnt = 0
                    while (dist_check==False or branch_check==False) and iter_cnt<len(secs_all):
                        
                        # get distance from soma of section start
                        dist1 = h.distance(0, sec=tree[sec])
                        # distance from soma of section end
                        dist2 = h.distance(1, sec=tree[sec])
                        
                        # if there is a distance requirement
                        if distance:
                            # get minimum distance requirement
                            min_distance=distance[0]
                            
                            # if there is no maximum distance requirement
                            if len(distance)==1:
                                # if min distance reuirement is met
                                if dist2>min_distance:
                                    # update distance check
                                    dist_check=True
                            
                            # if there is also a max distance requirement
                            if len(distance)>1:
                                # get max distance
                                max_distance=distance[0]
                                
                                # if there is a part of the section that satisfies both distance reuirements
                                if dist1<max_distance and dist2>min_distance:
                                    # update distance check
                                    dist_check=True
                        
                        # if there is no distance reuirement
                        else:
                            # update distance check
                            dist_check=True
                        
                        # if there is branch requirement
                        if branch:

                            # get section ref for the selected branch
                            sref = h.SectionRef(sec=tree[sec])

                            # if the section has no children, it is a branch terminal 
                            if sref.nchild() is 0:

                                # update branch check
                                branch_check=True

                        # if no branch reuirement 
                        else:
                            # update branch check
                            branch_check=True

                        # if either reuirement not met, choose a new branch
                        if (not dist_check) or (not branch_check):
                            iter_cnt+=1
                            sec = secs_all[np.random.choice(len(secs_all), 1, replace=False)]
                    
                    # if all sections are iterated through without satisfying requirements
                    if iter_cnt>=len(secs_all) and (not dist_check or not branch_check):
                        sec=[]
                        print 'no sections meet specified distance and branch requirements'

                    # if full path from branch terminal to soma is to be tracked
                    if full_path:

                        # current section ref starting branch terminal
                        sref_current = h.SectionRef(sec=tree[sec])

                        # while the current section has a parent
                        while sref_current.has_parent:
                            # retrieve parent section object
                            parent=sref_current.parent
                            # get index in geo structure
                            parent_sec_i = [section_i for section_i, section in enumerate(tree) if section is parent][0]
                            # add to parent list
                            parent_idx[tree_key][sec_i].append(parent_sec_i)
                            # update current section ref
                            sref_current = h.SectionRef(sec=parent)

                    # once current section is a branch terminal, store the section number in sec_idx
                    sec_idx[tree_key].append(sec)

        # store section list in parameter dictionary
        dic['sec_idx'] = sec_idx
        dic['parent_idx'] = parent_idx

        return dic

    def choose_seg_branch(self, geo, sec_idx, seg_dist, max_seg=[], spacing=0, distance=[]):
        """ choose active segments on branch based on spacing between segments

        Arguments:

        distance is a list with 1 or 2 entries.  If 1 entry is given, this is considered a minimum distance requirement.  If two entries are given, they are treated as min and max requirements respectively
        """

        #FIXME: add segments to active list start at section terminal
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
                
                # number of segments
                num_seg = geo[tree_key][sec].nseg

                # list of segment indeces in reverse order
                seg_i_list_reverse = range(num_seg)[::-1]

                # iterate backwards through segment indeces
                for seg_i in seg_i_list_reverse:

                    # relative segment location
                    seg_loc = float(seg_i+1)/float(num_seg) - 1./(2.*num_seg)

                    # get segment object
                    seg = geo[tree_key][sec](seg_loc)

                # iterate over segments
                # for seg_i, seg in enumerate(seg_list_temp.reverse()):
                    
                    # check distance requirement
                    dist_check=False

                    # if there is a distance requirement
                    if distance:
                        # print dist_check
                        # if there is a max distance requirement
                        if len(distance)>1:
                            
                            print distance[0], seg_dist[tree_key][sec][seg_i], distance[1]
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
                        # print dist_check
                        dist_check=True

                    if not dist_check:
                        print 'distance requirement not met for active segments'

                    # if segment has not been stored yet and min distance is met
                    if (not seg_idx[tree_key][sec_i]) and (dist_check):

                        # add segment to list
                        seg_idx[tree_key][sec_i].append(seg_i)

                    # otherwise check spacing requirement between current segment and previously added segment
                    elif seg_idx[tree_key][sec_i] and ((max_seg and len(seg_idx[tree_key][sec_i])< max_seg) or not max_seg) :
                        
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

                seg_idx[tree_key][sec_i].sort()

        # update parameter dictionary
        self.p['seg_idx'] = seg_idx

    def choose_seg_rand(self, trees, syns, syn_frac, seg_dist, syn_num=[],distance=[], replace=False):
        """ choose random segments to activate
        arguments:
        
        trees = list of subtrees with active synapses

        syn_list = list of all synapses to be chosen from ogranized as {trees}[section number][segment number][synapse type]

        syn_frac = fraction of synapses to be chosen, with equal probability for all synapses

        updates the parameter dictionary according to the chosen synapses
        """
        dic = {
        'sec_list':{},
        'seg_list': {},
        'sec_idx': {},
        'seg_idx': {},
        }
        


        # list all segments as [[section,segment, tree]] 
        segs_all = [[sec_i,seg_i, tree_key] for tree_key, tree in syns.iteritems() for sec_i,sec in enumerate(tree) for seg_i,seg in enumerate(tree[sec_i]) if tree_key in trees]

        # there are multiple distance requirements
        if len(distance)>0 and isinstance(distance[0],list):
            print 'distance:',distance
            sec_list_all = []
            seg_list_all = []
            tree_list_all =[]
            # for each distance requirement
            for distance_i, distances in enumerate(distance):
                # all segments that fit the current distance requirement
                segs_all_dist = [seg for seg_i, seg in enumerate(segs_all) if seg_dist[seg[2]][seg[0]][seg[1]]>distances[0] and seg_dist[seg[2]][seg[0]][seg[1]]<distances[1]] 

                # print len(segs_all_dist)
                # segs_all = segs_all_dist

                # if different synapse numbers are provided for each distance bin
                if isinstance(syn_num,list) and len(syn_num)>0:
                    
                    # choose segments to activate
                    segs_choose = np.random.choice(len(segs_all_dist), int(syn_num[distance_i]), replace=replace)

                # if a single scalar is given
                elif syn_num:
                    print 'syn_num:', int(syn_num)
                    print 'available segments:',len(segs_all_dist)
                    # choose segments to activate
                    segs_choose = np.random.choice(len(segs_all_dist), int(syn_num), replace=replace)
                # if no synapse number is given
                else:
                    # choose segments to activate
                    segs_choose = np.random.choice(len(segs_all_dist), int(syn_frac*len(segs_all)), replace=replace)

                # list of active sections (contains duplicates)
                sec_list_all_temp = [segs_all_dist[a][0] for a in segs_choose]
            
                # list of active segments
                seg_list_all_temp = [segs_all_dist[a][1] for a in segs_choose]

                # list of trees
                tree_list_all_temp = [segs_all_dist[a][2] for a in segs_choose]

                # add to total list for distance requirements
                for i, sec in enumerate(sec_list_all_temp):
                    sec_list_all.append(sec)
                    seg_list_all.append(seg_list_all_temp[i])
                    tree_list_all.append(tree_list_all_temp[i])

        # if only one distacne requirement is given
        elif len(distance) > 0:

            # print 'distance requirement'
            segs_all_dist = [seg for seg_i, seg in enumerate(segs_all) if seg_dist[seg[2]][seg[0]][seg[1]]>distance[0] and seg_dist[seg[2]][seg[0]][seg[1]]<distance[1]] 

            # print len(segs_all_dist)
            # segs_all = segs_all_dist

            # if synapse number is given
            if syn_num:
                print 'syn_num:', int(syn_num)
                print 'available segments:'
                # choose segments to activate
                segs_choose = np.random.choice(len(segs_all_dist), int(syn_num), replace=replace)

            else:
                # choose segments to activate
                segs_choose = np.random.choice(len(segs_all_dist), int(syn_frac*len(segs_all_dist)), replace=replace)

            # list of active sections (contains duplicates)
            sec_list_all = [segs_all_dist[a][0] for a in segs_choose]
        
            # list of active segments
            seg_list_all = [segs_all_dist[a][1] for a in segs_choose]

            # list of trees
            tree_list_all = [segs_all_dist[a][2] for a in segs_choose]

        # if no distance requirement given
        else:
            if syn_num:
                print 'syn_num:', int(syn_num)
                print 'available segments:'
                # choose segments to activate
                segs_choose = np.random.choice(len(segs_all), int(syn_num), replace=replace)

            else:
                # choose segments to activate
                segs_choose = np.random.choice(len(segs_all), int(syn_frac*len(segs_all)), replace=replace)

            # list of active sections (contains duplicates)
            sec_list_all = [segs_all[a][0] for a in segs_choose]
        
            # list of active segments
            seg_list_all = [segs_all[a][1] for a in segs_choose]

            # list of trees
            tree_list_all = [segs_all[a][2] for a in segs_choose]


        for tree_key in trees:

            sec_list = [sec_list_all[i] for i,tree in enumerate(tree_list_all) if tree==tree_key]

            seg_list = [seg_list_all[i] for i,tree in enumerate(tree_list_all) if tree==tree_key]

            sec_idx = list(set(sec_list))

            # list of active segments as [unique section][segments]
            seg_idx = []
            for sec in sec_idx:
                seg_idx.append(list(set([seg_list[sec_i] for sec_i,sec_num in enumerate(sec_list) if sec_num==sec])))

            # update parameter dictionary
            dic['sec_list'][tree_key] = sec_list
            dic['seg_list'][tree_key] = seg_list
            dic['sec_idx'][tree_key] = sec_idx
            dic['seg_idx'][tree_key] = seg_idx
            dic['syn_frac'] = syn_frac
            dic['trees']=trees

        return dic

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
            for sec in sec_idx[tree_key]:
                seg_idx[tree_key].append([seg_list[tree_key][sec_i] for sec_i,sec_num in enumerate(tree) if sec_num==sec])


        # update parameter dictionary
        self.p['sec_list'] = sec_list
        self.p['seg_list'] = seg_list
        self.p['sec_idx'] = sec_idx
        self.p['seg_idx'] = seg_idx

    def set_weights(self, seg_idx, sec_idx, sec_list, seg_list, w_mean, w_std, w_rand):
        """
        sets weights using nested list with same structure as seg_idx: [tree][section index][segment index]

        arguments: 
        seg_idx = nested list of segment numbers to be activated {trees}[section index][segment number]

        w_mean = mean synaptic weight

        w_std = standard deviation of weights, if w_rand is True

        w_rand = choose synaptic weights from normal distibution? (bool)
        """
        dic = {}
        w_list={}

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

                    # find number of repeats for the current segment
                    sec_num = sec_idx[tree_key][sec_i]
                    seg_num = seg
                    repeats = len([1 for i, section in enumerate(sec_list[tree_key]) if (section==sec_num and seg_list[tree_key][i]==seg_num)])
                    # print 'repeats:',repeats

                    # if weights are randomized
                    if w_rand:
                        # choose from normal distribution
                        w_list[tree_key][sec_i].append(np.random.normal(repeats*w_mean,w_std))
                    
                    # otherwise set all weights to the same
                    else:
                        w_list[tree_key][sec_i].append(repeats*w_mean)

        dic['w_list']=w_list
        # update parameter dictionary
        return dic

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

    def seg_location(self, sec):
        """ given a neuron section, output the 3d coordinates of each segment in the section

        ouput is a nested list as [xyz dimension][segment number], with x,y, z dimensions listed in that order

        """
        # number of 3d points in section
        tol =.001
        n3d = int( h.n3d( sec=sec))
        
        # preallocate 3d coordinates
        x = [None]*n3d
        y = [None]*n3d
        z = [None]*n3d
        position_3d =  [None]*n3d
                       
        # loop over 3d coordinates in each section
        for i in range(n3d):
            # retrieve x,y,z
            x[i] = h.x3d(i, sec=sec)
            y[i] = h.y3d(i, sec=sec)
            z[i] = h.z3d(i, sec=sec)

            # calculate total distance of each 3d point from start of section
            if i is 0:
                position_3d[i] = 0
            else:
                position_3d[i] = position_3d[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
        
        seg_x = []
        seg_y = []
        seg_z = []
        for seg_i,seg in enumerate(sec):
                # relative position within section (0-1)
                seg_pos = seg.x            
                
                # segment distance along section in 3D
                seg_dist = seg_pos*position_3d[-1]

                # find first 3D coordinate that contains the segment
                node_i = [dist_i for dist_i,dist in enumerate(position_3d) if dist >= seg_dist]
                
                # if segement occurs exactly at a node set its location to the node location
                if abs(position_3d[node_i[0]] - seg_dist) < tol:
                    seg_x.append( x[ node_i[ 0]])
                    seg_y.append( z[ node_i[ 0]])
                    seg_z.append( z[ node_i[ 0]])

                # otherwise if segment falls between two coordinates, interpolate to get location
                # FIXME clean up
                else:
                    pt1 = position_3d[ node_i[0]-1]
                    pt2 = position_3d[ node_i[0]]
                    scale = (seg_dist-pt1) / (pt2-pt1)
                    interpx = x[ node_i[0]-1] + scale*( x[ node_i[0]] - x[ node_i[0]-1])
                    interpy = y[ node_i[0]-1] + scale*( y[ node_i[0]] - y[ node_i[0]-1])
                    interpz = z[ node_i[0]-1] + scale*( z[ node_i[0]] - z[ node_i[0]-1])
                    seg_x.append( interpx)
                    seg_y.append( interpy)
                    seg_z.append( interpz)
        return [seg_x, seg_y, seg_z]

    def create_morpho(self, geo):
        """ create structure that stores morphology information for plotting with brian2 morphology

        each segment in morpho contains a tuple with seven entries
        (unique_segment_index, name, x, y, z, diam, unique_parent segment_index)

        root segment has index 0, with parent segment index -1
        """

        # initialize morpho structure with same dimensions as geo structure
        morpho = {}
        # iterate over trees
        for tree_key, tree in geo.iteritems():
            morpho[tree_key]=[]
            # iterate over sections
            for sec_i, sec in enumerate(tree):
                morpho[tree_key].append([])
                # iterate over segments
                for seg_i in enumerate(sec):
                    morpho[tree_key][sec_i].append([])

        # find root of cell 
        for tree_key, tree in geo.iteritems():
            for sec_i, sec in enumerate(tree):
                sref = h.SectionRef(sec=sec)
                root = sref.root
                break

        # create new secton list
        nrn_sec_list = h.SectionList()
        # add all seection to list, starting from root
        nrn_sec_list.wholetree()

        # copy nrn section list as a python list
        sec_list = []
        for sec_i_temp, sec_temp in enumerate(nrn_sec_list):
            sec_list.append(sec_temp)

        # nested list for storing segment objects [section_number][segment_number]
        seg_list= []
        # nested list for storing segment indices [section number][segment number]
        seg_list_idx = []
        # nested list for storing index of parent segment, matches seg_list_idx dimesions, [section_number][segment_number]
        parent_list_idx = []
        # keep track of total segment number
        idx = -1
        # iterate through sections in list
        for sec_i, sec in enumerate(sec_list):
            # keep track of the root section
            is_root=False

            # add section dimension to each list
            seg_list.append([])
            seg_list_idx.append([])
            parent_list_idx.append([])

            # reference for current section
            sec_ref =  h.SectionRef(sec=sec)
            
            # find parent section index
            if sec_ref.has_parent():
                parent_sec = sec_ref.parent
                parent_sec_i = [i for i, val in enumerate(sec_list) if parent_sec == val][0]
            else:
                parent_sec_i=-1
                is_root = True

            # iterate through segments in the current section
            for seg_i, seg in enumerate(sec):
                # add to total segments counter and store in lists
                idx+=1
                # copy index count to prevent overwrite during loop
                idx_count = copy.copy(idx)
                # add segment object and index to corresponding list
                seg_list[sec_i].append(seg)
                seg_list_idx[sec_i].append(idx_count)

                # if current segment is not the first in its section 
                if seg_i>0:
                    # set parent to previous segemnt in the section
                    parent_seg_idx = seg_list_idx[sec_i][seg_i-1]
                # else if it is the first segment 
                elif seg_i==0:
                    # if it is the root segment
                    if is_root:
                        parent_seg_idx=-1
                    else:
                        # set parent to the last segment in the parent section
                        parent_seg_idx = seg_list_idx[parent_sec_i][-1]

                # add to list of all parent segments
                parent_list_idx.append(parent_seg_idx)

                # find the current segment in geo structure
                # iterate through geo structure until you find matching segment
                for tree_key_local, tree_local in geo.iteritems():
                    for sec_i_local, sec_local in enumerate(tree_local):
                        for seg_i_local, seg_local in enumerate(sec_local):

                            # if section name and segment index match
                            if (sec.name() == sec_local.name()) and (seg_i == seg_i_local):
                                # segment diameter
                                diam = seg_local.diam
                                # segment xyz coordinates
                                xyz = self.seg_location(sec_local)
                                x = xyz[0][seg_i_local]
                                y = xyz[1][seg_i_local]
                                z = xyz[2][seg_i_local]

                                # segment name
                                name = tree_key_local + '_'+ str(sec_i_local) + '_'  +str(seg_i_local)
                                # create 7-tuple
                                morph_tuple = (idx_count, name, x, y, z, diam, parent_seg_idx)
                                # store in morphology structure
                                morpho[tree_key_local][sec_i_local][seg_i_local] = morph_tuple
            
                                break 
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue

        return morpho




        #     print sec.name()

        # # root_ref = h.SectionRef(sec=root)
        # # idx_count=-2
        # # nchildren_start = root_ref.nchild()
        # # children=[[]]
        # # for child_sec_i in range(int(nchildren_start)):
        # #     child_sec = root_ref.child[child_sec_i]
        # #     print child_sec.name()
        # idx_count=-2
        # children=[[]]
        # children[0].append({'sec':root,'seg':[], 'idx':[]})
        # for seg_i, seg in enumerate(root):
        #     idx_count += 1
        #     print 'idx_count:', idx_count
        #     children[0][0]['seg'].append(seg)
        #     children[0][0]['idx'].append(idx_count)
            
        #     # find corresponding tree, section, segment index
        #                     # iterate 
        #     for tree_key_local, tree_local in geo.iteritems():

        #         for sec_i_local, sec_local in enumerate(tree_local):

        #             for seg_i_local, seg_local in enumerate(sec_local):
        #                 # print sec.name(), sec_local.name()
        #                 # print seg_i, seg_i_local
        #                 if (root.name() == sec_local.name()) and (seg_i == seg_i_local):

        #                     if idx_count < 2:
        #                         print 'segment found:', idx_count
        #                     current_tree=tree_key_local
        #                     current_sec=sec_i_local
        #                     current_seg=seg_i_local
        #                     diam = seg_local.diam
        #                     xyz = self.seg_location(sec_local)
        #                     x = xyz[0][seg_i_local]
        #                     y = xyz[1][seg_i_local]
        #                     z = xyz[2][seg_i_local]
        #                     name = tree_key_local + '_'+ str(sec_i_local) + '_'  +str(seg_i_local)
        #                     morph_tuple = (idx_count, name, x, y, z, diam, -1)
        #                     morpho[tree_key_local][sec_i_local][seg_i_local] = morph_tuple

        #                     break
        #             else:
        #                 continue
        #             break
        #         else:
        #             continue
        #         break
        #     else:
        #         continue
        #     break
        # # print morpho['soma'][0][0]

        # has_children=True
        # while has_children:

        #     # reset check for children sections on each iteration
        #     has_children=False

        #     # children from previous iteration become parents
        #     parents=[]
        #     children_copy = copy.copy(children)
        #     # flatten children
        #     for parent_sec_i, parent_sec in enumerate(children_copy):
        #         for child_sec_i, child_sec in enumerate(parent_sec):
        #             # [sections], each list entry is a dictionary with the section object, list of segment objects, list of indeces for each segment 
        #             if child_sec['sec']:
        #                 parents.append(child_sec)
        #             # print 'child sec:', child_sec


        #     # collect new group of children
        #     children = []
        #     # iterate through parent sections
        #     for parent_sec_i, parent_sec in enumerate(parents):

        #         # add a list to store children for each parent section
        #         children.append([])

        #         # if there are no children, store empty list
        #         if int(h.SectionRef(sec=parent_sec['sec']).nchild()) is 0:
        #             children[parent_sec_i].append({'sec':[], 'seg':[], 'idx':[]})
        #         # else if there are children    
        #         else:
        #             # iterate over child sections
        #             for child_sec_i, child_sec in enumerate(h.SectionRef(sec=parent_sec['sec']).child):

        #                 # add child section to list, add list for storing segments and indeces
        #                 children[parent_sec_i].append({'sec':child_sec, 'seg':[], 'idx':[]})
        #                 # iterate over segments in child section
        #                 for seg_i, seg in enumerate(child_sec):
        #                     # each segment increases index by 1
        #                     idx_count += 1
        #                     # print 'count:', idx_count

        #                     # add segment to list
        #                     children[parent_sec_i][child_sec_i]['seg'].append(seg)
        #                     # add index to list
        #                     children[parent_sec_i][child_sec_i]['idx'].append(idx_count)

        #                     # get index of parent segment
        #                     # if this is the first segment in the child section
        #                     if seg_i is 0:
        #                         # its parent is the last segment from the parent section
        #                         current_parent_idx = parent_sec['idx'][-1]
        #                     # else its parent is the previous segment
        #                     else:
        #                         current_parent_idx = idx_count-1

        #                     # find corresponding tree, section, segment index
        #                     # iterate 
        #                     for tree_key_local, tree_local in geo.iteritems():
        #                         for sec_i_local, sec_local in enumerate(tree_local):

        #                             for seg_i_local, seg_local in enumerate(sec_local):
        #                                 # print sec.name(), sec_local.name()
        #                                 # print seg_i, seg_i_local
        #                                 if (sec.name() == sec_local.name()) and (seg_i == seg_i_local):
        #                                     # print sec.name()

        #                                     current_tree=tree_key_local
        #                                     current_sec=sec_i_local
        #                                     current_seg=seg_i_local
        #                                     diam = seg_local.diam
        #                                     xyz = self.seg_location(sec_local)
        #                                     x = xyz[0][seg_i_local]
        #                                     y = xyz[1][seg_i_local]
        #                                     z = xyz[2][seg_i_local]
        #                                     name = tree_key_local + '_'+ str(sec_i_local) + '_'  +str(seg_i_local)
        #                                     morph_tuple = (idx_count, name, x, y, z, diam, current_parent_idx)
        #                                     morpho[tree_key_local][sec_i_local][seg_i_local] = morph_tuple

        #                                     break
        #                             else:
        #                                 continue
        #                             break
        #                         else:
        #                             continue
        #                         break
        #                     else:
        #                         continue
        #                     break

        #     # check if there are children
        #         for parent_sec_i, child_group in enumerate(children):
        #             if len(child_group)>0:
        #                 has_children=True
        #                 break
        # print 'soma:', morpho['soma'][0][0]
        # return morpho

        # # assign index to each segment
        # seg_idx = {}
        # # count segments
        # counter = -1
        # # iterate over trees
        # for tree_key, tree in geo.iteritems():
        #     # add tree dimension
        #     seg_idx[tree_key]=[]
        #     # iterate over sections
        #     for sec_i, sec in enumerate(tree):
        #         # add section dimension
        #         seg_idx[tree_key].append([])
        #         # iterate over segments
        #         for seg_i, seg in enumerate(sec):
        #             # update counter for each segment
        #             counter+=1
        #             # copy counter to store as unique index for each segment
        #             idx = copy.copy(counter)
        #             # store unique index
        #             seg_idx[tree_key][sec_i].append(idx)

        # # create morpho structure that stores info needed for brian2.morphology methods
        # morpho = {}
        # # iterate over trees
        # for tree_key, tree in geo.iteritems():
        #     morpho[tree_key]=[]
        #     # iterate over sections
        #     for sec_i, sec in enumerate(tree):
        #         morpho[tree_key].append([])
        #         # get 3d location data [xyz][segment_num]
        #         xyz = self.seg_location(sec)
        #         # get section ref
        #         secref = h.SectionRef(sec)
        #         # if section has a parent
        #         parent_sec_idx=[-1]
        #         if secref.has_parent():
        #             # get parent section object
        #             parent = secref.parent
        #             # find unique index of parent seciton (list of all segments in the parent section)
        #             for tree_key_temp, tree_temp in seg_idx.iteritems():
        #                 for sec_i_temp, sec_temp in enumerate(tree_temp):
        #                     if parent is geo[tree_key_temp][sec_i_temp]:
        #                         # list of section indeces in parent section
        #                         parent_sec_idx = sec_temp
        #         # if no parent section, the unique index is -1
        #         else:
        #             parent_sec_idx = [-1]

        #         # iterate over segments
        #         for seg_i, seg in enumerate(sec):
        #             # if first segment in section
        #             if seg_i == 0:
        #                 # FIXME (what if child is attached to the 0 end of the parent section?)
        #                 # parent segment is the last segment in the parent section
        #                 parent_idx = parent_sec_idx[-1]
        #             else:
        #                 # otherwise parent segment is the previous segment in the current section
        #                 parent_idx = seg_idx[tree_key][sec_i][seg_i-1]
        #             # get unique index of current segment
        #             index = seg_idx[tree_key][sec_i][seg_i]
        #             # xyz coordinates
        #             x = xyz[0][seg_i]
        #             y = xyz[1][seg_i]
        #             z = xyz[2][seg_i]

        #             # diameter
        #             diam = seg.diam
        #             # name
        #             name = tree_key + '_'+ str(sec_i) + '_'  +str(seg_i)

        #             morph_tuple = (index, name, x, y, z, diam, parent_idx)
        #             morpho[tree_key][sec_i].append(morph_tuple)

        # return morpho

# set procedure if called as a script
if __name__ == "__main__":
    pass