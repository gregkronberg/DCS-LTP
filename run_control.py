

"""
run control
"""
# imports
# from mpi4py import MPI
# import multiprocessing 
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
import pickle

h.load_file("stdrun.hoc")

# 
class Experiment:
    """ Impliment experimental procedures.  
    """
    def __init__(self, **kwargs):
        """ choose experiment to run

        kwargs must be a dictionary with 'experiment' as a key, the corresponding value points to a given experiment, which is then fed the same kwarg dictionary and run
        """
        if not kwargs:
            pass
        else:
            # retrieve which experiment to run
            experiment = getattr(self, kwargs['experiment'])

            # run experiment
            experiment(**kwargs) 
    #%%
    def quick_run(self, **kwargs):
        """ single simulation to test facilitatio/depression parameters
        """
        w_mean = .001 # weight of single synapse uS
        trees = ['apical_tuft']
        nsyns = 2.
        syn_nums = [2.]
        syn_dist = [0., 200.]
        trials = 1
        stim_freqs=[100.]
        paths = {'1':{}}
        param_file='fd_parameters.pkl'
        self.kwargs = {
        'experiment' : kwargs['experiment'], 
        'trees' : trees,
        'nsyns':nsyns,
        'syn_num':[],
        'syn_dist': syn_dist,
        'f_ampa':1.,
        'tau_F_ampa':94.,
        'd1_ampa':1.,
        'tau_D1_ampa':380.,
        'd2_ampa':.5,
        'tau_D2_ampa':9200.,
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
        'trials' : trials,
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
        'pulses':0,
        'pulse_freq':100,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':70,
        'clopath_tau_r':8,
        'gna_inact': 0.
        }

        if 'load_fd' in kwargs:
            # load parameters
            #````````````````
            with open('Data/'+param_file, 'rb') as pkl_file:
                param_obj = pickle.load(pkl_file)

            params = param_obj.x

            self.kwargs['f_ampa'] = params[0]
            self.kwargs['tau_F_ampa'] = params[1]
            self.kwargs['d1_ampa'] = params[2]
            self.kwargs['tau_D1_ampa'] = params[3]
            self.kwargs['d2_ampa'] = params[4]
            self.kwargs['tau_D2_ampa'] = params[5]
            self.kwargs['d3_ampa'] = params[6]
            self.kwargs['tau_D3_ampa'] = params[7]


        # update kwargs
        for key, val in kwargs.iteritems():
            self.kwargs[key]=val
        
        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        threshold=-30

        # iterate over frequency
        # iterate over synapse number
        # iterate over trials
        # iterate over all segments in tree
        for freq_i, freq in enumerate(stim_freqs):
            self.p['tstop'] = (self.p['pulses']*1000/freq)+self.p['warmup']
            self.p['pulse_freq'] = freq
            self.p['field_off'] = self.p['tstop']
            for syn_num_i, syn_num in enumerate(syn_nums):
                for trial_i, trial in enumerate(range(self.p['trials'])):
                    self.p['syn_num']=syn_num
                    self.p['p_path']={}
                    
                    idx = self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'], replace=True)

                    for key, val in idx.iteritems():
                        self.p[key]=val

                    self.p['w_mean'] = self.p['nsyns']*w_mean

                    self.p['w_list'] = self.p_class.set_weights(seg_idx=self.p['seg_idx'], sec_idx=self.p['sec_idx'], sec_list=self.p['sec_list'], seg_list=self.p['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])['w_list']

                    self.p['sequence_delays'] = self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])['sequence_delays']

                    print 'syn_num:', self.p['syn_num']
                    print 'nsyn:',self.p['nsyns'], 'w (nS):',self.p['w_mean'] 
                    print 'distance from soma:', self.p['syn_dist']

                    # store trial number
                    self.p['trial']=trial
                                    
                    # create unique identifier for each trial
                    self.p['trial_id'] = str(uuid.uuid4())

                    for path_key, path in paths.iteritems():
                        self.p['p_path'][path_key]=copy.copy(self.p)
                                    
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
                    '_dist_'+str(self.p['syn_dist'][-1])+
                    '_freq_'+str(self.p['pulse_freq'])+
                    '_syn_num_'+str(self.p['syn_num'])+
                    '_trial_'+str(self.p['trial'])+
                    '_id_'+self.p['trial_id']
                    )

                    # save data for eahc trial
                    run.save_data(data=sim.data, file_name=file_name)

                    # plot traces
                    analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        sec_idx=self.p['sec_idx'], 
                        seg_idx=self.p['seg_idx'],
                        variables=self.p['plot_variables'],
                        x_variables=self.p['x_variables'],
                        file_name=file_name,
                        group_trees=self.p['group_trees'],
                        xlim=[self.p['warmup']-5,self.p['tstop']],
                        ylim=[])

    """
    EXPERIMENT 1
    Distance dependence of DCS effects
    """
    def exp_1a(self, **kwargs):
        """ 
        activate a varying number of synapses at varying frequency with varying distance from the soma.  Synapses are chosen from a window of 50 um, with the window moving along the apical dendrite.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of mean distance from the soma and frequency of synaptic activity.  

        """
        print 'running', kwargs['experiment'], 'on worker', pc.id()
        print kwargs
        w_mean = .001 # weight of single synapse uS
        trees = kwargs['trees']
        nsyns = kwargs['nsyns']
        syn_nums = kwargs['syn_num']
        syn_dist = kwargs['syn_dist']
        trials = 20
        self.kwargs = {
        'experiment' : 'exp_1a', 
        'trees' : trees,
        'nsyns':nsyns,
        'syn_num':[],
        'syn_dist': syn_dist,
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
        'trials' : trials,
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
        'pulses':4,
        'pulse_freq':100,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':70,
        'clopath_tau_r':8,
        'gna_inact': 0.
        }

        # update kwargs
        for key, val in kwargs.iteritems():
            self.kwargs[key]=val
        
        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        threshold=-30

        # iterate over frequency
        # iterate over synapse number
        # iterate over trials
        # iterate over all segments in tree
        for freq_i, freq in enumerate(kwargs['stim_freqs']):
            self.p['tstop'] = (self.p['pulses']*1000/freq)+self.p['warmup']
            self.p['pulse_freq'] = freq
            self.p['field_off'] = self.p['tstop']
            for syn_num_i, syn_num in enumerate(syn_nums):
                for trial_i, trial in enumerate(range(self.p['trials'])):
                    self.p['syn_num']=syn_num
                    self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'], replace=True)

                    self.p['w_mean'] = self.p['nsyns']*w_mean

                    self.p_class.set_weights(seg_idx=self.p['seg_idx'], sec_idx=self.p['sec_idx'], sec_list=self.p['sec_list'], seg_list=self.p['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

                    self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

                    print 'syn_num:', self.p['syn_num']
                    print 'nsyn:',self.p['nsyns'], 'w (nS):',self.p['w_mean'] 
                    print 'distance from soma:', self.p['syn_dist']

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
                    '_dist_'+str(self.p['syn_dist'][-1])+
                    '_freq_'+str(self.p['pulse_freq'])+
                    '_syn_num_'+str(self.p['syn_num'])+
                    '_trial_'+str(self.p['trial'])+
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

    def exp_1b(self, **kwargs):
        """ 
        activate a varying number of synapses with varying distance from the soma.  Synapses are chosen from a window of 200 um, with the window moving along the apical dendrite.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of mean distance from the soma and frequency of synaptic activity.  

        Similar to 1a, with larger distance window for synapses to be activated

        """
        print 'running', kwargs['experiment'], 'on worker', pc.id()
        print kwargs
        w_mean = .001 # weight of single synapse uS
        trees = kwargs['trees']
        nsyns = kwargs['nsyns']
        syn_nums = kwargs['syn_num']
        syn_dist = kwargs['syn_dist']
        trials = 20
        self.kwargs = {
        'experiment' : 'exp_1b', 
        'trees' : trees,
        'nsyns':nsyns,
        'syn_num':[],
        'syn_dist': syn_dist,
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
        'trials' : trials,
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
        'pulses':4,
        'pulse_freq':100,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':70,
        'clopath_tau_r':8,
        'gna_inact': 0.
        }

        # update kwargs
        for key, val in kwargs.iteritems():
            self.kwargs[key]=val
        
        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        threshold=-30

        # iterate over frequency
        # iterate over synapse number
        # iterate over trials
        # iterate over all segments in tree
        for freq_i, freq in enumerate(kwargs['stim_freqs']):
            self.p['tstop'] = (self.p['pulses']*1000/freq)+self.p['warmup']
            self.p['pulse_freq'] = freq
            self.p['field_off'] = self.p['tstop']
            for syn_num_i, syn_num in enumerate(syn_nums):
                for trial_i, trial in enumerate(range(self.p['trials'])):
                    self.p['syn_num']=syn_num
                    self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'], replace=True)

                    self.p['w_mean'] = self.p['nsyns']*w_mean

                    self.p_class.set_weights(seg_idx=self.p['seg_idx'], sec_idx=self.p['sec_idx'], sec_list=self.p['sec_list'], seg_list=self.p['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

                    self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

                    print 'syn_num:', self.p['syn_num']
                    print 'nsyn:',self.p['nsyns'], 'w (nS):',self.p['w_mean'] 
                    print 'distance from soma:', self.p['syn_dist']

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
                    '_dist_'+str(self.p['syn_dist'][-1])+
                    '_freq_'+str(self.p['pulse_freq'])+
                    '_syn_num_'+str(self.p['syn_num'])+
                    '_trial_'+str(self.p['trial'])+
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

    def exp_1c(self, **kwargs):
        """ 
        activate a varying number of synapses in proximal (0-200/0-300 um) and distal regions (400-600/300-600 um) simultameously.  Synapses are chosen from a window of 200 or 300 um.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of nnumber of synapses. Does pairing with proximal inputs (e.g. 0-200 um) cause distal inputs (eg 400-600 um) to come under greater control from the soma? 

        Similar to 1a and 1b, now pairing two distance windows (proximal and distal)

        """
        print kwargs
        w_mean = .001 # weight of single synapse uS
        trees = kwargs['trees']
        nsyns = kwargs['nsyns']
        syn_nums = kwargs['syn_num']
        syn_dist = kwargs['syn_dist']
        trials = kwargs['trials']
        self.kwargs = {
        'experiment' : 'exp_1c', 
        'trees' : trees,
        'nsyns':nsyns,
        'syn_num':[],
        'syn_dist': syn_dist,
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
        'trials' : trials,
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
        'pulses':4,
        'pulse_freq':100,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':70,
        'clopath_tau_r':8,
        'gna_inact': 0.
        }

        # update kwargs
        for key, val in kwargs.iteritems():
            self.kwargs[key]=val
        
        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        threshold=-30

        # iterate over frequency
        for freq_i, freq in enumerate(kwargs['stim_freqs']):
            # update simulation duration based bipolar pulse protocol
            self.p['tstop'] = (self.p['pulses']*1000/freq)+self.p['warmup']
            self.p['pulse_freq'] = freq
            self.p['field_off'] = self.p['tstop']
            # iterate over synapse number
            for syn_num_i, syn_num in enumerate(syn_nums):
                # iterate over trials
                for trial_i, trial in enumerate(range(self.p['trials'])):
                    # update synapse number in parameter dictionary
                    self.p['syn_num']=syn_num
                    # choose random segments based on distance requirement
                    idx = self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'], replace=True)
                    for key, val in idx.iteritems():
                        self.p[key]=val

                    # update weight based on number of synapses per segment (always 1 in this experiment)
                    self.p['w_mean'] = self.p['nsyns']*w_mean

                    # create nested list of weights to match seg_idx structure
                    self.p['w_list'] = self.p_class.set_weights(seg_idx=self.p['seg_idx'], sec_idx=self.p['sec_idx'], sec_list=self.p['sec_list'], seg_list=self.p['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])['w_list']

                    # create a list of time delays for each synapse, same organization as seg_idx 
                    self.p_class.set_branch_sequence_ordered(seg_idx=self.p['seg_idx'], delay=self.p['sequence_delay'], direction=self.p['sequence_direction'])

                    print 'syn_num:', self.p['syn_num']
                    print 'nsyn:',self.p['nsyns'], 'w (nS):',self.p['w_mean'] 
                    print 'distance from soma:', self.p['syn_dist']

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
                    '_dist_'+str(self.p['syn_dist'][-1])+
                    '_freq_'+str(self.p['pulse_freq'])+
                    '_syn_num_'+str(self.p['syn_num'])+
                    '_trial_'+str(self.p['trial'])+
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

    """
    EXPERIMENT 2

    Shapeplots for spike initiation as a function of synapse/segment location
    Recreate figure 1 from Bono and Clopath 2017
    """
    def exp_2c(self, **kwargs):
        """ active each segment with varying weights (number of synapses). monitor spike initiation in soma/dendrite as a function of distance from soma

        """
        w_mean = .001 # weight of single synapse uS
        trees = ['basal']
        self.kwargs = {
        'experiment' : 'exp_2c', 
        'trees' : ['basal'],
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
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        for tree_key_morph, tree_morph in self.p['morpho'].iteritems():
            for sec_i_morph, sec_morph in enumerate(tree_morph):
                for seg_i_morph, seg_morph in enumerate(sec_morph):
                    pass


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
                                
    def exp_2d(self, **kwargs):
        """ Apply a single theta burst for each segment, one segment at a time.  Step through increasing number of synapses (weight) to detect thresholds

        Simulations are run in parallel.  See the functions _f_parallel and _run_parallel

        """
        w_mean = .001 # weight of single synapse uS
        trees = ['apical_tuft','apical_trunk','basal']
        nsyns = kwargs['nsyns']
        self.kwargs = {
        'experiment' : 'exp_2d', 
        'trees' : trees,
        'nsyns':nsyns,
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
        'pulses':4,
        'pulse_freq':100,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':70
        }

        # update kwargs
        for key, val in kwargs.iteritems():
            self.kwargs[key]=val
        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in self.kwargs.iteritems():        # update parameters
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

        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        for tree_key_morph, tree_morph in self.p['morpho'].iteritems():
            for sec_i_morph, sec_morph in enumerate(tree_morph):
                for seg_i_morph, seg_morph in enumerate(sec_morph):
                    pass


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

    """
    EXPERIMENT 3

    Frequency dependence (compare 20 Hz to TBS) and dose response
    """
    def exp_3a1(self, **kwargs):
        """ 20 Hz simulation (40 pulses, 2 seconds) with sodium channel inactivation.  Track backpropagation of action potentials
        """
        # 
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':'1',
        'clopath_A_m':40E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up two pathways
        paths_update = {'1':{
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 40.,
        'pulse_freq': 20.,
        'bursts': 1,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            update_synapse_parameters(P, P.p, P.paths, cell1=cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # print cell1.syns
            # run simulation
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        variables=P.p['plot_variables'],
                        x_variables=P.p['x_variables'],
                        file_name=file_name,
                        group_trees=P.p['group_trees'],
                        xlim=[P.p['warmup']-5,P.p['tstop']],
                        ylim=[-75,-40])

    def exp_3a1_basal(self, **kwargs):
        """ 20 Hz simulation (40 pulses, 2 seconds) with sodium channel inactivation.  Track backpropagation of action potentials
        """
        # 
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':'1',
        'clopath_A_m':100E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-67,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':20,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up two pathways
        paths_update = {'1':{
        'trees': ['basal'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 40.,
        'pulse_freq': 20.,
        'bursts': 1,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            update_synapse_parameters(P, P.p, P.paths, cell1=cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # print cell1.syns
            # run simulation
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        variables=P.p['plot_variables'],
                        x_variables=P.p['x_variables'],
                        file_name=file_name,
                        group_trees=P.p['group_trees'],
                        xlim=[P.p['warmup']-5,P.p['tstop']],
                        ylim=[-75,-40])
    
    def exp_3a2(self, **kwargs):
        """ TBS simulation (40 pulses, 2 seconds) in full pyramidal cell with parameters from Migliore et al. 2005.  Track backpropagation of action potentials
        """
        trials = 1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':['1'],
        'clopath_A_m':3E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up synaptic pathway parameters
        paths_update = {'1':{
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 4.,
        'pulse_freq': 100.,
        'bursts': 10.,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update,)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            # create list of active synapses, weights, delays
            # stored in P.p['seg_idx', 'w_list', 'sequence_delays'], 
            update_synapse_parameters(P, P.p, P.paths, cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # run simulation
            print P.p['p_path']['1']['pulses']
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(
                data=sim.data, 
                variables=P.p['plot_variables'],
                x_variables=P.p['x_variables'],
                file_name=file_name,
                group_trees=P.p['group_trees'],
                xlim=[P.p['warmup']-5,P.p['tstop']],
                 ylim=[-75,-40])

    def exp_3a2_dose(self, **kwargs):
        """ TBS simulation (40 pulses, 2 seconds) in full pyramidal cell with parameters from Migliore et al. 2005.  Sweep across field intensities to get dose response
        """
        trials = 1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,-5.,-1.,-0.5, 0.,0.5, 1., 5., 20.],
        'plot_variables':['v','gbar'],
        'path_combo':['1'],
        'clopath_A_m':100E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-67,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':20,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up synaptic pathway parameters
        paths_update = {'1':{
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 4.,
        'pulse_freq': 100.,
        'bursts': 10.,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update,)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            # create list of active synapses, weights, delays
            # stored in P.p['seg_idx', 'w_list', 'sequence_delays'], 
            update_synapse_parameters(P, P.p, P.paths, cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # run simulation
            # print P.p['p_path']['1']['pulses']
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(
                data=sim.data, 
                variables=P.p['plot_variables'],
                x_variables=P.p['x_variables'],
                file_name=file_name,
                group_trees=P.p['group_trees'],
                xlim=[P.p['warmup']-5,P.p['tstop']],
                 ylim=[-75,-40])
    
    def exp_3a2_basal(self, **kwargs):
        """ TBS simulation (40 pulses, 2 seconds) in full pyramidal cell with parameters from Migliore et al. 2005.  Track backpropagation of action potentials
        """
        trials = 1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':['1'],
        'clopath_A_m':100E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-67,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':20,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up synaptic pathway parameters
        paths_update = {'1':{
        'trees': ['basal'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 4.,
        'pulse_freq': 100.,
        'bursts': 10.,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update,)

        print 'trials:', P.p['trials']
        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            # create list of active synapses, weights, delays
            # stored in P.p['seg_idx', 'w_list', 'sequence_delays'], 
            update_synapse_parameters(P, P.p, P.paths, cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # run simulation
            # print P.p['p_path']['1']['pulses']
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(
                data=sim.data, 
                variables=P.p['plot_variables'],
                x_variables=P.p['x_variables'],
                file_name=file_name,
                group_trees=P.p['group_trees'],
                xlim=[P.p['warmup']-5,P.p['tstop']],
                 ylim=[-75,-40])

    def exp_3a3(self, **kwargs):
        """ 1 Hz simulation (40 pulses, 40 seconds) in full pyramidal cell with parameters from Migliore et al. 2005.  Track backpropagation of action potentials
        """
        trials = 1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':['1'],
        'clopath_A_m':3E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up synaptic pathway parameters
        paths_update = {'1':{
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': 8,
        'nsyns': 1.,
        'syn_dist': [0,300],
        'pulses': 40.,
        'pulse_freq': 1.,
        'bursts': 1.,
        'burst_freq': 1.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update,)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            # create list of active synapses, weights, delays
            # stored in P.p['seg_idx', 'w_list', 'sequence_delays'], 
            update_synapse_parameters(P, P.p, P.paths, cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # run simulation
            print P.p['p_path']['1']['pulses']
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(
                data=sim.data, 
                variables=P.p['plot_variables'],
                x_variables=P.p['x_variables'],
                file_name=file_name,
                group_trees=P.p['group_trees'],
                xlim=[P.p['warmup']-5,P.p['tstop']],
                 ylim=[-75,-40])
    
    def exp_3b1(self, **kwargs):
        """ 20 Hz simulation (40 pulses, 2 seconds) with reduced cylindrical model
        """
        # 
        # kwargs['trials']=1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-5.,0.,5.],
        'plot_variables':['v','gbar'],
        'path_combo':'1',
        'nseg':1,
        'clopath_A_m':4E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up two pathways
        paths_update = {'1':{
        'trees': ['apical_dist'],
        'syn_num': 10,
        'nsyns': 1.,
        'syn_dist': [0,2000],
        'pulses': 0.,
        'pulse_freq': 20.,
        'bursts': 0,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='PyramidalCylinder',
            p_update=p_update,
            paths_update=paths_update,
            load_fd=False)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            update_synapse_parameters(P, P.p, P.paths, cell1=cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # print cell1.syns
            # run simulation

            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        variables=P.p['plot_variables'],
                        x_variables=P.p['x_variables'],
                        file_name=file_name,
                        group_trees=P.p['group_trees'],
                        xlim=[P.p['warmup']-5,P.p['tstop']],
                        ylim=[-75,-40])

    def exp_3b3(self, **kwargs):
        """ 1 Hz simulation (40 pulses, 40 seconds) with reduced cylindrical model
        """
        # 
        kwargs['trials']=1
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-5.,0.,5.],
        'plot_variables':['v','gbar'],
        'path_combo':'1',
        'nseg':1,
        'clopath_A_m':4E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up two pathways
        paths_update = {'1':{
        'trees': ['apical_dist'],
        'syn_num': 10,
        'nsyns': 1.,
        'syn_dist': [0,2000],
        'pulses': 40.,
        'pulse_freq': 1.,
        'bursts': 1,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='PyramidalCylinder',
            p_update=p_update,
            paths_update=paths_update,
            load_fd=False)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            update_synapse_parameters(P, P.p, P.paths, cell1=cell1)

            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # print cell1.syns
            # run simulation

            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        variables=P.p['plot_variables'],
                        x_variables=P.p['x_variables'],
                        file_name=file_name,
                        group_trees=P.p['group_trees'],
                        xlim=[P.p['warmup']-5,P.p['tstop']],
                        ylim=[-75,-40])

    def exp_3c1(self, **kwargs):
        """ Measure membrane polarization and create shapeplot
        """
        # 
        p_update = {
        'experiment' : kwargs['experiment'], 
        'trials' : kwargs['trials'],
        'field':[-20.,0.,20.],
        'plot_variables':['v','gbar'],
        'path_combo':'1',
        'clopath_A_m':40E-5, # depression magnitude parameter (mV^-1)
        'clopath_tetam':-70,#-41, # depression threshold (mV)
        'clopath_tetap':-65,#-38, # potentiation threshold (mV)
        'clopath_tau_r':8,#-38, # potentiation threshold (mV)
        'clopath_tau_0':30,#-38, # potentiation threshold (mV)
        'clopath_tau_y': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'clopath_A_p': 38E-5, # amplitude for potentiation (mV^-2)
        }

        # set up two pathways
        paths_update = {'1':{
        'trees': ['basal','apical_tuft','apical_trunk'],
        'syn_num': [],
        'syn_frac':1.,
        'replace_syn':False,
        'nsyns': 0.,
        'syn_dist': [],
        'pulses': 0.,
        'pulse_freq': 20.,
        'bursts': 1,
        'burst_freq': 5.,
        'warmup': 30,
        'w_mean': .0015,
        }}

        # setup cell and updated parameter structures
        P, cell1 = setup_parameters(
            default_p='migliore_2005',
            cell_class='CellMigliore2005',
            p_update=p_update,
            paths_update=paths_update)

        # iterate over trials
        for trial_i, trial in enumerate(range(P.p['trials'])):

            update_synapse_parameters(P, P.p, P.paths, cell1=cell1)

            P.p['tstop']=50
            # store trial number
            P.p['trial']=trial
                            
            # create unique identifier for each trial
            P.p['trial_id'] = str(uuid.uuid4())
                            
            # start timer
            start = time.time() 
            
            # print cell1.syns
            # run simulation
            sim = run.Run(p=P.p, cell=cell1) 

            # end timer
            end = time.time() 

            # print trial and simulation time
            print 'trial'+ str(P.p['trial']) + ' duration:' + str(end -start) 
            
            # set file name to save data
            file_name = str(
            P.p['experiment']+
            '_trial_'+str(P.p['trial'])+
            '_id_'+P.p['trial_id']
            )

            # save data for eahc trial
            run.save_data(data=sim.data, file_name=file_name)

            # plot traces
            # analysis.PlotRangeVar().plot_trace(data=sim.data, 
            #             variables=P.p['plot_variables'],
            #             x_variables=P.p['x_variables'],
            #             file_name=file_name,
            #             group_trees=P.p['group_trees'],
            #             xlim=[P.p['warmup']-5,P.p['tstop']],
            #             ylim=[-75,-40])
    """
    Experiment 4

    Pathway specificity, associativity
    """
    def exp_4a(self, **kwargs):
        """ Associative plasticity experiment

        Activate a strong (100 Hz burst) and weak (single pulse in the middle of burst) pathway (separately or together).
        Vary which pathways are active (strong only, weak only, strong+weak)
        Vary the distance from soma of each pathway (0-200,200-400,400-600 um).
        Vary the number of synapses in each pathway (10, 20, 40).
        Synapse locations are generated randomly with param.choose_seg_rand(). Multiple synapses can be placed on the same segment.  The weight is then multiplied by the number of "synapses"

        Keyword Arguments:
        syn_num= number of synapses for each pathway [iterable conditions][strong path, weak path]

        syn_dist= dist requirement for each path [pathway][min distance, max distance]

        path_combos= list of path combinations, e.g. ['strong', 'weak', 'strong_weak']

        trials= number of trials for each set of conditions

        Conditions that are specific to each path (e.g. stimulation parameters, seg_idx) are set up in the 'paths' dictionary locally.  Each entry in the local paths dictionary is transferred to the global parameter dictionary 'p'. Each path is organized as p{'p_path'}{'path_key'}{'parameters'}, where path_key is the name of the path e.g. 'strong', or '1'
        """
        print 'running', kwargs['experiment'], 'on worker', pc.id()
        print kwargs

        # update cell-wide parameters
        # weight of single synapse uS
        kwargs['w_mean'] = .001
        kwargs['plot_variables']=['v','gbar'] 

        # instantiate default parameter class
        self.p_class = param.Default()

        # reference to default parameters
        self.p = self.p_class.p

        # update parameters
        for key, val in kwargs.iteritems():        # update parameters
            self.p[key] = val

        # load cell and store in parameter dictionary
        cell1 = cell.CellMigliore2005(self.p)
        cell1.geometry(self.p)
        # insert mechanisms
        cell1.mechanisms(self.p)
        # measure distance of each segment from the soma and store in parameter dictionary
        self.p_class.seg_distance(cell1)
        # create morphology structure
        self.p['morpho'] = self.p_class.create_morpho(cell1.geo)

        # data and figure folder
        self.p['data_folder'] = 'Data/'+self.p['experiment']+'/'
        self.p['fig_folder'] =  'png figures/'+self.p['experiment']+'/'

        # set up two pathways
        paths={}
        paths['strong']={
        'sequence_delay':0,
        'sequence_direction': 'in',
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': [],
        'syn_dist': kwargs['syn_dist'][0],
        'pulses': 4.,
        'pulse_freq': 100.,
        'bursts': 1,
        'burst_freq': 5.,
        'warmup': self.p['warmup'],
        'noise': 0
        }
        paths['weak']={
        'sequence_delay':0,
        'sequence_direction': 'in',
        'trees': ['apical_tuft','apical_trunk'],
        'syn_num': [],
        'syn_dist': kwargs['syn_dist'][1],
        'pulses': 1.,
        'pulse_freq': 100.,
        'bursts': 1, 
        'burst_freq': 5.,
        'warmup': self.p['warmup'] + 1.5*1000/paths['strong']['pulse_freq'],
        'noise': 0
        }
        # iterate through different numbers of synapses and distance requirements
        # kwargs{'syn_num'}[condition number][path number (strong, then weak)]
        # kwargs{'syn_dist'}[condition number][path number (strong, then weak)][min distance, max distance]
        for syn_num_i, syn_num in enumerate(kwargs['syn_num']):
            # FIXME update self.p['syn_num']
            # update parameters for each pathway
            paths['strong']['syn_num']= syn_num[0]
            paths['weak']['syn_num']= syn_num[1]
            self.p['syn_num']=syn_num

            # iterae over path combinations
            for combo_i, combo in enumerate(kwargs['path_combos']):
                self.p['path_combo']=combo
                # update tstop for each combination
                tstops=[]
                for path_key, path in paths.iteritems():
                    tstops.append((path['pulses']*1000/path['pulse_freq'])+path['warmup'])
                self.p['tstop'] = max(tstops)
                self.p['field_off'] = self.p['tstop']

                # iterate over trials
                for trial_i, trial in enumerate(range(kwargs['trials'])):
                    self.p['p_path']={}
                    
                    # update pathways and add to global p structure
                    for path_key, path in paths.iteritems():

                        print path_key, ' warmup:', path['warmup']
                        
                        # if path is included in current combination
                        if path_key in combo:
                            
                            # choose active segments for this pathway
                            seg_dic = self.p_class.choose_seg_rand(trees=path['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=path['syn_num'], distance=path['syn_dist'], replace=True)
                            path.update(seg_dic)
                            
                            # set weights for each segment in this pathway
                            w_dic = self.p_class.set_weights(seg_idx=path['seg_idx'], sec_idx=path['sec_idx'], sec_list=path['sec_list'], seg_list=path['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])
                            path.update(w_dic)

                            # set delays for branch sequences in this pathway 
                            delay_dic = self.p_class.set_branch_sequence_ordered(seg_idx=path['seg_idx'], delay=path['sequence_delay'], direction=path['sequence_direction'])
                            path.update(delay_dic)

                            # update p_path dictionary in global p dictionary
                            self.p['p_path'][path_key]=copy.copy(path)

                    print combo, self.p['syn_num']

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
                    '_'+self.p['path_combo']+
                    '_dist_'+str(self.p['syn_dist'])+
                    '_freq_'+str(self.p['pulse_freq'])+
                    '_syn_num_'+str(syn_num)+
                    '_trial_'+str(self.p['trial'])+
                    '_id_'+self.p['trial_id']
                    )

                    # save data for eahc trial
                    run.save_data(data=sim.data, file_name=file_name)

                    # plot traces
                    analysis.PlotRangeVar().plot_trace(data=sim.data, 
                        variables=self.p['plot_variables'],
                        x_variables=self.p['x_variables'],
                        file_name=file_name,
                        group_trees=self.p['group_trees'],
                        xlim=[self.p['warmup']-5,self.p['tstop']],
                        ylim=[])

def setup_parameters(default_p, p_update, paths_update, cell_class, load_fd=True, data_folder='Data/',**kwargs):
    '''
    ===Args===
    -default_p  : string specifying default parameters.  calls the corresponding method in param.Default
    -p_update   : dictionary of global parameters that are specific to the current experiment. will be used to update the corresponding entries in the default dictionary specified by default_p
    -paths_update   : dictionary path-specific parameters.  Organized as paths_update{path name}{parameters}
    -cell       : string specifying the cell class to be instantiated from cell module as cell.CellClass()

    ===Out===
    -P  : default parameter class containing methods for specifying parameters
    -p  : updated parameter dictionary
    -paths  : updated path-specific parameter dictionary as paths{path name}{parameters}
    -cell   : cell class instance

    ===Updates===
    ===Comments===
    '''
    # instantiate default parameter class
    P = param.Default()

    # load parameters from migliore model
    getattr(P, default_p)()

    # reference to default parameters
    p = P.p
    paths = P.paths

    # load facilitation depression parameters
    if load_fd:
        P.load_fd_parameters(p=p, filename='Data/fd_parameters.pkl')

    # update parameter dictionaries
    p.update(p_update)
    for key, val in paths.iteritems():
        val.update(paths_update[key])

    # data and figure folder
    p['data_folder'] = data_folder+p['experiment']+'/'
    p['fig_folder'] =  'png figures/'+p['experiment']+'/'

    # load cell and store in parameter dictionary
    cell1 = getattr(cell, cell_class)(p)
    cell1.geometry(p)
    # insert mechanisms
    cell1.mechanisms(p)
    
    # measure distance of each segment from the soma and store in parameter dictionary
    P.seg_distance(cell1)

    # create morphology for shape plots
    p['morpho'] = P.create_morpho(cell1.geo)

    # update simulation duration
    p['tstop']  = p['warmup'] + 1000*(p['bursts']-1)/p['burst_freq'] + 1000*(p['pulses']+1)/p['pulse_freq'] 
    p['field_off'] = p['tstop']

    return P, cell1

def update_synapse_parameters(p_class, p, paths, cell1):
    ''' update parameter dictionaries for each pathway before running simulation

    ===Args===
    -p_class    : instance of default parameter class, containing parameter dictionaries and methods for updating parameters for simulations
    -p          : full parameter dictionary
    -paths      : parameter dictionary for separate synaptic pathways, organized as paths{path name}{parameters}
    cell1       : instance of cell class containing geometry and synapse structures (contain all hoc section and synapse objects)

    ===Out===

    ===Updates===
    -p          : updated path dictionaries are added to the main parameter dictionary p (organized as p['p_path']{path name}{parameters})
    -paths      : active synapses and their activity patterns are set and updated for each path

    ===Comments===
    -p  : p should have an entry called 'path_combo', which is a list of paths to be activated during the current simulation
    '''
    
    # update tstop for each combination
    tstops=[]
    warmups=[]
    for path_key, path in paths.iteritems():
        if path_key in p['path_combo']:
            warmups.append(path['warmup'])
            tstops.append(path['warmup'] + 1000*(path['bursts']-1)/path['burst_freq'] + 1000*(path['pulses']+1)/path['pulse_freq'] )
    p['tstop'] = max(tstops)
    p['warmup'] = min(warmups)
    p['field_off'] = p['tstop']

    p['p_path']={}
    
    # update pathways and add to global p structure
    for path_key, path in paths.iteritems():

        # print path_key, ' warmup:', path['warmup']
        
        # if path is included in current combination
        if path_key in p['path_combo']:
            
            # choose active segments for this pathway
            seg_dic = p_class.choose_seg_rand(trees=path['trees'], syns=cell1.syns, syn_frac=path['syn_frac'], seg_dist=p['seg_dist'], syn_num=path['syn_num'], distance=path['syn_dist'], replace=path['replace_syn'])
            path.update(seg_dic)
            
            # set weights for each segment in this pathway
            w_dic = p_class.set_weights(seg_idx=path['seg_idx'], sec_idx=path['sec_idx'], sec_list=path['sec_list'], seg_list=path['seg_list'], w_mean=path['w_mean'], w_std=p['w_std'], w_rand=p['w_rand'])
            path.update(w_dic)

            # set delays for branch sequences in this pathway 
            delay_dic = p_class.set_branch_sequence_ordered(seg_idx=path['seg_idx'], delay=path['sequence_delay'], direction=path['sequence_direction'])
            path.update(delay_dic)

            # update p_path dictionary in global p dictionary
            p['p_path'][path_key]=copy.copy(path)

class ExperimentsParallel:
    """ Organize parameters for distributing to multiple processors
    
    Contains a function corresponding to each experiment in Experiments class. 

    Arguments:
    experiment= experiment number to be run.  should be a string of the form 'exp_num', e.g. 'exp_4a'

    Each function ExperimentsParallel.exp_num() should output a list of parameter dictionaries. Each element (dictionary) in the list will sent to a different worker/processor.  Parameters should be designed so that simulations on each worker take about the same time (ie load balnced). Specific entries in the dictionary will depend on the details of the experiment. The dictionary will be passed to the corresponding Experiments.exp_num() funtion as **kwargs

    Once the list of parameter dictionaries is designed, the experiment can be run from command line with the syntax:

    _run_parallel(ExperimentsParallel('exp_num', **kwargs).parameters)    
    """
    def __init__(self, experiment, **kwargs):
        """ choose experiment to run and pass kwargs
        """
        
        # retrieve which experiment to run
        experiment_function = getattr(self, experiment)

        if kwargs:
            kwargs['experiment']=experiment
            experiment_function(**kwargs)

        else:
            kwargs={'experiment':experiment}
            experiment_function(**kwargs)

    def quick_run(self, **kwargs):
        self.parameters=[]
        n_workers=10
        for i in range(n_workers):
            self.parameters.append({'experiment':kwargs['experiment']})

        return self.parameters

    def exp_3a1(self, **kwargs):
        self.parameters = []
        n_workers = 10
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3a1_basal(self, **kwargs):
        self.parameters = []
        n_workers = 10
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3a2(self, **kwargs):
        self.parameters = []
        n_workers = 10
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3a2_basal(self, **kwargs):
        self.parameters = []
        n_workers = 1
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3a2_dose(self, **kwargs):
        self.parameters = []
        n_workers = 1
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3a3(self, **kwargs):
        self.parameters = []
        n_workers = 10
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_3b1(self, **kwargs):
        self.parameters = []
        n_workers = 10
        trials_per_worker=1
        for i in range(n_workers):
            self.parameters.append(
                {'experiment':kwargs['experiment'],
                'trials':trials_per_worker})
        return self.parameters

    def exp_4a(self, **kwargs):
        """ Associative plasticity experiment

        Activate a strong (100 Hz burst) and weak (single pulse in the middle of burst) pathway (separately or together).
        Vary which pathways are active (strong only, weak only, strong+weak)
        Vary the distance from soma of each pathway (0-200,200-400,400-600 um).
        Vary the number of synapses in each pathway (10, 20, 30, 40, 50).
        Synapse locations are generated randomly with param.choose_seg_rand(). Multiple synapses can be placed on the same segment.  The weight is then multiplied by the number of "synapses"
        
        Keyword Arguments:
        experiment= specifies which experiment to run in Experiments class

        Return a list of parameter dictionaries.
        Each dictionary contains a different synapse distance requirement (syn_dist), i.e. synapse distance varies across workers.  All other parameters that vary are passed as a list to each worker and are iteratede within each worker
        """
        experiment= kwargs['experiment']
        syn_num = [[12, 12],]
        paths = ['strong','weak']
        path_combos = ['strong','weak','strong_weak']
        syn_dists_strong = [[0,200],[200,400],[400,600]]
        syn_dists_weak = [[0,200],[200,400],[400,600]]
        # syn_dists_strong = [[0,200]]
        # syn_dists_weak = [[0,200]]
        trials=10
        w_mean=.001
        # distribute parameters
        self.parameters = []
        for syn_dist_i, syn_dist_val_strong in enumerate(syn_dists_strong):
            for syn_dist_i, syn_dist_val_weak in enumerate(syn_dists_weak):

                self.parameters.append({
                    'experiment':experiment,
                    'syn_num': syn_num,
                    'paths': paths,
                    'path_combos': path_combos,
                    'trials': trials, 
                    'w_mean': w_mean,
                    'syn_dist': [syn_dist_val_strong, syn_dist_val_weak]
                    })

        return self.parameters

# function to pass to parallel context message board
def _f_parallel(parameters):
    """ Wrap experiment function so it exists in global namespace

    Arguments: 
    parameters - dictionary with entries 'experiment' and parameters to be passed to Experiment.exp_num.  'experiment' should be of the form 'exp_4a' and specifies which experiment to run
    """
    # get experiment info
    experiment = parameters['experiment']
    
    # create experiment class instance
    exp_instance = Experiment()

    # get specific experiment function
    f = getattr(exp_instance, experiment)

    print f
    print parameters
    # run experiment
    return f(**parameters)

# function for controlling parallel    
def _run_parallel(parameters):
    """ Standard run procedure for parallel simulations

    Arguments:
    parameters= must be a list of parameter dictionaries.  Each dictionary in the list is passed to a different worker as the arguments for Experiment.exp_num

    Use ExperimentsParallel.exp_num to design parameters list

    Arguments:
    parameters= must be a list of parameter dictionaries.  Each dictionary in the list is passed to a different worker as the arguments for Experiment.exp_num

    Use ExperimentsParallel.exp_num to design parameters list

    To use multiple workers, python script must be called from the interpreter using syntax:
    'mpiexec -n 10 python script.py'
    the call to mpiexec initializes the mpi with 10 workers

    _run_parallel should be called as:
    if __name__=="__main__":
        _run_parallel(ExperimentsParallel('exp_4a').parameters)
    """

    # make parallel context global
    global pc

    print parameters
    # create parallel context instance
    pc = h.ParallelContext()

    print 'i am', pc.id(), 'of', pc.nhost()
    # start workers, begins an infinitely loop where master workers posts jobs and workers pull jobs until all jobs are finished
    pc.runworker()
    
    # print len(parameters)
    # # # distribute experiment and parameters to workers
    for param in parameters:
        # print len(parameters)
        # print param
        pc.submit(_f_parallel, param)
        # print param

    # # continue runnning until all workers are finished
    while pc.working():
        print pc.id(), 'is working'

    # # close parallel context 
    pc.done()




if __name__ =="__main__":
    # Experiment(experiment='exp_3a1', load_fd=False, trials=1)
    _run_parallel(ExperimentsParallel('exp_3a2_dose').parameters)
    # kwargs = {'experiment':'exp_1a'}
    # analysis.Experiment(**kwargs)
    # kwargs = {'experiment':'exp_1c'}
    # kwargs = Arguments('exp_1').kwargs
    # x = Experiment(**kwargs)
    # analysis.Experiment(**kwargs)
    # plots = analysis.PlotRangeVar()
    # plots.plot_all(x.p)
    # analysis.Experiment(exp='exp_3')
