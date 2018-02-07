

"""
run control
"""
# imports
from mpi4py import MPI
import multiprocessing 
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
        if not kwargs:
            pass
        else:
            # retrieve which experiment to run
            experiment = getattr(self, kwargs['experiment'])

            # run experiment
            experiment(**kwargs) 

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
                    self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'], replace=True)
                    # update weight based on number of synapses per segment (always 1 in this experiment)
                    self.p['w_mean'] = self.p['nsyns']*w_mean

                    # create nested list of weights to match seg_idx structure
                    self.p_class.set_weights(seg_idx=self.p['seg_idx'], sec_idx=self.p['sec_idx'], sec_list=self.p['sec_list'], seg_list=self.p['seg_list'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

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

    Frequency dependence (compare 20 Hz to TBS)
    """
    def exp_3a(self, **kwargs):
        """ Run full 20 Hz simulation (900 pulses, 45 seconds) with sodium channel inactivation.  Track backpropagation of action potentials
        """
        # print 'running', kwargs['experiment'], 'on worker', pc.id()
        print kwargs
        w_mean = .001 # weight of single synapse uS
        trees = ['apical_trunk', 'apical_tuft']
        nsyns = 1.
        syn_nums = [50.]
        syn_dist = [0,300]
        freqs=[20]
        trials = 4
        pulses=10
        self.kwargs = {
        'experiment' : 'exp_3a', 
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
        'pulses':pulses,
        'pulse_freq':20,
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
        for freq_i, freq in enumerate(freqs):
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
                
            # update parameters for each pathway
            paths['strong']['syn_num']= syn_num[0]
            paths['weak']['syn_num']= syn_num[1]

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

                    print combo

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


    def exp_4a(self, **kwargs):
        """ Associative plasticity experiment

        Activate a strong (100 Hz burst) and weak (single pulse in the middle of burst) pathway (separately or together).
        Vary which pathways are active (strong only, weak only, strong+weak)
        Vary the distance from soma of each pathway (0-200,200-400,400-600 um).
        Vary the number of synapses in each pathway (10, 20, 40).
        Synapse locations are generated randomly with param.choose_seg_rand(). Multiple synapses can be placed on the same segment.  The weight is then multiplied by the number of "synapses"
        
        Keyword Arguments:
        experiment= specifies which experiment to run in Experiments class

        Return a list of parameter dictionaries.
        Each dictionary contains a different synapse distance requirement (syn_dist), i.e. synapse distance varies across workers.  All other parameters that vary are passed as a list to each worker and are iteratede within each worker
        """
        experiment= kwargs['experiment']
        syn_num = [[10,10],[20,20],[40,40]]
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

    # run experiment
    return f(**parameters)

# function for controlling parallel    
def _run_parallel(parameters):
    """ Standard run procedure for parallel simulations

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
    # _run_parallel(experiment='exp_1c')
    _run_parallel(ExperimentsParallel('exp_4a').parameters)
    # kwargs = {'experiment':'exp_1a'}
    # analysis.Experiment(**kwargs)
    # kwargs = {'experiment':'exp_1c'}
    # kwargs = Arguments('exp_1').kwargs
    # x = Experiment(**kwargs)
    # analysis.Experiment(**kwargs)
    # plots = analysis.PlotRangeVar()
    # plots.plot_all(x.p)
    # analysis.Experiment(exp='exp_3')
