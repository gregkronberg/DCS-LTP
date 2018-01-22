

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
        for key, val in self.kwargs.iteritems():        # update parameters
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

    def exp_1a(self, **kwargs):
        """ 
        activate a varying number of synapses at varying frequency with varying distance from the soma.  Synapses are chosen from a window of 50 um, with the window moving along the dendrite.  Number of synapses is increased.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of mean distance from the soma and frequency of synaptic activity

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
        for key, val in self.kwargs.iteritems():        # update parameters
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
        for key, val in self.kwargs.iteritems():        # update parameters
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

    def exp_2e(self, **kwargs):
        """ choose a random subset of segments (fixed number of segments) within a given distance from the soma and apply theta burst

        vary number of synapses, weight of each synapse, and distance from soma.  For each combination of these variables, do several repeats to randomly choose specific synapse locations 

        run simulations in parallel.  distribute distance and number of synapses across workers. iterate over weights and random drawings within a given worker

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
        'experiment' : 'exp_2e', 
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
        'clopath_tau_r':10,
        'gna_inact': .5
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

        # iterate over all segments in tree
        for syn_num_i, syn_num in enumerate(syn_nums):
            for trial_i, trial in enumerate(range(self.p['trials'])):
                self.p['syn_num']=syn_num
                self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'])

                self.p['w_mean'] = self.p['nsyns']*w_mean

                self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

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
                '_weight_'+str(self.p['w_mean'])+
                '_trial_'+str(self.p['trial'])+
                '_dist_'+str(self.p['syn_dist'][-1])+
                '_syn_num_'+str(self.p['syn_num'])+
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

    def exp_2f(self, **kwargs):
        """ choose a random subset of segments (fixed number of segments) within a given distance from the soma and apply theta burst

        vary number of synapses, weight of each synapse, and distance from soma.  For each combination of these variables, do several repeats to randomly choose specific synapse locations 

        run simulations in parallel.  distribute distance and number of synapses across workers. iterate over weights and random drawings within a given worker

        same as 2e, added sodium slow inactivation, shortened clopath tau_r time constant

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
        'experiment' : 'exp_2f', 
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
        'gna_inact': .5
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

        # iterate over all segments in tree
        for syn_num_i, syn_num in enumerate(syn_nums):
            for trial_i, trial in enumerate(range(self.p['trials'])):
                self.p['syn_num']=syn_num
                self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'])

                self.p['w_mean'] = self.p['nsyns']*w_mean

                self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

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
                '_weight_'+str(self.p['w_mean'])+
                '_trial_'+str(self.p['trial'])+
                '_dist_'+str(self.p['syn_dist'][-1])+
                '_syn_num_'+str(self.p['syn_num'])+
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

    def exp_2g(self, **kwargs):
        """ choose a random subset of segments (fixed number of segments) within a given distance from the soma and apply theta burst

        vary number of synapses, weight of each synapse, and distance from soma.  For each combination of these variables, do several repeats to randomly choose specific synapse locations 

        run simulations in parallel.  distribute distance and number of synapses across workers. iterate over weights and random drawings within a given worker

        same as 2f, expanded distance range for activated synapses (e.g. from 0 to 200 um from soma)

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
        'experiment' : 'exp_2g', 
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
        'gna_inact': .4
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

        # iterate over all segments in tree
        for syn_num_i, syn_num in enumerate(syn_nums):
            for trial_i, trial in enumerate(range(self.p['trials'])):
                self.p['syn_num']=syn_num
                self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'])

                self.p['w_mean'] = self.p['nsyns']*w_mean

                self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

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
                '_weight_'+str(self.p['w_mean'])+
                '_trial_'+str(self.p['trial'])+
                '_dist_'+str(self.p['syn_dist'][-1])+
                '_syn_num_'+str(self.p['syn_num'])+
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

    def exp_2h(self, **kwargs):
        """ choose a random subset of segments (fixed number of segments) within a given distance from the soma and apply theta burst

        vary number of synapses, weight of each synapse, and distance from soma.  For each combination of these variables, do several repeats to randomly choose specific synapse locations 

        run simulations in parallel.  distribute distance and number of synapses across workers. iterate over weights and random drawings within a given worker

        same as 2f 2g, but with 20 Hz synaptic activity

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
        'experiment' : 'exp_2h', 
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
        'pulses':10,
        'pulse_freq':20,
        'group_trees':False,
        'plot_variables':['v','gbar'],
        'cell':[],
        'tstop':530,
        'clopath_tau_r':8,
        'gna_inact': 0.,
        'field_off':520
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

        # iterate over all segments in tree
        for syn_num_i, syn_num in enumerate(syn_nums):
            for trial_i, trial in enumerate(range(self.p['trials'])):
                self.p['syn_num']=syn_num
                self.p_class.choose_seg_rand(trees=self.p['trees'], syns=cell1.syns, syn_frac=0., seg_dist=self.p['seg_dist'], syn_num=syn_num, distance=self.p['syn_dist'])

                self.p['w_mean'] = self.p['nsyns']*w_mean

                self.p_class.set_weights(seg_idx=self.p['seg_idx'], w_mean=self.p['w_mean'], w_std=self.p['w_std'], w_rand=self.p['w_rand'])

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
                '_weight_'+str(self.p['w_mean'])+
                '_trial_'+str(self.p['trial'])+
                '_dist_'+str(self.p['syn_dist'][-1])+
                '_syn_num_'+str(self.p['syn_num'])+
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
    
# function to pass to parallel context message board
def _f_parallel(parameters):
    """ wrapper around experiment so that it exists in global namespace and can be accessed by mpi

    Arguments: 
    parameters - dictionary with entries 'experiment' and parameters to be iterated through.  'experiment specifies which experiment to run'
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
def _run_parallel(experiment):
    """ set up for running parallel simulations
    python script must be called from the interpreter using syntax

    mpiexec -n 10 python script.py

    the call to mpiexec initializes the mpi with 10 workers
    """

    # make parallel context global
    global pc

    # choose experiment
    experiment =experiment

    # divide up parameters
    # nsyns = np.arange(2.,4.,1.)
    nsyns = range(1, 2, 1)
    syn_nums = np.arange(26., 32.,2. )
    stim_freqs = [20,50,100,200]
    syn_dists = [[0, 50], [50, 100], [100, 150], [150, 200], [200, 250],[250,300],[300, 350], [350,400], [400,450],[450,500],[500,550],[550,600]]
    trees=['apical_tuft','apical_trunk']
    
    # list of parameter values to be sent to each worker [worker number]{'parameter':[values], 'experiment':exp_number}
    parameters=[]
    syns=nsyns
    for syns in nsyns:
        for syn_dist in syn_dists:
            parameters.append({'nsyns':syns, 'experiment':experiment, 'syn_num':syn_nums, 'syn_dist':syn_dist, 'trees':trees, 'stim_freqs':stim_freqs})

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
        'experiment' : 'exp_1a', 
        'tree' : ['apical_tuft','apical_trunk'],
        'trials' : 1,
        'w_mean' : weights,#[.001],
        'w_std' : [.002],
        'w_rand' : False, 
        'syn_frac' : .2
        }

if __name__ =="__main__":
    _run_parallel(experiment='exp_1a')
    # kwargs = {'experiment':'exp_2c'}
    # kwargs = Arguments('exp_1').kwargs
    # x = Experiment(**kwargs)
    # analysis.Experiment(**kwargs)
    # plots = analysis.PlotRangeVar()
    # plots.plot_all(x.p)
    # analysis.Experiment(exp='exp_3')
