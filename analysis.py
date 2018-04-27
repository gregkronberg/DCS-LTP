"""
analysis

data structure for each trial is organized as []['tree'][polarity][section][segment]
"""

from __future__ import division
import numpy as np
import scipy
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as it
import os
import glob
import cPickle as pickle
import param
import cell
import math
import run_control
import copy
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm as colormap


def _param2times(p_path, tree_key, sec_i, seg_i, t):
        """ Get synaptic input times from parameter dictionary for a given segment
        
        ===Args===
        -p_path     :   parameter dictionary for a given pathway
            -must contain the following entries
                -pulses : number of pulses per burst
                -pulse_freq : frequency of inputs within a burst (Hz)
                -bursts : number of bursts
                -burst_freq : frequency of bursts (Hz)
                -warmup : simulation warmup time to allow steady state (ms)
                -delay : delay (after warmup) for current segment to allow sequences (ms). 
                -delay of 0 corresponds to all input patterns starting after warmup 
        -tree_key   : tree containing the desired segment
        -sec_i      : section index containing the desired segment (matching indexing in p_path['seg_idx'])
        -sec_i      : index of desired segment (matching indexing in p_path['seg_idx'])
        -t          : time vector for the simulation
        
        ===Out===
        -input_times : list of input times in ms, length (dim: number of input pulses)
        -x           : boolean vector of input times with same length as t (dim: samples)
        
        ===Comments===
        
        """
        # get parameters from dictionary
        #```````````````````````````````````````````````````````````````
        pulses = int(p_path['pulses'])
        pulse_freq = p_path['pulse_freq']/1000
        burst_freq = p_path['burst_freq']/1000
        bursts  =int(p_path['bursts'])
        warmup = p_path['warmup']
        delay = p_path['sequence_delays'][tree_key][sec_i][seg_i]
        
        # array for storing input times
        input_times = np.zeros((1, bursts*pulses))
        
        # define a single burst 
        burst1 = np.arange(0., pulses/pulse_freq, 1./pulse_freq)
        
        # iterate through bursts
        for burst_i in range(bursts):
            
            # add times to array
            input_times[0,burst_i:burst_i+pulses] = warmup + delay + burst_i/burst_freq + burst1
            
        # convert to boolean vector (dim: samples)
        x = np.zeros(t.shape)
        # cut off number of decimals to prevent rounding error
        t = np.around(t, 4)
        # find input times and to boolean vector
        for t_i, t_t in enumerate(input_times[0,:]):
            
            # store input times
            x[np.where(t==t_t)]=1
                
        return input_times, x

def _seg_zip(data):
    """ From nested data structure, get list of all segments and corresponding conditions
    
    ===Args===
    -data : data structure organized as data{'polarity'}{'pathway'}{'tree_variable'}[section number][segment number][time series]
    
    ===Out===
    -seg_list : list of tuples.  each tuple contains (polarity_key, path_key, tree_key, sec_i, seg_i).  list has length equal to the total number of recorded segments in the data structure
    
    ===Comments===
    -used in _data2mat to store list of attributes for each data time series
    """
    # parameter structure
    p = data['p']
    # for storing list of all recorded segments
    seg_list = []
    for polarity_key, polarity in data.iteritems():
        # exclude parameter dictionary
        if polarity_key != 'p':        
            # iterate over pathways
            for path_key, path in polarity.iteritems():
                # get path-specific parameters
                p_path = p['p_path'][path_key]  
                # iterate over trees
                for tree_key, tree in p_path['seg_idx'].iteritems():                
                    # exclude time vectors
                    if tree_key!='t':                       
                        # iterate over sections in tree
                        for sec_i, sec in enumerate(tree):                        
                            # iterate over segments in section
                            for seg_i, seg in enumerate(sec):
                                # add tuple to segment list
                                seg_list.append((polarity_key, path_key, tree_key, sec_i, seg_i))
    return seg_list
    
def _data2mat(data, variable):
    """ Convert data structure to single matrix
    
    ===Args===
    -data : data structure organized as data{'polarity'}{'pathway'}{'tree_variable'}[section number][segment number][time series]
    -variable : string naming a variable to enter in matrix (eg 'v' for membrane voltage)
    
    ===Out===
    -data_mat : matrix of time series data for specified variable (segments x samples)
    -input_mat : boolean matrix with ones for input times (segments x samples)
    -conditions  : dictionary of condition types ('polarity', 'path' etc). each dictionary entry is a list with indices matching the fist dimension of mat
    -p : parameter dictionary for the simulations in data
    -t : single time vector for all data (ms), all simulations in a given data file have the same time vector
    
    ===Comments===
    -This is very sensitive to the organizaiton of the input data structure
    
    """
    # get parameter dictionary
    p = data['p']
    # get time vector (all time vectors are the same within a given data file, same simulation)
    t = data[data.keys()[0]].values()[0]['t']
    # get total number of recorded segments
    nsegs = len(_seg_zip(data))
    # preallocate output arrays (segments x samples)
    data_mat = np.zeros((nsegs, len(t)))
    input_mat = np.zeros((nsegs, len(t)))
    # dictionary to store conditions for each segment ['condition type'][segment number]
    conditions = {'polarity':[],
                  'path':[],
                  'tree':[],
                  'sec_i':[],
                  'sec_num':[],
                  'seg_i':[],
                  'seg_num':[],
                  'input_times':[],
                  }
    
    # get data for each segment
    #`````````````````````````````````````````````````````````````````````
    # data = {'polarity'}{'pathway'}{'tree_variable'}[section number][segment number][time series]
    # iterate over polarities
    seg_cnt=-1
    for polarity_key, polarity in data.iteritems():

        # exclude parameter dictionary
        if polarity_key != 'p':
            
            # iterate over pathways
            for path_key, path in polarity.iteritems():
                        
                # get path-specific parameters
                p_path = p['p_path'][path_key]
                
                # iterate over trees
                for tree_key, tree in p_path['seg_idx'].iteritems():
                    
                    # exclude time vectors
                    if tree_key!='t':   
                        
                        # iterate over sections in tree
                        for sec_i, sec in enumerate(tree):
                            
                            # get section number
                            sec_num = p_path['sec_idx'][tree_key][sec_i]
                            
                            # iterate over segments in section
                            for seg_i, seg in enumerate(sec):
                                
                                seg_cnt += 1
                                # segment number
                                seg_num = seg
                                
                                # get input times from parameter dictionary
                                # input_times: list of all input times in ms
                                # x: boolean vector. Same dimensions as t. 1=input, 0=no input
                                input_times, x = _param2times(p_path, tree_key, sec_i, seg_i, t)
                                
                                # reshape x to 2D row vector
                                x = x.reshape(1, x.shape[0])
                                
                                # add x to input matrix (boolean, dim: segments x samples)
                                input_mat[seg_cnt,:] = x
                                
                                # get time series data for current segment
                                vec =  path[tree_key+'_'+variable][sec_i][seg_i]
                                # reshape to 2D row vector
                                vec = vec.reshape(1, vec.shape[0])
                                # add to data matrix (dim: segments x samples)
                                data_mat[seg_cnt,:] = vec
                                
                                # update conditions matrix ['condition type'][segment number]
                                conditions['polarity'].append(polarity_key)
                                conditions['path'].append(path_key)
                                conditions['tree'].append(tree_key)
                                conditions['sec_i'].append(sec_i)
                                conditions['sec_num'].append(sec_num)
                                conditions['seg_i'].append(seg_i)
                                conditions['seg_num'].append(seg_num)
                                conditions['input_times'].append(input_times)

    return data_mat, input_mat, conditions, p, t

def _load_group_data(directory='', file_name=''):
        """ Load group data from folder
        
        ===Args===
        -directory : directory where group data is stored including /
        -file_name : file name for group data file, including .pkl
                    -file_name cannot contain the string 'data', since this string is used t search for individual data files

        ===Out===
        -group_data  : typically a dictionary.  If no file is found with the specified string, an empty dictionary is returned

        ===Updates===
        -none

        ===Comments===
        """
        
        # all files in directory
        files = os.listdir(directory)
        
        # if data file already exists
        if file_name in files:
            # load data
            print 'group data found:', file_name
            with open(directory+file_name, 'rb') as pkl_file:
                group_data= pickle.load(pkl_file)
            print 'group data loaded'
        # otherwise create data structure
        else:
            # data organized as {frequency}{syn distance}{number of synapses}{polarity}[trial]{data type}{tree}[section][segment][spikes]
            print 'no group data found'
            group_data= {}

        return group_data 

def _update_group_data(directory, search_string, group_data, variables):
    '''
    ===Args===
    -directory : directory where group data is stored including /
    -search_string : string that is unique individual data files to search directory.  typically '*data*'
    -group_data : group data structure organized as group_data{variable}{data_type}
            -group_data['v']['data_mat'] is a matrix with dimensions segments x samples

    ===Out===
    ===Updates===
    ===Comments===
    -group_data['processed'] = list of data file names that have already been added to the group 
    -variable = variable type that was recorded from neuron simulaitons, e.g. 'v' or 'gbar'
    -data_type for group_data structure:
        -'data_mat' : matrix of time series data for the specified variable
        -'input_mat': matrix of boolean time series of synpatic inputs for the corresponding data_mat (1=active input, 0=inactive input)
        -'conditions' : 
                -'polarity', 'path', 'tree', 'sec_i', 'sec_num', 'seg_i', 'seg_num', 'input_times'
                -e.g. group_data['v']['conditions']['polarity'] returns a list of field polarities with indices corresponsponding to rows group_data['v']['data_mat']
        't' : single 0D vector of time values (should be identical for all simulations in a given group data structure)
    '''
    # get list of new data files
    #`````````````````````````````````````````````````````````````
    # get list of all data files
    data_files = glob.glob(directory+search_string)

    # get list of processed data files
    if group_data:
        processed_data_files = group_data['processed']
    else:
        group_data['processed']=[]
        processed_data_files = group_data['processed']

    # get list of new data files
    new_data_files = list(set(data_files)-set(processed_data_files))
    print 'total data files:', len(data_files) 
    print 'new data fies:', len(new_data_files)
    
    # iterate over new files and update group_data structure
    #`````````````````````````````````````````````````````````````````
    print 'updating group data structure'
    # dictionary for temporary storage of individual simulation data
    dtemp={}
    # iterate over new data files
    for file_i, file in enumerate(new_data_files):

        # load data file
        with open(file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # iterate over variables to be updated
        for variable_i, variable in enumerate(variables):

            # convert data to matrix and get corresponding conditions
            dtemp['data_mat'], dtemp['input_mat'], dtemp['conditions'], dtemp['p'], dtemp['t'] = _data2mat(data, variable)


            # add to group data
            #``````````````````````````````````````````````````````````````
            # if variable does not exist in group_data
            if variable not in group_data:
                # create entry
                group_data[variable]={}
                # iterate through data types for individual simulation
                for key, val in dtemp.iteritems():
                    # set initial value in group_data to the individual value
                    group_data[variable][key]=val

            # if variable already exists in group_data structure
            else:
                # iterate through data types
                for key, val in dtemp.iteritems():
                    # for conditions info
                    if key=='conditions':
                        # iterate through the various conditions (e.g. 'polarity', 'path', 'tree', 'sec_i', 'seg_i')
                        for condition_key, condition_list in val.iteritems():
                            # add list for individual data to running list for group
                            group_data[variable][key][condition_key] += condition_list 
                    # if data type is a matrix
                    elif 'mat' in key:
                        # add to existing group matrix
                        group_data[variable][key] = np.append(group_data[variable][key], val, axis=0)
                    
                    # FIXME this is overwritten on each loop (should only be set once)
                    # create single time vector in group_data
                    elif key=='t':
                        group_data[variable][key] = val

            # add file to processed list to keep track of processed files
            group_data['processed'].append(file)

    print 'finished updating group data structure'

    # ouput group data structure 
    return group_data

def _save_group_data(group_data, directory, file_name):
    '''
    ===Args===
    ===Out===
    ===Updates===
    ===Comments===
    '''
    with open(directory+file_name, 'wb') as output:
        pickle.dump(group_data, output,protocol=pickle.HIGHEST_PROTOCOL)

    print 'group data saved as:', file_name 

def _plot_weights_mean(group_data, condition, condition_vals, colors):
    '''
    '''
    fig = plt.figure()
    for condition_i, condition_val in enumerate(condition_vals):
        # get indeces that match the conditions
        # print group_data['gbar']
        indices = [i for i,val in enumerate(group_data['gbar']['conditions'][condition]) if val==condition_val]
        color = colors[condition_i]
        w_mean = np.mean(group_data['gbar']['data_mat'][indices,:],axis=0, keepdims=True).transpose()
        w_std = np.std(group_data['gbar']['data_mat'][indices,:],axis=0, keepdims=True).transpose()
        t_vec = group_data['gbar']['t']
        t_vec.reshape(len(t_vec), 1)

        plt.plot( t_vec, w_mean, color=colors[condition_i],)
        plt.xlabel('time (ms)')
        plt.ylabel('weight (AU)')


    return fig

class Clopath():
    
    """ Apply Clopath learning rule to membrane voltage data
    
    ===Methods===
    
    ===Variables===
    """
    
    def __init__(self):
        """
        """
        pass
        
    def _run_clopath_individual(self,directory='Data/quick_run/',search_string='data*syn_num_2*',):
        """ Load data from a single simulation and apply learning rule
        
        ===Args===
        
        ===Out===
        
        ===Comments===
        """
        data_files = glob.glob(directory+search_string)
        data_file = data_files[1]
        with open(data_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        p = data['p']
        print data_file
        self.data_mat, self.input_mat, self.conditions, self.p, self.t = self._load_data(directory='', data_file=data_file)
        self.w = self._clopath(x=self.input_mat, u=self.data_mat, )
        fig_v = PlotRangeVar().plot_trace(data=data,
                            sec_idx=p['sec_idx'],
                            seg_idx=p['seg_idx'],
                            variables=p['plot_variables'],
                            x_variables=p['x_variables'],
                            file_name='',
                            group_trees=p['group_trees'],
                            xlim=[p['warmup']-5,p['tstop']],
                            ylim=[-70,-45])
        fig = self._plot_clopath(self.w, self.conditions)
        fig.savefig(directory+'_weights.png',dpi=250)
    
    def _plot_clopath(self, w, conditions):
        """ Plot weight changes
        """
        pols = ['-20.0','0.0','20.0']
        colors = ['blue','black','red']
        fig = plt.figure()
        nsyns = w.shape[0]
        for syn in range(nsyns):
            polarity = conditions['polarity'][syn]
            color = colors[pols.index(polarity)]
            plt.plot(w[syn,:],color=color)
        plt.xlabel('time')
        plt.ylabel('weight')
        plt.title('weight dynamics during single theta burst')
        
        return fig
        
    def _load_data(self, directory='Data/', param_file='fd_parameters.pkl',data_file='induction_slopes_all.mat', ):
        """
        
        ===Args===
        
        ===Out===
        """
        
        with open(directory+data_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            
        data_mat, input_mat, conditions, p, t = _data2mat(data, 'v')
        
        return data_mat, input_mat, conditions, p, t
            
    def _clopath(self, x, u, fs=40, w0=0.5, homeostatic=False, param={}):
        """ Determine weights from voltage traces and input times with Clopath rule
        
        ===Args===
        -x      :   input vector of spike times (1 for spike, 0 for no spike)
        -u      :   array of voltage time traces (compartments x time)
        -fs     :   sampling rate (kHz)
        -w0     :   initial weight 
        -param  :   dictionary of clopath parameters
        -homeostatic : if True, homeostatic term is included in LTD equation
        
        ===Out===
        -w  :    vector of weights
        
        ===Comments===
        -see Clopath et al. 2010 Nature Neuroscience for details of learning rule
        """
        
        # preallocate parameter dictionary
        p = {}
        
        # get number of synapses (to determine shape of preallocated arrays)
        n_syn   = u.shape[0]
        
        # create time vector
        #`````````````````````````````
        print 'data shape:', u.shape
        dt      = 1./fs
        T       = u.shape[1]/fs
        dur_samples = int(T*fs)
        time    = np.arange(0., T, dt)
        
        # plasticity parameters
        #```````````````````````````````````````````````````````````````````
        p['u_ref']      = 9         # reference value for homeostatic process mV
        p['adapt_t']    = 1000*fs   # time to integrate over for homeostatic process (samples)
        p['A_m0']       = 2E-5      # A_m: amplitude for the depression
        p['A_p']        = 38E-5     # A_p: amplitude for the potentiation
        p['tau_p']      = 5         # tau_p: time constant for voltage trace in the potentiation term [ms]
        p['tau_x']      = 10       # tau_x: time constant for presynaptic trace [ms]
        p['tau_m']      = 6         # tau_m: time constant for voltage trace in the depression term [ms]
        p['tetam']      = -60       # tetam: low threshold
        p['tetap']      = -53       # tetap: high threshold in the potentiation term
        p['E_L']        = -70.6     # resting potential [mV], used for LTD adaptation
        p['delay']      = 0        # conduction delay after action potential (ms)
        
        
        # update with parameters passed as arguments
        for key, val in param.iteritems():
            p[key] = val
        
        # preallocate learning rule variables
        #``````````````````````````````````````````````````````````````````````
        self.w           = w0*np.ones((n_syn,dur_samples))       # weights
        self.u_md        = p['E_L']*np.ones((n_syn,dur_samples)) # membrane potential for the depression
        self.u_mdbar     = np.zeros((n_syn,dur_samples))         # homeostatic membrane potential parameter
        self.u_mp        = p['E_L']*np.ones((n_syn,dur_samples)) # membrane potential for the potentiation
        self.u_sig       = p['E_L']*np.ones((n_syn,dur_samples)) # thresholded membrane potential
        self.u_md_sig    = np.zeros((n_syn,dur_samples))         # filtered and thresholded (thetam) membrane potential
        self.u_mp_sig    = np.zeros((n_syn,dur_samples))         # filtered and thresholded (thetap) membrane potential
        self.x_m0        = np.zeros((n_syn,dur_samples))         # presynaptic trace
        self.A_m         = p['A_m0']*np.ones((n_syn,dur_samples))# homeostatic LTD parameter

        # apply learning rule
        #`````````````````````````````````````````````````````````````````````` 
        # main loop
        for i, t in enumerate(time):
            
            # start simulation after specified delay
            if t>p['delay']:
                
                
                             
                # if include homeostatic LTD mechanism
                if homeostatic:
                    
                    # adaptation voltage
                    if t <= p['adapt_t']:
                        self.u_mdbar[:,i]   = np.mean(self.u_md[:,1:i]-p['E_L'],axis=1)
                    else:
                        self.u_mdbar[:,i]   = np.mean(self.u_md[:,i-p['adapt_t']:i-1]-p['E_L'],axis=1)
                
                    # homeostatic modulation of LTD rate based on adaptation voltage
                    self.A_m[:,i]   = p['A_m0']*( self.u_mdbar[:, i-1] **2) /( p['u_ref']**2)   
    
                else:
                    # constant LTD rate
                    self.A_m[:,i]   = p['A_m0']

                # trace of membrane potential (u) with time constant tau_d
                self.u_md[:,i]  = self.u_md[ :, i-1] + dt* ( u[ :, i-1]-self.u_md[ :, i-1])/p['tau_m']
                      
                # trace of membrane potential (u) with time constant tau_p
                self.u_mp[:,i]  = self.u_mp[:, i-1]  +dt*( u[:,i-1]-self.u_mp[:, i-1]) /p['tau_p']
                 
                # trace of input spike train (spikes0)
                self.x_m0[:,i]  = self.x_m0[:,i-1]  +dt*(x[:,i-1]-self.x_m0[:,i-1]) /p['tau_x']
                
                # membrane potential (u) thresholded by thetap
                self.u_sig[:,i] = (u[:,i] > p['tetap']) *( u[:,i] -p['tetap'])
                
                # membrane potential trace (u_md) thresholded by thetam (taken 3ms before since time of a spike is 2ms)
                self.u_md_sig[:,i]  = ( self.u_md[:, i-p['delay']*fs] > p['tetam']) *( self.u_md[:, i-p['delay']*fs] -p['tetam'])                  
                
                # membrane potential trace (u_md) thresholded by thetam (taken 3ms before since time of a spike is 2ms)
                self.u_mp_sig[:,i]  = ( (self.u_mp[:,i-p['delay']*fs] -p['tetam']) > 0) *(self.u_mp[:,i-p['delay']*fs] -p['tetam'])
                
                # update weights
                self.w[:,i] = self.w[:,i-1] - self.A_m[:,i] *self.u_md_sig[:,i] *x[:,i] + p['A_p']*self.u_sig[:,i] *self.u_mp_sig[:,i] *self.x_m0[:,i]
        
        self.p=p
        # output weight matrix (synapses x time)
        return self.w

class FitFD():
    """ Functions for fitting facillitation and depression parameters to train of synaptic inputs
    
    Based on FD model in Varela et al. 1997 Journal of Neuroscience
    """
    
    #__________________________________________________________________________
    def __init__(self, run_opt=False, ):
        """ Define optimization parameters here
        
        ---
        Args
        ---
        run_opt = if True, parameter optimization will be called on class instantiation
        """
        
        # run optimization
        #``````````````````````````````````````````````
        if run_opt:
            
            param0 = [.8,.1,
                      .999,.2,
                      .999,100,]
            bounds = [(0, 20), (0,.5), 
                     (0,1), (0,1),
                     (0, 1), (0, 100),
                     (.9, 1), (1, 2000),]
            self._fd_opt(param0=param0, 
                        bounds=bounds)
            
        else:
            pass

    #__________________________________________________________________________
    def _plot_fd_fromfile(self, 
                          directory='Data/', 
                          param_file='fd_parameters.pkl', 
                          data_file='induction_slopes_all.mat', 
                          plot_file='fd_fitted_traces.png',
                          fit_color='red', 
                          data_color='black'):
        
        """ Plot data and fd fit from saved files
        ---
        Args
        ---
        directory = folder for parameters and data
        param_file =  file name for fd parameter list
        data_file = file name for data to be fit
            -matlab file with a matrix for tetanus at different frequencies, x_tet = [input times x tetanus frequency]
            and a vector for tbs called x_tbs = [input times]
        
        ---
        Out
        ---
        fig = plot with data and fd fit
        
        -saved
            fig
        """
        
        # load parameters
        #`````````````````
        with open(directory+param_file, 'rb') as pkl_file:
            param_obj = pickle.load(pkl_file)
            
        params = param_obj.x
        
        print params
        
        # load data
        #````````````````
        self.data = scipy.io.loadmat(directory+data_file)
        
        t_data_list = [self.data['t_tbs'], self.data['t_tet'][:,-1]]
        y_data_list = [self.data['x_tbs'], self.data['x_tet'][:,-1]]

        # plot data and fit
        #`````````````````````
        fig = plt.figure()
        
        # iterate through input streams
        for i, t_data in enumerate(t_data_list):
            
            # get time series of amplitudes from FD parameters
            fit_opt = self._fd(params, t_data,)
            
            # plot data and fit
            plt.plot(t_data, y_data_list[i], color=data_color)
            plt.plot(t_data, fit_opt, color=fit_color)
            
        # format and save figure
        #``````````````````````
        plt.xlabel('time (s)')
        plt.ylabel('Normalized slope')
        plt.show()
            
        fig.savefig(directory+plot_file, dpi=250)
        
        # output figure
        return fig
    
    #__________________________________________________________________________
    def _fd_opt(self, 
                param0, 
                bounds, 
                directory='Data/', 
                param_file='fd_parameters.pkl', 
                data_file='induction_slopes_all.mat', 
                plot_file='fd_fitted_traces.png',
                method='differential_evolution'):
        
        """ optimize fd parameters
        
        ---Args---
        param0 = list of initial parameter guesses ordered as [f,tF, d1,tD1, d2,tD2, d3,tD3,...]
        bounds = list of tuples with bounds for each parameter as [(min,max), (min,max)...] with a tuple for each parameter in param0
        directory = string for directory containing data and optimized parameters
        param_file = file name string for storing optimized parameters (including file type extension)
        data_file = file name string for storing data to fit parameters to
        plot_file = file name string for saving plot
        method = optimization algorithm
        
        ---Out---
        param_opt_obj = output object from parameter optimization algorithm (parameters stored as param_opt_obj.x)
        
        ---Write---
            -param_opt_obj = optimized parameters
            -plot of data and fitted fd curves as directory+plot_file
        """
        
        # load data
        #`````````````````````````
        self.data = scipy.io.loadmat(directory+data_file)
        
        t_data_list = [self.data['t_tbs'], self.data['t_tet'][:,-1]]
        y_data_list = [self.data['x_tbs'], self.data['x_tet'][:,-1]]
        
        # run optimization
        #```````````````````````````````
        if method == 'differential_evolution':
            param_opt_obj = scipy.optimize.differential_evolution(self._fd_err_mult, bounds=bounds, args=(t_data_list, y_data_list))
            
        else:
            param_opt_obj = scipy.optimize.minimize(self._fd_err_mult, param0, (t_data_list, y_data_list), bounds=bounds)

        # save optimization parameters
        with open(directory+param_file, 'wb') as output:
            pickle.dump(param_opt_obj, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'saved optimized parameters'
        
        self._plot_fd_fromfile(directory=directory, param_file=param_file, plot_file=plot_file)
        
        return param_opt_obj
        
    #__________________________________________________________________________
    def _fd_loop(self, params, t_data,):
        """ Iterate through times steps to get FD curves

        ---
        Arguments
        ---
        t_data = 1D array of times at which inputs arrive (s) 
        params = list of parameters, first two parameters are for facilitation 
        (f increment and time constant respectively), remaining parameters for depression

        ---
        Output
        ---
        fit = dictionary of 1D facilitation and depression values at each input time in t_data
            ---
            Keys
            ---
            params = list of parameters, first two parameters are for facilitation, remaining parameters for depression
            F = facilitation
            DN = depression, N is the Nth depression term, e.g. D1, D2, D3 etc 

        """
        fit = {'params':params}
        f = params[0]
        tF = params[1]
        fit['F'] = np.ones(t_data.shape)
        
        d1 = params[2]
        tD1 = params[3]
        fit['D1'] =np.ones(t_data.shape)
        
        if len(params)>4:
            d2 = params[4]
            tD2 = params[5]
            fit['D2'] = np.ones(t_data.shape)
            
            if len(params)>6:
                d3 = params[6]
                tD3 = params[7]
                fit['D3'] = np.ones(t_data.shape)
        
        # FIXME set up to receive variable number of FD parameters
        # iterate over input times
        for i, t in enumerate(t_data):

            # start from seccond input
            if i>0:

                # add increment f to previous pulse, then calculate exponential decay based on time interval
                fit['F'][i] = 1 + (fit['F'][i-1]+f-1.)*np.exp(-(t-t_data[i-1])/tF)
                
                # add decrement d to previous pulse, then calculate exponential decay based on time interval
                fit['D1'][i] = 1 - (1-fit['D1'][i-1]*d1)*np.exp(-(t-t_data[i-1])/tD1)
                
                if 'D2' in fit:
                    fit['D2'][i] = 1- ( 1- fit['D2'][ i-1]*d2)* np.exp( -( t -t_data[ i-1])/ tD2)
                
                if 'D3' in fit:
                    fit['D3'][i] =  1- ( 1- fit['D3'][ i-1]*d3)* np.exp( -( t -t_data[ i-1])/ tD3)

        return fit 
    
    #__________________________________________________________________________
    def _fd(self, params, t_data,):
        """ fit FD parameters and return amplitude vectors
        
        ---
        Arguments
        ---
        t_data = 1D array of times at which inputs arrive (s) 
        params = list of parameters, first two parameters are for facilitation 
        (f increment and time constant respectively), remaining parameters for depression
        
        ---
        Output
        ---
        A = 1D array of amplitude values at each input time in t_data
        """

        # get time series for each F and D variable
        #`````````````````````````````````````````
        fit = self._fd_loop(params, t_data)
        
        # Combine all F and D variables to get amplitude time series A
        #````````````````````````````````````````````
        # temporary array for storing amplitudes
        A = np.ones(fit[fit.keys()[0]].shape)
        
        # iterate through F and D variables in fit
        for key, var in fit.iteritems():
            
            # if variable is an F or D variable (parameters are also stored in this dictionary)
            if 'F' in key or 'D' in key:
                
                # multiply all F and D terms 
                A = A*var
                
        # output 1D array of amplitudes at input times in t_data        
        return A    
    
    #__________________________________________________________________________
    def _fd_err(self, params, t_data, y_data, discount_slope=False):
        """ Given FD parameters and data calculate error for single time series
        
        ---
        Arguments
        ---
        t_data = 1D array of times at which inputs arrive (s) 
        y_data = 1D array of measured amplitude values at input times in t_data
        params = list of parameters, first two parameters are for facilitation 
        (f increment and time constant respectively), remaining parameters for depression
        
        ---
        Output
        ---
        ssq_error = sum of squares error between data in y_data and fd model with parameters in params
        
        """
        
        # get amplitudes from FD model with params
        #````````````````````````````````````````
        A = self._fd(params, t_data,)
        
        sq_error = (A-y_data)**2
        
        # add discount factor so that earlier values are weighted more
        #````````````````````````````````````````````````````````````
        if discount_slope:
            discount = np.exp(-t_data/discount_slope)
        else:
            discount = np.ones(t_data.shape)
        
        # determine error
        #````````````````
        ssq_error = np.sum(discount*sq_error)

        # output sum of squared error
        return ssq_error
    
    #__________________________________________________________________________
    def _fd_err_mult(self, params, t_data_list, y_data_list, ):
        """Given FD parameters and data calculate error for list of time series, where each list entry is a different input stream
        
        ---
        Arguments
        ---
        t_data_list = list of 1D arrays of times at which inputs arrive (s) [input stream][input times]
        y_data_list = list of 1D arrays of measured amplitude values at input times in t_data [input stream][input times]
        params = list of parameters, first two parameters are for facilitation 
        (f increment and time constant respectively), remaining parameters for depression
        
        ---
        Output
        ---
        ssq_error = total sum of squares error between data in y_data_list and corresponding fd model with parameters in params
        """
        
        # list to store error for each input stream
        ssq_error=[]
        
        # iterate over input streams
        for i, input_stream in enumerate(t_data_list):
            
            # add error for each stream to list
            ssq_error.append(self._fd_err(params, t_data_list[i], y_data_list[i]))
            
        print 'error:', ssq_error

        # output total error summed over all streams
        return sum(ssq_error)

class ShapePlot():
    """ create plot of neuron morphology with data as a colormap

    similar to ShapePlot in neuron
    """
    
    def __init__(self):
        """
        """
        pass

    #__________________________________________________________________________
    def basic(self, morpho, data, axes, width_scale=1, colormap=colormap.jet):
        """ creates a shapeplot of a given neuron morphology with the given data

        Arguments:
        morpho: morphology structure with dimensions {tree}[section][segment](index, name, x, y, z, diam, parent_index)

        data: single floating point values store in a structure with dimensions {tree}[section][segment]

        axes: pyplot axes object

        colormap: matplotlib.cm colormap
        """
        # create list of points
        morph_points=[]
        indexes = []
        data_values=[]
        for tree_key, tree in morpho.iteritems():
            for sec_i, sec in enumerate(tree):
                for seg_i, seg in enumerate(sec):
                    morph_points.append(seg)
                    indexes.append(copy.copy(seg[0]))
                    data_values.append(data[tree_key][sec_i][seg_i])
        
        # resort lists
        sort_list = np.argsort(indexes)
        morph_points_sorted = [morph_points[i] for i in sort_list]
        data_values_sorted = [data_values[i] for i in sort_list]

        # lists to store segment shapes and their data values (colors)
        patches=[]
        colors=[]

        # iterate through child sections
        for child_i, child in enumerate(morph_points_sorted):
            # if segment is the root segment (parent index is -1)
            if child[-1]==-1:
                # skip, the root will be added as a parent to its children 
                continue
            # if not a root segment
            else:
                # find parent segment
                parent_idx = child[-1]
                parent = [val for i,val in enumerate(morph_points_sorted) if val[0]==parent_idx][0]
            
            # interpolate xyz values between parent and child
            parent_point = (parent[2], parent[3], parent[4])
            child_point = (child[2],child[3],child[4])
            mid_point = self.interpolate_3d(point1=parent_point, point2=child_point, t=0.5)

            # get diameter of parent and child 
            parent_diam = width_scale*parent[5]
            child_diam = width_scale*child[5]
            mid_diam = (parent_diam+child_diam)/2.

            # get data values for parent and child
            parent_color = data_values_sorted[parent[0]]
            child_color = data_values_sorted[child_i]

            # create polygon patch to plot segment
            parent_polygon = self.make_polygon(point1=parent_point, point2=mid_point, d1=parent_diam/2., d2=mid_diam/2.)
            child_polygon =self.make_polygon(point1=child_point, point2=mid_point, d1=child_diam/2., d2=mid_diam/2.)

            # add to list of patches
            patches.append(parent_polygon)
            colors.append(parent_color)
            patches.append(child_polygon)
            colors.append(child_color)

        # create patch collection
        p = PatchCollection(patches, cmap=colormap, alpha=1.)
        # set colors
        p.set_array(np.array(colors))
        # plot collection
        # axes.add_collection(p)
        # show colorbar
        # plt.colorbar(p)
        # autoscale axes
        # axes.autoscale()
        return p

    # create 3d interpolation function
    def interpolate_3d(self, point1, point2, t):
        """
        Arguments:
        point1 and point2 are lists/tuples of 3d points as [x,y,z]

        t is the parameter specifying relative distance between the two points from 0 to 1

        returns a tuple with the 3d coordinates of the requested point along the line (x,y,z)
        """
        x = point1[0] + point2[0]*t  - point1[0]*t
        y = point1[1] + point2[1]*t  - point1[1]*t
        z = point1[2] + point2[2]*t  - point1[2]*t

        return (x,y,z)

    def make_polygon(self, point1, point2, d1, d2):
        """ make a matplotlib.patches.Polygon object for plotting

        Arguments:
        point1, point2: list/tuple containing 3d coordinates of points to be connected

        d1,d2: diameter of each point 

        the function determines the line between the two points, then adds the corresponding diameters to each point along the orthogonal direction to the line, this produces the four points that the polygon

        returns a matplotlib.patches.Polygon object for plotting
        """
        # find slope of connecting line
        dy = point2[1]-point1[1]
        dx = point2[0] - point1[0]
        m = dy/dx

        # find slope of orthogonal line
        m_inv = -1./m

        # find x and y changes to add diameter
        delta_y1 = m_inv*np.sqrt(d1**2/(m_inv**2+1))
        delta_x1  = np.sqrt(d1**2/(m_inv**2+1))
        delta_y2 = m_inv*np.sqrt(d2**2/(m_inv**2+1))
        delta_x2  = np.sqrt(d2**2/(m_inv**2+1))

        # add diameter to first point
        y1_pos = point1[1] + delta_y1
        y1_neg = point1[1] - delta_y1
        x1_pos = point1[0] + delta_x1
        x1_neg = point1[0] - delta_x1

        # add diameter to second point
        y2_pos = point2[1] + delta_y2
        y2_neg = point2[1] - delta_y2
        x2_pos = point2[0] + delta_x2
        x2_neg = point2[0] - delta_x2

        # store points as a 4 x 2 array
        points = np.array([[x1_pos, y1_pos],[x1_neg, y1_neg],[x2_neg, y2_neg],[x2_pos, y2_pos]])

        return Polygon(points)

class Weights():
    """
    measure weight change at group of synapses

    saves initial and final weights at each active synapse across all simulated neurons (dcs polarity x synapses)
    """
    def __init__(self,p):
        self.n_pol = len(p['field'])

    def dw(self, p):
        self.group_dw(p)
        self.save_dw(p)
        self.plot_dw_all(p, self.w_end_all)
        self.plot_dw_mean(p, self.w_end_all)

    def group_dw(self,p):
        # arrays for storing all weight changes across neurons
        self.w_end_all = np.empty([self.n_pol,0])
        self.w_start_all = np.empty([self.n_pol,0])
        # loop over experiments (neurons)
        for data_file in os.listdir(p['data_folder']):
            # check for proper data file format
            if 'data' in data_file:

                with open(p['data_folder']+data_file, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)

                    # load data file
                    # pkl_file = open(p['data_folder']+data_file, 'rb')
                    # data = pickle.load(pkl_file)

                    self.p = data['p']
                    
                    self.n_act_seg = len(self.p['seg_list'])
                    
                    # measure weight changes for individual neuron
                    self.measure_dw(data)
                    
                    # add individual neuron to the group
                    self.w_end_all = np.append(self.w_end_all,self.w_end,axis=1)
                    self.w_start_all = np.append(self.w_start_all,self.w_start,axis=1)

                # # close pickle file
                # pkl_file.close()

    def measure_dw(self, data):
        # set up arrays to record final and initial weights at each active synapse
        self.w_end = np.empty([self.n_pol,self.n_act_seg]) # polarity x segments
        self.w_start = np.empty([self.n_pol,self.n_act_seg]) # polarity x segments

        # find active synapses (all recorded segments were active)
        # measure weight change at each active synapse

        for tree_key,tree in data.iteritems():
            if (data['p']['tree'] in tree_key) and ('_w' in tree_key):
                for f_i,f in enumerate(data['p']['field']):
                    cnt = -1
                    for sec_i,sec in enumerate(data['p']['sec_idx']):
                        for seg_i,seg in enumerate(data['p']['seg_idx'][sec_i]):
                            cnt += 1
                            self.w_end[f_i,cnt] = data[tree_key][f_i][sec][seg][-1]
                            self.w_start[f_i,cnt] = data[tree_key][f_i][sec][seg][0]

    def save_dw(self,p):
        with open(p['data_folder']+'dw_all_'+p['experiment']+'.pkl', 'wb') as output:
            pickle.dump(self.w_end_all, output,protocol=pickle.HIGHEST_PROTOCOL)

    def plot_dw_all(self,p,dw):
        # create figure
        self.fig_dw_all = plt.figure()
        # loop over n_pol
        for field_i,field in enumerate(p['field']):
            # plot
            plt.plot(field_i*np.ones(len(dw[field_i,:])),dw[field_i,:],p['field_color'][field_i]+'.')
        # save figure
        self.fig_dw_all.savefig(p['data_folder']+'fig_dw_all'+'.png', dpi=250)
        plt.close(self.fig_dw_all)
    
    def plot_dw_mean(self,p,dw):
        # determine stats
        dw_mean = np.mean(dw,axis=1)
        dw_std = np.std(dw,axis=1)
        dw_sem = stats.sem(dw,axis=1)
        # create figure
        self.fig_dw_mean = plt.figure()
        # loop over n_pol
        for field_i,field in enumerate(p['field']):
            # plot
            plt.errorbar(field_i,dw_mean[field_i],yerr=dw_sem[field_i],color = p['field_color'][field_i],fmt='.')
        # save figure
        self.fig_dw_mean.savefig(p['data_folder']+'fig_dw_mean'+'.png', dpi=250)
        plt.close(self.fig_dw_mean)

class Spikes():
    """
    detect spikes and determine where they originated
    """
    def __init__(self):
        pass

    def analysis_function(self, p):
        self.initialize_vectors(p)
        self.group_spikes(p)
        self.spike_start(p)
        self.save_spikes(p)
        self.plot_spike_hist_soma(self.spiket_soma,p)
        self.plot_spike_hist_dend(self.spike_dend_init,p)

    def initialize_vectors(self,p):
        self.n_pol = len(p['field'])
        # initialize lists
        self.spiket_soma = [] # soma spike times [polarity list][spikes array]
        self.spiket_dend = [] # dendrite spike times [polarity list][spikes array]
        self.sec_list = [] # keep track of sections (same dimensions as spiket_dend)
        self.seg_list = [] # keep track of segments (same dimensions as spiket_dend)
        self.cell_list_soma = []
        self.cell_list_dend = []
        self.win_list_soma = []
        self.win_list_dend = []
        # loop over polarity
        for pol in range(self.n_pol):
            self.spiket_soma.append(np.empty([1,0]))
            self.spiket_dend.append(np.empty([1,0]))
            self.sec_list.append([])
            self.seg_list.append([])
            self.cell_list_soma.append([])
            self.cell_list_dend.append([])
            self.win_list_soma.append([])
            self.win_list_dend.append([])

    def group_spikes(self,p):
        cell_num = -1   # track which cell number
        for data_file in os.listdir(p['data_folder']):
            # check for proper data file format
            if 'data' in data_file:
                cell_num+=1
                # load data file
                pkl_file = open(p['data_folder']+data_file, 'rb')
                data = pickle.load(pkl_file)
                # get parameters from specific experiment
                self.p = data['p']
                self.n_act_seg = len(self.p['seg_list'])
                self.measure_spikes(data,self.p,cell_num)

    def spike_window(self,p):
        # determine windows for spike time detection [window number][min max]
        bursts = range(p['bursts'])
        pulses = range(p['pulses'])
        burst_freq = p['burst_freq']
        pulse_freq = p['pulse_freq']
        nrn_fs = 1000.
        fs = nrn_fs/p['dt']
        warmup = p['warmup']
        # for each input pulse their is a list containing the window start and stop time [window number][start,stop]
        return [[warmup*fs+(burst)*fs/burst_freq+(pulse)*fs/pulse_freq,warmup*fs+(burst)*fs/burst_freq+(pulse+1)*fs/pulse_freq] for burst in bursts for pulse in pulses]

    def measure_spikes(self,data,p,cell_num=0):
        # nrn_fs = 1000. # conversion from seconds to miliseconds
        # fs = nrn_fs/p['dt'] # sampling rate in samples/second
        window  = self.spike_window(p) # list of spike windows [window #][start,stop]
        
        # detect spikes for individual neurons
        for pol in range(self.n_pol):
            
            # detect soma spikes
            self.spikes_soma = self.detect_spikes(data['soma_v'][pol][0][0])['times']
            if self.spikes_soma.size!=0:
                
                # add spike times to array
                self.spiket_soma[pol] = np.append(self.spiket_soma[pol],self.spikes_soma*p['dt'],axis=1)

                # track cell number
                for spike_i,spike in enumerate(self.spikes_soma[0,:]):
                    # detect the window that the spike occurred in, indexed by the onset time of the window
                    spike_soma_win = [win[0] for win in window if (spike >= win[0]) and (spike < win[1]) ]
                    self.cell_list_soma[pol].append(cell_num)
                    self.win_list_soma[pol].append(spike_soma_win)
            
            # detect dendritic spikes and track location
            cnt=-1
            for sec_i,sec in enumerate(data[p['tree']+'_v'][pol]): # loop over sections
                for seg_i,seg in enumerate(data[p['tree']+'_v'][pol][sec_i]): # loop over segemnts
                    cnt+=1
                    # detect spikes
                    spikes_dend = self.detect_spikes(np.array(data[p['tree']+'_v'][pol][sec_i][seg_i]))['times']
                    if spikes_dend.size!=0:
                        # add spike times to array
                        self.spiket_dend[pol] = np.append(self.spiket_dend[pol],spikes_dend,axis=1)
                        # spiket_dend_track = np.append(spiket_dend_track,spikes_dend,axis=1)
                        # for each spike store the section, segment, cell number in the appropriate list
                        for spike in spikes_dend[0,:]:
                            spike_dend_win = [win[0] for win in window if (spike >= win[0]) and (spike < win[1]) ]
                            self.sec_list[pol].append(sec_i)
                            self.seg_list[pol].append(seg_i)
                            self.cell_list_dend[pol].append(cell_num)
                            self.win_list_dend[pol].append(spike_dend_win)

    def spike_start(self,p):
        # determine windows for spike time detection [window number][min max]
        bursts = range(p['bursts'])
        pulses = range(p['pulses'])
        burst_freq = p['burst_freq']
        pulse_freq = p['pulse_freq']
        fs = 1./p['dt']
        warmup = p['warmup']
        # for each input pulse their is a list containing the window start and stop time [window number][start,stop]
        window =  [[warmup*fs+(burst)*1000*fs/burst_freq+(pulse)*1000*fs/pulse_freq,warmup*fs+(burst)*1000*fs/burst_freq+(pulse+1)*1000*fs/pulse_freq] for burst in bursts for pulse in pulses]

        # numpy array for storing minumum spike time for each cell 
        self.spike_dend_init = []   # minimum spike time for each cell
        self.spike_dend_init_sec = []   # section where first spike occured
        self.spike_dend_init_seg = []   # segment where first spike occured
        self.spike_dend_init_cell = [] # keep track of cell number
        self.spike_dend_init_win = [] # timing of presynaptic input 
        # loop over polarities
        for pol in range(self.n_pol):
            # list all cells with a dendritic spike
            cells = list(set(self.cell_list_dend[pol]))
            # numpy array for storing minumum spike time for each cell 
            self.spike_dend_init.append([]) # minimum spike time for each cell
            self.spike_dend_init_sec.append([]) # section where first spike occured
            self.spike_dend_init_seg.append([]) # segment where first spike occured
            self.spike_dend_init_cell.append([]) # keep track of cell number
            self.spike_dend_init_win.append([]) # keep track of cell number
            # loop over cells
            for cell_i,cell in enumerate(cells):
                # print len(self.spiket_dend[pol][0,:])
                # print len(self.cell_list_dend[pol])
                # for each cell list all dendritic spike times
                spiket_dend = [spikes for spike_i,spikes in enumerate(self.spiket_dend[pol][0,:]) if self.cell_list_dend[pol][spike_i]==cell]
                # keep track of the index for in the full list of spike times
                spikei_dend = [spike_i for spike_i,spikes in enumerate(self.spiket_dend[pol][0,:]) if self.cell_list_dend[pol][spike_i]==cell]
                # loop over spike windows
                for win in window:
                    # return spikes for this cell that fit the window
                    spiket_dend_win = [spike for spike in spiket_dend if spike >= win[0] and spike < win[1]]
                    # keep track of indeces
                    spikei_dend_win = [spike for spike_i,spike in enumerate(spikei_dend) if spiket_dend[spike_i] >= win[0] and spiket_dend[spike_i] < win[1]]
                    # check if spike was found
                    if spiket_dend_win:
                        # print min(spiket_dend_win)/fs
                        # index in full list for first spike in current time window in current cell
                        spike_idx = spikei_dend_win[spiket_dend_win.index(min(spiket_dend_win))]
                        # store minimum spike time and keep track of section, segment, and cell number
                        self.spike_dend_init[pol].append(min(spiket_dend_win)/fs)
                        self.spike_dend_init_sec[pol].append(self.sec_list[pol][spike_idx])
                        self.spike_dend_init_seg[pol].append(self.seg_list[pol][spike_idx])
                        self.spike_dend_init_cell[pol].append(cell)
                        self.spike_dend_init_win[pol].append(win[0])
            
            self.spike_dend_init[pol] = np.array([self.spike_dend_init[pol]])

    def spike_start_compare(self,p):
        cell1 = cell.CellMigliore2005(p)
        spikes = {}
        for tree_i,tree in cell1.geo.iteritems():
            spikes[tree_key] = []
            for field_i,field in p['field']:
                spikes[tree_key].append([])
                for sec_i,sec in tree:
                    spikes[tree_key][field_i].append([])
                    for seg_i,seg in sec:
                        spikes[tree_key][field_i][sec_i].append({})
                        spikes[tree_key][field_i][sec_i][seg_i]['times'] = []
                        spikes[tree_key][field_i][sec_i][seg_i]['train'] = []
                        spikes[tree_key][field_i][sec_i][seg_i]['p'] = []
                        spikes[tree_key][field_i][sec_i][seg_i]['xcorr'] = []


        for data_file in os.listdir(p['data_folder']):
            # check for proper data file format
            if 'data' in data_file:
                with open(p['data_folder']+data_file, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)

                for tree_key,tree in spikes:
                    for field_i,field in tree:
                        for sec_i,sec in field:
                            for seg_i,seg in sec:
                                spike_temp = self.detect_spikes(data[tree_key+'_v'][field_i][sec_i][seg_i],thresshold=-20)
                                if tree_key is 'soma':
                                    spike_temp_soma =spike_temp 

                                seg['times'].append(spike_temp['times'])
                                seg['train'].append(spike_temp['train'])
                                seg['p'].append(data['p'])
                                if tree_key is not 'soma':
                                    xcorr_temp = scipy.signal.correlate(spike_temp['train'],spikes['soma'][field_i])

                                seg['xcorr'].append(spike_temp['xcorr'])
                                


                                spikes = self.detect_spikes(seg,threshold = -20)


        # loop over polarities
        # loop over cells
        # loop over spike windows
        # compare first dendritic spike to first somatic spike
        # get cross correlation for all time delays
        # determine whether these features influence field effects 

    def save_spikes(self,p):
        with open(p['data_folder']+'spiket_soma_all_'+p['experiment']+'.pkl', 'wb') as output:
            pickle.dump(self.spiket_soma, output,protocol=pickle.HIGHEST_PROTOCOL)

        with open(p['data_folder']+'spiket_dend_all_'+p['experiment']+'.pkl', 'wb') as output:
            pickle.dump(self.spiket_dend, output,protocol=pickle.HIGHEST_PROTOCOL)

    def plot_spike_hist_soma(self,data,p):
        warmup = p['warmup']
        finish = p['tstop']
        bins = np.linspace(warmup,finish ,(finish-warmup)/p['dt'])
        self.fig_spike_hist_soma = plt.figure()
        for pol in range(self.n_pol):
            plt.hist(data[pol][0,:],bins=bins,color = p['field_color'][pol])
        plt.title('Somatic spike time histogram')
        plt.xlabel('spike onset (ms)')
        plt.ylabel('count')
        # save figure
        self.fig_spike_hist_soma.savefig(p['data_folder']+'fig_spike_hist_soma'+'.png', dpi=250)
        plt.close(self.fig_spike_hist_soma)

    def plot_spike_hist_dend(self,data,p):
        warmup = p['warmup']
        finish = p['tstop']
        bins = np.linspace(warmup,finish ,(finish-warmup)/p['dt'])
        self.fig_spike_hist_dend = plt.figure()
        for pol in range(self.n_pol):
            plt.hist(data[pol][0,:],bins=bins,color = p['field_color'][pol])
        plt.title('Dendritic spike time histogram')
        plt.xlabel('spike onset (ms)')
        plt.ylabel('count')
        self.fig_spike_hist_dend.savefig(p['data_folder']+'fig_spike_hist_dend'+'.png', dpi=250)
        plt.close(self.fig_spike_hist_dend)

    def spikes_xcorr(self,data1,data2,p):
        pass
        # 


    def detect_spikes(self,data,threshold=-20):
        """
        """
        spike_times = np.asarray(np.where(np.diff(np.sign(data-threshold))>0))
        spike_train = np.zeros([1,len(data)])
        for time in spike_times:
            spike_train[0,time] = 1 
        # detect indeces where vector crosses threshold in the positive direction
        return {'times':spike_times,'train':spike_train}

class PlotRangeVar():
    """ plot different range variables over time 
    """
    def __init__(self):
        pass

    def plot_all(self, p):
        """ Plot times series data for all data files in folder without replotting

        Arguments:

        p : parameter dictionary containing the following entries
            'data_folder' : string specifying the folder containing data and plots

            'tree' : neuron subtree containing the data to plot

            'sec_idx' : list of section indeces to plot

            'seg_idx' : list of segments for each section to plot [section_num][segment] 

            'plot_variables' : list of range variables to plot; e.g. ['v', 'i_hd']

        """
        # all files in directory
        files = os.listdir(p['data_folder'])
        
        # files containing plot figures
        plot_files = [file for file in files if 'trace' in file]
        
        # data files
        data_files = [file for file in files if 'data' in file]
        
        # unique identifiers for each file type
        plot_files_id = [file[-39:-4] for file in files if 'trace' in file]
        data_files_id= [file[-39:-4] for file in files if 'data' in file]
        
        # iterate over all data files in folder
        for file_i, file in enumerate(data_files):
            
            # check if plot already exists matching uid's
            if data_files_id[file_i] not in plot_files_id:
                print data_files_id[file_i] 
                # open data file
                with open(p['data_folder']+file, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)

                # update specific experiment parameters
                p_data = data['p']
                
                for path_key, path in p_data['p_path'].iteritems():
                    # plot range variables (automatically saved to same folder)
                    self.plot_trace(data=data, 
                        trees=path['trees'], 
                        sec_idx=path['sec_idx'], 
                        seg_idx=path['seg_idx'],
                        variables=path['plot_variables'],
                        x_variables=path['x_variables'],
                        file_name=path['trial_id'],
                        group_trees=path['group_trees'])

    def plot_trace(self, data, sec_idx=[], seg_idx=[], soma=True, axon=True, group_trees=True, variables=['v'], x_variables='t', xlim=[], ylim=[], file_name=''):
        """ Create a figure containing a subplot for each segment specified in seg_idx

        variable is a list of range variable key words indicating which varaible to plot.  A new figure is created and saved for each variable

        data structure organized as:  data{'polarity'}{'path number'}{'tree'}[section number][segment number][data vector]

        parameters that are specific to a given pathway are stored in data{'p'}{'p_path'}[path number]


        """
        # load parameters
        p = data['p']

        # number field intensities/polarities
        n_pol = len(p['field'])
    
        # dictionary to hold figures
        fig={}  

        # iterate over pathways
        for path_key, path in p['p_path'].iteritems():
            sec_idx = path['sec_idx']
            seg_idx = path['seg_idx']
            print seg_idx
            print sec_idx
            
            fig[path_key]={}
            # iterate over list of y variables to plot
            for var in variables:

                fig[path_key][var] = {}

                # iterate over list of x variables to plot
                for x_var in x_variables:


                    # if data from all subtrees in one plot
                    if group_trees:

                        # create figure
                        fig[path_key][var][x_var] = plt.figure()

                        # number of segments to plot
                        nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for tree_key, tree in seg_idx.iteritems() for sec in tree if tree_key in path['trees'] ])+1

                        # print nseg
                        # plot soma trace?
                        if soma:
                            nseg+=1
                        if axon:
                            nseg+=1

                        # columns and rows for subplot array    
                        cols = int(math.ceil(math.sqrt(nseg)))
                        rows = int(math.ceil(math.sqrt(nseg)))

                    
                    # if each subtree to get its own figure
                    elif not group_trees:
                        
                        # create another dimension to store each subtree figure separately
                        fig[path_key][var][x_var]={}
                
                    # count segments
                    cnt=0

                    # iterate over trees
                    for tree_key, tree in seg_idx.iteritems():

                        if not group_trees:
                            cnt=0

                        # check that there are active sections in the tree
                        if tree:

                            # if each subtree gets its own figure
                            if not group_trees:
                                fig[path_key][var][x_var][tree_key] = plt.figure()

                                # number of segments to plot
                                nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for sec in tree])+1
                                
                                # print'nseg:',nseg
                                # plot soma trace?
                                if soma:
                                    nseg+=1
                                if axon:
                                    nseg+=1

                                # columns and rows for subplot array    
                                cols = int(math.ceil(math.sqrt(nseg)))
                                rows = int(math.ceil(math.sqrt(nseg)))

                                # print 'rows,cols:',rows,cols

                            # iterate over sections
                            for sec_i,sec in enumerate(tree):

                                # derive section number from index
                                sec_num = sec_idx[tree_key][sec_i]
                                
                                # iterate over segments
                                for seg_i, seg in enumerate(sec):
                                    
                                    # count subplots (segments)
                                    cnt+=1
                                    
                                    # create subplot
                                    plt.subplot(rows, cols, cnt)

                                    # get segment distance from soma
                                    seg_dist = p['seg_dist'][tree_key][sec_num][seg]
                                    
                                    # plot title
                                    plt.title(tree_key + ('%.2f'%seg_dist) )

                                    # adjust limits
                                    if var is 'v':
                                        if xlim:
                                            plt.xlim(xlim)
                                        if ylim:
                                            plt.ylim(ylim)
                                    
                                    # iterate over stimulation polarity
                                    for f_i, f in enumerate(p['field']):
                                    
                                        # check if variable exists in the current section
                                        if data[str(f)][path_key][tree_key+'_'+var][sec_i]:

                                            # if not plotting the soma trace
                                            if soma and cnt<nseg:

                                                # retrieve time series to plot
                                                v = data[str(f)][path_key][tree_key+'_'+var][sec_i][seg_i]

                                                # retrieve x variable
                                                if x_var =='t':
                                                    # time vector
                                                    xv = data[str(f)][path_key]['t']

                                                # other variable from arguments
                                                else:
                                                    xv = data[str(f)][path_key][tree_key+'_'+x_var][sec_i][seg_i]


                                            # set plot color based on stimulation polarity
                                            color = p['field_color'][f_i]
                                            
                                            # add trace to corresponding subplot
                                            plt.plot(xv, v, color=color)
                                            plt.xlabel(x_var)
                                            plt.ylabel(var)

                            if soma:
                                cnt+=1

                                # if plotting soma trace
                                for f_i, f in enumerate(p['field']):

        
                                    # if variable exists in soma data
                                    if 'soma_'+var in data[str(f)][path_key].keys():
                                        if len(data[str(f)][path_key]['soma_'+var][0][0])>0:
                                        
                                            # subplot for soma trace
                                            plt.subplot(rows, cols, cnt)

                                            # adjust limits
                                            if var is 'v':
                                                if xlim:
                                                    plt.xlim(xlim)
                                                if ylim:
                                                    plt.ylim(ylim)

                                            # retrieve data to plot
                                            v = data[str(f)][path_key]['soma_'+var][0][0] 

                                            # determine x variable to plot
                                            if x_var =='t':
                                                # time vector
                                                xv = data[str(f)][path_key]['t']
                                            else:
                                                xv = data[str(f)][path_key]['soma_'+x_var][0][0]
                                            
                                            # set plot color
                                            color = p['field_color'][f_i]
                                            
                                            plt.plot(xv, v, color=color)
                                            plt.title('soma')
                                            plt.xlabel(x_var)
                                            plt.ylabel(var)

                            if axon:
                                cnt+=1

                                # if plotting soma trace
                                for f_i, f in enumerate(p['field']):

                                    # if variable exists in soma data
                                    if 'axon_'+var in data[str(f)][path_key].keys():
                                        if len(data[str(f)][path_key]['axon_'+var][0][0])>0:
                                        
                                            # subplot for soma trace
                                            plt.subplot(rows, cols, cnt)

                                            # adjust limits
                                            if var is 'v':
                                                if xlim:
                                                    plt.xlim(xlim)
                                                if ylim:
                                                    plt.ylim(ylim)

                                            # retrieve data to plot
                                            v = data[str(f)][path_key]['axon_'+var][0][0] 

                                            # determine x variable to plot
                                            if x_var =='t':
                                                # time vector
                                                xv = data[str(f)][path_key]['t']
                                            else:
                                                xv = data[str(f)][path_key]['axon_'+x_var][0][0]
                                            
                                            # set plot color
                                            color = p['field_color'][f_i]
                                            
                                            plt.plot(xv, v, color=color)
                                            plt.title('axon')
                                            plt.xlabel(x_var)
                                            plt.ylabel(var)
                            
                            # save figure
                            # if each tree has separate figure
                            if not group_trees:
                                
                                # info to add to file name
                                file_name_add = 'path_'+path_key+'_'+tree_key+'_trace_'+x_var+'_x_'+var

                                # save figure
                                fig[path_key][var][x_var][tree_key].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

                                # close figure
                                plt.close(fig[path_key][var][x_var][tree_key])

                        # if all trees are in the same figure
                        if group_trees:
                            all_trees =''
                            for tree_key, tree in seg_idx.iteritems():
                                all_trees = all_trees+tree_key+'_'

                            # info to add to file name
                            file_name_add = all_trees+'path_'+path_key+'_''trace_'+x_var+'_x_'+var


                        # if all trees are in the same figure
                        if group_trees:
                            all_trees =''
                            for tree_key, tree in seg_idx.iteritems():
                                all_trees = all_trees+tree_key+'_'

                            # info to add to file name
                            file_name_add = all_trees+'path_'+path_key+'_''trace_'+x_var+'_x_'+var

                            # save and close figure
                            fig[path_key][var][x_var].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

                            plt.close(fig[path_key][var][x_var])

class IO():
    """ create shape plot 
    """
    pass

class Experiment:
    """analyses for individual experiments
    """
    def __init__(self, **kwargs):
        experiment = getattr(self, kwargs['experiment'])

        experiment(**kwargs) 

    def _sigmoid_opt(self, params, *data):
        """ function for fitting a sigmoid to data

        Positional Arguments
        params = list with 3 entries corresponding to parameters ymax, x50, and s respectively

        data[0] = input data values
        data[1] = output data values

        Return
        ssq_error = sum of squares error between sigmoid with parameters in param and output values in data[1]
        """
        ymax = params[0]
        x50 = params[1]
        s = params[2]
        x = data[0]
        y = data[1]
        y_fit = ymax/(1+np.exp((x50-x)/s))
        ssq_error = np.sum(np.square(y-y_fit))
        return ssq_error

    def _sigmoid2_opt(self, params, *data):
        """
        """
        ymax_1 = params[0]
        x50_1 = params[1]
        s_1 = params[2]
        ymax_2 = params[3]
        x50_2 = params[4]
        s_2 = params[5]
        x = data[0]
        y = data[1]
        y_fit = ymax_1/(1+np.exp((x50_1-x)/s_1)) - ymax_2/(1+np.exp((x50_2-x)/s_2))
        ssq_error = np.sum(np.square(y-y_fit))
        return ssq_error

    def _get_spikes(self, **kwargs):
        """ Retrieve spike data from individual simulations
        
        ==
        KEYWORD ARGUMENTS:
        data= structure containing time series voltage data organized as data{'polarity'}{'pathway'}{'tree_variable'}[section number][segment number][time series]
        data also contains a parameter dictionary in data['p']
        individual parameter dictionaries are stored for each pathway in data['p']['p_path']

        threshold= threshold for detecting spikes

        weakpath_bins=whether to include strong path time bins in weak path spike detection

        ==
        RETURN:
        A dictionary containing spike data organized as spike_dictionary{'polarity'}{'pathway'}{'data type'}

        ==
        DATATYPES:
        spikes_dend= list of all dendritic spike times
        
        spikes_dend_dist= for each dendritic spike in 'spikes_dend', list the distance from the soma, [distances]
        
        spikes_first= for each activated segment, whether there was a spike at all, and where it was initiated. 0= no spike, 1=dendrite first, 2=soma first

        spike_times= list spike times for each segment, same organization as 'seg_idx', {tree_key}[sec_num][seg_num][spike_times].  If no spikes are found, returns empty list

        dw= list weight change for each segment, same organization as 'seg_idx', {tree_key}[sec_num][seg_num=dw]

        time_bins= nested list of time bins [bin number][time1, time2], where each bin corresponds to the time between input pulses

        spikes_dend_bin= nested list of dendritic spike times [time bin][spike times]

        spikes_som_bin= nested list of somatic spike times [time bin][spike times]

        spikes_dend_bin_norm= nested list of dendritic spike times normalized to input pulse onset time [time bin][normalzied spike times]

        spikes_soma_bin_norm= nested list of somatic spike times normalized to input pulse onset time [time bin][normalized spike times]

        spikes_dend_dist_bin= nested list of dendritic spike distances [time bin][distances]

        spikes_dend_diff_bin= nested list of timing difference between dendrite and soma [time bin][spike time differences], soma first=negative, dendrite first=positive, if no soma spike=no subtraction(positive)

        syn_frac_bin= fraction of synapses with a spike in each bin, [time bin][fraction]
        """
        data = kwargs['data']
        threshold = kwargs['threshold']

        # parameter dictionary
        p = data['p']

        print p['path_combos']

        # load spike analysis functions
        spike_analysis = Spikes()

        # temporary store for current trial data. all dictionary entries will be transferred to full data structure (all trials)
        spike_dic = {}

        # print trial
        # parameter dictionary
        spike_dic['p'] = copy.copy(p)

        # iterate over polarity
        for polarity_key, polarity in data.iteritems():

            # exclude parameter dictionary
            if polarity_key != 'p':

                spike_dic[polarity_key]={}
                # iterate over pathways
                for path_key, path in polarity.iteritems():

                    # exclude time vectors
                    if path_key!='t':

                        spike_dic[polarity_key][path_key]={}
                        p_path = p['p_path'][path_key]

                        # temporary dictionary
                        dtemp={}

                        dtemp['p'] = copy.copy(p)

                        dtemp['p_path']=copy.copy(p_path)

                        # list all dendritic spikes [spike times]
                        dtemp['spikes_dend']=[] 
                        # for each dendritic spike in 'spikes_dend', list the distance from the soma, [distances]
                        dtemp['spikes_dend_dist']=[]
                        # for each activated segment, whether there was a spike at all, and where it was initiated
                        # 0= no spike, 1=dendrite first, 2=soma first
                        dtemp['spikes_first'] = []
                        # list spike times for each segment, same organization as 'seg_idx', {tree_key}[sec_num][seg_num][spike_times]
                        dtemp['spike_times'] ={}
                        # list weight change for each segment, same organization as 'seg_idx', {tree_key}[sec_num][seg_num=dw]
                        dtemp['dw']={}

                        # add soma data
                        dtemp['spike_times']['soma'] = [[]]
                        spike_times = spike_analysis.detect_spikes(data[polarity_key][path_key]['soma_v'][0][0], threshold=threshold)['times'][0]
                        dtemp['spike_times']['soma'][0].append(spike_times)
                        dtemp['spikes_soma'] = spike_times

                        # iterate over trees
                        for tree_key, tree in p_path['seg_idx'].iteritems():
                            # add dimension
                            dtemp['spike_times'][tree_key] =[]
                            dtemp['dw'][tree_key] =[]
                            # iterate over sections
                            for sec_i, sec in enumerate(tree):
                                # add dimensions
                                dtemp['spike_times'][tree_key].append([])
                                dtemp['dw'][tree_key].append([])
                                sec_num = p['p_path'][path_key]['sec_idx'][tree_key][sec_i]
                                # iterate over segments
                                for seg_i, seg in enumerate(sec):
                                    seg_num=seg
                                    distance = p['seg_dist'][tree_key][sec_num][seg_num]

                                    # list of spike times [spike_times]
                                    spike_times = spike_analysis.detect_spikes(data[polarity_key][path_key][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0]

                                    # scalar weight change
                                    dw = data[polarity_key][path_key][tree_key+'_gbar'][sec_i][seg_i][-1]/data[polarity_key][path_key][tree_key+'_gbar'][sec_i][seg_i][0]

                                    # add to dtemp structure, each segment contains a list of spike times
                                    dtemp['spike_times'][tree_key][sec_i].append(spike_times)
                                    # each segment is a scalar weight change
                                    dtemp['dw'][tree_key][sec_i].append(dw)

                                    # record whether there was a spike in soma or dendrite first [all segments]
                                    # no spike=0, dend first=1, soma first=2
                                    # if there are dendritic spikes
                                    if len(spike_times)>0:
                                        
                                        # iterate through spike times
                                        for spike_i, spike_time in enumerate(spike_times):

                                            # add to list of all dendritic spike times for this trial/cell
                                            # [all dendritic spike times]
                                            dtemp['spikes_dend'].append(spike_time)
                                            # [all dendritic spike distances]
                                            dtemp['spikes_dend_dist'].append(distance)

                                            # if this is the first spike
                                            if spike_i==0:
                                                
                                                # if there is also a somatic spike
                                                if len(dtemp['spikes_soma'])>0:
                                                    
                                                    # if the somatic spike occurs first
                                                    if spike_time > dtemp['spikes_soma'][0]:

                                                        # store as soma first
                                                        dtemp['spikes_first'].append(2)
                                                    
                                                    # otherwise the spike is dend first
                                                    else:
                                                        dtemp['spikes_first'].append(1)
                                                
                                                # if there is a dendritic but no somatic spike, it is dend first
                                                else:
                                                    dtemp['spikes_first'].append(1)
                                    
                                    # otherwise no spike at all
                                    else:
                                        dtemp['spikes_first'].append(0)

                        # create nested list of spike times with dimensions [pulse time bin][spike times]
                        # nested list of time bins [bin number][time1, time2]
                        dtemp['time_bins'] = []
                        # nested list of dendritic spike times [time bin][spike times]
                        dtemp['spikes_dend_bin'] = []
                        # nested list of somatic spike times [time bin][spike times]
                        dtemp['spikes_soma_bin'] = []
                        # nested list of dendritic spike times normalized to input pulse onset time [time bin][spike times]
                        dtemp['spikes_dend_bin_norm'] = []
                        # nested list of somatic spike times normalized to input pulse onset time [time bin][spike times]
                        dtemp['spikes_soma_bin_norm'] = []
                        # nested list of dendritic spike distances [time bin][distances]
                        dtemp['spikes_dend_dist_bin'] = []
                        # nested list of timing difference between dendrite and soma [time bin][spike time differences], soma first=negative, dendrite first=positive, if no soma spike=no subtraction(positive)
                        dtemp['spikes_dend_diff_bin'] = []
                        # fraction of synapses with a spike in each bin, [time bin][fraction]
                        dtemp['syn_frac_bin'] = []

                        if kwargs['weakpath_bins']:
                            pulses = p['pulses']
                        else:
                            pulses = p['p_path'][path_key]['pulses']
                        # iterate through pulses in the current pathway
                        for pulse_i, pulse in enumerate(range(int(pulses))):

                            if kwargs['weakpath_bins'] and path_key=='weak':
                                # determine time bins
                                dtemp['time_bins'].append([])
                                time1 = (p['warmup'] + 1000/p['pulse_freq']*pulse_i)/p['dt'] +1 
                                time2 = (p['warmup'] + 1000/p['pulse_freq']*(pulse_i+1))/p['dt']
                                dtemp['time_bins'][pulse_i].append(time1)
                                dtemp['time_bins'][pulse_i].append(time2)
                            else:
                                 # determine time bins
                                dtemp['time_bins'].append([])
                                time1 = (p_path['warmup'] + 1000/p_path['pulse_freq']*pulse_i)/p['dt'] +1 
                                time2 = (p_path['warmup'] + 1000/p_path['pulse_freq']*(pulse_i+1))/p['dt']
                                dtemp['time_bins'][pulse_i].append(time1)
                                dtemp['time_bins'][pulse_i].append(time2)


                            # get spike times that fall within the current bin 
                            binned_spikes_dist = []
                            binned_spikes_dend =[]
                            binned_spikes_dend_norm =[]
                            binned_spikes_soma = []
                            binned_spikes_soma_norm = []
                            
                            # if there are any somatic spikes
                            if len(dtemp['spikes_soma'])>0:
                                # list spikes in current bin, return empty list if no spikes
                                binned_spikes_soma = [spike for spike_i, spike in enumerate(dtemp['spikes_soma']) if (spike > time1 and spike <= time2)]
                                # if there are spikes in the current bin, normalized to the pulse onset
                                if len(binned_spikes_soma)>0:
                                    binned_spikes_soma_norm = [spike-time1 for spike in binned_spikes_soma]
                            
                            # if there are any dendritic spikes
                            if len(dtemp['spikes_dend'])>0:
                                # list spikes in current bin, return empty list if no spikes
                                binned_spikes_dend = [spike for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]

                                binned_spikes_dist = [dtemp['spikes_dend_dist'][spike_i] for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]

                                # if there are spikes in the current bin, normalized to the pulse onset
                                if len(binned_spikes_dend)>0:
                                    binned_spikes_dend_norm = [spike-time1 for spike in binned_spikes_dend]
                                
                            
                            # difference between dendritic and somatic spike times (dendrite first is positive, soma first is negative)
                            # if there are somatic and dendritic spikes in the current bin
                            if len(binned_spikes_soma)>0 and len(binned_spikes_dend)>0:
                                # list of time differences
                                binned_spikes_dend_diff = [binned_spikes_soma[0]-spike for spike_i, spike in enumerate(binned_spikes_dend)]
                            else: 
                                # otherwise dendritic spiek list remains unchanged (positive)
                                binned_spikes_dend_diff = binned_spikes_dend_norm

                            # fraction of synapses that spike in current bin
                            binned_distances =  list(set(binned_spikes_dist))
                            binned_syn_frac = float(len(binned_distances))/float(p_path['syn_num'])
                            

                            # add spike times for current bin to list of all bins
                            # [bin number][list of spike times]
                            dtemp['spikes_dend_bin'].append(binned_spikes_dend)
                            dtemp['spikes_soma_bin'].append(binned_spikes_soma)
                            dtemp['spikes_dend_bin_norm'].append(binned_spikes_dend_norm)
                            dtemp['spikes_soma_bin_norm'].append(binned_spikes_soma_norm)
                            dtemp['spikes_dend_dist_bin'].append(binned_spikes_dist)
                            dtemp['spikes_dend_diff_bin'].append(binned_spikes_dend_diff)
                            dtemp['syn_frac_bin'].append(binned_syn_frac)

                         # fraction of synapses that spike at all during simulation
                        distances = list(set(dtemp['spikes_dend_dist']))
                        dtemp['syn_frac'] = float(len(distances))/float(p_path['syn_num'])

                        # update main data structure
                        # for each data type
                        for dtype_key, dtype in dtemp.iteritems():

                            spike_dic[polarity_key][path_key][dtype_key]=dtype

        return spike_dic

    def _load_group_data(self, **kwargs):
        """ Load group data from folder
        
        ==
        Keyword Arguments:
        experiment: experiment name, typically a string of the form 'exp_4a'

        save_string: string for saving pickled group data, must include .pkl suffix
        
        ==
        Return:
        group data, typically a dictionary.  If no file is found with the specified string, an empty dictionary is returned

        """

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        
        # all files in directory
        files = os.listdir(data_folder)

        save_string = kwargs['save_string']
        # if data file already exists
        if save_string in files:
            # load data
            print 'group data found:', save_string
            with open(data_folder+save_string, 'rb') as pkl_file:
                group_data= pickle.load(pkl_file)
            print 'group data loaded'
        # otherwise create data structure
        else:
            # data organized as {frequency}{syn distance}{number of synapses}{polarity}[trial]{data type}{tree}[section][segment][spikes]
            print 'no group data found'
            group_data= {}

        return group_data

    def exp_1a(self, **kwargs):
        """ 
        activate a varying number of synapses at varying frequency with varying distance from the soma.  Synapses are chosen from a window of 50 um, with the window moving along the apical dendrite.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of mean distance from the soma and frequency of synaptic activity.  

        Plots: 
        number of active synapse x fraction of synapses with at least one spike (somatically and dendritically driven spikes are separated)

        number of active synapses x total number of spikes (normalized to number of active synapses, somatic and dendritic spikes are separated)

        number of active synapses x mean spike timing for each spike

        """
        # spike threshold 
        threshold =-30

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        
        # all files in directory
        files = os.listdir(data_folder)
        
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        # if list of processed files in folder, load list
        id_list_string_spike_times = 'id_list_spike_times.pkl'
        if id_list_string_spike_times in files:
            print 'id_list found'
            
            with open(data_folder+id_list_string_spike_times, 'rb') as pkl_file:
                id_list_spike_times = pickle.load(pkl_file)

            print 'id list loaded'
        
        # otherwise create empty list
        else:
            id_list_spike_times = []

        # string to save group data
        save_string_group_data_raw = 'spikes_grouped_raw'+'.pkl'
        # if data file already exists
        if save_string_group_data_raw in files:
            # load data
            print 'raw spike data found'
            with open(data_folder+save_string_group_data_raw, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
            print 'raw spike data loaded'
        # otherwise create data structure
        else:
            # data organized as {frequency}{syn_distance}{number of synapses}{polarity}[trial]{data type}{tree}[section][segment][spikes]
            spike_data= {}

        # load spike analysis functions
        spike_analysis = Spikes()

        # data types to be stored
        dtypes = ['spike_times','dw','p']

        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:# and data_file_i <=1000:

                # check if data has been processed already
                if data_file not in id_list_spike_times:

                    print data_file
                    # open unprocessed data

                    try:
                        with open(data_folder+data_file, 'rb') as pkl_file:
                            data = pickle.load(pkl_file)

                        print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'
                    # if data file is corrupted, skip it
                    except EOFError:
                        'EOF error, file not opened'
                        continue

                    # add to processed list
                    id_list_spike_times.append(data_file)

                    # parameter dictionary
                    p = data['p']

                    # temporary store for current trial data. all dictionary entries will be transferred to full data structure (all trials)
                    dtemp = {}

                    # print trial
                    # parameter dictionary
                    dtemp['p'] = copy.copy(p)

                    # retrieve experiment conditions
                    freq = p['pulse_freq']
                    syn_num = p['syn_num']
                    syn_dist = p['syn_dist'][1] 
                    freq_key = str(freq)
                    syn_num_key =str(syn_num)
                    syn_dist_key = str(syn_dist)

                    # update data structure dimensions
                    if str(freq) not in spike_data:
                        spike_data[str(freq)]={}

                    if str(syn_dist) not in spike_data[str(freq)]:
                        spike_data[str(freq)][str(syn_dist)]={}

                    if str(syn_num) not in spike_data[str(freq)][str(syn_dist)]:
                        spike_data[str(freq)][str(syn_dist)][str(syn_num)]={}
                        for polarity_i, polarity in enumerate(p['field']):
                            spike_data[str(freq)][str(syn_dist)][str(syn_num)][str(polarity)]={}

                    # iterate through polarities
                    for polarity_i, polarity in enumerate(p['field']):
                        polarity_key =str(polarity)

                        # for each trial get a list of dendritic spike times and a corresponding list with the location (distance from soma) they occured
                        dtemp['spikes_dend']=[] 
                        dtemp['spikes_dend_dist']=[]
                        dtemp['spikes_first'] = []
                        dtemp['spike_times'] ={}
                        dtemp['dw']={}

                        # add soma data
                        dtemp['spike_times']['soma'] = [[]]
                        # list of spike times [spike times]
                        spike_times = spike_analysis.detect_spikes(data[str(polarity)]['soma_v'][0][0], threshold=threshold)['times'][0]
                        dtemp['spike_times']['soma'][0].append(spike_times)
                        dtemp['spikes_soma'] = spike_times

                        # iterate through tree, section, segment
                        for tree_key, tree in p['sec_idx'].iteritems():
                            dtemp['spike_times'][tree_key] =[]
                            dtemp['dw'][tree_key] =[]
                            for sec_i, sec in enumerate(tree):
                                dtemp['spike_times'][tree_key].append([])
                                dtemp['dw'][tree_key].append([])
                                for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):

                                    # retrieve section and segment number, and associated distance from soma
                                    sec_num = sec
                                    seg_num = p['seg_idx'][tree_key][sec_i][seg_i]
                                    distance = p['seg_dist'][tree_key][sec_num][seg_num]

                                    # list of spike times [spike_times]
                                    spike_times = spike_analysis.detect_spikes(data[str(polarity)][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0]

                                    # scalar weight change
                                    dw = data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][-1]/data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][0]

                                    # add to dtemp structure
                                    dtemp['spike_times'][tree_key][sec_i].append(spike_times)
                                    dtemp['dw'][tree_key][sec_i].append(dw)

                                    # record whether there was a spike in soma or dendrite first [all dendritic spikes]
                                    # no spike=0, dend first=1, soma first=2
                                    # if there are dendritic spikes
                                    if len(spike_times)>0:
                                        # iterate through spike times
                                        for spike_i, spike_time in enumerate(spike_times):

                                            # add to list of all dendritic spike times for this trial/cell, regardless of location [spike times]
                                            dtemp['spikes_dend'].append(spike_time)
                                            # list of distances from soma [distances], indeces correspond with spikes_dend (there can be repeats, i.e. multiple spikes at the same location)
                                            dtemp['spikes_dend_dist'].append(distance)

                                            # if this is the first spike
                                            if spike_i==0:
                                                # and there is also a somatic spike
                                                if len(dtemp['spikes_soma'])>0:
                                                    # if the somatic spike occurs first
                                                    if spike_time > dtemp['spikes_soma'][0]:

                                                        # store as soma first
                                                        dtemp['spikes_first'].append(2)
                                                    # otherwise the spike is dend first
                                                    else:
                                                        dtemp['spikes_first'].append(1)
                                                # if there is a dendritic but no somatic spike, it is dend first
                                                else:
                                                    dtemp['spikes_first'].append(1)
                                    # otherwise no spike at all
                                    else:
                                        dtemp['spikes_first'].append(0)

                        # create nested list of spike times with dimensions [pulse time bins][spike times]
                        dtemp['time_bins'] = []
                        dtemp['spikes_dend_bin'] = []
                        dtemp['spikes_soma_bin'] = []
                        dtemp['spikes_dend_dist_bin'] = []
                        dtemp['spikes_dend_diff_bin'] = []
                        dtemp['syn_frac_bin'] = []
                        for pulse_i, pulse in enumerate(range(p['pulses'])):
                            # determine time bins
                            dtemp['time_bins'].append([])
                            time1 = (p['warmup'] + 1000/p['pulse_freq']*pulse_i)/p['dt'] +1 
                            time2 = (p['warmup'] + 1000/p['pulse_freq']*(pulse_i+1))/p['dt']
                            dtemp['time_bins'][pulse_i].append(time1)
                            dtemp['time_bins'][pulse_i].append(time2)

                            # get spike times that fall within the current bin 
                            binned_spikes_dist = []
                            binned_spikes_dend =[]
                            binned_spikes_soma = []
                            # if theres is a somatic spike
                            if len(dtemp['spikes_soma'])>0:
                                # somatic spikes for the current bin [spike times]
                                binned_spikes_soma = [spike for spike_i, spike in enumerate(dtemp['spikes_soma']) if (spike > time1 and spike <= time2)]
                            # if there is a dendritic spike
                            if len(dtemp['spikes_dend'])>0:
                                # list of dendritic spikes in current bin [spike times] (for all locations)
                                binned_spikes_dend = [spike for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]
                                # list of distances from soma for spikes in current bin [distances]
                                binned_spikes_dist = [dtemp['spikes_dend_dist'][spike_i] for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]

                            # difference between dendritic and somatic spike times within the current bin (dendrite first is positive, soma first is negative)
                            # if there is a dendritic and somatic spike
                            if len(binned_spikes_soma)>0 and len(binned_spikes_dend)>0:
                                # take the difference for the current bin
                                binned_spikes_dend_diff = [binned_spikes_soma[0]-spike for spike_i, spike in enumerate(binned_spikes_dend)]
                            # if there is no somatic spike, just copy dendritic spike times (positive spike time means dendrite first)
                            else: 
                                binned_spikes_dend_diff = binned_spikes_dend

                            # fraction of synapses that spike in current bin
                            binned_distances =  list(set(binned_spikes_dist))
                            binned_syn_frac = float(len(binned_distances))/float(syn_num_key)


                            # add spike times for current bin to list of all bins for current trial
                            dtemp['spikes_dend_bin'].append(binned_spikes_dend)
                            dtemp['spikes_soma_bin'].append(binned_spikes_soma)
                            dtemp['spikes_dend_dist_bin'].append(binned_spikes_dist)
                            dtemp['spikes_dend_diff_bin'].append(binned_spikes_dend_diff)
                            dtemp['syn_frac_bin'].append(binned_syn_frac)

                        # fraction of synapses that spike at all during simulation
                        distances = list(set(dtemp['spikes_dend_dist']))
                        dtemp['syn_frac'] = float(len(distances))/float(syn_num_key)

                        # update main data structure
                        # for each data type
                        for dtype_key, dtype in dtemp.iteritems():
                            
                            # if this type does not already exist in main group data structure
                            if dtype_key not in spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key]:
                                # create list to store data for each trial
                                spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key]=[]

                            # add data for current trial to list 
                            spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key].append(dtype)

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'id list saved'
        
        # save structure of all raw spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data_raw, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'

        
        # plot number of activated synapses vs. mean fraction of synapses with at least one soma/dendrite driven spike
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all='^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions, i.e. number of spikes/number of active synapses [trials]
                        soma_frac=[]
                        dend_frac=[]
                        all_frac=[]
                        syn_num_all=[]
                        # iterate through trials
                        for trial_i, trial in enumerate(polarity['spikes_first']):
                            # soma and dendrite spikes
                            # soma=2, dendrite=1
                            # note that trial contains an entry for all activated synapses, a place is held for none spiking synapses by a 0
                            soma_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==2]
                            dend_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==1]
                            # add to list with dimension [trials]
                            soma_frac.append(float(len(soma_first))/float(len(trial)))
                            dend_frac.append(float(len(dend_first))/float(len(trial)))
                            all_frac.append(float(len(dend_first)+len(soma_first))/float(len(trial)))


                        
                        # group stats
                        soma_frac_mean = np.mean(soma_frac)
                        soma_frac_std = np.std(soma_frac)
                        soma_frac_sem = stats.sem(soma_frac)
                        dend_frac_mean = np.mean(dend_frac)
                        dend_frac_std = np.std(dend_frac)
                        dend_frac_sem = stats.sem(dend_frac)
                        all_frac_mean = np.mean(all_frac)
                        all_frac_std = np.std(all_frac)
                        all_frac_sem = stats.sem(all_frac)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_frac_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_frac_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_frac_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_frac_mean, yerr=soma_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_frac_mean, yerr=dend_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_frac_mean, yerr=all_frac_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('fraction of spiking synapses')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spike_prob_location_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])
        
        
        # plot number of synapses vs. total number of soma/dendritic spikes normalized to total number of synapses
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all='^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions

                        # iterate through trials
                        soma_spikes_total=[]
                        dend_spikes_total=[]
                        all_spikes_total=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            # count somatic/dendritic spikes for each trial
                            soma_count =[]
                            dend_count=[]
                            for time_bin_i, time_bin in enumerate(trial):
                                # list of spike indeces with soma/dend spikes
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                # count total spieks per time bin
                                soma_count.append(float(len(soma_first)))
                                dend_count.append(float(len(dend_first)))
                            
                            syn_num_unique = float(len([seg_i for tree_key, tree in polarity['p'][trial_i]['seg_idx'].iteritems() for sec_i, sec in enumerate(tree) for seg_i, seg in enumerate(sec)]))
                            # syn_num_unique = len(polarity['p']['seg_list'])
                            # number of unique synaptic locations
                            # syn_num_unique = float(len(list(set(polarity['spikes_dend_dist'][trial_i]))))
                            # total spikes per trial, normalized
                            soma_spikes_norm = np.sum(soma_count)/syn_num_unique
                            dend_spikes_norm = np.sum(dend_count)/syn_num_unique
                            # add to list for all trials
                            soma_spikes_total.append(soma_spikes_norm)
                            dend_spikes_total.append(dend_spikes_norm)
                            dend_spikes_total.append(dend_spikes_norm+soma_spikes_norm)
                            
                            # print 'dend_spikes_total:',dend_spikes_total
                        
                        # group stats
                        soma_total_mean = np.mean(soma_spikes_total)
                        soma_total_std = np.std(soma_spikes_total)
                        soma_total_sem = stats.sem(soma_spikes_total)
                        dend_total_mean = np.mean(dend_spikes_total)
                        dend_total_std = np.std(dend_spikes_total)
                        dend_total_sem = stats.sem(dend_spikes_total)
                        all_total_mean = np.mean(all_spikes_total)
                        all_total_std = np.std(all_spikes_total)
                        all_total_sem = stats.sem(all_spikes_total)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_total_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_total_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_total_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_total_mean, yerr=soma_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_total_mean, yerr=dend_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_total_mean, yerr=all_total_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('Number of spikes/synapse')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_total_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])
        
        
        # plot number of synapses vs. mean spike timing
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        dt = polarity['p'][0]['dt']
                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions
                        soma_frac=[]
                        dend_frac=[]
                        # iterate through trials
                        soma_timing=[]
                        dend_timing=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            for time_bin_i, time_bin in enumerate(trial):
                                onset = polarity['time_bins'][trial_i][time_bin_i][0]
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                soma_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in soma_first]
                                dend_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in dend_first]
                                soma_timing.append(soma_time)
                                dend_timing.append(dend_time)

                        soma_timing_flat = [time*dt for times in soma_timing for time in times]
                        dend_timing_flat = [time*dt for times in dend_timing for time in times]
                        
                        # group stats
                        soma_timing_mean = np.mean(soma_timing_flat)
                        soma_timing_std = np.std(soma_timing_flat)
                        soma_timing_sem = stats.sem(soma_timing_flat)
                        dend_timing_mean = np.mean(dend_timing_flat)
                        dend_timing_std = np.std(dend_timing_flat)
                        dend_timing_sem = stats.sem(dend_timing_flat)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_timing_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_timing_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_timing_mean, yerr=soma_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_timing_mean, yerr=dend_timing_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('spike timing after epsp onset (ms)')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_timing_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])

    def exp_1b(self, **kwargs):
        """
        activate a varying number of synapses with varying distance from the soma.  Synapses are chosen from a window of 200 um, with the window moving along the apical dendrite.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of mean distance from the soma and frequency of synaptic activity.  

        Similar to 1a, with larger distance window for synapses to be activated

        Plots: 
        number of active synapse x fraction of synapses with at least one spike (somatically and dendritically driven spikes are separated)

        number of active synapses x total number of spikes (normalized to number of active synapses, somatic and dendritic spikes are separated)

        number of active synapses x mean spike timing for each spike
        """
        # spike threshold 
        threshold =-30

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        
        # all files in directory
        files = os.listdir(data_folder)
        
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        # if list of processed files in folder, load list
        id_list_string_spike_times = 'id_list_spike_times.pkl'
        if id_list_string_spike_times in files:
            print 'id_list found'
            
            with open(data_folder+id_list_string_spike_times, 'rb') as pkl_file:
                id_list_spike_times = pickle.load(pkl_file)

            print 'id list loaded'
        
        # otherwise create empty list
        else:
            id_list_spike_times = []

        # string to save group data
        save_string_group_data_raw = 'spikes_grouped_raw'+'.pkl'
        # if data file already exists
        if save_string_group_data_raw in files:
            # load data
            print 'raw spike data found'
            with open(data_folder+save_string_group_data_raw, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
            print 'raw spike data loaded'
        # otherwise create data structure
        else:
            # data organized as {frequency}{syn distance}{number of synapses}{polarity}[trial]{data type}{tree}[section][segment][spikes]
            spike_data= {}


        # load spike analysis functions
        spike_analysis = Spikes()

        dtypes = ['spike_times','dw','p']
    
        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:# and data_file_i <=1000:

                # check if data has been processed already
                if data_file not in id_list_spike_times:

                    print data_file
                    # open unprocessed data

                    try:
                        with open(data_folder+data_file, 'rb') as pkl_file:
                            data = pickle.load(pkl_file)

                        print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'
                    except EOFError:
                        'EOF error, file not opened'
                        continue

                    # add to processed list
                    id_list_spike_times.append(data_file)

                    # parameter dictionary
                    p = data['p']

                    # temporary store for current trial data. all dictionary entries will be transferred to full data structure (all trials)
                    dtemp = {}

                    # print trial
                    # parameter dictionary
                    dtemp['p'] = copy.copy(p)

                    # retrieve experiment conditions
                    freq = p['pulse_freq']
                    syn_num = p['syn_num']
                    syn_dist = p['syn_dist'][1] 
                    freq_key = str(freq)
                    syn_num_key =str(syn_num)
                    syn_dist_key = str(syn_dist)

                    # update data structure dimensions
                    if str(freq) not in spike_data:
                        spike_data[str(freq)]={}

                    if str(syn_dist) not in spike_data[str(freq)]:
                        spike_data[str(freq)][str(syn_dist)]={}

                    if str(syn_num) not in spike_data[str(freq)][str(syn_dist)]:
                        spike_data[str(freq)][str(syn_dist)][str(syn_num)]={}
                        for polarity_i, polarity in enumerate(p['field']):
                            spike_data[str(freq)][str(syn_dist)][str(syn_num)][str(polarity)]={}

                    # iterate through polarities
                    for polarity_i, polarity in enumerate(p['field']):
                        polarity_key =str(polarity)



                        # for each trial get a list of dendritic spike times and a corresponding list with the location (distance from soma) they occured
                        dtemp['spikes_dend']=[] 
                        dtemp['spikes_dend_dist']=[]
                        dtemp['spikes_first'] = []
                        dtemp['spike_times'] ={}
                        dtemp['dw']={}

                        # add soma data
                        dtemp['spike_times']['soma'] = [[]]
                        spike_times = spike_analysis.detect_spikes(data[str(polarity)]['soma_v'][0][0], threshold=threshold)['times'][0]
                        dtemp['spike_times']['soma'][0].append(spike_times)
                        dtemp['spikes_soma'] = spike_times
                        # print spike_times


                        # iterate through tree, section, segment
                        for tree_key, tree in p['sec_idx'].iteritems():
                            dtemp['spike_times'][tree_key] =[]
                            dtemp['dw'][tree_key] =[]
                        
                            for sec_i, sec in enumerate(tree):

                                dtemp['spike_times'][tree_key].append([])
                                dtemp['dw'][tree_key].append([])

                                for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):

                                    sec_num = sec
                                    seg_num = p['seg_idx'][tree_key][sec_i][seg_i]
                                    distance = p['seg_dist'][tree_key][sec_num][seg_num]

                                    # list of spike times [spike_times]
                                    spike_times = spike_analysis.detect_spikes(data[str(polarity)][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0]

                                    # scalar weight change
                                    dw = data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][-1]/data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][0]

                                    # add to dtemp structure
                                    dtemp['spike_times'][tree_key][sec_i].append(spike_times)
                                    dtemp['dw'][tree_key][sec_i].append(dw)

                                    # record whether whether there was a spike in soma or dendrite first [all dendritic spike]
                                    # no spike=0, dend first=1, soma first=2
                                    # if there are dendritic spikes
                                    if len(spike_times)>0:
                                        # iterate through spike times
                                        for spike_i, spike_time in enumerate(spike_times):

                                            # add to list of all dendritic spike times for this trial/cell
                                            dtemp['spikes_dend'].append(spike_time)
                                            dtemp['spikes_dend_dist'].append(distance)

                                            # if this is the first spike
                                            if spike_i==0:
                                                # if there is also a somatic spike
                                                if len(dtemp['spikes_soma'])>0:
                                                    # if the somatic spike occurs first
                                                    if spike_time > dtemp['spikes_soma'][0]:

                                                        # store as soma first
                                                        dtemp['spikes_first'].append(2)
                                                    # otherwise the spike is dend first
                                                    else:
                                                        dtemp['spikes_first'].append(1)
                                                # if there is a dendritic but no somatic spike, it is dend first
                                                else:
                                                    dtemp['spikes_first'].append(1)
                                    # otherwise no spike at all
                                    else:
                                        dtemp['spikes_first'].append(0)

                        # create nested list of spike times with dimensions [pulse time bins][spike times]
                        dtemp['time_bins'] = []
                        dtemp['spikes_dend_bin'] = []
                        dtemp['spikes_soma_bin'] = []
                        dtemp['spikes_dend_dist_bin'] = []
                        dtemp['spikes_dend_diff_bin'] = []
                        dtemp['syn_frac_bin'] = []
                        for pulse_i, pulse in enumerate(range(p['pulses'])):
                            # determine time bins
                            dtemp['time_bins'].append([])
                            time1 = (p['warmup'] + 1000/p['pulse_freq']*pulse_i)/p['dt'] +1 
                            time2 = (p['warmup'] + 1000/p['pulse_freq']*(pulse_i+1))/p['dt']
                            dtemp['time_bins'][pulse_i].append(time1)
                            dtemp['time_bins'][pulse_i].append(time2)

                            # get spike times that fall within the current bin 
                            binned_spikes_dist = []
                            binned_spikes_dend =[]
                            binned_spikes_soma = []
                            if len(dtemp['spikes_soma'])>0:
                                # print dtemp['spikes_soma']
                                binned_spikes_soma = [spike for spike_i, spike in enumerate(dtemp['spikes_soma']) if (spike > time1 and spike <= time2)]
                            if len(dtemp['spikes_dend'])>0:
                                binned_spikes_dend = [spike for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]
                                binned_spikes_dist = [dtemp['spikes_dend_dist'][spike_i] for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]
                            # difference between dendritic and somatic spike times (dendrite first is positive, soma first is negative)
                            if len(binned_spikes_soma)>0 and len(binned_spikes_dend)>0:
                                binned_spikes_dend_diff = [binned_spikes_soma[0]-spike for spike_i, spike in enumerate(binned_spikes_dend)]
                            else: #len(binned_spikes_dend)>0:
                                binned_spikes_dend_diff = binned_spikes_dend

                            # fraction of synapses that spike in current bin
                            binned_distances =  list(set(binned_spikes_dist))
                            binned_syn_frac = float(len(binned_distances))/float(syn_num_key)
                            

                            # add spike times for current bin to list of all bins
                            dtemp['spikes_dend_bin'].append(binned_spikes_dend)
                            dtemp['spikes_soma_bin'].append(binned_spikes_soma)
                            dtemp['spikes_dend_dist_bin'].append(binned_spikes_dist)
                            dtemp['spikes_dend_diff_bin'].append(binned_spikes_dend_diff)
                            dtemp['syn_frac_bin'].append(binned_syn_frac)

                         # fraction of synapses that spike at all during simulation
                        distances = list(set(dtemp['spikes_dend_dist']))
                        dtemp['syn_frac'] = float(len(distances))/float(syn_num_key)

                        # update main data structure
                        # for each data type
                        for dtype_key, dtype in dtemp.iteritems():
                            
                            # if this type does not already exist in main group data structure
                            if dtype_key not in spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key]:
                                # create list to store data for each trial
                                spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key]=[]

                            # add data for current trial to list 
                            spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key].append(dtype)

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'id list saved'
        
        # save structure of all raw spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data_raw, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'


        # plot number of activated synapses vs. mean fraction of synapses with at least one soma/dendrite driven spike
        plots={}
        plots_dw={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            plots_dw[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                plots_dw[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 20.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all= '^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions, i.e. number of spikes/number of active synapses [trials]
                        soma_frac=[]
                        dend_frac=[]
                        all_frac=[]
                        dw_list=[]
                        # iterate through trials
                        for trial_i, trial in enumerate(polarity['spikes_first']):
                            # soma and dendrite spikes
                            # soma=2, dendrite=1
                            # note that trial contains an entry for all activated synapses, a place is held for none spiking synapses by a 0
                            soma_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==2]
                            dend_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==1]
                            # add to list with dimension [trials]
                            soma_frac.append(float(len(soma_first))/float(len(trial)))
                            dend_frac.append(float(len(dend_first))/float(len(trial)))
                            all_frac.append(float(len(dend_first)+len(soma_first))/float(len(trial)))
                            dw_all = polarity['dw'][trial_i]
                            syn_count = 0
                            dw_trial=[]
                            for tree_key, tree in dw_all.iteritems():
                                for sec_i, sec in enumerate(tree):
                                    for seg_i, seg in enumerate(sec):
                                        syn_count+=1
                                        dw_trial.append(seg)
                            dw_mean = sum(dw_trial)/float(syn_count)
                            dw_list.append(dw_mean)

                        dw_mean = np.mean(dw_list)
                        dw_std = np.std(dw_list)
                        dw_sem = stats.sem(dw_list)

                        # group stats
                        soma_frac_mean = np.mean(soma_frac)
                        soma_frac_std = np.std(soma_frac)
                        soma_frac_sem = stats.sem(soma_frac)
                        dend_frac_mean = np.mean(dend_frac)
                        dend_frac_std = np.std(dend_frac)
                        dend_frac_sem = stats.sem(dend_frac)
                        all_frac_mean = np.mean(all_frac)
                        all_frac_std = np.std(all_frac)
                        all_frac_sem = stats.sem(all_frac)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_frac_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_frac_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_frac_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_frac_mean, yerr=soma_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_frac_mean, yerr=dend_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_frac_mean, yerr=all_frac_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('fraction of spiking synapses')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                         # plot dw with errorbars
                        plt.figure(plots_dw[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), dw_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dw_mean, yerr=dw_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('average weight change')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spike_prob_location_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])

                # save and close figure
                plt.figure(plots_dw[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_dw_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots_dw[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots_dw[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots_dw[freq_key][syn_dist_key])
        
        
        # plot number of synapses vs. total number of soma/dendritic spikes normalized to total number of synapses
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 20.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all='^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions

                        # iterate through trials
                        soma_spikes_total=[]
                        dend_spikes_total=[]
                        all_spikes_total=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            # count somatic/dendritic spikes for each trial
                            soma_count =[]
                            dend_count=[]
                            for time_bin_i, time_bin in enumerate(trial):
                                # list of spike indeces with soma/dend spikes
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                # count total spieks per time bin
                                soma_count.append(float(len(soma_first)))
                                dend_count.append(float(len(dend_first)))
                            
                            syn_num_unique = float(len([seg_i for tree_key, tree in polarity['p'][trial_i]['seg_idx'].iteritems() for sec_i, sec in enumerate(tree) for seg_i, seg in enumerate(sec)]))
                            # syn_num_unique = len(polarity['p']['seg_list'])
                            # number of unique synaptic locations
                            # syn_num_unique = float(len(list(set(polarity['spikes_dend_dist'][trial_i]))))
                            # total spikes per trial, normalized
                            soma_spikes_norm = np.sum(soma_count)/syn_num_unique
                            dend_spikes_norm = np.sum(dend_count)/syn_num_unique
                            # add to list for all trials
                            soma_spikes_total.append(soma_spikes_norm)
                            dend_spikes_total.append(dend_spikes_norm)
                            all_spikes_total.append(dend_spikes_norm+soma_spikes_norm)
                            
                            # print 'dend_spikes_total:',dend_spikes_total
                        
                        # group stats
                        soma_total_mean = np.mean(soma_spikes_total)
                        soma_total_std = np.std(soma_spikes_total)
                        soma_total_sem = stats.sem(soma_spikes_total)
                        dend_total_mean = np.mean(dend_spikes_total)
                        dend_total_std = np.std(dend_spikes_total)
                        dend_total_sem = stats.sem(dend_spikes_total)
                        all_total_mean = np.mean(all_spikes_total)
                        all_total_std = np.std(all_spikes_total)
                        all_total_sem = stats.sem(all_spikes_total)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_total_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity, )
                        plt.plot(float(syn_num_key), dend_total_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity, )
                        plt.plot(float(syn_num_key), all_total_mean, color=color, marker=marker_all, markersize=size, alpha=opacity, )
                        plt.errorbar(float(syn_num_key), all_total_mean, yerr=all_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_total_mean, yerr=soma_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_total_mean, yerr=dend_total_sem, color=color, alpha=opacity)
                        # plt.xlabel('Active synapses', fontsize=30, weight='bold')
                        # plt.ylabel('Spikes/synapse', fontsize=30, weight='bold')
                        # if syn_dist_key=='200':
                        #     plt.xlim([0, 40])
                        plt.xticks(fontsize=10, weight='bold')
                        plt.yticks(fontsize=10, weight='bold')
                        plt.title(syn_dist_key + ' um from soma', fontsize=30, weight='bold')
                        # plt.legend()

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                black_patch = patches.Patch(color='black', label='Control')
                red_patch =patches.Patch(color='red', label='Anodal')
                blue_patch =patches.Patch(color='blue', label='Cathodal')
                triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='All spikes')
                dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Somatic')
                cross = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Dendritic')
                # plt.legend(handles=[black_patch, red_patch, blue_patch, triangle, dot, cross])
                plot_file_name = 'syn_num_x_spikes_total_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250, bbox_inches='tight')
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])
        
    
        # plot number of synapses vs. mean spike timing
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        dt = polarity['p'][0]['dt']
                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions
                        soma_frac=[]
                        dend_frac=[]
                        # iterate through trials
                        soma_timing=[]
                        dend_timing=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            for time_bin_i, time_bin in enumerate(trial):
                                onset = polarity['time_bins'][trial_i][time_bin_i][0]
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                soma_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in soma_first]
                                dend_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in dend_first]
                                soma_timing.append(soma_time)
                                dend_timing.append(dend_time)

                        soma_timing_flat = [time*dt for times in soma_timing for time in times]
                        dend_timing_flat = [time*dt for times in dend_timing for time in times]
                        
                        # group stats
                        soma_timing_mean = np.mean(soma_timing_flat)
                        soma_timing_std = np.std(soma_timing_flat)
                        soma_timing_sem = stats.sem(soma_timing_flat)
                        dend_timing_mean = np.mean(dend_timing_flat)
                        dend_timing_std = np.std(dend_timing_flat)
                        dend_timing_sem = stats.sem(dend_timing_flat)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_timing_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_timing_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_timing_mean, yerr=soma_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_timing_mean, yerr=dend_timing_sem, color=color, alpha=opacity)
                        plt.xlabel('Number of active synapses', fontsize=30, weight='bold')
                        plt.ylabel('Spike timing after epsp onset (ms)', fontsize=30, weight='bold')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_timing_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])

    def exp_1c(self, **kwargs):
        """ 
        activate a varying number of synapses in proximal (0-200/0-300 um) and distal regions (400-600/300-600 um) simultameously.  Synapses are chosen from a window of 200 or 300 um.  As number of synapses is increased, multiple synapses may impinge on the same compartment/segment, effectively increasing the weight in that segment.  The threshold for generating a somatic or dendritic spike (in number of synapses) is measured as a function of nnumber of synapses. Does pairing with proximal inputs (e.g. 0-200 um) cause distal inputs (eg 400-600 um) to come under greater control from the soma? 

        Similar to 1a and 1b, now pairing two distance windows (proximal and distal)

        Plots: 
        number of active synapse x fraction of synapses with at least one spike (somatically and dendritically driven spikes are separated)

        number of active synapses x total number of spikes (normalized to number of active synapses, somatic and dendritic spikes are separated)

        number of active synapses x mean spike timing for each spike

        """
        # spike threshold 
        threshold =-30

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        
        # all files in directory
        files = os.listdir(data_folder)
        
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        # if list of processed files in folder, load list
        id_list_string_spike_times = 'id_list_spike_times.pkl'
        if id_list_string_spike_times in files:
            print 'id_list found'
            
            with open(data_folder+id_list_string_spike_times, 'rb') as pkl_file:
                id_list_spike_times = pickle.load(pkl_file)

            print 'id list loaded'
        
        # otherwise create empty list
        else:
            id_list_spike_times = []

        # string to save group data
        save_string_group_data_raw = 'spikes_grouped_raw'+'.pkl'
        # if data file already exists
        if save_string_group_data_raw in files:
            # load data
            print 'raw spike data found'
            with open(data_folder+save_string_group_data_raw, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
            print 'raw spike data loaded'
        # otherwise create data structure
        else:
            # data organized as {frequency}{syn distance}{number of synapses}{polarity}[trial]{data type}{tree}[section][segment][spikes]
            spike_data= {}

        # load spike analysis functions
        spike_analysis = Spikes()

        dtypes = ['spike_times','dw','p']
        
        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:# and data_file_i <=1000:

                # check if data has been processed already
                if data_file not in id_list_spike_times:

                    print data_file
                    # open unprocessed data

                    try:
                        with open(data_folder+data_file, 'rb') as pkl_file:
                            data = pickle.load(pkl_file)

                        print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'
                    except EOFError:
                        'EOF error, file not opened'
                        continue

                    # add to processed list
                    id_list_spike_times.append(data_file)

                    # parameter dictionary
                    p = data['p']

                    # temporary store for current trial data. all dictionary entries will be transferred to full data structure (all trials)
                    dtemp = {}

                    # print trial
                    # parameter dictionary
                    dtemp['p'] = copy.copy(p)

                    # retrieve experiment conditions
                    freq = p['pulse_freq']
                    syn_num = p['syn_num']
                    syn_dist = p['syn_dist'][-1] 
                    freq_key = str(freq)
                    syn_num_key =str(syn_num)
                    syn_dist_key = str(syn_dist)

                    # update data structure dimensions
                    if str(freq) not in spike_data:
                        spike_data[str(freq)]={}

                    if str(syn_dist) not in spike_data[str(freq)]:
                        spike_data[str(freq)][str(syn_dist)]={}

                    if str(syn_num) not in spike_data[str(freq)][str(syn_dist)]:
                        spike_data[str(freq)][str(syn_dist)][str(syn_num)]={}
                        for polarity_i, polarity in enumerate(p['field']):
                            spike_data[str(freq)][str(syn_dist)][str(syn_num)][str(polarity)]={}

                    # iterate through polarities
                    for polarity_i, polarity in enumerate(p['field']):
                        polarity_key =str(polarity)



                        # for each trial get a list of dendritic spike times and a corresponding list with the location (distance from soma) they occured
                        dtemp['spikes_dend']=[] 
                        dtemp['spikes_dend_dist']=[]
                        dtemp['spikes_first'] = []
                        dtemp['spike_times'] ={}
                        dtemp['dw']={}

                        # add soma data
                        dtemp['spike_times']['soma'] = [[]]
                        spike_times = spike_analysis.detect_spikes(data[str(polarity)]['soma_v'][0][0], threshold=threshold)['times'][0]
                        dtemp['spike_times']['soma'][0].append(spike_times)
                        dtemp['spikes_soma'] = spike_times
                        # print spike_times


                        # iterate through tree, section, segment
                        for tree_key, tree in p['sec_idx'].iteritems():
                            dtemp['spike_times'][tree_key] =[]
                            dtemp['dw'][tree_key] =[]
                        
                            for sec_i, sec in enumerate(tree):

                                dtemp['spike_times'][tree_key].append([])
                                dtemp['dw'][tree_key].append([])

                                for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):

                                    sec_num = sec
                                    seg_num = p['seg_idx'][tree_key][sec_i][seg_i]
                                    distance = p['seg_dist'][tree_key][sec_num][seg_num]

                                    # distance requirement, only include synapses in the distal region
                                    if distance > p['syn_dist'][-1][0] and distance <= p['syn_dist'][-1][1]:  


                                        # list of spike times [spike_times]
                                        spike_times = spike_analysis.detect_spikes(data[str(polarity)][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0]

                                        # scalar weight change
                                        dw = data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][-1]/data[str(polarity)][tree_key+'_gbar'][sec_i][seg_i][0]

                                        # add to dtemp structure
                                        dtemp['spike_times'][tree_key][sec_i].append(spike_times)
                                        dtemp['dw'][tree_key][sec_i].append(dw)

                                        # record whether whether there was a spike in soma or dendrite first [all dendritic spike]
                                        # no spike=0, dend first=1, soma first=2
                                        # if there are dendritic spikes
                                        if len(spike_times)>0:
                                            # iterate through spike times
                                            for spike_i, spike_time in enumerate(spike_times):

                                                # add to list of all dendritic spike times for this trial/cell
                                                dtemp['spikes_dend'].append(spike_time)
                                                dtemp['spikes_dend_dist'].append(distance)
                                                print 'distance:',distance

                                                # if this is the first spike
                                                if spike_i==0:
                                                    # if there is also a somatic spike
                                                    if len(dtemp['spikes_soma'])>0:
                                                        # if the somatic spike occurs first
                                                        if spike_time > dtemp['spikes_soma'][0]:

                                                            # store as soma first
                                                            dtemp['spikes_first'].append(2)
                                                        # otherwise the spike is dend first
                                                        else:
                                                            dtemp['spikes_first'].append(1)
                                                    # if there is a dendritic but no somatic spike, it is dend first
                                                    else:
                                                        dtemp['spikes_first'].append(1)
                                        # otherwise no spike at all
                                        else:
                                            dtemp['spikes_first'].append(0)

                        # create nested list of spike times with dimensions [pulse time bins][spike times]
                        dtemp['time_bins'] = []
                        dtemp['spikes_dend_bin'] = []
                        dtemp['spikes_soma_bin'] = []
                        dtemp['spikes_dend_dist_bin'] = []
                        dtemp['spikes_dend_diff_bin'] = []
                        dtemp['syn_frac_bin'] = []
                        for pulse_i, pulse in enumerate(range(p['pulses'])):
                            # determine time bins
                            dtemp['time_bins'].append([])
                            time1 = (p['warmup'] + 1000/p['pulse_freq']*pulse_i)/p['dt'] +1 
                            time2 = (p['warmup'] + 1000/p['pulse_freq']*(pulse_i+1))/p['dt']
                            dtemp['time_bins'][pulse_i].append(time1)
                            dtemp['time_bins'][pulse_i].append(time2)

                            # get spike times that fall within the current bin 
                            binned_spikes_dist = []
                            binned_spikes_dend =[]
                            binned_spikes_soma = []
                            if len(dtemp['spikes_soma'])>0:
                                # print dtemp['spikes_soma']
                                binned_spikes_soma = [spike for spike_i, spike in enumerate(dtemp['spikes_soma']) if (spike > time1 and spike <= time2)]
                            if len(dtemp['spikes_dend'])>0:
                                binned_spikes_dend = [spike for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]
                                binned_spikes_dist = [dtemp['spikes_dend_dist'][spike_i] for spike_i, spike in enumerate(dtemp['spikes_dend']) if (spike > time1 and spike <= time2)]
                            # difference between dendritic and somatic spike times (dendrite first is positive, soma first is negative)
                            if len(binned_spikes_soma)>0 and len(binned_spikes_dend)>0:
                                binned_spikes_dend_diff = [binned_spikes_soma[0]-spike for spike_i, spike in enumerate(binned_spikes_dend)]
                            else: #len(binned_spikes_dend)>0:
                                binned_spikes_dend_diff = binned_spikes_dend

                            # fraction of synapses that spike in current bin
                            binned_distances =  list(set(binned_spikes_dist))
                            binned_syn_frac = float(len(binned_distances))/float(syn_num_key)
                            

                            # add spike times for current bin to list of all bins
                            dtemp['spikes_dend_bin'].append(binned_spikes_dend)
                            dtemp['spikes_soma_bin'].append(binned_spikes_soma)
                            dtemp['spikes_dend_dist_bin'].append(binned_spikes_dist)
                            dtemp['spikes_dend_diff_bin'].append(binned_spikes_dend_diff)
                            dtemp['syn_frac_bin'].append(binned_syn_frac)

                         # fraction of synapses that spike at all during simulation
                        distances = list(set(dtemp['spikes_dend_dist']))
                        dtemp['syn_frac'] = float(len(distances))/float(syn_num_key)

                        # update main data structure
                        for dtype_key, dtype in dtemp.iteritems():
                            if dtype_key not in spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key]:
                                spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key]=[]
                            spike_data[freq_key][syn_dist_key][syn_num_key][polarity_key][dtype_key].append(dtype)

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'id list saved'
        
        # save structure of all raw spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data_raw, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'

        
        # plot number of synapses vs. mean fraction of synapses with at least one soma/dendrite driven spike
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all= '^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions
                        soma_frac=[]
                        dend_frac=[]
                        all_frac=[]
                        # iterate through trials
                        for trial_i, trial in enumerate(polarity['spikes_first']):
                            # soma and dendrite spikes
                            # soma=2, dendrite=1
                            soma_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==2]
                            dend_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==1]
                            # add to list with dimension [trials]
                            soma_frac.append(float(len(soma_first))/float(len(trial)))
                            dend_frac.append(float(len(dend_first))/float(len(trial)))
                            all_frac.append(float(len(dend_first)+len(soma_first))/float(len(trial)))

                        # group stats
                        soma_frac_mean = np.mean(soma_frac)
                        soma_frac_std = np.std(soma_frac)
                        soma_frac_sem = stats.sem(soma_frac)
                        dend_frac_mean = np.mean(dend_frac)
                        dend_frac_std = np.std(dend_frac)
                        dend_frac_sem = stats.sem(dend_frac)
                        all_frac_mean = np.mean(all_frac)
                        all_frac_std = np.std(all_frac)
                        all_frac_sem = stats.sem(all_frac)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_frac_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_frac_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_frac_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_frac_mean, yerr=soma_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_frac_mean, yerr=dend_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_frac_mean, yerr=all_frac_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('fraction of spiking synapses')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spike_prob_location_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])
        
        
        # plot number of synapses vs. total number of soma/dendritic spikes normalized to total number of synapses
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():


                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all= '^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions

                        # iterate through trials
                        soma_spikes_total=[]
                        dend_spikes_total=[]
                        all_spikes_total=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            # count somatic/dendritic spikes for each trial
                            soma_count =[]
                            dend_count=[]
                            for time_bin_i, time_bin in enumerate(trial):
                                # list of spike indeces with soma/dend spikes
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                # count total spieks per time bin
                                soma_count.append(float(len(soma_first)))
                                dend_count.append(float(len(dend_first)))
                            
                            syn_num_unique = float(len([seg_i for tree_key, tree in polarity['p'][trial_i]['seg_idx'].iteritems() for sec_i, sec in enumerate(tree) for seg_i, seg in enumerate(sec)]))
                            # syn_num_unique = len(polarity['p']['seg_list'])
                            # number of unique synaptic locations
                            # syn_num_unique = float(len(list(set(polarity['spikes_dend_dist'][trial_i]))))
                            # total spikes per trial, normalized
                            soma_spikes_norm = np.sum(soma_count)/syn_num_unique
                            dend_spikes_norm = np.sum(dend_count)/syn_num_unique
                            # add to list for all trials
                            soma_spikes_total.append(soma_spikes_norm)
                            dend_spikes_total.append(dend_spikes_norm)
                            all_spikes_total.append(soma_spikes_norm+dend_spikes_norm)
                            
                            # print 'dend_spikes_total:',dend_spikes_total
                        
                        # group stats
                        soma_total_mean = np.mean(soma_spikes_total)
                        soma_total_std = np.std(soma_spikes_total)
                        soma_total_sem = stats.sem(soma_spikes_total)
                        dend_total_mean = np.mean(dend_spikes_total)
                        dend_total_std = np.std(dend_spikes_total)
                        dend_total_sem = stats.sem(dend_spikes_total)
                        all_total_mean = np.mean(all_spikes_total)
                        all_total_std = np.std(all_spikes_total)
                        all_total_sem = stats.sem(all_spikes_total)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_total_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_total_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_total_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_total_mean, yerr=soma_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_total_mean, yerr=dend_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_total_mean, yerr=all_total_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('Number of spikes/synapse')
                        soma_frac=[]
                        dend_frac=[]
                        all_frac=[]
                        # iterate through trials
                        for trial_i, trial in enumerate(polarity['spikes_first']):
                            # soma and dendrite spikes
                            # soma=2, dendrite=1
                            soma_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==2]
                            dend_first = [spike_i for spike_i, spike_loc in enumerate(trial) if spike_loc==1]
                            # add to list with dimension [trials]
                            soma_frac.append(float(len(soma_first))/float(len(trial)))
                            dend_frac.append(float(len(dend_first))/float(len(trial)))
                            all_frac.append(float(len(dend_first)+len(soma_first))/float(len(trial)))

                        # group stats
                        soma_frac_mean = np.mean(soma_frac)
                        soma_frac_std = np.std(soma_frac)
                        soma_frac_sem = stats.sem(soma_frac)
                        dend_frac_mean = np.mean(dend_frac)
                        dend_frac_std = np.std(dend_frac)
                        dend_frac_sem = stats.sem(dend_frac)
                        all_frac_mean = np.mean(all_frac)
                        all_frac_std = np.std(all_frac)
                        all_frac_sem = stats.sem(all_frac)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_frac_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_frac_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_frac_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_frac_mean, yerr=soma_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_frac_mean, yerr=dend_frac_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_frac_mean, yerr=all_frac_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('fraction of spiking synapses')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_total_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plot_file_name = 'syn_num_x_spike_prob_location_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])
        
        
        # plot number of synapses vs. mean spike timing
        # plot number of synapses vs. total number of soma/dendritic spikes normalized to total number of synapses
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        dt = polarity['p'][0]['dt']
                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all='^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions
                        soma_frac=[]
                        dend_frac=[]
                        # iterate through trials
                        soma_timing=[]
                        dend_timing=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            for time_bin_i, time_bin in enumerate(trial):
                                onset = polarity['time_bins'][trial_i][time_bin_i][0]
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                soma_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in soma_first]
                                dend_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in dend_first]
                                soma_timing.append(soma_time)
                                dend_timing.append(dend_time)

                        soma_timing_flat = [time*dt for times in soma_timing for time in times]
                        dend_timing_flat = [time*dt for times in dend_timing for time in times]
                        all_timing_flat = soma_timing_flat+dend_timing_flat
                        
                        # group stats
                        soma_timing_mean = np.mean(soma_timing_flat)
                        soma_timing_std = np.std(soma_timing_flat)
                        soma_timing_sem = stats.sem(soma_timing_flat)
                        dend_timing_mean = np.mean(dend_timing_flat)
                        dend_timing_std = np.std(dend_timing_flat)
                        dend_timing_sem = stats.sem(dend_timing_flat)
                        all_timing_mean = np.mean(all_timing_flat)
                        all_timing_std = np.std(all_timing_flat)
                        all_timing_sem = stats.sem(all_timing_flat)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_timing_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_timing_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_timing_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_timing_mean, yerr=soma_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_timing_mean, yerr=dend_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_timing_mean, yerr=all_timing_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('spike timing after epsp onset (ms)')
                        marker_all= '^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions

                        # iterate through trials
                        soma_spikes_total=[]
                        dend_spikes_total=[]
                        all_spikes_total=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            # count somatic/dendritic spikes for each trial
                            soma_count =[]
                            dend_count=[]
                            for time_bin_i, time_bin in enumerate(trial):
                                # list of spike indeces with soma/dend spikes
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                # count total spieks per time bin
                                soma_count.append(float(len(soma_first)))
                                dend_count.append(float(len(dend_first)))
                            
                            syn_num_unique = float(len([seg_i for tree_key, tree in polarity['p'][trial_i]['seg_idx'].iteritems() for sec_i, sec in enumerate(tree) for seg_i, seg in enumerate(sec)]))
                            # syn_num_unique = len(polarity['p']['seg_list'])
                            # number of unique synaptic locations
                            # syn_num_unique = float(len(list(set(polarity['spikes_dend_dist'][trial_i]))))
                            # total spikes per trial, normalized
                            soma_spikes_norm = np.sum(soma_count)/syn_num_unique
                            dend_spikes_norm = np.sum(dend_count)/syn_num_unique
                            # add to list for all trials
                            soma_spikes_total.append(soma_spikes_norm)
                            dend_spikes_total.append(dend_spikes_norm)
                            all_spikes_total.append(soma_spikes_norm+dend_spikes_norm)
                            
                            # print 'dend_spikes_total:',dend_spikes_total
                        
                        # group stats
                        soma_total_mean = np.mean(soma_spikes_total)
                        soma_total_std = np.std(soma_spikes_total)
                        soma_total_sem = stats.sem(soma_spikes_total)
                        dend_total_mean = np.mean(dend_spikes_total)
                        dend_total_std = np.std(dend_spikes_total)
                        dend_total_sem = stats.sem(dend_spikes_total)
                        all_total_mean = np.mean(all_spikes_total)
                        all_total_std = np.std(all_spikes_total)
                        all_total_sem = stats.sem(all_spikes_total)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_total_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_total_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_total_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_total_mean, yerr=soma_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_total_mean, yerr=dend_total_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_total_mean, yerr=all_total_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('Number of spikes/synapse')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_timing_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plot_file_name = 'syn_num_x_spikes_total_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])

        
        
        # plot number of synapses vs. mean spike timing
        plots={}
        for freq_key, freq in spike_data.iteritems():
            plots[freq_key] = {}
            for syn_dist_key, syn_dist in freq.iteritems():
                plots[freq_key][syn_dist_key] = plt.figure()
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():

                        dt = polarity['p'][0]['dt']
                        # get polarity index
                        polarity_i = [f_i for f_i, f in enumerate(polarity['p'][0]['field']) if str(f)==polarity_key][0]

                        # plot color and marker
                        color = polarity['p'][0]['field_color'][polarity_i]
                        # size = 20.*float(syn_dist_key)/600.
                        size = 10.
                        opacity = 0.7
                        marker_soma = '.'
                        marker_dend= 'x'
                        marker_all='^'

                        # lsit of soma/dend spike fraction for each trial in current set of conditions
                        soma_frac=[]
                        dend_frac=[]
                        # iterate through trials
                        soma_timing=[]
                        dend_timing=[]
                        for trial_i, trial in enumerate(polarity['spikes_dend_diff_bin']):
                            for time_bin_i, time_bin in enumerate(trial):
                                onset = polarity['time_bins'][trial_i][time_bin_i][0]
                                soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                soma_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in soma_first]
                                dend_time = [polarity['spikes_dend_bin'][trial_i][time_bin_i][spike_i]-onset for spike_i in dend_first]
                                soma_timing.append(soma_time)
                                dend_timing.append(dend_time)

                        soma_timing_flat = [time*dt for times in soma_timing for time in times]
                        dend_timing_flat = [time*dt for times in dend_timing for time in times]
                        all_timing_flat = soma_timing_flat+dend_timing_flat
                        
                        # group stats
                        soma_timing_mean = np.mean(soma_timing_flat)
                        soma_timing_std = np.std(soma_timing_flat)
                        soma_timing_sem = stats.sem(soma_timing_flat)
                        dend_timing_mean = np.mean(dend_timing_flat)
                        dend_timing_std = np.std(dend_timing_flat)
                        dend_timing_sem = stats.sem(dend_timing_flat)
                        all_timing_mean = np.mean(all_timing_flat)
                        all_timing_std = np.std(all_timing_flat)
                        all_timing_sem = stats.sem(all_timing_flat)

                        # plot with errorbars
                        plt.figure(plots[freq_key][syn_dist_key].number)
                        plt.plot(float(syn_num_key), soma_timing_mean, color=color, marker=marker_soma, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), dend_timing_mean, color=color, marker=marker_dend, markersize=size, alpha=opacity)
                        plt.plot(float(syn_num_key), all_timing_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                        plt.errorbar(float(syn_num_key), soma_timing_mean, yerr=soma_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), dend_timing_mean, yerr=dend_timing_sem, color=color, alpha=opacity)
                        plt.errorbar(float(syn_num_key), all_timing_mean, yerr=all_timing_sem, color=color, alpha=opacity)
                        plt.xlabel('number of active synapses')
                        plt.ylabel('spike timing after epsp onset (ms)')
                        plt.title(freq_key + ' Hz, ' + syn_dist_key + ' um from soma')

                # save and close figure
                plt.figure(plots[freq_key][syn_dist_key].number)
                plot_file_name = 'syn_num_x_spikes_timing_'+freq_key+'Hz_' + syn_dist_key +'_um'
                plots[freq_key][syn_dist_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[freq_key][syn_dist_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plots[freq_key][syn_dist_key])


        # create least squares fit for syn_num vs spike_num curves

    def exp_2c(self, **kwargs):
        """ activate synapse in each segment until a spike either occurs in the soma or dendrite.  Plot the number of synapses(weight) required at each segment as a function of distance from soma
        """
        # number of stimulation polarities tested
        n_pol =3
        # spike threshold
        threshold = -30

        # instantiate default parameter class
        p_class = param.Default()
        p_temp = p_class.p
        # load cell and store in parameter dictionary
        cell1 = cell.CellMigliore2005(p_temp)
        cell1.geometry(p_temp)
        # insert mechanisms
        cell1.mechanisms(p_temp)
        # measure distance of each segment from the soma and store in parameter dictionary
        p_class.seg_distance(cell1)

        p_temp['morpho'] = p_class.create_morpho(cell1.geo)


        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        # all files in directory
        files = os.listdir(data_folder)
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        id_list_string_spike_times = 'id_list_spike_times.pkl'
        # if list of processed files in folder, load list
        if id_list_string_spike_times in files:
            print 'id_list found'
            
            with open(data_folder+id_list_string_spike_times, 'rb') as pkl_file:
                    id_list_spike_times = pickle.load(pkl_file)
        
        # otherwise create empty list
        else:
            id_list_spike_times = []

        # string to save group data
        save_string_group_data = 'spikes_grouped'+'.pkl'
        # if data file already exists
        if save_string_group_data in files:
            # load data
            with open(data_folder+save_string_group_data, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
        # otherwise create data structure
        else:
            # data organized as {location}{data_type}[trials][polarity]
            spike_data= {}

        variables = ['spikes', 'peak', 'distance', 'weight', 'seg_idx']
        # load spike analysis functions
        spike_analysis = Spikes()

        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:
                # print 'data_file:', data_file_i, 'out of', len(data_files)

                # check if data has been processed already
                if data_file not in id_list_spike_times:

                    # open unprocessed data
                    with open(data_folder+data_file, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)

                    print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'

                    # parameter dictionary
                    p = data['p']

                    if 'morpho' not in p.keys():
                        p['morpho'] = p_temp['morpho']
                        with open(data_folder+data_file, 'wb')  as output:
                            pickle.dump(data, output,protocol=pickle.HIGHEST_PROTOCOL)

                    if len(spike_data)==0:
                        for tree_key, tree in p['seg_dist'].iteritems():
                            spike_data[tree_key]=[]
                            for sec_i, sec in enumerate(tree):
                                spike_data[tree_key].append([])
                                for seg_i, seg in enumerate(sec):
                                    spike_data[tree_key][sec_i].append([])
                                    for pol_i, pol in enumerate(p['field']):
                                        spike_data[tree_key][sec_i][seg_i].append({'soma':{}, 'dend':{}})
                                        for loc_i, loc in spike_data[tree_key][sec_i][seg_i][pol_i].iteritems():
                                            for variable_i, variable in enumerate(variables):
                                                loc[variable]=[]

                    
                    # add to processed list
                    id_list_spike_times.append(data_file)

                    # get tree, section, segment info (only one segment is active)
                    tree = p['trees'][0]
                    sec = p['sec_idx'][tree][0]
                    sec_i=0
                    seg = p['seg_idx'][tree][0][0]
                    seg_i=0

                    # iterate over field polarities
                    for f_i, f in enumerate(p['field']):

                        # detect spikes in dendrite
                        spike_data[tree][sec][seg][f_i]['dend']['spikes'].append(spike_analysis.detect_spikes(data[str(f)][tree+'_v'][sec_i][seg_i], threshold=threshold)['times'][0])

                        # detect spikes in soma
                        spike_data[tree][sec][seg][f_i]['soma']['spikes'].append(spike_analysis.detect_spikes(data[str(f)]['soma_v'][0][0], threshold=threshold)['times'][0])

                        # peak dendrite voltage
                        peak_dend = np.amax(data[str(f)][tree+'_v'][sec_i][seg_i][int(p['warmup']/p['dt']):])
                        # peak soma voltage
                        peak_soma = np.amax(data[str(f)]['soma_v'][0][0][int(p['warmup']/p['dt']):])

                        # store peak voltages
                        spike_data[tree][sec][seg][f_i]['dend']['peak'].append(peak_dend)
                        spike_data[tree][sec][seg][f_i]['soma']['peak'].append(peak_soma)

                        # store distance from soma for each spike
                        spike_data[tree][sec][seg][f_i]['dend']['distance'].append(p['seg_dist'][tree][sec][seg])

                        # store synaptic weight
                        spike_data[tree][sec][seg][f_i]['dend']['weight'].append(p['w_mean'])

                        # store segment info
                        spike_data[tree][sec][seg][f_i]['dend']['seg_idx'].append(p['seg_idx'])

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        
        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)

        # plot number of synapses required for spike as a function of distance from the soma, with separate markers when spike initiated in soma or dendrite
        plot_file_name = 'distance_x_weight_spike_init'
        if plot_file_name + '.png' not in files: 
            distance_plot = plt.figure()
            plot_handles=[]

            for tree_key, tree in spike_data.iteritems():
                for sec_i, sec in enumerate(tree):
                    for seg_i, seg in enumerate(sec):
                        for f_i, f in enumerate(seg):

                            if len(f['dend']['weight'])>0 and tree_key is not 'soma' :

                                soma = f['soma']['spikes']
                                
                                # distance from soma
                                dist = [dist
                                for dist_i, dist 
                                in enumerate(f['dend']['distance']) 
                                if len(f['dend']['spikes'][dist_i]) > 0 
                                or len(soma[dist_i]) > 0]

                                if len(dist)>0:
                                    dist=dist[0]
                                
                                # weight
                                weight = [weight
                                for weight_i, weight 
                                in enumerate(f['dend']['weight']) 
                                if len(f['dend']['spikes'][weight_i]) > 0 
                                or len(soma[weight_i]) > 0]

                                if len(weight)>0:
                                    weight=weight[0]
                                
                                # plot color
                                color = p_temp['field_color'][f_i]
                                
                                # list of dendritic spike time (if no spike it is empty list)
                                dend = [spike[0] 
                                for spike_i, spike 
                                in enumerate(f['dend']['spikes'])
                                if len(spike) > 0]

                                soma = [spike[0] 
                                for spike_i, spike 
                                in enumerate(soma) 
                                if len(spike) > 0]

                                    # for storing whether dendritic or soamtic spike occured first

                                if len(dend)>0:
                                    dend = [dend[0]]

                                if len(soma)>0:
                                    soma=soma[0]
                                

                                dend_first=False
                                soma_first=False
                                # if there are both dendritic and somatic spikes
                                if dend and soma:
                                    # if dendrite first
                                    if (dend < soma):
                                        dend_first=True
                                        marker = 'o'
                                        label='dendrite first'
                                    # if soma first, if tie soma wins
                                    else:
                                        soma_first=True
                                        marker = 'x'
                                        label = 'soma_first'
                                # if only dendritic spike
                                elif dend and not soma:
                                    dend_first=True
                                    marker = 'o'
                                    label='dendrite first'

                                # if only somatic spike
                                elif soma and not dend:
                                    soma_first=True
                                    marker = 'x'
                                    label = 'soma_first'

                                if dend_first:
                                    f['dend']['first_spike'] = 'dend'
                                elif soma_first:
                                    f['dend']['first_spike'] = 'soma'
                                elif dend_first and soma_first:
                                    f['dend']['first_spike'] = 'none'
                                else:
                                    f['dend']['first_spike'] = 'none'


                                if dend_first or soma_first:

                                    f['dend']['first_spike_weight'] = weight

                                    # plot trial and add to list for legend
                                    plot_handles.append(plt.plot(dist, weight, color+marker, label=label))
                                    # labels
                                    plt.xlabel('distance from soma (um)')
                                    plt.ylabel('number of synapses/weight (uS)')

                                else:
                                    f['dend']['first_spike_weight'] = f['dend']['weight'][-1]
                        # legend
                        # plt.legend(handles=plot_handles)

            # save and close figure
            distance_plot.savefig(data_folder+plot_file_name+'.png', dpi=250)
            plt.close(distance_plot)

        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)

        # plot peak soma vs peak dendrite voltage for each segment
        distances = [[0,100],[100,200],[200,300],[300,600]]
        reset = -65
        peak_plot = []
        plot_handles=[]
        for distance_i, distance in enumerate(distances):
            plot_file_name = 'peak_voltage_distance_'+str(distance[0])+'_'+str(distance[1])
            if plot_file_name+'.png' not in files:

                peak_plot.append(plt.figure())
                plot_handles.append([])
                for tree_key, tree in spike_data.iteritems():
                    for sec_i, sec in enumerate(tree):
                        for seg_i, seg in enumerate(sec):
                            for f_i, f in enumerate(seg):
                                if len(f['dend']['weight'])>0 and tree_key is not 'soma' :

                                    # plot color
                                    color = p_temp['field_color'][f_i]
                                    # peak dendrite voltage
                                    dend_peak = f['dend']['peak']
                                    # peak soma voltage
                                    soma_peak = f['soma']['peak']

                                    weight = f['dend']['weight']

                                    dist = f['dend']['distance']

                                    soma = f['soma']['spikes']

                                    # list of dendritic spike time (if no spike it is empty list)
                                    dend = [spike[0] 
                                    for spike_i, spike 
                                    in enumerate(f['dend']['spikes'])
                                    if len(spike) > 0]

                                    soma = [spike[0] 
                                    for spike_i, spike 
                                    in enumerate(soma) 
                                    if len(spike) > 0]

                                        # for storing whether dendritic or soamtic spike occured first

                                    if len(dend)>0:
                                        dend = dend[0]

                                    if len(soma)>0:
                                        soma=soma[0]
                                    

                                    dend_first=False
                                    soma_first=False
                                    # if there are both dendritic and somatic spikes
                                    if dend and soma:
                                        # if dendrite first
                                        if (dend < soma):
                                            dend_first=True
                                            marker = 'o'
                                            label='dendrite first'
                                        # if soma first, if tie soma wins
                                        else:
                                            soma_first=True
                                            marker = 'x'
                                            label = 'soma_first'
                                    # if only dendritic spike
                                    elif dend and not soma:
                                        dend_first=True
                                        marker = 'o'
                                        label='dendrite first'

                                    # if only somatic spike
                                    elif soma and not dend:
                                        soma_first=True
                                        marker = 'x'
                                        label = 'soma_first'

                                    for trial_i, trial in enumerate(dend_peak):
                                        if soma_peak[trial_i]>threshold:
                                            soma_peak[trial_i]=reset
                                        if  dend_peak[trial_i] >threshold:
                                            dend_peak[trial_i]=reset

                                        if dist[trial_i]>=distance[0] and dist[trial_i] < distance[1]: 
                                            # plot trial and add to list for legend
                                            plot_handles[distance_i].append(plt.plot(dend_peak[trial_i], soma_peak[trial_i], color+marker, label=label))
                                            # labels
                                            plt.xlabel('peak dendrite voltage (mV)')
                                            plt.ylabel('peak soma voltage (mV)')
                                            plt.title('distnace from soma:' + str(distance[0]) + ' to ' + str(distance[1]) + ' um')
                            # legend
                            # plt.legend(handles=plot_handles[distance_i])

                    # save and close figure
                peak_plot[distance_i].savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(peak_plot[distance_i])

        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)


        # create shapeplot of spike threshold for each polarity
        # iterate over field polarities
        # create data structure to pass to morph plot for each field
        # iterate through spike data structure, just storing spike threshold weight for each segment, if soma first weights are negative, if dendrite first weights are positive
        morphplot_data = []
        for f_i, f in enumerate(p_temp['field']):
            morphplot_data.append({})
            for tree_key, tree in spike_data.iteritems():
                morphplot_data[f_i][tree_key]=[]
                for sec_i, sec in enumerate(tree):
                    morphplot_data[f_i][tree_key].append([])
                    for seg_i, seg in enumerate(sec):
                        if 'first_spike' not in seg[f_i]['dend'].keys():
                            # print 'first spike not found'
                            seg[f_i]['dend']['first_spike'] = 'none'

                        first = seg[f_i]['dend']['first_spike']
                        # print 'first spike:', first
                        if first=='soma':
                            weight = -1*seg[f_i]['dend']['first_spike_weight']
                        elif first=='dend':
                            weight = seg[f_i]['dend']['first_spike_weight']
                        elif first=='none':
                            weight= 0.

                        # print 'weight threshold:', weight
                        morphplot_data[f_i][tree_key][sec_i].append(weight)

        fig, ax = plt.subplots(nrows=1, ncols=len(p_temp['field']))
        plot_file_name = 'spike_threshold_shapeplots' 
        file_name = data_folder + plot_file_name + '.png'
        for f_i, f in enumerate(p_temp['field']):
            ShapePlot().basic(morpho=p_temp['morpho'], data=morphplot_data[f_i], axes=ax[f_i])

        fig.savefig(file_name, dpi=250)

    def exp_2d(self, **kwargs):
        """ Apply a single theta burst for each segment, one segment at a time.  Step through increasing number of synapses (weight) to detect thresholds

        plot number of spikes
        """
        # number of stimulation polarities tested
        n_pol =3
        # spike threshold
        threshold = -30

        # instantiate default parameter class
        p_class = param.Default()
        p_temp = p_class.p
        # load cell and store in parameter dictionary
        cell1 = cell.CellMigliore2005(p_temp)
        cell1.geometry(p_temp)
        # insert mechanisms
        cell1.mechanisms(p_temp)
        # measure distance of each segment from the soma and store in parameter dictionary
        p_class.seg_distance(cell1)

        p_temp['morpho'] = p_class.create_morpho(cell1.geo)

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        # all files in directory
        files = os.listdir(data_folder)
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        id_list_string_spike_times = 'id_list_spike_times.pkl'
        # if list of processed files in folder, load list
        if id_list_string_spike_times in files:
            print 'id_list found'
            
            with open(data_folder+id_list_string_spike_times, 'rb') as pkl_file:
                    id_list_spike_times = pickle.load(pkl_file)
        
        # otherwise create empty list
        else:
            id_list_spike_times = []

        # string to save group data
        save_string_group_data = 'spikes_grouped'+'.pkl'
        # if data file already exists
        if save_string_group_data in files:
            # load data
            with open(data_folder+save_string_group_data, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
        # otherwise create data structure
        else:
            # data organized as {location}{data_type}[trials][polarity]
            spike_data= {}

        variables = ['spikes', 'peak', 'distance', 'weight', 'seg_idx']
        # load spike analysis functions
        spike_analysis = Spikes()

        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:

                # check if data has been processed already
                if data_file not in id_list_spike_times:

                    # open unprocessed data
                    with open(data_folder+data_file, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)

                    print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'

                    # parameter dictionary
                    p = data['p']

                    if 'morpho' not in p.keys():
                        p['morpho'] = p_temp['morpho']
                        with open(data_folder+data_file, 'wb')  as output:
                            pickle.dump(data, output,protocol=pickle.HIGHEST_PROTOCOL)

                    if len(spike_data)==0:
                        for tree_key, tree in p['seg_dist'].iteritems():
                            spike_data[tree_key]=[]
                            for sec_i, sec in enumerate(tree):
                                spike_data[tree_key].append([])
                                for seg_i, seg in enumerate(sec):
                                    spike_data[tree_key][sec_i].append([])
                                    for pol_i, pol in enumerate(p['field']):
                                        spike_data[tree_key][sec_i][seg_i].append({'soma':{}, 'dend':{}})
                                        for loc_i, loc in spike_data[tree_key][sec_i][seg_i][pol_i].iteritems():
                                            for variable_i, variable in enumerate(variables):
                                                loc[variable]=[]

                    
                    # add to processed list
                    id_list_spike_times.append(data_file)

                    # get tree, section, segment info (only one segment is active)
                    tree = p['trees'][0]
                    sec = p['sec_idx'][tree][0]
                    sec_i=0
                    seg = p['seg_idx'][tree][0][0]
                    seg_i=0

                    # iterate over field polarities
                    for f_i, f in enumerate(p['field']):

                        # detect spikes in dendrite
                        spike_data[tree][sec][seg][f_i]['dend']['spikes'].append(spike_analysis.detect_spikes(data[str(f)][tree+'_v'][sec_i][seg_i], threshold=threshold)['times'][0])

                        # detect spikes in soma
                        spike_data[tree][sec][seg][f_i]['soma']['spikes'].append(spike_analysis.detect_spikes(data[str(f)]['soma_v'][0][0], threshold=threshold)['times'][0])

                        # peak dendrite voltage
                        peak_dend = np.amax(data[str(f)][tree+'_v'][sec_i][seg_i][int(p['warmup']/p['dt']):])
                        # peak soma voltage
                        peak_soma = np.amax(data[str(f)]['soma_v'][0][0][int(p['warmup']/p['dt']):])

                        # store peak voltages
                        spike_data[tree][sec][seg][f_i]['dend']['peak'].append(peak_dend)
                        spike_data[tree][sec][seg][f_i]['soma']['peak'].append(peak_soma)

                        # store distance from soma for each spike
                        spike_data[tree][sec][seg][f_i]['dend']['distance'].append(p['seg_dist'][tree][sec][seg])

                        # store synaptic weight
                        spike_data[tree][sec][seg][f_i]['dend']['weight'].append(p['w_mean'])

                        # store segment info
                        spike_data[tree][sec][seg][f_i]['dend']['seg_idx'].append(p['seg_idx'])

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        
        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)

        # plot number of synapses required for spike as a function of distance from the soma, with separate markers when spike initiated in soma or dendrite
        plot_file_name = 'distance_x_weight_spike_init'
        if plot_file_name + '.png' not in files: 
            distance_plot = plt.figure()
            plot_handles=[]

            for tree_key, tree in spike_data.iteritems():
                for sec_i, sec in enumerate(tree):
                    for seg_i, seg in enumerate(sec):
                        for f_i, f in enumerate(seg):

                            if len(f['dend']['weight'])>0 and tree_key is not 'soma' :

                                soma = f['soma']['spikes']
                                
                                # distance from soma
                                dist = [dist
                                for dist_i, dist 
                                in enumerate(f['dend']['distance']) 
                                if len(f['dend']['spikes'][dist_i]) > 0 
                                or len(soma[dist_i]) > 0]

                                if len(dist)>0:
                                    dist=dist[0]
                                
                                # weight
                                weight = [weight
                                for weight_i, weight 
                                in enumerate(f['dend']['weight']) 
                                if len(f['dend']['spikes'][weight_i]) > 0 
                                or len(soma[weight_i]) > 0]

                                if len(weight)>0:
                                    weight=weight[0]
                                
                                # plot color
                                color = p_temp['field_color'][f_i]
                                
                                # list of dendritic spike time (if no spike it is empty list)
                                dend = [spike[0] 
                                for spike_i, spike 
                                in enumerate(f['dend']['spikes'])
                                if len(spike) > 0]

                                soma = [spike[0] 
                                for spike_i, spike 
                                in enumerate(soma) 
                                if len(spike) > 0]

                                    # for storing whether dendritic or soamtic spike occured first

                                if len(dend)>0:
                                    dend = [dend[0]]

                                if len(soma)>0:
                                    soma=soma[0]
                                

                                dend_first=False
                                soma_first=False
                                # if there are both dendritic and somatic spikes
                                if dend and soma:
                                    # if dendrite first
                                    if (dend < soma):
                                        dend_first=True
                                        marker = 'o'
                                        label='dendrite first'
                                    # if soma first, if tie soma wins
                                    else:
                                        soma_first=True
                                        marker = 'x'
                                        label = 'soma_first'
                                # if only dendritic spike
                                elif dend and not soma:
                                    dend_first=True
                                    marker = 'o'
                                    label='dendrite first'

                                # if only somatic spike
                                elif soma and not dend:
                                    soma_first=True
                                    marker = 'x'
                                    label = 'soma_first'

                                if dend_first:
                                    f['dend']['first_spike'] = 'dend'
                                elif soma_first:
                                    f['dend']['first_spike'] = 'soma'
                                elif dend_first and soma_first:
                                    f['dend']['first_spike'] = 'none'
                                else:
                                    f['dend']['first_spike'] = 'none'


                                if dend_first or soma_first:

                                    f['dend']['first_spike_weight'] = weight

                                    # plot trial and add to list for legend
                                    plot_handles.append(plt.plot(dist, weight, color+marker, label=label))
                                    # labels
                                    plt.xlabel('distance from soma (um)')
                                    plt.ylabel('number of synapses/weight (uS)')

                                else:
                                    f['dend']['first_spike_weight'] = f['dend']['weight'][-1]
                        # legend
                        # plt.legend(handles=plot_handles)

            # save and close figure
            distance_plot.savefig(data_folder+plot_file_name+'.png', dpi=250)
            plt.close(distance_plot)

        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)

        # plot peak soma vs peak dendrite voltage for each segment
        distances = [[0,100],[100,200],[200,300],[300,600]]
        reset = -65
        peak_plot = []
        plot_handles=[]
        for distance_i, distance in enumerate(distances):
            plot_file_name = 'peak_voltage_distance_'+str(distance[0])+'_'+str(distance[1])
            if plot_file_name+'.png' not in files:

                peak_plot.append(plt.figure())
                plot_handles.append([])
                for tree_key, tree in spike_data.iteritems():
                    for sec_i, sec in enumerate(tree):
                        for seg_i, seg in enumerate(sec):
                            for f_i, f in enumerate(seg):
                                if len(f['dend']['weight'])>0 and tree_key is not 'soma' :

                                    # plot color
                                    color = p_temp['field_color'][f_i]
                                    # peak dendrite voltage
                                    dend_peak = f['dend']['peak']
                                    # peak soma voltage
                                    soma_peak = f['soma']['peak']

                                    weight = f['dend']['weight']

                                    dist = f['dend']['distance']

                                    soma = f['soma']['spikes']

                                    # list of dendritic spike time (if no spike it is empty list)
                                    dend = [spike[0] 
                                    for spike_i, spike 
                                    in enumerate(f['dend']['spikes'])
                                    if len(spike) > 0]

                                    soma = [spike[0] 
                                    for spike_i, spike 
                                    in enumerate(soma) 
                                    if len(spike) > 0]

                                        # for storing whether dendritic or soamtic spike occured first

                                    if len(dend)>0:
                                        dend = dend[0]

                                    if len(soma)>0:
                                        soma=soma[0]
                                    

                                    dend_first=False
                                    soma_first=False
                                    # if there are both dendritic and somatic spikes
                                    if dend and soma:
                                        # if dendrite first
                                        if (dend < soma):
                                            dend_first=True
                                            marker = 'o'
                                            label='dendrite first'
                                        # if soma first, if tie soma wins
                                        else:
                                            soma_first=True
                                            marker = 'x'
                                            label = 'soma_first'
                                    # if only dendritic spike
                                    elif dend and not soma:
                                        dend_first=True
                                        marker = 'o'
                                        label='dendrite first'

                                    # if only somatic spike
                                    elif soma and not dend:
                                        soma_first=True
                                        marker = 'x'
                                        label = 'soma_first'

                                    for trial_i, trial in enumerate(dend_peak):
                                        if soma_peak[trial_i]>threshold:
                                            soma_peak[trial_i]=reset
                                        if  dend_peak[trial_i] >threshold:
                                            dend_peak[trial_i]=reset

                                        if dist[trial_i]>=distance[0] and dist[trial_i] < distance[1]: 
                                            # plot trial and add to list for legend
                                            plot_handles[distance_i].append(plt.plot(dend_peak[trial_i], soma_peak[trial_i], color+marker, label=label))
                                            # labels
                                            plt.xlabel('peak dendrite voltage (mV)')
                                            plt.ylabel('peak soma voltage (mV)')
                                            plt.title('distnace from soma:' + str(distance[0]) + ' to ' + str(distance[1]) + ' um')
                            # legend
                            # plt.legend(handles=plot_handles[distance_i])

                    # save and close figure
                peak_plot[distance_i].savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(peak_plot[distance_i])

        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)


        # create shapeplot of spike threshold for each polarity
        # iterate over field polarities
        # create data structure to pass to morph plot for each field
        # iterate through spike data structure, just storing spike threshold weight for each segment, if soma first weights are negative, if dendrite first weights are positive
        morphplot_data = []
        soma_count=[]
        dend_count=[]
        min_val=0.
        max_val=0.
        for f_i, f in enumerate(p_temp['field']):
            soma_count.append([])
            dend_count.append([])
            morphplot_data.append({})
            for tree_key, tree in spike_data.iteritems():
                morphplot_data[f_i][tree_key]=[]
                for sec_i, sec in enumerate(tree):
                    morphplot_data[f_i][tree_key].append([])
                    for seg_i, seg in enumerate(sec):
                        if 'first_spike' not in seg[f_i]['dend'].keys():
                            # print 'first spike not found'
                            seg[f_i]['dend']['first_spike'] = 'none'

                        first = seg[f_i]['dend']['first_spike']
                        # print 'first spike:', first
                        if first=='soma':
                            weight = -1*seg[f_i]['dend']['first_spike_weight']
                            soma_count[f_i].append(weight)
                        elif first=='dend':
                            weight = seg[f_i]['dend']['first_spike_weight']
                            dend_count[f_i].append(weight)
                        elif first=='none':
                            weight= 0.02
                            dend_count[f_i].append(weight)


                        # print 'weight threshold:', weight
                        morphplot_data[f_i][tree_key][sec_i].append(weight)
                        if weight< min_val:
                            min_val=weight
                        elif weight>max_val:
                            max_val=weight

        # soma
        fig = plt.figure()
        # plot_locs = [[0,1], [3,4], [6,7]]
        plot_locs = [[0], [1], [2]]
        plot_colors = ['blue','black','red']
        for f_i, f in enumerate(soma_count):
            count = float(len(soma_count[f_i]))/float(len(dend_count[f_i])+len(soma_count[f_i]))
            plt.bar(plot_locs[f_i][0], count, facecolor=plot_colors[f_i], edgecolor='pink')
            # plt.bar(plot_locs[f_i][1], len(dend_count[f_i]), facecolor=plot_colors[f_i], edgecolor='pink')
        plt.title('Soma-controlled synapses',fontsize=25, weight='bold')
        plt.ylabel('Fraction of total synapses',fontsize=20, weight='bold')
        plt.xticks([0,1,2], ['Cathodal','Control','Anodal'],fontsize=20, weight='bold')
        plt.yticks(fontsize=8, weight='bold')
        plt.ylim([0.06, 0.14])
        plot_file_name = 'soma fraction' 
        file_name = data_folder + plot_file_name + '.png'
        fig.savefig(file_name, dpi=250)


        # soma average threshold
        fig = plt.figure()
        # plot_locs = [[0,1], [3,4], [6,7]]
        plot_locs = [[0], [1], [2]]
        plot_colors = ['blue','black','red']
        for f_i, f in enumerate(soma_count):
            count = float(len(soma_count[f_i]))/float(len(dend_count[f_i])+len(soma_count[f_i]))
            mean = -1*np.mean(soma_count[f_i])
            sem = stats.sem(soma_count[f_i])
            plt.bar(plot_locs[f_i][0], mean, facecolor=plot_colors[f_i], edgecolor='pink')
            plt.errorbar(plot_locs[f_i][0], mean, sem, color=plot_colors[f_i])
            # plt.bar(plot_locs[f_i][1], len(dend_count[f_i]), facecolor=plot_colors[f_i], edgecolor='pink')
        plt.title('Somatic Threshold',fontsize=25, weight='bold')
        plt.ylabel('Threshold synaptic weight',fontsize=20, weight='bold')
        plt.xticks([0,1,2], ['Cathodal','Control','Anodal'],fontsize=20, weight='bold')
        plt.yticks(fontsize=8, weight='bold')
        plt.ylim([0.006, 0.016])
        plot_file_name = 'soma threshold change' 
        file_name = data_folder + plot_file_name + '.png'
        fig.savefig(file_name, dpi=250)

        # dednritic average threshold
        fig = plt.figure()
        # plot_locs = [[0,1], [3,4], [6,7]]
        plot_locs = [[0], [1], [2]]
        plot_colors = ['blue','black','red']
        for f_i, f in enumerate(soma_count):
            count = float(len(soma_count[f_i]))/float(len(dend_count[f_i])+len(soma_count[f_i]))
            mean = np.mean(dend_count[f_i])
            sem = stats.sem(dend_count[f_i])
            plt.bar(plot_locs[f_i][0], mean, facecolor=plot_colors[f_i], edgecolor='pink')
            plt.errorbar(plot_locs[f_i][0], mean, sem, color=plot_colors[f_i])
            # plt.bar(plot_locs[f_i][1], len(dend_count[f_i]), facecolor=plot_colors[f_i], edgecolor='pink')
        plt.title('Dendritic Threshold', fontsize=25, weight='bold')
        plt.ylabel('Threshold synaptic weight',fontsize=20, weight='bold')
        plt.xticks([0,1,2], ['Cathodal','Control','Anodal'], fontsize=20, weight='bold')
        plt.yticks(fontsize=8, weight='bold')
        plt.ylim([0.006, 0.016])
        plot_file_name = 'dendrite threshold change' 
        file_name = data_folder + plot_file_name + '.png'
        fig.savefig(file_name, dpi=250)





        # morphplot_data_diff=[]
        # for f_i, f in enumerate(morphplot_data):
        #     morphplot_data_diff.append({})
        #     for tree_key, tree in spike_data.iteritems():
        #         morphplot_data_diff[f_i][tree_key]=[]
        #         for sec_i, sec in enumerate(tree):
        #             morphplot_data_diff[f_i][tree_key].append([])
        #             for seg_i, seg in enumerate(sec):
        #                 weight_stim = morphplot_data[f_i][tree_key][sec_i][seg_i]
        #                 weight_control = morphplot_data[1][tree_key][sec_i][seg_i]
        #                 weight_diff = weight_stim-weight_control
        #                 diff = morphplot_data[f_i][tree_key][sec_i][seg_i] - morphplot_data[1][tree_key][sec_i][seg_i]
        #                 morphplot_data_diff[f_i][tree_key][sec_i].append(diff)


        print min_val, max_val

        fig, ax = plt.subplots(nrows=1, ncols=len(p_temp['field']))
        patch_coll=[]
        plot_file_name = 'spike_threshold_shapeplots' 
        file_name = data_folder + plot_file_name + '.png'
        plot_labels = ['Cathodal', 'Control', 'Anodal']
        plot_label_colors = ['blue','black','red']
        for f_i, f in enumerate(p_temp['field']):
            patch_coll.append(ShapePlot().basic(morpho=p_temp['morpho'], data=morphplot_data[f_i], axes=ax[f_i], colormap=colormap.PiYG, width_scale=4))
            patch_coll[f_i].set_clim([min_val,max_val])
            ax[f_i].add_collection(patch_coll[f_i])
            ax[f_i].autoscale()
            ax[f_i].set_title(plot_labels[f_i], color=plot_label_colors[f_i], fontsize=20)
            ax[f_i].set_yticks([])
            ax[f_i].set_xticks([])
        # ax[0].set_ylabel('Distance from soma (um)', fontsize=20)
        # ax[1].set_xlabel('Distance from soma (um)', fontsize=20)
        cbar = plt.colorbar(patch_coll[f_i], ax=ax[f_i], ticks=None, orientation='vertical')
        cbar.set_label('Threshold synaptic input', fontsize=20)
        cbar.set_ticks([min_val,0, max_val])
        cbar.set_ticklabels(['Max','0','Max'])
        # cbar.set_ticks([])
        # cbar.set_ticklabels([])
        red_patch =patches.Patch(color='green', label='Dendrite first')
        blue_patch =patches.Patch(color='pink', label='Soma first')
        plt.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(0.,0), loc="upper left", frameon=False)


        fig.savefig(file_name, dpi=250)

    def exp_4a(self, **kwargs):
        """ Associative plasticity experiment

        Activate a strong (100 Hz burst) and weak (single pulse in the middle of burst) pathway (separately or together).
        Vary which pathways are active (strong only, weak only, strong+weak)
        Vary the distance from soma of each pathway (0-200,200-400,400-600 um).
        Vary the number of synapses in each pathway (10, 20, 40).
        Synapse locations are generated randomly with param.choose_seg_rand(). Multiple synapses can be placed on the same segment.  The weight is then multiplied by the number of "synapses"

        Plots: 
        number of active synapse x associativity factor (ratio of normalized number of spikes (number of spikes/number of active synapses) when pathways are paired vs unpaired)

        """
        # spike threshold 
        threshold =-30

        # identify data folder
        data_folder = 'Data/'+kwargs['experiment']+'/'
        
        # all files in directory
        files = os.listdir(data_folder)
        
        # data files
        data_files = [file for file in files if 'data' in file]
        plot_files = [file for file in files if 'trace' in file]
        
        # unique identifiers for each file type
        data_files_id= [file[-36:-1] for file in files if 'data' in file]
        plot_files_id = [file[-36:-1] for file in files if 'trace' in file]

        save_string = 'group_data'+'.pkl'
        self.group_spike_data = self._load_group_data(experiment=kwargs['experiment'], save_string=save_string)
        id_list_key = 'id_list'

        if save_string in data_files:
            data_files.remove(save_string)
        # data organized as {pathway combo}{syn distance}{number of synapses}{polarity}{pathway}{data type}[trial]{tree}[section][segment][spikes]


        # iterate over data files
        for data_file_i, data_file in enumerate(data_files):

            if data_file_i >=0:# and data_file_i <=1000:
                # if no id list, create one
                if id_list_key not in self.group_spike_data:
                    self.group_spike_data[id_list_key]=[]
                        
                # check if data has been processed already
                if data_file not in self.group_spike_data[id_list_key]:

                    print data_file
                    # open unprocessed data

                    try:
                        with open(data_folder+data_file, 'rb') as pkl_file:
                            data = pickle.load(pkl_file)

                        print 'data_file: ', data_file_i, ' out of ', len(data_files), ' opened'
                    # if data file is corrupted, skip it
                    except EOFError:
                        'EOF error, file not opened'
                        continue

                    # spike_dictionary{'polarity'}{'pathway'}{'data type'}
                    spike_data = self._get_spikes(data=data, threshold=threshold, weakpath_bins=True)

                    p = spike_data['p']
                    p_path = p['p_path']

                    # retrieve experimental conditions
                    if 'path_combo' not in p:
                        path_combo_key=''
                        for key in p_path:
                            if len(path_combo_key)==0:
                                path_combo_key+=key
                            else:
                                path_combo_key+='_'+key
                    else:
                        path_combo_key = str(p['path_combo'])
                    syn_dist_key = str(p['syn_dist'])

                    # FIXME
                    if isinstance(p['syn_num'][0], list):
                        for polarity_key, pathway in spike_data.iteritems():
                            if polarity_key != 'p':
                                for pathway_key, data_dic in pathway.iteritems():
                                    syn_num_path = data_dic['p_path']['syn_num']
                                    for syn_num_range in p['syn_num']:
                                        if syn_num_path in syn_num_range:
                                            syn_num_key=str(syn_num_path)
                    else:
                        syn_num_key=str(p['syn_num'][0])

                    # update data structure dimensions
                    if path_combo_key not in self.group_spike_data:
                        self.group_spike_data[path_combo_key]={}

                    if syn_dist_key not in self.group_spike_data[path_combo_key]:
                        self.group_spike_data[path_combo_key][syn_dist_key]={}

                    if syn_num_key not in self.group_spike_data[path_combo_key][syn_dist_key]:
                        self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key]={}

                    for polarity_key, pathway in spike_data.iteritems():
                        if polarity_key != 'p':
                            if polarity_key not in self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key]:
                                self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key]={}
                            for pathway_key, data_dic in pathway.iteritems():
                                if pathway_key not in self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key]:
                                    self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]={}

                                for dtype_key, dtype in data_dic.iteritems():
                                    if dtype_key not in self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]:
                                        self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key][dtype_key]=[]
                                    
                                    self.group_spike_data[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key][dtype_key].append(dtype)

                    self.group_spike_data[id_list_key].append(data_file)

        # save structure of all raw spike data
        save_group_data = self.group_spike_data
        with open(data_folder+save_string, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'
        
        
        # plot number of synapses vs. total number of spikes normalized to total number of synapses in weak pathway under paired and unpaired condition
        spikes_temp ={}
        for path_combo_key, path_combo in self.group_spike_data.iteritems():
            if path_combo_key != id_list_key:
                spikes_temp[path_combo_key]={}
                for syn_dist_key, syn_dist in path_combo.iteritems():
                    spikes_temp[path_combo_key][syn_dist_key]={}
                    sig_data=[]
                    for syn_num_key, syn_num in syn_dist.iteritems():
                        spikes_temp[path_combo_key][syn_dist_key][syn_num_key]={}
                        for polarity_key, polarity in syn_num.iteritems():
                            spikes_temp[path_combo_key][syn_dist_key][syn_num_key][polarity_key]={}
                            for pathway_key, pathway in polarity.iteritems():
                                spikes_temp[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]={}
                                dtemp = spikes_temp[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]

                                # get polarity index
                                polarity_i = [f_i for f_i, f in enumerate(pathway['p'][0]['field']) if str(f)==polarity_key][0]

                                # plot color and marker
                                color = pathway['p'][0]['field_color'][polarity_i]
                                # size = 20.*float(syn_dist_key)/600.
                                size = 10.
                                opacity = 0.7
                                marker_soma = '.'
                                marker_dend= 'x'
                                marker_all= '^'


                                # lsit of soma/dend spike fraction for each trial in current set of conditions

                                # iterate through trials
                                soma_spikes_total=[]
                                dend_spikes_total=[]
                                all_spikes_total=[]
                                dw_list=[]
                                
                                # spikes_dend_diff_bin = [trial][time bin][dendritic spike time], where soma first is negative, dendrite first is positive
                                for trial_i, trial in enumerate(pathway['spikes_dend_diff_bin']):


                                    # count somatic/dendritic spikes for each trial
                                    soma_count =[]
                                    dend_count=[]
                                    for time_bin_i, time_bin in enumerate(trial):
                                        # list of spike indeces with soma/dend spikes
                                        soma_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff<0.]
                                        dend_first = [spike_i for spike_i, spike_diff in enumerate(time_bin) if spike_diff>=0.]
                                        # count total spieks per time bin
                                        soma_count.append(float(len(soma_first)))
                                        dend_count.append(float(len(dend_first)))
                                    
                                    syn_num_unique = float(len([seg_i for tree_key, tree in pathway['p_path'][trial_i]['seg_idx'].iteritems() for sec_i, sec in enumerate(tree) for seg_i, seg in enumerate(sec)]))

                                    # total spikes per trial, normalized
                                    soma_spikes_norm = np.sum(soma_count)/syn_num_unique
                                    dend_spikes_norm = np.sum(dend_count)/syn_num_unique
                                    # add to list for all trials
                                    soma_spikes_total.append(soma_spikes_norm)
                                    dend_spikes_total.append(dend_spikes_norm)
                                    all_spikes_total.append(soma_spikes_norm+dend_spikes_norm)

                                    dw_all = pathway['dw'][trial_i]
                                    syn_count = 0
                                    dw_trial=[]
                                    for tree_key, tree in dw_all.iteritems():
                                        for sec_i, sec in enumerate(tree):
                                            for seg_i, seg in enumerate(sec):
                                                syn_count+=1
                                                dw_trial.append(seg)
                                    dw_mean = sum(dw_trial)/float(syn_count)
                                    dw_list.append(dw_mean)

                                dw_mean = np.mean(dw_list)
                                dw_std = np.std(dw_list)
                                dw_sem = stats.sem(dw_list)
                                
                                # group stats
                                soma_total_mean = np.mean(soma_spikes_total)
                                soma_total_std = np.std(soma_spikes_total)
                                soma_total_sem = stats.sem(soma_spikes_total)
                                dend_total_mean = np.mean(dend_spikes_total)
                                dend_total_std = np.std(dend_spikes_total)
                                dend_total_sem = stats.sem(dend_spikes_total)
                                all_total_mean = np.mean(all_spikes_total)
                                all_total_std = np.std(all_spikes_total)
                                all_total_sem = stats.sem(all_spikes_total)

                                # find max number of synapses while all conditions have <1 spike per synapse

                                # print path_combo_key, syn_num_key, pathway_key, all_total_mean
                                dtemp['all_total_mean']=all_total_mean
                                dtemp['all_total_std']=all_total_std
                                dtemp['all_total_sem']=all_total_sem
                                dtemp['dw_mean'] = dw_mean
                                dtemp['dw_std'] = dw_std
                                dtemp['dw_sem'] = dw_sem
        
        # fit sigmoid to num of synapses vs spikes/synapse
        io = {}
        for path_combo_key, path_combo in spikes_temp.iteritems():
            if path_combo_key not in io:
                io[path_combo_key]={}
            for syn_dist_key, syn_dist in path_combo.iteritems():
                if syn_dist_key not in io[path_combo_key]:
                    io[path_combo_key][syn_dist_key]={}
                for syn_num_key, syn_num in syn_dist.iteritems():
                    for polarity_key, polarity in syn_num.iteritems():
                        if polarity_key not in io[path_combo_key][syn_dist_key]:
                            io[path_combo_key][syn_dist_key][polarity_key]={}
                        for pathway_key, pathway in polarity.iteritems():
                            if pathway_key not in io[path_combo_key][syn_dist_key][polarity_key]:
                                io[path_combo_key][syn_dist_key][polarity_key][pathway_key]={'syn_num':[],'all_total_mean':[]}
                            if float(syn_num_key) not in io[path_combo_key][syn_dist_key][polarity_key][pathway_key]['syn_num']:
                                spikes_total = pathway['all_total_mean']

                                if spikes_total<1.1:
                                    io[path_combo_key][syn_dist_key][polarity_key][pathway_key]['syn_num'].append(float(syn_num_key))
                                
                                    io[path_combo_key][syn_dist_key][polarity_key][pathway_key]['all_total_mean'].append(spikes_total)

        # iterate over conditions, fit sigmoid to data
        save_string = 'sigmoid_params.pkl'
        sigmoid_params = self._load_group_data(experiment='exp_4a', save_string=save_string)
        plots={}
        polarities = ['-20', '0', '20']
        polarity_colors = ['blue','black','red']
        for path_combo_key, path_combo in io.iteritems():
            if path_combo_key not in sigmoid_params:
                sigmoid_params[path_combo_key]={}

            if '_'in path_combo_key:
                for syn_dist_key, syn_dist in path_combo.iteritems():
                    if syn_dist_key not in sigmoid_params[path_combo_key]:
                        sigmoid_params[path_combo_key][syn_dist_key]={}
                    if syn_dist_key not in plots:
                        plots[syn_dist_key] = {}
                        # plots_dw[syn_dist_key] = {}
                    for polarity_key, polarity in syn_dist.iteritems():
                        if polarity_key not in sigmoid_params[path_combo_key][syn_dist_key]:
                            sigmoid_params[path_combo_key][syn_dist_key][polarity_key]={}
                        for pathway_key, pathway in polarity.iteritems():
                            if pathway_key not in plots[syn_dist_key]:
                                plots[syn_dist_key][pathway_key]=plt.figure()
                            if pathway_key not in sigmoid_params[path_combo_key][syn_dist_key][polarity_key]:
                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]={}

                                # fit sigmoid 
                                params=[1.,10.,1.]
                                x = np.array(pathway['syn_num'])
                                y = np.array(pathway['all_total_mean'])
                                # print 'x:',x
                                # print 'y:',y
                                data_pass = [x,y]
                                param_opt_obj = scipy.optimize.minimize(self._sigmoid_opt, params, (x, y))
                                if param_opt_obj.success:
                                    param_opt=param_opt_obj.x
                                else:
                                    print 'optimization unsuccessful'
                                    print param_opt_obj.message
                                x = np.array(pathway['syn_num'])
                                ymax = float(param_opt[0])
                                x50 = float(param_opt[1])
                                s = float(param_opt[2])
                                # print 'ymax:', ymax, 'x50:', x50, 's:', s
                                x_fit = np.arange(min(pathway['syn_num']),max(pathway['syn_num']), 0.1)
                                y_fit = ymax/(1.+np.exp((x50-x_fit)/s))


                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['syn_num'] = pathway['syn_num']
                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['all_total_mean'] = pathway['all_total_mean']
                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['params'] = param_opt
                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['sigmoid_fit'] = y_fit
                                sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['x_fit'] = x_fit

                            

                            # plot color and marker
                            color = polarity_colors[polarities.index(polarity_key)]

                            size = 30.
                            opacity = 0.7
                            marker_soma = '.'
                            marker_dend= 'x'
                            if '_' in path_combo_key:
                                marker_all= '^'
                            else:
                                marker_all='.'
                                # print path_combo_key, pathway_key, all_total_mean, float(syn_num_key)
                            print 'xfit:',sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['x_fit']
                            print 'yfit:',sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['sigmoid_fit']
                            # plot with errorbars
                            plt.figure(plots[syn_dist_key][pathway_key].number)

                            plt.plot(sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['syn_num'], sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['all_total_mean'], color=color, marker=marker_all, markersize=size, alpha=opacity, linewidth=0)
                            
                            plt.plot(sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['x_fit'], sigmoid_params[path_combo_key][syn_dist_key][polarity_key][pathway_key]['sigmoid_fit'], color=color, linewidth=5)
                            plt.xlabel('Number of active synapses')
                            plt.ylabel('Number of spikes/synapse')
                            # if '200' in syn_dist_key:
                            #     plt.xlim([0,14])
                            # elif '600' in syn_dist_key:
                            #     plt.xlim([15,42])
                            # plt.ylim([-0.1, 1.1])
                            plt.title(pathway_key+'_pathway_' + syn_dist_key +'_um')

        # save structure of all raw spike data
        save_group_data = sigmoid_params
        with open(data_folder+save_string, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'

        # save and close figure
        for syn_dist_key, syn_dist in plots.iteritems():
            for pathway_key, plot in syn_dist.iteritems():
                plt.figure(plot.number)
                plot_file_name = 'syn_num_x_spikes_total_sigfit_'+pathway_key+'_pathway_' + syn_dist_key +'_um'
                plot.savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[syn_dist_key][pathway_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plot)

        # reorganize so the 
        spikes_temp_norm={}
        plots={}
        plots_dw={}
        for path_combo_key, path_combo in self.group_spike_data.iteritems():
            if path_combo_key != id_list_key:
                spikes_temp_norm[path_combo_key]={}
                for syn_dist_key, syn_dist in path_combo.iteritems():
                    if syn_dist_key not in plots:
                        plots[syn_dist_key] = {}
                        plots_dw[syn_dist_key] = {}
                    spikes_temp_norm[path_combo_key][syn_dist_key]={}
                    for syn_num_key, syn_num in syn_dist.iteritems():
                        spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key]={}
                        for polarity_key, polarity in syn_num.iteritems():
                            spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key]={}
                            for pathway_key, pathway in polarity.iteritems():
                                if pathway_key not in plots[syn_dist_key]:
                                    plots[syn_dist_key][pathway_key]=plt.figure()
                                    plots_dw[syn_dist_key][pathway_key]=plt.figure()
                                spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]={}
                                for dtype_key, dtype in spikes_temp[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key].iteritems():
                                    # print dtype_key, dtype
                                    # print 'denom:',spikes_temp[pathway_key][syn_dist_key][syn_num_key]['0'][pathway_key][dtype_key]
                                    spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key][dtype_key]= dtype#/spikes_temp[pathway_key][syn_dist_key][syn_num_key]['0'][pathway_key][dtype_key]
                                # print 'all_total_mean:', all_total_mean
                                all_total_mean = spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]['all_total_mean']

                                all_total_sem = spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]['all_total_sem']

                                dw_sem = spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]['dw_sem']

                                dw_mean = spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]['dw_mean']

                                dw_std = spikes_temp_norm[path_combo_key][syn_dist_key][syn_num_key][polarity_key][pathway_key]['dw_std']
                                
                                
                                # get polarity index
                                polarity_i = [f_i for f_i, f in enumerate(pathway['p'][0]['field']) if str(f)==polarity_key][0]

                                # plot color and marker
                                color = pathway['p'][0]['field_color'][polarity_i]

                                size = 25.
                                opacity = 0.7
                                marker_soma = '.'
                                marker_dend= 'x'
                                if '_' in path_combo_key:
                                    marker_all= '^'
                                else:
                                    marker_all='.'
                                    # print path_combo_key, pathway_key, all_total_mean, float(syn_num_key)

                                
                                # plot with errorbars
                                plt.figure(plots[syn_dist_key][pathway_key].number)

                                plt.plot(float(syn_num_key), all_total_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                                
                                plt.errorbar(float(syn_num_key), all_total_mean, yerr=all_total_sem, color=color, alpha=opacity)
                                plt.xlabel('Number of active synapses')
                                plt.ylabel('Number of spikes/synapse')
                                if '200' in syn_dist_key:
                                    plt.xlim([0,14])
                                elif '600' in syn_dist_key:
                                    plt.xlim([15,42])
                                plt.ylim([-0.1, 1.1])
                                plt.title(pathway_key+'_pathway_' + syn_dist_key +'_um')

                                # weight plots
                                # plot with errorbars
                                plt.figure(plots_dw[syn_dist_key][pathway_key].number)

                                plt.plot(float(syn_num_key), dw_mean, color=color, marker=marker_all, markersize=size, alpha=opacity)
                                
                                plt.errorbar(float(syn_num_key), dw_mean, yerr=dw_sem, color=color, alpha=opacity)
                                plt.xlabel('Number of synapses')
                                plt.ylabel('Weight change')
                                plt.title(pathway_key+'_pathway_' + syn_dist_key +'_um')

        # save and close figure
        for syn_dist_key, syn_dist in plots.iteritems():
            for pathway_key, plot in syn_dist.iteritems():
                plt.figure(plot.number)
                plot_file_name = 'syn_num_x_spikes_total_'+pathway_key+'_pathway_' + syn_dist_key +'_um'
                plot.savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots[syn_dist_key][pathway_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plot)

        # save and close figure
        for syn_dist_key, syn_dist in plots_dw.iteritems():
            for pathway_key, plot in syn_dist.iteritems():
                plt.figure(plot.number)
                plot_file_name = 'syn_num_x_dw_'+pathway_key+'_pathway_' + syn_dist_key +'_um'
                plot.savefig(data_folder+plot_file_name+'.png', dpi=250)
                with open(data_folder+plot_file_name+'.pkl', 'wb') as output:
                    pickle.dump(plots_dw[syn_dist_key][pathway_key], output,protocol=pickle.HIGHEST_PROTOCOL)
                print 'figure: ', plot_file_name, ' saved'
                plt.close(plot)


        # def _fit_sigmoid(self, params, x, y, **kwargs):
        #     """
        #     """
        #     data_folder = 'Data/'+kwargs['experiment']+'/'
        
        #     # all files in directory
        #     files = os.listdir(data_folder)

        #     save_string = kwargs['save_string']

        #     if save_string in files:
        #         print 'group data found:', save_string
        #         with open(data_folder+save_string, 'rb') as pkl_file:
        #         param_opt = pickle.load(pkl_file)
        #         print 'group data loaded'

        #     else:
        #         param_opt_obj = scipy.optimize.minimize(Experiment._sigmoid_opt(), params)
        #         if param_opt_obj.success:
        #             param_opt=param_opt_obj.x
        #         else:
        #             print 'optimization unsuccessful'
        #             print param_opt_obj.message

        #     return param_opt



        #     # check if parameters are already stored
        #     if 'save_string' in kwargs:
        #         print 'group data found:', save_string
        #     with open(data_folder+save_string, 'rb') as pkl_file:
        #         group_data= pickle.load(pkl_file)
        #     print 'group data loaded'


if __name__ =="__main__":
    # Weights(param.exp_3().p)
    # # Spikes(param.exp_3().p)
    # kwargs = run_control.Arguments('exp_8').kwargs
    # plots = Voltage()
    # plots.plot_all(param.Experiment(**kwargs).p)
    # kwargs = {'experiment':'exp_4a'}
    # Experiment(**kwargs)
    # FitFD()
    pass