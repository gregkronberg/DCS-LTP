''' Scripts for analysis of individual experiments
'''
import numpy as np
import analysis
import glob
import pickle
import copy
import matplotlib.pyplot as plt
import itertools
from neuron import h
from matplotlib import cm as colormap

class Experiment:
    """analyses for individual experiments
    """
    def __init__(self, **kwargs):

        if not kwargs:
            pass
        else:
            experiment = getattr(self, kwargs['experiment'])

            experiment(**kwargs) 

    def exp_3a1(self, **kwargs):
        ''' 20 Hz (40 pulses) in apical dendrites
        ===Args===
        -experiment  : experiment name (e.g. 'exp_3a')
        -run_clopath
        ===Out===
        ===Updates===
        ===Comments===
        '''

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        self.group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=self.group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':100E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-67,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':20,#-38, # potentiation threshold (mV)
        'tau_p': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 40E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        'LTD_delay':1,
        }

        if 'clopath_param' in kwargs:
            for key, val in kwargs['clopath_param']:
                clopath_param[key]=val

        if 'run_clopath' in kwargs and kwargs['run_clopath']:
            # apply clopath learning rule and return matrix of weight time series
            self.w_group = self.C._clopath(x=self.group_data['v']['input_mat'], u=self.group_data['v']['data_mat'], param=clopath_param)

            # update group data
            self.group_data['gbar']['data_mat'] = self.w_group
            self.group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=self.group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

        # final weight change for each condition as bar graph
        weight_plot_final = analysis._plot_weights_final(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_final.png'
        weight_plot_final.savefig(directory+plot_file_name, dpi=250)

    def exp_3a1_basal(self, **kwargs):

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        self.group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=self.group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':100E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-67,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':20,#-38, # potentiation threshold (mV)
        'tau_p': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 40E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        'LTD_delay':1,
        }

        if 'clopath_param' in kwargs:
            for key, val in kwargs['clopath_param']:
                clopath_param[key]=val

        if 'run_clopath' in kwargs and kwargs['run_clopath']:
            # apply clopath learning rule and return matrix of weight time series
            self.w_group = self.C._clopath(x=self.group_data['v']['input_mat'], u=self.group_data['v']['data_mat'], param=clopath_param)

            # update group data
            self.group_data['gbar']['data_mat'] = self.w_group
            self.group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=self.group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

        # final weight change for each condition as bar graph
        weight_plot_final = analysis._plot_weights_final(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_final.png'
        weight_plot_final.savefig(directory+plot_file_name, dpi=250)

    def exp_3a2(self, **kwargs):

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        self.group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=self.group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':100E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-67,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':20,#-38, # potentiation threshold (mV)
        'tau_p': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 40E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        'LTD_delay':1,
        }

        if 'run_clopath' in kwargs and kwargs['run_clopath']:
            # apply clopath learning rule and return matrix of weight time series
            self.w_group = self.C._clopath(x=self.group_data['v']['input_mat'], u=self.group_data['v']['data_mat'], param=clopath_param)

            # update group data
            self.group_data['gbar']['data_mat'] = self.w_group
            self.group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=self.group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

        # final weight change for each condition as bar graph
        weight_plot_final = analysis._plot_weights_final(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_final.png'
        weight_plot_final.savefig(directory+plot_file_name, dpi=250)

    def exp_3a2_basal(self, **kwargs):

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        self.group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=self.group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':100E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-67,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':20,#-38, # potentiation threshold (mV)
        'tau_p': 3, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 40E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        'LTD_delay':1,
        }

        if 'clopath_param' in kwargs:
            for key, val in kwargs['clopath_param']:
                clopath_param[key]=val

        if 'run_clopath' in kwargs and kwargs['run_clopath']:
            # apply clopath learning rule and return matrix of weight time series
            self.w_group = self.C._clopath(x=self.group_data['v']['input_mat'], u=self.group_data['v']['data_mat'], param=clopath_param)

            # update group data
            self.group_data['gbar']['data_mat'] = self.w_group
            self.group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=self.group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

        # final weight change for each condition as bar graph
        weight_plot_final = analysis._plot_weights_final(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_final.png'
        weight_plot_final.savefig(directory+plot_file_name, dpi=250)
    
    def exp_3a_find_clopath_range(self, **kwargs):
        ''' find clopath parameter combinations that qualitatively reproduce experimental results for 20 Hz and TBS LTP

        ===Args===
        ===Out===
        -w      : dictionary with parameter fits
                -dw_mean        : list of net weight change for each condition ([cathodal_20Hz, control_20Hz, anodal_20Hz, cathodal_TBS, control_TBS, anodal_TBS])
                -clopath_param      : dictionary of clopath parameter values for the corresponding weight change
        ===Updates===
        ===Comments===
        '''
        # iterate through range of clopath parameters
        # classify weight changes as being qualitatively fit to data or not

        # load group data for each experiment
        # get list of data files
        directory3a1 = 'Data/'+'exp_3a1'+'/'
        directory3a2 = 'Data/'+'exp_3a2'+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data_3a1 = analysis._load_group_data(directory=directory3a1, file_name=group_data_file_name)
        self.group_data_3a2 = analysis._load_group_data(directory=directory3a2, file_name=group_data_file_name)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()

        param_iter = {
        'A_m0':[.1E-5, 1E-5, 10E-5, 100E-5, ],
        'tetap': [-70, -68,],#-67, -65, -63, -61,],
        'tau_m':[10, 20, 30, 40,],
        'tau_p': [5],#range(3, 30, 5),
        'LTD_delay':[0.5, 1]
        }

        # if kwargs:
        #     for key, val in kwargs.iteritems():
        #         param_iter[key]=val

        param_val_list=[]
        param_key_list=[]
        for key, val in param_iter.iteritems():
            param_val_list.append(val)
            param_key_list.append(key)

        param_combos = list(itertools.product(*param_val_list))
        # print param_combos
        # updated clopath parameters
        clopath_param= {
        'A_m0':4E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-60,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':40,#-38, # potentiation threshold (mV)
        'tau_p': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 38E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        'LTD_delay':1,
        }

        file_name = 'weights_x_clopath_param_tetap'+'.pkl'
        print 'loading previous results'
        w = analysis._load_group_data(directory=directory3a1, file_name=file_name)
        if not w:    
            print 'no previous results found'
            w = {'dw_mean':[], 'clopath_param':[], 'fit':[]}

        for combo_i, combo in enumerate(param_combos):
            try:
                for param_val_i, param_val in enumerate(combo):
                    param_key = param_key_list[param_val_i]
                    clopath_param[param_key]=param_val

                if clopath_param not in w['clopath_param']:
                    # print clopath_param
                
                    # apply clopath learning rule and return matrix of weight time series
                    self.w_group_3a1 = self.C._clopath(x=self.group_data_3a1['v']['input_mat'], u=self.group_data_3a1['v']['data_mat'], param=clopath_param)
                    self.w_group_3a2 = self.C._clopath(x=self.group_data_3a2['v']['input_mat'], u=self.group_data_3a2['v']['data_mat'], param=clopath_param)

                    # update group data
                    self.group_data_3a1['gbar']['data_mat'] = self.w_group_3a1
                    self.group_data_3a1['p_clopath'] = clopath_param
                    # update group data
                    self.group_data_3a2['gbar']['data_mat'] = self.w_group_3a2
                    self.group_data_3a2['p_clopath'] = clopath_param

                    condition='polarity'
                    condition_vals = ['-20.0','0.0','20.0']
                    dw_mean_3a1=[]
                    dw_mean_3a2 =[]
                    # compute net weight change for each condition
                    for condition_i, condition_val in enumerate(condition_vals):
                        # get indeces that match the conditions
                        # print group_data['gbar']
                        indices_3a1 = [i for i,value in enumerate(self.group_data_3a1['gbar']['conditions'][condition]) if value==condition_val]
                        dw_3a1 = self.group_data_3a1['gbar']['data_mat'][indices_3a1,-1]/self.group_data_3a1['gbar']['data_mat'][indices_3a1,0]
                        dw_mean_3a1.append(np.mean(dw_3a1))

                        indices_3a2 = [i for i,value in enumerate(self.group_data_3a2['gbar']['conditions'][condition]) if value==condition_val]
                        dw_3a2 = self.group_data_3a2['gbar']['data_mat'][indices_3a2,-1]/self.group_data_3a2['gbar']['data_mat'][indices_3a2,0]
                        dw_mean_3a2.append(np.mean(dw_3a2))

                    # net LTP
                    condition1 = dw_mean_3a1[1] > 1 and dw_mean_3a2[1] > 1
                    # 20 Hz: cathodal boost, anodal reduce, cathodal boost larger than anodal reduce
                    condition2 = dw_mean_3a1[0] > dw_mean_3a1[1] and dw_mean_3a1[2] < dw_mean_3a1[1] and np.abs(dw_mean_3a1[0]-  dw_mean_3a1[1]) >= np.abs(dw_mean_3a1[2]-dw_mean_3a1[1])
                    # TBS: anodal boost, cathodal reduce, anodal boost larger than cathodal reduce
                    condition3 = dw_mean_3a2[0] < dw_mean_3a2[1] and dw_mean_3a2[2] > dw_mean_3a2[1] and np.abs(dw_mean_3a2[0]-  dw_mean_3a2[1]) <= np.abs(dw_mean_3a2[2]-dw_mean_3a2[1])

                    condition4 = dw_mean_3a2[2] > dw_mean_3a2[1] and np.abs(dw_mean_3a2[0]-  dw_mean_3a2[1]) <= 0.2*np.abs(dw_mean_3a2[2]-dw_mean_3a2[1])

                    
                    # store weight changes and parameters
                    w['dw_mean'].append(copy.copy(dw_mean_3a1+dw_mean_3a2))
                    w['clopath_param'].append(copy.copy(clopath_param))
                    # print w['clopath_param']
                    # print w['dw_mean']
                    if condition1 and condition2 and condition4:
                        w['fit'].append(True)
                        print 'GOOD PARAMETERS FOUND:', clopath_param 
                        print 'weights:',w['dw_mean'][-1]
                    else:
                        w['fit'].append(False)
                        print 'bad parameters:', clopath_param 
                        print 'weights:',w['dw_mean'][-1]
                else:   
                    print 'parameters tested already:',clopath_param


            except KeyboardInterrupt:

                print 'keyboard interrupt exception: saving w'
                # save updated group data
                file_name = 'weights_x_clopath_param_tetap'+'.pkl'
                analysis._save_group_data(group_data=w, directory=directory3a1, file_name=file_name)

                print 'w saved'
                raise


        # save updated group data
        file_name = 'weights_x_clopath_param_tetap'+'.pkl'
        analysis._save_group_data(group_data=w, directory=directory3a1, file_name=file_name)

    def exp_3a_find_clopath_analysis(self, **kwargs):
        directory3a1 = 'Data/'+'exp_3a1'+'/'
        file_name = 'weights_x_clopath_param_tetap'+'.pkl'
        with open(directory3a1+file_name, 'rb') as pkl_file:
            self.w= pickle.load(pkl_file)

        for i, dw_mean in enumerate(self.w['dw_mean']):

            dw_mean_3a1 = dw_mean[:3]
            dw_mean_3a2 = dw_mean[3:]
            # print dw_mean_3a1
            # print dw_mean_3a2
            # net LTP
            condition1 = dw_mean_3a1[1] > 1 and dw_mean_3a2[1] > 1
            # 20 Hz: cathodal boost, anodal reduce, cathodal boost larger than anodal reduce
            condition2 = dw_mean_3a1[0] > dw_mean_3a1[1] and dw_mean_3a1[2] < dw_mean_3a1[1] and np.abs(dw_mean_3a1[0]-  dw_mean_3a1[1]) >= np.abs(dw_mean_3a1[2]-dw_mean_3a1[1])
            # TBS: anodal boost, cathodal reduce, anodal boost larger than cathodal reduce
            condition3 = dw_mean_3a2[0] < dw_mean_3a2[1] and dw_mean_3a2[2] > dw_mean_3a2[1] and np.abs(dw_mean_3a2[0]-  dw_mean_3a2[1]) <= np.abs(dw_mean_3a2[2]-dw_mean_3a2[1])

            condition4 = dw_mean_3a2[2] > dw_mean_3a2[1] and np.abs(dw_mean_3a2[0]-  dw_mean_3a2[1]) <= 0.2*np.abs(dw_mean_3a2[2]-dw_mean_3a2[1])
            # print dw_mean
            # print condition1, condition2, condition4

            if condition1 and condition2 and condition4:
                print dw_mean, self.w['clopath_param'][i]

        ind = self.w['fit'].index(False)
        # print self.w['dw_mean'][ind], self.w['clopath_param'][ind]

    def exp_3a3(self, **kwargs):

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        self.group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        self.group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=self.group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':3E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-65,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':40,#-38, # potentiation threshold (mV)
        'tau_p': 5, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 38E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        }

        # apply clopath learning rule and return matrix of weight time series
        self.w_group = self.C._clopath(x=self.group_data['v']['input_mat'], u=self.group_data['v']['data_mat'], param=clopath_param)

        # update group data
        self.group_data['gbar']['data_mat'] = self.w_group
        self.group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=self.group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=self.group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

    def exp_3c1(self, **kwargs):
        '''
        ===Args===
        ===Out===
        ===Updates===
        ===Comments===
        '''
        # get list of all data files
        

        directory = 'Data/' + kwargs['experiment'] + '/'
        search_string = '*data_*'
        data_files = glob.glob(directory+search_string)
        print data_files
        with open(data_files[0], 'rb') as pkl_file:
            self.data = pickle.load(pkl_file)

        print self.data.keys()

        self.shape_data={}
        self.patches={}
        self.colors={}
        p=self.data['p']
        fig, ax = plt.subplots(figsize=(2,10),nrows=1, ncols=1)
        path = p['p_path'].values()[0]
        for field_key, field in self.data.iteritems():
            if field_key != 'p':
                self.shape_data[field_key]={}
                for tree_key, tree in p['seg_dist'].iteritems():
                    self.shape_data[field_key][tree_key]=[]
                    for sec_i, sec in enumerate(tree):
                        self.shape_data[field_key][tree_key].append([])
                        for seg_i, seg in enumerate(sec):
                            # print tree_key, sec_i, seg_i

                            if tree_key=='soma':
                                # print (field.keys())
                                # current = field.values()[0][tree_key+'_v']
                                # control = self.data['0.0'].values()[0][tree_key+'_v'][sec_i][-1]
                                current = field.values()[0][tree_key+'_v'][sec_i][seg_i][-1]
                                control = self.data['0.0'].values()[0][tree_key+'_v'][sec_i][seg_i][-1]
                            else:
                                current = field.values()[0][tree_key+'_v'][sec_i][seg_i][-1]
                                control = self.data['0.0'].values()[0][tree_key+'_v'][sec_i][seg_i][-1]

                            value = current-control
                            self.shape_data[field_key][tree_key][sec_i].append(value)

                self.patches[field_key], self.colors[field_key] = analysis.ShapePlot().basic(morpho=p['morpho'], data=self.shape_data[field_key], axes=ax, width_scale=3, colormap=colormap.coolwarm)
        

        # plot anodal
        # plt.show()
        # print self.patches['20.0']
        print self.shape_data['20.0']['soma'][0][0]
        plot_file_name = 'membrane_polarization_shapeplot_anodal.png' 
        # ax.plot(self.shape_data['20.0']['soma'][0][0])
        # create patch collection
        # p = PatchCollection(patches, cmap=colormap, alpha=1.)
        # set colors
        # p.set_array(np.array(colors))
        # plot collection
        mappable = ax.add_collection(self.patches['20.0'])
        # ax.autoscale()
        # ax.axis('equal')
        # ax.set_ylim(bottom=-200, top=600)
        # ax.set_xlim(left=-150,right=150)
        ax.axis('equal')
        # show colorbar
        fig.colorbar(self.patches['20.0'])
        # autoscale axes
        # print ax
        
        # fig.show()
        plt.show()

        fig.savefig(directory+plot_file_name, dpi=300)

        fig, ax = plt.subplots(figsize=(2,10),nrows=1, ncols=1)
        plot_file_name = 'membrane_polarization_shapeplot_cathodal.png' 
        # # create patch collection
        # # p = PatchCollection(patches, cmap=colormap, alpha=1.)
        # # set colors
        # # p.set_array(np.array(colors))
        # # plot collection
        ax.add_collection(self.patches['-20.0'])
        # # show colorbar
        fig.colorbar(self.patches['-20.0'])
        # # autoscale axes
        # ax.autoscale()
        ax.axis('equal')
        plt.show()
        fig.savefig(directory+plot_file_name, dpi=300)


        # create shape plot
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


    def exp_3a_find_clopath_range(self, **kwargs):
        self.parameters = []
        tetap = [-70, -68, -66, -64, -62, -60, -58, -56, -54, -52]
        for tetap_val in tetap:
            self.parameters.append(
                {'experiment':'exp_3a_find_clopath_range',
                'A_m0':[1E-5, 2E-5, 4E-5, 8E-5, 16E-5, 32E-5,64E-5],
                'tetap': [tetap_val],
                'tau_m':[10, 20, 30, 40, 50, 60],
                'tau_p': range(2, 40, 5)})
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

if __name__=='__main__':
    Experiment(experiment='exp_3a1_basal', run_clopath=True)
    # params=ExperimentsParallel('exp_3a_find_clopath_range').parameters
    # _run_parallel(params)

