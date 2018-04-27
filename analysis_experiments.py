''' Scripts for analysis of individual experiments
'''
import numpy as np
import analysis
import glob
import pickle
import copy
import matplotlib.pyplot as plt

class Experiment:
    """analyses for individual experiments
    """
    def __init__(self, **kwargs):
        experiment = getattr(self, kwargs['experiment'])

        experiment(**kwargs) 

    def exp_3a1(self, **kwargs):
        ''' 20 Hz (40 pulses, 2 seconds)

        ===Figures===
        -average weight change (across all synapses) over time
        '''

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':4E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-65,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':40,#-38, # potentiation threshold (mV)
        'tau_p': 10, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 38E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        }

        # apply clopath learning rule and return matrix of weight time series
        self.w_group = self.C._clopath(x=group_data['v']['input_mat'], u=group_data['v']['data_mat'], param=clopath_param)

        # update group data
        group_data['gbar']['data_mat'] = self.w_group
        group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

    def exp_3a2(self, **kwargs):

        # get list of data files
        directory = 'Data/'+kwargs['experiment']+'/'
        search_string = '*data*'

        # load group data
        group_data_file_name = 'group.pkl'
        group_data = analysis._load_group_data(directory=directory, file_name=group_data_file_name)

        # update group data
        variables = ['v', 'gbar']
        group_data = analysis._update_group_data(directory=directory, search_string=search_string, group_data=group_data, variables=variables)

        # apply clopath rule to all data
        # load clopath class
        self.C = analysis.Clopath()
        
        # updated clopath parameters
        clopath_param= {
        'A_m0':4E-5, # depression magnitude parameter (mV^-1)
        'tetam':-70,#-41, # depression threshold (mV)
        'tetap':-65,#-38, # potentiation threshold (mV)
        'tau_x':8,#-38, # potentiation threshold (mV)
        'tau_m':40,#-38, # potentiation threshold (mV)
        'tau_p': 10, # time constant (ms) for low pass filter post membrane potential for potentiation
        'A_p': 38E-5, # amplitude for potentiation (mV^-2)
        'delay':0, # conduction delay (ms)
        }

        # apply clopath learning rule and return matrix of weight time series
        self.w_group = self.C._clopath(x=group_data['v']['input_mat'], u=group_data['v']['data_mat'], param=clopath_param)

        # update group data
        group_data['gbar']['data_mat'] = self.w_group
        group_data['p_clopath'] = clopath_param

        # save updated group data
        analysis._save_group_data(group_data=group_data, directory=directory, file_name=group_data_file_name)

        # plot average weight change across all synapses for a given polarity
        condition='polarity'
        condition_vals = ['-20.0','0.0','20.0']
        colors = ['blue', 'black', 'red']
        weight_plot = analysis._plot_weights_mean(group_data=group_data, condition=condition, condition_vals=condition_vals, colors=colors)
        plt.show()

        plot_file_name = 'weight_plot_mean.png'
        weight_plot.savefig(directory+plot_file_name, dpi=250)

if __name__=='__main__':
    Experiment(experiment='exp_3a1')

