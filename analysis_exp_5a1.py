"""
analysis

"""
import figsetup_exp_5a1 as figsetup
import numpy as np
import analysis
import glob
import pickle
import copy
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm as colormap
from scipy import stats
import os

'''
'''
kwargs = {'experiment':'exp_5a1'}
dpi=350

#############################################################################
# directories and filenames
#############################################################################
# directories
directory = 'Data/'+kwargs['experiment']+'/'
figure_directory = 'Data/'+kwargs['experiment']+'/Figures/'
# make figure directory if doesnt exist
if os.path.isdir(figure_directory) is False:
    os.mkdir(figure_directory)
# filenames    
filename_vtrace = 'vtrace_df.h5'
# filename_w_clopath = 'w_clopath_df.pkl'
filename_w_clopath = 'w_clopath_df.pkl'
filename_w_clopath2 = 'w_clopath2_df.pkl'
filename_w_clopath_stdp = 'w_clopath_stdp_df.pkl'
filename_w_clopath_stdp2 = 'w_clopath_stdp2_df.pkl'
filename_spikes = 'spikes_df.pkl'

arrayfuncs = analysis.ArrayFunctions()


#############################################################################
# load variables
#############################################################################
# vtrace_df = analysis._load_group_data(directory=directory, filename=filename_vtrace)
# spikes_df = analysis._load_group_data(directory=directory, filename=filename_spikes)
w_clopath_df = analysis._load_group_data(directory=directory, filename=filename_w_clopath)
w_clopath2_df = analysis._load_group_data(directory=directory, filename=filename_w_clopath2)
w_clopath_stdp_df = analysis._load_group_data(directory=directory, filename=filename_w_clopath_stdp)
w_clopath_stdp2_df = analysis._load_group_data(directory=directory, filename=filename_w_clopath_stdp2)

#####################################################################
# frequency response
#####################################################################
run_weights=False
if run_weights:

    # dose response figures
    #-------------------------------------------------------------------
    figtype='frequency_response_stdp_param_'
    figdf = figsetup.BuildFigDF()._frequency_response()
    # figdf.reset_index()
    w_clopath_stdp_df.set_index('stdp_dt', inplace=True)
    figs, ax = analysis.PlotFuncs()._var2var_mean(df=w_clopath_stdp_df, figdf=figdf, variable='dw_clopath', x_variable='path_1_pulse_freq')
    figs, ax = analysis.PlotFuncs()._draw_lines(fig=figs, ax=ax, figdf=figdf)
    w_clopath_stdp_df.reset_index( inplace=True)
    # save figure
    #------------
    for fig_key, fig in figs.iteritems():
        fname = figure_directory+figtype+str(fig_key)+'.png'
        fig.savefig(fname, format='png', dpi=dpi)

#####################################################################
# frequency response
#####################################################################
run_weights=False
if run_weights:

    # dose response figures
    #-------------------------------------------------------------------
    figtype='frequency_response_stdp2_param_'
    figdf = figsetup.BuildFigDF()._frequency_response()
    # figdf.reset_index()
    w_clopath_stdp2_df.set_index('stdp_dt', inplace=True)
    figs, ax = analysis.PlotFuncs()._var2var_mean(df=w_clopath_stdp2_df, figdf=figdf, variable='dw_clopath', x_variable='path_1_pulse_freq')
    figs, ax = analysis.PlotFuncs()._draw_lines(fig=figs, ax=ax, figdf=figdf)
    w_clopath_stdp2_df.reset_index( inplace=True)
    # save figure
    #------------
    for fig_key, fig in figs.iteritems():
        fname = figure_directory+figtype+str(fig_key)+'.png'
        fig.savefig(fname, format='png', dpi=dpi)

#####################################################################
# frequency response
#####################################################################
run_weights=False
if run_weights:

    # dose response figures
    #-------------------------------------------------------------------
    figtype='frequency_response_'
    figdf = figsetup.BuildFigDF()._frequency_response()
    # figdf.reset_index()
    w_clopath_df.set_index('stdp_dt', inplace=True)
    figs, ax = analysis.PlotFuncs()._var2var_mean(df=w_clopath_df, figdf=figdf, variable='dw_clopath', x_variable='path_1_pulse_freq')
    figs, ax = analysis.PlotFuncs()._draw_lines(fig=figs, ax=ax, figdf=figdf)
    w_clopath_df.reset_index( inplace=True)
    # save figure
    #------------
    for fig_key, fig in figs.iteritems():
        fname = figure_directory+figtype+str(fig_key)+'.png'
        fig.savefig(fname, format='png', dpi=dpi)


#####################################################################
# frequency response
#####################################################################
run_weights=True
if run_weights:

    # dose response figures
    #-------------------------------------------------------------------
    figtype='frequency_response2_'
    figdf = figsetup.BuildFigDF()._frequency_response()
    # figdf.reset_index()
    w_clopath2_df.set_index('stdp_dt', inplace=True)
    figs, ax = analysis.PlotFuncs()._var2var_mean(df=w_clopath2_df, figdf=figdf, variable='dw_clopath', x_variable='path_1_pulse_freq')
    figs, ax = analysis.PlotFuncs()._draw_lines(fig=figs, ax=ax, figdf=figdf)
    w_clopath2_df.reset_index( inplace=True)
    # save figure
    #------------
    for fig_key, fig in figs.iteritems():
        fname = figure_directory+figtype+str(fig_key)+'.png'
        fig.savefig(fname, format='png', dpi=dpi)

plt.show(block=False)