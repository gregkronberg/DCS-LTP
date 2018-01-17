"""
analysis

data structure for each trial is organized as ['tree'][polarity][section][segment]
"""
from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as it
import os
import cPickle as pickle
import param
import cell
import math
import run_control
import copy
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm as colormap

class ShapePlot():
    """ create plot of neuron morphology with data as a colormap

    similar to ShapePlot in neuron
    """
    def __init__(self):
        """
        """
        pass

    def basic(self, morpho, data, axes, colormap=colormap.jet):
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
            parent_diam = parent[5]
            child_diam = child[5]
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
        axes.add_collection(p)
        # show colorbar
        # plt.colorbar(p)
        # autoscale axes
        axes.autoscale()
        return axes

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
            self.soma_spikes = self.detect_spikes(data['soma_v'][pol][0][0])['times']
            if self.soma_spikes.size!=0:
                
                # add spike times to array
                self.spiket_soma[pol] = np.append(self.spiket_soma[pol],self.soma_spikes*p['dt'],axis=1)

                # track cell number
                for spike_i,spike in enumerate(self.soma_spikes[0,:]):
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
                    dend_spikes = self.detect_spikes(np.array(data[p['tree']+'_v'][pol][sec_i][seg_i]))['times']
                    if dend_spikes.size!=0:
                        # add spike times to array
                        self.spiket_dend[pol] = np.append(self.spiket_dend[pol],dend_spikes,axis=1)
                        # spiket_dend_track = np.append(spiket_dend_track,dend_spikes,axis=1)
                        # for each spike store the section, segment, cell number in the appropriate list
                        for spike in dend_spikes[0,:]:
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
                
                # plot range variables (automatically saved to same folder)
                self.plot_trace(data=data, 
                    trees=p_data['trees'], 
                    sec_idx=p_data['sec_idx'], 
                    seg_idx=p_data['seg_idx'],
                    variables=p_data['plot_variables'],
                    x_variables=p_data['x_variables'],
                    file_name=p_data['trial_id'],
                    group_trees=p_data['group_trees'])

    def plot_trace(self, data, trees, sec_idx, seg_idx, soma=True, axon=True, group_trees=True, variables=['v'], x_variables='t', xlim=[], ylim=[], file_name=''):
        """ Create a figure containing a subplot for each segment specified in seg_idx

        variable is a list of range variable key words indicating which varaible to plot.  A new figure is created and saved for each variable
        """
        # load parameters
        p = data['p']

        # number field intensities/polarities
        n_pol = len(p['field'])
    
        # dictionary to hold figures
        fig={}  

        # iterate over list of y variables to plot
        for var in variables:

            fig[var] = {}

            # iterate over list of x variables to plot
            for x_var in x_variables:


                # if data from all subtrees in one plot
                if group_trees:

                    # create figure
                    fig[var][x_var] = plt.figure()

                    # number of segments to plot
                    nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for tree_key, tree in seg_idx.iteritems() for sec in tree if tree_key in p['trees'] ])+1

                    print nseg
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
                    fig[var][x_var]={}
            
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
                            fig[var][x_var][tree_key] = plt.figure()

                            # number of segments to plot
                            nseg =  sum([sum(seg_i+1 for seg_i,seg in enumerate(sec)) for sec in tree])+1
                            
                            print'nseg:',nseg
                            # plot soma trace?
                            if soma:
                                nseg+=1
                            if axon:
                                nseg+=1

                            # columns and rows for subplot array    
                            cols = int(math.ceil(math.sqrt(nseg)))
                            rows = int(math.ceil(math.sqrt(nseg)))

                            print 'rows,cols:',rows,cols

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
                                    if data[str(f)][tree_key+'_'+var][sec_i]:

                                        # if not plotting the soma trace
                                        if soma and cnt<nseg:

                                            # retrieve time series to plot
                                            v = data[str(f)][tree_key+'_'+var][sec_i][seg_i]

                                            # retrieve x variable
                                            if x_var =='t':
                                                # time vector
                                                xv = data[str(f)]['t']

                                            # other variable from arguments
                                            else:
                                                xv = data[str(f)][tree_key+'_'+x_var][sec_i][seg_i]


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
                                if 'soma_'+var in data[str(f)].keys():
                                    if len(data[str(f)]['soma_'+var][0][0])>0:
                                    
                                        # subplot for soma trace
                                        plt.subplot(rows, cols, cnt)

                                        # adjust limits
                                        if var is 'v':
                                            if xlim:
                                                plt.xlim(xlim)
                                            if ylim:
                                                plt.ylim(ylim)

                                        # retrieve data to plot
                                        v = data[str(f)]['soma_'+var][0][0] 

                                        # determine x variable to plot
                                        if x_var =='t':
                                            # time vector
                                            xv = data[str(f)]['t']
                                        else:
                                            xv = data[str(f)]['soma_'+x_var][0][0]
                                        
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
                                if 'axon_'+var in data[str(f)].keys():
                                    if len(data[str(f)]['axon_'+var][0][0])>0:
                                    
                                        # subplot for soma trace
                                        plt.subplot(rows, cols, cnt)

                                        # adjust limits
                                        if var is 'v':
                                            if xlim:
                                                plt.xlim(xlim)
                                            if ylim:
                                                plt.ylim(ylim)

                                        # retrieve data to plot
                                        v = data[str(f)]['axon_'+var][0][0] 

                                        # determine x variable to plot
                                        if x_var =='t':
                                            # time vector
                                            xv = data[str(f)]['t']
                                        else:
                                            xv = data[str(f)]['axon_'+x_var][0][0]
                                        
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
                            file_name_add = tree_key+'_trace_'+x_var+'_x_'+var

                            # save figure
                            fig[var][x_var][tree_key].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

                            # close figure
                            plt.close(fig[var][x_var][tree_key])

                    # if all trees are in the same figure
                    if group_trees:
                        all_trees =''
                        for tree_key, tree in seg_idx.iteritems():
                            all_trees = all_trees+tree_key+'_'

                        # info to add to file name
                        file_name_add = all_trees+'trace_'+x_var+'_x_'+var

                        # save and close figure
                        fig[var][x_var].savefig(p['data_folder']+file_name_add+file_name+'.png', dpi=300)

                        plt.close(fig[var][x_var])

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

    def exp_1(self, **kwargs):
        plots = PlotRangeVar()
        plots.plot_all(kwargs['p'])

    
    def exp_2(self, **kwargs):
        pass
        # npol = 3 
        # data_folder = 'Data/'+kwargs['exp']+'/'
        # files = os.listdir(data_folder)
        # dw = np.zeros([npol, len(files)])
        # syn_weight = np.zeros([1,len(files)])
        # for data_file_i, data_file in enumerate(files):
        #   # check for proper data file format
        #   if 'data' in data_file:

        #       with open(data_folder+data_file, 'rb') as pkl_file:
        #           data = pickle.load(pkl_file)
        #       p = data['p']

        #       for field_i, field in enumerate(p['field']):
        #           for sec_i,sec in enumerate(p['sec_idx']):
        #               for seg_i,seg in enumerate(p['seg_idx'][sec_i]):
        #                   w_end = data[p['tree']+'_w'][field_i][sec][seg][-1]
        #                   w_start = data[p['tree']+'_w'][field_i][sec][seg][0]
        #                   dw[field_i,data_file_i] = w_end/w_start
        #                   syn_weight[0,data_file_i] = p['w_list'][sec_i][seg]


        # fig = plt.figure()
        # for pol in range(npol):
        #       plt.plot(syn_weight[0,:], dw[pol, :], '.', color=p['field_color'][pol])
        # plt.xlabel('peak conductance (uS)')
        # plt.ylabel('weight change')
        # fig.savefig(data_folder+'fig_dw'+'.png', dpi=250)
        # plt.close(fig)

    def exp_2b(self, **kwargs):
        """ activate groups of synapses at varying distance from soma until dendritic or somatic spikes are observed
        """
        pass

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

    def exp_2e(self, **kwargs):
        """ Apply a single theta burst to synapses with varying mean distance from soma 
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
            # data organized as {location}{data_type}[trials][polarity]
            spike_data= []

        # string to save group data
        save_string_group_data_proc = 'spikes_grouped_proc'+'.pkl'
        # if data file already exists
        if save_string_group_data_proc in files:
            # load data
            print 'processed spike data found'
            with open(data_folder+save_string_group_data_proc, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
            print 'processed spike data loaded'
        # otherwise create data structure
        else:
            # data organized as {location}{data_type}[trials][polarity]
            spike_data_group = {}

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

                    spike_data.append({})

                    # add to processed list
                    id_list_spike_times.append(data_file)

                    spike_data[-1]['spikes'] = {'soma':[],'dend':{}, 'dw':{}}

                    for tree_key, tree in p['sec_idx'].iteritems():

                        spike_data[-1]['spikes']['dend'][tree_key]=[]
                        spike_data[-1]['spikes']['dw'][tree_key]=[]
                        
                        for sec_i, sec in enumerate(tree):

                            spike_data[-1]['spikes']['dend'][tree_key].append([])
                            spike_data[-1]['spikes']['dw'][tree_key].append([])

                            for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):

                                spike_data[-1]['spikes']['dend'][tree_key][-1].append([])
                                spike_data[-1]['spikes']['dw'][tree_key][-1].append([])
                                                    # iterate over field polarities
                                for f_i, f in enumerate(p['field']):

                                    # detect spikes in dendrite
                                    spike_data[-1]['spikes']['dend'][tree_key][-1][-1].append(spike_analysis.detect_spikes(data[str(f)][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0])

                                    # detect spikes in dendrite
                                    spike_data[-1]['spikes']['dw'][tree_key][-1][-1].append(data[str(f)][tree_key+'_gbar'][sec_i][seg_i][-1]/data[str(f)][tree_key+'_gbar'][sec_i][seg_i][0])

                    for f_i, f in enumerate(p['field']):
                                    # detect spikes in soma
                        spike_data[-1]['spikes']['soma'].append(spike_analysis.detect_spikes(data[str(f)]['soma_v'][0][0], threshold=threshold)['times'][0])

                    spike_data[-1]['p']=p

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'id list saved'
        
        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data_raw, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'

        # spike data organized as {syn_num}{w_mean}[trials]{variable}[polarity]
        spike_data_group ={}
        spike_data_dist_bin = {}
        plots_spikes={} 
        plots_weights ={}
        plots_weights_mean = {}
        time_bins = [[1200, 1600],[1600, 2000],[2000, 2400],[2400, 2800]]
        dist_bins = [[0., 100.],[100., 200.],[200., 300.],[300.,400.],[400., 600.]]
        for trial_i, trial in enumerate(spike_data):
            p = spike_data[trial_i]['p']
            syn_num = p['syn_num']
            w_mean = p['w_mean']
            # print syn_num, w_mean
            if str(syn_num) not in spike_data_group:
                spike_data_group[str(syn_num)]={}
                spike_data_dist_bin[str(syn_num)]={}
                plots_spikes[str(syn_num)]={}
                plots_weights[str(syn_num)]={}
                plots_weights_mean[str(syn_num)]={}
            if str(w_mean) not in spike_data_group[str(syn_num)]:
                spike_data_group[str(syn_num)][str(w_mean)]=[]
                spike_data_dist_bin[str(syn_num)][str(w_mean)]=[]
                for f_i, f in enumerate(p['field']):
                    spike_data_group[str(syn_num)][str(w_mean)].append({'dw':[],'dw_mean':[],'dist':[],'dist_spikes':[],'dist_mean':[],'spikes_dend':[],'spikes_soma':[], 'dist_spikes_binned':[],'spikes_dend_binned':[],'spikes_soma_binned':[], 'spikes_dend_diff_binned':[],'dist_bin_i':[]})
                    spike_data_dist_bin[str(syn_num)][str(w_mean)].append([])
                    for dist_bin_i, dist_bin in enumerate(dist_bins):
                        spike_data_dist_bin[str(syn_num)][str(w_mean)][f_i].append({'pre':[],'post':[]})
                plots_spikes[str(syn_num)][str(w_mean)]=plt.figure()
                plots_weights[str(syn_num)][str(w_mean)]=plt.figure()
                plots_weights_mean[str(syn_num)][str(w_mean)]=plt.figure()
            
            for f_i, f in enumerate(p['field']):
                # spike_data_group[str(syn_num)][str(w_mean)].append({'dw':[],'dw_mean':[],'dist':[],'dist_spikes':[],'dist_mean':[],'spikes_dend':[],'spikes_soma':[], 'dist_spikes_binned':[],'spikes_dend_binned':[],'spikes_soma_binned':[], 'spikes_dend_diff_binned':[],'dist_bin_i':[]})
                # spike_data_dist_bin[str(syn_num)][str(w_mean)].append([])
                color=p['field_color'][f_i]
                marker='.'
                spikes_soma = trial['spikes']['soma'][f_i]
                spikes_dend = []
                dist=[]
                dw_list=[]
                dist_spikes=[]
                for tree_key, tree in p['sec_idx'].iteritems():
                    for sec_i, sec in enumerate(tree):
                        for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):
                            distance = p['seg_dist'][tree_key][sec][seg]
                            spikes_dend_temp = trial['spikes']['dend'][tree_key][sec_i][seg_i][f_i]
                            dw = trial['spikes']['dw'][tree_key][sec_i][seg_i][f_i]
                            for spike_i, spike in enumerate(spikes_dend_temp):
                                spikes_dend.append(spike)
                                dist_spikes.append(distance)
                            dist.append(distance)
                            dw_list.append(dw)

                dist_mean  = np.mean(dist)
                

                dist_bin_i = [dist_bin_i for dist_bin_i, dist_bin in enumerate(dist_bins) if (dist_mean > dist_bin[0] and dist_mean <= dist_bin[1])][0]
                dw_mean = np.mean(dw_list)

                # divide spikes in into bins 
                spikes_soma_binned =[]
                spikes_dend_binned=[]
                spikes_dist_binned=[]
                spikes_dend_diff_binned=[]
                for time_bin_i, time_bin in enumerate(time_bins):
                    # find spikes within current time bin
                    binned_spikes_soma = [spike for spike_i, spike in enumerate(spikes_soma) if (spike > time_bin[0] and spike <= time_bin[1])]
                    binned_spikes_dend = [spike for spike_i, spike in enumerate(spikes_dend) if (spike > time_bin[0] and spike <= time_bin[1])]
                    binned_spikes_dist = [dist_spikes[spike_i] for spike_i, spike in enumerate(spikes_dend) if (spike > time_bin[0] and spike <= time_bin[1])]

                    # count dendritic spikes
                    spike_frac_dend = len(binned_spikes_dend)/len(dist)
                    spike_time_diff=[]
                    if len(binned_spikes_soma)==0:
                        spike_time_diff = binned_spikes_dend
                    elif len(binned_spikes_dend)>0:
                        spike_time_diff = [binned_spikes_soma[0]-t for t in binned_spikes_dend]
                    spikes_soma_binned.append(binned_spikes_soma) 
                    spikes_dend_binned.append(binned_spikes_dend)
                    spikes_dist_binned.append(binned_spikes_dist)
                    spikes_dend_diff_binned.append(spike_time_diff)

                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dw'].append(dw_list)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dw_mean'].append(dw_mean)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_spikes'].append(dist_spikes)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_spikes_binned'].append(spikes_dist_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist'].append(dist)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_mean'].append(dist_mean)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_bin_i'].append(dist_bin_i)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend'].append(spikes_dend)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_soma_binned'].append(spikes_soma_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend_binned'].append(spikes_dend_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend_diff_binned'].append(spikes_dend_diff_binned)



                # print plots[str(syn_num)][str(w_mean)].number
                plt.figure(plots_spikes[str(syn_num)][str(w_mean)].number)
                plt.plot(dist_mean, len(dist_spikes), color+marker)
                plt.figure(plots_weights[str(syn_num)][str(w_mean)].number)
                plt.plot(dist, dw_list, color+marker)
                plt.figure(plots_weights_mean[str(syn_num)][str(w_mean)].number)
                plt.plot(dist_mean, dw_mean, color+marker)

        # group simulations into distance bins and pre/post spikes
        plots = {}
        for syn_num_key, syn_num in spike_data_group.iteritems():
            plots[syn_num_key]={}
            for w_mean_key, w_mean in syn_num.iteritems():
                plots[syn_num_key][w_mean_key] = plt.figure()
                plot_file_name = 'distance_x_prepost_spikes_'+'syn_num_'+syn_num_key+ '_w_'+w_mean_key
                for f_i, f in enumerate(w_mean):
                    print p['field_color'], f_i
                    color = p['field_color'][f_i]
                    pre_marker = 'x'
                    post_marker = '.'
                    for trial_i, trial in enumerate(f['dist_bin_i']):
                        syn_tot = 4.*float(syn_num_key)
                        dist_bin_i = trial
                        pre = [time_diff for time_bin_i, time_bin in enumerate(f['spikes_dend_diff_binned'][trial_i]) for time_diff in time_bin if time_diff >0 ]
                        post = [time_diff for time_bin_i, time_bin in enumerate(f['spikes_dend_diff_binned'][trial_i]) for time_diff in time_bin if time_diff <0 ]
                        pre_norm = float(len(pre))/syn_tot
                        post_norm = float(len(post))/syn_tot
                        spike_data_dist_bin[syn_num_key][w_mean_key][f_i][dist_bin_i]['pre'].append(pre_norm)
                        spike_data_dist_bin[syn_num_key][w_mean_key][f_i][dist_bin_i]['post'].append(post_norm)

                    for dist_bin_i, dist_bin in enumerate(spike_data_dist_bin[syn_num_key][w_mean_key][f_i]):
                        print dist_bin_i, len(spike_data_dist_bin[syn_num_key][w_mean_key][f_i])
                        distance = np.mean(dist_bins[dist_bin_i])
                        pre_mean = np.mean(dist_bin['pre'])
                        post_mean = np.mean(dist_bin['post'])
                        pre_std = np.std(dist_bin['pre'])
                        post_std = np.std(dist_bin['post'])
                        pre_sem = stats.sem(dist_bin['pre'])
                        post_sem = stats.sem(dist_bin['post'])
                        plt.plot(distance, pre_mean, color+pre_marker)
                        plt.plot(distance, post_mean, color+post_marker)
                        plt.errorbar(distance, pre_mean, yerr=pre_sem, ecolor=color)
                        plt.errorbar(distance, post_mean, yerr=post_sem, ecolor=color)
                        plt.xlabel('distance from soma (um)')
                        plt.ylabel('fraction of synapses that spike')

                plots[syn_num_key][w_mean_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plots[syn_num_key][w_mean_key])

        # save and close figure
        for plot_key1, plot1 in plots_spikes.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_spikes_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)

        for plot_key1, plot1 in plots_weights.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_weights_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)

        for plot_key1, plot1 in plots_weights_mean.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_weights_mean_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)

    def exp_2g(self, **kwargs):
        """ Apply a single theta burst to synapses with varying mean distance from soma 
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
            # data organized as {location}{data_type}[trials][polarity]
            spike_data= []

        # string to save group data
        save_string_group_data_proc = 'spikes_grouped_proc'+'.pkl'
        # if data file already exists
        if save_string_group_data_proc in files:
            # load data
            print 'processed spike data found'
            with open(data_folder+save_string_group_data_proc, 'rb') as pkl_file:
                spike_data= pickle.load(pkl_file)
            print 'processed spike data loaded'
        # otherwise create data structure
        else:
            # data organized as {location}{data_type}[trials][polarity]
            spike_data_group = {}

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

                    spike_data.append({})

                    # add to processed list
                    id_list_spike_times.append(data_file)

                    spike_data[-1]['spikes'] = {'soma':[],'dend':{}, 'dw':{}}

                    for tree_key, tree in p['sec_idx'].iteritems():

                        spike_data[-1]['spikes']['dend'][tree_key]=[]
                        spike_data[-1]['spikes']['dw'][tree_key]=[]
                        
                        for sec_i, sec in enumerate(tree):

                            spike_data[-1]['spikes']['dend'][tree_key].append([])
                            spike_data[-1]['spikes']['dw'][tree_key].append([])

                            for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):

                                spike_data[-1]['spikes']['dend'][tree_key][-1].append([])
                                spike_data[-1]['spikes']['dw'][tree_key][-1].append([])
                                                    # iterate over field polarities
                                for f_i, f in enumerate(p['field']):

                                    # detect spikes in dendrite
                                    spike_data[-1]['spikes']['dend'][tree_key][-1][-1].append(spike_analysis.detect_spikes(data[str(f)][tree_key+'_v'][sec_i][seg_i], threshold=threshold)['times'][0])

                                    # detect spikes in dendrite
                                    spike_data[-1]['spikes']['dw'][tree_key][-1][-1].append(data[str(f)][tree_key+'_gbar'][sec_i][seg_i][-1]/data[str(f)][tree_key+'_gbar'][sec_i][seg_i][0])

                    for f_i, f in enumerate(p['field']):
                                    # detect spikes in soma
                        spike_data[-1]['spikes']['soma'].append(spike_analysis.detect_spikes(data[str(f)]['soma_v'][0][0], threshold=threshold)['times'][0])

                    spike_data[-1]['p']=p

        # save processed file list
        with open(data_folder+id_list_string_spike_times, 'wb') as output:pickle.dump(id_list_spike_times, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'id list saved'
        
        # save structure of all spike data
        save_group_data = spike_data
        with open(data_folder+save_string_group_data_raw, 'wb') as output:
            pickle.dump(save_group_data, output,protocol=pickle.HIGHEST_PROTOCOL)
        print 'spike data saved'

        # spike data organized as {syn_num}{w_mean}[trials]{variable}[polarity]
        spike_data_group ={}
        spike_data_dist_bin = {}
        plots_spikes={} 
        plots_weights ={}
        plots_weights_mean = {}
        time_bins = [[1200, 1600],[1600, 2000],[2000, 2400],[2400, 2800]]
        dist_bins = [[0., 250.],[100., 350.],[200., 450.],[300.,550.],[400., 650.]]
        for trial_i, trial in enumerate(spike_data):
            p = spike_data[trial_i]['p']
            syn_num = p['syn_num']
            w_mean = p['w_mean']
            # print syn_num, w_mean
            if str(syn_num) not in spike_data_group:
                spike_data_group[str(syn_num)]={}
                spike_data_dist_bin[str(syn_num)]={}
                plots_spikes[str(syn_num)]={}
                plots_weights[str(syn_num)]={}
                plots_weights_mean[str(syn_num)]={}
            if str(w_mean) not in spike_data_group[str(syn_num)]:
                spike_data_group[str(syn_num)][str(w_mean)]=[]
                spike_data_dist_bin[str(syn_num)][str(w_mean)]=[]
                for f_i, f in enumerate(p['field']):
                    spike_data_group[str(syn_num)][str(w_mean)].append({'dw':[],'dw_mean':[],'dist':[],'dist_spikes':[],'dist_mean':[],'spikes_dend':[],'spikes_soma':[], 'dist_spikes_binned':[],'spikes_dend_binned':[],'spikes_soma_binned':[], 'spikes_dend_diff_binned':[],'dist_bin_i':[]})
                    spike_data_dist_bin[str(syn_num)][str(w_mean)].append([])
                    for dist_bin_i, dist_bin in enumerate(dist_bins):
                        spike_data_dist_bin[str(syn_num)][str(w_mean)][f_i].append({'pre':[],'post':[]})
                plots_spikes[str(syn_num)][str(w_mean)]=plt.figure()
                plots_weights[str(syn_num)][str(w_mean)]=plt.figure()
                plots_weights_mean[str(syn_num)][str(w_mean)]=plt.figure()
            
            for f_i, f in enumerate(p['field']):
                # spike_data_group[str(syn_num)][str(w_mean)].append({'dw':[],'dw_mean':[],'dist':[],'dist_spikes':[],'dist_mean':[],'spikes_dend':[],'spikes_soma':[], 'dist_spikes_binned':[],'spikes_dend_binned':[],'spikes_soma_binned':[], 'spikes_dend_diff_binned':[],'dist_bin_i':[]})
                # spike_data_dist_bin[str(syn_num)][str(w_mean)].append([])
                color=p['field_color'][f_i]
                marker='.'
                spikes_soma = trial['spikes']['soma'][f_i]
                spikes_dend = []
                dist=[]
                dw_list=[]
                dist_spikes=[]
                for tree_key, tree in p['sec_idx'].iteritems():
                    for sec_i, sec in enumerate(tree):
                        for seg_i, seg in enumerate(p['seg_idx'][tree_key][sec_i]):
                            distance = p['seg_dist'][tree_key][sec][seg]
                            spikes_dend_temp = trial['spikes']['dend'][tree_key][sec_i][seg_i][f_i]
                            dw = trial['spikes']['dw'][tree_key][sec_i][seg_i][f_i]
                            for spike_i, spike in enumerate(spikes_dend_temp):
                                spikes_dend.append(spike)
                                dist_spikes.append(distance)
                            dist.append(distance)
                            dw_list.append(dw)

                dist_mean  = np.mean(dist)
                

                dist_bin_i = [dist_bin_i for dist_bin_i, dist_bin in enumerate(dist_bins) if (dist_mean > dist_bin[0] and dist_mean <= dist_bin[1])][0]
                dw_mean = np.mean(dw_list)

                # divide spikes in into bins 
                spikes_soma_binned =[]
                spikes_dend_binned=[]
                spikes_dist_binned=[]
                spikes_dend_diff_binned=[]
                for time_bin_i, time_bin in enumerate(time_bins):
                    # find spikes within current time bin
                    binned_spikes_soma = [spike for spike_i, spike in enumerate(spikes_soma) if (spike > time_bin[0] and spike <= time_bin[1])]
                    binned_spikes_dend = [spike for spike_i, spike in enumerate(spikes_dend) if (spike > time_bin[0] and spike <= time_bin[1])]
                    binned_spikes_dist = [dist_spikes[spike_i] for spike_i, spike in enumerate(spikes_dend) if (spike > time_bin[0] and spike <= time_bin[1])]

                    # count dendritic spikes
                    spike_frac_dend = len(binned_spikes_dend)/len(dist)
                    spike_time_diff=[]
                    if len(binned_spikes_soma)==0:
                        spike_time_diff = binned_spikes_dend
                    elif len(binned_spikes_dend)>0:
                        spike_time_diff = [binned_spikes_soma[0]-t for t in binned_spikes_dend]
                    spikes_soma_binned.append(binned_spikes_soma) 
                    spikes_dend_binned.append(binned_spikes_dend)
                    spikes_dist_binned.append(binned_spikes_dist)
                    spikes_dend_diff_binned.append(spike_time_diff)

                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dw'].append(dw_list)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dw_mean'].append(dw_mean)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_spikes'].append(dist_spikes)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_spikes_binned'].append(spikes_dist_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist'].append(dist)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_mean'].append(dist_mean)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['dist_bin_i'].append(dist_bin_i)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend'].append(spikes_dend)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_soma_binned'].append(spikes_soma_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend_binned'].append(spikes_dend_binned)
                spike_data_group[str(syn_num)][str(w_mean)][f_i]['spikes_dend_diff_binned'].append(spikes_dend_diff_binned)



                # print plots[str(syn_num)][str(w_mean)].number
                plt.figure(plots_spikes[str(syn_num)][str(w_mean)].number)
                plt.plot(dist_mean, len(dist_spikes), color+marker)
                plt.figure(plots_weights[str(syn_num)][str(w_mean)].number)
                plt.plot(dist, dw_list, color+marker)
                plt.figure(plots_weights_mean[str(syn_num)][str(w_mean)].number)
                plt.plot(dist_mean, dw_mean, color+marker)

        # group simulations into distance bins and pre/post spikes
        plots = {}
        for syn_num_key, syn_num in spike_data_group.iteritems():
            plots[syn_num_key]={}
            for w_mean_key, w_mean in syn_num.iteritems():
                plots[syn_num_key][w_mean_key] = plt.figure()
                plot_file_name = 'distance_x_prepost_spikes_'+'syn_num_'+syn_num_key+ '_w_'+w_mean_key
                for f_i, f in enumerate(w_mean):
                    print p['field_color'], f_i
                    color = p['field_color'][f_i]
                    pre_marker = 'x'
                    post_marker = '.'
                    for trial_i, trial in enumerate(f['dist_bin_i']):
                        syn_tot = 4.*float(syn_num_key)
                        dist_bin_i = trial
                        pre = [time_diff for time_bin_i, time_bin in enumerate(f['spikes_dend_diff_binned'][trial_i]) for time_diff in time_bin if time_diff >0 ]
                        post = [time_diff for time_bin_i, time_bin in enumerate(f['spikes_dend_diff_binned'][trial_i]) for time_diff in time_bin if time_diff <0 ]
                        pre_norm = float(len(pre))/syn_tot
                        post_norm = float(len(post))/syn_tot
                        spike_data_dist_bin[syn_num_key][w_mean_key][f_i][dist_bin_i]['pre'].append(pre_norm)
                        spike_data_dist_bin[syn_num_key][w_mean_key][f_i][dist_bin_i]['post'].append(post_norm)

                    for dist_bin_i, dist_bin in enumerate(spike_data_dist_bin[syn_num_key][w_mean_key][f_i]):
                        print dist_bin_i, len(spike_data_dist_bin[syn_num_key][w_mean_key][f_i])
                        distance = np.mean(dist_bins[dist_bin_i])
                        pre_mean = np.mean(dist_bin['pre'])
                        post_mean = np.mean(dist_bin['post'])
                        pre_std = np.std(dist_bin['pre'])
                        post_std = np.std(dist_bin['post'])
                        pre_sem = stats.sem(dist_bin['pre'])
                        post_sem = stats.sem(dist_bin['post'])
                        plt.plot(distance, pre_mean, color+pre_marker)
                        plt.plot(distance, post_mean, color+post_marker)
                        plt.errorbar(distance, pre_mean, yerr=pre_sem, ecolor=color)
                        plt.errorbar(distance, post_mean, yerr=post_sem, ecolor=color)
                        plt.xlabel('distance from soma (um)')
                        plt.ylabel('fraction of synapses that spike')

                plots[syn_num_key][w_mean_key].savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plots[syn_num_key][w_mean_key])

        # save and close figure
        for plot_key1, plot1 in plots_spikes.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_spikes_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)

        for plot_key1, plot1 in plots_weights.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_weights_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)

        for plot_key1, plot1 in plots_weights_mean.iteritems():
            for plot_key2, plot2 in plot1.iteritems():
                plot_file_name = 'distance_x_weights_mean_'+'syn_num_'+plot_key1+ '_w_'+plot_key2
                plot2.savefig(data_folder+plot_file_name+'.png', dpi=250)
                plt.close(plot2)





if __name__ =="__main__":
    # Weights(param.exp_3().p)
    # # Spikes(param.exp_3().p)
    # kwargs = run_control.Arguments('exp_8').kwargs
    # plots = Voltage()
    # plots.plot_all(param.Experiment(**kwargs).p)
    kwargs = {'experiment':'exp_2g'}
    Experiment(**kwargs)




