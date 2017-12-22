"""
docstring
"""
# imports
from neuron import h
from mpi4py import MPI
import multiprocessing 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cell 
import itertools as it
import stims
import pickle
import param
import os
import copy

# load standard runtime settings.  this is critical.
h.load_file("stdrun.hoc")

# run control
class Run():
    """
    Arguments list:
    p - dictionary of parameters

    each experiment appends a list to the appropriate key in the data dictionary
    data is organized as data['type of data'][experiments][sections][time series data vector]
    details of each experiment are tracked via the data['detail'][experiment number], e.g. data['field'][2]
    """
    def __init__(self, p, cell):

        # create cell
        # self.cell1 = cell.CellMigliore2005(p)
        # self.cell1 = cell.PyramidalCylinder(p) 
        # self.cell1 = p['cell'] # cell must be deleted from p before saving
        self.cell1 = cell
        self.update_clopath( p, syns=self.cell1.syns)
        self.activate_synapses(p)
        self.recording_vectors(p)
        self.run_sims(p)

    # update clopath parameters
    def update_clopath(self, p, syns):
        """
        """
        # iterate over parameters
        for parameter_key, parameter in p.iteritems():
            # if it is a clopath learning rule parameter
            if 'clopath_' in parameter_key:
                # get parameter name
                p_clopath = parameter_key[len('clopath_'):]

                for tree_key, tree in syns.iteritems():
                    # iterate over sections
                    for sec_i,sec in enumerate(tree):
                        # iterate over segments
                        for seg_i,seg in enumerate(sec):
                            # if segment contains a clopath synapse
                            if 'clopath' in list(seg.keys()): 
                                # set synapse parameter value
                                setattr(seg['clopath'], p_clopath, p['clopath_'+p_clopath])

    # activate synapses
    def activate_synapses(self, p):
        """
        """
        # bipolar = stims.Bipolar()
        # bipolar.tbs(bursts=p['bursts'], warmup=p['warmup'], pulses=p['pulses'], pulse_freq=p['pulse_freq'])
        # self.stim = bipolar.stim
        uncage = stims.Uncage()
        uncage.branch_sequence(seg_idx=p['seg_idx'], 
            delays=p['sequence_delays'], 
            bursts=p['bursts'], pulses=p['pulses'], 
            pulse_freq=p['pulse_freq'],
            burst_freq=p['burst_freq'],
            warmup=p['warmup'],
            noise=p['noise'] )
        self.stim = uncage.stim
        self.nc = cell.Syn_act(p=p, syns=self.cell1.syns, stim=self.stim)

    def shape_plot(self, p):
        # highlight active sections
        self.shapeplot = h.PlotShape()
        
        # create section list of active sections
        self.sl = h.SectionList()    # secetion list of included sections
        for sec_i,sec in enumerate(p['sec_idx']):
            self.sl.append(sec=self.cell1.geo[p['trees']][sec])
            self.shapeplot.color(2, sec=self.cell1.geo[p['trees']][sec])

    def recording_vectors(self, p):
        # set up recording vectors
        self.rec =  {}
        
        seg_idx = copy.copy(p['seg_idx'])
        sec_idx = copy.copy(p['sec_idx'])
        sec_idx['soma']=[0]
        sec_idx['axon']=[0]
        seg_idx['soma']= [[0]]
        seg_idx['axon']=[[0]]
        # loop over trees
        for tree_key, tree in seg_idx.iteritems():
            
            # iterate over variables to record
            for var_key, var_dic in p['record_variables'].iteritems():

                # FIXME 
                    # do not create recording vector if section does not have variable
                # create entry for each variable
                self.rec[tree_key+'_'+var_key] = []
            
                # loop over sections
                for sec_i,sec in enumerate(tree):
                    
                    # section number
                    sec_num = sec_idx[tree_key][sec_i]
                    
                    # add list for each section
                    self.rec[tree_key+'_'+var_key].append([])
                    
                    # loop over segments
                    for seg_i,seg in enumerate(sec):
                        
                        # determine relative segment location in (0-1) 
                        seg_loc = float(seg+1)/(self.cell1.geo[tree_key][sec_num].nseg+1)

                        # if variable occurs in a synapse object
                        if 'syn' in var_dic:

                            # check if synapse type exists in this segment
                            if var_dic['syn'] in self.cell1.syns[tree_key][sec_num][seg].keys():

                                # if the desired variable exists in the corresponding synapse
                                if var_key in dir(self.cell1.syns[tree_key][sec_num][seg][var_dic['syn']]): 
                                    
                                    # point to variable to record
                                    var_rec = getattr(self.cell1.syns[tree_key][sec_num][seg][var_dic['syn']], '_ref_'+var_key)

                                    # create recording vector
                                    self.rec[tree_key+'_'+var_key][sec_i].append(h.Vector())

                                    # record variable
                                    self.rec[tree_key+'_'+var_key][sec_i][seg_i].record(var_rec)

                                # append empty vector to hold place and check if data variable exists
                                else: 
                                    self.rec[tree_key+'_'+var_key][sec_i].append([])
                            else:
                                self.rec[tree_key+'_'+var_key][sec_i].append([])




                        # if variable is a range variable
                        if 'range' in var_dic:
                            
                            # if variable belongs to a range mechanism that exists in this section
                            if var_dic['range'] in dir(self.cell1.geo[tree_key][sec_num](seg_loc)):
                                
                                # point to variable for recording
                                var_rec = getattr(self.cell1.geo[tree_key][sec_num](seg_loc), '_ref_'+var_key)
                                
                                # create recording vector
                                self.rec[tree_key+'_'+var_key][sec_i].append(h.Vector())
                                
                                # record variable
                                self.rec[tree_key+'_'+var_key][sec_i][seg_i].record(var_rec)
                            else:
                                # create recording vector
                                self.rec[tree_key+'_'+var_key][sec_i].append([])


        # create time vector
        self.rec['t'] = h.Vector()
        
        # record time
        self.rec['t'].record(h._ref_t)

    def run_sims(self,p):
        # data organized as ['tree']['polarity'][section][segment]
        
        # loop over dcs fields
        self.data={'p':p}
        for f_i,f in enumerate(p['field']):

            # add dimension for each DCS polarity
            self.data[str(f)] = {}
            
            # insert extracellular field
            dcs = stims.DCS(cell=0, field_angle=p['field_angle'], intensity=f, field_on=p['field_on'], field_off=p['field_off'])

            # run time
            h.dt = p['dt']
            h.tstop = p['tstop']
            h.celsius= p['celsius']

            # h.finitialize()
            h.run()

            print 'run complete'


            # store recording vectors as arrays
            # loop over trees
            for tree_key,tree in self.rec.iteritems():
                
                # add list for each field polarity
                self.data[str(f)][tree_key]=[]

                if tree_key != 't':

                    # loop over sections
                    for sec_i,sec in enumerate(self.rec[tree_key]):

                        self.data[str(f)][tree_key].append([])
                        
                        # loop over segments
                        for seg_i,seg in enumerate(sec):

                            self.data[str(f)][tree_key][sec_i].append(np.array(self.rec[tree_key][sec_i][seg_i]))

            self.data[str(f)]['t'] = np.array(self.rec['t'])

        # clear synapses for repeated simulations
        self.nc=[]

def save_data(data, file_name): # save data
    p = data['p']
    # delete cell hoc object (can't be pickled)
    if p['cell']:
        p['cell']=[]
    # check if folder exists with experiment name
    
    if os.path.isdir(p['data_folder']) is False:
        os.mkdir(p['data_folder'])

    with open(p['data_folder']+'data_'+
        file_name+
        '.pkl', 'wb') as output:

        pickle.dump(data, output,protocol=pickle.HIGHEST_PROTOCOL)

# procedures to be initialized if called as a script
if __name__ =="__main__":
    plot_sections(None,None)

