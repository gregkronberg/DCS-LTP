"""
create cells and activate subsets of synapses
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import stims
import param
import run_control


class PyramidalCylinder:
    """ 4 compartment pyramidal cell with HH dynamics
    """
    def __init__(self, p):
        self.geometry(p)
        self.mechanisms(p)

    def geometry(self, p):
        """
        areas determined from cell geo5038804 from Migliore 2005
        basal: 19966.3598 um2
        apical: 23700.5664916 (tuft) + 11461.4440485 (trunk) = 35162.0105401 um2
        soma: 176.290723263 um2
        axon: 305.021056197 um2
        """
        # list sections
        trees = ['basal', 'soma', 'apical_prox', 'apical_dist']
        areas = [19966.36, 176.29, 35162.01/2., 35162.01/2.]
        
        # store geometry
        self.geo = {}
        # store synapses
        self.syns = {}
        # create sections
        for tree_i, tree in enumerate(trees):
            self.geo[tree] = []
            self.syns[tree] = [] 

            for sec_i in range(p['nsec_'+tree]):
                self.geo[tree].append( h.Section( name=tree))

                # diameter basd on area of full morphology
                diam1 = p['diam1_'+tree]
                diam2 = p['diam2_'+tree] 

                if tree=='soma':    
                    # create 3d specification, with cell arranged vertically
                    h.pt3dadd(0, 0, 0, diam1, sec=self.geo[tree][sec_i])
                    h.pt3dadd(0, p['L_'+tree], 0, diam2, sec=self.geo[tree][sec_i])

                if tree=='basal':
                    h.pt3dadd(0, 0, 0, diam1, sec=self.geo[tree][sec_i])
                    h.pt3dadd(0, -p['L_'+tree], 0, diam2, sec=self.geo[tree][sec_i])

                if tree=='apical_prox':
                    h.pt3dadd(0, p['L_soma'], 0, diam1, sec=self.geo[tree][sec_i])
                    h.pt3dadd(0, p['L_soma']+p['L_'+tree], 0, diam2, sec=self.geo[tree][sec_i])

                if tree=='apical_dist':
                    h.pt3dadd(0, p['L_soma']+p['L_apical_prox'], 0, diam1, sec=self.geo[tree][sec_i])
                    h.pt3dadd(0, p['L_soma']+p['L_apical_prox']+p['L_'+tree], 0, diam2, sec=self.geo[tree][sec_i])

                # add list to store synapses for each section
                self.syns[tree].append([])

                # insert passive mechanism
                self.geo[tree][sec_i].insert('pas')
                # passive conductance (S/cm2)
                self.geo[tree][sec_i].g_pas = 1./p['RmAll']         
                # leak reversal potential (mV)  
                self.geo[tree][sec_i].e_pas = p['Vrest']                
                # specific capacitance (uf/cm2)
                self.geo[tree][sec_i].cm = p['Cm']          
                # axial resistance (ohm cm)         
                self.geo[tree][sec_i].Ra = 1.*p['RaAll'] 

                self.geo[tree][sec_i].L = p['L_'+tree]
                # self.geo[tree][sec_i].diam = p['diam_'+tree]

                self.geo[tree][sec_i].nseg=p['nseg']

        self.geo['basal'][0].connect(self.geo['soma'][0](0),0)
        self.geo['apical_prox'][0].connect(self.geo['soma'][0](1),0)
        self.geo['apical_dist'][0].connect(self.geo['apical_prox'][0](1),0)

        if p['fixnseg']==True:
            h.xopen('fixnseg.hoc')

        # set temperature in hoc
        h.celsius = p['celsius']
        # set soma as origin for distance measurements
        h.distance(sec=self.geo['soma'][0])

    def rotate(self, theta):
        """Rotate the cell about the Z axis.
        """
        for sec in h.allsec():
            for i in range(int(h.n3d(sec=sec))):
                x = h.x3d(i, sec=sec)
                y = h.y3d(i, sec=sec)
                c = np.cos(theta)
                s = np.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                h.pt3dchange(i, xprime, yprime, h.z3d(i, sec=sec), h.diam3d(i, sec=sec), sec=sec)

    def mechanisms(self, p):

        for tree_key, tree in self.geo.iteritems():
            for sec_i, sec in enumerate(tree): 

                if tree_key == 'soma':
                    # voltage gated sodium
                    sec.insert('na3')
                    sec.gbar_na3 = p['gna']*p['AXONM']
                    # h-current         
                    sec.insert('hd')
                    sec.ghdbar_hd = p['ghd']                
                    sec.vhalfl_hd = p['vhalfl_hd_prox']
                    sec.kl_hd = p['kl_hd']
                    sec.ehd_hd =  p['ehd']

                    # delayed rectifier potassium       
                    sec.insert('kdr')
                    sec.gkdrbar_kdr = p['gkdr'] 

                    # a-type potassium      
                    sec.insert('kap')
                    sec.gkabar_kap = p['KMULTP']
                    sec.vhalfl_kap = p['vhalfl_kap']
                    sec.vhalfn_kap = p['vhalfn_kap']

                    # L-type calcium channel
                    sec.insert('calH')
                    sec.gcalbar_calH = p['gcalbar']

                    # sodium reversal potential 
                    sec.ena = p['ena']      
                    # potassium reversal potential 
                    sec.ek = p['ek']

                    for seg_i,seg in enumerate(sec):
                        self.syns[tree_key][sec_i].append({})
                
                elif ((tree_key == 'basal') or 
                (tree_key == 'apical_prox') or 
                (tree_key == 'apical_dist')):

                    # h-current
                    sec.insert('hd')
                    sec.ghdbar_hd = p['ghd']
                    sec.kl_hd = p['kl_hd']
                    sec.ehd_hd =  p['ehd']
                    
                    # voltage gated sodium      
                    sec.insert('na3')
                    sec.gbar_na3 = p['gna']

                    # delayed rectifier potassium   
                    sec.insert('kdr')
                    sec.gkdrbar_kdr = p['gkdr'] 
                    # a-type potassium proximal
                    sec.insert('kap')
                    sec.gkabar_kap = 0  
                    # a-type potassium distal       
                    sec.insert('kad')
                    sec.gkabar_kad = 0  

                    # L-type calcium channel
                    sec.insert('calH')
                    sec.gcalbar_calH = p['gcalbar']

                    # sodium reversal potential 
                    sec.ena = p['ena']
                    # potassium reversal potential
                    sec.ek = p['ek']        

                    # mechanisms that vary with distance from soma
                    # loop over segments
                    for seg_i,seg in enumerate(sec):

                        # add segment dimension to syns structure
                        self.syns[tree_key][sec_i].append([])

                        # distance from soma
                        seg_dist = h.distance(seg.x, sec=sec)
                        
                        # sodium
                        seg.gbar_na3 = p['gna'] + p['dgna']*seg_dist
                        # print seg_dist, seg.gbar_na3
                        
                        # h current
                        seg.ghdbar_hd = p['ghd']*(1+p['ghd_grad']*(seg_dist/100.)/(p['L_apical_prox']/200.))

                        # h current
                        if seg_dist < p['ghd_cutoff_distance']*(p['L_apical_prox']/200.):
                            seg.ghdbar_hd = p['ghd']*(1+p['ghd_grad']*(seg_dist/100.)/(p['L_apical_prox']/200.))
                        else:
                            seg.ghdbar_hd = p['ghd']*(1+p['ghd_grad']*(p['ghd_cutoff_distance']/100.)/(p['L_apical_prox']/200.))
                        
                        # A-type potassium
                        if seg_dist > 100.*(p['L_apical_prox']/200.): # distal
                            seg.vhalfl_hd = p['vhalfl_hd_dist']
                            seg.vhalfl_kad = p['vhalfl_kad']
                            seg.vhalfn_kad = p['vhalfn_kad']
                            seg.gkabar_kad = p['KMULT']*(1+p['ka_grad']*(seg_dist/100.)/(p['L_apical_prox']/200.))
                            if seg_dist < p['ka_cutoff_distance']*(p['L_apical_prox']/200.):
                                seg.gkabar_kad = p['KMULT']*(1+p['ka_grad']*(seg_dist/100.)/(p['L_apical_prox']/200.))
                            else:
                                seg.gkabar_kad = p['KMULT']*(1+p['ka_grad']*(p['ka_cutoff_distance']/100.)/(p['L_apical_prox']/200.))
                        else:   # proximal
                            seg.vhalfl_hd = p['vhalfl_hd_prox']
                            seg.vhalfl_kap = p['vhalfl_kap']
                            seg.vhalfn_kap = p['vhalfn_kap']
                            seg.gkabar_kap = p['KMULTP']*(1+p['ka_grad']*(seg_dist/100.)/(p['L_apical_prox']/200.))

                        self.syns[tree_key][sec_i][seg_i] = {
                        'ampa':[],
                        'nmda':[],
                        'clopath':[],
                        }

                        for syn_key,syn in self.syns[tree_key][sec_i][seg_i].iteritems():
                            if syn_key is 'ampa':
                                # Regular ampa synapse
                                # self.syns[tree_key][sec_i][seg_i][syn_key] = h.Exp2Syn(sec(seg.x))
                                # self.syns[tree_key][sec_i][seg_i][syn_key].tau1 = p['tau1_ampa']
                                # self.syns[tree_key][sec_i][seg_i][syn_key].tau2 = p['tau2_ampa']
                                # self.syns[tree_key][sec_i][seg_i][syn_key].i = p['i_ampa']

                                # FD adapting exponential synapse based on model in Varela et al. 1997
                                self.syns[tree_key][sec_i][seg_i][syn_key] = h.FDSExp2Syn_D3(sec(seg.x))
                                self.syns[tree_key][sec_i][seg_i][syn_key].f = p['f_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_F = p['tau_F_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d1 = p['d1_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D1 = p['tau_D1_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d2 = p['d2_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D2 = p['tau_D2_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d3 = p['d3_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D3 = p['tau_D3_ampa']

                            elif syn_key is 'nmda':
                                # print syn_key
                                self.syns[tree_key][sec_i][seg_i][syn_key]= h.Exp2SynNMDA(sec(seg.x))
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau1 = p['tau1_nmda']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau2 = p['tau2_nmda']
                                # print syn

                            elif syn_key is 'clopath':
                                # print syn_key
                                self.syns[tree_key][sec_i][seg_i][syn_key] = h.STDPSynCCNon(sec(seg.x))

class CellMigliore2005:
    """ pyramidal neuron based on Migliore et al. 2005

    An instance of this object will creates a cell (hoc objects) at the top level of the hoc interpreter using the hoc files in _init_geometry.  The .geo attribute contains a python mapping to these hoc objects.  The geo object is organized as geo['section tree'][section](segment location)

    the syns attribute creates a container for synapse objects that are added to each segment in the hoc cell.  syns is organized as syns['section tree']['synapse type'][section][segment number]
    
    """
    def __init__(self,p):
        '''
        '''
        pass

    def geometry(self,p):
        '''create cell geometry at hoc top level
        ===Args===
        -p : parameter dictionary
        
        ===Out===
        -geo : geometry structure containing hoc section objects
                -geo{tree}[section][segment]
        -syns : structure containing synapse objects
                -syns{tree}[section][segment]{synapse type}
                -synapse types include ampa, nmda, clopath
        
        ===Updates===
        -hoc objects are created based on the geometry in the loaded hoc file
        
        ===Comments===
        '''
        print 'loading cell geometry:', self.__class__.__name__
        # load cell geometry into hoc interpreter
        h.load_file('geo5038804.hoc')  
        # h.load_file('geoc62564.hoc')
        # set discretization based on dlambda rule (set dlambda in hoc file) 
        h.load_file('fixnseg.hoc')      
        # dictionary for storing geometry ['tree'][sec](seg location)
        self.geo = {}
        # dictionary for storing synapse objects ['tree']['type'][sec][seg]
        self.syns = {}
        # add section trees to geometry dictionary
        self.geo['soma'] = h.soma
        self.geo['axon'] =  h.axon
        self.geo['basal'] = h.dendrite
        self.geo['apical_trunk'] = h.user5
        self.geo['apical_tuft'] = h.apical_dendrite
        
        # set temperature in hoc
        h.celsius = p['celsius']
        # set soma as origin for distance measurements
        h.distance(sec = self.geo['soma'][0])

        return self.geo, self.syns

    def mechanisms(self,p):
        """ insert membrane mechanisms into cell geometry
        
        ==Args==
        -p : parameter dictionary

        ==Out==
        -geo : geometry structure containing hoc section objects
                -geo{tree}[section][segment].mechanism
        -syns : structure of containing synapse mechanisms
                -syns{tree}[section][segment]{synapse type}
        ==Updates==
        -range mechanisms and their parameters are updated according to the parameters in p
        ==Comments==
        self.syns is updated to store an object for each synapse.  It is organized as ['tree']['synapse type'][section][segment].  Note that the last index will depend on how the cell is discretized as the number segments changes in each sections 

        the parameters for each membrane mechanism  are store in a dictionary called p.  See the param module for details.
        """
        print 'loading cell range mechanisms'
        
        # loop over trees
        for tree_key,tree in self.geo.iteritems():
            
            # list to store synapse mechanisms
            self.syns[tree_key] = []

            # loop over sections in tree
            for sec_i,sec in enumerate(tree):
                
                # add dimension for each section
                self.syns[tree_key].append([])

                # common passive biophysics for all sections
                sec.insert('pas')
                # passive conductance (S/cm2)
                sec.g_pas = 1/p['RmAll']            
                # leak reversal potential (mV)  
                sec.e_pas = p['Vrest']              
                # specific capacitance (uf/cm2)
                sec.cm = p['Cm']            
                # axial resistance (ohm cm)         
                sec.Ra = p['RaAll'] 
                                        
                # axon active bipophysics
                if tree_key == 'axon':
                    # voltage gated sodium
                    sec.insert('nax')                       
                    sec.gbar_nax = p['gna']*p['AXONM']
                    # print 'axon sodium conductance:', sec.gbar_nax*10000
                    # delayed rectifier potassium
                    sec.insert('kdr')                       
                    sec.gkdrbar_kdr = p['gkdr']
                    # a-type potassium
                    sec.insert('kap')                       
                    sec.gkabar_kap = p['KMULTP']
                    sec.vhalfl_kap = p['vhalfl_kap']
                    sec.vhalfn_kap = p['vhalfn_kap']
                    # sodium reversal potential 
                    sec.ena = p['ena']      
                    # potassium reversal potential 
                    sec.ek = p['ek']
                    sec.Ra = p['RaAx']


                    for seg_i, seg in enumerate(sec):
                        self.syns[tree_key][sec_i].append({})
                    
                # soma active biophysics
                elif tree_key == 'soma':

                    # voltage gated sodium
                    sec.insert('na3')
                    sec.gbar_na3 = p['gna']*p['SOMAM']
                    sec.ar_na3 = p['gna_inact']
                    # print 'soma sodium conductance:', sec.gbar_na3*10000
                    # h-current         
                    sec.insert('hd')
                    sec.ghdbar_hd = p['ghd']                
                    sec.vhalfl_hd = p['vhalfl_hd_prox']
                    sec.kl_hd = p['kl_hd']
                    sec.ehd_hd = p['ehd']       


                    # delayed rectifier potassium       
                    sec.insert('kdr')
                    sec.gkdrbar_kdr = p['gkdr'] 
                    # a-type potassium      
                    sec.insert('kap')
                    sec.gkabar_kap = p['KMULTP']
                    sec.vhalfl_kap = p['vhalfl_kap']
                    sec.vhalfn_kap = p['vhalfn_kap']

                    sec.insert('calH')
                    sec.gcalbar_calH = p['gcalbar']
                    # sodium reversal potential 
                    sec.ena = p['ena']      
                    # potassium reversal potential 
                    sec.ek = p['ek']    

                    for seg_i,seg in enumerate(sec):
                        self.syns[tree_key][sec_i].append({})

                    
                # dendrites active biophysics
                elif ((tree_key == 'basal') or 
                (tree_key == 'apical_trunk') or 
                (tree_key == 'apical_tuft')):
                    # h-current
                    sec.insert('hd')
                    sec.ghdbar_hd = p['ghd']
                    sec.kl_hd = p['kl_hd']
                    sec.ehd_hd = p['ehd']
                    
                    # voltage gated sodium      
                    sec.insert('na3')
                    sec.gbar_na3 = p['gna']
                    sec.ar_na3 = p['gna_inact']

                    # delayed rectifier potassium   
                    sec.insert('kdr')
                    sec.gkdrbar_kdr = p['gkdr'] 
                    # a-type potassium proximal
                    sec.insert('kap')
                    sec.gkabar_kap = 0  
                    # a-type potassium distal       
                    sec.insert('kad')
                    sec.gkabar_kad = 0  

                    # L-type calcium channel
                    sec.insert('calH')
                    sec.gcalbar_calH = p['gcalbar']

                    # sodium reversal potential 
                    sec.ena = p['ena']
                    # potassium reversal potential
                    sec.ek = p['ek']        

                    # mechanisms that vary with distance from soma
                    # loop over segments
                    for seg_i,seg in enumerate(sec):
                        
                        # print seg_i
                        self.syns[tree_key][sec_i].append({'ampa':[],
                        'nmda':[],
                        'clopath':[]})

                        for syn_key,syn in self.syns[tree_key][sec_i][seg_i].iteritems():
                            
                            if syn_key is 'ampa':
                                
                                # adapting exponential synapse based on model in Varela et al. 1997
                                self.syns[tree_key][sec_i][seg_i][syn_key] = h.FDSExp2Syn_D3(sec(seg.x))
                                self.syns[tree_key][sec_i][seg_i][syn_key].f = p['f_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_F = p['tau_F_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d1 = p['d1_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D1 = p['tau_D1_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d2 = p['d2_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D2 = p['tau_D2_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].d3 = p['d3_ampa']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau_D3 = p['tau_D3_ampa']

                                # regular double exponential synapse
                                # self.syns[tree_key][sec_i][seg_i][syn_key] = h.Exp2Syn(sec(seg.x))
                                # self.syns[tree_key][sec_i][seg_i][syn_key].tau1 = p['tau1_ampa']
                                # self.syns[tree_key][sec_i][seg_i][syn_key].tau2 = p['tau2_ampa']
                                # self.syns[tree_key][sec_i][seg_i][syn_key].i = p['i_ampa']
                                # print syn

                            elif syn_key is 'nmda':
                                # print syn_key
                                self.syns[tree_key][sec_i][seg_i][syn_key]= h.Exp2SynNMDA(sec(seg.x))
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau1 = p['tau1_nmda']
                                self.syns[tree_key][sec_i][seg_i][syn_key].tau2 = p['tau2_nmda']
                                # print syn

                            elif syn_key is 'clopath':
                                # print syn_key
                                self.syns[tree_key][sec_i][seg_i][syn_key] = h.STDPSynCCNon(sec(seg.x))

                        # distance from soma
                        seg_dist = h.distance(seg.x,sec=sec)
                        
                        # sodium
                        if abs(p['dgna']*seg_dist)<p['gna']:
                            seg.gbar_na3 = p['gna'] + p['dgna']*seg_dist
                        else:
                            seg.gbar_na3 = 0.
                        
                        # h current
                        if seg_dist < p['ghd_cutoff_distance']:
                            seg.ghdbar_hd = p['ghd']*(1+p['ghd_grad']*seg_dist/100.)
                        else:
                            seg.ghdbar_hd = p['ghd']*(1+p['ghd_grad']*p['ghd_cutoff_distance']/100.)

                        
                        # A-type potassium
                        if seg_dist > 100.: # distal
                            seg.vhalfl_hd = p['vhalfl_hd_dist']
                            seg.vhalfl_kad = p['vhalfl_kad']
                            seg.vhalfn_kad = p['vhalfn_kad']
                            if seg_dist < p['ka_cutoff_distance']:
                                seg.gkabar_kad = p['KMULT']*(1+p['ka_grad']*seg_dist/100.)
                            else:
                                seg.gkabar_kad = p['KMULT']*(1+p['ka_grad']*p['ka_cutoff_distance']/100.)
                        else:   # proximal
                            seg.vhalfl_hd = p['vhalfl_hd_prox']
                            seg.vhalfl_kap = p['vhalfl_kap']
                            seg.vhalfn_kap = p['vhalfn_kap']
                            seg.gkabar_kap = p['KMULTP']*(1+p['ka_grad']*seg_dist/100.)

    def _activate_synapses(self, p, stim, syns):
        '''
        ==Args==
        -p : parameter dictionary
                -must contain p['syn_idx'], a zipped list of tuples

        -stim  : nested list of stim objects
                -[segment][burst], with the first index matching entries in p['syn_idx']

        -syns  : structure containing synapse objects
                -syns{tree}[section][segment]{synapse type}
                -synapse types include ampa, nmda, clopath
        ==Out==
        -nc    : structure containing hoc NetCon objects
                -nc[segment]{synapse type}[burst number]
                -segment index matches p['syn_idx']
        ==Updates==
        -NetCon objects are created

        ==Comments==
        '''

        # for storing NetCon objects nc[segment]{synapse type}[burst number]
        self.nc = []

        # iterate over segments to be activated
        for seg_i, seg in enumerate(p['syn_idx']):

            # add dimension for segments
            self.nc.append({})

            # get segment location info
            tree, sec_num, seg_num = seg

            # iterate over synapse types
            for syntype_key,syntype in syns[tree][sec_num][seg_num].iteritems():

                # create list for storing NetCon objects, one for each burst in stim
                self.nc[seg_i][syntype_key]=[]

                # iterate over bursts in stim
                for burst_i, burst in enumerate(stim[seg_i]):

                    # create netcon object
                    netcon = h.NetCon(burst, syntype, 0, 0, p['w_idx'][seg_i])

                    # add netcon object to list
                    self.nc[seg_i][syntype_key].append(netcon)

        return self.nc

    def set_branch_nseg(self, geo, sec_idx, seg_L):
        """ set number of segments for branch that was selected to activate synapses
        
        arguments:

        """

        # iterate over trees in section list
        for tree_key, tree in sec_idx.iteritems():

            for sec_i, sec in enumerate(tree):

                section = geo[tree_key][sec]

                # get section length
                sec_L = section.L

                # determine number of segments
                n_seg = int(np.ceil(sec_L/seg_L))

                # # check that number of segments is odd
                if n_seg % 2 != 0:
                    n_seg+=1

                # # set number of segments
                section.nseg = n_seg
    
    def measure_area(self, tree):
        """
        given a tree measure the total area of all sections in the tree

        tree is a list of sections (hoc objects)
        """
        # FIXME
        area_all = []
        for sec_i, sec in enumerate(tree):
            # convert to um to cm (*.0001)
            L = .0001*sec.L
            a = .0001*sec.diam/2.
            rL = sec.Ra
            rm = 1/sec.g_pas
            area = 2*np.pi*a*L
            lam = np.sqrt(a*rm/(2*rL))
            area_all.append(area)

        return sum(area_all)

class CellKim2015:
    """
    """
    def __init__(self):
        pass

class Syn_act:
    """Activate a specific subset of synpases with NetCon objects
    
    arguments: 
    syns = nested dictionary containing synapse objects organized as ['section tree']['synapse type'][section number][segment number]

    p =  dictionary of parameters including sec_idx and seg_idx lists, including section and segment numbers to be activated.  p also contains 'w_list', which sets the weight for each of the activated synapses. w_list has the same dimensions as seg_idx

    stim = list of NetStim objects to be connected (via NetCon) to the synapses in syns that designated by sec_idx and seg_idx. The stim list will be iterated through and a distinct NetCon object will be created for each NetStim for each activated synapse.  The need for multiple NetStim objects arises from complicated stimulation patterns, like theta bursts, which are not easily programmed with a single NetStim

    The created NetCon objects are referenced by the nc object, which is organized the same way as syns, namely ['section tree']['synapse type'][section number][segment number][NetStim object]
    """
    def __init__(self, syns, p, stim):
        # store netcon objects ['tree']['syn type'][section][segment][list of netstim objects]
        self.nc = {}
        
        # iterate over dendritic subtrees
        for tree_key,tree in syns.iteritems():

            if tree_key in p['seg_idx']:

                self.nc[tree_key] =[]
                # loop over active sections
                for sec_i,sec in enumerate(p['sec_idx'][tree_key]):

                    # add dimension to NetCon structure
                    self.nc[tree_key].append([])
                    
                    # loop over active segments
                    for seg_i,seg in enumerate(p['seg_idx'][tree_key][sec_i]):
                        
                        # add segment dimension to NetCon structure
                        self.nc[tree_key][sec_i].append({})

                        # iterate over synapse types (e.g. ampa, nmda, clopath)
                        for syntype_key,syntype in syns[tree_key][sec][seg].iteritems():
                            
                            # add dimension for synapse type to NetCon structure
                            self.nc[tree_key][sec_i][seg_i][syntype_key]=[]
                            
                            # loop over stimulation bursts
                            for syn_stim_i,syn_stim in enumerate(stim[tree_key][sec_i][seg_i]):

                                # store NetCon
                                # print p['w_list'][tree_key][sec_i][seg_i]
                                self.nc[tree_key][sec_i][seg_i][syntype_key].append(h.NetCon(syn_stim, syns[tree_key][sec][seg][syntype_key], 0, 0, p['w_list'][tree_key][sec_i][seg_i]))

class DendriteTransform:
    def __init__(self, p):
        cell1 = CellMigliore2005(p)
        apical_transform = self.dendrite_transform(geo=cell1.geo, python_tree=['apical_trunk','apical_tuft'], neuron_tree=['user5', 'apical_dendrite'])
        basal_transform = self.dendrite_transform(geo=cell1.geo, python_tree=['basal'], neuron_tree=['dendrite'])
        print 'apical:', apical_transform['a_cable'], apical_transform['L_cable']
        print 'basal:', basal_transform['a_cable'], basal_transform['L_cable']

    def measure_area(self, tree):
        """
        given a tree measure the total area of all sections in the tree

        tree is a list of sections (hoc objects)
        """
        area_all = []
        for sec_i, sec in enumerate(tree):
            # convert to um to cm (*.0001)
            L = .0001*sec.L
            a = .0001*sec.diam/2.
            rL = sec.Ra
            rm = 1/sec.g_pas
            area = 2*np.pi*a*L
            lam = np.sqrt(a*rm/(2*rL))
            area_all.append(area)

        return sum(area_all)

    def measure_length(self, geo):
        """ measure electrotonic length for each path along a cells dendritic tree
        """
        # keep track of most recent section in each path [paths]
        secs = [geo['soma'][0]]

        # keep track of all sections in paths [paths][sections]
        # does not include soma
        paths=[[]]
        
        # iterate over paths (most recent section)
        for sec_i, sec in enumerate(secs):
            
            # current section
            current_sec_ref = h.SectionRef(sec=sec)
            # current children
            current_children = current_sec_ref.child
            
            # if there are children
            while len(current_children)>0:
                
                # iterate over current children 
                for child_i, child in enumerate(current_children):

                    # add first child to current path
                    if child_i==0:
                        paths[sec_i].append(child)
                        # update current section
                        sec = child
                    
                    # if multiple children
                    if child_i > 0:
                        # create new path to copy previous path when tree splits
                        new_path=[]
                        for section in paths[sec_i]:
                            new_path.append(section)

                        # copy current path in list
                        # if split occurs at soma, do not copy previous list
                        if h.SectionRef(sec=child).parent.name()!='soma':
                            paths.append(new_path)
                        else:
                            # create new list, not including soma
                            paths.append([])
                        
                        # add corresponding child to the current section list
                        secs.append(child)
                        
                        # add corresponding child to new path 
                        paths[sec_i + child_i].append(child)
                
                # update current section and children       
                current_sec_ref = h.SectionRef(sec=sec)
                current_children = current_sec_ref.child


        # calculate electrotonic length of each path
        path_l = [] # [paths][electrotonic section length]
        sec_name = [] # [paths][section names]
        for path_i, path in enumerate(paths):
            path_l.append([])
            sec_name.append([])
            for sec_i, sec in enumerate(path):
                # convert all distances in cm
                # section length
                L = .0001*sec.L
                # section radius
                a = .0001*sec.diam/2
                # membrane resistivity
                rm = 1/sec.g_pas
                # axial resistivity
                rL = sec.Ra
                # space constant lambda
                lam = np.sqrt( (a*rm) / (2*rL) )
                # electrotonic length
                e_length = L/lam
                # electrotonic lengths for all paths and sections [paths][sections]
                path_l[path_i].append(e_length) 
                # keep track of section names [paths][sections]
                sec_name[path_i].append(sec.name())
        # print path_l[0]
        return {'path_l': path_l,
        'sec_name':sec_name}

    def dendrite_transform(self, geo, python_tree, neuron_tree):
        """ equivalent cable transform for dendritic tree
        """
        # FIXME
        rL_cable = 150.
        rm_cable = 28000.
        # area
        A_full=0
        for tree_i, tree in enumerate(python_tree):
            A_full += self.measure_area(geo[tree])

        paths = self.measure_length(geo)
        e_lengths = []
        # iterate over all paths
        for path_i, path in enumerate(paths['path_l']):
            # only keep paths in the neuron_tree argument
            for tree in neuron_tree:
                for sec in paths['sec_name'][path_i]:
                    if tree in sec:
                        e_lengths.append(sum(path))
                        break

                        

        EL_full = np.mean(e_lengths) 

        # convert cm back to um (*10000)
        L_cable = (EL_full**(2./3)) * (A_full*rm_cable/(4.*np.pi*rL_cable))**(1./3)
        a_cable = A_full/(2.*np.pi*L_cable)

        return {'a_cable':10000*a_cable, 'L_cable':10000*L_cable}
# set procedure if called as a script

if __name__ == "__main__":
    args = run_control.Arguments('exp_1').kwargs
    p = param.Experiment(**args).p
    DendriteTransform(p)
