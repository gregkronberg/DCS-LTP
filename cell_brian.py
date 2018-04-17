# -*- coding: utf-8 -*-
"""
Simulate 4 compartment neuron with Clopath STDP learning rule using Brian2 simulator

Created on Thu Apr 05 14:29:25 2018

@author: Greg Kronberg
"""
import numpy as np
from brian2 import*
prefs.codegen.target = 'numpy'

def _test_run3():
    # Parameters
    #`````````````````````````````````````````````````````````````````````````
    # compartments
    n=1 # number of neurons
    E_L = -69*mV # mV
    g_L = 40*nS # nS
    delta_T = 2*mV # mV
    C = 281*pF
    t_noise = 20*ms
    t_V_T = 50*ms
    V_Trest = -50*mV
    V_T_max = -30*mV

    # axial conductance between compartments
    g_axial = 1000*nS
    g_axial_soma_basal = 1250*nS
    g_axial_basal_soma = 50*nS
    g_axial_soma_proximal = 1250*nS
    g_axial_proximal_soma = 50*nS
    g_axial_proximal_distal = 1500*nS
    g_axial_distal_proximal = 225*nS
    I_axial1=0
    I_axial2=0

    # synapses
    # ampa
    g_max_ampa = 100*nS
    t_ampa = 2*ms
    E_ampa = 0*mV
    # nmda
    g_max_nmda = 50*nS
    t_nmda = 50*ms
    E_nmda = 0*mV

    # facilation/depression
    f = 5
    t_F = 94*ms
    d1 = 0.45
    t_D1 = 540*ms
    d2 = 0.12
    t_D2 = 45*ms
    d3 = 0.98
    t_D3 = 120E3*ms

    # compartment thresholds
    threshold_soma = '-55*mV'
    threshold_prox = '-45*mV'
    threshold_dist = '-40*mV'
    threshold_basal = '-45*mV'
    reset_soma = '-55*mV'
    reset_prox = '-55*mV'
    reset_dist = '-55*mV'
    reset_basal = '-55*mV'

    # connection parameters
    connect_prob = 0.1

    # input stimulus
    bursts = 1
    pulses = 4
    burst_freq = 5
    pulse_freq = 100
    warmup = 20 

    # field
    field = 0*mV

    method = 'euler'

    w=1


    eqs_compartment = '''
    du/dt = (I_L + I_exp + I_syn)/C : volt

    I_L = -g_L*(u - E_L) : amp

    I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

    dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

    I_syn : amp
    # C : farad
    '''

    eqs_syn = '''
    dg_ampa/dt = -g_ampa/t_ampa : siemens (clock-driven)
    I_ampa = g_ampa*(u_post-E_ampa) : amp
    I_syn_post = I_ampa : amp (summed)

    # facilitation/depression

    # parameters
    # w : 1
    # g_max_ampa : siemens

    '''

    pre_syn = '''
    g_ampa += w*g_max_ampa 
    '''

    # create compartments
    #======================================================================
    soma = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset='u='+reset_soma, method=method)
    soma.u = 0

    # input stimuli
    #======================================================================
    input_times = zeros(pulses*bursts)*ms
    indices = zeros(pulses*bursts)
    cnt=-1
    for burst in range(bursts):
        for pulse in range(pulses):
            cnt+=1
            time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
            input_times[cnt] = time*ms
    
    print input_times
    # input_times = [20*ms, 30*ms, 40*ms, 50*ms]
    input_spikes = SpikeGeneratorGroup(1, indices, input_times )

    input_syn = Synapses(input_spikes, soma, eqs_syn, on_pre=pre_syn)
    input_syn.connect(j='i')
            
    # connect neurons
    #========================================================================
    #FIXME

    # record variables
    #========================================================================
    rec_soma = StateMonitor(soma, ('u'), record=True)
    rec_g = StateMonitor(input_syn, 'g_ampa', record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses)/pulse_freq 
    print run_time
    run(run_time*ms)
        
    # plot
    #=======================================================================
    plot(rec_soma.t/ms, rec_soma.u.T/mV)
    show()

def _test_run2():
    # Parameters
    #`````````````````````````````````````````````````````````````````````````
    # compartments
    n=1 # number of neurons
    E_L = -69*mV # mV
    g_L = 40*nS # nS
    delta_T = 2*mV # mV
    C = 281*pF
    t_noise = 20*ms
    t_V_T = 50*ms
    V_Trest = -50*mV
    V_T_max = -30*mV

    # axial conductance between compartments
    g_axial = 1000*nS
    g_axial_soma_basal = 1250*nS
    g_axial_basal_soma = 50*nS
    g_axial_soma_proximal = 1250*nS
    g_axial_proximal_soma = 50*nS
    g_axial_proximal_distal = 1500*nS
    g_axial_distal_proximal = 225*nS
    I_axial1=0
    I_axial2=0

    # synapses
    # ampa
    g_max_ampa = 100*nS
    t_ampa = 2*ms
    E_ampa = 0*mV
    # nmda
    g_max_nmda = 50*nS
    t_nmda = 50*ms
    E_nmda = 0*mV

    # facilation/depression
    f = 5
    t_F = 94*ms
    d1 = 0.45
    t_D1 = 540*ms
    d2 = 0.12
    t_D2 = 45*ms
    d3 = 0.98
    t_D3 = 120E3*ms

    # compartment thresholds
    threshold_soma = '-55*mV'
    threshold_prox = '-45*mV'
    threshold_dist = '-40*mV'
    threshold_basal = '-45*mV'
    reset_soma = '-55*mV'
    reset_prox = '-55*mV'
    reset_dist = '-55*mV'
    reset_basal = '-55*mV'

    # connection parameters
    connect_prob = 0.1

    # input stimulus
    bursts = 1
    pulses = 4
    burst_freq = 5
    pulse_freq = 100
    warmup = 20 

    # field
    field = 0*mV

    method = 'euler'

    w=1


    eqs_compartment = '''
    du/dt = (I_L + I_exp + I_syn)/C : volt

    I_L = -g_L*(u - E_L) : amp

    I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

    dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

    I_syn : amp
    # C : farad
    '''

    eqs_syn = '''
    dg_nmda/dt = -g_nmda/t_nmda : siemens (clock-driven)
    dg_ampa/dt = -g_ampa/t_ampa : siemens (clock-driven)
    B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 
    I_nmda = g_nmda*B*(u_post-E_nmda) : amp
    I_ampa = g_ampa*(u_post-E_ampa) : amp
    I_syn_post = I_nmda + I_ampa : amp (summed)

    # facilitation/depression
    dF/dt = (1-F)/t_F : 1 (clock-driven)
    dD1/dt = (1-D1)/t_D1 : 1 (clock-driven)
    dD2/dt = (1-D2)/t_D2 : 1 (clock-driven)
    dD3/dt = (1-D3)/t_D3 : 1 (clock-driven)
    A = F*D1*D2*D3 : 1 

    # parameters
    # w : 1
    # g_max_nmda : siemens
    # g_max_ampa : siemens
    # A : 1
    # t_F : second
    # f : 1
    # t_D1 : second
    # d1 : 1
    # t_D2 : second
    # d2 : 1
    # t_D3 : second
    # d3 : 1
    '''

    pre_syn = '''
    g_nmda += w*g_max_nmda*A
    g_ampa += w*g_max_ampa*A
    F += f
    D1 *= d1
    D2 *= d2
    D3 *= d3
    '''

    # create compartments
    #======================================================================
    soma = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset='u='+reset_soma, method=method)
    soma.u = 0

    # input stimuli
    #======================================================================
    input_times = zeros(pulses*bursts)*ms
    indices = zeros(pulses*bursts)
    cnt=-1
    for burst in range(bursts):
        for pulse in range(pulses):
            cnt+=1
            time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
            input_times[cnt] = time*ms
    
    print input_times
    input_spikes = SpikeGeneratorGroup(1, indices, input_times )

    input_syn = Synapses(input_spikes, soma, eqs_syn, on_pre=pre_syn)
    
    input_syn.connect(j='i')

    input_syn.F=1
    input_syn.D1=1
    input_syn.D2=1
    input_syn.D3=1
            
    # connect neurons
    #========================================================================
    #FIXME

    # record variables
    #========================================================================
    rec_soma = StateMonitor(soma, ('u'), record=True)
    rec_g = StateMonitor(input_syn, 'g_ampa', record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses+1)/pulse_freq 
    print run_time
    run(run_time*ms)
        
    # plot
    #=======================================================================
    # plot(rec_soma.t/ms, rec_soma.u.T/mV)
    plot(rec_soma.t/ms, rec_g.g_ampa.T)
    show()

def _run():
    '''
    '''
    cell = FourCompartment()
    #set up to record voltage in each compartment and weights in input compartment
    rec_comps = ['basal', 'soma', 'proximal', 'distal']
    rec_vars = [['u'],['u'],['u'],['u']] 
    # rec_u = cell._record_variables(cell=cell.cell, compartments=rec_comps, variables=rec_vars)
    rec_u = StateMonitor(cell.cell['soma'], ('u'), record=True)
    # rec_w = StateMonitor(cell.syn_input, ('w_clopath'), record=True)
    cell._run(cell.p)
    print rec_u.u
    figure()
    plot(rec_u.t/ms, rec_u.u.T/mV)

    show()

    return cell

class FourCompartment():
    '''
    '''
    def __init__(self, n_neuron=1):
        '''
        '''
        self.p = self._load_parameters(n_neuron=n_neuron)
        self.eqs = self._set_equations()
        self.cell = self._create_compartments(
            p=self.p, 
            eqs_compartment=self.eqs['compartment'],
            eqs_reset=self.eqs['reset'],
            method=self.p['method'])
        
        self.connect = self._connect_compartments(
            p=self.p,
            cell=self.cell,
            eqs_connect1=self.eqs['connect1'],
            eqs_connect2=self.eqs['connect2'],
            )

        self.input_spikes = self._make_input_spikes(
            bursts=1, 
            pulses=4, 
            pulse_freq=100, 
            burst_freq=5, 
            warmup=10, 
            n_inputs=1, 
            index=0)

        self.syn_input = self._connect_inputs(
            input_spikes=self.input_spikes, 
            post_group=self.cell['proximal'],
            post_indices=range(len(self.cell['proximal'])),
            eqs_syn=self.eqs['syn']+self.eqs['syn_clopath'],
            on_pre=self.eqs['pre_syn']+self.eqs['pre_syn_clopath'])

        self._set_initial_conditions(
            cell=self.cell,
            syn=self.syn_input,
            p=self.p )

    def _load_parameters(self, **kwargs):
        '''
        '''
        p = {
        'n_compartment':4,
        'n_neuron':2,
        # compartments
        'E_L' : -69*mV, # mV
        'g_L' : 40*nS, # nS
        'delta_T' : 2*mV, # mV
        'C' : 281*pF,
        't_noise' : 20*ms,
        't_V_T' : 50*ms,
        'refractory_time' : 1*ms,
        'spike_hold_time':0.9*ms, # must be less than refractory time
        'reset' : -55*mV,
        'spike_peak':-20*mV,

        # FIXME
        # axial conductance between compartments
        'g_axial_soma_basal' : 1250*nS,
        'g_axial_basal_soma' : 50*nS,
        'g_axial_soma_proximal' : 1250*nS,
        'g_axial_proximal_soma' : 50*nS,
        'g_axial_proximal_distal' : 1500*nS,
        'g_axial_distal_proximal' : 225*nS,
        # I_axial1=0
        # I_axial2=0

        # synapses
        # ampa
        'g_max_ampa' : 3*100*nS,
        't_ampa' : 2*ms,
        'E_ampa' : 0*mV,
        # nmda
        'g_max_nmda' : 100*nS,
        't_nmda' : 50*ms,
        'E_nmda' : 0*mV,

        # facilation/depression
        'f' : 5,
        't_F' : 94*ms,
        'd1' : 0.45,
        't_D1' : 540*ms,
        'd2' : 0.12,
        't_D2' : 45*ms,
        'd3' : 0.98,
        't_D3' : 120E3*ms,

        # compartment thresholds
        'threshold_soma' : '-50*mV',
        'threshold_proximal' : '-30*mV',
        'threshold_distal' : '-30*mV',
        'threshold_basal' : '-30*mV',
        'reset_soma' : '-55*mV',
        'reset_prox' : '-55*mV',
        'reset_dist' : '-55*mV',
        'reset_basal' : '-55*mV',
        'V_Trest_soma' : -55*mV,
        'V_Trest_proximal' : -35*mV,
        'V_Trest_distal' : -35*mV,
        'V_Trest_basal' : -35*mV,
        'V_T_max_soma' : -30*mV,
        'V_T_max_proximal' : -20*mV,
        'V_T_max_distal' : -20*mV,
        'V_T_max_basal' : -20*mV,
        'V_hold_soma' : 20*mV,
        'V_hold_proximal' : -20*mV,
        'V_hold_distal' : -20*mV,
        'V_hold_basal' : -20*mV,

        # clopath rule parameters
        'v_target' : -55*mV*mV,


        # connection parameters
        'connect_prob' : 0.1,

        # input stimulus
        'bursts' : 1,
        'pulses' : 4,
        'burst_freq' : 5,
        'pulse_freq' : 100,
        'warmup' : 20, 

        # field
        # field = 0*mV

        'method' : 'euler',

        't_reset' : 0.1*ms,

        'w':1,

        }

        # update parameter dictionary from keyword arguments
        for (key, val) in kwargs.items():
            p[key]=val

        return p

    def _set_equations(self):
        '''
        '''

        eqs = {}
        eqs['compartment'] = Equations('''
        # voltage dynamics including holding spike potential
            du/dt = int(not_refractory)*(I_L + I_exp + I_axial + I_syn)/C  + (t-lastspike>spike_hold_time)*(1-int(not_refractory))*(reset-u)/t_reset : volt 
        
        # leak current
            I_L = -g_L*(u - E_L) : amp

        # exponential current for adaptive exponential spiking
            I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

        # adaptation variable
            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

        # FIXME
        # sum axial current
            I_axial = I_axial1 + I_axial2 : amp

        # parameters
            I_axial1 : amp
            I_axial2 : amp
            I_syn : amp
            I_ext : amp
            V_Trest : volt
        ''')

        eqs['connect1'] = Equations('''
        # connect compartments
        
        # total current entering compartment from all connected compartments
            I_axial1_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 

        # parameters
            field : volt # axial current due to electric field
            g_axial_in : siemens
            g_axial_out : siemens
        ''')

        # FIXME
        eqs['connect2'] = Equations('''
        # connect compartments
        
        # total current entering compartment from all connected compartments
            I_axial2_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 

        # parameters
            field : volt # axial current due to electric field
            g_axial_in : siemens
            g_axial_out : siemens
        ''')

        eqs['syn'] = Equations('''
        # ampa conductance
            dg_ampa/dt = -g_ampa/t_ampa : siemens (clock-driven)
        
        # nmda conductance
            dg_nmda/dt = -g_nmda/t_nmda : siemens (clock-driven)
        
        # nmda magnesium gate
            B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 

        # synaptic current
            I_nmda = -g_nmda*B*(u_post-E_nmda) : amp
            I_ampa = -g_ampa*(u_post-E_ampa) : amp
            I_syn_post = I_nmda + I_ampa : amp (summed)

        # facilitation/depression
            dF/dt = (1-F)/t_F : 1 (clock-driven)
            dD1/dt = (1-D1)/t_D1 : 1 (clock-driven)
            dD2/dt = (1-D2)/t_D2 : 1 (clock-driven)
            dD3/dt = (1-D3)/t_D3 : 1 (clock-driven)
            A = F*D1*D2*D3 : 1

        # weight
            w : 1
        ''')

        eqs['pre_syn'] = '''
        g_nmda += w*g_max_nmda*A
        g_ampa += w*g_max_ampa*A 
        F += f
        D1 *= d1
        D2 *= d2
        D3 *= d3
        '''

        # equations executed at every timestep
        eqs['syn_clopath'] =   Equations('''
        # # low threshold filtered membrane potential
        #     du_lowpass1/dt = (u_post-u_lowpass1)/tau_lowpass1 : volt     

        # # high threshold filtered membrane potential
        #     du_lowpass2/dt = (u_post-u_lowpass2)/tau_lowpass2 : volt     

        # # homeostatic term
        #     du_homeo/dt = (u_post-V_rest-u_homeo)/tau_homeo : volt       
        
        # # lowpass presynaptic variable
        #     dx_trace/dt = -x_trace/taux :1                          

        # # clopath rule for potentiation (depression implented with on_pre)
        #     dw_clopath/dt = A_LTP*x_trace_post*(u_lowpass2/mV - Theta_low/mV)*int(u_lowpass2/mV - Theta_low/mV > 0)  : 1

        # # homeostatic depression amplitude
        #     A_LTD_u = A_LTD*(u_homeo**2/v_target) : 1                   

            ''')

        # equations executed only when a presynaptic spike occurs
        eqs['pre_syn_clopath'] = '''
            # # depression term
            #     w_minus = A_LTD_u*(u_lowpass1/mV - Theta_low/mV)*int(u_lowpass1/mV - Theta_low/mV > 0)  

            # # update overall weight
            #     w_clopath = clip(w_clopath-w_minus, 0, w_max)

            # # update presynaptic trace
            #     x_trace += 1
            ''' 

        eqs['reset'] = '''
        u = spike_peak
        '''

        return eqs

    def _create_compartments(self, p, eqs_compartment, eqs_reset, method):
        '''
        '''
        # create compartments
        #===================================================================
        n_compartment = p['n_compartment']
        n_neuron = p['n_neuron']
        n_total = n_compartment*n_neuron

        cell={}
        cell['soma'] = NeuronGroup(n_neuron, eqs_compartment, threshold='u>'+p['threshold_soma'], reset=eqs_reset, refractory=p['refractory_time'],method=method, namespace=p)
        cell['basal'] = NeuronGroup(n_neuron, eqs_compartment, threshold='u>'+p['threshold_basal'], reset=eqs_reset, refractory=p['refractory_time'], method=method, namespace=p)
        cell['proximal'] = NeuronGroup(n_neuron, eqs_compartment, threshold='u>'+p['threshold_proximal'], reset=eqs_reset, refractory=p['refractory_time'],method=method, namespace=p)
        cell['distal'] = NeuronGroup(n_neuron, eqs_compartment, threshold='u>'+p['threshold_distal'], reset=eqs_reset, refractory=p['refractory_time'], method=method, namespace=p)
        # cell['neurons'] = []
        # for i in range(n_neuron):
        #     cell['neurons'].append([ cell['soma'][i], cell['basal'][i], cell['proximal'][i], cell['distal'][i] ])

        return cell

    def _connect_compartments(self, p, cell, eqs_connect1, eqs_connect2):
        '''
        '''
        # connect compartments
        #======================================================================
        n_neuron = p['n_neuron']
        connect={}
        connect['soma_basal'] = Synapses(cell['soma'], cell['basal'], eqs_connect1)
        connect['basal_soma'] = Synapses(cell['basal'], cell['soma'], eqs_connect1)
        connect['proximal_soma'] = Synapses(cell['proximal'], cell['soma'], eqs_connect2)
        connect['soma_proximal'] = Synapses(cell['soma'], cell['proximal'], eqs_connect1)
        connect['distal_proximal'] = Synapses(cell['distal'], cell['proximal'], eqs_connect2)
        connect['proximal_distal'] = Synapses(cell['proximal'], cell['distal'], eqs_connect1)

        for (key, val) in connect.items():
            for i in range(p['n_neuron']):
                val.connect(condition='i==j')

        # set axial conductances
        #====================================================================
        # iterate over compartment connections
        for (key, val) in connect.items():
            # pre compartment connection
            pre = key[0:key.index('_')]
            # post compartment connection
            post = key[key.index('_')+1:]
            # list all possible axial conductances
            conductances = [a for a in p if 'g_axial' in a]
            # iterate over conductances
            for conductance in conductances:
                # find conductances that match the compartments in the current connection
                if pre in conductance and post in conductance:
                    # if the order is the same (ie pre before post)
                    if conductance.index(pre) < conductance.index(post):
                        # set the inward conductance
                        val.g_axial_in = p[conductance]
                    # if the order is backwards (post before pre)
                    elif conductance.index(pre) > conductance.index(post):
                        # set the outward conductance
                        val.g_axial_out = p[conductance]


        return connect

    def _make_input_spikes(self, bursts=1, pulses=4, pulse_freq=100, burst_freq=5, warmup=10, n_inputs=1, index=0):
        # input stimuli
        #======================================================================
        input_times = np.zeros(pulses*bursts)*ms
        indices = index*np.ones(pulses*bursts)
        cnt=-1
        for burst in range(bursts):
            for pulse in range(pulses):
                cnt+=1
                time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
                input_times[cnt] = time*ms
                
        input_spikes = SpikeGeneratorGroup(n_inputs, indices, input_times )
        return input_spikes

    def _connect_inputs(self, input_spikes, post_group, eqs_syn, on_pre, p_connect=1, post_indices=[]):
        '''
        '''

        syn_input = Synapses(input_spikes, post_group, eqs_syn, on_pre=on_pre)
        # if post_indices:
        #     syn_input.connect('j=post_indices')
        # else:
        #     syn_input.connect(p=p_connect)
        # syn_input.connect(p=p_connect)
        syn_input.connect(j='i')

        return syn_input

    def _set_initial_conditions(self, cell, syn, p):
        '''
        '''
        # iterate through compartments and update initial conditions
        for (comp_key, comp) in cell.items():
            print comp
            comp.u = p['E_L']
            comp.V_T = p['V_Trest_'+comp_key]
            comp.V_Trest = p['V_Trest_'+comp_key]

        # facilitation depression parameters
        syn.F=1
        syn.D1=1
        syn.D2=1
        syn.D3=1

        syn.w = 1


    def _record_variables(self, cell, compartments, variables, indices=[]):
        '''
        '''
        rec={}
        for i, compartment in enumerate(compartments):
            if indices:
                rec[compartment] = StateMonitor(cell[compartment], variables[i], record=indices[i])

            else:
                rec[compartment] = StateMonitor(cell[compartment], variables[i], record=True)

        return rec


        # self.rec_soma = StateMonitor(soma, ('u'), record=True)
        # self.rec_proximal = StateMonitor(proximal, ('u'), record=True)
        # self.rec_distal = StateMonitor(distal, ('I_axial1','u'), record=True)
        # self.rec_basal = StateMonitor(basal, ('u'), record=True)

    def _run(self, p):
        run_time = p['warmup'] + 1000*(p['bursts']-1)/p['burst_freq'] + 1000*(p['pulses']+1)/p['pulse_freq'] 
        print run_time
        run(run_time*ms)

    def _plots(self):
        pass

def _test_run():
    '''
    '''

    # Parameters
    #========================================================================
    n=1 # number of neurons
    E_L = -69*mV # mV
    g_L = 40*nS # nS
    delta_T = 2*mV # mV
    C = 281*pF
    t_noise = 20*ms
    t_V_T = 50*ms
    refractory_time = 1.2*ms
    spike_hold_time=1*ms # must be at least 0.2 ms less than refractory time
    reset = -55*mV
    # time constant for resetting voltage after holding spike (should be equal to dt)
    t_reset = .1*ms

    # axial conductance between compartments (g_axial_fromcompartment_tocompartment)
    #````````````````````````````````````````
    g_axial_soma_basal = 1250*nS
    g_axial_basal_soma = 50*nS
    g_axial_soma_proximal = 1250*nS
    g_axial_proximal_soma = 50*nS
    g_axial_proximal_distal = 1500*nS
    g_axial_distal_proximal = 225*nS

    # synapses
    #````````````````````````````````````````````````````````````````````````
    # ampa
    #`````````````````````````
    g_max_ampa = 3*100*nS
    t_ampa = 2*ms
    E_ampa = 0*mV
    # nmda
    #`````````````````````````
    g_max_nmda = 100*nS
    t_nmda = 50*ms
    E_nmda = 0*mV

    # facilation/depression
    #``````````````````````````
    f = 5
    t_F = 94*ms
    d1 = 0.45
    t_D1 = 540*ms
    d2 = 0.12
    t_D2 = 45*ms
    d3 = 0.98
    t_D3 = 120E3*ms

    # clopath plasticity parameters
    #```````````````````````````````````````````````````````````````````
    # reference value for homeostatic process mV
    v_target     = 100*mV*mV         
    # amplitude for the depression
    A_LTD      = 2E-5    
    # amplitude for the potentiation 
    A_LTP        = 38E-6/ms   
    # time constant for voltage trace in the potentiation term 
    tau_lowpass2      = 5*ms  
    # time constant for presynaptic trace        
    tau_x      = 10*ms   
    # time constant for voltage trace in the depression term    
    tau_lowpass1      = 6*ms   
    # time constant for homeostatic process      
    tau_homeo = 1000*ms
    # low threshold
    theta_low      = -60*mV  
    # high threshold in the potentiation term    
    theta_high    = -53*mV       
    # maximum weight
    w_max=10

    # compartment-specific thresholds, resets, adaptation
    #```````````````````````````````````````````````````````````````````````
    threshold_soma = '-50*mV'
    threshold_prox = '-30*mV'
    threshold_dist = '-35*mV'
    threshold_basal = '-30*mV'
    reset_soma = '-55*mV'
    reset_prox = '-55*mV'
    reset_dist = '-55*mV'
    reset_basal = '-55*mV'
    V_Trest_soma = -55*mV
    V_Trest_proximal = -35*mV
    V_Trest_distal = -35*mV
    V_Trest_basal = -35*mV
    V_Tmax_soma = -30*mV
    V_Tmax_proximal = -20*mV
    V_Tmax_distal = -20*mV
    V_Tmax_basal = -20*mV
    V_hold_soma = 20*mV
    V_hold_proximal = 20*mV
    V_hold_distal = 20*mV
    V_hold_basal = 20*mV

    # connection parameters
    #``````````````````````````````
    connect_prob = 0.1

    # input stimulus
    #``````````````````````````````
    bursts = 1
    pulses = 4
    burst_freq = 5
    pulse_freq = 100
    warmup = 20 

    method = 'euler'

    w=1

    # equations
    #=======================================================================
    # set voltage to holding potential after threshold crossing
    #``````````````````````````````````````````````````````````
    eqs_reset = '''
    u = u_hold
    V_T = V_Tmax
    '''

    # voltage dynamics for each comparment
    #````````````````````````````````````````````````````````````````````
    eqs_compartment = '''
    du/dt = int(not_refractory)*(I_L + I_exp + I_axial + I_syn)/C  + ((t-lastspike)>spike_hold_time)*(1-int(not_refractory))*(reset-u)/t_reset : volt 

    I_L = -g_L*(u - E_L) : amp

    I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

    dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

    # FIXME
    I_axial = I_axial1 + I_axial2 : amp
    I_axial1 : amp
    I_axial2 :amp
    I_syn : amp
    I_ext : amp
    V_Trest : volt
    V_Tmax : volt
    u_hold : volt
    # C : farad
    '''

    # connection between compartments (axial conductance)
    #`````````````````````````````````````````````````````````````````````
    eqs_connect1 = '''
    
    field : volt # axial current due to electric field
    g_axial_in : siemens
    g_axial_out : siemens
    I_axial1_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 
    '''

    # connection between compartments (axial conductance), if compartment has a second connection
    #`````````````````````````````````````````````````````````````````````
    eqs_connect2 = '''
    
    field : volt # axial current due to electric field
    g_axial_in : siemens
    g_axial_out : siemens
    I_axial2_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 

    '''

    # synapse equations
    #`````````````````````````````````````````````````````````````````````
    eqs_syn = '''
    # ampa and nmda
    #``````````````````````````````````````````````````````````
    dg_nmda/dt = -g_nmda/t_nmda : siemens (clock-driven)
    dg_ampa/dt = -g_ampa/t_ampa : siemens (clock-driven)
    B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 
    I_nmda = -g_nmda*B*(u_post-E_nmda) : amp
    I_ampa = -g_ampa*(u_post-E_ampa) : amp
    I_syn_post = I_nmda + I_ampa : amp (summed)

    # facilitation/depression
    #````````````````````````````````````````````````````````
    dF/dt = (1-F)/t_F : 1 (clock-driven)
    dD1/dt = (1-D1)/t_D1 : 1 (clock-driven)
    dD2/dt = (1-D2)/t_D2 : 1 (clock-driven)
    dD3/dt = (1-D3)/t_D3 : 1 (clock-driven)
    A = F*D1*D2*D3 : 1

    # parameters
    #`````````````````````````````````````````````````````````
    # w : 1
    # g_max_nmda : siemens
    # g_max_ampa : siemens
    # t_F : second
    # f : 1
    # t_D1 : second
    # d1 : 1
    # t_D2 : second
    # d2 : 1
    # t_D3 : second
    # d3 : 1

    # clopath learning rule
    #`````````````````````````````````````````````````````````
    # low threshold filtered membrane potential
    du_lowpass1/dt = (u_post-u_lowpass1)/tau_lowpass1 : volt (clock-driven)   

    # high threshold filtered membrane potential
        du_lowpass2/dt = (u_post-u_lowpass2)/tau_lowpass2 : volt     

    # homeostatic term
        du_homeo/dt = (u_post-E_L-u_homeo)/tau_homeo : volt       
    
    # lowpass presynaptic variable
        dx_trace/dt = -x_trace/tau_x : 1                          

    # clopath rule for potentiation (depression implented with on_pre)
        dw_clopath/dt = A_LTP*x_trace*(u_lowpass2/mV - theta_low/mV)*int(u_lowpass2/mV - theta_low/mV > 0)*(u_post/mV-theta_high/mV)*int(u_post/mV-theta_high/mV >0)  : 1

    # homeostatic depression amplitude
        A_LTD_u = A_LTD*(u_homeo**2/v_target) : 1   
    '''

    # to be executed on each presynaptic spike
    #``````````````````````````````````````````````````````````````````````
    pre_syn = '''
    g_nmda += w*g_max_nmda*A  # nmda 
    g_ampa += w*g_max_ampa*A  # ampa
    F += f                    # facilitation/depression
    D1 *= d1
    D2 *= d2
    D3 *= d3 

    w_minus = A_LTD_u*(u_lowpass1/mV - theta_low/mV)*int(u_lowpass1/mV - theta_low/mV > 0)   # update LTD on eahc input spike

    w_clopath = clip(w_clopath-w_minus, 0, w_max)  # apply LTD

    x_trace += 1   # update presynaptic trace with each input
    '''

    # create compartments
    #======================================================================
    soma = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset=eqs_reset, refractory=refractory_time,method=method)
    proximal = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_prox, reset=eqs_reset, refractory=refractory_time, method=method)
    distal = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_dist, reset=eqs_reset, refractory=refractory_time, method=method)
    basal = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_basal, reset=eqs_reset, refractory=refractory_time, method=method)

    # update initial conditions and parameters for each compartment
    #======================================================================
    soma.u = E_L
    proximal.u = E_L
    distal.u = E_L
    basal.u = E_L
    soma.V_T= V_Trest_soma
    proximal.V_T = V_Trest_proximal
    distal.V_T = V_Trest_distal
    basal.V_T = V_Trest_basal
    soma.V_Trest= V_Trest_soma
    proximal.V_Trest = V_Trest_proximal
    distal.V_Trest = V_Trest_distal
    basal.V_Trest = V_Trest_basal
    soma.V_Tmax= V_Tmax_soma
    proximal.V_Tmax = V_Tmax_proximal
    distal.V_Tmax = V_Tmax_distal
    basal.V_Tmax = V_Tmax_basal
    soma.u_hold= V_hold_soma
    proximal.u_hold = V_hold_proximal
    distal.u_hold = V_hold_distal
    basal.u_hold = V_hold_basal

    # connect compartments
    #======================================================================
    connect_soma_basal = Synapses(soma, basal, eqs_connect1)
    connect_soma_proximal = Synapses(soma, proximal, eqs_connect2)
    connect_proximal_distal = Synapses(proximal, distal, eqs_connect1)
    connect_basal_soma = Synapses(basal, soma, eqs_connect1)
    connect_proximal_soma = Synapses(proximal, soma, eqs_connect2)
    connect_distal_proximal = Synapses(distal, proximal, eqs_connect1)
    for i in range(n):
        connect_soma_basal.connect(i=i, j=i)
        connect_soma_proximal.connect(i=i, j=i)
        connect_proximal_distal.connect(i=i, j=i)
        connect_basal_soma.connect(i=i, j=i)
        connect_proximal_soma.connect(i=i, j=i)
        connect_distal_proximal.connect(i=i, j=i)

    # update axial conductances
    #=======================================================================
    connect_soma_basal.g_axial_in = g_axial_soma_basal
    connect_soma_basal.g_axial_out = g_axial_basal_soma
    connect_soma_proximal.g_axial_in = g_axial_soma_proximal
    connect_soma_proximal.g_axial_out = g_axial_proximal_soma
    connect_proximal_distal.g_axial_in = g_axial_proximal_distal
    connect_proximal_distal.g_axial_out = g_axial_distal_proximal
    connect_basal_soma.g_axial_in = g_axial_basal_soma
    connect_basal_soma.g_axial_out = g_axial_soma_basal
    connect_proximal_soma.g_axial_in = g_axial_proximal_soma
    connect_proximal_soma.g_axial_out = g_axial_soma_proximal
    connect_distal_proximal.g_axial_in = g_axial_distal_proximal
    connect_distal_proximal.g_axial_out = g_axial_proximal_distal
        
    # generate input stimuli
    #======================================================================
    input_times = np.zeros(pulses*bursts)*ms
    indices = np.zeros(pulses*bursts)
    cnt=-1
    for burst in range(bursts):
        for pulse in range(pulses):
            cnt+=1
            time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
            input_times[cnt] = time*ms
            
    input_spikes = SpikeGeneratorGroup(1, indices, input_times )

    input_syn = Synapses(input_spikes, distal, eqs_syn, on_pre=pre_syn)

    input_syn.connect(j='i')

    # set FD variables to 1
    #`````````````````````````````
    input_syn.F=1
    input_syn.D1=1
    input_syn.D2=1
    input_syn.D3=1

    # set initial clopath weights
    #```````````````````````````````````
    input_syn.w_clopath = 0.5
            
    # connect neurons
    #========================================================================
    #FIXME

    # record variables
    #========================================================================
    # FIXME
    rec_soma = StateMonitor(soma, ('u'), record=True)
    rec_proximal = StateMonitor(proximal, ('u'), record=True)
    rec_distal = StateMonitor(distal, ('V_T','u'), record=True)
    rec_basal = StateMonitor(basal, ('u'), record=True)
    rec_w = StateMonitor(input_syn, ('w_clopath'), record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses+1)/pulse_freq 
    run(run_time*ms)
        
    # plot
    #=======================================================================
    figure()
    plot(rec_distal.t/ms, rec_distal.V_T.T/mV)
    figure()
    plot(rec_distal.t/ms, rec_distal.u.T/mV)
    plot(rec_distal.t/ms, rec_proximal.u.T/mV)
    figure()
    plot(rec_distal.t/ms, rec_soma.u.T/mV)
    figure()
    plot(rec_w.t/ms, rec_w.w_clopath.T)
    
    show()

if __name__ == '__main__':
    _test_run()


    