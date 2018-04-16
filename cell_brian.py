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
    g_axial_somas_basals = 1250*nS
    g_axial_basals_somas = 50*nS
    g_axial_somas_proximals = 1250*nS
    g_axial_proximals_somas = 50*nS
    g_axial_proximals_distals = 1500*nS
    g_axial_distals_proximals = 225*nS
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
    somas = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset='u='+reset_soma, method=method)
    somas.u = 0

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

    input_syn = Synapses(input_spikes, somas, eqs_syn, on_pre=pre_syn)
    input_syn.connect(j='i')
            
    # connect neurons
    #========================================================================
    #FIXME

    # record variables
    #========================================================================
    rec_somas = StateMonitor(somas, ('u'), record=True)
    rec_g = StateMonitor(input_syn, 'g_ampa', record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses)/pulse_freq 
    print run_time
    run(run_time*ms)
        
    # plot
    #=======================================================================
    plot(rec_somas.t/ms, rec_somas.u.T/mV)
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
    g_axial_somas_basals = 1250*nS
    g_axial_basals_somas = 50*nS
    g_axial_somas_proximals = 1250*nS
    g_axial_proximals_somas = 50*nS
    g_axial_proximals_distals = 1500*nS
    g_axial_distals_proximals = 225*nS
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
    somas = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset='u='+reset_soma, method=method)
    somas.u = 0

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

    input_syn = Synapses(input_spikes, somas, eqs_syn, on_pre=pre_syn)
    
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
    rec_somas = StateMonitor(somas, ('u'), record=True)
    rec_g = StateMonitor(input_syn, 'g_ampa', record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses+1)/pulse_freq 
    print run_time
    run(run_time*ms)
        
    # plot
    #=======================================================================
    # plot(rec_somas.t/ms, rec_somas.u.T/mV)
    plot(rec_somas.t/ms, rec_g.g_ampa.T)
    show()



class BonoReduced():
    '''
    '''
    def __init__(self):
        '''
        '''

    def _load_parameters(self):
        '''
        '''
        self.p = {
        'n_compartment':4,
        'n_neuron':2,
        # compartments
        'E_L' : -69*mV # mV
        'g_L' : 40*nS # nS
        'delta_T' : 2*mV # mV
        'C' : 281*pF
        't_noise' : 20*ms
        't_V_T' : 50*ms
        'refractory_time' : 1*ms
        'spike_hold_time':0.9*ms # must be less than refractory time
        'reset' : -55*mV

        # FIXME
        # axial conductance between compartments
        'g_axial_somas_basals' : 1250*nS
        'g_axial_basals_somas' : 50*nS
        'g_axial_somas_proximals' : 1250*nS
        'g_axial_proximals_somas' : 50*nS
        'g_axial_proximals_distals' : 1500*nS
        'g_axial_distals_proximals' : 225*nS
        # I_axial1=0
        # I_axial2=0

        # synapses
        # ampa
        'g_max_ampa' : 3*100*nS
        't_ampa' : 2*ms
        'E_ampa' : 0*mV
        # nmda
        'g_max_nmda' : 100*nS
        't_nmda' : 50*ms
        'E_nmda' : 0*mV

        # facilation/depression
        'f' : 5
        't_F' : 94*ms
        'd1' : 0.45
        't_D1' : 540*ms
        'd2' : 0.12
        't_D2' : 45*ms
        'd3' : 0.98
        't_D3' : 120E3*ms

        # compartment thresholds
        'threshold_soma' : '-50*mV'
        'threshold_prox' : '-30*mV'
        'threshold_dist' : '-30*mV'
        'threshold_basal' : '-30*mV'
        'reset_soma' : '-55*mV'
        'reset_prox' : '-55*mV'
        'reset_dist' : '-55*mV'
        'reset_basal' : '-55*mV'
        'V_Trest_soma' : -55*mV
        'V_Trest_proximal' : -35*mV
        'V_Trest_distal' : -35*mV
        'V_Trest_basal' : -35*mV
        'V_T_max_soma' : -30*mV
        'V_T_max_proximal' : -20*mV
        'V_T_max_distal' : -20*mV
        'V_T_max_basal' : -20*mV
        'V_hold_soma' : 20*mV
        'V_hold_proximal' : -20*mV
        'V_hold_distal' : -20*mV
        'V_hold_basal' : -20*mV

        # clopath rule parameters


        # connection parameters
        'connect_prob' : 0.1

        # input stimulus
        'bursts' : 1
        'pulses' : 4
        'burst_freq' : 5
        'pulse_freq' : 100
        'warmup' : 20 

        # field
        # field = 0*mV

        'method' : 'euler'

        't_reset' : 0.1*ms

        }

    def _set_equations(self):
        '''
        '''

        self.eqs_compartment = Equations('''
        # voltage dynamics including holding spike potential
            du/dt = int(not_refractory)*(I_L + I_exp + I_axial + I_syn)/C  + (t-lastspike>spike_hold_time)*(1-int(not_refractory))*(reset-u)/t_reset : volt 
        
        # leak current
            I_L = -g_L*(u - E_L) : amp

        # exponential current for adaptive exponential spiking
            I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

        # adaptation variable
            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

        # parameters
            I_axial : amp
            I_syn : amp
            I_ext : amp
            V_Trest : volt
        ''')

        self.eqs_connect = Equations('''
        # connect compartments
        
        # total current entering compartment from all connected compartments
            I_axial_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 

        # parameters
            field : volt # axial current due to electric field
            g_axial_in : siemens
            g_axial_out : siemens
        '''

        self.eqs_syn = '''
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
        ''')

        self.pre_syn = Equations('''
        g_nmda += w*g_max_nmda*A
        g_ampa += w*g_max_ampa*A
        F += f
        D1 *= d1
        D2 *= d2
        D3 *= d3
        ''')

        # equations executed at every timestep
        self.eqs_syn_clopath =   Equations('''
        # low threshold filtered membrane potential
            du_lowpass1/dt = (u_post-u_lowpass1)/tau_lowpass1 : volt     

        # high threshold filtered membrane potential
            du_lowpass2/dt = (u_post-u_lowpass2)/tau_lowpass2 : volt     

        # homeostatic term
            du_homeo/dt = (u_post-V_rest-u_homeo)/tau_homeo : volt       
        
        # lowpass presynaptic variable
            dx_trace/dt = -x_trace/taux :1                          

        # clopath rule for potentiation (depression implented with on_pre)
            dw_clopath/dt = A_LTP*x_trace_post*(u_lowpass2/mV - Theta_low/mV)*int(u_lowpass2/mV - Theta_low/mV > 0)  

        # homeostatic depression amplitude
            A_LTD_u = A_LTD*(u_homeo**2/v_target)                            

            ''')

        # equations executed only when a presynaptic spike occurs
        self.pre_syn_clopath = Equations('''
            # depression term
                w_minus = A_LTD_u*(u_lowpass1/mV - Theta_low/mV)*int(u_lowpass1/mV - Theta_low/mV > 0)  

            # update overall weight
                w_clopath = clip(w_clopath-w_minus, 0, w_max)

            # update presynaptic trace
                x_trace += 1
            ''' )

        # clopath equations
    def _create_compartments(self, n_compartment=4, n_neuron=1):
        '''
        '''
        # create compartments
        #===================================================================
        n_total = n_compartment*n_neuron
        self.basals_i = range(0, n_total, n_compartment)
        self.somas_i = range(1, n_total, n_compartment)
        self.proximals_i = range(2, n_total, n_compartment)
        self.distals_i = range(3, n_total, n_compartment)
        self.compartments = NeuronGroup(n_total, eqs_compartment)
        self.basals = compartments[self.basals_i]
        self.somas = compartments[self.somas_i]
        self.proximals = compartments[self.proximals_i]
        self.distals = compartments[self.distals_i]
        self.neurons=[]
        for neuron in range(n_neuron):
            self.neurons.append(compartments[neuron:neuron+n_compartment])

        # # create compartments
        # #======================================================================
        # self.somas = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset=eqs_reset, refractory=refractory_time, method=method)
        # self.proximals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_prox, reset=eqs_reset, refractory=refractory_time, method=method)
        # self.distals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_dist, reset=eqs_reset, refractory=refractory_time, method=method)
        # self.basals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_basal, reset=eqs_reset, refractory=refractory_time, method=method)

    def _connect_compartments(self):
        '''
        '''
        # connect compartments
        #======================================================================
        n_neuron = self.p['n_neuron']
        self.connect_basals = Synapses(self.somas, self.basals, eqs_connect)
        self.connect_somas = Synapses(compartments[self.basals_i+self.proximals_i], self.somas, eqs_connect)
        self.connect_proximals = Synapses(compartments[self.proximals_i+self.distals_i], self.proximals, eqs_connect)
        self.connect_distals = Synapses(self.proximals, self.distals, eqs_connect)
        for i in range(self.p['n_neuron']):
            self.connect_basals.connect(i=i, j=i)
            self.connect_somas.connect(i=[i, i+n_neuron], j=i)
            self.connect_proximals.connect(i=[i, i+n_neuron], j=i)
            self.connect_distals.connect(i=i, j=i)

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
                
        self.input_spikes = SpikeGeneratorGroup(n_inputs, indices, input_times )

    def _connect_inputs(self, input_spikes, post_compartment, post_indices, p_connect=1):
        '''
        '''

        self.input_syn = Synapses(input_spikes, proximals, eqs_syn, pre=pre_syn)

        self.input_syn.connect(p=p_connect)

    def _set_initial_conditions(self):
        '''
        '''
        pass
    def _record_variables(self, compartments, variables):
        '''
        '''
        self.rec_somas = StateMonitor(somas, ('u'), record=True)
        self.rec_proximals = StateMonitor(proximals, ('u'), record=True)
        self.rec_distals = StateMonitor(distals, ('I_axial1','u'), record=True)
        self.rec_basals = StateMonitor(basals, ('u'), record=True)

    def _run(self):
        run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses+1)/pulse_freq 
        run(run_time*ms)

    def _plots(self):
        pass

def _test_run():
    # Parameters
    #`````````````````````````````````````````````````````````````````````````
    p = {
    'E_L': -69*mV
    }

    # compartments
    n=1 # number of neurons
    E_L = -69*mV # mV
    g_L = 40*nS # nS
    delta_T = 2*mV # mV
    C = 281*pF
    t_noise = 20*ms
    t_V_T = 50*ms
    refractory_time = 1*ms
    spike_hold_time=0.9*ms # must be less than refractory time
    reset = -55*mV

    # FIXME
    # axial conductance between compartments
    # g_axial = 1000*nS
    g_axial_somas_basals = 1250*nS
    g_axial_basals_somas = 50*nS
    g_axial_somas_proximals = 1250*nS
    g_axial_proximals_somas = 50*nS
    g_axial_proximals_distals = 1500*nS
    g_axial_distals_proximals = 225*nS
    # I_axial1=0
    # I_axial2=0

    # synapses
    # ampa
    g_max_ampa = 3*100*nS
    t_ampa = 2*ms
    E_ampa = 0*mV
    # nmda
    g_max_nmda = 100*nS
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
    threshold_soma = '-50*mV'
    threshold_prox = '-30*mV'
    threshold_dist = '-30*mV'
    threshold_basal = '-30*mV'
    reset_soma = '-55*mV'
    reset_prox = '-55*mV'
    reset_dist = '-55*mV'
    reset_basal = '-55*mV'
    V_Trest_soma = -55*mV
    V_Trest_proximal = -35*mV
    V_Trest_distal = -35*mV
    V_Trest_basal = -35*mV
    V_T_max_soma = -30*mV
    V_T_max_proximal = -20*mV
    V_T_max_distal = -20*mV
    V_T_max_basal = -20*mV
    V_hold_soma = 20*mV
    V_hold_proximal = -20*mV
    V_hold_distal = -20*mV
    V_hold_basal = -20*mV

    # connection parameters
    connect_prob = 0.1

    # input stimulus
    bursts = 1
    pulses = 4
    burst_freq = 5
    pulse_freq = 100
    warmup = 20 

    # field
    # field = 0*mV

    method = 'euler'

    w=1

    @implementation('numpy')
    @check_units(x=volt, result=volt)
    def _rectify_positive(x):
        if x <= 0.:
            x=0.
        return x

    @implementation('numpy')
    @check_units(x=volt, result=volt)
    def _rectify_negative(x):
        if x >= 0.:
            x=0.
        return x
    # rectify_positive = Function(_rectify_positive, arg_units=[volt], return_unit=volt, implementation='numpy')
    # rectify_negative = Function(_rectify_negative, arg_units=[volt], return_unit=volt, implementation='numpy')

    t_reset = .1*ms

    # du/dt = (I_L + I_exp + I_axial + I_syn)/C : volt (unless refractory)
    # u = V_hold*(1-int(not_refractory)) + reset*(int(not_refractory))
    eqs_reset = '''
    
    u = -20*mV

    '''
    eqs_compartment = '''
    du/dt = int(not_refractory)*(I_L + I_exp + I_axial + I_syn)/C  + (t-lastspike>spike_hold_time)*(1-int(not_refractory))*(reset-u)/t_reset : volt 

    I_L = -g_L*(u - E_L) : amp

    I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

    # I_noise = s*(2*t_noise)**(1/2) : amp

    dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

    # ds/dt = -(s + xi1)/t_noise : amp

    # xi1 = randn()*amp : amp (constant over dt)

    # FIXME
    I_axial = I_axial1 + I_axial2 : amp
    I_axial1 : amp
    I_axial2 :amp
    I_syn : amp
    I_ext : amp
    V_Trest : volt
    # C : farad
    '''

    # # FIXME
    # eqs_connect1 = '''
    
    # field : volt # axial current due to electric field
    # g_axial : siemens
    # I_axial1_post = g_axial*(u_pre-u_post) + g_axial*field :  amp (summed) 
    # '''

    # FIXME
    eqs_connect1 = '''
    
    field : volt # axial current due to electric field
    g_axial_in : siemens
    g_axial_out : siemens
    I_axial1_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 
    '''

    eqs_connect2 = '''
    
    field : volt # axial current due to electric field
    g_axial_in : siemens
    g_axial_out : siemens
    I_axial2_post = g_axial_in*clip(u_pre-u_post, 0*volt, 1000*volt) + g_axial_out*clip(u_pre-u_post, -1000*volt, 0*volt)  :  amp (summed) 

    '''

    eqs_syn = '''
    dg_nmda/dt = -g_nmda/t_nmda : siemens (clock-driven)
    dg_ampa/dt = -g_ampa/t_ampa : siemens (clock-driven)
    B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 
    I_nmda = -g_nmda*B*(u_post-E_nmda) : amp
    I_ampa = -g_ampa*(u_post-E_ampa) : amp
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
    somas = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_soma, reset=eqs_reset, refractory=refractory_time,method=method)
    proximals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_prox, reset=eqs_reset, refractory=refractory_time, method=method)
    distals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_dist, reset=eqs_reset, refractory=refractory_time, method=method)
    basals = NeuronGroup(n, eqs_compartment, threshold='u>'+threshold_basal, reset=eqs_reset, refractory=refractory_time, method=method)

    somas.u = E_L
    proximals.u = E_L
    distals.u = E_L
    basals.u = E_L
    somas.V_T= V_Trest_soma
    proximals.V_T = V_Trest_proximal
    distals.V_T = V_Trest_distal
    basals.V_T = V_Trest_basal
    somas.V_Trest= V_Trest_soma
    proximals.V_Trest = V_Trest_proximal
    distals.V_Trest = V_Trest_distal
    basals.V_Trest = V_Trest_basal
    # somas.V_hold= V_hold_soma
    # proximals.V_hold = V_hold_proximal
    # distals.V_hold = V_hold_distal
    # basals.V_hold = V_hold_basal
    # connect compartments
    #======================================================================
    connect_somas_basals = Synapses(somas, basals, eqs_connect1)
    # connect_somas_proximals = Synapses(somas, proximals, eqs_connect2)
    connect_proximals_distals = Synapses(proximals, distals, eqs_connect1)
    connect_basals_somas = Synapses([basals,proximals], somas, eqs_connect1)
    connect_proximals_somas = Synapses(proximals, somas, eqs_connect2)
    connect_distals_proximals = Synapses(distals, proximals, eqs_connect1)
    for i in range(n):
        connect_somas_basals.connect(i=i, j=i)
        # connect_somas_proximals.connect(i=i, j=i)
        connect_proximals_distals.connect(i=i, j=i)
        connect_basals_somas.connect(i=i, j=i)
        connect_proximals_somas.connect(i=i, j=i)
        connect_distals_proximals.connect(i=i, j=i)

    # update axial conductances
    #``````````````````````````
    connect_somas_basals.g_axial_in = g_axial_somas_basals
    connect_somas_basals.g_axial_out = g_axial_basals_somas
    # connect_somas_proximals.g_axial_in = g_axial_somas_proximals
    # connect_somas_proximals.g_axial_out = g_axial_proximals_somas
    connect_proximals_distals.g_axial_in = g_axial_proximals_distals
    connect_proximals_distals.g_axial_out = g_axial_distals_proximals
    connect_basals_somas.g_axial_in = g_axial_basals_somas
    connect_basals_somas.g_axial_out = g_axial_somas_basals
    connect_proximals_somas.g_axial_in = g_axial_proximals_somas
    connect_proximals_somas.g_axial_out = g_axial_somas_proximals
    connect_distals_proximals.g_axial_in = g_axial_distals_proximals
    connect_distals_proximals.g_axial_out = g_axial_proximals_distals
        
    # input stimuli
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

    input_syn = Synapses(input_spikes, proximals, eqs_syn, pre=pre_syn)

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
    rec_somas = StateMonitor(somas, ('u'), record=True)
    rec_proximals = StateMonitor(proximals, ('u'), record=True)
    rec_distals = StateMonitor(distals, ('I_axial1','u'), record=True)
    rec_basals = StateMonitor(basals, ('u'), record=True)

    # run
    #=======================================================================
    run_time = warmup + 1000*(bursts-1)/burst_freq + 1000*(pulses+1)/pulse_freq 
    run(run_time*ms)
        
    # plot
    #=======================================================================
    figure()
    plot(rec_distals.t/ms, rec_distals.I_axial1.T/pA)
    figure()
    plot(rec_distals.t/ms, rec_distals.u.T/mV)
    plot(rec_distals.t/ms, rec_proximals.u.T/mV)
    figure()
    plot(rec_distals.t/ms, rec_somas.u.T/mV)
    figure()
    
    show()

if __name__ == '__main__':
    _test_run()


    