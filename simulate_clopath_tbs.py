'''
'''
from brian2 import *
import numpy as np
import pandas as pd
from scipy import stats
import clopath
import copy

prefs.codegen.target = 'numpy'

class Param:
    '''
    '''
    def __init__(self,):
        '''
        '''
        
        self.p = {
        'trials':1,
        'dt':0.1*ms, 
        'run_time':300*ms,
        'E_L':-70*mV,
        'g_L':40*nS,
        'delta_T':2*mV,
        'C':281*pF,
        't_noise': 20*ms,
        't_V_T':50*ms,
        'refractory':2*ms,
        'V_Trest': -55*mV,
        'V_Tmax':-30*mV,
        'reset_condition':'u=-70*mV',
        'threshold_condition':'u>V_T+20*mV',
        'I_after' : 400*pA,
        'a_adapt' : 4*nS,
        'b_adapt' : 0.805*pA,
        't_w_adapt' : 144*ms,
        't_z_after' : 40*ms,
        'u_reset' : -70*mV,

        'u_hold':30*mV,
        'refractory_time':2*ms,
        'spike_hold_time':1*ms, # must be at least 0.2 ms less than refractory time
        'spike_hold_time2': 2*ms - 2*defaultclock.dt,
        't_reset':0.5*ms, # time constant for resetting voltage after holding spike (should be equal to dt)
        'hold_spike':1,
        'update_ampa_online':0,
        'update_nmda_online':0,   

        # synapse parameters
    #========================================================================
        # ampa
        #''''''''''''''''''''

        'g_max_ampa' :100*nS,
        't_ampa' : 2*ms,
        'E_ampa' : 0*mV,
        'w_ampa' : 0.2,

        # nmda
        #''''''''''''''''''''''
        'g_max_nmda' : 50*nS, #g_max_ampa/2, #75*nS
        't_nmda' : 50*ms,
        'E_nmda' : 0*mV,
        'w_nmda' : 0.5,

        # short term plasticity
        #'''''''''''''''''''''''
        'f' : 5.3,
        't_F' : 94*ms,
        'd1' : 0.45,
        't_D1' : 540*ms,
        'd2' : 0.12,
        't_D2' : 45*ms,
        'd3' : 0.98,
        't_D3' : 120E3*ms,

        # clopath
        #'''''''''''''''''''''''''
        'v_target' : 100*mV*mV,
        'A_LTD' : 50*100E-5,
        'A_LTP' : 50*40E-5/ms,
        'tau_lowpass2' : 5*ms,
        'tau_x' : 10*ms,
        'tau_lowpass1' : 6*ms,
        'tau_homeo' : 1000*ms,
        'theta_low' : -60*mV,
        'theta_high' : -50*mV,
        'w_max_clopath' : 2,
        'x_reset':1,
        
        

        # input/stimulation parameters
        #============================================================================
        'pulses' : 4,
        'bursts' : 4,
        'pulse_freq' : 100,
        'burst_freq' : 5,
        'warmup' : 10,

        'I_input':0*pA,

        # network parameters
        #===================================================================
        'N' : 3, 

        'syn_condition': 'i==1',

        # variables to record
        #====================================================================
        'rec_variables_nrn':['u','A_LTD_homeo', 'I_nmda'],
        'rec_variables_input_syn': ['w_clopath', 'x_trace', ],


        }

        self.init_nrn = {
        'I_field':  0*pA,
        'u':self.p['E_L'],
        'V_T':self.p['V_Trest'],
        'w_adapt':0*pA,
        'z_after':0*pA 
        }

        self.init_input_syn={
        'F':1,
        'D1':1,
        'D2':1,
        'D3':1,
        'u_lowpass1':self.p['E_L'],
        'u_lowpass2':self.p['E_L'],
        'u_homeo':0*mV,
        'w_clopath':0.5,
        }

# create poisson inputs for receptive field development
# create discretized array of input neurons
# the current location (center) shifts along the neurons, where the neurons firing rate is a gaussian function of its distance from the center

class Inputs:
    '''
    '''
    def __init__(self, ):
        '''
        '''
        pass

    def _circular_track_firing_rates(self, track_length=1, dt=0.1, loops=2, speed=1, amplitude= 30, variance = .2, neuron_groups=10):
        '''
        '''
        x = 2*np.pi/track_length*np.linspace(0, loops*track_length, int(loops*track_length/speed/dt+1))
        y_locations = np.sin(x)
        x_locations = np.cos(x)

        locations = np.arctan2(y_locations, x_locations)
        nrn_locations = (np.linspace(-np.pi,np.pi, neuron_groups))

        N_nrn = nrn_locations.shape[0]
        N_t = locations.shape[0]
        locations_norm = abs(np.tile(locations, [N_nrn, 1]) - np.tile(nrn_locations,[N_t,1]).T)
        locations_norm[locations_norm>np.pi] = 2*np.pi-locations_norm[abs(locations_norm)>np.pi]

        rates = amplitude*np.exp(-(locations_norm**2)/(2.*variance**2))


        return rates, locations

    def _tbs(self, p):
        '''
        '''
        pulses = p['pulses']
        bursts = p['bursts']
        warmup = p['warmup']
        burst_freq = p['burst_freq']
        pulse_freq = p['pulse_freq']

        input_times = np.zeros(pulses*bursts)
        indices = np.zeros(pulses*bursts)
        cnt=-1
        for burst in range(bursts):
            for pulse in range(pulses):
                cnt+=1
                time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
                print time
                input_times[cnt] = time

        input_group = SpikeGeneratorGroup(1, indices, input_times)
        return input_group

class Run:
    '''
    '''
    def __init__(self, P):
        '''
        '''
        self.P=P

    def _setup(self, P):
        '''
        '''

        # create neuron group
        #========================================================================
        # nrn = NeuronGroup(N, eq_nrn , threshold=threshold_condition, reset=eq_nrn_reset, refractory=refractory_time, method='euler')
        Eq = Eqs()
        self.nrn = NeuronGroup(N, Eq.nrn , threshold=threshold_condition, reset=Eq.nrn_reset,   refractory=refractory_time,  method='euler')


        # inputs to network
        #====================================================================
        self.input_nrn = Inputs()._tbs(P.p)

        # create synapses
        #====================================================================
        self.eq_syn =  Eq.syn_stp + '\n' + Eq.syn_clopath  

        self.eq_syn_pre = Eq.syn_ampa_pre + '\n' + Eq.syn_nmda_pre + '\n'+ Eq.syn_stp_pre + '\n' + Eq.syn_clopath_pre

        self.input_syn = Synapses(self.input_nrn, self.nrn, self.eq_syn, on_pre=self.eq_syn_pre,)
        self.input_syn.connect(condition=P.p['input_syn_condition'])

        # setup recording variables
        #===================================================================

        self.rec={}
        self.rec['nrn'] = StateMonitor(self.nrn, P.p['rec_variables_nrn'], record=True)
        self.rec['nrn_spikes'] = SpikeMonitor(self.nrn)
        self.rec['syn'] = StateMonitor(self.input_syn, P.p['rec_variables_input_syn'], record=True)
        self.rec['input'] = SpikeMonitor(self.input_nrn, record=True)

        # initial conditions
        #===================================================================
        for param, val in P.init_nrn.iteritems():
            setattr(self.nrn, param, val)

        for param, val in P.init_input_syn.iteritems():
            setattr(self.input_syn, param, val)

        self.net = Network(self.nrn, self.input_nrn, self.input_syn, self.rec)

        return self.net

    def _run(self, P):
        '''
        '''
        # run
        #=======================================================================
        run_time = 300*ms
        net.store()
        P = Param()

        # run(run_time)
        for trial in range(trials):
            net.restore()
            net.run(run_time, namespace=P.p)


def _run():
    '''
    '''


   
    # create neuron group
    #========================================================================
    # nrn = NeuronGroup(N, eq_nrn , threshold=threshold_condition, reset=eq_nrn_reset, refractory=refractory_time, method='euler')
    Eq = Eqs()
    nrn = NeuronGroup(N, Eq.nrn , threshold=threshold_condition, reset=Eq.nrn_reset,   refractory=refractory_time,  method='euler')

    # nrn = NeuronGroup(N, eq_nrn, threshold=threshold_condition, reset='u=u_refractory', refractory=refractory_time, method='euler')

    # circular track input
    #========================================================================
    # loops=3
    # track_length=1
    # speed=1./500
    # dt=.1
    # input_dt = loops*track_length/int(track_length/speed/dt+1)
    rates, locations = _circular_track_firing_rates(track_length=1, dt=0.1, loops=8, speed=1./2000, amplitude= 50, variance = .2, neuron_groups=10)
    # input_rates = TimedArray(_circular_track_firing_rates(track_length=1, dt=0.1, loops=3, speed=1./2000, amplitude= 20, variance = .2, neuron_groups=10).T*Hz,dt=.1*ms)
    input_rates = TimedArray(rates.T*Hz, dt=0.1*ms)
    # figure()
    # plot(input_rates(:,0))
    # show()
    # input_nrn = NeuronGroup(N=10, model='temp:1', threshold='rand()<input_rates(t,i)*dt' )


    # poisson input group
    #========================================================================
    # input_nrn = PoissonGroup(N=2, rates=[100*Hz, 20*Hz])


    # theta burst input
    #========================================================================
    input_times1 = np.zeros(pulses*bursts)
    indices1 = np.zeros(pulses*bursts)
    cnt=-1
    for burst in range(bursts):
        for pulse in range(pulses):
            cnt+=1
            time = warmup + 1000*burst/burst_freq + 1000*pulse/pulse_freq
            print time
            input_times1[cnt] = time

    pulses2=1
    bursts2 = 4
    pulse_freq2 = 100
    burst_freq2 = 5
    warmup2 = warmup + 10
    input_times2 = np.zeros(pulses2*bursts2)
    indices2 = np.ones(pulses2*bursts2)
    cnt=-1
    for burst2 in range(bursts2):
        for pulse2 in range(pulses2):
            cnt+=1
            time2 = warmup2 + 1000*burst2/burst_freq2 + 1000*pulse2/pulse_freq2
            input_times2[cnt] = time2
    print input_times2
    indices = np.hstack([indices1,indices2])
    input_times = np.hstack([input_times1, input_times2])*ms
    # print indices, input_times
    input_nrn = SpikeGeneratorGroup(2, indices, input_times )

# set maximum weights
#=======================================================================
 
    # weight_max_event = {'weight_max':'w_clopath>=w_max_clopath'}
    # weight_reset = 'w_clopath=w_max_clopath' 

# create synapses
#========================================================================
    
    
    eq_syn =  Eq.syn_stp + '\n' + Eq.syn_clopath  

    eq_syn_pre = Eq.syn_ampa_pre + '\n' + Eq.syn_nmda_pre + '\n'+ Eq.syn_stp_pre + '\n' + Eq.syn_clopath_pre

    syn = Synapses(input_nrn, nrn, eq_syn, on_pre=eq_syn_pre,)
    # syn.run_on_event('weight_max', weight_reset)
    condition = 'i==1'
    syn.connect(condition=condition)

    # recording
    #============================================================================
    rec={}
    rec['nrn'] = StateMonitor(nrn, ('u','A_LTD_homeo', 'I_nmda'), record=True)
    rec['nrn_spikes'] = SpikeMonitor(nrn)
    rec['syn'] = StateMonitor(syn, ('w_clopath', 'x_trace', ), record=True)
    rec['input'] = SpikeMonitor(input_nrn, record=True)
    # rec['locations']=locations
    # rec_input = SpikeMonitor(input_nrn, record=True)
    # rec_syn = StateMonitor(syn, ('w_clopath', 'x_trace', ), record=True)

    # set initial conditions
    #========================================================================
    # nrn.u['u>20*mV']=20*mV
    # nrn.u['int(not_refractory)'] = u_refractory
    # nrn.u['t-lastspike==refractory_time']=u_reset

    # nrn.u['u>20*mV']=u_hold
    field = [5*20*pA, 0*pA, -5*20*pA]
    field_color = ['red', 'black', 'blue']
    input_marker = ['--','-']
    nrn.I_field[0] = 5*20*pA
    nrn.I_field[1] = 0*pA
    nrn.I_field[2] = -5*20*pA
    nrn.u = E_L
    nrn.V_T = V_Trest
    nrn.w_adapt = 0*pA
    nrn.z_after = 0*pA
    syn.F=1
    syn.D1=1
    syn.D2=1
    syn.D3=1
    syn.u_lowpass1=E_L
    syn.u_lowpass2=E_L
    # syn.u_homeo=E_L
    syn.w_clopath=0.5
    # syn.w_ampa=0.5


    # create network
    #=======================================================================
    net = Network(nrn, input_nrn, syn, rec)
    # run
    #=======================================================================
    run_time = 300*ms
    net.store()
    P = Param()

    # run(run_time)
    for trial in range(trials):
        net.restore()
        net.run(run_time, namespace=P.p)

    # net.run(run_time)

    # figures
    #========================================================================

    rec['locations']=locations
    return rec, input_rates

class Analysis:
    '''
    '''
    # to do
        # metric for the selectivity of the developed receptive field
        # plot 
# plot firing rate as a function of location on track
class Plots:
    '''
    '''
    def __init__(self,):
        '''
        '''

    def _plot_voltage(self, rec):
        '''
        '''
        self.u  = rec['nrn'].u
        linewidth=4
        colors = ['r','k','b']
        plt.figure()
        for i in range(self.u.shape[0]):
            plt.plot(rec['nrn'].t/ms, self.u[i,:]/mV, color=colors[i], linewidth=linewidth)

        plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
        plt.ylabel('Membrane potential (mV)', fontsize=20, fontweight='bold')
        plt.ylim([-72,-20])
        plt.show(block=False)


    def _plot_weights(self, rec, npre, npost):
        '''
        '''
        self.w = rec['syn'].w_clopath
        self.w = self.w.reshape(npre,npost,self.w.shape[1])
        markers = ['-','--']
        colors = ['r','k','b']
        linewidth=4
        plt.figure()
        for pre_i in range(npre-1):
            for post_i in range(npost):
                plt.plot(rec['syn'].t/ms, self.w[pre_i,post_i,:], color=colors[post_i], linestyle=markers[pre_i], linewidth=linewidth)
        plt.xlabel('Time (ms)', fontsize=20, fontweight='bold')
        plt.ylabel('Synaptic Weight (AU)', fontsize=20, fontweight='bold')
        plt.ylim([0.4,1.5])
        plt.show(block=False)
    # FIXME
    def _rate_x_location(self, rec):
        '''
        '''
        tol = .01
        integration_window = 1000
        firing_rate_filter = np.ones(integration_window)
        locations=rec['locations']

        anodal_idx = rec['nrn_spikes'].i == 0
        control_idx = rec['nrn_spikes'].i == 1
        cathodal_idx = rec['nrn_spikes'].i == 2
        anodal_spikes = rec['nrn_spikes'].t[anodal_idx]
        control_spikes = rec['nrn_spikes'].t[control_idx]
        cathodal_spikes = rec['nrn_spikes'].t[cathodal_idx]
        time = np.linspace(0, locations.shape[0]*defaultclock.dt, locations.shape[0])
        locations_unique = np.array(list(set(np.round(locations, 0))))
        print locations_unique.shape[0],time.shape[0]
        spike_array_anodal = np.zeros((locations_unique.shape[0],time.shape[0]  ))
        
        for spike_i, spike in enumerate(anodal_spikes):
            # print np.round(spike/ms,2)
            # print np.round(time[10]/ms,2)
            # time_i = np.where(np.round(time/ms,2)==np.round(spike/ms,2))[0]
            # time_i = np.where(abs(time/ms-spike/ms) < tol)
            time_i = np.argmin(abs(time/ms-spike/ms))
            location = locations[time_i]
            # print np.round(spike/ms,2), time_i, location, min(abs(time/ms-spike/ms))

            location_unique_i = np.argmin(abs(locations_unique-location))
            print location_unique_i, time_i
            # print location_unique_i, time_i
            spike_array_anodal[location_unique_i,time_i]=1

        firing_rate_array_anodal = np.zeros(spike_array_anodal.shape)
        for loc_i in range(spike_array_anodal.shape[0]):
            firing_rate_array_anodal[loc_i, :] = np.convolve(spike_array_anodal[loc_i,:], firing_rate_filter, mode='same')


        rec['time'] = time
        rec['locations_unique'] = locations_unique
        rec['anodal_spikes'] = anodal_spikes
        rec['cathodal_spikes'] =  cathodal_spikes
        rec['control_spikes'] = control_spikes

        plt.figure()
        plt.imshow(firing_rate_array_anodal, aspect='auto')
        plt.colorbar()
        plt.show(block=False)

        return spike_array_anodal

    def _time_series(self, rec, variables, indices):
        '''
        ==Args==
        -rec : dictionary of recorded StateMonitor objects
        -variables : list of variables to plot
        ==Out==
        ==Update==
        ==Comments==
        '''
        # object for dimensionless variable
        dimensionless = units.fundamentalunits.DIMENSIONLESS
        # map dimension to specific unit
        dimension_map = [(volt, mV, 'mV'), (amp,pA,'pA'), (siemens,nS,'nS'), (dimensionless,1, 'AU')]
        # iterate over variables
        for variable_i, variable in enumerate(variables):
            # iterate through recorded data types
            for rec_type, rec_data in rec.iteritems():

                if isinstance(rec_data, StateMonitor) and variable in rec_data.recorded_variables.keys():
                    figure()
                    plot_time = getattr(rec_data, 't')
                    plot_data = getattr(rec_data, variable)
                    plot_dimensions = get_dimensions(plot_data)
                    print plot_dimensions
                    plot_units = [temp[1] for temp_i, temp in enumerate(dimension_map) if plot_dimensions==get_dimensions(temp[0])][0]
                    print plot_units
                    plot(plot_time/ms, plot_data[indices[variable_i]].T/plot_units,)
                    show(block=False)

                elif isinstance(rec_data, SpikeMonitor) and variable in list(rec_data.record_variables):
                    figure()
                    plot_time = getattr(rec_data, 't')
                    plot_data = getattr(rec_data, variable)
                    plot_dimensions = get_dimensions(plot_data)
                    plot_units = [temp[1] for temp_i, temp in enumerate(dimension_map) if plot_dimensions==temp[0]][0] 
                    plot(plot_time/ms, plot_data[indices[variable_i]].T/plot_units)
                    show(block=False)

class Eqs:
    '''
    '''
    def __init__(self, ):
        '''
        '''
        self.nrn = '''

            du/dt = int(not_refractory)*I_total/C  + I_reset  + ((t-lastspike)>spike_hold_time2)*(1-int(not_refractory))*I_total/C :volt

            I_total = (I_L + I_syn + I_exp + I_field + I_input - w_adapt + z_after ) : amp

            I_reset = ((t-lastspike)<spike_hold_time2)*((t-lastspike)>spike_hold_time)*(1-int(not_refractory))*((u_reset-u)/t_reset + z_after/C) : volt/second     

            I_L = -g_L*(u - E_L) : amp

            I_exp = g_L*delta_T*exp((u - V_T)/delta_T) : amp

            dV_T/dt = -(V_T-V_Trest)/t_V_T : volt

            dw_adapt/dt = a_adapt*(u-E_L)/t_w_adapt - w_adapt/t_w_adapt : amp

            dz_after/dt = -z_after/t_z_after : amp 

            # synaptic
            #=======================================
            # ampa
            #``````````````
                dg_ampa/dt = -g_ampa/t_ampa : siemens 
                I_ampa = -g_ampa*(u-E_ampa) : amp

            # nmda
            #`````````````````
                dg_nmda/dt = -g_nmda/t_nmda : siemens 
                B =  1/(1 + exp(-0.062*u/mV)/3.57) : 1 
                I_nmda = -g_nmda*B*(u-E_nmda) : amp

                I_syn = I_ampa + I_nmda : amp

            # clopath
            #```````````````````
            # low threshold filtered membrane potential
                du_lowpass1/dt = (u-u_lowpass1)/tau_lowpass1 : volt 

            # high threshold filtered membrane potential
                du_lowpass2/dt = (u-u_lowpass2)/tau_lowpass2 : volt     

            # homeostatic term
                du_homeo/dt = (u-E_L-u_homeo)/tau_homeo : volt       

            # LTP voltage dependence
                LTP_u = (u_lowpass2/mV - theta_low/mV)*int((u_lowpass2/mV - theta_low/mV) > 0)*(u/mV-theta_high/mV)*int((u/mV-theta_high/mV) >0)  : 1

            # LTD voltage dependence
                LTD_u = (u_lowpass1/mV - theta_low/mV)*int((u_lowpass1/mV - theta_low/mV) > 0)  : 1

            # homeostatic depression amplitude
                #``````````````````````````````````
                A_LTD_homeo = A_LTD*(u_homeo**2/v_target) : 1  
            
            # parameters
            #```````````````
            # I_input : amp
            I_field : amp
            # I_syn : amp
            # I_after : amp
            # C : farad
            # g_L : siemens
            # delta_T : volt 
            # t_V_T : second
            # a_adapt : siemens
            # t_w_adapt : second
            # t_z_after : second
            # u_reset : volt
            # b_adapt : amp
            # V_Tmax : volt
            # V_Trest:volt
            # E_L : volt
        '''

        # voltage rest
        #``````````````````````````````
        self.nrn_reset ='''
            z_after = I_after 
            u = int(hold_spike)*u_hold + int(1-hold_spike)*(u_reset + dt*I_after/C)
            V_T = V_Tmax 
            w_adapt += b_adapt    
        '''
        # ampa synapses
        #'````````````````````````````````
        self.syn_ampa = '''
            dg_ampa/dt = -g_ampa/t_ampa : siemens 
            I_ampa = -g_ampa*(u_post-E_ampa) : amp
            
            # w_ampa :1
            # g_max_ampa : siemens
            # t_ampa : second
            # E_ampa : volt
        '''

        self.syn_ampa_pre = '''
        g_ampa += int(update_ampa_online)*w_clopath*g_max_ampa*A + int(1-update_ampa_online)*w_ampa*g_max_ampa*A 
    '''

        self.syn_nmda_pre = '''
        g_nmda += int(update_nmda_online)*w_clopath*g_max_nmda*A + int(1-update_nmda_online)*w_nmda*g_max_nmda*A 
    '''
        # nmda synapses
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.eq_syn_nmda = '''

        dg_nmda/dt = -g_nmda/t_nmda : siemens 
        B =  1/(1 + exp(-0.062*u_post/mV)/3.57) : 1 
        I_nmda = -g_nmda*B*(u_post-E_nmda) : amp
        
    '''

        # short term plasticity
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_stp = '''

        dF/dt = (1-F)/t_F : 1 
        dD1/dt = (1-D1)/t_D1 : 1 
        dD2/dt = (1-D2)/t_D2 : 1 
        dD3/dt = (1-D3)/t_D3 : 1 
        A = F*D1*D2*D3 : 1
    '''

        self.syn_stp_pre = '''
        F += f 
        D1 *= d1
        D2 *= d2
        D3 *= d3 
    '''

        # clopath plasticity rule
        #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.syn_clopath = '''      
        
        # lowpass presynaptic variable
            dx_trace/dt = -x_trace/tau_x : 1                          

        # clopath rule for potentiation (depression implented with on_pre)

            # dw_clopath/dt = int(w_clopath<w_max_clopath)*A_LTP*x_trace*LTP_u_post :1
            
            dw_clopath/dt = saturated*(w_max_clopath-w_clopath)/dt + (1-saturated)*A_LTP*x_trace*LTP_u_post : 1

            saturated = int((w_clopath+A_LTP*x_trace*LTP_u_post*dt)>w_max_clopath) : 1 # indicates that next weight update brings synapse to saturation
    '''

        self.syn_clopath_pre = '''

        w_minus = A_LTD_homeo_post*LTD_u_post
        
        w_clopath = clip(w_clopath-w_minus, 0, w_max_clopath)  # apply LTD

        x_trace += dt*x_reset/tau_x  # update presynaptic trace with each input
    '''

    def _add_eq(*list_args):

        '''
        '''
        equation  = ''
        for eq in list_args:
            equation = equation + '\n' + eq 

        return equation 