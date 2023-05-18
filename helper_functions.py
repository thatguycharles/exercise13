from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Helper functions for Exercise 10 part 1 and 2

h.load_file("import3d.hoc") # loads NEURON library for importing 3D morphologies
morphology_file = "morphologies/cell1.asc" # morphology file
h.load_file("models/L5PCbiophys.hoc")
h.load_file("models/L5PCtemplate.hoc")

def createL5PC(morphology_file):    
    L5PC = None 
    for sec in h.allsec():
        h.delete_section(sec=sec)
    L5PC = h.L5PCtemplate(morphology_file,'apical',-1,0) # don't add spines
    return L5PC


def plot_result(t_vec, v_soma, v_syn, stim_current, syn_current, show_from = 0):
    # Adapted from https://github.com/orena1/NEURON_tutorial/blob/master/Jupyter_notebooks/Layer_5b_pyramidal_cell_Calcium_Spike.ipynb
    show_from = xind(t_vec,show_from) # convert from time to index
    t_vec = np.array(t_vec)[show_from:]
    v_soma = np.array(v_soma)[show_from:]
    v_syn = np.array(v_syn)[show_from:]
    stim_current = np.array(stim_current)[show_from:]
    syn_current = np.array(syn_current)[show_from:]
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize  = (6.5,3),gridspec_kw = {'height_ratios':[4, 1,1]})
    ax0.plot(t_vec,v_soma,  label = 'soma')
    ax0.plot(t_vec,v_syn,  label = 'synapse')
    ax0.set_ylabel('Vm (mV)')
    ax0.set_ylim(-80,40)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.get_xaxis().set_visible(False)
    ax0.legend(frameon=False)

    ax1.plot(t_vec, np.array(syn_current), color='red', label='EPSC')
    # ax1.set_ylim(-1.1*np.abs(np.array(syn_current).min()),0.1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.legend(frameon=False)

    ax2.plot(t_vec, stim_current, color='black', label='somatic step current')
    ax2.set_ylabel('current (nA)', ha='left', labelpad=15)
    ax2.set_xlabel('time (ms)')
    ax2.set_ylim(-0.02,1.05*np.array(stim_current).max())
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(frameon=False)
    return fig

# Helper function to get index of closest value in vector
def xind(vec,val):
    # finds closest value to x in x_vals vector, outputs index    
    # if x>1, outputs index for x = 1, and if x<0, outputs index for x = 0
    # Input arguments:
    #   vec - numpy array of values 
    #   val - value to query within vec  
    # Example: ind = xind(t_vec,5) # find index of time point closest to 5 ms
    return np.argmin(np.abs(vec-val))

def get_seclist_dists(cell,seclist_name):
    # Get distance of middle of compartment of sections in a SectionList within a cell template in microns
    # Input arguments:
    #   cell - cell template object
    #   seclist_name - string, name of SectionList
    # Example: dists = get_seclist_dists(cell,'apical') # get distance from soma of middle compartment in all apical dendritic sections 
    h('{}.soma[0] distance()'.format(cell.hname()))
    h('objref dists')
    h('dists = new Vector()')
    h('forsec {}.{} dists.append(distance(0.5))'.format(cell.hname(),seclist_name))
    dists = np.array(h.dists)
    return dists

def get_peaks(t_vec,v_recs,stim_delay):
    # Get peak of AP/bAP in list of voltage recordings, taken after a delay and baseline subtracted to change in Vm (in mV)
    # Input arguments:
    #   t_vec - Vector or array of time values
    #   v_recs - list of Vectors of recorded voltage values
    #   stim_delay - scalar time to start of stimulus (in same units as time vector, ms)    
    # Example: 
    # v_apics = [h.Vector().record(cell.apic[i](0.5)._ref_v) for i in [1,2,3]] # record Vm in apic[1], apic[2], and apic[3]
    # h.run() # run simulation
    # peaks = get_peaks(t_vec,v_apics,100) # get peak after 100 ms for 3 recordings in v_apics
    peaks = []
    t_ind = xind(t_vec,stim_delay)
    for v in v_recs:
        v = np.array(v)
        v0 = v[t_ind]
        peak_v = np.max(v[t_ind:]) # take max after stim delay
        peaks.append(peak_v - v0) # subtract baseline immediately before stimulus was aplplied
    return peaks


def setParams(obj_list, indices,settings):
    # obj_list - list of synapses, netstims, or netcons
    # indices - array of indices (integers) of NetStims within the lists to modify
    # settings - dictionary of key, value pairs, should match fields of object in list
    for i in indices:
        obji = obj_list[i]        
        for key, val in settings.items():
            h('{}.{} = {}'.format(obji.hname(),key,val))      

def turnOnSynapses(n_syn,distribute_mode,netstims,netstim_params):
    # Turn on a set number of synapses using setParams
    # Input arguments:
    #   n_syn - number of synapses to turn on
    #   distribute_mode - 'sequential' or 'even', 'sequential' turns on synapses sequentially from proximal to distal in section (e.g.,
    #                       with n_syn = 3, the 1st, 2nd, and 3rd synapse would be turned on)
    #                      'even' turns on synapses distributed evenly throughout the section (e.g. with n_syn = 3, the synapse at the 
    #                       beginning, middle, and end of the section would be turned on)
    #   netstims - list of NetStims connected to each synapse
    #   netstim_params - dictionary of NetStim parameters to assign, should at least include 'number' to set the number of APs to activate
    # Example:
    # turnOnSynapses(10,'even',netstims,{'number':1,'start':100}) # turns on 10 synapses, evenly distributed, with 1 AP delivered at 100 ms
    
    setParams(netstims,range(len(netstims)),{'number':0}) # turn all off to initialize
     # generate indices of synapses to turn on
    if distribute_mode == 'sequential': # turns on synapses sequentially from proximal to distal in section
        if n_syn > 1:
            syn_indices = range(n_syn)
        else:
            syn_indices = [0]
    elif distribute_mode == 'even': # turns on synapses evenly distributed in section
        if n_syn > 1:
            syn_indices = list(np.linspace(0,len(netstims)-1,n_syn,dtype=int))
        else:
            syn_indices = [int(len(netstims)/2)]
    setParams(netstims,syn_indices,netstim_params) # assign parameters for all synpases in syn_indices

    return syn_indices

def plot_V_recs(t_vec,v_soma,v_recs,rec_names,x_lim=None,title=None):    
    fig = make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=t_vec,y=v_soma,name='soma',line=dict(color='rgb(0,0,0)')))    
    for v, name in zip(v_recs,rec_names):
        fig.add_trace(go.Scatter(x=t_vec,y=v,name=name))
    fig['layout']['yaxis']['title'] = 'Vm (mV)'
    fig['layout']['xaxis']['title'] = 'time (ms)'    
    if x_lim is not None:
        fig.update_layout(xaxis_range=x_lim)
    if title is not None:
        fig.update_layout(title=title)
    fig.show()
    return fig

def toggleChannelSeclist(channel_type,cell,seclist,turn_off):
    # Turns off/on set of ion channel conductances in input seclist
    # Input arguments:
    #   channel_type - string of channel type to turn off, see key of mech_names dictionary for possible channel types
    #   cell - Cell template object
    #   seclist - NEURON h.SectionList object
    #   turn_off - True or False. True to turn off channels (set gbar to 0), False to revert channels back to default conductances
    # example: 
    # L5PC = h.L5PCtemplate(morphology_file)
    # toggleChannelSeclist('Cav',L5PC,L5PC.apical,1) # turns off Cav channels
    # toggleChannelSeclist('Cav',L5PC,L5PC.apical,0) # turns back on
    mech_names = {
        'Cav': [('gCa_LVAstbar','Ca_LVAst'),('gCa_HVAbar','Ca_HVA')],
        'Kv': [('gK_Tstbar','K_Tst'),('gK_Pstbar','K_Pst'),('gSKv3_1bar','SKv3_1'),('gImbar','Im')],
        'Nav': [('gNaTa_tbar','NaTa_t'),('gNap_Et2bar','Nap_Et2')],
        'Kca': [('gSK_E2bar','SK_E2')],
        'Ih': [('gIhbar','Ih')]
    }
    if turn_off:
        for sec in seclist: # loop through sections in section list
            for gbar,mech in mech_names[channel_type]: # grab the name of the peak conductance and mechanism
                if h.ismembrane(mech,sec=sec): # check if mechanism is present in this section
                    h('{} {}_{} = 0'.format(sec.hname(),gbar,mech)) # turn off conductance             
        print('Turned off {} currents in SectionList'.format(channel_type))
    else:
       cell.biophys() # resets conductances to default values
       print('Reverted conductances back to default values')
