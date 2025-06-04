import numpy as np

def string_method(timestamps, fs, neurons_idxs, sim_time_samples=float('NaN')):

    '''String method algorithm for the Burst Detection:
    calculate timestamps list of the first and last spike for all bursts and other parameters
    '''
    maxISI = 100                                                        # in ms
    minintraburstspikes = 5 
    burst_event_total = []
    end_burst_total = []
    intraburstspikes = []
    mbr = []
    burstcounts = []
    single_neur_burstlengths = []
    for i, neuron in enumerate(timestamps) :
        if len(neuron)>0 and i in neurons_idxs:

            neuron = np.array(neuron)
            fake_spike=neuron[-1]+(maxISI*fs/1000)+1                        # in samples
            neuron = np.append(neuron, fake_spike)
    
            delta_time_spike = (neuron[1:] - neuron[:-1])*1000/fs           # in ms
            temp_mask_detection = delta_time_spike > maxISI                 # Change burst focusing when time delta >= 100 ms
            temp_mask_detection = np.append(True, temp_mask_detection)
            temp_time_burst_events = neuron[temp_mask_detection]
    
            burst_event_pos = np.where(np.in1d(neuron,temp_time_burst_events))[0]
            number_inburst_spike = burst_event_pos[1:] - burst_event_pos[:-1]
            mask_detection = number_inburst_spike >= minintraburstspikes    # Change the number of spikes in the burst >= 5
            mask_detection = np.append(mask_detection, False)
            time_burst_events = neuron[temp_mask_detection][mask_detection] # in samples
    
            idx_end_burst = np.where(np.in1d(neuron,time_burst_events))[0] + number_inburst_spike[mask_detection[:-1]] - 1
            time_end_burst = neuron[idx_end_burst]*1000/fs                  # in ms
    
            burst_event_total.append(time_burst_events*1000/fs)             # in ms
            end_burst_total.append(time_end_burst)                          # in ms
            intraburstspikes.append(number_inburst_spike[mask_detection[:-1]])
            if not np.isnan(sim_time_samples):
                mbr.append(len(time_burst_events)/(sim_time_samples/fs/60)) # in bursts/min
            burstcounts.append(len(time_burst_events))
            single_neur_burstlengths.append(time_end_burst-(time_burst_events*1000/fs)) # in ms
    non_zeros_mbr = [i for i in mbr if i>0.2]
    idxs_non_zeros_mbr = [i for i, x in enumerate(mbr) if x>0.2]            # saving indexes of non-zero mbr neurons, mbr >0.2
    non_zeros_burst_event_total = [burst_event_total[i] for i in idxs_non_zeros_mbr]
    non_zeros_end_burst_total = [end_burst_total[i] for i in idxs_non_zeros_mbr]
    non_zeros_single_neur_burstlengths = [single_neur_burstlengths[i] for i in idxs_non_zeros_mbr]
    if len(non_zeros_single_neur_burstlengths) > 0:
        overall_burstlengths = np.concatenate(non_zeros_single_neur_burstlengths, axis=0)
    else:
        overall_burstlengths = np.zeros(len(timestamps))
        
    '''calculate the ibi of neuron signals only for neurons with a bursting activity (mbr>0.2)'''
    number_no_bursts_neurons = 0
    single_neur_ibi = []
    for k in range(len(non_zeros_burst_event_total)):
        if len(non_zeros_burst_event_total[k]) > 1:
            single_neur_ibi.append((non_zeros_burst_event_total[k][1:] - non_zeros_end_burst_total[k][:-1])) # in ms
        if len(non_zeros_burst_event_total[k]) == 0:
            number_no_bursts_neurons += 1
    if len(single_neur_ibi)>0:
        overall_ibi = np.concatenate(single_neur_ibi, axis=0)
    else:
        overall_ibi = []
    number_no_bursts_neurons_perc = number_no_bursts_neurons/len(timestamps)

    return non_zeros_mbr, overall_burstlengths, overall_ibi