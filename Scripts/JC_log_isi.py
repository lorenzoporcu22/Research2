from scipy.sparse import csr_matrix
import numpy as np
from scipy.signal import find_peaks
import os
import h5py
import numpy as np
import neo
import quantities as pq



def BurstDetection(spikes, n_spikes_min=5, isi_th=100., mbr_min=0.4, freqSam=10000, acqTime=60, log_isi_max_th_flag=False, min_mfb_flag=False):

    chBurstStart = [np.nan for _ in range(len(spikes))]
    chBurstEnd = [np.nan for _ in range(len(spikes))]
    chMBR = [np.nan for _ in range(len(spikes))]
    chBD = [np.nan for _ in range(len(spikes))]
    chSPB = [np.nan for _ in range(len(spikes))]
    chRS = [np.nan for _ in range(len(spikes))]

    if log_isi_max_th_flag:
        chISITh = LogISIHistogram(spikes, freqSam=freqSam)

    for el, sp in enumerate(spikes):

        if len(sp) <= 1:
            continue

        sp = np.asarray(sp)

        up_small, down_small = BurstEvent(spikes=sp,
                                          n_spikes_min=n_spikes_min,
                                          isi_th=isi_th,
                                          freqSam=freqSam)

        if np.isnan(chISITh[el]) or not log_isi_max_th_flag:
            burst_up = up_small
            burst_down = down_small

        else:
            if len(up_small) >= 2:
                cond_merge = (sp[up_small][1:] - sp[down_small][:-1]) > 500  # self.chISITh[el]
                up_small = up_small[np.concatenate(([True], cond_merge))]
                # down_small = down_small[np.concatenate((cond_merge, [True]))]  # Not needed

            up_large, down_large = BurstEvent(spikes=sp,
                                              n_spikes_min=n_spikes_min,
                                              isi_th=chISITh[el])

            burst_up = []
            burst_down = []

            for u_large, d_large in zip(up_large, down_large):

                burst_nested = np.logical_and(u_large <= up_small, d_large >= up_small)

                if sum(burst_nested) < 2:
                    burst_up.append(u_large)
                    burst_down.append(d_large)
                else:
                    burst_up.append(u_large)
                    burst_up += up_small[burst_nested][1:].tolist()
                    burst_down += (up_small[burst_nested][1:] - 1).tolist()
                    burst_down.append(d_large)

        burst_up = np.asarray(burst_up)
        burst_down = np.asarray(burst_down)

        assert len(burst_up) == len(burst_down)

        if min_mfb_flag and burst_up.size > 0:
            cond_min_mfb = freqSam*(burst_down - burst_up + 1)/(sp[burst_down] - sp[burst_up]) >= 50  # sp/s
            burst_up = burst_up[cond_min_mfb]
            burst_down = burst_down[cond_min_mfb]

        if len(burst_up)/(acqTime/60) < mbr_min:
            continue

        chBurstStart[el] = sp[burst_up]-1  # samples
        chBurstEnd[el] = sp[burst_down]+1  # samples
        chMBR[el] = len(burst_up)/(acqTime/60)  # bursts/min
        chBD[el] = 1000*(sp[burst_down] - sp[burst_up])/freqSam  # ms
        chSPB[el] = burst_down - burst_up + 1  # number of spikes
        chRS[el] = 100*(1 - sum(burst_down - burst_up+1) / len(sp))  # %

    return chBurstStart, chBurstEnd, chMBR, chBD, chSPB, chRS

def BurstEvent(spikes, n_spikes_min=5, isi_th=100., freqSam=10000):

    isi_th = int(isi_th/1000*freqSam)  # from ms to samples

    burst_train = np.concatenate(([0], np.diff(spikes) <= isi_th, [0]))
    burst_up = np.where(np.diff(burst_train) == 1)[0]
    burst_down = np.where(np.diff(burst_train) == -1)[0]

    cond = burst_down - burst_up + 1 > n_spikes_min
    burst_up = burst_up[cond]
    burst_down = burst_down[cond]

    assert len(burst_up) == len(burst_down)

    return burst_up, burst_down

def LogISIHistogram(spikes, freqSam=10000):

    bins_per_decade = 10
    smooth_size = 3
    min_peak_dist = 2
    void_param_th = 0.7
    chISITh = np.zeros(len(spikes))

    for el, sp in enumerate(spikes):

        if len(sp) <= 1:
            continue

        all_isi = np.diff(sp)/freqSam*1000  # in ms

        max_isi = np.ceil(np.log10(np.amax(all_isi)))
        bins = np.logspace(0, max_isi, int(bins_per_decade*max_isi))
        isi_hist, _ = np.histogram(all_isi, bins=np.concatenate((bins, [np.inf])))
        isi_smooth = np.convolve(isi_hist/sum(isi_hist), np.ones(smooth_size) / smooth_size, mode='same')

        isi_peak, _ = find_peaks(isi_smooth, distance=min_peak_dist)

        if not np.any(bins[isi_peak] > 100):
            continue

        isi_peak = isi_peak[np.where(bins[isi_peak] > 100)[0][0]-1:]

        if len(isi_peak) <= 1:
            continue

        void_param = np.zeros(len(isi_peak) - 1)
        idx_minima = np.zeros(len(isi_peak) - 1, dtype=int)

        for idx, (peak_l, peak_r) in enumerate(zip(isi_peak[:-1], isi_peak[1:])):
            idx_minima[idx] = peak_l + np.argmin(isi_smooth[peak_l:peak_r])
            void_param[idx] = 1 - isi_smooth[idx_minima[idx]] / np.sqrt(isi_smooth[peak_l] * isi_smooth[peak_r])

        if not np.any(void_param >= void_param_th):
            continue

        isi_th = bins[idx_minima[np.argmax(void_param)]]

        if isi_th > 1000:
            continue

        chISITh[el] = isi_th

    return chISITh

filename = '../templates/spiketrains.h5'
duration = 120  # stesso t_stop usato prima



spike_trains = []
with h5py.File(filename, 'r') as f:
    for key in f.keys():
        times = f[key][()]  # array di spike times in secondi
        st = neo.SpikeTrain(times * pq.s, t_start=0 * pq.s, t_stop=duration * pq.s)
        spike_trains.append(st)

fs = 32000
Spikes = []
for sp in spike_trains:
    sp_values = sp.rescale('s').magnitude  # solo valori float in secondi
    spikes_samples = (sp_values * fs).astype(int).tolist()
    Spikes.append(spikes_samples)

burst_up, burst_down, mbr, bd, spb, rs = BurstDetection(
    Spikes,
    n_spikes_min=8,
    isi_th=100.,
    mbr_min=0.5,
    freqSam=fs,
    acqTime=duration,
    log_isi_max_th_flag=True,
    min_mfb_flag=True
)

burst_detection = np.array([burst_up, burst_down, mbr, bd, spb, rs], dtype=np.ndarray)
np.save('../templates/burst_detection', burst_detection)



# Carica il file
burst_detection = np.load('../templates/burst_detection.npy', allow_pickle=True)

# Estrai i singoli array
burst_up, burst_down, mbr, bd, spb, rs = burst_detection


for i, burst_starts in enumerate(burst_up):
    if isinstance(burst_starts, float) and np.isnan(burst_starts):
        print(f"Neurone {i} non ha burst rilevati")
    else:
        print(f"Neurone {i} ha {len(burst_starts)} burst")
        
        