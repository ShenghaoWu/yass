import numpy as np
import logging
from scipy.signal import argrelmax

def deconvolve(recordings, idx_local, idx,
               spt_list, shifted_templates,
               temporal_features, spatial_features,
               temp_temp, n_explore, threshold_d):

    """
    run greedy deconvolution algorithm

    Parameters
    ----------

    recording: numpy.ndarray (T, n_channels)
       A 2D array of a recording

    templates: numpy.ndarray (n_channels, n_timebins, n_templates)
       A 3D array of templates

    spike_index: numpy.ndarray (n_spikes, 2)
       A 2D array containing spikes information with two columns,
       where the first column is spike time and the second is channel.

    n_explore: int
       parameter for a function, get_longer_spt_list

    n_rf: int
       refractory period violation in timebin unit

    upsample_factor: int
       number of shifted templates to create

    threshold_a: int
        threhold on waveform scale when fitted to template
        (check make_tf_tensors)

    threshold_dd: int
        threshold on decrease in l2 norm of recording after
        subtracting a template (check make_tf_tensors)

    Returns
    -------
    spike_train: numpy.ndarray (n_spikes_recovered, 2)
        A 2D array of deconvolved spike train with two columns,
        where the first column is spike time and the second is
        cluster id.

    Notes
    -----
    [Add a brief description of the method]
    """    
    
    data_start = idx[0].start
    data_end = idx[0].stop
    # get offset that will be applied
    offset = idx_local[0].start

    n_templates, n_shifts, waveform_size, n_channels = shifted_templates.shape
    R = int((waveform_size-1)/2)

    principal_channels = np.argmax(np.max(np.abs(shifted_templates),(1,2)), 1)
    norms = np.sum(np.square(shifted_templates),(2,3))

    d_matrix = np.ones((recordings.shape[0],
                        n_templates,
                        n_shifts))*-np.Inf

    for c in range(n_channels):
        tmp_idx = np.where(principal_channels==c)[0]

        if tmp_idx.shape[0] > 0:

            spt = spt_list[c]
            spt = spt[np.logical_and(spt >= data_start,
                                     spt < data_end)]
            spt = spt - data_start + offset

            wf = recordings[
                spt[:, np.newaxis] + np.arange(
                    -R-n_explore, n_explore+R+1)]

            spatial_dot = np.matmul(spatial_features[tmp_idx][np.newaxis, np.newaxis],
                                    wf[:, :, np.newaxis, np.newaxis,:,
                                       np.newaxis]
                                   ).transpose(0, 2, 3, 1, 4, 5)[:, :, :, :, :, 0]

            dot = np.zeros((spt.shape[0], 2*n_explore+1, tmp_idx.shape[0], n_shifts))
            for j in range(2*n_explore+1):
                dot[:, j, :] = np.sum(spatial_dot[
                    :, :, :, j:j+2*R+1]*temporal_features[tmp_idx][np.newaxis], (3, 4))

            for j, k in enumerate(tmp_idx):
                d_matrix[spt[:, np.newaxis] + np.arange(
                    -n_explore,n_explore+1), k] = 2*dot[:, :, j]  - \
                norms[k][np.newaxis, np.newaxis]

    d_max_along_shift = np.max(d_matrix, 2)

    spike_train = np.zeros((0, 2), 'int32')

    while np.max(d_max_along_shift) > threshold_d:
        print(np.max(d_max_along_shift))
        max_d = np.max(d_max_along_shift, 1)
        peaks = argrelmax(max_d)[0]
        idx_good = peaks[np.argmax(
            max_d[peaks[:, np.newaxis] + np.arange(-2*R,2*R+1)],1) == (2*R)]
        spike_time = idx_good[max_d[idx_good] > threshold_d]
        template_id, max_shift = np.unravel_index(
            np.argmax(np.reshape(d_matrix[spike_time],
                                 (spike_time.shape[0], -1)),1),
            [n_templates, n_shifts])

        rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
        rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
        d_matrix[rf_area, rf_area_t] = -np.Inf
        d_max_along_shift[rf_area, rf_area_t] = -np.Inf

        for j in range(spike_time.shape[0]):
            t_neigh, k_neigh = np.where(
                d_max_along_shift[spike_time[j]-2*R:spike_time[j]+2*R] > -np.Inf)
            t_neigh_abs = spike_time[j] + t_neigh - 2*R
            d_matrix[t_neigh_abs, k_neigh]
            d_matrix[t_neigh_abs, k_neigh] -= 2*temp_temp[
                template_id[j], k_neigh, max_shift[j], :, t_neigh]
            d_max_along_shift[t_neigh_abs, k_neigh] = np.max(
                d_matrix[t_neigh_abs, k_neigh], 1)

        spike_train_temp = np.hstack((spike_time[:, np.newaxis],
                                      template_id[:, np.newaxis]))
        spike_train = np.concatenate((spike_train, spike_train_temp), 0)         

    return spike_train


def fix_indexes(spike_train, idx_local, idx, buffer_size):
    """Fixes indexes from detected spikes in batches

    Parameters
    ----------
    res: tuple
        A tuple with the results from the nnet detector
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size
    """

    # get limits for the data (exlude indexes that have buffer data)
    data_start = idx_local[0].start
    data_end = idx_local[0].stop
    # get offset that will be applied
    offset = idx[0].start

    # fix clear spikes
    spike_times = spike_train[:, 0]
    # get only observations outside the buffer
    train_not_in_buffer = spike_train[np.logical_and(spike_times >= data_start,
                                                     spike_times <= data_end)]
    # offset spikes depending on the absolute location
    train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + offset
                                 - buffer_size)

    return train_not_in_buffer
