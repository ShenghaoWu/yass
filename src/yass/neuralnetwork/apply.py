import tensorflow as tf
import numpy as np
import datetime

def run_detect_triage_featurize(recordings, x_tf, output_tf, NND, NNAE, NNT, neighbors):
    """Detect spikes using a neural network

    Parameters
    ----------
    recordings: numpy.ndarray (n_observations, n_channels)
        Neural recordings

    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    output_tf: tuple of tf.tensors
        a tuple of tensorflow tensors that produce score, spike_index_clear,
        and spike_index_collision

    Returns
    -------
    scores: numpy.ndarray (n_clear_spikes, n_features, n_neigh)
        3D array with the scores for the clear spikes, first dimension is
        the number of spikes, second is the nymber of features and third the
        number of neighboring channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    """

    # get values of above tensors
    with tf.Session() as sess:
        NND.saver.restore(sess, NND.path_to_detector_model)
        NNAE.saver_ae.restore(sess, NNAE.path_to_ae_model)
        NNT.saver.restore(sess, NNT.path_to_triage_model)

        score, spike_index_clear, spike_index_collision = sess.run(
            output_tf, feed_dict={x_tf: recordings})

    return (score, spike_index_clear, spike_index_collision)


def run_detect_triage_featurize2(recordings, x_tf, output_tf, NND, NNAE, NNT, neighbors):
    """Detect spikes using a neural network

    Parameters
    ----------
    recordings: numpy.ndarray (n_observations, n_channels)
        Neural recordings

    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    output_tf: tuple of tf.tensors
        a tuple of tensorflow tensors that produce score, spike_index_clear,
        and spike_index_collision

    Returns
    -------
    scores: numpy.ndarray (n_clear_spikes, n_features, n_neigh)
        3D array with the scores for the clear spikes, first dimension is
        the number of spikes, second is the nymber of features and third the
        number of neighboring channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    """

    # get values of above tensors
    with tf.Session() as sess:
        NND.saver.restore(sess, NND.path_to_detector_model)
        NNAE.saver_ae.restore(sess, NNAE.path_to_ae_model)
        NNT.saver.restore(sess, NNT.path_to_triage_model)

        _b = datetime.datetime.now()
        score, spike_index, idx_clean = sess.run(
            output_tf, feed_dict={x_tf: recordings})
        print((datetime.datetime.now()-_b).total_seconds())
        
        _b = datetime.datetime.now()
        rot = NNAE.load_rotation()
        energy = np.ptp(np.matmul(score[:, :, 0], rot.T), axis=1)
        print((datetime.datetime.now()-_b).total_seconds())
        
        T, C = recordings.shape
        #neighbors = neighbors - np.eye(C)
        killed = remove_axons(spike_index, energy, neighbors, T, C)
        print((datetime.datetime.now()-_b).total_seconds())
        
        idx_keep = np.logical_and(~killed, idx_clean)
        score_clear = score[idx_keep]
        spike_index_clear = spike_index[idx_keep]
        
    return (score_clear, spike_index_clear, spike_index)

def remove_axons(spike_index, energy, neighbors, T, C):
    
    n_data = spike_index.shape[0]
    
    temp = np.ones((T, C), 'int32')*-1
    temp[spike_index[:, 0], spike_index[:, 1]] = np.arange(n_data)
    
    kill = np.zeros(n_data, 'bool')
    energy_killed = np.zeros(n_data, 'float32')
    search_order = np.argsort(energy)[::-1]

    for j in range(n_data):
        kill, energy_killed = kill_spikes(temp, neighbors, spike_index, energy, kill,
                                          energy_killed, search_order[j])

    return kill

def kill_spikes(temp, neighbors, spike_index, energy, kill, energy_killed, current_idx):
        
    tt, cc = spike_index[current_idx]
    energy_threshold = max(energy_killed[current_idx], energy[current_idx])
    ch_idx = np.where(neighbors[cc])[0]
    w = 5
    indices = temp[tt-w:tt+w+1, ch_idx].ravel()
    indices = indices[indices > -1]
 
    for j in indices:
        if energy[j] < energy_threshold:
            kill[j] = 1
            energy_killed[j] = energy_threshold

    return kill, energy_killed



def fix_indexes(res, idx_local, idx, buffer_size):
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
    score, clear, collision = res

    # get limits for the data (exlude indexes that have buffer data)
    data_start = idx_local[0].start
    data_end = idx_local[0].stop
    # get offset that will be applied
    offset = idx[0].start

    # fix clear spikes
    clear_times = clear[:, 0]
    # get only observations outside the buffer
    idx_not_in_buffer = np.logical_and(clear_times >= data_start,
                                       clear_times <= data_end)
    clear_not_in_buffer = clear[idx_not_in_buffer]
    score_not_in_buffer = score[idx_not_in_buffer]

    # offset spikes depending on the absolute location
    clear_not_in_buffer[:, 0] = (clear_not_in_buffer[:, 0] + offset
                                 - buffer_size)

    # fix collided spikes
    col_times = collision[:, 0]
    # get only observations outside the buffer
    col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
                                                 col_times <= data_end)]
    # offset spikes depending on the absolute location
    col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size

    return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer
