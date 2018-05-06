"""
Detection pipeline
"""
import logging
import os.path
from functools import reduce

import numpy as np

from yass import read_config, GPU_ENABLED
from yass.batch import BatchProcessor, RecordingsReader
from yass.threshold.detect import threshold
from yass.threshold import detect
from yass.threshold.dimensionality_reduction import pca
from yass import neuralnetwork
from yass.preprocess import whiten
from yass.geometry import n_steps_neigh_channels
from yass.util import file_loader, save_numpy_object


def run(standardized_recording, standarized_path, standarized_params,
        channel_index, whiten_filter, output_directory='tmp/',
        if_file_exists='skip', save_results=False):
    """Execute detect step

    Parameters
    ----------
    standarized_path: str or pathlib.Path
        Path to standarized data binary file

    standarized_params: dict, str or pathlib.Path
        Dictionary with standarized data parameters or path to a yaml file

    channel_index: numpy.ndarray, str or pathlib.Path
        Channel index or path to a npy file

    whiten_filter: numpy.ndarray, str or pathlib.Path
        Whiten matrix or path to a npy file

    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for every
      generated file. If 'overwrite' it replaces the files if any exist,
      if 'abort' it raises a ValueError exception if any file exists,
      if 'skip' if skips the operation if any file exists

    save_results: bool, optional
        Whether to save results to disk, defaults to False

    Returns
    -------
    clear_scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_call: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for all spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/ (if save_results is
    True):

    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_all.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``scores_clear.npy`` - Scores for clear spikes

    Threshold detector runs on CPU, neural network detector runs CPU and GPU,
    depending on how tensorflow is configured.

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    CONFIG = read_config()

    # load files in case they are strings or Path objects
    standarized_params = file_loader(standarized_params)
    channel_index = file_loader(channel_index)
    whiten_filter = file_loader(whiten_filter)


    return run_neural_network(standardized_recording,
                              standarized_path,
                              standarized_params,
                              channel_index,
                              whiten_filter,
                              output_directory,
                              if_file_exists,
                              save_results)

def run_neural_network(standardized_recording, standarized_path, 
                       standarized_params, channel_index, whiten_filter, 
                       output_directory, if_file_exists, save_results):
                           
    """Run neural network detection and autoencoder dimensionality reduction

    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

    # check if all scores, clear and collision spikes exist..
    path_to_score = os.path.join(TMP_FOLDER, 'scores_clear.npy')
    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')
    path_to_spike_index_all = os.path.join(TMP_FOLDER, 'spike_index_all.npy')
    path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')

    paths = [path_to_score, path_to_spike_index_clear, path_to_spike_index_all]
    exists = [os.path.exists(p) for p in paths]

    max_memory = (CONFIG.resources.max_memory_gpu if GPU_ENABLED else
                  CONFIG.resources.max_memory)

    # make tensorflow tensors and neural net classes
    print (" making tensorflow tensors and neural net classes ") 
    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename
    n_channels = CONFIG.recordings.n_channels
  
    # prepare nn 
    (x_tf, output_tf, NND,
     NNAE, NNT) = neuralnetwork.prepare_nn(channel_index,
                                           whiten_filter,
                                           detection_th,
                                           triage_th,
                                           detection_fname,
                                           ae_fname,
                                           triage_fname)

    # run nn preprocess batch-wsie
    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
    
    # Cat: list make is not pythonic; can be updated, 
    # making list of indexes for batch processing of detect step
    rec_len= standardized_recording.shape[0]
    
    chunk_len = 20000      # Cat: this is 20sec for 49 channel data; to detect automatically
    chunks = np.arange(0,rec_len, chunk_len)
    indexes = []
    for k in range(len(chunks[:-1])):
        indexes.append([chunks[k],chunks[k+1]])
    if indexes[-1][1]!= rec_len:
        indexes.append([indexes[-1][1],rec_len])
    
    scores_list = []
    clear_spikes_list = []
    spikes_all_list = []
    for ctr, index in enumerate(indexes): 
        logger.info('processing index: %s  %s/%s', str(index), str(ctr), str(len(indexes)))
        
        # buffer data going into neural network: 
        buffer_size = 200   # Cat: to set this in CONFIG file
        # read chunk plus buffer at end
        if ctr==0: 
            data_temp = standardized_recording[index[0]:index[1]+buffer_size] 
            data_temp = np.vstack((np.zeros((buffer_size,n_channels),'float32'),data_temp))
        
        # read chunk plus buffer at end
        # TODO: Cat: may wish to check for end of file and pad
        else: 
            data_temp = standardized_recording[index[0]-buffer_size:index[1]+buffer_size]  # read chunk plus buffer at end
        
        print data_temp.shape
        
        res = neuralnetwork.run_detect_triage_featurize(data_temp,
                        x_tf, output_tf, NND, NNAE, NNT, neighbors)
        
        print (" Spikes: ", res[2].shape)


        # modified fix index file
        res = fix_indexes_firstbatch(res, buffer_size, chunk_len ,index[0])

        # get clear spikes
        clear = res[1] 
        #logger.info('Removing clear indexes outside the allowed range to '
        #            'draw a complete waveform...')
        _n_observations = standardized_recording.shape[0]
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spike_size + CONFIG.templates.max_shift,
            _n_observations)

        # append 
        clear_spikes_list.append(clear)

        # get scores for clear spikes
        scores = res[0][idx]
        scores_list.append(scores)

        # get and clean all spikes
        spikes_all = res[2] 
        #logger.info('Removing all indexes outside the allowed range to '
        #            'draw a complete waveform...')
        spikes_all, _ = detect.remove_incomplete_waveforms(
            spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
            _n_observations)

        spikes_all_list.append(spikes_all)

    scores = np.vstack(scores_list)
    clear = np.vstack(clear_spikes_list)
    spikes_all = np.vstack(spikes_all_list)

    np.save(os.path.join(TMP_FOLDER,'scores_clear.npy'),scores)
    np.save(os.path.join(TMP_FOLDER,'spikes_clear.npy'),clear)
    np.save(os.path.join(TMP_FOLDER,'spikes_all.npy'),spikes_all)

    return scores, clear, spikes_all

def fix_indexes_firstbatch(res, buffer_size, chunk_len, offset):
    """Fixes indexes from the first batch; 

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
    data_start = buffer_size
    data_end = buffer_size + chunk_len

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




def get_locations_features(scores, rotation, main_channel,
                           channel_index, channel_geometry,
                           threshold):

    n_data, n_features, n_neigh = scores.shape

    reshaped_score = np.reshape(np.transpose(scores, [0, 2, 1]),
                                [n_data*n_neigh, n_features])
    energy = np.reshape(np.ptp(np.matmul(
        reshaped_score, rotation.T), 1), (n_data, n_neigh))

    energy = np.piecewise(energy, [energy < threshold,
                                   energy >= threshold],
                          [0, lambda x:x-threshold])

    channel_index_per_data = channel_index[main_channel, :]
    channel_geometry = np.vstack((channel_geometry, np.zeros((1, 2), 'int32')))
    channel_locations_all = channel_geometry[channel_index_per_data]

    xy = np.divide(np.sum(np.multiply(energy[:, :, np.newaxis],
                                      channel_locations_all), axis=1),
                   np.sum(energy, axis=1, keepdims=True))
    noise = np.random.randn(xy.shape[0], xy.shape[1])*(0.00001)
    xy += noise

    scores = np.concatenate((xy, scores[:, :, 0]), 1)

    if scores.shape[0] != n_data:
        raise ValueError('Number of clear spikes changed from {} to {}'
                         .format(n_data, scores.shape[0]))

    if scores.shape[1] != (n_features+channel_geometry.shape[1]):
        raise ValueError('There are {} shape features and {} location features'
                         'but {} features are created'.
                         format(n_features,
                                channel_geometry.shape[1],
                                scores.shape[1]))

    return scores[:, :, np.newaxis]


def get_locations_features_threshold(scores, main_channel,
                                     channel_index, channel_geometry):

    n_data, n_features, n_neigh = scores.shape

    energy = np.linalg.norm(scores, axis=1)

    channel_index_per_data = channel_index[main_channel, :]

    channel_geometry = np.vstack((channel_geometry, np.zeros((1, 2), 'int32')))
    channel_locations_all = channel_geometry[channel_index_per_data]
    xy = np.divide(np.sum(np.multiply(energy[:, :, np.newaxis],
                                      channel_locations_all), axis=1),
                   np.sum(energy, axis=1, keepdims=True))
    scores = np.concatenate((xy, scores[:, :, 0]), 1)

    if scores.shape[0] != n_data:
        raise ValueError('Number of clear spikes changed from {} to {}'
                         .format(n_data, scores.shape[0]))

    if scores.shape[1] != (n_features+channel_geometry.shape[1]):
        raise ValueError('There are {} shape features and {} location features'
                         'but {} features are created'
                         .format(n_features,
                                 channel_geometry.shape[1],
                                 scores.shape[1]))

    scores = np.divide((scores - np.mean(scores, axis=0, keepdims=True)),
                       np.std(scores, axis=0, keepdims=True))

    return scores[:, :, np.newaxis]



def remove_incomplete_waveforms(spike_index, spike_size, recordings_length):
    """

    Parameters
    ----------
    spikes: numpy.ndarray
        A 2D array of detected spikes as returned from detect.threshold

    Returns
    -------
    numpy.ndarray
        A new 2D array with some spikes removed. If the spike index is in a
        position (beginning or end of the recordings) where it is not possible
        to draw a complete waveform, it will be removed
    numpy.ndarray
        A boolean 1D array with True entries meaning that the index is within
        the valid range
    """
    max_index = recordings_length - 1 - spike_size
    min_index = spike_size
    include = np.logical_and(spike_index[:, 0] <= max_index,
                             spike_index[:, 0] >= min_index)
    return spike_index[include], include

