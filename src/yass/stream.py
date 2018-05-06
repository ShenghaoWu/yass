# Functions to process data in streaming mode using previously computed templates

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
from yass.preprocess.filter import filter_standardize_stream
from yass.geometry import (n_steps_neigh_channels, make_channel_index)
from yass.util import file_loader, save_numpy_object

class Stream:
    ''' Class that processes mini-chunks of data 
        Input: computed teamplates and scores from a firstbatch of data
        Function: preprocess, detect and match spikes to templates/scores
                  from firstbatch
        Note: chunk of data limited to maximum size that fits into GPU
        TODO:   asynchronous processing with a barrier between GPU-detect 
                and CPU steps
    '''
    
    
    def __init__(self, templates, scores, output_directory):
        
        self.templates = templates
        self.scores = scores
        self.output_directory = output_directory
        

    def run(self): 
        
        """ Initilize network detection and autoencoder dimensionality reduction
            - do it once then run all data through

        """
    
        logger = logging.getLogger(__name__)
        
        CONFIG = read_config()
        TMP_FOLDER = os.path.join(CONFIG.data.root_folder, 
                                                self.output_directory)

        # check if all scores, clear and collision spikes exist..
        max_memory = (CONFIG.resources.max_memory_gpu if GPU_ENABLED else
                      CONFIG.resources.max_memory)

        # make tensorflow tensors and neural net classes
        detection_th = CONFIG.detect.neural_network_detector.threshold_spike
        triage_th = CONFIG.detect.neural_network_triage.threshold_collision
        detection_fname = CONFIG.detect.neural_network_detector.filename
        ae_fname = CONFIG.detect.neural_network_autoencoder.filename
        triage_fname = CONFIG.detect.neural_network_triage.filename
        n_channels = CONFIG.recordings.n_channels
        channel_index = make_channel_index(CONFIG.neigh_channels, 
                                                        CONFIG.geom, 2)      
        
        
        whiten_filter = np.load(os.path.join(CONFIG.data.root_folder, 
                                        self.output_directory,'whitening.npy'))
        # prepare nn 
        (x_tf, output_tf, NND,
         NNAE, NNT) = neuralnetwork.prepare_nn(channel_index,
                                               whiten_filter,
                                               detection_th,
                                               triage_th,
                                               detection_fname,
                                               ae_fname,
                                               triage_fname)

        # compute neighbor channel distances (?)
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
        
        # make indexes; read config params
        n_channels = CONFIG.recordings.n_channels

        # compute len of recording
        filename_dat = os.path.join(CONFIG.data.root_folder,
                                    CONFIG.data.recordings)
        fp = np.memmap(filename_dat, dtype='int16', mode='r')
        fp_len = fp.shape[0]

        # compute batch indexes
        buffer_size = 200
        chunk_len = 20000
        indexes = np.arange(0, fp_len / n_channels, chunk_len)
        if indexes[-1] != fp_len / n_channels:
            indexes = np.hstack((indexes, fp_len / n_channels))

        idx_list = []
        for k in range(len(indexes) - 1):
            idx_list.append([
                indexes[k], indexes[k + 1], buffer_size,
                indexes[k + 1] - indexes[k] + buffer_size
            ])

        idx_list = np.int64(np.vstack(idx_list))
        proc_indexes = np.arange(len(idx_list))

        print idx_list
        
        spikes_clear_list = []
        spikes_all_list = []
        for ctr, idx in enumerate(idx_list): 
            logger.info('processing index: %s/%s', str(ctr), str(len(idx_list)))
            
            # read and preprocess chunk
            standardized_recording = filter_standardize_stream(
                            idx,
                            CONFIG.preprocess.filter.low_pass_freq,
                            CONFIG.preprocess.filter.high_factor,
                            CONFIG.preprocess.filter.order,
                            CONFIG.recordings.sampling_rate,
                            buffer_size,
                            filename_dat,
                            n_channels,
                            CONFIG.data.root_folder,
                            self.output_directory)
            
            # run nn over chunk
            res = neuralnetwork.run_detect_triage_featurize(
                            standardized_recording,
                            x_tf, 
                            output_tf, 
                            NND, NNAE, NNT, neighbors)
            
            # clean up nn output
            spikes_clear, spikes_all, scores = cleanup_nn(
                                            standardized_recording, res,
                                            buffer_size,chunk_len, idx,
                                            CONFIG)
            spikes_all_list.append(spikes_all)
            
            # spikematch: Peter and Cat to write
            spikes_clear_postmatch = spike_match(spikes_clear, scores, 
                                                 self.templates)
            spikes_clear_list.append(spikes_clear_postmatch)
            
        clear = np.vstack(spikes_clear_list)
        spikes_all = np.vstack(spikes_all_list)

        np.save(os.path.join(TMP_FOLDER,'stream_spikes_clear.npy'),clear)
        np.save(os.path.join(TMP_FOLDER,'stream_spikes_all.npy'),spikes_all)
        


def cleanup_nn(standardized_recording, res, buffer_size,chunk_len, idx, 
                                                                CONFIG):
        
    # modified fix index file
    res = fix_indexes_stream(res, buffer_size, chunk_len ,idx[0])

    # get clear spikes
    clear = res[1] 
    _n_observations = standardized_recording.shape[0]
    clear, idx = detect.remove_incomplete_waveforms(
        clear, CONFIG.spike_size + CONFIG.templates.max_shift,
        _n_observations)

    # get scores for clear spikes
    scores = res[0][idx]

    # get and clean all spikes
    spikes_all = res[2] 
    spikes_all, _ = detect.remove_incomplete_waveforms(
        spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
        _n_observations)

    return clear, spikes_all, scores


def spike_match(spikes_clear, scores, templates):
    
    print (" Peter/Cat to write spike_match function ")
    
    return []


def fix_indexes_stream(res, buffer_size, chunk_len, offset):
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

