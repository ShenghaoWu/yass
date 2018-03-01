import os.path
import logging

import numpy as np

from yass.deconvolute.util import calculate_temp_temp, small_shift_templates, make_spt_list, get_smaller_shifted_templates 
from yass.deconvolute.deconvolve import deconvolve, fix_indexes
from yass.geometry import make_channel_index
from yass import read_config
from yass.batch import BatchProcessor


def run(spike_index, templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index: numpy.ndarray (n_data, 2)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    # read config file
    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'
                  .format(templates.shape, spike_index.shape))

    # channel index    
    channel_index = make_channel_index(CONFIG.neighChannels, CONFIG.geom)

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    #n_shifts = CONFIG.deconvolution.upsample_factor
    n_shifts = 3
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd

    # get spike_index as list
    spt_list = make_spt_list(spike_index, n_channels)    

    # upsample template
    shifted_templates = small_shift_templates(templates, n_shifts)

    # principal channels
    principal_channels = np.argmax(np.max(np.abs(shifted_templates), (0,2)), 0)

    # localize it
    templates_small = get_smaller_shifted_templates(shifted_templates,
                                                    channel_index,
                                                    principal_channels,
                                                    CONFIG.spikeSize)

    temp_temp = calculate_temp_temp(shifted_templates, channel_index,
                                    (2*CONFIG.spikeSize+1))

    # run nn preprocess batch-wsie
    recording_path = os.path.join(CONFIG.data.root_folder,
                                  output_directory,
                                  recordings_filename)
    bp = BatchProcessor(recording_path,
                        buffer_size=n_temporal_big)
    mc = bp.multi_channel_apply
    res = mc(
        deconvolve,
        mode='memory',
        cleanup_function=fix_indexes,
        pass_batch_info=True,
        channel_index=channel_index,
        spt_list=spt_list,
        shifted_templates=shifted_templates,
        templates_small=templates_small,
        principal_channels=principal_channels,
        n_explore=n_explore,
        temp_temp=temp_temp,
        threshold_d=threshold_d
    )   

    spike_train = np.concatenate([element for element in res], axis=0)

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    # sort spikes by time
    spike_train = spike_train[np.argsort(spike_train[:, 0])]

    return spike_train
