import os.path
import logging

import numpy as np

from yass.deconvolute.util import svd_shifted_templates, calculate_temp_temp, small_shift_templates, make_spt_list, clean_up
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

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index.shape: {}'
                  .format(templates.shape, spike_index.shape))

    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates,(2,1,0))

    n_shifts = CONFIG.deconvolution.upsample_factor
    n_explore = CONFIG.deconvolution.n_explore
    threshold_d = CONFIG.deconvolution.threshold_dd
    n_features = CONFIG.deconvolution.n_features
    max_spikes = CONFIG.deconvolution.max_spikes

    # get spike_index as list
    spt_list = make_spt_list(spike_index, n_channels)    
    spike_index = 0
    print('make list done')
 
    # upsample template
    shifted_templates = small_shift_templates(templates, n_shifts)

    print('shifting done')
    # svd templates
    temporal_features, spatial_features = svd_shifted_templates(shifted_templates, n_features)

    print('svd done')

    # calculate convolution of pairwise templates
    temp_temp = calculate_temp_temp(temporal_features, spatial_features)
    temp_temp *= 2

    # run nn preprocess batch-wsie
    recording_path = os.path.join(CONFIG.data.root_folder,
                                  output_directory,
                                  recordings_filename)
    bp = BatchProcessor(recording_path,
                        max_memory=CONFIG.resources.max_memory,
                        buffer_size=2*n_temporal_big)
    mc = bp.multi_channel_apply
    res = mc(
        deconvolve,
        mode='memory',
        cleanup_function=fix_indexes,
        pass_batch_info=True,
        
	spt_list=spt_list,
        shifted_templates=shifted_templates,
        temporal_features=temporal_features,
        spatial_features=spatial_features,
        temp_temp=temp_temp,
        n_explore=n_explore,
        threshold_d=threshold_d
	
    )   

    spike_train = np.concatenate([element for element in res], axis=0)

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    # sort spikes by time
    spike_train, templates = clean_up(spike_train, templates, max_spikes)

    return spike_train, np.transpose(templates)
