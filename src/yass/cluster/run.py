import logging
import datetime

from yass import read_config
from yass.geometry import make_channel_index
from yass.cluster.list import make_list
from yass.cluster.subsample import random_subsample
from yass.cluster.triage import triage
from yass.cluster.coreset import coreset
from yass.cluster.mask import getmask
from yass.cluster.util import run_cluster, run_cluster_location, recover_clear_spikes
import os
import numpy as np

def run(scores, spike_index):
    """Spike clustering

    Parameters
    ----------
    score: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    Returns
    -------
    spike_train: (TODO add documentation)

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/cluster.py

    """
    CONFIG = read_config()

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    # transform data structure of scores and spike_index
    scores, spike_index = make_list(scores, spike_index,
                                    CONFIG.recordings.n_channels)

    ##########
    # Triage #
    ##########

    _b = datetime.datetime.now()
    logger.info("Triaging...")

    path_to_triage_score = CONFIG.data.root_folder+ 'tmp/triage_score.npy'
    path_to_triage_spike_index = CONFIG.data.root_folder+ 'tmp/triage_spike_index.npy'
    if os.path.exists(path_to_triage_score):
        score_triage = np.load(path_to_triage_score)
        spike_index_triage = np.load(path_to_triage_spike_index)
    else:
        score_triage, spike_index_triage = triage(scores, spike_index, CONFIG.cluster.triage.nearest_neighbors, CONFIG.cluster.triage.percent)
        np.save(path_to_triage_score, score_triage)
        np.save(path_to_triage_spike_index,spike_index_triage)


    logger.info("Randomly subsampling...")
    scores_subsampled, spike_index_subsampled, scores_excluded, spike_index_excluded = random_subsample(score_triage, spike_index_triage,
                                           CONFIG.cluster.max_n_spikes)
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    if CONFIG.cluster.method == 'location':
        ##############
        # Clustering #
        ##############
        _b = datetime.datetime.now()
        logger.info("Clustering...")
        spike_train, outlier_vbPar = run_cluster_location(scores_subsampled, spike_index_subsampled, CONFIG)
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

        spike_train_excluded = recover_clear_spikes(scores_excluded, spike_index_excluded, outlier_vbPar)
        print (spike_train_excluded.shape)
	    
        #Concatenate spike_train and spike_train_excluded
        spike_train = np.concatenate([spike_train, spike_train_excluded],axis=0)
        print (spike_train.shape)
    
    else:
        ###########
        # Coreset #
        ###########
        _b = datetime.datetime.now()
        logger.info("Coresetting...")
        groups = coreset(scores,
                         CONFIG.cluster.coreset.clusters,
                         CONFIG.cluster.coreset.threshold)
        Time['c'] += (datetime.datetime.now() - _b).total_seconds()

        ###########
        # Masking #
        ###########
        _b = datetime.datetime.now()
        logger.info("Masking...")
        masks = getmask(scores, groups,
                        CONFIG.cluster.masking_threshold)
        Time['m'] += (datetime.datetime.now() - _b).total_seconds()

        ##############
        # Clustering #
        ##############
        _b = datetime.datetime.now()
        logger.info("Clustering...")
        channel_index = make_channel_index(CONFIG.neigh_channels,
                                           CONFIG.geom)
        spike_train = run_cluster(scores, masks, groups,
                                  spike_index, CONFIG.channel_groups,
                                  channel_index,
                                  CONFIG.detect.temporal_features,
                                  CONFIG)
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime - startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))

    return spike_train
