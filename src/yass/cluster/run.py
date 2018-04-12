import logging
import datetime

from yass import read_config
from yass.geometry import make_channel_index
from yass.cluster.list import make_list
from yass.cluster.standardize import standardize
from yass.cluster.subsample import random_subsample
from yass.cluster.triage import triage
from yass.cluster.coreset import coreset
from yass.cluster.mask import getmask
from yass.cluster.util import run_cluster, run_cluster_location
from yass.cluster.merge import merge_units


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

    ##########
    # Triage #
    ##########

    _b = datetime.datetime.now()
    logger.info("Triaging...")
    print(scores.shape)
    scores, spike_index = random_subsample(scores, spike_index,
                                       CONFIG.cluster.max_n_spikes)
    scores, spike_index = triage(scores, spike_index,
                                CONFIG.cluster.triage.nearest_neighbors,
                                CONFIG.cluster.triage.percent)
    print(scores.shape)
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    if CONFIG.cluster.method == 'location':
        
        ##############
        # Clustering #
        ##############
        _b = datetime.datetime.now()
        logger.info("Clustering...")
        vbParam, tmp_loc, score, spike_index = run_cluster_location(
            scores, spike_index, CONFIG)
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

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
        spike_train, _, _ = run_cluster(scores, masks, groups,
                                        spike_index, 
                                        CONFIG.channel_groups,
                                        channel_index,
                                        CONFIG.detect.temporal_features,
                                        CONFIG)
        vbParam = 0
        tmp_loc = 0
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime - startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))

    return vbParam, tmp_loc, score, spike_index
