"""
Tests for functions that create training data for the neural networks
"""
import os.path as path

import numpy as np

import yass
from yass.augment import make_training_data
from yass.templates.util import get_templates
from yass.augment.crop import crop_and_align_templates


spike_train = np.array([100, 0,
                        150, 0,
                        200, 1,
                        250, 1,
                        300, 2,
                        350, 2]).reshape(-1, 2)

chosen_templates = [0, 1, 2]
min_amplitude = 2
n_spikes_to_make = 500

filters = [8, 4]


def test_can_make_training_data(path_to_tests, path_to_data_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    make_training_data(CONFIG, spike_train, chosen_templates,
                       min_amplitude, n_spikes_to_make,
                       data_folder=path_to_data_folder)


def test_can_crop_and_align_templates(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    crop_and_align_templates(templates_uncropped,
                             CONFIG.spike_size,
                             CONFIG.neigh_channels, CONFIG.geom)


def test_can_make_clean():
    pass


def test_can_make_collided():
    pass


def test_can_make_misaligned():
    pass


def test_can_make_noise():
    pass
