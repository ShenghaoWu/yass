from scipy.interpolate import interp1d
import numpy as np

def calculate_temp_temp(shifted_templates, channel_index, waveform_size_small):

    n_shifts, n_channels, waveform_size, n_templates = shifted_templates.shape
    n_neigh = channel_index.shape[1]
    principal_channels = np.argmax(np.max(np.abs(shifted_templates), (0,2)), 0)

    mid_point = int((waveform_size-1)/2)
    R_small = int((waveform_size_small-1)/2)
    temp_temp = np.zeros((n_templates, n_templates, n_shifts, n_shifts,
                          waveform_size+waveform_size_small-1), 'float32')

    for s1 in range(n_shifts):
        for s2 in range(n_shifts):
            for k in range(n_templates):
                ch_idx = channel_index[principal_channels[k]]
                ch_idx = ch_idx[ch_idx < n_channels]
                temp_ = shifted_templates[:, ch_idx]
                temp_k = np.flip(temp_[s1, :, mid_point-R_small:mid_point+R_small+1, k], 1)

                for k2 in range(n_templates):
                    for j in range(ch_idx.shape[0]):
                        temp_temp[k, k2, s1, s2] += np.convolve(temp_[s2, j, :, k2], temp_k[j])

    return temp_temp


def upsample_templates(templates, upsample_factor):
    # get shapes
    n_channels, waveform_size, n_templates = templates.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size-1, num=waveform_size, endpoint=True)
    shifts = np.linspace(-0.5, 0.5, upsample_factor, endpoint=False)
    xnew = np.sort(np.reshape(x[:, np.newaxis] + shifts, -1))

    upsampled_templates = np.zeros((n_channels, waveform_size*upsample_factor, n_templates))
    for j in range(n_templates):
        ff = interp1d(x, templates[:, :, j], kind='cubic')
        idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size-1)
        upsampled_templates[:, idx_good, j] = ff(xnew[idx_good])

    return upsampled_templates

def make_spike_index_per_template(spike_index, templates, n_explore):

    n_channels, n_temporal_big, n_templates = templates.shape
    
    principal_channels = np.argmax(np.max(np.abs(templates), 1), 0)
    
    spt_list = make_spt_list(spike_index, n_channels)
    
    for c in range(n_channels):
        spt_list[c] = get_longer_spt_list(spt_list[c], n_explore)
    
    spike_index_template = np.zeros((0, 2), 'int32')
    template_id = np.zeros(0, 'int32')
    for k in range(n_templates):
        
        mainc = principal_channels[k]
        spt = spt_list[mainc]

        spike_index_template = np.concatenate((
            spike_index_template,
            np.concatenate((spt[:, np.newaxis],
                            np.ones((spt.shape[0], 1), 'int32')*mainc),
                           1)), 0)
        template_id = np.hstack((template_id,
                                 np.ones((spt.shape[0]), 'int32')*k))
        
    idx_sort = np.argsort(spike_index_template[:, 0])
    
    
    return spike_index_template[idx_sort], template_id[idx_sort]
    

def get_smaller_shifted_templates(shifted_templates, channel_index,
                                  principal_channels,
                                  spike_size):
    
    n_shifts, n_channels, waveform_size, n_templates = shifted_templates.shape
    n_neigh = channel_index.shape[1]

    shifted_templates = np.transpose(shifted_templates, (3, 2, 1, 0))
    shifted_templates = np.concatenate(
        (shifted_templates,np.zeros((n_templates,
                                     waveform_size,
                                     1,
                                     n_shifts))), 2)

    mid_t = int((waveform_size-1)/2)
    templates_small = np.zeros((n_templates, 2*spike_size+1, n_neigh, n_shifts))
    for k in range(n_templates):
        mainc = principal_channels[k]
        temp = shifted_templates[k, mid_t-spike_size: mid_t+spike_size+1]
        templates_small[k] = temp[:, channel_index[mainc]]
        
    return templates_small

def small_shift_templates(templates, n_shifts=5):
    """
    get n_shifts number of shifted templates.
    The amount of shift is evenly distributed from
    -0.5 to 0.5 timebin including -0.5 and 0.5.

    Parameters
    ----------

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
       A 2D array of a template

    n_shifts: int
       number of shifted templates to make

    Returns
    -------
    shifted_templates: numpy.ndarray (n_shifts, n_channels,
                                      waveform_size)
        A 3D array with shifted templates
    """

    # get shapes
    n_channels, waveform_size, n_templates = templates.shape

    # upsample using cubic interpolation
    x = np.linspace(0, waveform_size-1, num=waveform_size, endpoint=True)
    shifts = np.linspace(-0.5, 0.5, n_shifts, endpoint=False)

    shifted_templates = np.zeros((n_shifts, n_channels, waveform_size, n_templates))    
    for k in range(n_templates):
        ff = interp1d(x, templates[:, :, k], kind='cubic')

        # get shifted templates
        for j in range(n_shifts):
            xnew = x - shifts[j]
            idx_good = np.logical_and(xnew >= 0, xnew <= waveform_size-1)
            shifted_templates[j][:, idx_good, k] = ff(xnew[idx_good])

    return shifted_templates


def make_spt_list(spike_index, n_channels):
    """
    Change a data structure of spike_index from an array of two
    columns to a list

    Parameters
    ----------

    spike_index: numpy.ndarray (n_spikes, 2)
       A 2D array containing spikes information with two columns,
       where the first column is spike time and the second is channel.

    n_channels: int
       the number of channels in recording (or length of output list)

    Returns
    -------
    spike_index_list: list (n_channels)
        A list such that spike_index_list[c] cointains all spike times
        whose channel is c
    """

    spike_index_list = [None]*n_channels

    for c in range(n_channels):
        spike_index_list[c] = spike_index[spike_index[:, 1] == c, 0]

    return spike_index_list


def get_longer_spt_list(spt, n_explore):
    """
    Given a spike time, -n_explore to n_explore time points
    around the spike time is also included as spike times

    Parameters
    ----------

    spt: numpy.ndarray (n_spikes)
       A list of spike times

    n_explore: int
       2*n_explore additional points will be included into spt

    Returns
    -------
    spt_long: numpy.ndarray
        A new list containing additions spike times
    """

    # sort spike time
    spt = np.sort(spt)

    # add -n_explore to n_explore points around each spike time
    all_spikes = np.reshape(np.add(spt[:, np.newaxis],
                                   np.arange(-n_explore, n_explore+1)
                                   [np.newaxis, :]), -1)

    # if there are any duplicate remove it
    spt_long = np.sort(np.unique(all_spikes))

    return spt_long
