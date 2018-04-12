"""
[Description of file content]
"""
import numpy as np
from scipy import sparse
import logging

from yass.batch import BatchProcessor

logger = logging.getLogger(__name__)


# TODO: remove this function and use the explorer directly
def get_templates(spike_train, path_to_recordings, max_memory, spike_size,
                  channel_index=None):
    logger.info('Computing templates...')

    # number of templates
    n_templates = np.max(spike_train[:, 1]) + 1

    # read recording
    bp = BatchProcessor(path_to_recordings,
                        max_memory=max_memory,
                        buffer_size=spike_size)

    # run nn preprocess batch-wsie
    mc = bp.multi_channel_apply
    res = mc(
        compute_weighted_templates,
        mode='memory',
        pass_batch_info=True,
        pass_batch_results=True,
        spike_train=spike_train,
        spike_size=spike_size,
        n_templates=n_templates,
        channel_index=channel_index)

    templates = res[0]
    weights = res[1]
    weights[weights == 0] = 1
    templates = templates/weights[np.newaxis, np.newaxis, :]

    return templates, weights


def compute_weighted_templates(recording, idx_local, idx, previous_batch,
                               spike_train, spike_size, n_templates,
                               channel_index):

    n_channels = recording.shape[1]

    # batch info
    data_start = idx[0].start
    data_end = idx[0].stop
    # get offset that will be applied
    offset = idx_local[0].start

    # shift location of spikes according to the batch info
    spike_time = spike_train[:, 0]
    spike_train = spike_train[np.logical_and(spike_time >= data_start,
                                             spike_time < data_end)]
    spike_train[:, 0] = spike_train[:, 0] - data_start + offset

    # calculate weight templates
    weighted_templates = np.zeros((n_templates, n_channels, 2*spike_size+1),
                                  dtype=np.float32)
    weights = np.zeros(n_templates)
    #ch_idx = np.arange(n_channels)
    #vectorized_rec = recording.ravel()
    
    if channel_index is None:
        for k in range(n_templates):
            spt = spike_train[spike_train[:, 1] == k, 0]
            n_spikes = spt.shape[0]
            if n_spikes > 0:
                #times = spt[:, np.newaxis] + np.arange(-spike_size, spike_size+1)
                #idx = (ch_idx + (times * n_channels).reshape(-1,1)).ravel()
                #weighted_templates[k] = np.sum(
                #    vectorized_rec[idx].reshape(
                #        times.size, ch_idx.size).reshape(
                #        n_spikes, -1 , n_channels), 0).T 

                weighted_templates[k] = np.sum(recording[
                    spt[:, np.newaxis] + np.arange(-spike_size, spike_size+1)], 0).T
                weights[k] = n_spikes
    else:
        vectorized_rec = recording.ravel()
        for k in range(n_templates):
            spt = spike_train[spike_train[:, 1] == k, 0]
            n_spikes = spt.shape[0]
            if n_spikes > 0:
                times = spt[:, np.newaxis] + np.arange(-spike_size, spike_size+1)
                ch_idx = np.where(channel_index[k])[0]

                idx = (ch_idx + (times * n_channels).reshape(-1,1)).ravel()
                weighted_templates[k, ch_idx] = np.sum(
                    vectorized_rec[idx].reshape(
                        times.size, ch_idx.size).reshape(
                        n_spikes, 2*spike_size+1, ch_idx.shape[0]), 0).T 
                
                #weighted_templates[k, ch_idx] = np.sum(recording[
                #    spt[:, np.newaxis] + np.arange(-spike_size, spike_size+1), ch_idx], 0).T
                weights[k] = n_spikes
        
    weighted_templates = np.transpose(weighted_templates, (1, 2, 0))

    if previous_batch is not None:
        weighted_templates += previous_batch[0]
        weights += previous_batch[1]

    return weighted_templates, weights


# TODO: remove this function and use the explorer directly
def get_templates2(spike_train, path_to_recordings,
                   max_memory, spike_size, channel_index=None):
    
    logger.info('Computing templates...')

    # number of templates
    n_templates = int(np.max(spike_train[:, 1]) + 1)

    # read recording
    bp = BatchProcessor(path_to_recordings,
                        max_memory=max_memory,
                        buffer_size=spike_size)

    # run nn preprocess batch-wsie
    mc = bp.multi_channel_apply
    res = mc(
        compute_weighted_templates2,
        mode='memory',
        pass_batch_info=True,
        pass_batch_results=True,
        spike_train=spike_train,
        spike_size=spike_size,
        n_templates=n_templates,
        channel_index=channel_index)

    templates = res[0]
    weights = res[1]
    weights[weights == 0] = 1
    templates = templates/weights[np.newaxis, np.newaxis, :]

    return templates, weights


# TODO: remove this function and use the explorer directly
def get_templates3(spike_train, path_to_recordings,
                   max_memory, spike_size, channel_index=None):
    
    logger.info('Computing templates...')

    # number of templates
    n_templates = int(np.max(spike_train[:, 1]) + 1)

    # read recording
    bp = BatchProcessor(path_to_recordings,
                        max_memory=max_memory,
                        buffer_size=spike_size)

    # run nn preprocess batch-wsie
    mc = bp.multi_channel_apply
    res = mc(
        compute_weighted_templates3,
        mode='memory',
        pass_batch_info=True,
        pass_batch_results=True)
    
    return 0


def compute_weighted_templates2(recording, idx_local, idx, previous_batch,
                                spike_train, spike_size, n_templates, 
                                channel_index):

    n_channels = recording.shape[1]

    # batch info
    data_start = idx[0].start
    data_end = idx[0].stop
    # get offset that will be applied
    offset = idx_local[0].start

    # shift location of spikes according to the batch info
    spike_time = spike_train[:, 0]
    spike_train = spike_train[np.logical_and(spike_time >= data_start,
                                             spike_time < data_end)]
    spike_train[:, 0] = spike_train[:, 0] - data_start + offset
    
    # calculate weight templates
    weighted_templates = np.zeros((n_templates, 2*spike_size+1, n_channels),
                                  dtype=np.float32)
    weights = np.zeros(n_templates)
    if channel_index is None:
        for k in range(n_templates):
            spt = spike_train[spike_train[:, 1] == k]
            n_spikes = spt.shape[0]
            if n_spikes > 0:
                weighted_templates[k] = np.average(
                    recording[spt[:, [0]].astype('int32')
                              + np.arange(-spike_size, spike_size+1)],
                    axis=0,
                    weights=spt[:, 2])
                weights[k] = np.sum(spt[:, 2])
                weighted_templates[k] *= weights[k]
    else:
        vectorized_rec = recording.ravel()
        for k in range(n_templates):
            spt = spike_train[spike_train[:, 1] == k]
            n_spikes = spt.shape[0]
            if n_spikes > 0:
                times = spt[:, [0]].astype('int32') + np.arange(-spike_size, spike_size+1)
                ch_idx = np.where(channel_index[k])[0]

                idx = (ch_idx + (times * n_channels).reshape(-1,1)).ravel()
                weighted_templates[k, :, ch_idx] = np.average(
                    vectorized_rec[idx].reshape(
                        times.size, ch_idx.size).reshape(
                        n_spikes, 2*spike_size+1, ch_idx.shape[0]),
                    axis=0,
                    weights=spt[:, 2]).T 
                weights[k] = np.sum(spt[:, 2])
                weighted_templates[k] *= weights[k]
        
    weighted_templates = np.transpose(weighted_templates, (2, 1, 0))

    if previous_batch is not None:
        weighted_templates += previous_batch[0]
        weights += previous_batch[1]

    return weighted_templates, weights


def compute_weighted_templates3(recording, idx_local, idx, previous_batch):

    return 0


# TODO: docs
def get_and_merge_templates(spike_train_clear, path_to_recordings, max_memory,
                            nearby_units, spike_size, template_max_shift,
                            t_merge_th):
    templates, weights = get_templates(spike_train_clear, path_to_recordings,
                                       max_memory,
                                       2*(spike_size + template_max_shift))

    templates = align_templates(templates, template_max_shift)

    #spike_train_clear, templates = mergeTemplates(templates, weights,
    #                                              spike_train_clear,
    #                                              nearby_units,
    #                                              template_max_shift,
    #                                              t_merge_th)
    templates = templates[:, template_max_shift:(
        template_max_shift+(4*spike_size+1))]

    return spike_train_clear, templates


def align_templates(templates, spike_train, max_shift):
    C, R, K = templates.shape
    spike_size = int((R-1)/2 - max_shift)

    # get main channel for each template
    mainc = np.argmax(
        np.max(templates[:, max_shift:(
            max_shift+2*spike_size+1)], axis=1), axis=0)

    # get templates on their main channel only
    templates_mainc = np.zeros((R, K))
    for k in range(K):
        templates_mainc[:, k] = templates[mainc[k], :, k]

    # reference template
    biggest_template_k = np.argmax(np.max(templates_mainc, axis=0))
    biggest_template = templates_mainc[
        max_shift:(max_shift + 2*spike_size+1), biggest_template_k]

    # find best shift
    fit_per_shift = np.zeros((2*max_shift+1, K))
    for s in range(2*max_shift+1):
        fit_per_shift[s] = np.matmul(
            biggest_template[
                np.newaxis, :], templates_mainc[s:(s+2*spike_size+1)])
    best_shift = np.argmax(fit_per_shift, axis=0)

    templates_final = np.zeros((C, 2*spike_size+1, K))
    for k in range(K):
        s = best_shift[k]
        templates_final[:, :, k] = templates[:, s:(s+2*spike_size+1), k]
        spike_train[spike_train[:, 1]==k, 0] += (s - max_shift)

    return templates_final, spike_train


# TODO: documentation
# TODO: comment code, it's not clear what it does
def mergeTemplates(templates, weights, spike_train, neighbors,
                   template_max_shift, t_merge_th):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, R, K = templates.shape
    th = t_merge_th
    W = template_max_shift

    energy = np.ptp(templates, 1)
    visible_channels = energy > 0.5
    main_channels = np.argmax(energy, 0)
    
    sparseConnection = sparse.lil_matrix((K, K), dtype='bool')
    for k1 in range(K):
        for k2 in range(k1, K):
            if neighbors[main_channels[k1], main_channels[k2]]:
                ch_idx = np.logical_or(visible_channels[:, k1],
                                        visible_channels[:, k2])
                t1 = templates[ch_idx, :, k1]
                t2 = templates[ch_idx, :, k2]
                if TemplatesSimilarity(t1, t2, th, W):
                    sparseConnection[k1, k2] = 1
                    sparseConnection[k2, k1] = 1
                               
    edges = {x: sparse.find(sparseConnection[x])[1] for x in range(K)}

    groups = list()
    for scc in strongly_connected_components_iterative(np.arange(K), edges):
        groups.append(np.array(list(scc)))

    Knew = len(groups)
    templatesNew = np.zeros((C, R, Knew))
    weightNew = np.zeros(Knew)
    spt_new = np.zeros(spike_train.shape[0], 'int32')
    id_new = np.zeros(spike_train.shape[0], 'int32')
    for k in range(Knew):
        temp = groups[k]
        templatesNew_temp = np.zeros((C, R, temp.shape[0]))
        weight_temp = np.zeros(temp.shape[0])
        if temp.shape[0] > 1:
            ch_idx = np.unique(main_channels[temp])
            shift_temp = determine_shift(
                templates[[ch_idx]][:, :, temp], W)
            for j2 in range(temp.shape[0]):
                weight_temp[j2] = weights[temp[j2]]
                s = shift_temp[j2]
                if s > 0:
                    templatesNew_temp[
                        :, :(R-s), j2] = templates[:, s:, temp[j2]]
                elif s < 0:
                    templatesNew_temp[
                        :, (-s):, j2] = templates[:, :(R+s), temp[j2]]
                elif s == 0:
                    templatesNew_temp[:, :, j2] = templates[:, :, temp[j2]]

                idx_old_id = spike_train[:, 1] == temp[j2]
                spt_new[idx_old_id] = spike_train[idx_old_id, 0] + s
                id_new[idx_old_id] = k

            weightNew[k] = np.sum(weight_temp)
            templatesNew[:, :, k] = np.average(
                templatesNew_temp, axis=2, weights=weight_temp)

        else:
            weightNew[k] = weights[temp[0]]
            templatesNew[:, :, k] = templates[:, :, temp[0]]

            idx_old_id = spike_train[:, 1] == temp[0]
            spt_new[idx_old_id] = spike_train[idx_old_id, 0]
            id_new[idx_old_id] = k

    spike_train_clear_new = np.hstack((
        spt_new[:, np.newaxis], id_new[:, np.newaxis]))

    return spike_train_clear_new, templatesNew, groups


# TODO: documentation
# TODO: comment code, it's not clear what it does
def determine_shift(tt, W):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, RW, K = tt.shape
    R = RW - 2*W
    t1 = tt[:, W:(W+R), 0]
    norm1 = np.linalg.norm(t1)
    
    shift = np.zeros(K, 'int16')
    for k in range(1, K):
        cos = np.zeros(2*W+1)
        for j in range(2*W+1):
            t2 = tt[:, j:(j+R), k]
            norm2 = np.linalg.norm(t2)
            cos[j] = np.sum(t1*t2)/norm1/norm2
        ii = np.argmax(cos)
        shift[k] = ii - W

    amps = np.max(np.abs(tt), axis=(0, 1))
    k_max = np.argmax(amps)
    shift = shift - shift[k_max]

    return shift


# TODO: documentation
# TODO: comment code, it's not clear what it does
def strongly_connected_components_iterative(vertices, edges):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    identified = set()
    stack = []
    index = {}
    boundaries = []

    for v in vertices:
        if v not in index:
            to_do = [('VISIT', v)]
            while to_do:
                operation_type, v = to_do.pop()
                if operation_type == 'VISIT':
                    index[v] = len(stack)
                    stack.append(v)
                    boundaries.append(index[v])
                    to_do.append(('POSTVISIT', v))
                    # We reverse to keep the search order identical to that of
                    # the recursive code;  the reversal is not necessary for
                    # correctness, and can be omitted.
                    to_do.extend(
                        reversed([('VISITEDGE', w) for w in edges[v]]))
                elif operation_type == 'VISITEDGE':
                    if v not in index:
                        to_do.append(('VISIT', v))
                    elif v not in identified:
                        while index[v] < boundaries[-1]:
                            boundaries.pop()
                else:
                    # operation_type == 'POSTVISIT'
                    if boundaries[-1] == index[v]:
                        boundaries.pop()
                        scc = set(stack[index[v]:])
                        del stack[index[v]:]
                        identified.update(scc)
                        yield scc


# TODO: documentation
# TODO: comment code, it's not clear what it does
def TemplatesSimilarity(t1, t2, th, W):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    C, RW = t1.shape
    R = RW - 2*W

    t1 = np.reshape(t1[:, W:(W+R)], R*C)
    t2_shifted = np.zeros((2*W+1, R*C))
    for j in range(2*W+1):
        t2_shifted[j] = np.reshape(t2[:, j:(j+R)], R*C)

    norm1 = np.sqrt(np.sum(np.square(t1), axis=0))
    norm2 = np.sqrt(np.sum(np.square(t2_shifted), axis=1))
    cos = np.matmul(t2_shifted, t1)/(norm1*norm2)

    ii = np.argmax(cos)
    cos = cos[ii]

    similar = 0
    if cos > th[0]:
        t1 = np.reshape(t1, [C, R])
        t2 = np.reshape(t2_shifted[ii], [C, R])

        scale = np.sum(t1*t2)/np.sum(np.square(t1))
        if scale > th[1] and scale < 2 - th[1]:
            similar = 1

    return similar
