import numpy as np
import logging

from yass import mfm
from yass.cluster.merge import merge_units
from scipy.sparse import csc_matrix, lil_matrix

def run_cluster(scores, masks, groups, spike_times,
                channel_groups, channel_index,
                n_features, CONFIG):
    """
    run clustering algorithm using MFM

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    masks: list (n_channels)
        mask for each data in scores
        masks[c] is the mask of spikes in scores[c]

    groups: list (n_channels)
        coreset represented as group id.
        groups[c] is the group id of spikes in scores[c]

    spike_index: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    channel_groups: list (n_channel_groups)
        Using divide-and-conquer approach, data will be split
        based on main channel. As an example, data in group g
        will be data whose main channel is one of channel_groups[g]

    channel_index: np.array (n_channels, n_neigh)
        neighboring channel information
        channel_index[c] contains the index of neighboring channels of
        channel c

    n_features: int
       number of features in each data per channel

    CONFIG: class
       configuration class

    Returns
    -------
    spike_train: np.array (n_data, 2)
        spike_train such that spike_train[j, 0] and spike_train[j, 1]
        are the spike time and spike id of spike j
    """

    # FIXME: mutating parameter
    # this function is passing a config object and mutating it,
    # this is not a good idea as having a mutable object lying around the code
    # can break things and make it hard to debug
    # (09/27/17) Eduardo

    n_channel_groups = len(channel_groups)
    n_channels, n_neigh = channel_index.shape

    # biggest cluster id is -1 since there is no cluster yet
    max_cluster_id = -1
    spike_train = np.zeros((0, 2), 'int32')
    for g in range(n_channel_groups):

        # channels in the group
        core_channels = channel_groups[g]
        # include all channels neighboring to core channels
        neigh_cores = np.unique(channel_index[core_channels])
        neigh_cores = neigh_cores[neigh_cores < n_channels]
        n_neigh_channels = neigh_cores.shape[0]

        # initialize data for this channel group
        score = np.zeros((0, n_features, n_neigh_channels))
        mask = np.zeros((0, n_neigh_channels))
        group = np.zeros(0, 'int32')
        spike_time = np.zeros((0), 'int32')

        # gather information
        max_group_id = -1
        for _, channel in enumerate(core_channels):
            if scores[channel].shape[0] > 0:

                # number of data
                n_data_channel = scores[channel].shape[0]
                # neighboring channels in this group
                neigh_channels = channel_index[channel][
                    channel_index[channel] < n_channels]

                # expand the number of channels and
                # re-organize data to match it
                score_temp = np.zeros((n_data_channel, n_features,
                                       n_neigh_channels))
                mask_temp = np.zeros((n_data_channel,
                                      n_neigh_channels))
                for j in range(neigh_channels.shape[0]):
                    c_idx = neigh_cores == neigh_channels[j]
                    score_temp[:, :, c_idx
                               ] = scores[channel][:, :, [j]]

                    mask_temp[:, c_idx] = masks[channel][:, [j]]

                # collect all data in this group
                score = np.concatenate((score, score_temp), axis=0)
                mask = np.concatenate((mask, mask_temp), axis=0)
                spike_time = np.concatenate((spike_time, spike_times[channel]),
                                            axis=0)
                group = np.concatenate((group,
                                        groups[channel] + max_group_id + 1),
                                       axis=0)
                max_group_id += np.max(groups[channel]) + 1

        if score.shape[0] > 0:

            # run clustering
            cluster_id = spikesort(score, mask, group, CONFIG)

            # model based triage
            idx_triage = cluster_id == -1
            cluster_id = cluster_id[~idx_triage]
            spike_time = spike_time[~idx_triage]

            spike_train = np.vstack((spike_train, np.hstack(
                (spike_time[:, np.newaxis],
                 cluster_id[:, np.newaxis] + max_cluster_id + 1))))

            max_cluster_id += (np.max(cluster_id) + 1)

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort], 0, 0


def run_cluster_location(scores, spike_index, CONFIG):
    """
    run clustering algorithm using MFM and location features

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    spike_times: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    CONFIG: class
        configuration class

    Returns
    -------
    spike_train: np.array (n_data, 2)
        spike_train such that spike_train[j, 0] and spike_train[j, 1]
        are the spike time and spike id of spike j
    """
    logger = logging.getLogger(__name__)

    n_channels = np.max(spike_index[:, 1]) + 1
    global_score = None
    global_vbParam = None
    global_spike_index = None
    #global_cluster_id = None
    global_tmp_loc = None
    # run clustering algorithm per main channel
    for channel in range(n_channels):

        logger.info('Processing channel {}'.format(channel))

        idx_data = np.where(spike_index[:, 1]==channel)[0]
        score_channel = scores[idx_data]
        spike_index_channel = spike_index[idx_data]
        n_data = score_channel.shape[0]

        if n_data > 1:

            # make a fake mask of ones to run clustering algorithm
            mask = np.ones((n_data, 1))
            group = np.arange(n_data)
            vbParam = mfm.spikesort(np.copy(score_channel),
                                    mask,
                                    group, CONFIG)

            #cluster_id = mfm.cluster_triage(vbParam, score, 15)
            #idx_triage = (cluster_id == -1)
            
            #core_data = mfm.get_core_data(vbParam, score, 100, 5)
            
            #spike_time = spike_index[idx_data, 0][core_data[:,0]]
            #cluster_id = core_data[:,1]
            
            #print(cluster_id.shape)
            #cluster_id = cluster_id[~idx_triage]
            #spike_time = spike_time[~idx_triage]
            #score = score[~idx_triage]
            vbParam.rhat[vbParam.rhat < 0.1] = 0
            vbParam.rhat = vbParam.rhat/np.sum(vbParam.rhat, 1, keepdims=True)
            vbParam = clean_empty_cluster2(vbParam)
            
            print(vbParam.rhat.shape)
            #(vbParam, spike_time,
            # cluster_id) = clean_empty_cluster(vbParam,
            #                                   spike_time,
            #                                   cluster_id)
            #print(cluster_id.shape)
            # gather clustering information into global variable
            #(global_vbParam,
            # global_score, global_spike_time,
            # global_cluster_id) = global_cluster_info(vbParam,
            #                                          score,
            #                                          spike_time,
            #                                          cluster_id,
            #                                          global_vbParam,
            #                                          global_score,
            #                                          global_spike_time,
            #                                          global_cluster_id)

            (global_vbParam,
             global_tmp_loc,
             global_score,
             global_spike_index) = global_cluster_info2(
                vbParam, channel, score_channel, spike_index_channel,
                global_vbParam, global_tmp_loc,
                global_score, global_spike_index)

    #print('calc rhat')
    #rhat = calculate_sparse_rhat(global_vbParam, global_tmp_loc, scores,
    #                             spike_index, CONFIG.neigh_channels)
    #global_vbParam.rhat = rhat
    #print('done')
    
    #spike_train = np.hstack((
    #    spike_index[:, [0]], rhat.argmax(axis=1)))

    # global merge
    #maha = calculate_mahalanobis(global_vbParam)
    #check = np.logical_or(maha<15, maha.T<15)
    #while np.any(check):
    #    cluster = np.where(np.any(check, axis = 1))[0][0]
    #    neigh_clust = list(np.where(check[cluster])[0])
    #    vbParam, maha = merge_move_patches(cluster, neigh_clust, scores,
    #                                      global_vbParam, maha, CONFIG)
    #    check = np.logical_or(maha<15, maha.T<15)

    #
    #global_spike_time, global_cluster_id, global_score = clean_empty_cluster(
    #    global_spike_time, global_cluster_id, global_score[:,:,0])

    # make spike train
    #spike_train = np.hstack(
    #    (global_spike_time[:, np.newaxis],
    #     global_cluster_id[:, np.newaxis]))

    #spike_train = merge_units(global_scores, spike_train, 2*global_scores.shape[1])

    # sort based on spike_time
    #idx_sort = np.argsort(spike_train[:, 0])

    #return spike_train[idx_sort], global_score[idx_sort]
    #return spike_train, global_vbParam, global_tmp_loc
    return global_vbParam, global_tmp_loc, global_score, global_spike_index


def calculate_sparse_rhat(vbParam, tmp_loc, scores, spike_index):
    # vbParam.rhat calculation
    n_channels = np.max(spike_index[:, 1]) + 1
    n_templates = tmp_loc.shape[0]

    rhat = lil_matrix((scores.shape[0], n_templates))
    rhat = None
    for channel in range(n_channels):

        idx_data = np.where(spike_index[:, 1]==channel)[0]
        score = scores[idx_data]
        n_data = score.shape[0]
        cluster_idx = np.where(tmp_loc == channel)[0]
        
        if n_data > 0 and cluster_idx.shape[0] > 0:        
            #ch_idx = np.where(neighbors[channel])[0]
            #cluster_idx = np.zeros(n_templates, 'bool')
            #for _, c in enumerate(ch_idx):
            #    cluster_idx[tmp_loc == c] = 1
            #cluster_idx = np.where(cluster_idx)[0]
            
            local_vbParam = mfm.vbPar(None)
            local_vbParam.muhat = vbParam.muhat[:, cluster_idx]
            local_vbParam.Vhat = vbParam.Vhat[:, :, cluster_idx]
            local_vbParam.invVhat = vbParam.invVhat[:, :, cluster_idx]
            local_vbParam.nuhat = vbParam.nuhat[cluster_idx]
            local_vbParam.lambdahat = vbParam.lambdahat[cluster_idx]
            local_vbParam.ahat = vbParam.ahat[cluster_idx]

            mask = np.ones([n_data, 1])
            group = np.arange(n_data)
            masked_data = mfm.maskData(score, mask, group)

            local_vbParam.update_local(masked_data)
            local_vbParam.rhat[local_vbParam.rhat < 0.1] = 0
            local_vbParam.rhat = local_vbParam.rhat/ \
            np.sum(local_vbParam.rhat, axis=1, keepdims=True)

                            
            row_idx, col_idx = np.where(local_vbParam.rhat > 0)
            val = local_vbParam.rhat[row_idx, col_idx]
            row_idx = idx_data[row_idx]
            col_idx = cluster_idx[col_idx]
            rhat_local = np.hstack((row_idx[:, np.newaxis],
                                    col_idx[:, np.newaxis],
                                    val[:, np.newaxis]))
            if rhat is None:
                rhat = rhat_local
            else:
                rhat = np.vstack((rhat, rhat_local))
           
    return rhat


def calculate_maha_clusters(vbParam):
    diff = np.transpose(vbParam.muhat, [1,2,0]) - vbParam.muhat[...,0].T
    clustered_prec = np.transpose(vbParam.Vhat[:,:,:,0] * vbParam.nuhat,[2,0,1])
    maha = np.squeeze(np.matmul(diff[:,:,np.newaxis],np.matmul(clustered_prec[:,np.newaxis], diff[..., np.newaxis]) ), axis = [2,3])
    maha[np.diag_indices(maha.shape[0])] = np.inf
    
    return maha

def merge_move_patches(cluster, neigh_clusters, scores, vbParam, maha, cfg):
    
    while len(neigh_clusters) > 0:
        i = neigh_clusters[-1]
        #indices = np.logical_or(clusterid == cluster, clusterid == i)
        indices, temp = vbParam.rhat[:, [cluster, i]].nonzero()
        indices = np.unique(indices)
        ka,kb = min(cluster, i), max(cluster, i)
        local_scores = scores[indices]
        local_vbParam = mfm.vbPar(vbParam.rhat[:, [cluster, i]].toarray()[indices])
        local_vbParam.muhat = vbParam.muhat[:,[cluster, i]]
        local_vbParam.Vhat = vbParam.Vhat[:,:,[cluster, i]]
        local_vbParam.invVhat = vbParam.invVhat[:,:,[cluster, i]]
        local_vbParam.nuhat = vbParam.nuhat[[cluster, i]]
        local_vbParam.lambdahat = vbParam.lambdahat[[cluster, i]]
        local_vbParam.ahat = vbParam.ahat[[cluster, i]]
        mask = np.ones([local_scores.shape[0],1])
        group = np.arange(local_scores.shape[0])
        local_maskedData = mfm.maskData(local_scores, mask, group)
        #local_vbParam.update_local(local_maskedData)
        local_suffStat = mfm.suffStatistics(local_maskedData,local_vbParam)
        
        ELBO = mfm.ELBO_Class(local_maskedData, local_suffStat, local_vbParam, cfg)
        L = np.ones(2)
        (local_vbParam, local_suffStat,
         merged,_ ,_) = mfm.check_merge(local_maskedData, 
                                        local_vbParam,
                                        local_suffStat, 0, 1, 
                                        cfg, L, ELBO)
        if merged:
            print("merging {}, {}".format(cluster,i))
            print('test1')
            vbParam.muhat = np.delete(vbParam.muhat, kb, 1)
            vbParam.muhat[:,ka] = local_vbParam.muhat[:,0]
            
            vbParam.Vhat = np.delete(vbParam.Vhat, kb, 2)
            vbParam.Vhat[:,:,ka] = local_vbParam.Vhat[:,:,0]
            
            vbParam.invVhat = np.delete(vbParam.invVhat, kb, 2)
            vbParam.invVhat[:,:,ka] = local_vbParam.invVhat[:,:,0]
            
            vbParam.nuhat = np.delete(vbParam.nuhat, kb, 0)
            vbParam.nuhat[ka] = local_vbParam.nuhat[0]
            
            vbParam.lambdahat = np.delete(vbParam.lambdahat, kb, 0)
            vbParam.lambdahat[ka] = local_vbParam.lambdahat[0]
            
            vbParam.ahat = np.delete(vbParam.ahat, kb, 0)
            vbParam.ahat[ka] = local_vbParam.ahat[0]
            
            print('test2')
            vbParam.rhat[:, ka] = vbParam.rhat[:, ka] + vbParam.rhat[:, kb]
            n_data_all, n_templates_all = vbParam.rhat.shape
            to_keep = list(set(np.arange(n_templates_all))-set([kb]))  
            vbParam.rhat = vbParam.rhat[:,to_keep]

            print('test4')
            #clusterid[indices] = ka
            #clusterid[clusterid > kb] = clusterid[clusterid > kb] - 1
            neigh_clusters.pop()
            
            maha = np.delete(maha, kb, 1)
            maha = np.delete(maha, kb, 0)
            
            diff =  vbParam.muhat[:,:,0] - local_vbParam.muhat[:,:,0]
            
            prec = local_vbParam.Vhat[...,0] * local_vbParam.nuhat[0]
            maha[ka] = np.squeeze(np.matmul(diff.T[:,np.newaxis,:],np.matmul(prec[:,:,0], diff.T[..., np.newaxis]) ))
            
            prec = np.transpose(vbParam.Vhat[...,0] * vbParam.nuhat, [2,0,1])
            maha[:,ka] = np.squeeze(np.matmul(diff.T[:,np.newaxis,:],np.matmul(prec, diff.T[..., np.newaxis]) ))

            maha[ka,ka] = np.inf
            neigh_clusters = list(np.where(np.logical_or(maha[ka]< 15, maha.T[ka]< 15))[0])
            cluster = ka
            
            print('test5')
        if not merged:
            maha[ka,kb] = maha[kb,ka] = np.inf
            neigh_clusters.pop()
    
    return vbParam, maha


def try_merge(k1, k2, scores, vbParam, maha, cfg):
    
    ka, kb = min(k1, k2), max(k1, k2)

    assignment = vbParam.rhat[:, :2].astype('int32')
    
    idx_ka = assignment[:, 1] == ka
    idx_kb = assignment[:, 1] == kb
    
    indices = np.unique(assignment[
        np.logical_or(idx_ka, idx_kb), 0])
    
    rhat = np.zeros((scores.shape[0], 2))
    rhat[assignment[idx_ka, 0], 0] = vbParam.rhat[idx_ka, 2]
    rhat[assignment[idx_kb, 0], 1] = vbParam.rhat[idx_kb, 2]
    rhat = rhat[indices]
    
    local_scores = scores[indices]
    local_vbParam = mfm.vbPar(rhat)
    local_vbParam.muhat = vbParam.muhat[:,[ka, kb]]
    local_vbParam.Vhat = vbParam.Vhat[:,:,[ka, kb]]
    local_vbParam.invVhat = vbParam.invVhat[:,:,[ka, kb]]
    local_vbParam.nuhat = vbParam.nuhat[[ka, kb]]
    local_vbParam.lambdahat = vbParam.lambdahat[[ka, kb]]
    local_vbParam.ahat = vbParam.ahat[[ka, kb]]
    
    mask = np.ones([local_scores.shape[0], 1])
    group = np.arange(local_scores.shape[0])
    local_maskedData = mfm.maskData(local_scores, mask, group)
    #local_vbParam.update_local(local_maskedData)
    local_suffStat = mfm.suffStatistics(local_maskedData,local_vbParam)

    ELBO = mfm.ELBO_Class(local_maskedData, local_suffStat, local_vbParam, cfg)
    L = np.ones(2)
    (local_vbParam, local_suffStat,
     merged,_ ,_) = mfm.check_merge(local_maskedData, 
                                    local_vbParam,
                                    local_suffStat, 0, 1, 
                                    cfg, L, ELBO)
    if merged:
        print("merging {}, {}".format(ka,kb))

        vbParam.muhat = np.delete(vbParam.muhat, kb, 1)
        vbParam.muhat[:,ka] = local_vbParam.muhat[:,0]

        vbParam.Vhat = np.delete(vbParam.Vhat, kb, 2)
        vbParam.Vhat[:,:,ka] = local_vbParam.Vhat[:,:,0]

        vbParam.invVhat = np.delete(vbParam.invVhat, kb, 2)
        vbParam.invVhat[:,:,ka] = local_vbParam.invVhat[:,:,0]

        vbParam.nuhat = np.delete(vbParam.nuhat, kb, 0)
        vbParam.nuhat[ka] = local_vbParam.nuhat[0]

        vbParam.lambdahat = np.delete(vbParam.lambdahat, kb, 0)
        vbParam.lambdahat[ka] = local_vbParam.lambdahat[0]

        vbParam.ahat = np.delete(vbParam.ahat, kb, 0)
        vbParam.ahat[ka] = local_vbParam.ahat[0]

        idx_delete = np.where(np.logical_or(idx_ka, idx_kb))[0]
        vbParam.rhat = np.delete(vbParam.rhat, idx_delete, 0)
        vbParam.rhat[vbParam.rhat[:, 1] > kb, 1] -= 1
        
        rhat_temp = np.hstack((indices[:, np.newaxis],
                               np.ones((indices.size, 1))*ka,
                               np.sum(rhat, 1, keepdims=True)))
        vbParam.rhat = np.vstack((vbParam.rhat, rhat_temp))
        
        maha = np.delete(maha, kb, 1)
        maha = np.delete(maha, kb, 0)

        diff =  vbParam.muhat[:,:,0] - local_vbParam.muhat[:,:,0]

        prec = local_vbParam.Vhat[...,0] * local_vbParam.nuhat[0]
        maha[ka] = np.squeeze(np.matmul(diff.T[:,np.newaxis,:],np.matmul(prec[:,:,0], diff.T[..., np.newaxis]) ))

        prec = np.transpose(vbParam.Vhat[...,0] * vbParam.nuhat, [2,0,1])
        maha[:,ka] = np.squeeze(np.matmul(diff.T[:,np.newaxis,:],np.matmul(prec, diff.T[..., np.newaxis]) ))

        maha[ka,ka] = np.inf

    if not merged:
        maha[ka,kb] = maha[kb,ka] = np.inf
    
    return vbParam, maha
    

def global_cluster_info(vbParam, score, spike_time, cluster_id,
                        global_vbParam, global_score,
                        global_spike_time, global_cluster_id):
    """
    Gather clustering information from each run
    Parameters
    ----------
    vbParam, maskedData: class
        cluster information output from MFM
    score: np.array (n_data, n_features, 1)
        score used for each clustering
    spike_time: np.array (n_data, 1)
        spike time that matches with each score
    global_vbParam, global_maskedData: class
        a class that contains cluster information from all
        previous run,
    global_score: np.array (n_data_all, n_features, 1)
        all scores from previous runs
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    Returns
    -------
    global_vbParam, global_maskedData: class
        a class that contains cluster information after
        adding the current one
    global_score: np.array (n_data_all, n_features, 1)
        all scores after adding the current one
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    """
    if global_vbParam is None:
        global_vbParam = vbParam
        global_score = score
        global_spike_time = spike_time
        global_cluster_id = cluster_id
    else:

        # append global_vbParam
        global_vbParam.muhat = np.concatenate(
            [global_vbParam.muhat, vbParam.muhat], axis=1)
        global_vbParam.Vhat = np.concatenate(
            [global_vbParam.Vhat, vbParam.Vhat], axis=2)
        global_vbParam.invVhat = np.concatenate(
            [global_vbParam.invVhat, vbParam.invVhat],
            axis=2)
        global_vbParam.lambdahat = np.concatenate(
            [global_vbParam.lambdahat, vbParam.lambdahat],
            axis=0)
        global_vbParam.nuhat = np.concatenate(
            [global_vbParam.nuhat, vbParam.nuhat],
            axis=0)
        global_vbParam.ahat = np.concatenate(
            [global_vbParam.ahat, vbParam.ahat],
            axis=0)

        # append score
        global_score = np.concatenate([global_score, score], axis=0)

        # append spike_time
        global_spike_time = np.hstack((global_spike_time,
                                           spike_time))
        
        # append assignment
        cluster_id_max = np.max(global_cluster_id)
        global_cluster_id = np.hstack([
            global_cluster_id,
            cluster_id + cluster_id_max + 1])

    return (global_vbParam, global_score,
            global_spike_time, global_cluster_id)


def global_cluster_info2(vbParam, main_channel,
                         score, spike_index,
                         global_vbParam, global_tmp_loc,
                         global_score, global_spike_index):
    """
    Gather clustering information from each run
    Parameters
    ----------
    vbParam, maskedData: class
        cluster information output from MFM
    score: np.array (n_data, n_features, 1)
        score used for each clustering
    spike_time: np.array (n_data, 1)
        spike time that matches with each score
    global_vbParam, global_maskedData: class
        a class that contains cluster information from all
        previous run,
    global_score: np.array (n_data_all, n_features, 1)
        all scores from previous runs
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    Returns
    -------
    global_vbParam, global_maskedData: class
        a class that contains cluster information after
        adding the current one
    global_score: np.array (n_data_all, n_features, 1)
        all scores after adding the current one
    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score
    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    """
    
    n_idx, k_idx = np.where(vbParam.rhat > 0) 
    prob_val = vbParam.rhat[n_idx, k_idx] 
    vbParam.rhat = np.hstack((n_idx[:, np.newaxis],
                              k_idx[:, np.newaxis],
                              prob_val[:, np.newaxis]))

    if global_vbParam is None:
        global_vbParam = vbParam
        global_tmp_loc = np.ones(
            vbParam.muhat.shape[1], 'int16')*main_channel
        global_score = score
        global_spike_index = spike_index

    else:

        # append global_vbParam
        global_vbParam.muhat = np.concatenate(
            [global_vbParam.muhat, vbParam.muhat], axis=1)
        global_vbParam.Vhat = np.concatenate(
            [global_vbParam.Vhat, vbParam.Vhat], axis=2)
        global_vbParam.invVhat = np.concatenate(
            [global_vbParam.invVhat, vbParam.invVhat],
            axis=2)
        global_vbParam.lambdahat = np.concatenate(
            [global_vbParam.lambdahat, vbParam.lambdahat],
            axis=0)
        global_vbParam.nuhat = np.concatenate(
            [global_vbParam.nuhat, vbParam.nuhat],
            axis=0)
        global_vbParam.ahat = np.concatenate(
            [global_vbParam.ahat, vbParam.ahat],
            axis=0)
        
        n_max, k_max = np.max(global_vbParam.rhat[:, :2], axis=0)
        vbParam.rhat[:,0] += n_max + 1
        vbParam.rhat[:,1] += k_max + 1
        global_vbParam.rhat = np.concatenate(
            [global_vbParam.rhat, vbParam.rhat],
            axis=0)
        
        global_tmp_loc = np.hstack((global_tmp_loc,
                                    np.ones(vbParam.muhat.shape[1],
                                            'int16')*main_channel))
        
        # append score
        global_score = np.concatenate([global_score,
                                       score], axis=0)
        
        # append spike_index
        global_spike_index = np.concatenate([global_spike_index,
                                             spike_index], axis=0)

    return (global_vbParam, global_tmp_loc,
            global_score, global_spike_index)
             

def clean_empty_cluster(vbParam, spike_time, cluster_id, max_spikes=10):

    n_units = np.max(cluster_id) + 1
    units_keep  = np.zeros(n_units, 'bool')
    for k in range(n_units):
        if np.sum(cluster_id == k) >= max_spikes:
            units_keep[k] = 1

    Ks = np.where(units_keep)[0]
    spike_time_clean = np.zeros(0, 'int32')
    cluster_id_clean = np.zeros(0, 'int32')
    for j, k in enumerate(Ks):

        spt_temp = spike_time[cluster_id == k]

        spike_time_clean = np.hstack((spike_time_clean,
                                      spt_temp))
        cluster_id_clean = np.hstack((cluster_id_clean,
                                      np.ones(spt_temp.shape[0], 'int32')*j))

    idx_sort = np.argsort(spike_time_clean)
    
    vbParam.muhat = vbParam.muhat[:, Ks]
    vbParam.Vhat = vbParam.Vhat[:, :, Ks]
    vbParam.invVhat = vbParam.invVhat[:, :, Ks]
    vbParam.lambdahat = vbParam.lambdahat[Ks]
    vbParam.nuhat = vbParam.nuhat[Ks]
    vbParam.ahat = vbParam.ahat[Ks]        

    return vbParam, spike_time_clean[idx_sort], cluster_id_clean[idx_sort]


def clean_empty_cluster2(vbParam, max_spikes=20):

    n_hat = np.sum(vbParam.rhat, 0)
    Ks = n_hat > max_spikes

    vbParam.muhat = vbParam.muhat[:, Ks]
    vbParam.Vhat = vbParam.Vhat[:, :, Ks]
    vbParam.invVhat = vbParam.invVhat[:, :, Ks]
    vbParam.lambdahat = vbParam.lambdahat[Ks]
    vbParam.nuhat = vbParam.nuhat[Ks]
    vbParam.ahat = vbParam.ahat[Ks]  
    vbParam.rhat = vbParam.rhat[:, Ks]

    return vbParam

