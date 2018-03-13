import numpy as np


def clean_up_templates(spike_train, templates, geometry,
                       snr_threshold, duplicate_threshold,
                       spread_threshold):
    
    # get size
    n_channels, temporal_size, n_templates = templates.shape
    
    # get energy
    energy = np.max(templates, 1) - np.min(templates, 1)
    
    # template code: '0' clean template, '1' overly spread template,
    # '2' collided template, '3' low snr template 
    template_code = np.zeros((n_templates), 'int16')

    # check for overly spread template first
    too_spread = np.zeros(n_templates, 'bool')
    for k in range(n_templates):
        
        idx_c = energy[:, k] > np.max(energy[:, k])*0.5
        if np.sum(idx_c) > 1:
            center = np.average(geometry[idx_c], axis=0, weights=energy[idx_c,k])
            lam, V = np.linalg.eig(np.cov(geometry[idx_c].T, aweights=energy[idx_c,k]))
            lam[lam<0] = 0
            if np.sqrt(np.max(lam)) > spread_threshold:
                too_spread[k] = 1

    templates_good = templates[:,:,~too_spread]
    templates_bad = templates[:,:,too_spread]
    template_code[too_spread] = 1

    # try to make overly spread templates using good ones.
    mid_t = int((temporal_size-1)/2)
    built_templates = np.zeros(templates_bad.shape)
    norms = np.sum(np.square(templates_good), (0,1))
    # do it twice so that it can be built using two templates
    for count in range(2):
        fit = np.zeros((templates_bad.shape[2], templates_good.shape[2], temporal_size))
        for k in range(templates_bad.shape[2]):
            t_bad = templates_bad[:,:,k] - built_templates[:, :, k]
            for k2 in range(templates_good.shape[2]):
                t_good = np.flip(templates_good[:,:,k2], 1)  
                for c in range(n_channels):
                    fit[k,k2] += 2*np.convolve(t_bad[c], t_good[c], 'same')
        fit -= norms[np.newaxis, :, np.newaxis]
        
        for k in range(templates_bad.shape[2]):
            best_fit = np.where(fit[k] == np.max(fit[k]))
            k_fit = best_fit[0][0]
            loc_fit = best_fit[1][0]
            shift = loc_fit - mid_t
            if shift < 0:
                built_templates[:, :shift, k] += templates_good[:, -shift:, k_fit]
            elif shift > 0:
                built_templates[:, shift:, k] += templates_good[:, :-shift, k_fit]
            else:
                built_templates[:, :, k] += templates_good[:, :, k_fit]

    decrease = 2*np.sum(templates_bad*built_templates,(0,1)) - np.sum(np.square(built_templates),(0,1))
    percent_decrease = decrease/np.sum(np.square(templates_bad),(0,1)) 
    template_code[np.where(template_code==1)[0][percent_decrease > duplicate_threshold]] = 2

    # triage by snr
    template_code[np.max(energy, 0) < snr_threshold] = 3

    # rebuild spike train and template based on the result
    # keep templates with code 0 and 1 only
    spike_train_new = np.zeros((0, 2), 'int32')
    good_units = np.where(np.logical_or(template_code == 0, template_code == 1))[0]
    for j, k in enumerate(good_units):
        spike_train_temp = spike_train[spike_train[:, 1] == k]
        spike_train_temp[:, 1] = j
        spike_train_new = np.vstack((spike_train_new, spike_train_temp))

    templates_new = templates[:, :, good_units]
    template_code_new = template_code[good_units]
    spread_template = np.where(template_code_new == 1)[0]

    idx_sort = np.argsort(spike_train_new[:, 0])
    
    return spike_train_new[idx_sort], templates_new, spread_template