import numpy as np

def merge_units(score, spike_train_clear, threshold):
    all_merged = 0
    n_clusters = np.max(spike_train_clear[:,1]) + 1
    unit_mean = None
    unit_cov = None
    while not all_merged:
        
        if unit_mean is None:
            unit_mean = np.zeros((n_clusters, score.shape[1]))
            prec = np.zeros((n_clusters,  score.shape[1], score.shape[1]))

            for k in range(n_clusters):
                score_temp = score[spike_train_clear[:,1]==k]
                if score_temp.shape[0] > 1:
                    unit_mean[k] = np.mean(score_temp, 0)
                    unit_cov = np.cov(score_temp.T)
                else:
                    unit_cov = np.eye(score.shape[1])
                    
                try:
                    prec[k] = linalg.inv(unit_cov)
                except LinAlgErr as err:
                    prec[k] = linalg.inv(unit_cov + np.eye(scores.shape[1])*1e-16)
            
        prec = np.linalg.inv(unit_cov)
        diff = unit_mean[np.newaxis] - unit_mean[:, np.newaxis]
        maha = np.sum(np.matmul( diff[:, :, np.newaxis], np.matmul(prec[np.newaxis], diff[:, :, :, np.newaxis])), (2,3))
        maha[np.arange(n_clusters), np.arange(n_clusters)] = np.inf
        ranks = np.min(maha, 1)

        if np.min(ranks) < threshold:
            k1 = np.argmin(ranks)
            k2 = np.argmin(maha[k1])
            k_small = np.min((k1,k2))
            k_big = np.max((k1,k2))

            spike_train_clear[spike_train_clear[:,1] == k_big,1] = k_small
            spike_train_clear[spike_train_clear[:,1] > k_big, 1] -= 1
            
            n_clusters -= 1
            unit_mean = np.delete(unit_mean, k_big, 0)
            unit_cov = np.delete(unit_cov, k_big, 0)

            score_temp = score[spike_train_clear[:,1]==k_small]
            if score_temp.shape[0] > 1:
                unit_mean[k_small] = np.mean(score_temp, 0)
                unit_cov[k_small] = np.cov(score_temp.T)
            else:
                unit_cov[k_small] = np.eye(score.shape[1])
        else:
            all_merged = 1
            
    return spike_train_clear

    