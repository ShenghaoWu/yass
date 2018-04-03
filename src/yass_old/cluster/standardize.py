import numpy as np

def standardize(scores):
    """
    Triage based on KNN distance.
    It removes triage_percent*100% of data

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    Returns
    -------
    scores: list (n_channels)
        scores after triage
    """
    # relevant info
    n_channels = len(scores)

    for channel in range(n_channels):
        scores_channel = scores[channel][:, :, 0]
        
        scores_channel = np.divide((scores_channel - np.mean(scores_channel, axis=0, keepdims=True)),
                   np.std(scores_channel, axis=0, keepdims=True))

        scores[channel] = scores_channel[:, :, np.newaxis]

    return scores
