import os.path
import logging
import datetime
import numpy as np
import parmap
import os.path as path
import time

from yass.deconvolute.util import svd_shifted_templates, small_shift_templates, make_spt_list, make_spt_list_parallel, clean_up, calculate_temp_temp_parallel
from yass.deconvolute.deconvolve import deconvolve_new_allcores, fix_indexes
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
    n_processors = CONFIG.resources.n_processors

    TMP_FOLDER = CONFIG.data.root_folder

    # get spt_list
    print('making spt list')
    path_to_spt_list = path.join(TMP_FOLDER, 'tmp/spt_list.npy')
    if os.path.exists(path_to_spt_list):
        spt_list = np.load(path_to_spt_list)
    else:
	#n_processors = 30
	spike_index_chunks = np.array_split(spike_index,n_processors)
	spt_list = parmap.map(make_spt_list_parallel, spike_index_chunks, n_channels, processes=n_processors)
	
	#Recombine the spikes across all channels from different processors
	spt_list_total = []
	for k in range(n_channels):
	    spt_list_total.append([])
	    for p in range(n_processors):
		spt_list_total[k].append(spt_list[p][k])
		
	    spt_list_total[k]=np.hstack(spt_list_total[k])
	np.save(path_to_spt_list, spt_list_total)	

    spike_index = 0
 
    # upsample template
    print('computing shifted templates')
    path_to_shifted_templates = path.join(TMP_FOLDER, 'tmp/shifted_templates.npy')
    if os.path.exists(path_to_shifted_templates):
        shifted_templates= np.load(path_to_shifted_templates)
    else:
	shifted_templates = small_shift_templates(templates, n_shifts)
        np.save(path_to_shifted_templates, shifted_templates)	

    # svd templates
    print('computing svd templates')
    path_to_temporal_features = path.join(TMP_FOLDER, 'tmp/temporal_features.npy')
    path_to_spatial_features = path.join(TMP_FOLDER, 'tmp/spatial_features.npy')
    if os.path.exists(path_to_temporal_features):
        #data = np.load(path_to_svd_templates)
	temporal_features=np.load(path_to_temporal_features)
	spatial_features=np.load(path_to_spatial_features)
    else:
	temporal_features, spatial_features = svd_shifted_templates(shifted_templates, n_features)
        np.save(path_to_spatial_features, spatial_features)
        np.save(path_to_temporal_features, temporal_features)


    # calculate convolution of pairwise templates
    print ("computing temp_temp")
    path_to_temp_temp = path.join(TMP_FOLDER, 'tmp/temp_temp.npy')
    if os.path.exists(path_to_temp_temp):
        temp_temp = np.load(path_to_temp_temp)
    else:
	n_processors = CONFIG.resources.n_processors
	#n_chunks = n_processors
	indexes = np.arange(temporal_features.shape[0])
	template_list = np.array_split(indexes,n_processors)
	#template_list = np.array_split(indexes,20)#[:66]

	temp_temp_array = parmap.map(calculate_temp_temp_parallel, template_list, temporal_features,spatial_features, processes=n_processors)
	print ("Completed temp_temp, saving...")
	temp_temp = np.concatenate((temp_temp_array),axis=1)*2
	print temp_temp.shape
	np.save(path_to_temp_temp, temp_temp)

	#Single core: 
	#temp_temp = calculate_temp_temp(temporal_features, spatial_features)
	#temp_temp *= 2
	#np.save(path_to_temp_temp, temp_temp)	


    #******************** SINGLE PROCESSOR RUN ************************
    # run nn preprocess batch-wsie
    if False:
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

    #********************** PARALLELIZED RUN ***********************
    else:
	buffer_size=2*n_temporal_big+1
	print buffer_size
	n_channels = CONFIG.recordings.n_channels
  
	#filename_bin = '/home/cat/code/yass/data/tmp/standarized.bin'
	filename_bin = os.path.join(CONFIG.data.root_folder, output_directory, recordings_filename)
				      
	fp = np.memmap(filename_bin, dtype='float32', mode='r')
	fp_len = fp.shape[0]
	
	#Make index lists for reading chunks from file;
	#chunk_length = int(CONFIG.resources.max_memory/float(n_processors))
	#indexes = np.int32(np.arange(0,fp_len, chunk_length)/float(n_channels))
	#indexes = np.append(indexes,fp_len/n_channels)

	n_explore = CONFIG.deconvolution.n_explore
	threshold_d = CONFIG.deconvolution.threshold_dd
	n_processors = CONFIG.resources.n_processors
	n_channels = CONFIG.recordings.n_channels  
	n_parallel_chunks = CONFIG.resources.n_parallel_chunks

	buffer_size=2*n_temporal_big+n_explore

	#n_parallel_chunks = 600
	indexes = np.linspace(0,fp_len, n_processors*n_parallel_chunks+1)/n_channels
	#indexes = np.arange(0,fp_len, 20000)
	#print indexes

	idx_list = []
	for k in range(len(indexes)-1):
	    idx_list.append([indexes[k],indexes[k+1],buffer_size, indexes[k+1]-indexes[k]+buffer_size])
	#idx_list.append([indexes[k+1],fp_len,buffer_size, fp_len-indexes[k+1]+buffer_size])
	
	idx_list = np.int32(np.vstack(idx_list))
	print idx_list	

	if n_processors==0:
	    print ("# of chunks: ", len(idx_list), " # cores: ", 1)
	else:
	    print ("# of chunks: ", len(idx_list), " # cores: ", n_processors)
	
	#************************************************************************************************************
	#****************************************f********************************************************************
	#************************************************************************************************************
	#Run parallel deconvolution
	
	if CONFIG.resources.multi_processing:
	    start_time = time.time()
	    print ("Start time for deconvolution: ", datetime.datetime.now().strftime('%H:%M:%S'))

	    proc_indexes = np.arange(len(idx_list))

	    #****************************************************************************************************

	    spike_trains = parmap.map(deconvolve_new_allcores, zip(idx_list,proc_indexes), filename_bin, path_to_spt_list, path_to_temp_temp, path_to_shifted_templates, buffer_size, n_channels, temporal_features, spatial_features, n_explore, threshold_d, processes=n_processors)

	    #****************************************************************************************************
	    
	    print ("Total time for deconvolution: ", time.time()-start_time)

	else:
	    spike_trains = []
	    total_time = time.time()
	    for k in range(len(idx_list)):
		start_time = time.time()
		print "..iteration: ", k
	    
		spike_trains.append(deconvolve_new_allcores([idx_list[k],k], filename_bin, path_to_spt_list, path_to_temp_temp, path_to_shifted_templates, buffer_size, n_channels, temporal_features, spatial_features, n_explore, threshold_d))
		
		print ("Single batch of deconvolfution: ", time.time()-start_time, " total from start: ", time.time()-total_time)

	spike_train = np.vstack(spike_trains)
	print (spike_train.shape)
	
	np.savetxt(filename_bin[:-4]+"_spike_trains.txt" ,spike_train, fmt='%d',)

    logger.debug('spike_train.shape: {}'
                 .format(spike_train.shape))

    # sort spikes by time
    spike_train, templates = clean_up(spike_train, templates, max_spikes)

    return spike_train, np.transpose(templates)
    
    
    
    
    
