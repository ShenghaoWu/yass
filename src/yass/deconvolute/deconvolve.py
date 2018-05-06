import numpy as np
import logging
from scipy.signal import argrelmax
import os
import time

#********************************************************************************************************************************************
#********************************************************************************************************************************************
#********************************************************************************************************************************************
def deconvolve_new_allcores_updated(data_in, filename_bin, filename_spt_list, filename_temp_temp, filename_shifted_templates, buffer_size, n_channels, temporal_features, spatial_features, n_explore, threshold_d):
    
    start_time_chunk = time.time()

    idx_list, chunk_idx = data_in[0], data_in[1]
    #print idx_list, proc_idx

    #prPurple("Loading data files for chunk: "+str(chunk_idx))

    spt_list = np.load(filename_spt_list)
    temp_temp = np.load(filename_temp_temp)
    shifted_templates = np.load(filename_shifted_templates)

    fname = os.path.split(filename_shifted_templates)[0]+'/spike_train_clear_after_merge.npy'
    spike_train_clear = np.load(fname)

    #print ("shifted_templates shape: ", shifted_templates.shape)

    #New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]
   
    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    # get offset that will be applied
    offset = idx_local   #idx_local[0].start


    #************************************************************************************************************
    #************************************* LOAD RAW RECORDING ***** ********************************************
    #************************************************************************************************************
    #print (data_start, data_end, offset)
    #print ("   Chunk: ", chunk_idx, "  Loading : ", ((data_end-data_start)*4*n_channels)*1.E-6, "MB")
    with open(filename_bin, "rb") as fin:
	if data_start==0:
	    # Seek position and read N bytes
	    recordings_1D = np.fromfile(fin, dtype='float32', count=(data_end+buffer_size)*n_channels)
	    recordings_1D = np.hstack((np.zeros(buffer_size*n_channels,dtype='float32'),recordings_1D))
	else:
	    fin.seek((data_start-buffer_size)*4*n_channels, os.SEEK_SET)         #Advance to idx_start but go back 
	    recordings_1D =  np.fromfile(fin, dtype='float32', count=((data_end-data_start+buffer_size*2)*n_channels))	#Grab 2 x template_widthx2 buffers

	if len(recordings_1D)!=((data_end-data_start+buffer_size*2)*n_channels):
	    recordings_1D = np.hstack((recordings_1D,np.zeros(buffer_size*n_channels,dtype='float32')))

    fin.close()

    #print ("...data loading time: ", time.time()-start_time_chunk)
    #print buffer_size
    #print recordings_1D.shape
    
    #********************************************************************************************************************************************
    #*************************************************** SETUP ARRAYS ***************************************************************************
    #********************************************************************************************************************************************
    #array_time = time.time()
    rec_len = (data_end-data_start+buffer_size*2)		#Need rec_len in n_samples for below

    #Convert to 2D array
    recordings = recordings_1D.reshape(-1,n_channels)

    n_templates, n_shifts, waveform_size, n_channels = shifted_templates.shape
    R = int((waveform_size-1)/2)
    R2 = int((waveform_size-1)/4)
    principal_channels = np.argmax(np.max(np.abs(shifted_templates),(1,2)), 1)
    
    norms = np.sum(np.square(shifted_templates),(2,3))		#||V|| L2 norms (square of this) for the shifted templates 
    visible_channels = np.max(np.abs(spatial_features), (1,2)) > np.min(np.max(np.abs(spatial_features), (1,2,3)))*0.5
    temporal_features = temporal_features[:, :, R2:3*R2+1]
    #print ("...array setup time: ", time.time()-array_time)

    #print ("Total load time: ", time.time()-start_time_chunk)
    
    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************


    #*********************************************************************************************************************************************
    #******************************* MAKE D_MATRIX ***********************************************************************************************
    #*********************************************************************************************************************************************
    #prPurple("Making d_matrix...")
    start_dmatrix = time.time()
    #d_matrix = np.ones((recordings.shape[0],n_templates,n_shifts))*-np.Inf
    #d_matrix = np.empty((recordings.shape[0],n_templates,n_shifts))*-np.Inf
    d_matrix = np.full((recordings.shape[0],n_templates,n_shifts),-np.Inf)
    #print ("...d_matrix make time: ", time.time()-start_dmatrix)

    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************


    #**************************************************************************************************************************************
    #******************************* REMOVE CLEAR SPIKES **********************************************************************************
    #**************************************************************************************************************************************
    #prPurple("Clearing spikes...")
    #spike_clearing_time=time.time()
    
    #Identify clear spikes in time window
    indexes = np.logical_and(spike_train_clear[:,0]>=data_start, spike_train_clear[:,0]<=data_end)
    clear_spiketimes = spike_train_clear[indexes]

    #indexes_unique = np.unique(clear_spiketimes[:,0],return_index=True)[1]     #Used to also remove any spike times falling within same time window (regardless of template)
    #clear_spiketimes_unique = clear_spiketimes[indexes_unique]                 #No longer exclude these as they could come from different cells on large arrays
    #print "# of clear spikes: ", clear_spiketimes.shape[0]

    #****************************************************************
    #********************** DE-DUPLICATE SPIKES *********************
    #****************************************************************
    #Deduplicate spikes: remove spikes from same template that are within e.g. 20 time steps
    #print("   Deduplicating clear spikes")
    deduplication_time = time.time()
    indexes=[]      #list to hold duplicate spike time indexes
    ctr=0           #loop ctr; probably can write this loop simpler 
    t_steps = 20    #window of time-steps to remove duplicates
    for k in range(len(clear_spiketimes)):
        nearby_indexes = np.where(np.logical_and(clear_spiketimes[:,0]>=clear_spiketimes[k,0]-t_steps, 
                                                 clear_spiketimes[:,0]<=clear_spiketimes[k,0]+t_steps))[0]
        matches = np.where(clear_spiketimes[nearby_indexes,1]==clear_spiketimes[k,1])[0]
        
        if len(matches)>1:
            if ctr!=0:
                if indexes[ctr-1][0]!=nearby_indexes[matches][0]:
                    indexes.append(nearby_indexes[matches])
                    ctr+=1
            else:
                indexes.append(nearby_indexes[matches])
                ctr+=1

    #Make list of duplicated events, keep in mind some are triplets, so delete all but one!!
    duplicated_event_indexes = []
    for k in range(len(indexes)):
        duplicated_event_indexes.append(indexes[k][:-1])
    
    if len(duplicated_event_indexes)!=0:
        duplicated_event_indexes = np.hstack(duplicated_event_indexes)
        #print ("   # of deduplicated events: ", len(duplicated_event_indexes), ",  # of original clear events: ", len(clear_spiketimes))
        clear_spiketimes_unique = np.delete(clear_spiketimes,duplicated_event_indexes,axis=0)
        #print("...deduplication time: ", time.time()-deduplication_time)
    else: 
        clear_spiketimes_unique = clear_spiketimes

    #****************************************************************
    #*************************** REMOVING CLEAR SPIKES **************
    #****************************************************************
    window = 30 #Window in time steps to load template from spike time, e.g. -30..+31               *************************** CAN USE EVEN SMALLER CHUNKS OF TEMPLATE DATA TO FIT
    recordings_copy = recordings.copy()
    #clearing_time = time.time()
    for p in range(len(clear_spiketimes_unique)):
        #Load correct templates_shifted - but only chunk in centre of event using window # of time-steps
        templates_shifted = shifted_templates[clear_spiketimes_unique[p,1],:,:,principal_channels[clear_spiketimes_unique[p,1]]][:,30-window:30+window+1]
        
        #Select times and expand into window
        times = (clear_spiketimes_unique[p,0,np.newaxis] + np.arange(-R2-n_explore, n_explore+R2+1))

        #Select entire channel trace on max channel
        recording = recordings[:, principal_channels[clear_spiketimes_unique[p,1]]]

        #Select X number of recoding chunks that are time_shifted                              ***************************** TRY TO PYTHONIZE THIS STEP
        recording_chunks = []
        for time_ in times:
            recording_chunks.append(recording[buffer_size+time_-window-data_start:buffer_size+time_+window+1-data_start])
        recording_chunks = np.array(recording_chunks)

        #Dot product of shifted recording chunks and shifted raw data
        dot_product = np.matmul(templates_shifted,recording_chunks.T)
        index = np.unravel_index(dot_product.argmax(), dot_product.shape)

        recordings_copy[buffer_size+time_+index[1]-window-2*(R2+n_explore)-data_start:buffer_size+time_+index[1]-2*(R2+n_explore)+window+1-data_start,:]-=shifted_templates[clear_spiketimes_unique[p,1],index[0],30-window:30+window+1]
        #quit()

    #print("...spike clearing time: ", time.time()-clearing_time)

    #Save spike_cleared recordings into recordings
    recordings=recordings_copy

    #****************************************************************
    #***************************** CLEAN SPT_LIST *******************
    #****************************************************************
    #First make spt_list for local analysis only...
    spt_list_local = []
    for k in range(len(spt_list)):
        spt = spt_list[k]
        spt_list_local.append(spt[np.logical_and(spt >= data_start,spt < data_end)]-data_start)			#Find spikes on each channel within time window and offset to 0

    #Remove clear spikes and save to spt_list_local        
    clear_spike_times = clear_spiketimes_unique[:,0]          #Select spike times as 1st column of array 
    for k in range(len(spt_list_local)):                      #Loop over all spt_list, i.e. channels
        common = np.intersect1d(clear_spike_times+data_start, spt_list_local[k])      # find intersetion of clear spikes and spt_list on all channels
        if len(common)>0:
            indexes = np.hstack([(spt_list_local[k]==i).nonzero()[0] for i in common])    # 
            spt_list_local[k] = np.delete(spt_list_local[k],indexes)                            #

    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************
    #*********************************************************************************************************************************************

    #*********************************************************************************************************************************************
    #********************************* DOT PRODUCT LOOP  *****************************************************************************************
    #*********************************************************************************************************************************************

    ctr_times=0				#Marker to ensure some spikes are present in the specific time chunk
    for k in range(n_templates):
        #Select spikes on the max channel of the template picked and offset to t=0 + buffer
        if False: 
            spt = spt_list[principal_channels[k]]
            spt = spt[np.logical_and(spt >= data_start,spt < data_end)]		
            spt = np.int32(spt - data_start + offset)
        else: 
            spt = spt_list_local[principal_channels[k]]                         #Cat use cleared_spike spt_list, also index locally only
            spt = np.int32(spt + offset)
        
        #Pick channels around template 
        ch_idx = np.where(visible_channels[k])[0]	# Pick indexes for channels above threshold

        times = (spt[:, np.newaxis] + np.arange(-R2-n_explore, n_explore+R2+1))	#duplicate spikes -32..+33 sampletimes (65 in total) for each spike
        if len(times)==0: continue		#Skip this chunk of time, no spikes in it


        #Use 2D original arrays
        #This step is equivalent to: wf = recordings[times][ch_idx]; it picks 65 time shifted versions (i.e. from times variable) of each original spike on visible chans (i.e. ch_idx)
        wf = ((recordings.ravel()[(ch_idx + (times * recordings.shape[1]).reshape((-1,1))).ravel()]).reshape(times.size, ch_idx.size)).reshape((spt.shape[0], -1, ch_idx.shape[0]))	
        #SUMMARY: wf contains n_spikes X n_sample_pts X n_channels
        
        spatial_dot = np.matmul(spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx],
                    wf[:, :, np.newaxis, :, np.newaxis])[:,:,:,:,0].transpose(0, 2, 1, 3)

        dot = np.zeros((spt.shape[0], 2*n_explore+1, n_shifts))
        for j in range(2*n_explore+1):
            dot[:, j] = np.sum(spatial_dot[:, :, j:j+2*R2+1]*temporal_features[k][np.newaxis], (2, 3))
        
        d_matrix[spt[:, np.newaxis] + np.arange(-n_explore,n_explore+1), k] = 2*dot - norms[k][np.newaxis, np.newaxis]
        
        ctr_times+=1
    
	
    if ctr_times==0: 
        print ("********************************************************** no spikes 2...")
        return None
	#continue	#Skip this chunk of time entirely
	    
	    
    #*****************************************************************************************************
    #*********************************** THRESHOLD LOOP **************************************************
    #*****************************************************************************************************
    #print ("Processor: ", proc_idx, "chunk: ", chunk_idx, " in thresholding loop ...") 
    spike_train = np.zeros((0, 2), 'int32')
   
    max_d = np.max(d_matrix, (1,2))
    max_d_array = []
    max_d_array.append(max_d.copy())

    spike_times = []

    max_val = np.max(max_d)
    threshold_ctr=0
    
    while max_val > threshold_d:
        #print max_val

        # find spike time; get spike time from objective function; and which template it is
        peaks = argrelmax(max_d)[0]
        
        idx_good = peaks[np.argmax(max_d[peaks[:, np.newaxis] + np.arange(-R,R+1)],1) == R]
        #idx_good = peaks
        
        spike_time = idx_good[max_d[idx_good] > threshold_d]
        spike_times.append(spike_time)
                                #argmax finds the location (in template_id space) of the maximum value at each peak time - and it does so by also considering time shifts
        template_id, max_shift = np.unravel_index(np.argmax(np.reshape(d_matrix[spike_time],(spike_time.shape[0], -1)),1),[n_templates, n_shifts])	#This finds the template_id and max_shift for each peak detected

        ## prevent refractory period violation
        rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
        rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
        d_matrix[rf_area, rf_area_t] = -np.Inf

        #****************BOTTLENECK #1************************
        ## update nearby times
        for j in range(spike_time.shape[0]):
            #Original
            t_neigh, k_neigh = np.where(d_matrix[spike_time[j]-2*R : spike_time[j]+2*R, :, 0] > -np.Inf)		#Find non-Inf values in the spike time window, returns times and template to which times belongs

            #new 
            #t_neigh, k_neigh2 = np.where(d_matrix[spike_time[j]-2*R : spike_time[j]+2*R, overlap_channels[template_id[j]], 0] > -np.Inf)		#Find non-Inf values in the spike time window, returns times and template to which times belongs
            #k_neigh = overlap_channels[template_id[j]][k_neigh2]
            
            t_neigh_abs = spike_time[j] + t_neigh - 2*R			#Shift the absolute times back to real-time (in specific block)
            
            d_matrix[t_neigh_abs, k_neigh] -= temp_temp[template_id[j], k_neigh, max_shift[j], :, t_neigh]		#***************** BOTTLNECK  #1

            #Peter subtract all chunk of temp_temp not just for nearby space/time
            #d_matrix[spike_time[j] -2*R : spike_time[j] + 2*R +1 , :] -= np.transpose(temp_temp[template_id[j], :, max_shift[j], :, :], (2,0,1))		#***************** BOTTLNECK  #1

        #***********Optional update of d_matrix at specific times
        #start_time=time.time()
        #time_affected = np.reshape(spike_time[:, np.newaxis] + np.arange(-2*R,2*R+1), -1)
        #time_affected = time_affected[max_d[time_affected] > -np.Inf]
        
        #****************BOTTLENECK #2************************
        #check_pt2=time.time()
        #max_d[time_affected] = np.max(d_matrix[time_affected], (1,2))		#***************** BOTTLNECK  #2: Updates the maximum values of objective function d_matrix
        max_d = np.max(d_matrix, (1,2))					
        #check_pt2 = time.time()-check_pt2
        #print " 4-1 ", check_pt2
        #*****************************************************
        
        #start_time1=time.time()
        max_val = np.max(max_d)
        #print " 4-2 ", time.time()-start_time1
        
        spike_train_temp = np.hstack((spike_time[:, np.newaxis],
                          template_id[:, np.newaxis]))
        spike_train = np.concatenate((spike_train, spike_train_temp), 0)         
        #print "4 ", time.time()-start_time, "\n\n\n"

        #print "max_val: ", max_val, "  total threshold loop time: ", time.time()-threshold_time, " % in dmax: ", int((check_pt2)/(time.time()-threshold_time)*100)
        
        threshold_ctr+=1
        
        max_d_array.append(max_d.copy())
        
        #sys.stdout.write("...# events removed: %d,   max_val: %f,   threshold counter: %d   \r" % (spike_time.shape[0], max_val, threshold_ctr) )
        #sys.stdout.flush()

    #Fix indexes
    spike_times = spike_train[:, 0]
    # get only observations outside the buffer
    train_not_in_buffer = spike_train[np.logical_and(spike_times >= offset,
						     spike_times <= idx_local_end)]
    # offset spikes depending on the absolute location
    train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + data_start
				 - buffer_size)

    if len(clear_spiketimes_unique)>0: 
        final_spikes = np.concatenate((clear_spiketimes_unique,train_not_in_buffer),axis=0)
    else:
        final_spikes = train_not_in_buffer

    prRed("Chunk: "+str(chunk_idx)), prGreen(" time: "+ str(time.time() - start_time_chunk)),
    print("#decon spks: ", len(train_not_in_buffer), " # clear spks: ", len(clear_spiketimes_unique), " Total: ", len(final_spikes), "  # threshold loops: ",threshold_ctr)
    #print('')

    #.append(train_not_in_buffer)
    #chunk_idx+=1

    return final_spikes


def prRed(prt): print("\033[91m {}\033[00m" .format(prt)),
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt)),
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt)),
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))


def deconvolve_new(data_in, filename_bin, filename_spt_list, filename_temp_temp, filename_shifted_templates, buffer_size, n_channels, temporal_features, spatial_features, n_explore, threshold_d):
    
    idx_list, proc_idx = data_in[0], data_in[1]
    #print idx_list, proc_idx

    spt_list = np.load(filename_spt_list)
    temp_temp = np.load(filename_temp_temp)
    shifted_templates = np.load(filename_shifted_templates)


    #New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]
   
    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    # get offset that will be applied
    offset = idx_local   #idx_local[0].start

    parallel=1
    if parallel:
	data_start_array = []
	data_end_array = []
	offset_array = []
	spike_train_array = []
	
	n_chunks = 300
	chunks = np.int32(np.linspace(data_start,data_end, n_chunks+1))
	
	data_start_array= (chunks[:-1])
	data_end_array=chunks[1:]
	offset_array = np.repeat(offset,n_chunks)
	

	
    #************************************************************************************************************
    #************************************* LOOP OVER BATCHES OF DATA ********************************************
    #************************************************************************************************************
    chunk_idx = 0
    for data_start, data_end, offset in zip(data_start_array,data_end_array,offset_array):
	#************************* LOAD RECORDINGS  **************************
	print (data_start, data_end, offset)
	print ("Processor: ", proc_idx, "  Loading : ", ((data_end-data_start)*4*n_channels)*1.E-6, "MB,  chunk: ", chunk_idx, "  of ", n_chunks)
	with open(filename_bin, "rb") as fin:
	    if data_start==0:
		# Seek position and read N bytes
		recordings_1D = np.fromfile(fin, dtype='float32', count=(data_end+buffer_size)*n_channels)
		recordings_1D = np.hstack((np.zeros(buffer_size*n_channels,dtype='float32'),recordings_1D))
	    else:
		fin.seek((data_start-buffer_size)*4*n_channels, os.SEEK_SET)         #Advance to idx_start but go back 
		recordings_1D =  np.fromfile(fin, dtype='float32', count=((data_end-data_start+buffer_size*2)*n_channels))	#Grab 2 x template_widthx2 buffers

	    if len(recordings_1D)!=((data_end-data_start+buffer_size*2)*n_channels):
		recordings_1D = np.hstack((recordings_1D,np.zeros(buffer_size*n_channels,dtype='float32')))

	fin.close()
	
	#print buffer_size
	#print recordings_1D.shape
	rec_len = (data_end-data_start+buffer_size*2)		#Need rec_len in n_samples for below

	#Convert to 2D array
	recordings = recordings_1D.reshape(-1,n_channels)
	#print recordings.shape
	#**************************** PROCESSING **************************

	n_templates, n_shifts, waveform_size, n_channels = shifted_templates.shape
	R = int((waveform_size-1)/2)
	R2 = int((waveform_size-1)/4)
	principal_channels = np.argmax(np.max(np.abs(shifted_templates),(1,2)), 1)
	norms = np.sum(np.square(shifted_templates),(2,3))
	
	visible_channels = np.max(np.abs(spatial_features), (1,2)) > np.min(np.max(np.abs(spatial_features), (1,2,3)))*0.5
	#n_visible_channels = 25
	#ptp = np.max(shifted_templates, (1, 2)) - np.min(shifted_templates, (1,2))
	#visible_channels = np.argsort(ptp, 1)[:, -n_visible_channels:]
	#print visible_channels[0]
	#print principal_channels[0]
	#print visible_channels[1]
	#print principal_channels[1]
	#print visible_channels.shape

	temporal_features = temporal_features[:, :, R2:3*R2+1]

	#Use 2D arrays
	d_matrix = np.ones((recordings.shape[0],n_templates,n_shifts))*-np.Inf
	
	#************************************************************************************************************
	#******************************************** DOT PRODUCTS **************************************************
	#************************************************************************************************************
	#Dot products between waveforms and tempaltes
	print ("Processor: ", proc_idx, "chunk: ", chunk_idx, " in dot product loop ...") 
	
	ctr_times=0
	if True: 
	    
	    #print (n_templates)					# 96
	    #print (spatial_features.shape)			# (96, 5, 3, 49)		#templates  x shift   x features  x channesl
	    #print (temporal_features.shape)			# (96, 5, 61, 3)
	    #print (spt_list.shape)				# 49 channels; contains each channel with it's spike times
	    #print (len(principal_channels))			# the max channels of the 96 templates
	    #for p in range(len(visible_channels)):
	#	print (sum(visible_channels[p]==True))				# the channels above some threshold!? - 15 in this case
	    #print (len(visible_channels[0]==True))				# the channels above some threshold!? - 15 in this case
	    #print (visible_channels[1])				# the channels above some threshold!? - 15 in this case

	    for k in range(n_templates):
		#start_time = time.time()
		#Select spikes on the max channel of the template picked and offset to t=0 + buffer
		spt = spt_list[principal_channels[k]]
		spt = spt[np.logical_and(spt >= data_start,spt < data_end)]			#Why is this 100 sampltieim offset here!?
		spt = spt - data_start + offset
		
		#Pick channels around template 
		ch_idx = np.where(visible_channels[k])[0]	# Pick indexes for channels above threshold

		#print (n_explore)
		#print (R)
		#print (spt[:, np.newaxis])
		times = (spt[:, np.newaxis] + np.arange(-R2-n_explore, n_explore+R2+1))	#duplicate spikes -32..+33 sampletimes (65 in total) for each spike
		#times = (spt[:, np.newaxis] + np.arange(-R2-n_explore, n_explore+R2+1))	#duplicate spikes -32..+33 sampletimes (65 in total) for each spike
		#print (times)
		#print times
		if len(times)==0: continue		#Skip this chunk of time, no spikes in it


		#Use 2D original arrays
		#wf_time = time.time()
		wf = ((recordings.ravel()[(
		   ch_idx + (times * recordings.shape[1]).reshape((-1,1))
		   ).ravel()]).reshape(times.size, ch_idx.size)).reshape((spt.shape[0], -1, ch_idx.shape[0]))	#This is equivalent to: wf = recordings[times][ch_idx], 
		#print "wf time: ", time.time() - wf_time				     # - it picks 65 time shifted versions of each original spike (on ch_idx channels)
	
		#print (wf.shape)
		#wf = recordings[times][:,:,ch_idx]
		#print (wf.shape)
		#quit()
		
		#SUMMARY: wf contains n_spikes X n_sample_pts X n_channels
		
		#print (spatial_features[k][:, :, ch_idx].shape)					#Contains n_features X n_shifts X n_channels; e.g. 5 x 3 x 15
		#print (spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx].shape)
		#print (wf[:, :, :].shape)							#Contains n_spikes X times X channels; e.g. 571 x 65 x 15
		#print (wf[:, :, np.newaxis, :, np.newaxis].shape)

		#spatial_time = time.time()
		spatial_dot = np.matmul(spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx],
					wf[:, :, np.newaxis, :, np.newaxis])[:,:,:,:,0].transpose(0, 2, 1, 3)
		#print "Spatial time: ", time.time() - spatial_time
		#print (spatial_dot.shape)

		
		dot = np.zeros((spt.shape[0], 2*n_explore+1, n_shifts))
		#dot_time = time.time()
		for j in range(2*n_explore+1):
		    #dot[:, j] = np.sum(spatial_dot[:, :, j:j+2*R+1]*temporal_features[k][np.newaxis], (2, 3))
		    dot[:, j] = np.sum(spatial_dot[:, :, j:j+2*R2+1]*temporal_features[k][np.newaxis], (2, 3))
		#print "dot time2: ", time.time()-dot_time
		
		d_matrix[spt[:, np.newaxis] + np.arange(-n_explore,n_explore+1), k] = 2*dot - norms[k][np.newaxis, np.newaxis]
	    
		ctr_times+=1
		#print "Total loop time: ", time.time() - start_time, "\n"
		    
	if ctr_times==0: 
	    print ("********************************************************** no spikes 2...")
	    continue	#Skip this chunk of time entirely
		
		
	#************************************************************************************************************
	#******************************************** THRESHOLDING **************************************************
	#************************************************************************************************************
	print ("Processor: ", proc_idx, "chunk: ", chunk_idx, " in thresholding loop ...") 

	spike_train = np.zeros((0, 2), 'int32')
	max_d = np.max(d_matrix, (1,2))
	max_val = np.max(max_d)
	while max_val > threshold_d:	
	    # find spike time; get spike time from objective function; and which template it is
	    peaks = argrelmax(max_d)[0]
	    idx_good = peaks[np.argmax(
		max_d[peaks[:, np.newaxis] + np.arange(-2*R,2*R+1)],1) == (2*R)]
	    spike_time = idx_good[max_d[idx_good] > threshold_d]
	    template_id, max_shift = np.unravel_index(
		np.argmax(np.reshape(d_matrix[spike_time],
				     (spike_time.shape[0], -1)),1),
		[n_templates, n_shifts])

	    # prevent refractory period violation
	    rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
	    rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
	    d_matrix[rf_area, rf_area_t] = -np.Inf
	    rf_area = np.reshape(rf_area, -1)
	    max_d[rf_area] = np.max(d_matrix[rf_area], (1,2))
	    
	    # update nearby times
	    time_affected = np.zeros(max_d.shape[0], 'bool')
	    for j in range(spike_time.shape[0]):					#Looping over number of spikes with batch chunk
		t_neigh, k_neigh = np.where(					#most expensive step						#WHY IS THERE A COMPARISON WITH -INF? IS THIS EXPENSIVE?
		    d_matrix[spike_time[j]-2*R:spike_time[j]+2*R, :, 0] > -np.Inf)
		t_neigh_abs = spike_time[j] + t_neigh - 2*R
		d_matrix[t_neigh_abs, k_neigh] -= temp_temp[			#d_matrix subtracting temp_temp, predefined deconvolution subtraction
		    template_id[j], k_neigh, max_shift[j], :, t_neigh]
		time_affected[t_neigh_abs] = 1
		
	    max_d[time_affected] = np.max(d_matrix[time_affected], (1,2))
	    max_val = np.max(max_d)
	    
	    spike_train_temp = np.hstack((spike_time[:, np.newaxis],
					  template_id[:, np.newaxis]))
	    spike_train = np.concatenate((spike_train, spike_train_temp), 0)         

	
	#Fix indexes
	spike_times = spike_train[:, 0]
	# get only observations outside the buffer
	train_not_in_buffer = spike_train[np.logical_and(spike_times >= offset,
							 spike_times <= idx_local_end)]
	# offset spikes depending on the absolute location
	train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + data_start
				     - buffer_size)

	print ("Processor: ", proc_idx, "cleaned spikes: ", len(train_not_in_buffer), "  chunk: ", chunk_idx, "  of ", n_chunks)
	
	spike_train_array.append(train_not_in_buffer)
	chunk_idx+=1

    return np.vstack(spike_train_array)


#********************************************************************************************************************************************
#********************************************************************************************************************************************
#********************************************************************************************************************************************
def deconvolve_new_allcores(data_in, filename_bin, filename_spt_list, filename_temp_temp, filename_shifted_templates, buffer_size, n_channels, temporal_features, spatial_features, n_explore, threshold_d):
    
    idx_list, chunk_idx = data_in[0], data_in[1]
    #print idx_list, proc_idx

    chunk_time = time.time()
    spt_list = np.load(filename_spt_list)
    temp_temp = np.load(filename_temp_temp)
    shifted_templates = np.load(filename_shifted_templates)

    #New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]
   
    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    # get offset that will be applied
    offset = idx_local   #idx_local[0].start

    #************************************************************************************************************
    #************************************* LOOP OVER BATCHES OF DATA ********************************************
    #************************************************************************************************************
    #chunk_idx = 0
    #for data_start, data_end, offset in zip(data_start_array,data_end_array,offset_array):
    #************************* LOAD RECORDINGS  **************************
    #print (data_start, data_end, offset)
    print ("Chunk: ", chunk_idx, "  Loading : ", ((data_end-data_start)*4*n_channels)*1.E-6, "MB")
    with open(filename_bin, "rb") as fin:
	if data_start==0:
	    # Seek position and read N bytes
	    recordings_1D = np.fromfile(fin, dtype='float32', count=(data_end+buffer_size)*n_channels)
	    recordings_1D = np.hstack((np.zeros(buffer_size*n_channels,dtype='float32'),recordings_1D))
	else:
	    fin.seek((data_start-buffer_size)*4*n_channels, os.SEEK_SET)         #Advance to idx_start but go back 
	    recordings_1D =  np.fromfile(fin, dtype='float32', count=((data_end-data_start+buffer_size*2)*n_channels))	#Grab 2 x template_widthx2 buffers

	if len(recordings_1D)!=((data_end-data_start+buffer_size*2)*n_channels):
	    recordings_1D = np.hstack((recordings_1D,np.zeros(buffer_size*n_channels,dtype='float32')))

    fin.close()
    
    #print buffer_size
    #print recordings_1D.shape
    rec_len = (data_end-data_start+buffer_size*2)		#Need rec_len in n_samples for below

    #Convert to 2D array
    recordings = recordings_1D.reshape(-1,n_channels)
    #print recordings.shape
    #**************************** PROCESSING **************************

    n_templates, n_shifts, waveform_size, n_channels = shifted_templates.shape
    R = int((waveform_size-1)/2)
    R2 = int((waveform_size-1)/4)
    principal_channels = np.argmax(np.max(np.abs(shifted_templates),(1,2)), 1)
    norms = np.sum(np.square(shifted_templates),(2,3))

    visible_channels = np.max(np.abs(spatial_features), (1,2)) > np.min(np.max(np.abs(spatial_features), (1,2,3)))*0.5
    #n_visible_channels = 15
    #ptp = np.max(shifted_templates, (1, 2)) - np.min(shifted_templates, (1,2))
    #visible_channels = np.argsort(ptp, 1)[:, -n_visible_channels:]
    #print visible_channels[0]
    #print principal_channels[0]
    #print visible_channels[1]
    #print principal_channels[1]
    #print visible_channels.shape


    temporal_features = temporal_features[:, :, R2:3*R2+1]

    #Use 2D arrays
    d_matrix = np.ones((recordings.shape[0],n_templates,n_shifts))*-np.Inf
    
    #************************************************************************************************************
    #******************************************** DOT PRODUCTS **************************************************
    #************************************************************************************************************
    #Dot products between waveforms and tempaltes
    #print ("Processor: ", proc_idx, "chunk: ", chunk_idx, " in dot product loop ...") 
    
    
    #print (n_templates)					# 96
    #print (spatial_features.shape)			# (96, 5, 3, 49)		#templates  x shift   x features  x channesl
    #print (temporal_features.shape)			# (96, 5, 61, 3)
    #print (spt_list.shape)				# 49 channels; contains each channel with it's spike times
    #print (len(principal_channels))			# the max channels of the 96 templates
    #for p in range(len(visible_channels)):
#	print (sum(visible_channels[p]==True))				# the channels above some threshold!? - 15 in this case
    #print (len(visible_channels[0]==True))				# the channels above some threshold!? - 15 in this case
    #print (visible_channels[1])				# the channels above some threshold!? - 15 in this case

    #dot_product = time.time()
    ctr_times=0				#Marker to ensure some spikes are present in the specific time chunk
    #n_templates = int(n_templates/2)
    #print ("Chunk: ", chunk_idx, "  in dot product loop ...")
    for k in range(n_templates):
	start_time = time.time()
	#Select spikes on the max channel of the template picked and offset to t=0 + buffer
	spt = spt_list[principal_channels[k]]
	spt = spt[np.logical_and(spt >= data_start,spt < data_end)]			#Why is this 100 sampltieim offset here!?
	spt = spt - data_start + offset
	
	#Pick channels around template 
	ch_idx = np.where(visible_channels[k])[0]	# Pick indexes for channels above threshold

	#print (n_explore)
	#print (R)
	#print (spt[:, np.newaxis])
	times = (spt[:, np.newaxis] + np.arange(-R2-n_explore, n_explore+R2+1))	#duplicate spikes -32..+33 sampletimes (65 in total) for each spike
	#print (times)
	#print times
	if len(times)==0: continue		#Skip this chunk of time, no spikes in it


	#Use 2D original arrays
	#wf_time = time.time()
	wf = ((recordings.ravel()[(
	   ch_idx + (times * recordings.shape[1]).reshape((-1,1))
	   ).ravel()]).reshape(times.size, ch_idx.size)).reshape((spt.shape[0], -1, ch_idx.shape[0]))	#This is equivalent to: wf = recordings[times][ch_idx], 
	#print "wf time: ", time.time() - wf_time				     # - it picks 65 time shifted versions of each original spike (on ch_idx channels)

	#print (wf.shape)
	#wf = recordings[times][:,:,ch_idx]
	#print (wf.shape)
	#quit()
	
	#SUMMARY: wf contains n_spikes X n_sample_pts X n_channels
	
	#print (spatial_features[k][:, :, ch_idx].shape)					#Contains n_features X n_shifts X n_channels; e.g. 5 x 3 x 15
	#print (spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx].shape)
	#print (wf[:, :, :].shape)							#Contains n_spikes X times X channels; e.g. 571 x 65 x 15
	#print (wf[:, :, np.newaxis, :, np.newaxis].shape)

	#spatial_time = time.time()
	spatial_dot = np.matmul(spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx],
				wf[:, :, np.newaxis, :, np.newaxis])[:,:,:,:,0].transpose(0, 2, 1, 3)
	#print "Spatial time: ", time.time() - spatial_time
	#print (spatial_dot.shape)

	
	dot = np.zeros((spt.shape[0], 2*n_explore+1, n_shifts))
	#dot_time = time.time()
	for j in range(2*n_explore+1):
	    dot[:, j] = np.sum(spatial_dot[:, :, j:j+2*R2+1]*temporal_features[k][np.newaxis], (2, 3))
	#print "dot time2: ", time.time()-dot_time
	
	d_matrix[spt[:, np.newaxis] + np.arange(-n_explore,n_explore+1), k] = 2*dot - norms[k][np.newaxis, np.newaxis]
    
	ctr_times+=1
    
    #print "Total dot product: ", time.time() - dot_product

	
    if ctr_times==0: 
	print ("********************************************************** no spikes 2...")
	return None
	#continue	#Skip this chunk of time entirely
	    
	    
    #************************************************************************************************************
    #******************************************** THRESHOLDING **************************************************
    #************************************************************************************************************
    #print ("Processor: ", proc_idx, "chunk: ", chunk_idx, " in thresholding loop ...") 
    print ("Chunk: ", chunk_idx, " in thresholding loop ...")

    spike_train = np.zeros((0, 2), 'int32')
    max_d = np.max(d_matrix, (1,2))
    max_val = np.max(max_d)
    threshold_time = time.time()
    thresh_ctr = 0
    while max_val > threshold_d:	
	# find spike time; get spike time from objective function; and which template it is
	#print max_val
	#check_pt1 = time.time()
	peaks = argrelmax(max_d)[0]
	idx_good = peaks[np.argmax(
	    max_d[peaks[:, np.newaxis] + np.arange(-2*R,2*R+1)],1) == (2*R)]
	spike_time = idx_good[max_d[idx_good] > threshold_d]
	template_id, max_shift = np.unravel_index(
	    np.argmax(np.reshape(d_matrix[spike_time],
				 (spike_time.shape[0], -1)),1),
	    [n_templates, n_shifts])
	#print "1 ", time.time()-check_pt1

	## prevent refractory period violation
	#start_time = time.time()
	#check_pt2=time.time()
	rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
	rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
	d_matrix[rf_area, rf_area_t] = -np.Inf
	#print "2 ", time.time()-check_pt2
	
	#*****************************************************
	#****************BOTTLENECK #1************************
	#*****************************************************
	#rf_area = np.reshape(rf_area, -1)
	#check_pt1=time.time()
	#max_d[rf_area] = np.max(d_matrix[rf_area], (1,2))
	#check_pt1 = time.time() - check_pt1
	#print "  ", check_pt1

	#*****************************************************
	#*****************************************************
	#*****************************************************

	#print "2 ", time.time()-start_time
	## update nearby times
	#start_time = time.time()
	#time_affected = np.zeros(max_d.shape[0], 'bool')

	# update nearby times
	#check_pt3=time.time()
	for j in range(spike_time.shape[0]):
	    t_neigh, k_neigh = np.where(
		d_matrix[spike_time[j]-2*R:spike_time[j]+2*R, :, 0] > -np.Inf)
	    t_neigh_abs = spike_time[j] + t_neigh - 2*R
	    d_matrix[t_neigh_abs, k_neigh] -= temp_temp[
		template_id[j], k_neigh, max_shift[j], :, t_neigh]
	#print "3 ", time.time()-check_pt3

	#for j in range(spike_time.shape[0]):					#Looping over number of spikes with batch chunk
	    #t_neigh, k_neigh = np.where(					#most expensive step						#WHY IS THERE A COMPARISON WITH -INF? IS THIS EXPENSIVE?
		#d_matrix[spike_time[j]-2*R:spike_time[j]+2*R, :, 0] > -np.Inf)
	    #t_neigh_abs = spike_time[j] + t_neigh - 2*R
	    #d_matrix[t_neigh_abs, k_neigh] -= temp_temp[			#d_matrix subtracting temp_temp, predefined deconvolution subtraction
		#template_id[j], k_neigh, max_shift[j], :, t_neigh]
	    #time_affected[t_neigh_abs] = 1
	#print "3 ", time.time()-start_time
	
	#start_time=time.time()

	#*****************************************************
	#****************BOTTLENECK #2************************
	#*****************************************************
	#check_pt4=time.time()
	time_affected = np.reshape(spike_time[:, np.newaxis] + np.arange(-2*R,2*R+1), -1)
	time_affected = time_affected[max_d[time_affected] > -np.Inf]
	max_d[time_affected] = np.max(d_matrix[time_affected], (1,2))
	#print "4 ", time.time()-check_pt4

	#max_d[time_affected] = max_function(d_matrix[time_affected])

	#max_d[time_affected] = max_function(d_matrix[time_affected])		#OLD ONE
	#check_pt2 = time.time()-check_pt2
	#print "  ", check_pt2
	#*****************************************************
	#*****************************************************
	#*****************************************************
	
	#start_time1=time.time()
	#check_pt5=time.time()
	max_val = np.max(max_d)
	#print "  ", time.time()-start_time1
	
	spike_train_temp = np.hstack((spike_time[:, np.newaxis],
				      template_id[:, np.newaxis]))
	spike_train = np.concatenate((spike_train, spike_train_temp), 0)         
	#print "4 ", time.time()-start_time, "\n\n\n"
	#print "5 ", time.time()-check_pt5, '\n\n\n\n'

	#print "total threshold loop time: ", time.time()-threshold_time, " % in dmax: ", int((check_pt2)/(time.time()-threshold_time)*100)
    
	thresh_ctr+=1
    
    #Fix indexes
    spike_times = spike_train[:, 0]
    # get only observations outside the buffer
    train_not_in_buffer = spike_train[np.logical_and(spike_times >= offset,
						     spike_times <= idx_local_end)]
    # offset spikes depending on the absolute location
    train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + data_start
				 - buffer_size)

    print ("Chunk: ", chunk_idx, "cleaned spikes: ", len(train_not_in_buffer), " chunk time: ", int(time.time() - chunk_time), " thershold time: ", int(time.time() - threshold_time), " thresh_ctr: ", thresh_ctr)

    #.append(train_not_in_buffer)
    #chunk_idx+=1

    return train_not_in_buffer


#********************************************************************************************************************************************
#********************************************************************************************************************************************
#********************************************************************************************************************************************
def deconvolve_single(data_in, filename_bin, path_to_spt_list, path_to_temp_temp, path_to_shifted_templates, template_width, n_channels, temporal_features, spatial_features, n_explore, threshold_d):

    spt_list = np.load(path_to_spt_list)
    temp_temp = np.load(path_to_temp_temp)
    shifted_templates = np.load(path_to_shifted_templates)


    idx_list, proc_idx = data_in[0], data_in[1]
    #New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]
    idx_local_end = idx_list[3]

    data_start = idx_start  #idx[0].start
    data_end = idx_stop    #idx[0].stop
    # get offset that will be applied
    offset = idx_local   #idx_local[0].start

    parallel=1
    if parallel:
	data_start_array = []
	data_end_array = []
	offset_array = []
	spike_train_array = []
	
	n_chunks = 50
	#print data_end, data_start,n_chunks
	chunks = np.int32(np.linspace(data_start,data_end, n_chunks+1))
	#print chunks
	
	data_start_array= (chunks[:-1])
	data_end_array=chunks[1:]
	offset_array = np.repeat(offset,n_chunks)
	
	#print data_start_array
	#print data_end_array
	#print offset_array
	#quit()
	
    chunk_idx = 0
    for data_start, data_end, offset in zip(data_start_array,data_end_array,offset_array):
	#************************* LOAD RECORDINGS  **************************
	print "Processor: ", proc_idx, "  Loading : ", ((data_end-data_start)*4*n_channel)*1.E-6, "MB,  chunk: ", chunk_idx, "  of ", n_chunks
	with open(filename_bin, "rb") as fin:
	    if data_start==0:
		# Seek position and read N bytes
		recordings_1D =  np.fromfile(fin, dtype='float32', count=(data_end+template_width*2)*n_channels*4)
		recordings_1D = np.hstack((np.zeros(template_width*2*n_channels,dtype='float32'),recordings_1D))
	    else:
		fin.seek((data_start-template_width*2)*4*n_channel, os.SEEK_SET)         #Advance to idx_start but go back 
		recordings_1D =  np.fromfile(fin, dtype='float32', count=((data_end-data_start+template_width*4)*n_channels*4))	#Grab 2 x template_widthx2 buffers
		
	#print (recordings_1D.shape)
	rec_len = (data_end-data_start+template_width*4)		#Need rec_len in n_samples for below
	#print rec_len

	#Convert to 2D array
	recordings = recordings_1D.reshape(-1,n_channels)
	#recordings=recordings_1D
	#print (recordings.shape)
	#print recordings
	#**************************** PROCESSING **************************

	n_templates, n_shifts, waveform_size, n_channels = shifted_templates.shape
	R = int((waveform_size-1)/2)

	principal_channels = np.argmax(np.max(np.abs(shifted_templates),(1,2)), 1)
	norms = np.sum(np.square(shifted_templates),(2,3))

	visible_channels = np.max(np.abs(spatial_features), (1,2)) > np.min(np.max(np.abs(spatial_features), (1,2,3)))*0.5

	#Use 2D arrays
	d_matrix = np.ones((recordings.shape[0],
	#Use 1D arrays
	#d_matrix = np.ones((rec_len,
			    n_templates,
			    n_shifts))*-np.Inf
	
	#Dot products between waveforms and tempaltes
	print ("... in dot product loop ...") 
	for k in range(n_templates):
	    spt = spt_list[principal_channels[k]]
	    spt = spt[np.logical_and(spt >= data_start + 100,
				     spt < data_end - 100)]
	    spt = spt - data_start + offset
	    ch_idx = np.where(visible_channels[k])[0]
	    times = (spt[:, np.newaxis] + np.arange(
		    -R-n_explore, n_explore+R+1))

	    #Use 2D original arrays
	    wf = (((recordings.ravel()[(
	       ch_idx + (times * recordings.shape[1]).reshape((-1,1))
	    
	    #Use 1D array so don't have to reformat
	    #wf = (((recordings[(
	    #   ch_idx + (times * n_channels).reshape((-1,1))
	    
	       ).ravel()]).reshape(times.size, ch_idx.size)).reshape((spt.shape[0], -1, ch_idx.shape[0])))

	    spatial_dot = np.matmul(spatial_features[k][np.newaxis, np.newaxis, :, :, ch_idx],
				    wf[:, :, np.newaxis, :, np.newaxis]
				   )[:,:,:,:,0].transpose(0, 2, 1, 3)
	    
	    dot = np.zeros((spt.shape[0], 2*n_explore+1, n_shifts))
	    for j in range(2*n_explore+1):
		dot[:, j] = np.sum(spatial_dot[
		    :, :, j:j+2*R+1]*temporal_features[k][np.newaxis], (2, 3))
	    
	    d_matrix[spt[:, np.newaxis] + np.arange(-n_explore,n_explore+1), k] = 2*dot  - \
		norms[k][np.newaxis, np.newaxis]
	
	
	spike_train = np.zeros((0, 2), 'int32')
	max_d = np.max(d_matrix, (1,2))
	max_val = np.max(max_d)
	print ("... in thresholding loop...")
	while max_val > threshold_d:	
	#while max_val > 10000:	
	    #print "Processor: ", proc_idx, "max_val: ", max_val				
	    # find spike time; get spike time from objective function; and which template it is
	    peaks = argrelmax(max_d)[0]
	    idx_good = peaks[np.argmax(
		max_d[peaks[:, np.newaxis] + np.arange(-2*R,2*R+1)],1) == (2*R)]
	    spike_time = idx_good[max_d[idx_good] > threshold_d]
	    template_id, max_shift = np.unravel_index(
		np.argmax(np.reshape(d_matrix[spike_time],
				     (spike_time.shape[0], -1)),1),
		[n_templates, n_shifts])

	    # prevent refractory period violation
	    rf_area = spike_time[:, np.newaxis] + np.arange(-R,R+1)
	    rf_area_t = np.tile(template_id[:,np.newaxis],(1, 2*R+1))
	    d_matrix[rf_area, rf_area_t] = -np.Inf
	    rf_area = np.reshape(rf_area, -1)
	    max_d[rf_area] = np.max(d_matrix[rf_area], (1,2))
	    
	    # update nearby times
	    time_affected = np.zeros(max_d.shape[0], 'bool')
	    for j in range(spike_time.shape[0]):					#Looping over number of spikes with batch chunk
		t_neigh, k_neigh = np.where(					#most expensive step
		    d_matrix[spike_time[j]-2*R:spike_time[j]+2*R, :, 0] > -np.Inf)
		t_neigh_abs = spike_time[j] + t_neigh - 2*R
		d_matrix[t_neigh_abs, k_neigh] -= temp_temp[			#d_matrix subtracting temp_temp, predefined deconvolution subtraction
		    template_id[j], k_neigh, max_shift[j], :, t_neigh]
		time_affected[t_neigh_abs] = 1
		
	    max_d[time_affected] = np.max(d_matrix[time_affected], (1,2))
	    max_val = np.max(max_d)
	    
	    spike_train_temp = np.hstack((spike_time[:, np.newaxis],
					  template_id[:, np.newaxis]))
	    spike_train = np.concatenate((spike_train, spike_train_temp), 0)         
	
	
	#Fix indexes
	spike_times = spike_train[:, 0]
	# get only observations outside the buffer
	train_not_in_buffer = spike_train[np.logical_and(spike_times >= idx_local,
							 spike_times <= idx_local_end)]
	# offset spikes depending on the absolute location
	train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + idx_start
				     - template_width*2)
	#print ("fixed indexes, len: ", len(train_not_in_buffer))


	spike_train_array.append(train_not_in_buffer)
	chunk_idx+=1
	
    return np.vstack(spike_train_array)
    
    

#def fix_indexes(spike_train, idx_local, idx, buffer_size):
def fix_indexes(spike_train, data_start,data_end,offset,buffer_size):
    """Fixes indexes from detected spikes in batches

    Parameters
    ----------
    res: tuple
        A tuple with the results from the nnet detector
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size
    """

    # get limits for the data (exlude indexes that have buffer data)
    #data_start = idx_local[0].start
    #data_end = idx_local[0].stop
    # get offset that will be applied
    #offset = idx[0].start

    # fix clear spikes
    spike_times = spike_train[:, 0]
    # get only observations outside the buffer
    train_not_in_buffer = spike_train[np.logical_and(spike_times >= data_start,
                                                     spike_times <= data_end)]
    # offset spikes depending on the absolute location
    train_not_in_buffer[:, 0] = (train_not_in_buffer[:, 0] + offset
                                 - buffer_size)
    print ("fixed indexes, len: ", len(train_not_in_buffer))
    return train_not_in_buffer
