import os,re,sys,joblib,scipy, copy, time, torch, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.signal import medfilt, resample
from scipy.ndimage import gaussian_filter1d
sys.path.append('../../../')
from signal_processing_functions import lowpass_filter
from torch.utils.data import Dataset

# read data from CHARIS database
def read_data_charis(file_path,sorted_dat_files,j):
    headfile_name = sorted_dat_files[j].replace('.dat','.hea')
    head = open(file_path+headfile_name, 'r').read()
    data = np.fromfile(file_path+sorted_dat_files[j], dtype='int16')
    header = re.split(r"[ )(|\n]", head)
    fs = int(header[2])
    # ABP = (data[0::3]-float(header[7]))/float(header[6])
    ICP = (data[2::3]-float(header[29]))/float(header[28])
    # time_axis = np.arange(0, len(ICP), 1)/fs/60
    ICP_filtered = lowpass_filter(ICP, 10, 15, 1, 50, fs)
    ICP_filtered = resample(ICP_filtered, int(len(ICP_filtered)*25/50))
    fs = 25
    return ICP_filtered,fs

# read data from KCH database
def read_data_kch(file_path,sorted_dat_files):
    ICP_na_files = [file for file in os.listdir(file_path+sorted_dat_files) if file.startswith("ICP,na")]
    ICP_na_files = sorted(ICP_na_files)
    ICP_na_data = []
    for j in ICP_na_files:
        ICP_na_data.append(scipy.io.loadmat(file_path+sorted_dat_files+'/'+j)['measurement_data'])
    ICP_na_data = np.vstack(ICP_na_data).reshape(-1)
    ICP_filtered = lowpass_filter(ICP_na_data, 10, 15, 1, 50, 125)
    # donwsample to 50 Hz
    ICP_filtered = resample(ICP_filtered, int(len(ICP_filtered)*25/125))
#     return ICP_filtered,int(1/(scipy.io.loadmat(file_path+sorted_dat_files+'/'+j)['time_vector'][0][1] - scipy.io.loadmat(file_path+sorted_dat_files+'/'+j)['time_vector'][0][0]))
    fs = 25
    return ICP_filtered,fs

# reshape the data into 5 minutes segements
def get_reshaped_data(file_path,file_name, j,time_minutes,look_back_minutes, start_point, end_point):
    # Calculate the number of samples in each chunk (window)
    periodic_component = joblib.load(file_path[0] + file_name[0][j])
    slow_wave = joblib.load(file_path[1] + file_name[1][j])
    if j==0:
        raw_data,fs = read_data_kch(file_path[2], file_name[2][j])
        raw_data = raw_data[433000:433000+4995000]
        slow_wave = slow_wave[0:4995000]
    elif j==1:
        raw_data,fs = read_data_kch(file_path[2], file_name[2][j])
        raw_data = raw_data[12300:12300+3030000]
        slow_wave = slow_wave[0:3030000]
    else:
        raw_data,fs = read_data_charis(file_path[3], file_name[3],j-2)
    # return periodic_components, slow_wave, raw_data
    
    time_samples = int(time_minutes)
    stride_samples = int((time_minutes - look_back_minutes))
    data_reshaped = []
    # Extract the relevant portion of the data
    for chunk in range(len(start_point)):
        periodic_components = periodic_component[start_point[chunk]:end_point[chunk]]
        slow_waves = slow_wave[start_point[chunk]:end_point[chunk]]
        raw_datas = raw_data[start_point[chunk]:end_point[chunk]]
        total_samples = len(periodic_components)
        # print(total_samples)
        num_rows = ((total_samples - time_samples) // stride_samples) + 1
        num_cols = time_samples
        # Initialize a list to hold the chunks
        for i in range(num_rows):
            start_idx = i * stride_samples
            end_idx = start_idx + time_samples
            if end_idx > total_samples:
                # If there are not enough samples left for a full chunk, break the loop
                break
            periodic_components_chunk = periodic_components[start_idx:end_idx]
            slow_wave_chunk = slow_waves[start_idx:end_idx]
            raw_data_chunk = raw_datas[start_idx:end_idx]
            chunk_info = {
                            'patient_id': file_name[0][j][19:-4],
                            'chunk_number': i+1,
                            'segments': chunk,
                            'start_index': start_idx + start_point[chunk],
                            'end_index': end_idx + start_point[chunk],
                            'periodic_components':periodic_components_chunk,
                            'slow_wave_chunk':slow_wave_chunk,
                            'raw_data':raw_data_chunk
                        }
            data_reshaped.append(chunk_info)

    return data_reshaped

# load data from raw files to dict
def load_data(time_minutes,look_back_minutes,full=1,remote = False):
    if remote:
        periodic_components_path = 'codes/Python/mimic-iii/variables/periodic_components_51_10/'
        slow_wave_tvd_path = 'codes/Python/mimic-iii/variables/slow_wave_tvd/'
        raw_wave_charis_path = 'data/charis database/'
        raw_wave_kch_path = 'data/Sharon/'
    else:
        periodic_components_path = '../../../../variables/periodic_components_51_10/'
        slow_wave_tvd_path = '../../../../variables/slow_wave_tvd/'
        raw_wave_charis_path = '../../../../../../../data/charis database/'
        raw_wave_kch_path = '../../../../../../../data/Sharon/'
    
    file_path = [periodic_components_path,slow_wave_tvd_path,raw_wave_kch_path,raw_wave_charis_path]
    
    periodic_components_files = os.listdir(periodic_components_path)
    periodic_components_files.sort()
    
    slow_wave_files = os.listdir(slow_wave_tvd_path)
    # slow_wave_files = [f for f in slow_wave_files if f.startswith('slow_wave_tvd_ch')]
    slow_wave_files.sort()
    
    raw_wave_charis_files = os.listdir(raw_wave_charis_path)
    raw_wave_charis_files = [f for f in raw_wave_charis_files if f.endswith('.dat')]
    raw_wave_charis_files.sort()
    
    raw_wave_kch_files = os.listdir(raw_wave_kch_path)
    raw_wave_kch_files = [f for f in raw_wave_kch_files if f.startswith('KCH173 ')]
    raw_wave_kch_files.sort()
    
    file_name = [periodic_components_files, slow_wave_files, raw_wave_kch_files, raw_wave_charis_files]

    # time_minutes = 5
    # look_back_minutes = 2.5
    
    # Initialize data containers
    train_data = []
    val_data = []
    test_data = []

      # Test data: patient 11 (all data)
    test_data = test_data + get_reshaped_data(file_path, file_name, 0, time_minutes, look_back_minutes, start_point=[0, 740000, 740000+1350000+1200000, 740000+1350000+1200000+1440000+500000], end_point=[600000, 740000+1350000, 740000+1350000+1200000+1440000, 740000+1350000+1200000+1440000+500000+1500000])
    # print('Loaded test data: patient 11')
    
    # Validation data: patient 12 (all data)
    val_data = val_data + get_reshaped_data(file_path, file_name, 1, time_minutes, look_back_minutes, start_point=[0, 1995000+2500000], end_point=[1995000, 1995000+2500000+1695000])
    # print('Loaded validation data: patient 12')
    
    # Training data: patients 0, 1, 3, 8, 9, 10 (all data)
    train_data = train_data + get_reshaped_data(file_path, file_name, 3, time_minutes, look_back_minutes, start_point=[0], end_point=[4995000])
    # print('Loaded training data: patient 0/6')
    
    if full!=1:
        return train_data, val_data, test_data
        
    train_data = train_data + get_reshaped_data(file_path, file_name, 8, time_minutes, look_back_minutes, start_point=[0], end_point=[3030000])
    # print('Loaded training data: patient 1/6')
    
    train_data = train_data + get_reshaped_data(file_path, file_name, 9, time_minutes, look_back_minutes, start_point=[0, 3450000], end_point=[2250000, 6540000])
    # print('Loaded training data: patient 9/6')
    
    train_data = train_data + get_reshaped_data(file_path, file_name, 10, time_minutes, look_back_minutes, start_point=[0, 14150000], end_point=[12990000, 14150000+12000000])
    # print('Loaded training data: patient 10/6')
    
    train_data = train_data + get_reshaped_data(file_path, file_name, 11, time_minutes, look_back_minutes, start_point=[0, 740000, 740000+1350000+1200000, 740000+1350000+1200000+1440000+500000], end_point=[600000, 740000+1350000, 740000+1350000+1200000+1440000, 740000+1350000+1200000+1440000+500000+1500000])
    # print('Loaded training data: patient 11/6')

    train_data = train_data + get_reshaped_data(file_path, file_name, 12, time_minutes, look_back_minutes, start_point=[0, 1995000+2500000], end_point=[1995000, 1995000+2500000+1695000])
    # print('Loaded training data: patient 12/6')

    
    return train_data,val_data, test_data

# create sequences and contain missing parts
def create_sequences(data, num_prev_chunks=2, num_missing_chunks=12, num_next_chunks=2):
    sequences = []
    data_by_patient_segment = {}
    
    # Group data by patient_id and segment
    for chunk in data:
        pid = chunk['patient_id']
        segment_id = chunk['segments']
        if pid not in data_by_patient_segment:
            data_by_patient_segment[pid] = {}
        if segment_id not in data_by_patient_segment[pid]:
            data_by_patient_segment[pid][segment_id] = []
        data_by_patient_segment[pid][segment_id].append(chunk)
    
    # For each patient and segment, create sequences
    for pid, segments in data_by_patient_segment.items():
        for segment_id, chunks in segments.items():
            # Sort chunks by chunk_number within the segment
            chunks.sort(key=lambda x: x['chunk_number'])
            num_chunks = len(chunks)
            # Adjust the range based on the number of chunks
            for i in range(num_prev_chunks, num_chunks - num_missing_chunks - num_next_chunks + 1):
                # Context chunks
                prev_chunks = chunks[i - num_prev_chunks:i]
                # Missing chunks
                missing_chunks = chunks[i:i + num_missing_chunks]
                # Later chunks
                next_chunks = chunks[i + num_missing_chunks:i + num_missing_chunks + num_next_chunks]
                
                # Check if we have enough later chunks for context
                if len(next_chunks) < num_next_chunks:
                    continue  # Skip if not enough later chunks
                
                # Create individual data points for each missing chunk
                for idx, missing_chunk in enumerate(missing_chunks):
                    sequence = {
                        'patient_id': pid,
                        'relative_chunk_number': idx,
#                         'segment_id': segment_id,
                        'prev_chunks': prev_chunks,
                        'next_chunks': next_chunks,
                        'missing_chunk': missing_chunk
                    }
                    sequences.append(sequence)
    return sequences

def positional_encoding(positions, d_model):
    """
    Generate sinusoidal positional encodings for a batch of positions.

    Args:
        positions (torch.Tensor): Shape (batch_size,)
        d_model (int): Embedding dimension

    Returns:
        torch.Tensor: Shape (batch_size, d_model)
    """
    positions = positions.unsqueeze(1)  # (batch_size, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(positions.size(0), d_model)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

# create dataset
class ICPDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.patient_ids = list({seq['patient_id'] for seq in sequences})
        # Create a mapping from patient_id to integer index
        self.patient_id_to_idx = {pid: idx for idx, pid in enumerate(self.patient_ids)}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        patient_id = sequence['patient_id']
        patient_idx = self.patient_id_to_idx[patient_id]
        missing_chunk = sequence['missing_chunk']
        relative_chunk_number = sequence['relative_chunk_number']
#         segment_id =  sequence['segment_id']
        
        # Prepare context data separately
        prev_chunks = sequence['prev_chunks']
        next_chunks = sequence['next_chunks']
        
        # Extract periodic_components from previous and next chunks separately
        prev_context_pc = np.concatenate([chunk['periodic_components'] for chunk in prev_chunks])
        next_context_pc = np.concatenate([chunk['periodic_components'] for chunk in next_chunks])
        
        # Extract periodic_components from the missing chunk
        target_pc = missing_chunk['periodic_components']
        
        # Conditional data
        chunk_number = relative_chunk_number  # Relative chunk number within the missing sequence
        start_index = missing_chunk['start_index']
        end_index = missing_chunk['end_index']
#         position_embedding_dim = 256
#         start_pos_enc = positional_encoding(torch.tensor([start_index]), position_embedding_dim).squeeze(0)
#         end_pos_enc = positional_encoding(torch.tensor([end_index]), position_embedding_dim).squeeze(0)

        
        # Convert to tensors
        prev_context_pc = torch.from_numpy(prev_context_pc).float()
        next_context_pc = torch.from_numpy(next_context_pc).float()
        target_pc = torch.from_numpy(target_pc).float()
        patient_idx = torch.tensor(patient_idx).long(),
        chunk_number = torch.tensor(chunk_number).long(),
#       segment_id': torch.tensor(segment_id).long(),
        start_pos = start_index,
        end_pos = end_index

        return prev_context_pc, next_context_pc, target_pc, patient_idx[0], chunk_number, start_pos, end_pos

