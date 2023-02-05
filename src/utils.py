""" Utility functions to process audio-visual dataset"""
import numpy as np
import torch

DSIZE = 227
DATAPATH = "./drive/MyDrive/Multimodal/Data/"

def _get_file(dtype: str, phase: str):
    """
    Return numpy array from filename

    Args:
        dtype: data type
        phase: phase type
    Returns:
        Numpy array from file
    """
    return np.load(
        DATAPATH + dtype + '/' + phase + '/' + phase + '_' + dtype + '.npy',
        allow_pickle=True
        )

def np_to_torch(data: np.ndarray):
    """
    Convert numpy data to pytorch Tensor

    Args:
        data: numpy array
    Returns:
        Float Tensor
    """
    return [torch.from_numpy(np.array(item)).float() for item in data]

def train_test_split(data: np.ndarray, percent: float):
    """
    Generate train, test, split data
    """
    datasize = data.shape[0]
    print(f"Splitting {datasize} sets")
    pct = int(datasize * percent)
    t_data, rem_data = data[:pct], data[pct:]
    return t_data, rem_data

def load_data(key: str):
    """
    Load dataset from file directories baased on type

    Args: 
        key: data type
    """
    drivedata = {
        'audio':'audio/audio.npy',
        'visual':'visual/visual.npy',
        'labels':'labels/labels.npy'
    }
    return np.load(DATAPATH + drivedata[key], allow_pickle=True)

def save_data(data: np.ndarray, dtype: str, phase: str):
    """
    Stor numpy array locally in pickle format

    Args:
        data: Numpy array to store
        dtype: data type
        phase: phase type
    """
    print(f"Saving {dtype} file in {phase} folder.")
    np.save(
        DATAPATH + dtype + '/' + phase + '/' + phase + '_'+ dtype + '.npy',
        data, 
        allow_pickle=True)
    print("Done saving.")

def process_data(dtype: str):
    """
    Split and store train test val dataset

    Args:
        dtype: type of data
    """
    data = load_data(dtype)
    print(dtype)
    t_x, rem_data = train_test_split(data, percent=0.8)
    v_x, test_x = train_test_split(rem_data, percent=0.5)
    save_data(t_x, dtype, 'train')
    save_data(v_x, dtype, 'val')
    save_data(test_x, dtype, 'test')

def load_processed_data(phase: str):
    """
    Load processed data

    """
    print(f"Loading {phase} data...")
    audio = _get_file('audio', phase)
    visual = _get_file('visual', phase)
    labels = _get_file('labels', phase)
    return np_to_torch(audio), np_to_torch(visual), np_to_torch(labels)
    