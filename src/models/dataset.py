"""Define Audio-Visual Dataset as pytorch Dataset"""

from torch.utils.data import TensorDataset
from torch import is_tensor

class AudioVisualDataset(TensorDataset):
    """
    Audio Visual dataset treats audio and visual data as
    TensorDataset to yield (audio, visual, label) items
    """
    def __init__(self, 
               audio_file, 
               visual_file, 
               labels_file,
               *args,
               **kwargs
               ):
        """
        Args:
        audio_file: file name for audio
        visual_file: file name for images
        labels_file: file with labels for image and audio
        """
        super().__init__(*args, **kwargs)
        self.audio_file = audio_file
        self.visual_file = visual_file
        self.labels = labels_file

    def __len__(self):
        return len(self.audio_file) 

    def __getitem__(self, idx):

        if is_tensor(idx):
            idx = idx.tolist()
            image = self.visual_file[idx]
            audio = self.audio_file[idx]

        label = self.labels[idx]

        return audio, image, label
