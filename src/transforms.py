"""Image and Spectogram Transforms for Training"""
import random

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import numpy as np
import librosa

SEED = 239
random.seed(SEED)
np.random.seed(SEED)

class SpecAugment(ImageOnlyTransform):
    """Shifting time axis"""
    def __init__(
        self,
        num_mask=5,
        freq_masking=0.6,
        time_masking=0.5,
        always_apply=False,
        prob=0.7
        ):
        super(SpecAugment, self).__init__(always_apply, prob)
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def apply(self, data: np.ndarray):
        """
        Method to apply transform on dataset

        Args:
            data: raw data
        Returns:
            Augmented dataset
        """
        melspec = data
        spec_aug = self.spec_augment(melspec,)
        return spec_aug

    def spec_augment(
        self,
        spec: np.ndarray
        ):
        """
        Method to apply spectogram augmentations

        Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation

        Args:
            spec: spectogram data
        Return:
            Augmented spectotgram
        """
        spec = spec.copy()
        value = spec.min()
        num_mask = random.randint(1, self.num_mask)
        for _ in range(num_mask):
            all_freqs_num, all_frames_num  = spec.shape
            freq_percentage = random.uniform(0.0, self.freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f_0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f_0 = int(f_0)
            spec[f_0:f_0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, self.time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t_0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t_0 = int(t_0)
            spec[:, t_0:t_0 + num_frames_to_mask] = value

        return spec


class SpectToImage(ImageOnlyTransform):
    """
    Convert Spectogram to image data
        by stacking image first and second derivate information
    """

    def apply(self, data: np.ndarray):
        """
        Method to apply transform on dataset

        Args:
            data: raw data
        Returns:
            Augmented dataset
        """
        image = data.copy()
        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=-1)
        image = image.astype(np.float32) / 100.0
        assert image.shape[-1] == 3
        return image

def generate_transform(data_type: str):
    """
    Transforms to use for train and validation data

    Args:
        data_type: data to return
    Returns:
        Dictionary containing transforms for visual and audio data
    """
    if data_type == "train":
        return {
            'visual':A.Compose([
                A.Resize(227,227),
                A.RandomRotate90(),
                A.Flip(),
                A.OneOf([A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                    ),
                A.RGBShift(
                    r_shift_limit=25,
                    g_shift_limit=25,
                    b_shift_limit=25,
                    p=0.5
                    ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                    ),
                A.HueSaturationValue(p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]),
            'audio':A.Compose([
                SpecAugment(),
                SpectToImage(),
                A.Resize(227,227),
                #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
                ])
            }
    # Do not apply probabilistic transforms to validation data
    if data_type == "val":

        return {'visual': A.Compose([
            A.Resize(227,227),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ]),
            'audio':A.Compose([
                SpectToImage(),
                A.Resize(227,227),
                #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
                ])
            }
            