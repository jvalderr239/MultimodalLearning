#!/usr/bin/env python

from distutils.core import setup
import setuptools

setup(
    name="Audio-Visual-Multimodal-Learning",
    version='1.0.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires="==3.9",
    install_requires=[
        "albumentations==1.3.0",
        "fastai==2.7.10",
        "librosa==0.9.1",
        "matplotlib==3.4.3",
        "numpy>=1.19",
        "pandas>=1.3.1",
        "scikit_learn==1.2.1",
        "seaborn==0.11.2",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm==4.62.3"
        ],
    packages=setuptools.find_packages(),
    scripts=['src/scripts/run_preprocessing.py', 'src/scripts/train.py']
)