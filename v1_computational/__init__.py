"""
Computational V1 Vision Pipeline

A fast, functional implementation of primary visual cortex (V1) processing
that replicates the MDPI2021 V1 orientation column architecture.

Components:
- Gabor feature extraction (orientation-selective filters)
- Spike encoding (latency coding)
- V1 model (4 orientation columns, 8 layers each, 4,668 neurons)
- Orientation map decoding

Usage:
    from v1_computational.pipeline import V1VisionPipeline
    
    pipeline = V1VisionPipeline()
    results = pipeline.process_frame(frame)
"""

__version__ = '1.0.0'
__author__ = 'Blindsight Project'

from .pipeline import V1VisionPipeline
from .v1_model import ComputationalV1Model
from .v1_column import V1OrientationColumn
from .neurons import LIFNeuron, NeuronPopulation
from .gabor_extractor import GaborFeatureExtractor
from .spike_encoder import SpikeEncoder
from .v1_decoder import V1Decoder

__all__ = [
    'V1VisionPipeline',
    'ComputationalV1Model',
    'V1OrientationColumn',
    'LIFNeuron',
    'NeuronPopulation',
    'GaborFeatureExtractor',
    'SpikeEncoder',
    'V1Decoder',
]

