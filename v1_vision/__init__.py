"""
V1 Vision Pipeline
Biologically-inspired visual processing using spiking neural networks
"""

__version__ = '0.1.0'
__author__ = 'Alexander Speer'

from .gabor_feature_extractor import GaborFeatureExtractor
from .spike_encoder import SpikeEncoder
from .v1_model_interface import V1ModelInterface
from .v1_decoder import V1Decoder
from .visualization import SpikeRasterPlot, PipelineMonitor, MultiWindowDisplay

__all__ = [
    'GaborFeatureExtractor',
    'SpikeEncoder',
    'V1ModelInterface',
    'V1Decoder',
    'SpikeRasterPlot',
    'PipelineMonitor',
    'MultiWindowDisplay',
]

