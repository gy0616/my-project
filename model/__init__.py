"""
DUHA-Net Model Package

This package contains the complete implementation of DUHA-Net:
    - Haar Enhancement Module (HEM)
    - Semantic-Guided Decoupling Aggregation (S-GDA)
    - Uncertainty-aware Dual-branch Training (UDT)
    - Main DUHA-Net model
"""

from .hem import HaarEnhancementModule
from .sgda import SemanticGuidedDecouplingAggregation
from .udt import UncertaintyAwareDualTraining, UDTTrainer
from .duha_net import DUHANet, create_model, count_parameters

__all__ = [
    'HaarEnhancementModule',
    'SemanticGuidedDecouplingAggregation',
    'UncertaintyAwareDualTraining',
    'UDTTrainer',
    'DUHANet',
    'create_model',
    'count_parameters',
]