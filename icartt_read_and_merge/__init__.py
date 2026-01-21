"""Set of Functions that allow you to read in Icartt Files and Merge them."""

from .icartt_read_and_merge import (
    read_icartt,
    master_icartt_time_parser,
    align2master_timeline,
    icartt_merger,
    icartt_time_to_datetime,
)
from .ancillary_utils import (
    read_size_distribution_radii,
    ensure_temporal_alignment,
)

__all__ = [
    'read_icartt',
    'master_icartt_time_parser',
    'align2master_timeline',
    'icartt_merger',
    'icartt_time_to_datetime',
    'read_size_distribution_radii',
    'ensure_temporal_alignment',
]

__version__ = '0.1.0'

