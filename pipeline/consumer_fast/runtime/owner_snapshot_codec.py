"""Trusted local snapshot encoding preserving immutable mapping wrappers."""
import copyreg
from types import MappingProxyType
def restore_mapping(items):return MappingProxyType(items)
def reduce_mapping(value):return restore_mapping,(dict(value),)
copyreg.pickle(type(MappingProxyType({})),reduce_mapping)
