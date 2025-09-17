"""
NILE - NeuroSymbolic Interactive Language Engine

日本語自然言語理解とシンボリック推論を組み合わせた実験的プロトタイプ
"""

__version__ = "0.2.0"
__author__ = "NILE Development Team"
__email__ = "nile@example.com"

from .core.reasoning.symbolic_engine import SymbolicEngine
from .core.storage.knowledge_base import KnowledgeBase
from .core.nlp.japanese_processor import JapaneseProcessor

__all__ = [
    "SymbolicEngine",
    "KnowledgeBase", 
    "JapaneseProcessor"
]
