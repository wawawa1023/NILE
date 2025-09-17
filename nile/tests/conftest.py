"""
Test Configuration

テスト設定ファイル
"""

import pytest
import tempfile
import os
from nile.core.storage.knowledge_base import KnowledgeBase
from nile.core.reasoning.symbolic_engine import SymbolicEngine

@pytest.fixture
def temp_knowledge_file():
    """一時的な知識ベースファイルのフィクスチャ"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    temp_file.close()
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def knowledge_base(temp_knowledge_file):
    """知識ベースのフィクスチャ"""
    return KnowledgeBase(temp_knowledge_file)

@pytest.fixture
def symbolic_engine(temp_knowledge_file):
    """シンボリックエンジンのフィクスチャ"""
    return SymbolicEngine(temp_knowledge_file)

@pytest.fixture
def sample_facts():
    """サンプル事実のフィクスチャ"""
    return [
        {"subject": "猫", "relation": "種類", "object": "動物"},
        {"subject": "犬", "relation": "種類", "object": "動物"},
        {"subject": "鳥", "relation": "種類", "object": "動物"},
        {"subject": "猫", "relation": "属性", "object": "かわいい"},
        {"subject": "犬", "relation": "属性", "object": "忠実"},
        {"subject": "鳥", "relation": "属性", "object": "自由"}
    ]

@pytest.fixture
def sample_questions():
    """サンプル質問のフィクスチャ"""
    return [
        "猫は動物ですか？",
        "犬は何ですか？",
        "鳥の特徴は何ですか？",
        "動物の種類は何がありますか？"
    ]
