"""
Knowledge Base Tests

知識ベースのテスト
"""

import pytest
import tempfile
import os
from nile.core.storage.knowledge_base import KnowledgeBase

class TestKnowledgeBase:
    """知識ベースのテストクラス"""
    
    def setup_method(self):
        """テストのセットアップ"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.kb = KnowledgeBase(self.temp_file.name)
    
    def teardown_method(self):
        """テストのクリーンアップ"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_add_fact(self):
        """事実の追加テスト"""
        result = self.kb.add_fact("猫", "動物", "種類")
        assert result is True
        
        facts = self.kb.get_facts()
        assert len(facts) > 0
        assert any(fact["subject"] == "猫" and fact["object"] == "動物" for fact in facts)
    
    def test_duplicate_fact(self):
        """重複事実のテスト"""
        # 最初の追加
        result1 = self.kb.add_fact("犬", "動物", "種類")
        assert result1 is True
        
        # 重複する追加
        result2 = self.kb.add_fact("犬", "動物", "種類")
        assert result2 is False
    
    def test_get_relation(self):
        """関係の取得テスト"""
        self.kb.add_fact("鳥", "動物", "種類")
        
        relation = self.kb.get_relation("鳥", "動物")
        assert relation is not None
        assert relation["relation"] == "種類"
    
    def test_get_related_nodes(self):
        """関連ノードの取得テスト"""
        self.kb.add_fact("猫", "動物", "種類")
        self.kb.add_fact("猫", "かわいい", "属性")
        
        related = self.kb.get_related_nodes("猫")
        assert "動物" in related
        assert "かわいい" in related
    
    def test_has_path(self):
        """パスの存在テスト"""
        self.kb.add_fact("猫", "動物", "種類")
        self.kb.add_fact("動物", "生き物", "種類")
        
        assert self.kb.has_path("猫", "生き物") is True
        assert self.kb.has_path("猫", "植物") is False
    
    def test_get_path(self):
        """パスの取得テスト"""
        self.kb.add_fact("猫", "動物", "種類")
        self.kb.add_fact("動物", "生き物", "種類")
        
        path = self.kb.get_path("猫", "生き物")
        assert path is not None
        assert path == ["猫", "動物", "生き物"]
    
    def test_search_facts(self):
        """事実の検索テスト"""
        self.kb.add_fact("猫", "動物", "種類")
        self.kb.add_fact("犬", "動物", "種類")
        
        results = self.kb.search_facts("動物")
        assert len(results) >= 2
    
    def test_get_statistics(self):
        """統計情報の取得テスト"""
        self.kb.add_fact("猫", "動物", "種類")
        
        stats = self.kb.get_statistics()
        assert "total_facts" in stats
        assert stats["total_facts"] >= 1
