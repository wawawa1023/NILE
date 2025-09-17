"""
Japanese Processor Tests

日本語処理のテスト
"""

import pytest
from nile.core.nlp.japanese_processor import JapaneseProcessor

class TestJapaneseProcessor:
    """日本語処理のテストクラス"""
    
    def setup_method(self):
        """テストのセットアップ"""
        # 軽量なモデルを使用してテスト
        self.processor = JapaneseProcessor("cl-tohoku/bert-base-japanese-char")
    
    def test_preprocess_text(self):
        """テキスト前処理のテスト"""
        text = "  猫は動物です。  "
        processed = self.processor.preprocess_text(text)
        assert processed == "猫は動物です。"
    
    def test_extract_relations(self):
        """関係抽出のテスト"""
        text = "猫は動物です"
        relations = self.processor.extract_relations(text)
        assert len(relations) > 0
        assert relations[0]["subject"] == "猫"
        assert relations[0]["object"] == "動物"
    
    def test_extract_entities(self):
        """エンティティ抽出のテスト"""
        text = "田中さんは先生です"
        entities = self.processor.extract_entities(text)
        # エンティティが抽出されることを確認
        assert isinstance(entities, list)
    
    def test_analyze_sentence_structure(self):
        """文構造解析のテスト"""
        text = "猫は動物です"
        structure = self.processor.analyze_sentence_structure(text)
        
        assert structure["text"] == text
        assert structure["length"] == len(text)
        assert structure["has_question_marker"] is False
        assert "entities" in structure
        assert "relations" in structure
    
    def test_handle_ellipsis(self):
        """省略表現処理のテスト"""
        text = "猫は"
        handled = self.processor.handle_ellipsis(text)
        assert "何" in handled or "？" in handled
    
    def test_extract_temporal_expressions(self):
        """時間表現抽出のテスト"""
        text = "今日は良い天気です"
        temporal_exprs = self.processor.extract_temporal_expressions(text)
        
        assert len(temporal_exprs) > 0
        assert temporal_exprs[0]["expression"] == "今日"
        assert temporal_exprs[0]["type"] == "today"
    
    def test_normalize_temporal_expression(self):
        """時間表現正規化のテスト"""
        temporal_expr = {
            "expression": "今日",
            "type": "today",
            "value": None,
            "position": 0
        }
        
        normalized = self.processor.normalize_temporal_expression(temporal_expr)
        assert "normalized_date" in normalized
    
    def test_generate_response_variations(self):
        """応答バリエーション生成のテスト"""
        base_response = "猫は動物です"
        variations = self.processor.generate_response_variations(base_response)
        
        assert len(variations) > 0
        assert base_response in variations
    
    def test_disambiguate_meaning(self):
        """意味曖昧性解消のテスト"""
        text = "猫"
        context = ["動物", "ペット"]
        
        disambiguation = self.processor.disambiguate_meaning(text, context)
        assert disambiguation["original_text"] == text
        assert disambiguation["context"] == context
        assert "possible_meanings" in disambiguation
