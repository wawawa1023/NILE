"""
Symbolic Engine Tests

シンボリックエンジンのテスト
"""

import pytest
import tempfile
import os
from nile.core.reasoning.symbolic_engine import SymbolicEngine

class TestSymbolicEngine:
    """シンボリックエンジンのテストクラス"""
    
    def setup_method(self):
        """テストのセットアップ"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.engine = SymbolicEngine(self.temp_file.name)
    
    def teardown_method(self):
        """テストのクリーンアップ"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_fact_processing(self):
        """事実処理のテスト"""
        response = self.engine.process_input("猫は動物です")
        assert "記録" in response or "了解" in response
    
    def test_question_processing(self):
        """質問処理のテスト"""
        # まず事実を追加
        self.engine.process_input("猫は動物です")
        
        # 質問を処理
        response = self.engine.process_input("猫は動物ですか？")
        assert "はい" in response or "ええ" in response or "その通り" in response
    
    def test_list_command(self):
        """リスト表示コマンドのテスト"""
        # 事実を追加
        self.engine.process_input("犬は動物です")
        
        # リスト表示
        response = self.engine.process_input("一覧")
        assert "知識ベース" in response or "犬" in response
    
    def test_empty_input(self):
        """空入力のテスト"""
        response = self.engine.process_input("")
        assert "空" in response or "入力" in response
    
    def test_unknown_input(self):
        """不明な入力のテスト"""
        response = self.engine.process_input("不明な入力")
        assert "理解" in response or "分かりません" in response
    
    def test_question_detection(self):
        """質問検出のテスト"""
        assert self.engine._is_question("猫は動物ですか？") is True
        assert self.engine._is_question("猫は動物です") is False
        assert self.engine._is_question("猫は動物？") is True
    
    def test_list_command_detection(self):
        """リストコマンド検出のテスト"""
        assert self.engine._is_list_command("一覧") is True
        assert self.engine._is_list_command("list") is True
        assert self.engine._is_list_command("表示") is True
        assert self.engine._is_list_command("猫は動物です") is False
    
    def test_time_expression_parsing(self):
        """時間表現解析のテスト"""
        result = self.engine._parse_time_expression("今日")
        assert result == 0
        
        result = self.engine._parse_time_expression("明日")
        assert result == 1
        
        result = self.engine._parse_time_expression("昨日")
        assert result == -1
    
    def test_show_knowledge(self):
        """知識表示のテスト"""
        # 事実を追加
        self.engine.process_input("鳥は動物です")
        
        # 知識を表示
        response = self.engine.show_knowledge()
        assert "知識ベース" in response
        assert "鳥" in response or "動物" in response
