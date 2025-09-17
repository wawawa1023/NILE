"""
Enhanced Japanese Processor

改善された日本語自然言語処理機能
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from .pattern_matcher import PatternMatcher
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

class JapaneseProcessor:
    """改善された日本語処理クラス"""
    
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3"):
        self.model_name = model_name
        self.pattern_matcher = PatternMatcher()
        self.entity_extractor = EntityExtractor()
        
        # 関係性パターンの拡張
        self.relation_patterns = {
            "種類": [
                r"(.*?)は(.*?)です",
                r"(.*?)は(.*?)である",
                r"(.*?)は(.*?)だ",
                r"(.*?)は(.*?)の一種です",
                r"(.*?)は(.*?)に属します",
                r"(.*?)は(.*?)の仲間です",
                r"(.*?)は(.*?)の分類です"
            ],
            "属性": [
                r"(.*?)は(.*?)です",
                r"(.*?)は(.*?)である",
                r"(.*?)は(.*?)だ",
                r"(.*?)は(.*?)の特徴です",
                r"(.*?)は(.*?)の性質です",
                r"(.*?)は(.*?)の状態です",
                r"(.*?)は(.*?)を持っています",
                r"(.*?)は(.*?)が特徴です"
            ],
            "行動": [
                r"(.*?)は(.*?)します",
                r"(.*?)は(.*?)する",
                r"(.*?)は(.*?)の行動です",
                r"(.*?)は(.*?)の動作です",
                r"(.*?)は(.*?)の振る舞いです",
                r"(.*?)は(.*?)をします",
                r"(.*?)は(.*?)を行います"
            ],
            "場所": [
                r"(.*?)は(.*?)にあります",
                r"(.*?)は(.*?)に存在します",
                r"(.*?)は(.*?)の生息地です",
                r"(.*?)は(.*?)の住処です",
                r"(.*?)は(.*?)の場所です",
                r"(.*?)は(.*?)に住んでいます",
                r"(.*?)は(.*?)に生息しています"
            ],
            "時間": [
                r"(.*?)は(.*?)の時です",
                r"(.*?)は(.*?)の時期です",
                r"(.*?)は(.*?)に起こります",
                r"(.*?)は(.*?)に発生します"
            ],
            "原因": [
                r"(.*?)は(.*?)の原因です",
                r"(.*?)は(.*?)の理由です",
                r"(.*?)は(.*?)の要因です",
                r"(.*?)によって(.*?)が起こります"
            ],
            "結果": [
                r"(.*?)は(.*?)の結果です",
                r"(.*?)は(.*?)の影響です",
                r"(.*?)は(.*?)の効果です",
                r"(.*?)により(.*?)が生じます"
            ]
        }
        
        # 敬語・丁寧語パターン
        self.polite_patterns = {
            "です": ["です", "でございます", "であります"],
            "ます": ["ます", "まして", "ました"],
            "である": ["である", "であります", "でござる"],
            "だ": ["だ", "である", "であります"]
        }
        
        # 省略表現の補完パターン
        self.ellipsis_patterns = {
            r"^(.*?)は$": r"\1は何ですか？",
            r"^(.*?)の$": r"\1の何について知りたいですか？",
            r"^(.*?)が$": r"\1がどうしたのですか？"
        }
        
        # 曖昧性解消のための文脈キーワード
        self.context_keywords = {
            "動物": ["生き物", "生物", "ペット", "野生動物"],
            "植物": ["草", "木", "花", "野菜", "果物"],
            "人物": ["人", "男性", "女性", "子供", "大人"],
            "場所": ["場所", "地域", "国", "都市", "建物"],
            "時間": ["時", "時期", "季節", "年", "月", "日"]
        }
        
        # 高速起動モード（環境変数 NILE_FAST_START=1）ではモデル読み込みをスキップ
        self.fast_start = os.getenv("NILE_FAST_START", "0") in ("1", "true", "True")
        if not self.fast_start:
            self._init_model()
        else:
            self.tokenizer = None
            self.model = None
    
    def _init_model(self):
        """モデルを初期化"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("GPUを使用してモデルを初期化しました")
            else:
                logger.info("CPUを使用してモデルを初期化しました")
                
        except Exception as e:
            logger.error(f"モデルの初期化中にエラーが発生しました: {e}")
            # フォールバックモデルを試行
            try:
                self.model_name = "cl-tohoku/bert-base-japanese-char"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("フォールバックモデルを使用しました")
            except Exception as e2:
                logger.error(f"フォールバックモデルの初期化も失敗しました: {e2}")
                # それでも失敗した場合はモデルなしで継続（正規表現ベースのみ）
                self.tokenizer = None
                self.model = None
    
    def preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        # 空白の正規化
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 句読点の正規化
        text = re.sub(r'[。、]{2,}', '。', text)
        
        # 不要な文字の除去
        text = re.sub(r'[^\w\s。、！？]', '', text)
        
        return text
    
    def handle_ellipsis(self, text: str) -> str:
        """省略表現の補完"""
        for pattern, replacement in self.ellipsis_patterns.items():
            if re.match(pattern, text):
                return re.sub(pattern, replacement, text)
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """エンティティを抽出"""
        return self.entity_extractor.extract(text)
    
    def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """関係性を抽出（改善版）"""
        relations = []
        
        # 前処理
        processed_text = self.preprocess_text(text)
        
        # 省略表現の補完
        processed_text = self.handle_ellipsis(processed_text)
        
        # パターンマッチングによる関係抽出
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, processed_text)
                for match in matches:
                    subject = match.group(1).strip()
                    object_ = match.group(2).strip()
                    
                    # 無意味なマッチを除外
                    if self._is_valid_relation(subject, object_):
                        relations.append({
                            "subject": subject,
                            "relation": relation_type,
                            "object": object_,
                            "confidence": self._calculate_confidence(subject, object_, relation_type)
                        })
        
        return relations
    
    def _is_valid_relation(self, subject: str, object_: str) -> bool:
        """関係の妥当性をチェック"""
        # 空文字列チェック
        if not subject or not object_:
            return False
        
        # 無意味なトークンチェック
        meaningless_tokens = {"は", "です", "か", "？", "。", "、", "の", "が", "を", "に"}
        if subject in meaningless_tokens or object_ in meaningless_tokens:
            return False
        
        # 長さチェック
        if len(subject) < 1 or len(object_) < 1:
            return False
        
        # 同じ文字列チェック
        if subject == object_:
            return False
        
        return True
    
    def _calculate_confidence(self, subject: str, object_: str, relation_type: str) -> float:
        """関係の信頼度を計算"""
        confidence = 0.5  # ベース信頼度
        
        # 文脈キーワードとの一致度
        for context_type, keywords in self.context_keywords.items():
            if any(keyword in subject or keyword in object_ for keyword in keywords):
                confidence += 0.2
                break
        
        # 関係タイプの信頼度調整
        if relation_type in ["種類", "属性"]:
            confidence += 0.1
        elif relation_type in ["行動", "場所"]:
            confidence += 0.05
        
        # 文字列の長さによる調整
        if len(subject) > 2 and len(object_) > 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """文の構造を解析"""
        # 基本的な文構造の解析
        structure = {
            "text": text,
            "length": len(text),
            "has_question_marker": "か" in text or "？" in text or "?" in text,
            "has_polite_form": any(polite in text for polite in ["です", "ます", "でございます"]),
            "has_casual_form": any(casual in text for casual in ["だ", "である", "じゃない"]),
            "entities": self.extract_entities(text),
            "relations": self.extract_relations(text)
        }
        
        return structure
    
    def disambiguate_meaning(self, text: str, context: List[str] = None) -> Dict[str, Any]:
        """意味の曖昧性を解消"""
        if context is None:
            context = []
        
        # 文脈情報を考慮した曖昧性解消
        disambiguation = {
            "original_text": text,
            "context": context,
            "possible_meanings": [],
            "confidence": 0.0
        }
        
        # 文脈キーワードとの照合
        for context_type, keywords in self.context_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                disambiguation["possible_meanings"].append({
                    "type": context_type,
                    "confidence": matches / len(keywords)
                })
        
        # 最も可能性の高い意味を選択
        if disambiguation["possible_meanings"]:
            best_match = max(disambiguation["possible_meanings"], 
                           key=lambda x: x["confidence"])
            disambiguation["confidence"] = best_match["confidence"]
        
        return disambiguation
    
    def generate_response_variations(self, base_response: str) -> List[str]:
        """応答のバリエーションを生成"""
        variations = [base_response]
        
        # 敬語レベルによるバリエーション
        if "です" in base_response:
            variations.append(base_response.replace("です", "でございます"))
            variations.append(base_response.replace("です", "だ"))
        
        if "ます" in base_response:
            variations.append(base_response.replace("ます", "る"))
        
        # 語順の変更
        if "は" in base_response and "です" in base_response:
            parts = base_response.split("は")
            if len(parts) == 2:
                variations.append(f"{parts[1].replace('です', '')}は{parts[0]}です")
        
        return list(set(variations))  # 重複を除去
    
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """時間表現を抽出"""
        temporal_expressions = []
        
        # 時間表現のパターン
        time_patterns = {
            r"(\d+)年前": "years_ago",
            r"(\d+)年後": "years_later",
            r"(\d+)ヶ月前": "months_ago",
            r"(\d+)ヶ月後": "months_later",
            r"(\d+)日前": "days_ago",
            r"(\d+)日後": "days_later",
            r"(\d+)週間前": "weeks_ago",
            r"(\d+)週間後": "weeks_later",
            r"今日": "today",
            r"明日": "tomorrow",
            r"昨日": "yesterday",
            r"明後日": "day_after_tomorrow",
            r"一昨日": "day_before_yesterday"
        }
        
        for pattern, time_type in time_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                temporal_expressions.append({
                    "expression": match.group(0),
                    "type": time_type,
                    "value": match.group(1) if match.groups() else None,
                    "position": match.start()
                })
        
        return temporal_expressions
    
    def normalize_temporal_expression(self, temporal_expr: Dict[str, Any]) -> Dict[str, Any]:
        """時間表現を正規化"""
        from datetime import datetime, timedelta
        
        normalized = temporal_expr.copy()
        now = datetime.now()
        
        if temporal_expr["type"] == "today":
            normalized["normalized_date"] = now.date()
        elif temporal_expr["type"] == "tomorrow":
            normalized["normalized_date"] = (now + timedelta(days=1)).date()
        elif temporal_expr["type"] == "yesterday":
            normalized["normalized_date"] = (now - timedelta(days=1)).date()
        elif temporal_expr["type"] == "day_after_tomorrow":
            normalized["normalized_date"] = (now + timedelta(days=2)).date()
        elif temporal_expr["type"] == "day_before_yesterday":
            normalized["normalized_date"] = (now - timedelta(days=2)).date()
        elif temporal_expr["value"]:
            value = int(temporal_expr["value"])
            if "days_ago" in temporal_expr["type"]:
                normalized["normalized_date"] = (now - timedelta(days=value)).date()
            elif "days_later" in temporal_expr["type"]:
                normalized["normalized_date"] = (now + timedelta(days=value)).date()
            elif "weeks_ago" in temporal_expr["type"]:
                normalized["normalized_date"] = (now - timedelta(weeks=value)).date()
            elif "weeks_later" in temporal_expr["type"]:
                normalized["normalized_date"] = (now + timedelta(weeks=value)).date()
            elif "months_ago" in temporal_expr["type"]:
                # 簡易的な月計算（30日として扱う）
                normalized["normalized_date"] = (now - timedelta(days=value*30)).date()
            elif "months_later" in temporal_expr["type"]:
                normalized["normalized_date"] = (now + timedelta(days=value*30)).date()
            elif "years_ago" in temporal_expr["type"]:
                normalized["normalized_date"] = (now - timedelta(days=value*365)).date()
            elif "years_later" in temporal_expr["type"]:
                normalized["normalized_date"] = (now + timedelta(days=value*365)).date()
        
        return normalized
