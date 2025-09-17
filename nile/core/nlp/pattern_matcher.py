"""
Pattern Matcher

高度なパターンマッチング機能を提供
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

class PatternMatcher:
    """高度なパターンマッチングクラス"""
    
    def __init__(self):
        self.compiled_patterns = {}
        self._init_patterns()
    
    def _init_patterns(self):
        """パターンを初期化"""
        # 基本的な文パターン
        self.basic_patterns = {
            "fact": [
                r"^(.*?)は(.*?)です$",
                r"^(.*?)は(.*?)である$",
                r"^(.*?)は(.*?)だ$",
                r"^(.*?)は(.*?)の一種です$",
                r"^(.*?)は(.*?)に属します$",
                r"^(.*?)は(.*?)を持っています$",
                r"^(.*?)は(.*?)が特徴です$"
            ],
            "question": [
                r"^(.*?)は(.*?)ですか？$",
                r"^(.*?)は(.*?)でしょうか？$",
                r"^(.*?)は(.*?)ですか$",
                r"^(.*?)は(.*?)でしょうか$",
                r"^(.*?)は(.*?)？$",
                r"^(.*?)は(.*?)？$"
            ],
            "negation": [
                r"^(.*?)は(.*?)ではない$",
                r"^(.*?)は(.*?)じゃない$",
                r"^(.*?)は(.*?)ではありません$",
                r"^(.*?)は(.*?)ではないです$"
            ],
            "conditional": [
                r"もし(.*?)なら(.*?)です$",
                r"もし(.*?)なら(.*?)である$",
                r"もし(.*?)なら(.*?)だ$",
                r"(.*?)なら(.*?)です$"
            ]
        }
        
        # 複合的なパターン
        self.complex_patterns = {
            "comparison": [
                r"^(.*?)は(.*?)より(.*?)です$",
                r"^(.*?)は(.*?)よりも(.*?)です$",
                r"^(.*?)と(.*?)は(.*?)です$"
            ],
            "possession": [
                r"^(.*?)は(.*?)を持っています$",
                r"^(.*?)は(.*?)を所有しています$",
                r"^(.*?)の(.*?)は(.*?)です$"
            ],
            "location": [
                r"^(.*?)は(.*?)にあります$",
                r"^(.*?)は(.*?)に存在します$",
                r"^(.*?)は(.*?)に住んでいます$",
                r"^(.*?)は(.*?)に生息しています$"
            ],
            "temporal": [
                r"^(.*?)は(.*?)の時です$",
                r"^(.*?)は(.*?)の時期です$",
                r"^(.*?)は(.*?)に起こります$",
                r"^(.*?)は(.*?)に発生します$"
            ]
        }
        
        # 関係性パターン
        self.relation_patterns = {
            "is_a": [
                r"^(.*?)は(.*?)の一種です$",
                r"^(.*?)は(.*?)に属します$",
                r"^(.*?)は(.*?)の仲間です$",
                r"^(.*?)は(.*?)の分類です$"
            ],
            "has_attribute": [
                r"^(.*?)は(.*?)を持っています$",
                r"^(.*?)は(.*?)が特徴です$",
                r"^(.*?)は(.*?)の性質です$",
                r"^(.*?)は(.*?)の状態です$"
            ],
            "performs_action": [
                r"^(.*?)は(.*?)します$",
                r"^(.*?)は(.*?)する$",
                r"^(.*?)は(.*?)の行動です$",
                r"^(.*?)は(.*?)の動作です$"
            ],
            "located_at": [
                r"^(.*?)は(.*?)にあります$",
                r"^(.*?)は(.*?)に存在します$",
                r"^(.*?)は(.*?)の生息地です$",
                r"^(.*?)は(.*?)の住処です$"
            ]
        }
        
        # パターンをコンパイル
        self._compile_patterns()
    
    def _compile_patterns(self):
        """パターンをコンパイルしてキャッシュ"""
        all_patterns = {
            **self.basic_patterns,
            **self.complex_patterns,
            **self.relation_patterns
        }
        
        for category, patterns in all_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern) for pattern in patterns
            ]
    
    @lru_cache(maxsize=128)
    def match_pattern(self, text: str, pattern_type: str) -> Optional[Dict[str, Any]]:
        """パターンマッチングを実行"""
        if pattern_type not in self.compiled_patterns:
            return None
        
        for pattern in self.compiled_patterns[pattern_type]:
            match = pattern.match(text)
            if match:
                return {
                    "pattern": pattern.pattern,
                    "groups": match.groups(),
                    "type": pattern_type,
                    "confidence": self._calculate_pattern_confidence(pattern, match)
                }
        
        return None
    
    def _calculate_pattern_confidence(self, pattern: re.Pattern, match: re.Match) -> float:
        """パターンマッチの信頼度を計算"""
        confidence = 0.5  # ベース信頼度
        
        # パターンの複雑さによる調整
        if len(pattern.pattern) > 20:
            confidence += 0.1
        
        # マッチしたグループの数による調整
        if len(match.groups()) >= 2:
            confidence += 0.1
        
        # 特殊文字の使用による調整
        if any(char in pattern.pattern for char in ["$", "^", "\\"]):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def find_all_matches(self, text: str) -> List[Dict[str, Any]]:
        """全てのパターンタイプでマッチングを実行"""
        matches = []
        
        for pattern_type in self.compiled_patterns.keys():
            match = self.match_pattern(text, pattern_type)
            if match:
                matches.append(match)
        
        # 信頼度でソート
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches
    
    def extract_entities_from_pattern(self, text: str, pattern_type: str) -> List[Dict[str, str]]:
        """パターンからエンティティを抽出"""
        match = self.match_pattern(text, pattern_type)
        if not match:
            return []
        
        entities = []
        groups = match["groups"]
        
        if len(groups) >= 2:
            entities.append({
                "text": groups[0],
                "type": "subject",
                "position": 0
            })
            entities.append({
                "text": groups[1],
                "type": "object",
                "position": 1
            })
        
        return entities
    
    def is_question(self, text: str) -> bool:
        """質問かどうかを判定"""
        question_indicators = ["か", "？", "?", "でしょうか", "ですか"]
        return any(indicator in text for indicator in question_indicators)
    
    def is_negation(self, text: str) -> bool:
        """否定文かどうかを判定"""
        negation_indicators = ["ない", "じゃない", "ではありません", "ではない"]
        return any(indicator in text for indicator in negation_indicators)
    
    def is_conditional(self, text: str) -> bool:
        """条件文かどうかを判定"""
        conditional_indicators = ["もし", "なら", "ば", "たら"]
        return any(indicator in text for indicator in conditional_indicators)
    
    def extract_relation_type(self, text: str) -> Optional[str]:
        """関係タイプを抽出"""
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return relation_type
        return None
    
    def normalize_text(self, text: str) -> str:
        """テキストを正規化"""
        # 空白の正規化
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 句読点の正規化
        text = re.sub(r'[。、]{2,}', '。', text)
        
        # 不要な文字の除去
        text = re.sub(r'[^\w\s。、！？]', '', text)
        
        return text
    
    def split_compound_sentence(self, text: str) -> List[str]:
        """複文を分割"""
        # 接続詞で分割
        conjunctions = ["そして", "また", "さらに", "しかし", "でも", "だから", "それで"]
        
        sentences = [text]
        for conjunction in conjunctions:
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split(conjunction)
                if len(parts) > 1:
                    new_sentences.extend([part.strip() for part in parts if part.strip()])
                else:
                    new_sentences.append(sentence)
            sentences = new_sentences
        
        return sentences
    
    def extract_modifiers(self, text: str) -> List[Dict[str, str]]:
        """修飾語を抽出"""
        modifiers = []
        
        # 形容詞の修飾
        adj_pattern = r"([^は]+)な([^は]+)は"
        for match in re.finditer(adj_pattern, text):
            modifiers.append({
                "type": "adjective",
                "modifier": match.group(1),
                "modified": match.group(2),
                "position": match.start()
            })
        
        # 副詞の修飾
        adv_pattern = r"([^は]+)に([^は]+)は"
        for match in re.finditer(adv_pattern, text):
            modifiers.append({
                "type": "adverb",
                "modifier": match.group(1),
                "modified": match.group(2),
                "position": match.start()
            })
        
        return modifiers
    
    def analyze_sentence_complexity(self, text: str) -> Dict[str, Any]:
        """文の複雑さを解析"""
        complexity = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_conjunctions": any(conj in text for conj in ["そして", "また", "しかし", "でも"]),
            "has_conditionals": any(cond in text for cond in ["もし", "なら", "ば", "たら"]),
            "has_negations": any(neg in text for neg in ["ない", "じゃない", "ではありません"]),
            "has_questions": any(q in text for q in ["か", "？", "?"]),
            "has_modifiers": len(self.extract_modifiers(text)) > 0,
            "complexity_score": 0
        }
        
        # 複雑さスコアの計算
        score = 0
        if complexity["length"] > 50:
            score += 1
        if complexity["word_count"] > 10:
            score += 1
        if complexity["has_conjunctions"]:
            score += 1
        if complexity["has_conditionals"]:
            score += 1
        if complexity["has_negations"]:
            score += 1
        if complexity["has_modifiers"]:
            score += 1
        
        complexity["complexity_score"] = score
        return complexity
    
    def suggest_corrections(self, text: str) -> List[str]:
        """文の修正提案"""
        suggestions = []
        
        # 一般的な誤りパターン
        corrections = {
            r"はは": "は",
            r"ですです": "です",
            r"ますます": "ます",
            r"だだ": "だ",
            r"であるである": "である"
        }
        
        corrected_text = text
        for error, correction in corrections.items():
            if re.search(error, corrected_text):
                corrected_text = re.sub(error, correction, corrected_text)
                suggestions.append(f"'{error}' → '{correction}'")
        
        if suggestions:
            suggestions.insert(0, f"修正版: {corrected_text}")
        
        return suggestions
