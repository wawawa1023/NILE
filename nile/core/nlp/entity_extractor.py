"""
Entity Extractor

エンティティ抽出機能を提供
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityExtractor:
    """エンティティ抽出クラス"""
    
    def __init__(self):
        self._init_entity_patterns()
        self._init_entity_types()
    
    def _init_entity_patterns(self):
        """エンティティパターンを初期化"""
        self.entity_patterns = {
            "person": [
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)さん",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)君",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)ちゃん",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)先生",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)博士"
            ],
            "animal": [
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という動物",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という生き物",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という生物"
            ],
            "place": [
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という場所",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という地域",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という国",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という都市"
            ],
            "object": [
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という物",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という道具",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という機械"
            ],
            "concept": [
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という概念",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という考え",
                r"([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+)という理論"
            ]
        }
        
        # 一般的なエンティティキーワード
        self.entity_keywords = {
            "person": ["人", "男性", "女性", "子供", "大人", "老人", "学生", "先生", "医者", "弁護士"],
            "animal": ["動物", "生き物", "生物", "ペット", "野生動物", "哺乳類", "鳥類", "魚類", "昆虫"],
            "plant": ["植物", "草", "木", "花", "野菜", "果物", "樹木", "草花", "作物"],
            "place": ["場所", "地域", "国", "都市", "町", "村", "建物", "家", "学校", "病院"],
            "object": ["物", "道具", "機械", "車", "本", "服", "食べ物", "飲み物", "家具"],
            "concept": ["概念", "考え", "理論", "方法", "技術", "科学", "芸術", "文化", "歴史"],
            "time": ["時間", "時", "時期", "季節", "年", "月", "日", "朝", "昼", "夜"],
            "emotion": ["感情", "気持ち", "喜び", "悲しみ", "怒り", "驚き", "恐れ", "愛", "憎しみ"]
        }
    
    def _init_entity_types(self):
        """エンティティタイプを初期化"""
        self.entity_types = {
            "person": {
                "description": "人物",
                "examples": ["田中さん", "山田君", "佐藤先生"],
                "attributes": ["名前", "年齢", "職業", "性格"]
            },
            "animal": {
                "description": "動物",
                "examples": ["猫", "犬", "鳥", "魚"],
                "attributes": ["種類", "生息地", "習性", "特徴"]
            },
            "plant": {
                "description": "植物",
                "examples": ["桜", "松", "バラ", "トマト"],
                "attributes": ["種類", "生息地", "特徴", "用途"]
            },
            "place": {
                "description": "場所",
                "examples": ["東京", "日本", "学校", "公園"],
                "attributes": ["位置", "特徴", "人口", "気候"]
            },
            "object": {
                "description": "物体",
                "examples": ["車", "本", "机", "電話"],
                "attributes": ["材質", "用途", "色", "大きさ"]
            },
            "concept": {
                "description": "概念",
                "examples": ["愛", "自由", "正義", "美"],
                "attributes": ["定義", "特徴", "例", "関連概念"]
            },
            "time": {
                "description": "時間",
                "examples": ["今日", "明日", "春", "朝"],
                "attributes": ["期間", "順序", "特徴", "関連イベント"]
            },
            "emotion": {
                "description": "感情",
                "examples": ["喜び", "悲しみ", "怒り", "驚き"],
                "attributes": ["強度", "原因", "表現", "関連感情"]
            }
        }
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """テキストからエンティティを抽出"""
        entities = []
        
        # パターンマッチングによる抽出
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group(1)
                    entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "position": match.start(),
                        "confidence": self._calculate_entity_confidence(entity_text, entity_type),
                        "context": self._extract_context(text, match.start(), match.end())
                    })
        
        # キーワードベースの抽出
        keyword_entities = self._extract_by_keywords(text)
        entities.extend(keyword_entities)
        
        # 重複を除去
        entities = self._remove_duplicates(entities)
        
        # 信頼度でソート
        entities.sort(key=lambda x: x["confidence"], reverse=True)
        
        return entities
    
    def _extract_by_keywords(self, text: str) -> List[Dict[str, Any]]:
        """キーワードベースでエンティティを抽出"""
        entities = []
        
        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # キーワードの前後の文脈を取得
                    start_pos = text.find(keyword)
                    context_start = max(0, start_pos - 10)
                    context_end = min(len(text), start_pos + len(keyword) + 10)
                    context = text[context_start:context_end]
                    
                    entities.append({
                        "text": keyword,
                        "type": entity_type,
                        "position": start_pos,
                        "confidence": 0.7,  # キーワードベースは中程度の信頼度
                        "context": context
                    })
        
        return entities
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: str) -> float:
        """エンティティの信頼度を計算"""
        confidence = 0.5  # ベース信頼度
        
        # 文字列の長さによる調整
        if len(entity_text) >= 2:
            confidence += 0.1
        
        # エンティティタイプによる調整
        type_confidence = {
            "person": 0.9,
            "animal": 0.8,
            "plant": 0.8,
            "place": 0.7,
            "object": 0.6,
            "concept": 0.5,
            "time": 0.8,
            "emotion": 0.6
        }
        
        confidence += type_confidence.get(entity_type, 0.5) * 0.2
        
        return min(confidence, 1.0)
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """エンティティの前後の文脈を抽出"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _remove_duplicates(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複するエンティティを除去"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = (entity["text"], entity["type"])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def classify_entity(self, entity_text: str) -> Optional[str]:
        """エンティティのタイプを分類"""
        # キーワードベースの分類
        for entity_type, keywords in self.entity_keywords.items():
            if any(keyword in entity_text for keyword in keywords):
                return entity_type
        
        # パターンベースの分類
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity_text):
                    return entity_type
        
        return None
    
    def get_entity_attributes(self, entity_type: str) -> List[str]:
        """エンティティタイプの属性を取得"""
        return self.entity_types.get(entity_type, {}).get("attributes", [])
    
    def get_entity_examples(self, entity_type: str) -> List[str]:
        """エンティティタイプの例を取得"""
        return self.entity_types.get(entity_type, {}).get("examples", [])
    
    def extract_relations_between_entities(self, text: str) -> List[Dict[str, Any]]:
        """エンティティ間の関係を抽出"""
        entities = self.extract(text)
        relations = []
        
        # エンティティペアの関係を抽出
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # エンティティ間のテキストを取得
                start = min(entity1["position"], entity2["position"])
                end = max(entity1["position"] + len(entity1["text"]), 
                         entity2["position"] + len(entity2["text"]))
                between_text = text[start:end]
                
                # 関係性を推定
                relation_type = self._infer_relation_type(between_text)
                if relation_type:
                    relations.append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "relation": relation_type,
                        "text": between_text,
                        "confidence": self._calculate_relation_confidence(between_text, relation_type)
                    })
        
        return relations
    
    def _infer_relation_type(self, text: str) -> Optional[str]:
        """テキストから関係タイプを推定"""
        relation_indicators = {
            "is_a": ["は", "である", "だ", "です"],
            "has": ["を持っている", "がある", "を所有している"],
            "located_at": ["にいる", "に住んでいる", "に生息している"],
            "part_of": ["の一部", "の構成要素", "の要素"],
            "causes": ["の原因", "の理由", "によって"],
            "results_in": ["の結果", "の影響", "により"]
        }
        
        for relation_type, indicators in relation_indicators.items():
            if any(indicator in text for indicator in indicators):
                return relation_type
        
        return None
    
    def _calculate_relation_confidence(self, text: str, relation_type: str) -> float:
        """関係の信頼度を計算"""
        confidence = 0.5  # ベース信頼度
        
        # 関係タイプによる調整
        type_confidence = {
            "is_a": 0.9,
            "has": 0.8,
            "located_at": 0.7,
            "part_of": 0.6,
            "causes": 0.5,
            "results_in": 0.5
        }
        
        confidence += type_confidence.get(relation_type, 0.5) * 0.3
        
        return min(confidence, 1.0)
    
    def get_entity_statistics(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """エンティティの統計情報を取得"""
        stats = {
            "total_entities": len(entities),
            "entity_types": defaultdict(int),
            "average_confidence": 0.0,
            "high_confidence_entities": 0
        }
        
        if not entities:
            return stats
        
        total_confidence = 0
        for entity in entities:
            entity_type = entity["type"]
            confidence = entity["confidence"]
            
            stats["entity_types"][entity_type] += 1
            total_confidence += confidence
            
            if confidence >= 0.8:
                stats["high_confidence_entities"] += 1
        
        stats["average_confidence"] = total_confidence / len(entities)
        stats["entity_types"] = dict(stats["entity_types"])
        
        return stats
