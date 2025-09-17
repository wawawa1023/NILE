"""
Enhanced Knowledge Base

改善された知識ベース実装
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
import logging
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """改善された知識ベースクラス"""
    
    def __init__(self, file_path: str = "knowledge.json"):
        self.file_path = file_path
        self.lock = Lock()
        self.facts = self._load_facts()
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """メタデータを読み込む"""
        try:
            if os.path.exists(self.file_path):
                # 空ファイルは空メタデータ扱い
                if os.path.getsize(self.file_path) == 0:
                    return {
                        "version": "1.0",
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "total_facts": 0
                    }
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return {
                            "version": "1.0",
                            "created_at": datetime.now().isoformat(),
                            "last_updated": datetime.now().isoformat(),
                            "total_facts": 0
                        }
                    data = json.loads(content)
                    if isinstance(data, dict):
                        return data.get("metadata", {
                            "version": "1.0",
                            "created_at": datetime.now().isoformat(),
                            "last_updated": datetime.now().isoformat(),
                            "total_facts": 0
                        })
        except Exception as e:
            logger.error(f"メタデータの読み込み中にエラーが発生しました: {e}")
        
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_facts": 0
        }

    def _init_basic_facts(self):
        """基本的な事実を初期化"""
        basic_facts = [
            {"subject": "猫", "relation": "種類", "object": "動物", "confidence": 1.0, "source": "system"},
            {"subject": "犬", "relation": "種類", "object": "動物", "confidence": 1.0, "source": "system"},
            {"subject": "鳥", "relation": "種類", "object": "動物", "confidence": 1.0, "source": "system"},
            {"subject": "猫", "relation": "属性", "object": "かわいい", "confidence": 0.8, "source": "system"},
            {"subject": "犬", "relation": "属性", "object": "忠実", "confidence": 0.9, "source": "system"},
            {"subject": "鳥", "relation": "属性", "object": "自由", "confidence": 0.7, "source": "system"}
        ]
        self.facts.extend(basic_facts)
        self.save_facts()

    def _load_facts(self) -> List[Dict[str, Any]]:
        """知識ベースをファイルから読み込む"""
        try:
            if os.path.exists(self.file_path):
                # 空ファイルは空配列扱い
                if os.path.getsize(self.file_path) == 0:
                    return []
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    data = json.loads(content)
                    if isinstance(data, dict) and "facts" in data:
                        return data["facts"]
                    elif isinstance(data, list):
                        return data
                    return []
            return []
        except Exception as e:
            logger.error(f"知識ベースの読み込み中にエラーが発生しました: {e}")
            return []

    def save_facts(self):
        """知識ベースをファイルに保存する"""
        with self.lock:
            try:
                # 保存先のディレクトリが存在しない場合は作成
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                
                # メタデータを更新
                self.metadata["last_updated"] = datetime.now().isoformat()
                self.metadata["total_facts"] = len(self.facts)
                
                data = {
                    "metadata": self.metadata,
                    "facts": self.facts
                }
                
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"知識ベースを保存しました: {len(self.facts)}件の事実")
            except Exception as e:
                logger.error(f"知識ベースの保存中にエラーが発生しました: {e}")

    def add_fact(self, subject: str, object_: str, relation: str = "is_a", 
                 confidence: float = 1.0, source: str = "user") -> bool:
        """
        新しい事実を追加します。
        
        Args:
            subject: 主語
            object_: 目的語
            relation: 関係性
            confidence: 信頼度 (0.0-1.0)
            source: 情報源
            
        Returns:
            新規追加ならTrue、既存ならFalse
        """
        fact = {
            "subject": subject, 
            "relation": relation, 
            "object": object_,
            "confidence": confidence,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "id": self._generate_fact_id(subject, object_, relation)
        }
        
        # 重複チェック
        if self._fact_exists(fact):
            return False
            
        self.facts.append(fact)
        self.save_facts()
        return True

    def _generate_fact_id(self, subject: str, object_: str, relation: str) -> str:
        """事実の一意IDを生成"""
        content = f"{subject}|{relation}|{object_}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]

    def _fact_exists(self, fact: Dict[str, Any]) -> bool:
        """事実が既に存在するかチェック"""
        for existing_fact in self.facts:
            if (existing_fact["subject"] == fact["subject"] and
                existing_fact["object"] == fact["object"] and
                existing_fact["relation"] == fact["relation"]):
                return True
        return False

    def get_facts(self) -> List[Dict[str, Any]]:
        """登録済みの全事実を取得します。"""
        return [{
            "subject": fact["subject"],
            "relation": fact["relation"],
            "object": fact["object"],
            "confidence": fact.get("confidence", 1.0),
            "source": fact.get("source", "unknown"),
            "created_at": fact.get("created_at", ""),
            "id": fact.get("id", "")
        } for fact in self.facts]

    def get_facts_by_confidence(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """信頼度でフィルタリングした事実を取得"""
        return [
            fact for fact in self.get_facts()
            if fact.get("confidence", 1.0) >= min_confidence
        ]

    def get_facts_by_source(self, source: str) -> List[Dict[str, Any]]:
        """情報源でフィルタリングした事実を取得"""
        return [
            fact for fact in self.get_facts()
            if fact.get("source", "") == source
        ]

    def get_relation(self, subject: str, object_: str) -> Optional[Dict[str, Any]]:
        """
        2つのノード間の関係性を取得します。
        直接の関係がない場合はNoneを返します。
        """
        for fact in self.facts:
            if fact["subject"] == subject and fact["object"] == object_:
                return {
                    "relation": fact["relation"],
                    "confidence": fact.get("confidence", 1.0),
                    "source": fact.get("source", "unknown")
                }
        return None

    def get_related_nodes(self, node: str) -> Set[str]:
        """
        指定されたノードに関連する全てのノードを取得します。
        主語としても目的語としても関連するノードを返します。
        """
        related = set()
        for fact in self.facts:
            if fact["subject"] == node:
                related.add(fact["object"])
            elif fact["object"] == node:
                related.add(fact["subject"])
        return related

    def has_path(self, start: str, end: str) -> bool:
        """start→…→end の経路があるかを確認します（深さ優先探索）。"""
        return self._find_path(start, end) is not None
    
    def has_reverse_path(self, start: str, end: str, visited: Optional[set] = None) -> bool:
        """
        逆方向の関係を確認します（end→…→start の経路があるかを確認）。
        """
        if visited is None:
            visited = set()
        if start == end:
            return True
        visited.add(end)
        
        # 直接の逆関係を確認
        for fact in self.facts:
            if fact["subject"] == end and fact["object"] == start:
                return True
        
        # 間接的な逆関係を確認
        for fact in self.facts:
            if fact["object"] == end:
                subj = fact["subject"]
                if subj not in visited:
                    if self.has_reverse_path(start, subj, visited):
                        return True
        return False

    def get_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        start→…→end の経路上のノードリストを返します。経路がなければ None。
        例: ["A", "B", "C"]
        """
        return self._find_path(start, end)

    def _find_path(self, start: str, end: str, visited: Optional[set] = None) -> Optional[List[str]]:
        if visited is None:
            visited = set()
        if start == end:
            return [start]
        visited.add(start)
        # start を主語とする事実を探す
        for fact in self.facts:
            if fact["subject"] == start:
                obj = fact["object"]
                if obj not in visited:
                    path = self._find_path(obj, end, visited)
                    if path:
                        return [start] + path
        return None

    def get_reverse_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        逆方向の経路を取得します（end→…→start の経路）。
        """
        # 直接の逆関係を確認
        for fact in self.facts:
            if fact["subject"] == end and fact["object"] == start:
                return [end, start]

        def _find_reverse_path(current: str, target: str, visited: set) -> Optional[List[str]]:
            if current == target:
                return [current]
            visited.add(current)
            
            for fact in self.facts:
                if fact["object"] == current:
                    subj = fact["subject"]
                    if subj not in visited:
                        path = _find_reverse_path(subj, target, visited)
                        if path:
                            return [current] + path
            return None
        
        return _find_reverse_path(end, start, set())

    def get_facts_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """指定された日付に関連する事実を取得
        
        Args:
            date_str: 日付文字列（YYYY-MM-DD形式）
            
        Returns:
            該当する事実のリスト
        """
        date_facts = []
        for fact in self.facts:
            # 時間表現を含む事実を検索
            if any(time_expr in fact["subject"] for time_expr in ["今日", "昨日", "明日", "明後日", "一昨日"]):
                date_facts.append(fact)
            elif any(time_expr in fact["object"] for time_expr in ["今日", "昨日", "明日", "明後日", "一昨日"]):
                date_facts.append(fact)
        return date_facts

    def get_facts_by_time_expression(self, time_expr: str) -> List[Dict[str, Any]]:
        """時間表現に関連する事実を取得
        
        Args:
            time_expr: 時間表現（"今日", "昨日"など）
            
        Returns:
            該当する事実のリスト
        """
        return [
            fact for fact in self.facts
            if time_expr in fact["subject"] or time_expr in fact["object"]
        ]

    def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """事実を検索する"""
        results = []
        query_lower = query.lower()
        
        for fact in self.facts:
            if (query_lower in fact["subject"].lower() or
                query_lower in fact["object"].lower() or
                query_lower in fact["relation"].lower()):
                results.append(fact)
                if len(results) >= limit:
                    break
                    
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """知識ベースの統計情報を取得"""
        total_facts = len(self.facts)
        sources = {}
        relations = {}
        
        for fact in self.facts:
            source = fact.get("source", "unknown")
            relation = fact.get("relation", "unknown")
            
            sources[source] = sources.get(source, 0) + 1
            relations[relation] = relations.get(relation, 0) + 1
        
        return {
            "total_facts": total_facts,
            "sources": sources,
            "relations": relations,
            "metadata": self.metadata
        }
