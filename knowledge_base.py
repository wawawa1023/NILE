import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

class KnowledgeBase:
    def __init__(self, file_path: str = "knowledge.json"):
        self.file_path = file_path
        self.facts = self._load_facts()
        # 初期データがない場合は基本的な事実を追加
        if not self.facts:
            self._init_basic_facts()

    def _init_basic_facts(self):
        """基本的な事実を初期化"""
        basic_facts = [
            {"subject": "猫", "relation": "種類", "object": "動物"},
            {"subject": "犬", "relation": "種類", "object": "動物"},
            {"subject": "鳥", "relation": "種類", "object": "動物"},
            {"subject": "猫", "relation": "属性", "object": "かわいい"},
            {"subject": "犬", "relation": "属性", "object": "忠実"},
            {"subject": "鳥", "relation": "属性", "object": "自由"}
        ]
        self.facts.extend(basic_facts)
        self.save_facts()

    def _load_facts(self) -> List[Dict[str, str]]:
        """知識ベースをファイルから読み込む"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "facts" in data:
                        return data["facts"]
                    elif isinstance(data, list):
                        return data
                    return []
            return []
        except Exception as e:
            print(f"知識ベースの読み込み中にエラーが発生しました: {e}")
            return []

    def save_facts(self):
        """知識ベースをファイルに保存する"""
        try:
            # 保存先のディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"facts": self.facts}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"知識ベースの保存中にエラーが発生しました: {e}")

    def add_fact(self, subject: str, object_: str, relation: str = "is_a") -> bool:
        """
        新しい事実を追加します。
        - 既に同じ事実があれば何もしないで False を返します。
        - 新規追加なら True を返します。
        """
        fact = {"subject": subject, "relation": relation, "object": object_}
        if fact in self.facts:
            return False
        self.facts.append(fact)
        self.save_facts()
        return True

    def get_facts(self) -> List[Dict[str, str]]:
        """登録済みの全事実を取得します。"""
        return [{
            "subject": fact["subject"],
            "relation": fact["relation"],
            "object": fact["object"]
        } for fact in self.facts]

    def get_relation(self, subject: str, object_: str) -> Optional[str]:
        """
        2つのノード間の関係性を取得します。
        直接の関係がない場合はNoneを返します。
        """
        for fact in self.facts:
            if fact["subject"] == subject and fact["object"] == object_:
                return fact["relation"]
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

    def get_facts_by_date(self, date_str: str) -> List[Dict[str, str]]:
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

    def get_facts_by_time_expression(self, time_expr: str) -> List[Dict[str, str]]:
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
