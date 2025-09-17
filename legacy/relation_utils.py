from typing import List, Dict, Any, Optional, Set, Tuple
import re

class RelationUtils:
    """関係性に関する共通の処理を提供するユーティリティクラス"""
    
    # 関係パターンの定義
    RELATION_PATTERNS = {
        "種類": [
            r"(.*?)は(.*?)です",
            r"(.*?)は(.*?)である",
            r"(.*?)は(.*?)だ",
            r"(.*?)は(.*?)の一種です",
            r"(.*?)は(.*?)に属します",
            r"(.*?)は(.*?)の仲間です"
        ],
        "属性": [
            r"(.*?)は(.*?)です",
            r"(.*?)は(.*?)である",
            r"(.*?)は(.*?)だ",
            r"(.*?)は(.*?)の特徴です",
            r"(.*?)は(.*?)の性質です",
            r"(.*?)は(.*?)の状態です"
        ],
        "行動": [
            r"(.*?)は(.*?)します",
            r"(.*?)は(.*?)する",
            r"(.*?)は(.*?)の行動です",
            r"(.*?)は(.*?)の動作です",
            r"(.*?)は(.*?)の振る舞いです"
        ],
        "場所": [
            r"(.*?)は(.*?)にあります",
            r"(.*?)は(.*?)に存在します",
            r"(.*?)は(.*?)の生息地です",
            r"(.*?)は(.*?)の住処です",
            r"(.*?)は(.*?)の場所です"
        ]
    }

    # 逆関係のマッピング
    REVERSE_RELATIONS = {
        "種類": "の一種",
        "属性": "の特徴",
        "行動": "の行動",
        "場所": "の場所"
    }

    # 関係性の逆変換マップ
    RELATION_REVERSE_MAP = {
        "の一種": "の上位概念",
        "に属する": "の下位概念",
        "の特徴": "の特徴を持つ",
        "の性質": "の性質を持つ",
        "の状態": "の状態を持つ",
        "の行動": "の行動をする",
        "の動作": "の動作をする",
        "の振る舞い": "の振る舞いをする",
        "の生息地": "の生息地である",
        "の住処": "の住処である",
        "の場所": "の場所である"
    }

    @classmethod
    def infer_relation(cls, subject: str, object_: str) -> str:
        """関係性を推測する"""
        # 基本的な関係性を推測
        if any(adj in object_ for adj in ["の一種", "に属する", "の仲間"]):
            return "種類"
        elif any(adj in object_ for adj in ["の特徴", "の性質", "の状態"]):
            return "属性"
        elif any(adj in object_ for adj in ["の行動", "の動作", "の振る舞い"]):
            return "行動"
        elif any(adj in object_ for adj in ["の生息地", "の住処", "の場所"]):
            return "場所"
        elif any(adj in object_ for adj in ["します", "する"]):
            return "行動"
        elif any(adj in object_ for adj in ["にあります", "に存在します"]):
            return "場所"
        
        # デフォルトは「種類」として扱う
        return "種類"

    @classmethod
    def get_reverse_relation(cls, relation: str) -> str:
        """関係の逆を取得"""
        return cls.REVERSE_RELATIONS.get(relation, relation)

    @classmethod
    def find_indirect_path(cls, start: str, end: str, facts: List[Dict[str, str]], max_depth: int = 3) -> Optional[List[str]]:
        """間接的な関係のパスを探索"""
        def _find_path(current: str, target: str, path: List[str], depth: int) -> Optional[List[str]]:
            if depth > max_depth:
                return None
            if current == target:
                return path

            for fact in facts:
                if fact["subject"] == current and fact["object"] not in path:
                    new_path = path + [fact["object"]]
                    result = _find_path(fact["object"], target, new_path, depth + 1)
                    if result:
                        return result
                elif fact["object"] == current and fact["subject"] not in path:
                    new_path = path + [fact["subject"]]
                    result = _find_path(fact["subject"], target, new_path, depth + 1)
                    if result:
                        return result

            return None

        return _find_path(start, end, [start], 0)

    @classmethod
    def find_relation_between(cls, subject: str, object_: str, facts: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """2つのノード間の関係を検索"""
        for fact in facts:
            if fact["subject"] == subject and fact["object"] == object_:
                return fact
            elif fact["subject"] == object_ and fact["object"] == subject:
                return {
                    "subject": subject,
                    "relation": cls.get_reverse_relation(fact["relation"]),
                    "object": object_
                }
        return None

    @classmethod
    def extract_subject_object(cls, text: str) -> Optional[Tuple[str, str]]:
        """テキストから主語と目的語を抽出"""
        # 基本的なパターン
        patterns = [
            r"(.*?)は(.*?)です",
            r"(.*?)は(.*?)である",
            r"(.*?)は(.*?)だ",
            r"(.*?)は(.*?)ですか",
            r"(.*?)は(.*?)でしょうか"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                subject, object_ = match.groups()
                return subject.strip(), object_.strip()
        
        return None

    @classmethod
    def find_relevant_facts(cls, subject: str, object_: str, facts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """関連する事実を抽出"""
        relevant = []
        for fact in facts:
            if (fact["subject"] == subject or fact["object"] == subject or
                fact["subject"] == object_ or fact["object"] == object_):
                relevant.append(fact)
        return relevant 