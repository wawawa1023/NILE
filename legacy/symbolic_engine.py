import re
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from knowledge_base import KnowledgeBase
from datetime import datetime, timedelta
from relation_utils import RelationUtils
from functools import lru_cache
from collections import defaultdict
import concurrent.futures
from threading import Lock
import multiprocessing
import random

class SymbolicEngine:
    def __init__(self, knowledge_file: str = "knowledge.json"):
        self.max_workers = multiprocessing.cpu_count()
        self._init_locks()
        self._init_thread_pool()
        self._init_time_expressions()
        self._init_attribute_adjectives()
        self.knowledge_base = KnowledgeBase(knowledge_file)
        self._init_patterns()
        self._init_indices()
        self._init_llm_interface()

    def _init_locks(self):
        """ロックの初期化"""
        self.index_lock = Lock()
        self.kb_lock = Lock()

    def _init_patterns(self):
        """パターンのコンパイル"""
        # 基本的な「AはBです」パターン
        self.simple_patterns = [
            # 最も基本的なパターン（最初に試す）
            re.compile(r"^(.*?)は(.*?)です$"),
            re.compile(r"^(.*?)は(.*?)だ$"),
            re.compile(r"^(.*?)は(.*?)である$"),
            # 形容詞で終わるパターン
            re.compile(r"^(.*?)は(.*?)い$"),
            re.compile(r"^(.*?)は(.*?)な$"),
            # 否定表現
            re.compile(r"^(.*?)は(.*?)ではない$"),
            re.compile(r"^(.*?)は(.*?)じゃない$"),
            re.compile(r"^(.*?)は(.*?)ではありません$"),
            # 「です」の前に助詞が入るパターン
            re.compile(r"^(.*?)は(.*?)が(.*?)です$"),
            re.compile(r"^(.*?)は(.*?)に(.*?)です$"),
            re.compile(r"^(.*?)は(.*?)を(.*?)です$"),
            # 「です」の前に形容詞が入るパターン
            re.compile(r"^(.*?)は(.*?)が(.*?)いです$"),
            re.compile(r"^(.*?)は(.*?)が(.*?)なです$"),
            # 「ます」系
            re.compile(r"^(.*?)は(.*?)ます$"),
            re.compile(r"^(.*?)は(.*?)ました$"),
            # 「である」系
            re.compile(r"^(.*?)は(.*?)である$"),
            re.compile(r"^(.*?)は(.*?)であった$"),
            # 「だ」系
            re.compile(r"^(.*?)は(.*?)だ$"),
            re.compile(r"^(.*?)は(.*?)だった$"),
            # その他の一般的なパターン
            re.compile(r"^(.*?)の(.*?)は(.*?)です$"),
            re.compile(r"^(.*?)が(.*?)は(.*?)です$"),
            re.compile(r"^(.*?)に(.*?)は(.*?)です$"),
            re.compile(r"^(.*?)を(.*?)は(.*?)です$"),
            # 状態を表すパターン
            re.compile(r"^(.*?)は(.*?)状態です$"),
            re.compile(r"^(.*?)は(.*?)状態である$"),
            # 特徴を表すパターン
            re.compile(r"^(.*?)は(.*?)特徴です$"),
            re.compile(r"^(.*?)は(.*?)特徴である$"),
            # 種類を表すパターン
            re.compile(r"^(.*?)は(.*?)の一種です$"),
            re.compile(r"^(.*?)は(.*?)に属します$"),
            # 場所を表すパターン
            re.compile(r"^(.*?)は(.*?)にあります$"),
            re.compile(r"^(.*?)は(.*?)に存在します$"),
            # 時間を表すパターン
            re.compile(r"^(.*?)は(.*?)の時です$"),
            re.compile(r"^(.*?)は(.*?)の時期です$")
        ]

    def _init_time_expressions(self):
        """時間表現の初期化"""
        self.time_expressions = {
            # 相対的な時間表現
            "今日": 0,
            "明日": 1,
            "明後日": 2,
            "昨日": -1,
            "一昨日": -2,
            "今週": 0,
            "来週": 7,
            "先週": -7,
            "今月": 0,
            "来月": 30,
            "先月": -30,
            # 絶対的な時間表現
            "朝": "morning",
            "昼": "noon",
            "夕方": "evening",
            "夜": "night",
            # 期間表現
            "毎日": "daily",
            "毎週": "weekly",
            "毎月": "monthly",
            "毎年": "yearly"
        }
        
        # 時間表現のパターン
        self.time_patterns = [
            re.compile(r"(\d+)日前"),
            re.compile(r"(\d+)日後"),
            re.compile(r"(\d+)週間前"),
            re.compile(r"(\d+)週間後"),
            re.compile(r"(\d+)ヶ月前"),
            re.compile(r"(\d+)ヶ月後"),
            re.compile(r"(\d+)年前"),
            re.compile(r"(\d+)年後")
        ]

    def _init_attribute_adjectives(self):
        """属性を表す形容詞の初期化"""
        self.attribute_adjectives = frozenset({
            "青い", "赤い", "白い", "黒い", "大きい", "小さい",
            "高い", "低い", "暑い", "寒い", "新しい", "古い",
            "美しい", "醜い", "強い", "弱い", "速い", "遅い"
        })

    def _init_indices(self):
        """インデックスの初期化"""
        self.subject_index = defaultdict(set)  # 主語から関係を検索
        self.object_index = defaultdict(set)   # 目的語から関係を検索
        self.relation_index = defaultdict(set) # 関係から事実を検索
        self._rebuild_indices()

    def _init_thread_pool(self):
        """スレッドプールの初期化"""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="SymbolicEngine"
        )

    def _rebuild_indices(self):
        """インデックスを並列で再構築（最適化版）"""
        self.subject_index.clear()
        self.object_index.clear()
        self.relation_index.clear()

        facts = self.knowledge_base.get_facts()
        futures = []
        
        # 並列処理でインデックスを構築
        for fact in facts:
            if isinstance(fact, dict) and all(k in fact for k in ["subject", "relation", "object"]):
                futures.append(
                    self.thread_pool.submit(
                        self._update_indices,
                        fact["subject"],
                        fact["object"],
                        fact["relation"]
                    )
                )

        # 全ての処理が完了するのを待つ
        concurrent.futures.wait(futures)

    def _update_indices(self, subject: str, object_: str, relation: str):
        """インデックスを更新（スレッドセーフ）"""
        with self.index_lock:
            self.subject_index[subject].add((object_, relation))
            self.object_index[object_].add((subject, relation))
            self.relation_index[relation].add((subject, object_))

    def _find_related_nodes(self, node: str) -> Set[str]:
        """ノードに関連する全てのノードを並列で取得（最適化版）"""
        futures = []
        
        # 主語と目的語の関係を並列で処理
        futures.append(self.thread_pool.submit(
            lambda: {obj for obj, _ in self.subject_index[node]}
        ))
        futures.append(self.thread_pool.submit(
            lambda: {subj for subj, _ in self.object_index[node]}
        ))
        
        # 結果を待機して結合
        related = set()
        for future in concurrent.futures.as_completed(futures):
            related.update(future.result())
            
        return related

    def _find_relation_between(self, subject: str, object_: str) -> Optional[str]:
        """2つのノード間の関係を並列で検索（最適化版）"""
        futures = []
        
        # 主語と目的語の関係を並列で検索
        futures.append(self.thread_pool.submit(
            lambda: next((rel for obj, rel in self.subject_index[subject] if obj == object_), None)
        ))
        futures.append(self.thread_pool.submit(
            lambda: next((RelationUtils.get_reverse_relation(rel) 
                         for subj, rel in self.object_index[object_] if subj == subject), None)
        ))
        
        # 最初に見つかった有効な関係を返す
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                return result
                
        return None

    def _find_indirect_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """間接的な関係のパスを並列で探索（最適化版）"""
        def process_path(current: str, target: str, path: List[str], depth: int) -> Optional[List[str]]:
            if depth > max_depth:
                return None
            if current == target:
                return path

            # 関連するノードを並列で処理
            related_nodes = self._find_related_nodes(current)
            futures = []
            
            for next_node in related_nodes:
                if next_node not in path:
                    new_path = path + [next_node]
                    futures.append(
                        self.thread_pool.submit(
                            process_path, next_node, target, new_path, depth + 1
                        )
                    )

            # 最初に見つかった有効なパスを返す
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    return result

            return None

        return process_path(start, end, [start], 0)

    @lru_cache(maxsize=128)
    def _is_list_command(self, text: str) -> bool:
        """リスト表示コマンドかどうかを判定（キャッシュ付き）"""
        list_commands = {
            "list", "リスト", "表示", "一覧", "show", "all",
            "知識", "内容", "確認", "見る", "見せる"
        }
        return text.strip() in list_commands

    @lru_cache(maxsize=128)
    def _is_question(self, text: str) -> bool:
        """質問かどうかを判定（キャッシュ付き）"""
        print(f"\n=== 質問判定開始: {text} ===")  # デバッグ用
        is_q = "ですか" in text or "？" in text or "?" in text
        print(f"質問判定結果: {is_q}")  # デバッグ用
        return is_q

    def _parse_time_expression(self, time_expr: str) -> Optional[Union[int, str]]:
        """時間表現を解析"""
        # 直接の時間表現
        if time_expr in self.time_expressions:
            return self.time_expressions[time_expr]
            
        # パターンマッチング
        for pattern in self.time_patterns:
            match = pattern.match(time_expr)
            if match:
                number = int(match.group(1))
                if "前" in time_expr:
                    return -number
                return number
                
        return None

    @lru_cache(maxsize=64)
    def _handle_time_question(self, subject: str, object_: str) -> Optional[str]:
        """時間に関する質問を処理（改善版）"""
        time_value = self._parse_time_expression(subject)
        if time_value is None:
            return None

        # 時間表現に関連する事実を取得
        facts = self.knowledge_base.get_facts_by_time_expression(subject)
        if not facts:
            return f"{subject}の{object_}については分かりません。"

        # 直接の関係を確認
        for fact in facts:
            if fact["object"] == object_:
                return f"はい、{subject}と{object_}の関係が存在します。"
            elif fact["subject"] == object_:
                return f"はい、{object_}と{subject}の関係が存在します。"

        # 間接的な関係を探索
        for fact in facts:
            if fact["subject"] == subject:
                indirect_path = self._find_indirect_path(fact["object"], object_)
                if indirect_path:
                    response = [f"はい、{subject}と{object_}の間接的な関係が存在します："]
                    for i in range(len(indirect_path) - 1):
                        current = indirect_path[i]
                        next_node = indirect_path[i + 1]
                        relation = self._find_relation_between(current, next_node)
                        if relation:
                            response.append(f"- {current}と{next_node}の関係が存在します。")
                    return "\n".join(response)

        return f"{subject}の{object_}については分かりません。"

    def _init_llm_interface(self):
        """言語モデルインターフェースの初期化"""
        from llm_interface import LLMInterface
        self.llm = LLMInterface()

    def _handle_fact(self, text: str) -> str:
        """事実を処理（ハイブリッドアプローチ）"""
        print(f"\n=== 入力テキスト: {text} ===")  # デバッグ用

        try:
            # 言語モデルによる解析を試みる
            llm_result = self.llm.analyze_fact(text)
            if llm_result:
                subject = llm_result.get("subject")
                object_ = llm_result.get("object")
                relation = llm_result.get("relation")
                
                if all([subject, object_, relation]):
                    # 知識ベースに追加
                    if self.knowledge_base.add_fact(subject, object_, relation):
                        self._update_indices(subject, object_, relation)
                        if relation == "の一種":
                            return f"了解しました。{subject}は{object_}の一種として記録しました。"
                        else:
                            return f"了解しました。{subject}と{object_}の関係を記録しました。"
                    else:
                        return f"{subject}と{object_}の関係は既に記録されています。"

            # 言語モデルによる解析が失敗した場合、従来のルールベース処理を実行
            return self._handle_fact_rule_based(text)
        except Exception as e:
            print(f"\n=== エラーが発生しました: {e} ===")  # デバッグ用
            return "処理中にエラーが発生しました。"

    def _handle_fact_rule_based(self, text: str) -> str:
        """ルールベースの事実処理（既存の_handle_factの内容）"""
        # 既存の_handle_factの内容をここに移動
        # ... 既存のコード ...

    def _handle_question(self, text: str) -> str:
        """質問を処理（ハイブリッドアプローチ）"""
        try:
            print(f"質問処理: {text}")  # デバッグ用

            # 言語モデルによる解析を試みる
            llm_result = self.llm.analyze_question(text)
            if llm_result:
                subject = llm_result.get("subject")
                object_ = llm_result.get("object")
                
                if subject and object_:
                    # 時間に関する質問の場合
                    if subject in self.time_expressions:
                        response = self._handle_time_question(subject, object_)
                        if response:
                            return response

                    # 直接の関係を確認
                    relation = self._find_relation_between(subject, object_)
                    if relation:
                        # より自然な応答パターンを生成
                        response_patterns = [
                            f"はい、{subject}は{object_}です。",
                            f"ええ、{subject}は{object_}と言えます。",
                            f"その通りです。{subject}は{object_}です。",
                            f"はい、{subject}は{object_}という特徴があります。"
                        ]
                        return random.choice(response_patterns)

                    # 間接的な関係を探索
                    indirect_path = self._find_indirect_path(subject, object_)
                    if indirect_path:
                        # より自然な説明を生成
                        intro_patterns = [
                            "はい、間接的な関係があります。",
                            "ええ、関連性が見つかりました。",
                            "その通りです。以下のような関係があります。"
                        ]
                        response = [random.choice(intro_patterns)]
                        
                        for i in range(len(indirect_path) - 1):
                            current = indirect_path[i]
                            next_node = indirect_path[i + 1]
                            relation = self._find_relation_between(current, next_node)
                            if relation:
                                # より自然な関係の説明を生成
                                relation_patterns = [
                                    f"- {current}は{next_node}と関連しています。",
                                    f"- {current}と{next_node}の間に関係があります。",
                                    f"- {current}は{next_node}に関連する特徴を持っています。"
                                ]
                                response.append(random.choice(relation_patterns))
                        return "\n".join(response)

                    # 関係が見つからない場合の応答
                    not_found_patterns = [
                        f"申し訳ありません。{subject}と{object_}の関係については分かりません。",
                        f"すみません。{subject}の{object_}に関する情報は見つかりませんでした。",
                        f"残念ながら、{subject}と{object_}の関係については情報がありません。"
                    ]
                    return random.choice(not_found_patterns)

            # 言語モデルによる解析が失敗した場合、従来のルールベース処理を実行
            return self._handle_question_rule_based(text)
        except Exception as e:
            print(f"質問処理中にエラーが発生しました: {e}")
            error_patterns = [
                "申し訳ありません。処理中にエラーが発生しました。",
                "すみません。エラーが発生してしまいました。",
                "申し訳ありませんが、エラーが発生しました。"
            ]
            return random.choice(error_patterns)

    def _handle_question_rule_based(self, text: str) -> str:
        """ルールベースの質問処理"""
        try:
            # 基本的なパターンで主語と目的語を抽出
            for pattern in self.simple_patterns:
                match = pattern.match(text)
                if match:
                    subject = match.group(1).strip()
                    object_ = match.group(2).strip()
                    
                    # 時間に関する質問の場合
                    if subject in self.time_expressions:
                        response = self._handle_time_question(subject, object_)
                        if response:
                            return response

                    # 直接の関係を確認
                    relation = self._find_relation_between(subject, object_)
                    if relation:
                        # より自然な応答パターンを生成
                        response_patterns = [
                            f"はい、{subject}は{object_}です。",
                            f"ええ、{subject}は{object_}と言えます。",
                            f"その通りです。{subject}は{object_}です。",
                            f"はい、{subject}は{object_}という特徴があります。"
                        ]
                        return random.choice(response_patterns)

                    # 間接的な関係を探索
                    indirect_path = self._find_indirect_path(subject, object_)
                    if indirect_path:
                        # より自然な説明を生成
                        intro_patterns = [
                            "はい、間接的な関係があります。",
                            "ええ、関連性が見つかりました。",
                            "その通りです。以下のような関係があります。"
                        ]
                        response = [random.choice(intro_patterns)]
                        
                        for i in range(len(indirect_path) - 1):
                            current = indirect_path[i]
                            next_node = indirect_path[i + 1]
                            relation = self._find_relation_between(current, next_node)
                            if relation:
                                # より自然な関係の説明を生成
                                relation_patterns = [
                                    f"- {current}は{next_node}と関連しています。",
                                    f"- {current}と{next_node}の間に関係があります。",
                                    f"- {current}は{next_node}に関連する特徴を持っています。"
                                ]
                                response.append(random.choice(relation_patterns))
                        return "\n".join(response)

                    # 関係が見つからない場合の応答
                    not_found_patterns = [
                        f"申し訳ありません。{subject}と{object_}の関係については分かりません。",
                        f"すみません。{subject}の{object_}に関する情報は見つかりませんでした。",
                        f"残念ながら、{subject}と{object_}の関係については情報がありません。"
                    ]
                    return random.choice(not_found_patterns)

            # パターンに一致しない場合
            return "申し訳ありません。質問の形式が理解できませんでした。"
        except Exception as e:
            print(f"ルールベースの質問処理中にエラーが発生しました: {e}")
            error_patterns = [
                "申し訳ありません。処理中にエラーが発生しました。",
                "すみません。エラーが発生してしまいました。",
                "申し訳ありませんが、エラーが発生しました。"
            ]
            return random.choice(error_patterns)

    def show_knowledge(self) -> str:
        """知識ベースの内容を表示"""
        facts = self.knowledge_base.get_facts()
        if not facts:
            return "知識ベースは空です。"
        
        result = ["知識ベースの内容："]
        for fact in facts:
            result.append(f"- {fact['subject']}は{fact['relation']} {fact['object']}です。")
        return "\n".join(result)

    def process_input(self, text: str) -> str:
        """入力を処理（最適化版）"""
        try:
            text = text.strip()
            if not text:
                return "入力が空です。"

            # 入力タイプの判定を並列で実行
            futures = []
            futures.append(self.thread_pool.submit(self._is_list_command, text))
            futures.append(self.thread_pool.submit(self._is_question, text))
            
            is_list, is_question = [f.result() for f in concurrent.futures.as_completed(futures)]

            # 応答の生成
            if is_list:
                response = self.show_knowledge()
            elif is_question:
                response = self._handle_question(text)
            else:
                response = self._handle_fact(text)

            # 応答の自然さを向上
            response = self._enhance_response(response, text)
            
            return response
        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")
            return "申し訳ありません。処理中にエラーが発生しました。"

    def _enhance_response(self, response: str, original_text: str) -> str:
        """応答をより自然にする"""
        # 応答が空の場合
        if not response:
            return "申し訳ありません。適切な応答を生成できませんでした。"

        # 応答の種類に応じて接頭辞を追加
        if response.startswith("はい"):
            prefixes = ["はい、", "ええ、", "その通りです。"]
            response = random.choice(prefixes) + response[2:]
        elif response.startswith("いいえ"):
            prefixes = ["いいえ、", "申し訳ありませんが、", "残念ながら、"]
            response = random.choice(prefixes) + response[3:]
        elif "分かりません" in response:
            response = "申し訳ありません。" + response

        # 文末の処理
        if not response.endswith("。") and not response.endswith("！") and not response.endswith("？"):
            response += "。"

        # 連続する句読点を整理
        response = re.sub(r'[。、]{2,}', '。', response)

        # 空白を整理
        response = re.sub(r'\s+', ' ', response).strip()

        return response

    def __del__(self):
        """スレッドプールのクリーンアップ"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)