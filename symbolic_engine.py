import re
from typing import List, Dict, Any, Optional, Tuple, Set
from knowledge_base import KnowledgeBase
from datetime import datetime, timedelta
from relation_utils import RelationUtils
from functools import lru_cache
from collections import defaultdict
import concurrent.futures
from threading import Lock
import multiprocessing

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
            "今日": 0,
            "明日": 1,
            "明後日": 2,
            "昨日": -1,
            "一昨日": -2
        }

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

    @lru_cache(maxsize=64)
    def _handle_time_question(self, subject: str, object_: str) -> Optional[str]:
        """時間に関する質問を処理（キャッシュ付き）"""
        if subject not in self.time_expressions:
            return None

        # 時間表現に関連する事実を取得
        facts = self.knowledge_base.get_facts_by_time_expression(subject)
        if not facts:
            return f"{subject}の{object_}については分かりません。"

        # 直接の関係を確認
        for fact in facts:
            if fact["object"] == object_:
                return f"はい、{subject}は{fact['relation']} {object_}です。"
            elif fact["subject"] == object_:
                return f"はい、{object_}は{fact['relation']} {subject}です。"

        # 間接的な関係を探索
        for fact in facts:
            if fact["subject"] == subject:
                # 目的語から間接的な関係を探索
                indirect_path = self._find_indirect_path(fact["object"], object_)
                if indirect_path:
                    response = [f"はい、{subject}は{fact['relation']} {fact['object']}で、"]
                    for i in range(len(indirect_path) - 1):
                        current = indirect_path[i]
                        next_node = indirect_path[i + 1]
                        relation = self._find_relation_between(current, next_node)
                        if relation:
                            response.append(f"{current}は{relation} {next_node}です。")
                    return "".join(response)

        return f"{subject}の{object_}については分かりません。"

    def _handle_question(self, text: str) -> str:
        """質問を処理（インデックスを使用）"""
        try:
            print(f"質問処理: {text}")  # デバッグ用

            match = RelationUtils.extract_subject_object(text)
            if not match:
                return "質問の形式が理解できません。"
                
            subject, object_ = match
            print(f"抽出: 主語={subject}, 目的語={object_}")  # デバッグ用

            # 時間に関する質問の場合
            if subject in self.time_expressions:
                response = self._handle_time_question(subject, object_)
                if response:
                    return response

            # 直接の関係を確認
            relation = self._find_relation_between(subject, object_)
            if relation:
                print(f"直接の関係を発見: {relation}")  # デバッグ用
                return f"はい、{subject}は{relation} {object_}です。"

            # 間接的な関係を探索
            indirect_path = self._find_indirect_path(subject, object_)
            if indirect_path:
                print(f"間接的な関係を発見: {indirect_path}")  # デバッグ用
                response = ["はい、間接的な関係があります："]
                for i in range(len(indirect_path) - 1):
                    current = indirect_path[i]
                    next_node = indirect_path[i + 1]
                    relation = self._find_relation_between(current, next_node)
                    if relation:
                        response.append(f"- {current}は{relation} {next_node}です。")
                return "\n".join(response)

            print(f"関係が見つかりません: {subject}と{object_}")  # デバッグ用
            return f"{subject}と{object_}の関係についての情報がありません。"
        except Exception as e:
            print(f"質問処理中にエラーが発生しました: {e}")
            return "処理中にエラーが発生しました。"

    def _handle_fact(self, text: str) -> str:
        """事実を処理"""
        print(f"\n=== 入力テキスト: {text} ===")  # デバッグ用

        try:
            # 基本的な「AはBです」パターンを処理
            for i, pattern in enumerate(self.simple_patterns):
                try:
                    print(f"\nパターン {i+1} を試行: {pattern.pattern}")  # デバッグ用
                    match = pattern.match(text)
                    if match:
                        print(f"パターン {i+1} にマッチしました！")  # デバッグ用
                        groups = match.groups()
                        print(f"マッチしたグループ: {groups}")  # デバッグ用
                        
                        # グループの数に応じて処理を分岐
                        if len(groups) == 2:
                            subject, object_ = groups
                            print(f"2グループの場合: 主語={subject}, 目的語={object_}")  # デバッグ用
                        elif len(groups) == 3:
                            # 3つのグループがある場合、文脈に応じて主語と目的語を決定
                            if "の" in groups[0]:
                                subject = groups[0]
                                object_ = groups[2]
                                print(f"3グループ（の）の場合: 主語={subject}, 目的語={object_}")  # デバッグ用
                            elif any(particle in groups[1] for particle in ["が", "に", "を"]):
                                subject = groups[0]
                                object_ = groups[2]
                                print(f"3グループ（助詞）の場合: 主語={subject}, 目的語={object_}")  # デバッグ用
                            else:
                                subject = groups[0]
                                object_ = groups[1] + groups[2]
                                print(f"3グループ（その他）の場合: 主語={subject}, 目的語={object_}")  # デバッグ用
                        else:
                            print(f"グループ数が不正: {len(groups)}")  # デバッグ用
                            continue

                        subject = subject.strip()
                        object_ = object_.strip()
                        print(f"最終的な抽出: 主語={subject}, 目的語={object_}")  # デバッグ用
                        
                        # 関係性を推測
                        relation = None
                        
                        # 否定表現のチェック
                        is_negative = any(neg in text for neg in ["ではない", "じゃない", "ではありません"])
                        if is_negative:
                            print("否定表現を検出")  # デバッグ用
                        
                        # 文脈に基づく関係性の推測
                        if any(phrase in object_ for phrase in ["の一種", "に属する", "の仲間", "の一種です", "に属します"]):
                            # 関係性を「の一種」として保持し、object_から「の一種」を除去
                            relation = "の一種"
                            object_ = object_.replace("の一種", "").replace("の一種です", "").replace("に属します", "").strip()
                            print("関係性: の一種")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の特徴", "の性質", "の状態", "の特徴です", "の性質です"]):
                            relation = "属性"
                            print("関係性: 属性")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の行動", "の動作", "の振る舞い", "の行動です", "の動作です"]):
                            relation = "行動"
                            print("関係性: 行動")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の生息地", "の住処", "の場所", "の生息地です", "の場所です"]):
                            relation = "場所"
                            print("関係性: 場所")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["します", "する", "できます", "できる"]):
                            relation = "行動"
                            print("関係性: 行動")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["にあります", "に存在します", "に存在する"]):
                            relation = "場所"
                            print("関係性: 場所")  # デバッグ用
                        elif any(adj in object_ for adj in self.attribute_adjectives):
                            relation = "属性"
                            print("関係性: 属性（形容詞）")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の一部", "の構成要素", "の要素"]):
                            relation = "構成要素"
                            print("関係性: 構成要素")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の目的", "の目標", "の狙い"]):
                            relation = "目的"
                            print("関係性: 目的")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の原因", "の理由", "の要因"]):
                            relation = "原因"
                            print("関係性: 原因")  # デバッグ用
                        elif any(phrase in object_ for phrase in ["の結果", "の影響", "の効果"]):
                            relation = "結果"
                            print("関係性: 結果")  # デバッグ用
                        elif "状態" in object_:
                            relation = "状態"
                            print("関係性: 状態")  # デバッグ用
                        elif "特徴" in object_:
                            relation = "特徴"
                            print("関係性: 特徴")  # デバッグ用
                        elif "時" in object_ or "時期" in object_:
                            relation = "時間"
                            print("関係性: 時間")  # デバッグ用
                        
                        # 関係性が推測できない場合は、より詳細な分析を試みる
                        if relation is None:
                            print("関係性の推測を試みます...")  # デバッグ用
                            # 形容詞や状態を表す表現をチェック
                            if any(adj in object_ for adj in ["い", "な", "だ", "です"]):
                                relation = "属性"
                                print("関係性: 属性（終止形）")  # デバッグ用
                            # 動作を表す表現をチェック
                            elif any(verb in object_ for verb in ["る", "た", "ている", "ます"]):
                                relation = "行動"
                                print("関係性: 行動（終止形）")  # デバッグ用
                            # 場所を表す表現をチェック
                            elif any(loc in object_ for loc in ["で", "に", "へ", "から"]):
                                relation = "場所"
                                print("関係性: 場所（助詞）")  # デバッグ用
                            # それでも推測できない場合は、文脈から判断
                            else:
                                relation = "関連"  # デフォルトは「関連」
                                print("関係性: 関連（デフォルト）")  # デバッグ用

                        print(f"最終的な関係性: {relation}")  # デバッグ用

                        # 知識ベースに追加
                        if self.knowledge_base.add_fact(subject, object_, relation):
                            self._update_indices(subject, object_, relation)
                            if relation == "の一種":
                                response = f"了解しました。{subject}は{object_}の一種として記録しました。"
                            else:
                                response = f"了解しました。{subject}は{relation} {object_}として記録しました。"
                            if is_negative:
                                response = f"了解しました。{subject}は{relation} {object_}ではないとして記録しました。"
                            print(f"応答: {response}")  # デバッグ用
                            return response
                        else:
                            response = f"{subject}と{object_}の関係は既に記録されています。"
                            print(f"応答: {response}")  # デバッグ用
                            return response

                except Exception as e:
                    print(f"パターン {i+1} の処理中にエラーが発生しました: {e}")  # デバッグ用
                    continue

            print("\n=== パターンマッチング失敗 ===")  # デバッグ用
            return "すみません、理解できませんでした。"
        except Exception as e:
            print(f"\n=== エラーが発生しました: {e} ===")  # デバッグ用
            return "処理中にエラーが発生しました。"

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

            if is_list:
                return self.show_knowledge()
            elif is_question:
                return self._handle_question(text)
            else:
                return self._handle_fact(text)
        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")
            return "処理中にエラーが発生しました。"

    def __del__(self):
        """スレッドプールのクリーンアップ"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)