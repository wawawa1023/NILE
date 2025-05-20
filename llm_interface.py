import os
from typing import List, Dict, Any, Optional, Tuple
import re
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from relation_utils import RelationUtils

class LLMInterface:
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese"):
        try:
            self._init_model(model_name)
            self._init_patterns()
            load_dotenv()
            # 基本的な関係性を定義
            self.relations = {
                "種類": ["の一種", "の仲間", "の分類"],
                "属性": ["の特徴", "の性質", "の特徴"],
                "行動": ["の行動", "の習性", "の動作"],
                "場所": ["の生息地", "の住処", "の場所"]
            }
        except Exception as e:
            print(f"モデルの初期化中にエラーが発生しました: {e}")
            print("フォールバックとして軽量なモデルを使用します。")
            self._init_model("cl-tohoku/bert-base-japanese-char")

    def _init_model(self, model_name: str):
        """モデルとトークナイザーを初期化"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except Exception as e:
            print(f"モデル '{model_name}' の読み込みに失敗しました: {e}")
            raise

    def _init_patterns(self):
        """パターンを初期化"""
        self.fact_patterns = [
            r"(.*?)は(.*?)です",
            r"(.*?)は(.*?)の一種です",
            r"(.*?)は(.*?)に属します",
            r"(.*?)は(.*?)を持っています",
            r"(.*?)は(.*?)が特徴です"
        ]
        self.relation_patterns = {
            "種類": ["の一種", "に属する", "の仲間"],
            "属性": ["の特徴", "の性質", "の状態"],
            "行動": ["の行動", "の動作", "の振る舞い"],
            "場所": ["の生息地", "の住処", "の場所"]
        }

    def extract_facts(self, text: str) -> List[Dict[str, str]]:
        """テキストから事実を抽出する"""
        facts = []
        for pattern in RelationUtils.RELATION_PATTERNS.values():
            for p in pattern:
                matches = re.finditer(p, text)
                for match in matches:
                    subject, object_ = match.groups()
                    relation = RelationUtils.infer_relation(subject, object_)
                    facts.append({
                        "subject": subject.strip(),
                        "relation": relation,
                        "object": object_.strip()
                    })
        return facts

    def _infer_relation(self, subject: str, object_: str) -> str:
        """関係性を推測する"""
        for relation, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if pattern in object_:
                    return relation
        return "その他"

    def explain_reasoning(self, question: str, facts: List[Dict[str, str]]) -> str:
        """推論過程を説明する"""
        if not facts:
            return "関連する事実が見つかりませんでした。"

        # 質問から主語と目的語を抽出
        match = RelationUtils.extract_subject_object(question)
        if not match:
            return "質問の形式が理解できません。"
            
        subject, object_ = match

        # 関連する事実を抽出
        relevant_facts = RelationUtils.find_relevant_facts(subject, object_, facts)
        
        # 推論過程を構築
        reasoning = self._build_reasoning(subject, object_, relevant_facts)
        
        return reasoning

    def _find_relevant_facts(self, subject: str, object_: str, facts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """関連する事実を抽出する"""
        relevant_facts = []
        for fact in facts:
            if (fact["subject"] == subject or fact["object"] == subject or
                fact["subject"] == object_ or fact["object"] == object_):
                relevant_facts.append(fact)
        return relevant_facts

    def _build_reasoning(self, subject: str, object_: str, facts: List[Dict[str, str]]) -> str:
        """推論過程を構築する"""
        if not facts:
            return "関連する事実が見つかりませんでした。"

        reasoning = f"質問：{subject}は{object_}ですか？\n"
        reasoning += "以下の事実に基づいて推論しました：\n"

        # 直接の関係を確認
        direct_relation = RelationUtils.find_relation_between(subject, object_, facts)
        if direct_relation:
            reasoning += f"- {subject}は{direct_relation['relation']} {direct_relation['object']}です。\n"
            return reasoning

        # 間接的な関係を探索
        indirect_path = RelationUtils.find_indirect_path(subject, object_, facts)
        if indirect_path:
            reasoning += "間接的な関係が見つかりました：\n"
            for i in range(len(indirect_path) - 1):
                current = indirect_path[i]
                next_node = indirect_path[i + 1]
                relation = RelationUtils.find_relation_between(current, next_node, facts)
                if relation:
                    reasoning += f"- {current}は{relation['relation']} {next_node}です。\n"
            return reasoning

        # 関係が見つからない場合
        reasoning += "この質問に答えるためには、知識ベースに以下の情報が必要です：\n"
        reasoning += f"- {subject}の{object_}に関する情報\n"
        reasoning += f"- {object_}との関係性\n"
        return reasoning

    def generate_response(self, text: str) -> str:
        """テキストに対する応答を生成する"""
        try:
            # 入力テキストをトークン化
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # 応答を生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )

            # 生成されたテキストをデコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"応答生成中にエラーが発生しました: {e}")
            return "すみません、応答の生成に失敗しました。"

    def validate_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽出された事実を検証する"""
        validated_facts = []
        for fact in facts:
            if all(key in fact for key in ["subject", "relation", "object"]):
                if fact["subject"] and fact["object"]:
                    # 無意味なトークンを除外
                    if fact["object"] not in ["は", "です", "か", "？", "。", "、"]:
                        validated_facts.append(fact)
        return validated_facts
