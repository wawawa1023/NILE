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
            self.relation_patterns = {
                "の一種": ["の一種", "に属する", "の仲間", "の一種です", "に属します"],
                "属性": ["の特徴", "の性質", "の状態", "の特徴です", "の性質です"],
                "行動": ["の行動", "の動作", "の振る舞い", "の行動です", "の動作です"],
                "場所": ["の生息地", "の住処", "の場所", "の生息地です", "の場所です"],
                "構成要素": ["の一部", "の構成要素", "の要素"],
                "目的": ["の目的", "の目標", "の狙い"],
                "原因": ["の原因", "の理由", "の要因"],
                "結果": ["の結果", "の影響", "の効果"],
                "状態": ["状態"],
                "特徴": ["特徴"],
                "時間": ["時", "時期"]
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

    def analyze_fact(self, text: str) -> Optional[Dict[str, str]]:
        """事実を解析（言語モデルを使用）"""
        try:
            # 言語モデルによる解析を試みる
            prompt = f"""以下の文を解析し、主語、目的語、関係性を抽出してください。
文: {text}
出力形式:
主語: [主語]
目的語: [目的語]
関係性: [関係性]

注意: 必ず上記の形式で出力してください。余分な文字は含めないでください。"""
            
            response = self.generate_response(prompt)
            
            # 応答を解析
            subject = None
            object_ = None
            relation = None
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('主語:'):
                    subject = line.replace('主語:', '').strip()
                elif line.startswith('目的語:'):
                    object_ = line.replace('目的語:', '').strip()
                elif line.startswith('関係性:'):
                    relation = line.replace('関係性:', '').strip()
            
            if all([subject, object_, relation]):
                return {
                    "subject": subject,
                    "object": object_,
                    "relation": relation
                }
            
            # 言語モデルの解析が失敗した場合、従来のパターンマッチングにフォールバック
            pattern = re.compile(r"^(.*?)は(.*?)です$")
            match = pattern.match(text)
            
            if not match:
                return None
                
            subject = match.group(1).strip()
            object_ = match.group(2).strip()
            relation = self._infer_relation(subject, object_)
            
            return {
                "subject": subject,
                "object": object_,
                "relation": relation
            }
        except Exception as e:
            print(f"事実の解析中にエラーが発生しました: {e}")
            return None

    def analyze_question(self, text: str) -> Optional[Dict[str, str]]:
        """質問を解析（言語モデルを使用）"""
        try:
            # 言語モデルによる解析を試みる
            prompt = f"""以下の質問を解析し、主語と目的語を抽出してください。
質問: {text}
出力形式:
主語: [主語]
目的語: [目的語]

注意: 必ず上記の形式で出力してください。余分な文字は含めないでください。"""
            
            response = self.generate_response(prompt)
            
            # 応答を解析
            subject = None
            object_ = None
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('主語:'):
                    subject = line.replace('主語:', '').strip()
                elif line.startswith('目的語:'):
                    object_ = line.replace('目的語:', '').strip()
            
            if subject and object_:
                return {
                    "subject": subject,
                    "object": object_
                }
            
            # 言語モデルの解析が失敗した場合、従来のパターンマッチングにフォールバック
            pattern = re.compile(r"^(.*?)は(.*?)ですか？$")
            match = pattern.match(text)
            
            if not match:
                return None
                
            subject = match.group(1).strip()
            object_ = match.group(2).strip()
            
            return {
                "subject": subject,
                "object": object_
            }
        except Exception as e:
            print(f"質問の解析中にエラーが発生しました: {e}")
            return None

    def _infer_relation(self, subject: str, object_: str) -> str:
        """関係性を推測"""
        # 文脈に基づく関係性の推測
        for relation, patterns in self.relation_patterns.items():
            if any(pattern in object_ for pattern in patterns):
                return relation
                
        # 形容詞や状態を表す表現をチェック
        if any(adj in object_ for adj in ["い", "な", "だ", "です"]):
            return "属性"
            
        # 動作を表す表現をチェック
        if any(verb in object_ for verb in ["る", "た", "ている", "ます"]):
            return "行動"
            
        # 場所を表す表現をチェック
        if any(loc in object_ for loc in ["で", "に", "へ", "から"]):
            return "場所"
            
        # デフォルトは「関連」
        return "関連"

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
        """テキストに対する応答を生成する（改善版）"""
        try:
            # 入力テキストをトークン化
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # 応答を生成（パラメータを調整）
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,  # より長い応答を許可
                    num_return_sequences=1,
                    temperature=0.7,  # より創造的な出力
                    top_p=0.9,       # 核サンプリング
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.5,  # 適度な繰り返し防止
                    no_repeat_ngram_size=3,  # 3-gramの繰り返しを防ぐ
                    num_beams=5,  # ビームサーチを使用
                    early_stopping=True,  # 適切な終了を促進
                    length_penalty=1.0,  # 長さのペナルティを調整
                    min_length=10  # 最小の応答長を設定
                )

            # 生成されたテキストをデコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 応答から元のプロンプトを除去
            response = response.replace(text, "").strip()
            
            # 応答の整形
            response = self._format_response(response)
            
            return response
        except Exception as e:
            print(f"応答生成中にエラーが発生しました: {e}")
            return "すみません、応答の生成に失敗しました。"

    def _format_response(self, response: str) -> str:
        """応答を整形する"""
        # 不要な文字を除去
        response = re.sub(r'\s+', ' ', response)
        
        # 文末の句読点を適切に処理
        response = re.sub(r'[。、]+$', '。', response)
        
        # 文頭の不要な文字を除去
        response = re.sub(r'^[、。\s]+', '', response)
        
        # 文末に句点がない場合は追加
        if not response.endswith('。'):
            response += '。'
            
        # 連続する句読点を整理
        response = re.sub(r'[。、]{2,}', '。', response)
        
        # 空白を整理
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response

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
