from typing import Dict, Any, List
import json
from llm_interface import LLMInterface
from symbolic_engine import SymbolicEngine

class Controller:
    def __init__(self, llm: LLMInterface, knowledge_base: Dict[str, Any]):
        self.llm = llm
        self.symbolic_engine = SymbolicEngine("knowledge.json")

    def process_query(self, query: str) -> str:
        """ユーザーのクエリを処理する"""
        # 一覧表示の処理
        if query.strip() in ["一覧", "list", "show"]:
            return self.symbolic_engine.show_knowledge()

        # 事実の抽出
        facts = self.llm.extract_facts(query)
        
        # 事実の検証
        validated_facts = self.llm.validate_facts(facts)
        
        # 質問の場合
        if any(fact.get("type") == "question" for fact in validated_facts):
            # シンボリックエンジンで推論
            response = self.symbolic_engine.process_input(query)
            if response.startswith("はい") or response.startswith("いいえ"):
                # 推論の説明を生成
                explanation = self.llm.explain_reasoning(validated_facts, query)
                return f"{response}\n\n{explanation}"
            return response
        
        # 事実の登録の場合
        if any(fact.get("type") == "fact" for fact in validated_facts):
            # シンボリックエンジンで事実を登録
            return self.symbolic_engine.process_input(query)
        
        return "すみません、理解できませんでした。"

    def _save_knowledge_base(self):
        """知識ベースをファイルに保存する"""
        with open("knowledge.json", "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2) 