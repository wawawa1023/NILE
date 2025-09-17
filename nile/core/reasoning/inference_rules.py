"""
Inference Rules

推論ルールの定義と実行を提供
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InferenceType(Enum):
    """推論タイプ"""
    DEDUCTIVE = "deductive"  # 演繹推論
    INDUCTIVE = "inductive"  # 帰納推論
    ABDUCTIVE = "abductive"  # 仮説推論
    TEMPORAL = "temporal"    # 時間推論
    CAUSAL = "causal"        # 因果推論

@dataclass
class InferenceRule:
    """推論ルール"""
    name: str
    description: str
    inference_type: InferenceType
    premises: List[str]  # 前提
    conclusion: str      # 結論
    confidence: float    # 信頼度
    conditions: List[str] = None  # 条件

@dataclass
class InferenceResult:
    """推論結果"""
    rule_name: str
    conclusion: str
    confidence: float
    premises_used: List[str]
    inference_type: InferenceType
    explanation: str

class InferenceRuleEngine(ABC):
    """推論ルールエンジンの基底クラス"""
    
    @abstractmethod
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """ルールが適用可能かチェック"""
        pass
    
    @abstractmethod
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """ルールを適用"""
        pass

class DeductiveReasoner(InferenceRuleEngine):
    """演繹推論エンジン"""
    
    def __init__(self):
        self.rules = [
            InferenceRule(
                name="transitivity",
                description="推移性ルール",
                inference_type=InferenceType.DEDUCTIVE,
                premises=["AはBです", "BはCです"],
                conclusion="AはCです",
                confidence=0.9
            ),
            InferenceRule(
                name="symmetric_relation",
                description="対称関係ルール",
                inference_type=InferenceType.DEDUCTIVE,
                premises=["AはBと関連しています"],
                conclusion="BはAと関連しています",
                confidence=0.8
            ),
            InferenceRule(
                name="reflexive_relation",
                description="反射関係ルール",
                inference_type=InferenceType.DEDUCTIVE,
                premises=["Aは存在します"],
                conclusion="AはAと関連しています",
                confidence=0.7
            )
        ]
    
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """演繹推論が適用可能かチェック"""
        return len(facts) >= 2
    
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """演繹推論を適用"""
        for rule in self.rules:
            if self._matches_rule(facts, rule):
                return self._execute_rule(facts, rule)
        return None
    
    def _matches_rule(self, facts: List[Dict[str, Any]], rule: InferenceRule) -> bool:
        """事実がルールにマッチするかチェック"""
        if rule.name == "transitivity":
            return self._check_transitivity(facts)
        elif rule.name == "symmetric_relation":
            return self._check_symmetric_relation(facts)
        elif rule.name == "reflexive_relation":
            return self._check_reflexive_relation(facts)
        return False
    
    def _check_transitivity(self, facts: List[Dict[str, Any]]) -> bool:
        """推移性をチェック"""
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if (fact1["object"] == fact2["subject"] and 
                    fact1["relation"] == fact2["relation"]):
                    return True
        return False
    
    def _check_symmetric_relation(self, facts: List[Dict[str, Any]]) -> bool:
        """対称関係をチェック"""
        symmetric_relations = {"関連", "関係", "類似"}
        for fact in facts:
            if fact["relation"] in symmetric_relations:
                return True
        return False
    
    def _check_reflexive_relation(self, facts: List[Dict[str, Any]]) -> bool:
        """反射関係をチェック"""
        for fact in facts:
            if fact["subject"] == fact["object"]:
                return True
        return False
    
    def _execute_rule(self, facts: List[Dict[str, Any]], rule: InferenceRule) -> InferenceResult:
        """ルールを実行"""
        if rule.name == "transitivity":
            return self._execute_transitivity(facts, rule)
        elif rule.name == "symmetric_relation":
            return self._execute_symmetric_relation(facts, rule)
        elif rule.name == "reflexive_relation":
            return self._execute_reflexive_relation(facts, rule)
        return None
    
    def _execute_transitivity(self, facts: List[Dict[str, Any]], rule: InferenceRule) -> InferenceResult:
        """推移性ルールを実行"""
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if (fact1["object"] == fact2["subject"] and 
                    fact1["relation"] == fact2["relation"]):
                    
                    conclusion = f"{fact1['subject']}は{fact2['object']}です"
                    explanation = f"{fact1['subject']}は{fact1['object']}で、{fact2['subject']}は{fact2['object']}なので、推移性により{conclusion}"
                    
                    return InferenceResult(
                        rule_name=rule.name,
                        conclusion=conclusion,
                        confidence=rule.confidence,
                        premises_used=[f"{fact1['subject']}は{fact1['object']}です", f"{fact2['subject']}は{fact2['object']}です"],
                        inference_type=rule.inference_type,
                        explanation=explanation
                    )
        return None
    
    def _execute_symmetric_relation(self, facts: List[Dict[str, Any]], rule: InferenceRule) -> InferenceResult:
        """対称関係ルールを実行"""
        for fact in facts:
            if fact["relation"] in {"関連", "関係", "類似"}:
                conclusion = f"{fact['object']}は{fact['subject']}と関連しています"
                explanation = f"{fact['subject']}は{fact['object']}と関連しているので、対称性により{conclusion}"
                
                return InferenceResult(
                    rule_name=rule.name,
                    conclusion=conclusion,
                    confidence=rule.confidence,
                    premises_used=[f"{fact['subject']}は{fact['object']}と関連しています"],
                    inference_type=rule.inference_type,
                    explanation=explanation
                )
        return None
    
    def _execute_reflexive_relation(self, facts: List[Dict[str, Any]], rule: InferenceRule) -> InferenceResult:
        """反射関係ルールを実行"""
        for fact in facts:
            if fact["subject"] == fact["object"]:
                conclusion = f"{fact['subject']}は{fact['subject']}と関連しています"
                explanation = f"反射性により、{conclusion}"
                
                return InferenceResult(
                    rule_name=rule.name,
                    conclusion=conclusion,
                    confidence=rule.confidence,
                    premises_used=[f"{fact['subject']}は存在します"],
                    inference_type=rule.inference_type,
                    explanation=explanation
                )
        return None

class InductiveReasoner(InferenceRuleEngine):
    """帰納推論エンジン"""
    
    def __init__(self):
        self.rules = [
            InferenceRule(
                name="pattern_generalization",
                description="パターン一般化ルール",
                inference_type=InferenceType.INDUCTIVE,
                premises=["複数の類似した事実"],
                conclusion="一般的なパターン",
                confidence=0.6
            ),
            InferenceRule(
                name="statistical_inference",
                description="統計的推論ルール",
                inference_type=InferenceType.INDUCTIVE,
                premises=["統計データ"],
                conclusion="統計的結論",
                confidence=0.7
            )
        ]
    
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """帰納推論が適用可能かチェック"""
        return len(facts) >= 3
    
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """帰納推論を適用"""
        # パターン一般化
        pattern_result = self._find_patterns(facts)
        if pattern_result:
            return pattern_result
        
        # 統計的推論
        statistical_result = self._statistical_inference(facts)
        if statistical_result:
            return statistical_result
        
        return None
    
    def _find_patterns(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """パターンを検索"""
        # 関係タイプの分布を分析
        relation_counts = {}
        for fact in facts:
            relation = fact["relation"]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        # 最も頻繁な関係を特定
        if relation_counts:
            most_common_relation = max(relation_counts, key=relation_counts.get)
            if relation_counts[most_common_relation] >= 3:
                conclusion = f"多くの事実で{most_common_relation}関係が観察されます"
                explanation = f"{len(facts)}件の事実中{relation_counts[most_common_relation]}件で{most_common_relation}関係が見つかりました"
                
                return InferenceResult(
                    rule_name="pattern_generalization",
                    conclusion=conclusion,
                    confidence=0.6,
                    premises_used=[f"{len(facts)}件の事実"],
                    inference_type=InferenceType.INDUCTIVE,
                    explanation=explanation
                )
        
        return None
    
    def _statistical_inference(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """統計的推論"""
        # 信頼度の統計
        confidences = [fact.get("confidence", 1.0) for fact in facts]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > 0.8:
                conclusion = "事実の信頼度が高い傾向があります"
                explanation = f"平均信頼度: {avg_confidence:.2f}"
                
                return InferenceResult(
                    rule_name="statistical_inference",
                    conclusion=conclusion,
                    confidence=0.7,
                    premises_used=[f"{len(facts)}件の事実"],
                    inference_type=InferenceType.INDUCTIVE,
                    explanation=explanation
                )
        
        return None

class AbductiveReasoner(InferenceRuleEngine):
    """仮説推論エンジン"""
    
    def __init__(self):
        self.rules = [
            InferenceRule(
                name="best_explanation",
                description="最良の説明ルール",
                inference_type=InferenceType.ABDUCTIVE,
                premises=["観察された現象"],
                conclusion="最良の説明",
                confidence=0.5
            )
        ]
    
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """仮説推論が適用可能かチェック"""
        return len(facts) >= 1
    
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """仮説推論を適用"""
        # 観察された現象を分析
        observations = self._analyze_observations(facts)
        if observations:
            # 最良の説明を生成
            best_explanation = self._generate_best_explanation(observations)
            if best_explanation:
                return InferenceResult(
                    rule_name="best_explanation",
                    conclusion=best_explanation["conclusion"],
                    confidence=best_explanation["confidence"],
                    premises_used=observations,
                    inference_type=InferenceType.ABDUCTIVE,
                    explanation=best_explanation["explanation"]
                )
        
        return None
    
    def _analyze_observations(self, facts: List[Dict[str, Any]]) -> List[str]:
        """観察を分析"""
        observations = []
        for fact in facts:
            observation = f"{fact['subject']}は{fact['relation']} {fact['object']}です"
            observations.append(observation)
        return observations
    
    def _generate_best_explanation(self, observations: List[str]) -> Optional[Dict[str, Any]]:
        """最良の説明を生成"""
        if not observations:
            return None
        
        # 簡単な説明生成（実際の実装ではより高度な処理が必要）
        if len(observations) == 1:
            return {
                "conclusion": f"観察された現象: {observations[0]}",
                "confidence": 0.5,
                "explanation": "単一の観察に基づく説明"
            }
        else:
            return {
                "conclusion": f"{len(observations)}件の観察に基づく説明",
                "confidence": 0.6,
                "explanation": "複数の観察に基づく説明"
            }

class TemporalReasoner(InferenceRuleEngine):
    """時間推論エンジン"""
    
    def __init__(self):
        self.rules = [
            InferenceRule(
                name="temporal_sequence",
                description="時間順序ルール",
                inference_type=InferenceType.TEMPORAL,
                premises=["時間順序のある事実"],
                conclusion="時間的関係",
                confidence=0.8
            )
        ]
    
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """時間推論が適用可能かチェック"""
        # 時間表現を含む事実があるかチェック
        time_keywords = ["今日", "昨日", "明日", "朝", "昼", "夜", "時", "時期"]
        for fact in facts:
            if any(keyword in fact["subject"] or keyword in fact["object"] 
                   for keyword in time_keywords):
                return True
        return False
    
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """時間推論を適用"""
        temporal_facts = self._extract_temporal_facts(facts)
        if temporal_facts:
            return self._analyze_temporal_sequence(temporal_facts)
        return None
    
    def _extract_temporal_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """時間的事実を抽出"""
        time_keywords = ["今日", "昨日", "明日", "朝", "昼", "夜", "時", "時期"]
        temporal_facts = []
        
        for fact in facts:
            if any(keyword in fact["subject"] or keyword in fact["object"] 
                   for keyword in time_keywords):
                temporal_facts.append(fact)
        
        return temporal_facts
    
    def _analyze_temporal_sequence(self, temporal_facts: List[Dict[str, Any]]) -> InferenceResult:
        """時間順序を分析"""
        conclusion = "時間的な関係が観察されました"
        explanation = f"{len(temporal_facts)}件の時間的事実を分析しました"
        
        return InferenceResult(
            rule_name="temporal_sequence",
            conclusion=conclusion,
            confidence=0.8,
            premises_used=[f"{len(temporal_facts)}件の時間的事実"],
            inference_type=InferenceType.TEMPORAL,
            explanation=explanation
        )

class CausalReasoner(InferenceRuleEngine):
    """因果推論エンジン"""
    
    def __init__(self):
        self.rules = [
            InferenceRule(
                name="causal_inference",
                description="因果推論ルール",
                inference_type=InferenceType.CAUSAL,
                premises=["原因と結果の関係"],
                conclusion="因果関係",
                confidence=0.7
            )
        ]
    
    def can_apply(self, facts: List[Dict[str, Any]]) -> bool:
        """因果推論が適用可能かチェック"""
        causal_keywords = ["原因", "結果", "影響", "により", "によって", "のため"]
        for fact in facts:
            if any(keyword in fact["relation"] or keyword in fact["subject"] or keyword in fact["object"]
                   for keyword in causal_keywords):
                return True
        return False
    
    def apply(self, facts: List[Dict[str, Any]]) -> Optional[InferenceResult]:
        """因果推論を適用"""
        causal_facts = self._extract_causal_facts(facts)
        if causal_facts:
            return self._analyze_causal_relationships(causal_facts)
        return None
    
    def _extract_causal_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """因果的事実を抽出"""
        causal_keywords = ["原因", "結果", "影響", "により", "によって", "のため"]
        causal_facts = []
        
        for fact in facts:
            if any(keyword in fact["relation"] or keyword in fact["subject"] or keyword in fact["object"]
                   for keyword in causal_keywords):
                causal_facts.append(fact)
        
        return causal_facts
    
    def _analyze_causal_relationships(self, causal_facts: List[Dict[str, Any]]) -> InferenceResult:
        """因果関係を分析"""
        conclusion = "因果関係が観察されました"
        explanation = f"{len(causal_facts)}件の因果的事実を分析しました"
        
        return InferenceResult(
            rule_name="causal_inference",
            conclusion=conclusion,
            confidence=0.7,
            premises_used=[f"{len(causal_facts)}件の因果的事実"],
            inference_type=InferenceType.CAUSAL,
            explanation=explanation
        )

class InferenceRules:
    """推論ルール管理クラス"""
    
    def __init__(self):
        self.reasoners = {
            InferenceType.DEDUCTIVE: DeductiveReasoner(),
            InferenceType.INDUCTIVE: InductiveReasoner(),
            InferenceType.ABDUCTIVE: AbductiveReasoner(),
            InferenceType.TEMPORAL: TemporalReasoner(),
            InferenceType.CAUSAL: CausalReasoner()
        }
    
    def apply_inference(self, facts: List[Dict[str, Any]], 
                       inference_types: List[InferenceType] = None) -> List[InferenceResult]:
        """推論を適用"""
        if inference_types is None:
            inference_types = list(InferenceType)
        
        results = []
        
        for inference_type in inference_types:
            reasoner = self.reasoners.get(inference_type)
            if reasoner and reasoner.can_apply(facts):
                result = reasoner.apply(facts)
                if result:
                    results.append(result)
        
        # 信頼度でソート
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def get_available_inference_types(self, facts: List[Dict[str, Any]]) -> List[InferenceType]:
        """適用可能な推論タイプを取得"""
        available_types = []
        
        for inference_type, reasoner in self.reasoners.items():
            if reasoner.can_apply(facts):
                available_types.append(inference_type)
        
        return available_types
    
    def explain_inference(self, result: InferenceResult) -> str:
        """推論結果の説明を生成"""
        explanation = f"推論タイプ: {result.inference_type.value}\n"
        explanation += f"ルール: {result.rule_name}\n"
        explanation += f"結論: {result.conclusion}\n"
        explanation += f"信頼度: {result.confidence:.2f}\n"
        explanation += f"使用された前提: {', '.join(result.premises_used)}\n"
        explanation += f"説明: {result.explanation}"
        
        return explanation
