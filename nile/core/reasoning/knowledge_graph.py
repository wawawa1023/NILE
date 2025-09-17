"""
Knowledge Graph

知識グラフの構築と操作を提供
"""

import networkx as nx
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """知識グラフクラス"""
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.graph = nx.DiGraph()
        self.node_attributes = defaultdict(dict)
        self.edge_attributes = defaultdict(dict)
        self._build_graph()
    
    def _build_graph(self):
        """知識ベースからグラフを構築"""
        facts = self.knowledge_base.get_facts()
        
        for fact in facts:
            subject = fact["subject"]
            object_ = fact["object"]
            relation = fact["relation"]
            confidence = fact.get("confidence", 1.0)
            source = fact.get("source", "unknown")
            
            # ノードを追加
            if not self.graph.has_node(subject):
                self.graph.add_node(subject, type="entity")
                self.node_attributes[subject] = {
                    "type": "entity",
                    "degree": 0,
                    "in_degree": 0,
                    "out_degree": 0
                }
            
            if not self.graph.has_node(object_):
                self.graph.add_node(object_, type="entity")
                self.node_attributes[object_] = {
                    "type": "entity",
                    "degree": 0,
                    "in_degree": 0,
                    "out_degree": 0
                }
            
            # エッジを追加
            self.graph.add_edge(subject, object_, 
                              relation=relation, 
                              confidence=confidence,
                              source=source)
            
            # ノード属性を更新
            self.node_attributes[subject]["out_degree"] += 1
            self.node_attributes[object_]["in_degree"] += 1
            self.node_attributes[subject]["degree"] += 1
            self.node_attributes[object_]["degree"] += 1
    
    def add_relation(self, subject: str, object_: str, relation: str, 
                    confidence: float = 1.0, source: str = "user"):
        """関係を追加"""
        # ノードを追加
        if not self.graph.has_node(subject):
            self.graph.add_node(subject, type="entity")
            self.node_attributes[subject] = {
                "type": "entity",
                "degree": 0,
                "in_degree": 0,
                "out_degree": 0
            }
        
        if not self.graph.has_node(object_):
            self.graph.add_node(object_, type="entity")
            self.node_attributes[object_] = {
                "type": "entity",
                "degree": 0,
                "in_degree": 0,
                "out_degree": 0
            }
        
        # エッジを追加
        self.graph.add_edge(subject, object_, 
                          relation=relation, 
                          confidence=confidence,
                          source=source)
        
        # ノード属性を更新
        self.node_attributes[subject]["out_degree"] += 1
        self.node_attributes[object_]["in_degree"] += 1
        self.node_attributes[subject]["degree"] += 1
        self.node_attributes[object_]["degree"] += 1
    
    def find_path(self, start: str, end: str, max_length: int = 5) -> Optional[List[str]]:
        """2つのノード間の最短パスを検索"""
        try:
            if not self.graph.has_node(start) or not self.graph.has_node(end):
                return None
            
            path = nx.shortest_path(self.graph, start, end)
            if len(path) <= max_length:
                return path
            return None
        except nx.NetworkXNoPath:
            return None
    
    def find_all_paths(self, start: str, end: str, max_length: int = 3) -> List[List[str]]:
        """2つのノード間の全てのパスを検索"""
        try:
            if not self.graph.has_node(start) or not self.graph.has_node(end):
                return []
            
            paths = list(nx.all_simple_paths(self.graph, start, end, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_neighbors(self, node: str, direction: str = "both") -> Set[str]:
        """ノードの隣接ノードを取得"""
        if not self.graph.has_node(node):
            return set()
        
        if direction == "in":
            return set(self.graph.predecessors(node))
        elif direction == "out":
            return set(self.graph.successors(node))
        else:  # both
            return set(self.graph.neighbors(node))
    
    def get_relations(self, node1: str, node2: str) -> List[Dict[str, Any]]:
        """2つのノード間の関係を取得"""
        relations = []
        
        if self.graph.has_edge(node1, node2):
            edge_data = self.graph[node1][node2]
            relations.append({
                "subject": node1,
                "object": node2,
                "relation": edge_data.get("relation", "unknown"),
                "confidence": edge_data.get("confidence", 1.0),
                "source": edge_data.get("source", "unknown")
            })
        
        if self.graph.has_edge(node2, node1):
            edge_data = self.graph[node2][node1]
            relations.append({
                "subject": node2,
                "object": node1,
                "relation": edge_data.get("relation", "unknown"),
                "confidence": edge_data.get("confidence", 1.0),
                "source": edge_data.get("source", "unknown")
            })
        
        return relations
    
    def get_node_centrality(self, node: str) -> Dict[str, float]:
        """ノードの中心性を計算"""
        if not self.graph.has_node(node):
            return {}
        
        try:
            # 次数中心性
            degree_centrality = nx.degree_centrality(self.graph).get(node, 0)
            
            # 近接中心性
            closeness_centrality = nx.closeness_centrality(self.graph).get(node, 0)
            
            # 媒介中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(node, 0)
            
            return {
                "degree_centrality": degree_centrality,
                "closeness_centrality": closeness_centrality,
                "betweenness_centrality": betweenness_centrality
            }
        except Exception as e:
            logger.error(f"中心性計算中にエラー: {e}")
            return {}
    
    def get_most_central_nodes(self, n: int = 10) -> List[Tuple[str, Dict[str, float]]]:
        """最も中心的なノードを取得"""
        try:
            centrality_scores = {}
            
            for node in self.graph.nodes():
                centrality_scores[node] = self.get_node_centrality(node)
            
            # 次数中心性でソート
            sorted_nodes = sorted(
                centrality_scores.items(),
                key=lambda x: x[1].get("degree_centrality", 0),
                reverse=True
            )
            
            return sorted_nodes[:n]
        except Exception as e:
            logger.error(f"中心ノード取得中にエラー: {e}")
            return []
    
    def find_communities(self) -> Dict[int, List[str]]:
        """コミュニティを検出"""
        try:
            # 無向グラフに変換
            undirected_graph = self.graph.to_undirected()
            
            # コミュニティ検出（Louvain法）
            communities = nx.community.louvain_communities(undirected_graph)
            
            # 結果を辞書形式に変換
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[i] = list(community)
            
            return community_dict
        except Exception as e:
            logger.error(f"コミュニティ検出中にエラー: {e}")
            return {}
    
    def get_subgraph(self, nodes: List[str]) -> nx.DiGraph:
        """指定されたノードのサブグラフを取得"""
        return self.graph.subgraph(nodes)
    
    def get_relation_types(self) -> Dict[str, int]:
        """関係タイプの分布を取得"""
        relation_counts = defaultdict(int)
        
        for edge in self.graph.edges(data=True):
            relation = edge[2].get("relation", "unknown")
            relation_counts[relation] += 1
        
        return dict(relation_counts)
    
    def explain_reasoning(self, query: str) -> str:
        """推論過程の説明を生成"""
        # クエリからエンティティを抽出
        entities = self._extract_entities_from_query(query)
        
        if len(entities) < 2:
            return "推論に必要なエンティティが不足しています。"
        
        start, end = entities[0], entities[1]
        
        # パスを検索
        path = self.find_path(start, end)
        
        if not path:
            return f"{start}と{end}の間に関係が見つかりませんでした。"
        
        # 推論過程を説明
        explanation = f"推論過程：{start}から{end}への関係\n\n"
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            relations = self.get_relations(current, next_node)
            
            if relations:
                relation = relations[0]
                explanation += f"{i+1}. {current}は{relation['relation']} {next_node}です。\n"
        
        explanation += f"\n結論：{start}と{end}は関連しています。"
        
        return explanation
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """クエリからエンティティを抽出"""
        # 簡単なエンティティ抽出（実際の実装ではより高度な処理が必要）
        entities = []
        
        # "は"で分割してエンティティを抽出
        if "は" in query:
            parts = query.split("は")
            if len(parts) >= 2:
                entities.append(parts[0].strip())
                # 目的語を抽出
                object_part = parts[1].replace("ですか", "").replace("？", "").strip()
                entities.append(object_part)
        
        return entities
    
    def get_statistics(self) -> Dict[str, Any]:
        """グラフの統計情報を取得"""
        try:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph),
                "average_clustering": nx.average_clustering(self.graph.to_undirected()),
                "relation_types": self.get_relation_types(),
                "communities": len(self.find_communities())
            }
        except Exception as e:
            logger.error(f"統計情報取得中にエラー: {e}")
            return {}
    
    def export_to_json(self, file_path: str) -> bool:
        """グラフをJSONファイルにエクスポート"""
        try:
            # ノードデータ
            nodes = []
            for node in self.graph.nodes(data=True):
                node_data = {
                    "id": node[0],
                    "type": node[1].get("type", "entity"),
                    "attributes": self.node_attributes.get(node[0], {})
                }
                nodes.append(node_data)
            
            # エッジデータ
            edges = []
            for edge in self.graph.edges(data=True):
                edge_data = {
                    "source": edge[0],
                    "target": edge[1],
                    "relation": edge[2].get("relation", "unknown"),
                    "confidence": edge[2].get("confidence", 1.0),
                    "source_type": edge[2].get("source", "unknown")
                }
                edges.append(edge_data)
            
            # グラフデータ
            graph_data = {
                "nodes": nodes,
                "edges": edges,
                "statistics": self.get_statistics()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"エクスポート中にエラー: {e}")
            return False
    
    def import_from_json(self, file_path: str) -> bool:
        """JSONファイルからグラフをインポート"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # グラフをクリア
            self.graph.clear()
            self.node_attributes.clear()
            self.edge_attributes.clear()
            
            # ノードを追加
            for node_data in graph_data.get("nodes", []):
                node_id = node_data["id"]
                self.graph.add_node(node_id, type=node_data.get("type", "entity"))
                self.node_attributes[node_id] = node_data.get("attributes", {})
            
            # エッジを追加
            for edge_data in graph_data.get("edges", []):
                self.graph.add_edge(
                    edge_data["source"],
                    edge_data["target"],
                    relation=edge_data.get("relation", "unknown"),
                    confidence=edge_data.get("confidence", 1.0),
                    source=edge_data.get("source_type", "unknown")
                )
            
            return True
        except Exception as e:
            logger.error(f"インポート中にエラー: {e}")
            return False
