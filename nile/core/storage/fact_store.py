"""
Fact Store

事実データの効率的なストレージと検索を提供
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)

class FactStore:
    """効率的な事実ストレージ"""
    
    def __init__(self, db_path: str = "facts.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._init_database()
    
    def _init_database(self):
        """データベースを初期化"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 事実テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # インデックスを作成
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject ON facts(subject)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_object ON facts(object)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON facts(relation)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON facts(confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON facts(source)')
            
            # 複合インデックス
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_subject_relation ON facts(subject, relation)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_object ON facts(relation, object)')
            
            conn.commit()
            conn.close()
    
    def _generate_fact_id(self, subject: str, object_: str, relation: str) -> str:
        """事実の一意IDを生成"""
        content = f"{subject}|{relation}|{object_}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    def add_fact(self, subject: str, object_: str, relation: str = "is_a",
                 confidence: float = 1.0, source: str = "user") -> bool:
        """事実を追加"""
        with self.lock:
            fact_id = self._generate_fact_id(subject, object_, relation)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # 重複チェック
                cursor.execute(
                    'SELECT id FROM facts WHERE subject = ? AND relation = ? AND object = ?',
                    (subject, relation, object_)
                )
                if cursor.fetchone():
                    return False
                
                # 事実を挿入
                cursor.execute('''
                    INSERT INTO facts (id, subject, relation, object, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (fact_id, subject, relation, object_, confidence, source))
                
                conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"事実の追加中にエラー: {e}")
                return False
            finally:
                conn.close()
    
    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """IDで事実を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM facts WHERE id = ?', (fact_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "subject": row[1],
                    "relation": row[2],
                    "object": row[3],
                    "confidence": row[4],
                    "source": row[5],
                    "created_at": row[6],
                    "updated_at": row[7]
                }
            return None
    
    def get_facts(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """全事実を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def search_facts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """事実を検索"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            search_term = f"%{query}%"
            cursor.execute('''
                SELECT * FROM facts 
                WHERE subject LIKE ? OR relation LIKE ? OR object LIKE ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            ''', (search_term, search_term, search_term, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def get_facts_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """主語で事実を検索"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE subject = ?
                ORDER BY confidence DESC
            ''', (subject,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def get_facts_by_object(self, object_: str) -> List[Dict[str, Any]]:
        """目的語で事実を検索"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE object = ?
                ORDER BY confidence DESC
            ''', (object_,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def get_facts_by_relation(self, relation: str) -> List[Dict[str, Any]]:
        """関係で事実を検索"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE relation = ?
                ORDER BY confidence DESC
            ''', (relation,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def get_relation_between(self, subject: str, object_: str) -> Optional[Dict[str, Any]]:
        """2つのノード間の関係を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE subject = ? AND object = ?
                ORDER BY confidence DESC
                LIMIT 1
            ''', (subject, object_))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "subject": row[1],
                    "relation": row[2],
                    "object": row[3],
                    "confidence": row[4],
                    "source": row[5],
                    "created_at": row[6],
                    "updated_at": row[7]
                }
            return None
    
    def get_related_nodes(self, node: str) -> Set[str]:
        """ノードに関連する全てのノードを取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            related = set()
            
            # 主語としての関係
            cursor.execute('SELECT object FROM facts WHERE subject = ?', (node,))
            for row in cursor.fetchall():
                related.add(row[0])
            
            # 目的語としての関係
            cursor.execute('SELECT subject FROM facts WHERE object = ?', (node,))
            for row in cursor.fetchall():
                related.add(row[0])
            
            conn.close()
            return related
    
    def get_facts_by_confidence(self, min_confidence: float) -> List[Dict[str, Any]]:
        """信頼度でフィルタリング"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE confidence >= ?
                ORDER BY confidence DESC
            ''', (min_confidence,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def get_facts_by_source(self, source: str) -> List[Dict[str, Any]]:
        """情報源でフィルタリング"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM facts 
                WHERE source = ?
                ORDER BY created_at DESC
            ''', (source,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "id": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "confidence": row[4],
                "source": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in rows]
    
    def update_fact_confidence(self, fact_id: str, confidence: float) -> bool:
        """事実の信頼度を更新"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE facts 
                    SET confidence = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (confidence, fact_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                logger.error(f"信頼度の更新中にエラー: {e}")
                return False
            finally:
                conn.close()
    
    def delete_fact(self, fact_id: str) -> bool:
        """事実を削除"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('DELETE FROM facts WHERE id = ?', (fact_id,))
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                logger.error(f"事実の削除中にエラー: {e}")
                return False
            finally:
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 総数
            cursor.execute('SELECT COUNT(*) FROM facts')
            total_facts = cursor.fetchone()[0]
            
            # 信頼度の統計
            cursor.execute('SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM facts')
            avg_conf, min_conf, max_conf = cursor.fetchone()
            
            # 情報源の分布
            cursor.execute('SELECT source, COUNT(*) FROM facts GROUP BY source')
            sources = dict(cursor.fetchall())
            
            # 関係の分布
            cursor.execute('SELECT relation, COUNT(*) FROM facts GROUP BY relation ORDER BY COUNT(*) DESC LIMIT 10')
            relations = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_facts": total_facts,
                "confidence_stats": {
                    "average": avg_conf,
                    "minimum": min_conf,
                    "maximum": max_conf
                },
                "sources": sources,
                "top_relations": relations
            }
    
    def export_to_json(self, file_path: str) -> bool:
        """JSONファイルにエクスポート"""
        try:
            facts = self.get_facts(limit=10000)  # 最大10000件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({"facts": facts}, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"エクスポート中にエラー: {e}")
            return False
    
    def import_from_json(self, file_path: str) -> int:
        """JSONファイルからインポート"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            facts = data.get("facts", [])
            imported_count = 0
            
            for fact in facts:
                if self.add_fact(
                    subject=fact.get("subject", ""),
                    object_=fact.get("object", ""),
                    relation=fact.get("relation", "is_a"),
                    confidence=fact.get("confidence", 1.0),
                    source=fact.get("source", "import")
                ):
                    imported_count += 1
            
            return imported_count
            
        except Exception as e:
            logger.error(f"インポート中にエラー: {e}")
            return 0
