"""
Cache Manager

効率的なキャッシュ管理を提供
"""

import time
import threading
from typing import Any, Dict, Optional, Callable
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class LRUCache:
    """LRU（Least Recently Used）キャッシュ実装"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        with self.lock:
            if key in self.cache:
                # アクセスされたアイテムを最後に移動
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """キャッシュに値を保存"""
        with self.lock:
            if key in self.cache:
                # 既存のキーを更新
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 最も古いアイテムを削除
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """キャッシュから値を削除"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュの統計情報を取得"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

class TimedCache:
    """時間ベースのキャッシュ実装"""
    
    def __init__(self, default_ttl: int = 3600):  # デフォルト1時間
        self.default_ttl = default_ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得（TTLチェック付き）"""
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.default_ttl:
                    self.hits += 1
                    return self.cache[key]
                else:
                    # TTL切れのアイテムを削除
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """キャッシュに値を保存"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """キャッシュから値を削除"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def cleanup_expired(self) -> int:
        """期限切れのアイテムを削除"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp >= self.default_ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュの統計情報を取得"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "expired_items": self.cleanup_expired()
        }

class CacheManager:
    """統合キャッシュマネージャー"""
    
    def __init__(self):
        # 異なる用途のキャッシュ
        self.reasoning_cache = LRUCache(max_size=500)  # 推論結果
        self.embedding_cache = TimedCache(default_ttl=7200)  # BERT埋め込み（2時間）
        self.pattern_cache = LRUCache(max_size=200)  # パターンマッチング結果
        self.fact_cache = TimedCache(default_ttl=1800)  # 事実検索結果（30分）
        
        # バックグラウンドクリーンアップ
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """バックグラウンドクリーンアップスレッドを開始"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # 5分ごと
                try:
                    self.embedding_cache.cleanup_expired()
                    self.fact_cache.cleanup_expired()
                except Exception as e:
                    logger.error(f"キャッシュクリーンアップ中にエラー: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get_reasoning_result(self, query: str) -> Optional[Any]:
        """推論結果をキャッシュから取得"""
        return self.reasoning_cache.get(query)
    
    def cache_reasoning_result(self, query: str, result: Any) -> None:
        """推論結果をキャッシュに保存"""
        self.reasoning_cache.put(query, result)
    
    def get_embedding(self, text: str) -> Optional[Any]:
        """埋め込みベクトルをキャッシュから取得"""
        return self.embedding_cache.get(text)
    
    def cache_embedding(self, text: str, embedding: Any) -> None:
        """埋め込みベクトルをキャッシュに保存"""
        self.embedding_cache.put(text, embedding)
    
    def get_pattern_result(self, pattern: str, text: str) -> Optional[Any]:
        """パターンマッチング結果をキャッシュから取得"""
        key = f"{pattern}:{text}"
        return self.pattern_cache.get(key)
    
    def cache_pattern_result(self, pattern: str, text: str, result: Any) -> None:
        """パターンマッチング結果をキャッシュに保存"""
        key = f"{pattern}:{text}"
        self.pattern_cache.put(key, result)
    
    def get_fact_search_result(self, query: str) -> Optional[Any]:
        """事実検索結果をキャッシュから取得"""
        return self.fact_cache.get(query)
    
    def cache_fact_search_result(self, query: str, result: Any) -> None:
        """事実検索結果をキャッシュに保存"""
        self.fact_cache.put(query, result)
    
    def clear_all(self) -> None:
        """全てのキャッシュをクリア"""
        self.reasoning_cache.clear()
        self.embedding_cache.clear()
        self.pattern_cache.clear()
        self.fact_cache.clear()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """全てのキャッシュの統計情報を取得"""
        return {
            "reasoning_cache": self.reasoning_cache.get_stats(),
            "embedding_cache": self.embedding_cache.get_stats(),
            "pattern_cache": self.pattern_cache.get_stats(),
            "fact_cache": self.fact_cache.get_stats()
        }
    
    def cached_function(self, cache_type: str, ttl: Optional[int] = None):
        """関数の結果をキャッシュするデコレータ"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # キャッシュキーを生成
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # キャッシュから取得を試行
                if cache_type == "reasoning":
                    result = self.get_reasoning_result(key)
                elif cache_type == "embedding":
                    result = self.get_embedding(key)
                elif cache_type == "pattern":
                    result = self.get_pattern_result(key, str(args))
                elif cache_type == "fact":
                    result = self.get_fact_search_result(key)
                else:
                    result = None
                
                if result is not None:
                    return result
                
                # キャッシュにない場合は関数を実行
                result = func(*args, **kwargs)
                
                # 結果をキャッシュに保存
                if cache_type == "reasoning":
                    self.cache_reasoning_result(key, result)
                elif cache_type == "embedding":
                    self.cache_embedding(key, result)
                elif cache_type == "pattern":
                    self.cache_pattern_result(key, str(args), result)
                elif cache_type == "fact":
                    self.cache_fact_search_result(key, result)
                
                return result
            return wrapper
        return decorator

# グローバルキャッシュマネージャーインスタンス
cache_manager = CacheManager()
