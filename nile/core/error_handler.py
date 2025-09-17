"""
Error Handler

エラーハンドリング機能を提供
"""

import logging
import traceback
import functools
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import json

from .exceptions import NILEException

logger = logging.getLogger(__name__)

class ErrorHandler:
    """エラーハンドリングクラス"""
    
    def __init__(self):
        self.error_log = []
        self.max_log_size = 1000
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """エラーを処理してユーザーフレンドリーなメッセージを返す"""
        try:
            # エラーログに記録
            self._log_error(error, context)
            
            # エラータイプに応じた処理
            if isinstance(error, NILEException):
                return self._handle_nile_error(error)
            elif isinstance(error, ValueError):
                return self._handle_value_error(error)
            elif isinstance(error, TypeError):
                return self._handle_type_error(error)
            elif isinstance(error, KeyError):
                return self._handle_key_error(error)
            elif isinstance(error, FileNotFoundError):
                return self._handle_file_not_found_error(error)
            elif isinstance(error, PermissionError):
                return self._handle_permission_error(error)
            elif isinstance(error, MemoryError):
                return self._handle_memory_error(error)
            elif isinstance(error, TimeoutError):
                return self._handle_timeout_error(error)
            else:
                return self._handle_generic_error(error)
                
        except Exception as e:
            logger.critical(f"エラーハンドリング中にエラーが発生しました: {e}")
            return "申し訳ありません。システムエラーが発生しました。"
    
    def _handle_nile_error(self, error: NILEException) -> str:
        """NILE例外を処理"""
        if error.error_code == "MODEL_ERROR":
            return f"モデルの処理中にエラーが発生しました: {error.message}"
        elif error.error_code == "KNOWLEDGE_BASE_ERROR":
            return f"知識ベースの操作中にエラーが発生しました: {error.message}"
        elif error.error_code == "REASONING_ERROR":
            return f"推論処理中にエラーが発生しました: {error.message}"
        elif error.error_code == "PROCESSING_ERROR":
            return f"テキスト処理中にエラーが発生しました: {error.message}"
        elif error.error_code == "CONFIGURATION_ERROR":
            return f"設定エラーが発生しました: {error.message}"
        elif error.error_code == "CACHE_ERROR":
            return f"キャッシュ処理中にエラーが発生しました: {error.message}"
        elif error.error_code == "VALIDATION_ERROR":
            return f"入力値の検証中にエラーが発生しました: {error.message}"
        elif error.error_code == "API_ERROR":
            return f"API処理中にエラーが発生しました: {error.message}"
        elif error.error_code == "DATABASE_ERROR":
            return f"データベース処理中にエラーが発生しました: {error.message}"
        else:
            return f"システムエラーが発生しました: {error.message}"
    
    def _handle_value_error(self, error: ValueError) -> str:
        """ValueErrorを処理"""
        return f"入力値に問題があります: {str(error)}"
    
    def _handle_type_error(self, error: TypeError) -> str:
        """TypeErrorを処理"""
        return f"データ型に問題があります: {str(error)}"
    
    def _handle_key_error(self, error: KeyError) -> str:
        """KeyErrorを処理"""
        return f"必要な情報が見つかりません: {str(error)}"
    
    def _handle_file_not_found_error(self, error: FileNotFoundError) -> str:
        """FileNotFoundErrorを処理"""
        return f"ファイルが見つかりません: {str(error)}"
    
    def _handle_permission_error(self, error: PermissionError) -> str:
        """PermissionErrorを処理"""
        return f"ファイルアクセス権限がありません: {str(error)}"
    
    def _handle_memory_error(self, error: MemoryError) -> str:
        """MemoryErrorを処理"""
        return "メモリが不足しています。処理を簡素化してください。"
    
    def _handle_timeout_error(self, error: TimeoutError) -> str:
        """TimeoutErrorを処理"""
        return "処理がタイムアウトしました。再度お試しください。"
    
    def _handle_generic_error(self, error: Exception) -> str:
        """一般的なエラーを処理"""
        return f"予期しないエラーが発生しました: {str(error)}"
    
    def _log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """エラーをログに記録"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        # エラーログに追加
        self.error_log.append(error_info)
        
        # ログサイズを制限
        if len(self.error_log) > self.max_log_size:
            self.error_log = self.error_log[-self.max_log_size:]
        
        # ログレベルに応じて記録
        if isinstance(error, NILEException):
            logger.error(f"NILE Error: {error.error_code} - {error.message}")
        else:
            logger.error(f"Unexpected Error: {error}", exc_info=True)
    
    def get_error_log(self, limit: int = 100) -> list:
        """エラーログを取得"""
        return self.error_log[-limit:]
    
    def clear_error_log(self):
        """エラーログをクリア"""
        self.error_log.clear()
    
    def export_error_log(self, file_path: str) -> bool:
        """エラーログをファイルにエクスポート"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.error_log, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"エラーログのエクスポート中にエラー: {e}")
            return False

def error_handler_decorator(func: Callable) -> Callable:
    """エラーハンドリングデコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handler = ErrorHandler()
            error_message = handler.handle_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            logger.error(f"Function {func.__name__} failed: {error_message}")
            return error_message
    return wrapper

def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[str]]:
    """安全に実行して結果とエラーメッセージを返す"""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        handler = ErrorHandler()
        error_message = handler.handle_error(e, {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        return None, error_message

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """エラー時のリトライデコレータ"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            # 最後のエラーを処理
            handler = ErrorHandler()
            return handler.handle_error(last_error, {
                "function": func.__name__,
                "max_retries": max_retries
            })
        
        return wrapper
    return decorator

def validate_input(func: Callable) -> Callable:
    """入力検証デコレータ"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 基本的な入力検証
            for arg in args:
                if arg is None:
                    raise ValueError("引数にNoneが含まれています")
                if isinstance(arg, str) and not arg.strip():
                    raise ValueError("空の文字列が含まれています")
            
            return func(*args, **kwargs)
        except Exception as e:
            handler = ErrorHandler()
            return handler.handle_error(e, {
                "function": func.__name__,
                "validation_error": True
            })
    
    return wrapper

# グローバルエラーハンドラーインスタンス
error_handler = ErrorHandler()
