"""
Custom Exceptions

カスタム例外クラスを定義
"""

from typing import Optional, Dict, Any

class NILEException(Exception):
    """NILEの基底例外クラス"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class ModelError(NILEException):
    """モデル関連のエラー"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)
        self.model_name = model_name

class KnowledgeBaseError(NILEException):
    """知識ベース関連のエラー"""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "KNOWLEDGE_BASE_ERROR", details)
        self.operation = operation

class ReasoningError(NILEException):
    """推論関連のエラー"""
    
    def __init__(self, message: str, inference_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "REASONING_ERROR", details)
        self.inference_type = inference_type

class ProcessingError(NILEException):
    """処理関連のエラー"""
    
    def __init__(self, message: str, input_text: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROCESSING_ERROR", details)
        self.input_text = input_text

class ConfigurationError(NILEException):
    """設定関連のエラー"""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key

class CacheError(NILEException):
    """キャッシュ関連のエラー"""
    
    def __init__(self, message: str, cache_key: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.cache_key = cache_key

class ValidationError(NILEException):
    """バリデーション関連のエラー"""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field

class APIError(NILEException):
    """API関連のエラー"""
    
    def __init__(self, message: str, status_code: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)
        self.status_code = status_code

class DatabaseError(NILEException):
    """データベース関連のエラー"""
    
    def __init__(self, message: str, query: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)
        self.query = query
