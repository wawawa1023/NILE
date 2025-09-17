"""
Configuration Settings

設定管理機能を提供
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelType(Enum):
    """モデルタイプ"""
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"

@dataclass
class ModelConfig:
    """モデル設定"""
    name: str = "cl-tohoku/bert-base-japanese-v3"
    type: ModelType = ModelType.BERT
    max_length: int = 512
    batch_size: int = 16
    use_gpu: bool = True
    cache_dir: str = "./cache"

@dataclass
class DatabaseConfig:
    """データベース設定"""
    type: str = "sqlite"
    path: str = "nile.db"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "nile"

@dataclass
class CacheConfig:
    """キャッシュ設定"""
    enabled: bool = True
    max_size: int = 1000
    ttl: int = 3600  # 秒
    backend: str = "memory"  # memory, redis
    redis_url: str = "redis://localhost:6379"

@dataclass
class LoggingConfig:
    """ログ設定"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "nile.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class APIConfig:
    """API設定"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    rate_limit: int = 100  # requests per minute

@dataclass
class UIConfig:
    """UI設定"""
    theme: str = "light"
    language: str = "ja"
    auto_save: bool = True
    show_confidence: bool = True
    max_history: int = 100

@dataclass
class NILEConfig:
    """NILE全体の設定"""
    model: ModelConfig = None
    database: DatabaseConfig = None
    cache: CacheConfig = None
    logging: LoggingConfig = None
    api: APIConfig = None
    ui: UIConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.ui is None:
            self.ui = UIConfig()

class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> NILEConfig:
        """設定を読み込み"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return self._dict_to_config(config_data)
            else:
                # デフォルト設定を作成
                config = NILEConfig()
                self.save_config(config)
                return config
        except Exception as e:
            logger.error(f"設定の読み込み中にエラーが発生しました: {e}")
            return NILEConfig()
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> NILEConfig:
        """辞書を設定オブジェクトに変換"""
        try:
            # モデル設定
            model_data = config_data.get("model", {})
            model_config = ModelConfig(
                name=model_data.get("name", "cl-tohoku/bert-base-japanese-v3"),
                type=ModelType(model_data.get("type", "bert")),
                max_length=model_data.get("max_length", 512),
                batch_size=model_data.get("batch_size", 16),
                use_gpu=model_data.get("use_gpu", True),
                cache_dir=model_data.get("cache_dir", "./cache")
            )
            
            # データベース設定
            db_data = config_data.get("database", {})
            db_config = DatabaseConfig(
                type=db_data.get("type", "sqlite"),
                path=db_data.get("path", "nile.db"),
                host=db_data.get("host", "localhost"),
                port=db_data.get("port", 5432),
                username=db_data.get("username", ""),
                password=db_data.get("password", ""),
                database=db_data.get("database", "nile")
            )
            
            # キャッシュ設定
            cache_data = config_data.get("cache", {})
            cache_config = CacheConfig(
                enabled=cache_data.get("enabled", True),
                max_size=cache_data.get("max_size", 1000),
                ttl=cache_data.get("ttl", 3600),
                backend=cache_data.get("backend", "memory"),
                redis_url=cache_data.get("redis_url", "redis://localhost:6379")
            )
            
            # ログ設定
            log_data = config_data.get("logging", {})
            log_config = LoggingConfig(
                level=LogLevel(log_data.get("level", "info")),
                format=log_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file_path=log_data.get("file_path", "nile.log"),
                max_file_size=log_data.get("max_file_size", 10 * 1024 * 1024),
                backup_count=log_data.get("backup_count", 5)
            )
            
            # API設定
            api_data = config_data.get("api", {})
            api_config = APIConfig(
                host=api_data.get("host", "localhost"),
                port=api_data.get("port", 8000),
                debug=api_data.get("debug", False),
                cors_enabled=api_data.get("cors_enabled", True),
                rate_limit=api_data.get("rate_limit", 100)
            )
            
            # UI設定
            ui_data = config_data.get("ui", {})
            ui_config = UIConfig(
                theme=ui_data.get("theme", "light"),
                language=ui_data.get("language", "ja"),
                auto_save=ui_data.get("auto_save", True),
                show_confidence=ui_data.get("show_confidence", True),
                max_history=ui_data.get("max_history", 100)
            )
            
            return NILEConfig(
                model=model_config,
                database=db_config,
                cache=cache_config,
                logging=log_config,
                api=api_config,
                ui=ui_config
            )
            
        except Exception as e:
            logger.error(f"設定の変換中にエラーが発生しました: {e}")
            return NILEConfig()
    
    def _config_to_dict(self, config: NILEConfig) -> Dict[str, Any]:
        """設定オブジェクトを辞書に変換"""
        return {
            "model": asdict(config.model),
            "database": asdict(config.database),
            "cache": asdict(config.cache),
            "logging": asdict(config.logging),
            "api": asdict(config.api),
            "ui": asdict(config.ui)
        }
    
    def save_config(self, config: Optional[NILEConfig] = None) -> bool:
        """設定を保存"""
        try:
            if config is None:
                config = self.config
            
            config_dict = self._config_to_dict(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            self.config = config
            logger.info(f"設定を保存しました: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"設定の保存中にエラーが発生しました: {e}")
            return False
    
    def _setup_logging(self):
        """ログ設定を適用"""
        try:
            log_config = self.config.logging
            
            # ログレベルを設定
            level = getattr(logging, log_config.level.value.upper())
            
            # ログフォーマットを設定
            formatter = logging.Formatter(log_config.format)
            
            # ファイルハンドラーを設定
            file_handler = logging.FileHandler(log_config.file_path, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            # コンソールハンドラーを設定
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            
            # ルートロガーを設定
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            logger.info("ログ設定を適用しました")
            
        except Exception as e:
            print(f"ログ設定の適用中にエラーが発生しました: {e}")
    
    def get_config(self) -> NILEConfig:
        """現在の設定を取得"""
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """設定を更新"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"不明な設定キー: {key}")
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"設定の更新中にエラーが発生しました: {e}")
            return False
    
    def reset_to_default(self) -> bool:
        """設定をデフォルトにリセット"""
        try:
            self.config = NILEConfig()
            return self.save_config()
        except Exception as e:
            logger.error(f"設定のリセット中にエラーが発生しました: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []
        
        try:
            # モデル設定の検証
            if not self.config.model.name:
                errors.append("モデル名が設定されていません")
            
            if self.config.model.max_length <= 0:
                errors.append("最大長は正の値である必要があります")
            
            if self.config.model.batch_size <= 0:
                errors.append("バッチサイズは正の値である必要があります")
            
            # データベース設定の検証
            if not self.config.database.path:
                errors.append("データベースパスが設定されていません")
            
            # キャッシュ設定の検証
            if self.config.cache.max_size <= 0:
                errors.append("キャッシュサイズは正の値である必要があります")
            
            if self.config.cache.ttl <= 0:
                errors.append("TTLは正の値である必要があります")
            
            # API設定の検証
            if self.config.api.port <= 0 or self.config.api.port > 65535:
                errors.append("ポート番号は1-65535の範囲である必要があります")
            
            if self.config.api.rate_limit <= 0:
                errors.append("レート制限は正の値である必要があります")
            
        except Exception as e:
            errors.append(f"設定検証中にエラーが発生しました: {e}")
        
        return errors
    
    def get_environment_config(self) -> Dict[str, str]:
        """環境変数から設定を取得"""
        env_config = {}
        
        # 環境変数のマッピング
        env_mappings = {
            "NILE_MODEL_NAME": "model.name",
            "NILE_MODEL_TYPE": "model.type",
            "NILE_DB_PATH": "database.path",
            "NILE_CACHE_ENABLED": "cache.enabled",
            "NILE_LOG_LEVEL": "logging.level",
            "NILE_API_HOST": "api.host",
            "NILE_API_PORT": "api.port"
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                env_config[config_path] = value
        
        return env_config
    
    def apply_environment_config(self) -> bool:
        """環境変数の設定を適用"""
        try:
            env_config = self.get_environment_config()
            
            for config_path, value in env_config.items():
                # 設定パスを分割
                parts = config_path.split('.')
                if len(parts) == 2:
                    section, key = parts
                    
                    if hasattr(self.config, section):
                        section_obj = getattr(self.config, section)
                        if hasattr(section_obj, key):
                            # 型変換
                            if key in ["max_length", "batch_size", "port", "max_size", "ttl", "rate_limit"]:
                                value = int(value)
                            elif key in ["use_gpu", "enabled", "debug", "cors_enabled", "auto_save", "show_confidence"]:
                                value = value.lower() in ["true", "1", "yes", "on"]
                            
                            setattr(section_obj, key, value)
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"環境変数設定の適用中にエラーが発生しました: {e}")
            return False

# グローバル設定マネージャーインスタンス
config_manager = ConfigManager()
