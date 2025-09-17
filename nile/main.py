"""
Enhanced NILE Main Application

改善されたNILEメインアプリケーション
"""

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nile.core.reasoning.symbolic_engine import SymbolicEngine
from nile.core.storage.knowledge_base import KnowledgeBase
from nile.core.error_handler import error_handler, error_handler_decorator
from nile.config.settings import config_manager

logger = logging.getLogger(__name__)

class NILEApplication:
    """NILEアプリケーションクラス"""
    
    def __init__(self):
        self.config = config_manager.get_config()
        self.knowledge_base = None
        self.symbolic_engine = None
        self._initialize()
    
    def _initialize(self):
        """アプリケーションを初期化"""
        try:
            logger.info("NILEアプリケーションを初期化しています...")
            
            # 知識ベースの初期化
            knowledge_file = os.path.join(project_root, "knowledge.json")
            self.knowledge_base = KnowledgeBase(knowledge_file)
            
            # シンボリックエンジンの初期化
            self.symbolic_engine = SymbolicEngine(knowledge_file)
            
            logger.info("NILEアプリケーションの初期化が完了しました")
            
        except Exception as e:
            error_message = error_handler.handle_error(e, {"context": "initialization"})
            logger.error(f"初期化中にエラーが発生しました: {error_message}")
            raise
    
    @error_handler_decorator
    def process_input(self, user_input: str) -> str:
        """ユーザー入力を処理"""
        try:
            if not user_input or not user_input.strip():
                return "入力が空です。質問や事実を入力してください。"
            
            # 入力の前処理
            processed_input = user_input.strip()
            
            # シンボリックエンジンで処理（タイムアウト付き）
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            response = None
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="NILE-Input") as ex:
                fut = ex.submit(self.symbolic_engine.process_input, processed_input)
                try:
                    # 既定タイムアウト（秒）
                    timeout_sec = int(os.getenv("NILE_RESPONSE_TIMEOUT", "15"))
                    response = fut.result(timeout=timeout_sec)
                except FutureTimeout:
                    return "処理がタイムアウトしました。環境変数 NILE_FAST_START=1 を設定して再実行してください。"
            
            return response
            
        except Exception as e:
            error_message = error_handler.handle_error(e, {
                "context": "input_processing",
                "input": user_input
            })
            return error_message
    
    def get_statistics(self) -> dict:
        """アプリケーションの統計情報を取得"""
        try:
            stats = {
                "knowledge_base": self.knowledge_base.get_statistics(),
                "symbolic_engine": self.symbolic_engine.get_statistics(),
                "config": {
                    "model": self.config.model.name,
                    "cache_enabled": self.config.cache.enabled,
                    "log_level": self.config.logging.level.value
                }
            }
            return stats
        except Exception as e:
            error_message = error_handler.handle_error(e, {"context": "statistics"})
            logger.error(f"統計情報取得中にエラー: {error_message}")
            return {"error": error_message}
    
    def show_help(self) -> str:
        """ヘルプメッセージを表示"""
        help_text = """
NILE (NeuroSymbolic Interactive Language Engine) ヘルプ

基本的な使用方法:
1. 事実の登録: "猫は動物です"
2. 質問: "猫は動物ですか？"
3. 知識の確認: "一覧" または "list"

コマンド:
- "help" または "ヘルプ": このヘルプを表示
- "stats" または "統計": 統計情報を表示
- "clear" または "クリア": 画面をクリア
- "exit" または "quit" または "終了": アプリケーションを終了

例:
- 事実登録: "犬は動物です", "鳥は空を飛びます"
- 質問: "犬は動物ですか？", "鳥は何をしますか？"
- 関係性: "猫と犬の関係は？"

注意:
- 日本語での入力を推奨します
- 複雑な文は短く分割してください
- エラーが発生した場合は、別の表現で試してください
        """
        return help_text.strip()
    
    def clear_screen(self):
        """画面をクリア"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run_interactive(self):
        """インタラクティブモードで実行"""
        print("NILEシステムが起動しました。")
        print("終了するには 'exit' または 'quit' と入力してください。")
        print("ヘルプを見るには 'help' と入力してください。")
        
        while True:
            try:
                user_input = input("\n質問を入力してください: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '終了']:
                    print("システムを終了します。")
                    break
                
                if user_input.lower() in ['help', 'ヘルプ']:
                    print(self.show_help())
                    continue
                
                if user_input.lower() in ['stats', '統計']:
                    stats = self.get_statistics()
                    print(f"統計情報: {stats}")
                    continue
                
                if user_input.lower() in ['clear', 'クリア']:
                    self.clear_screen()
                    continue
                
                if not user_input:
                    continue
                
                # 入力の処理
                response = self.process_input(user_input)
                print(f"\n回答: {response}")
                
            except KeyboardInterrupt:
                print("\nシステムを終了します。")
                break
            except Exception as e:
                error_message = error_handler.handle_error(e, {"context": "interactive_mode"})
                print(f"\nエラーが発生しました: {error_message}")

def main():
    """メイン関数"""
    try:
        # アプリケーションの初期化
        app = NILEApplication()
        
        # インタラクティブモードで実行
        app.run_interactive()
        
    except Exception as e:
        error_message = error_handler.handle_error(e, {"context": "main"})
        print(f"アプリケーションの起動中にエラーが発生しました: {error_message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
