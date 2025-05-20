import os
from dotenv import load_dotenv
from symbolic_engine import SymbolicEngine

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # 知識ベースのファイルパスを絶対パスで指定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_file = os.path.join(current_dir, "knowledge.json")
    
    # シンボリックエンジンの初期化
    engine = SymbolicEngine(knowledge_file)
    
    print("NILEシステムが起動しました。")
    print("終了するには 'exit' または 'quit' と入力してください。")
    
    while True:
        try:
            user_input = input("\n質問を入力してください: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("システムを終了します。")
                break
                
            if not user_input:
                continue
                
            # 質問の処理
            response = engine.process_input(user_input)
            print("\n回答:", response)
            
        except KeyboardInterrupt:
            print("\nシステムを終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
