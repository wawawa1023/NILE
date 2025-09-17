"""
Web Interface

StreamlitベースのWebインターフェース
"""

import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from typing import Dict, List, Any
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nile.core.reasoning.symbolic_engine import SymbolicEngine
from nile.core.storage.knowledge_base import KnowledgeBase
from nile.core.error_handler import error_handler

class NILEWebInterface:
    """NILE Webインターフェースクラス"""
    
    def __init__(self):
        self.engine = None
        self.knowledge_base = None
        self._initialize()
    
    def _initialize(self):
        """アプリケーションを初期化"""
        try:
            knowledge_file = project_root / "knowledge.json"
            self.knowledge_base = KnowledgeBase(str(knowledge_file))
            self.engine = SymbolicEngine(str(knowledge_file))
        except Exception as e:
            st.error(f"初期化エラー: {error_handler.handle_error(e)}")
    
    def run(self):
        """Webインターフェースを実行"""
        st.set_page_config(
            page_title="NILE - NeuroSymbolic Interactive Language Engine",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # メインタイトル
        st.title("🧠 NILE - NeuroSymbolic Interactive Language Engine")
        st.markdown("日本語自然言語理解とシンボリック推論を組み合わせた実験的プロトタイプ")
        
        # サイドバー
        self._render_sidebar()
        
        # メインコンテンツ
        tab1, tab2, tab3, tab4 = st.tabs(["💬 対話", "📊 統計", "🕸️ 知識グラフ", "⚙️ 設定"])
        
        with tab1:
            self._render_chat_interface()
        
        with tab2:
            self._render_statistics()
        
        with tab3:
            self._render_knowledge_graph()
        
        with tab4:
            self._render_settings()
    
    def _render_sidebar(self):
        """サイドバーをレンダリング"""
        st.sidebar.header("📋 ナビゲーション")
        
        # クイックアクション
        st.sidebar.subheader("🚀 クイックアクション")
        
        if st.sidebar.button("🗑️ 知識ベースをクリア"):
            if st.sidebar.button("⚠️ 本当にクリアしますか？", key="confirm_clear"):
                self._clear_knowledge_base()
                st.sidebar.success("知識ベースをクリアしました")
        
        if st.sidebar.button("📥 知識ベースをエクスポート"):
            self._export_knowledge_base()
        
        if st.sidebar.button("📤 知識ベースをインポート"):
            self._import_knowledge_base()
        
        # ヘルプ
        st.sidebar.subheader("❓ ヘルプ")
        if st.sidebar.button("📖 使い方を見る"):
            self._show_help()
    
    def _render_chat_interface(self):
        """チャットインターフェースをレンダリング"""
        st.header("💬 NILEとの対話")
        
        # チャット履歴の初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # チャット履歴の表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 入力フォーム
        if prompt := st.chat_input("質問や事実を入力してください..."):
            # ユーザーメッセージを追加
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # NILEの応答を生成
            with st.chat_message("assistant"):
                with st.spinner("考え中..."):
                    try:
                        response = self.engine.process_input(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = error_handler.handle_error(e)
                        st.error(f"エラー: {error_msg}")
                        st.session_state.messages.append({"role": "assistant", "content": f"エラー: {error_msg}"})
        
        # チャット履歴の管理
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 履歴をクリア"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("💾 履歴を保存"):
                self._save_chat_history()
    
    def _render_statistics(self):
        """統計情報をレンダリング"""
        st.header("📊 統計情報")
        
        try:
            # 知識ベースの統計
            kb_stats = self.knowledge_base.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("総事実数", kb_stats.get("total_facts", 0))
            
            with col2:
                sources = kb_stats.get("sources", {})
                st.metric("情報源数", len(sources))
            
            with col3:
                relations = kb_stats.get("relations", {})
                st.metric("関係タイプ数", len(relations))
            
            # 関係タイプの分布
            if relations:
                st.subheader("📈 関係タイプの分布")
                relation_df = pd.DataFrame(list(relations.items()), columns=["関係タイプ", "件数"])
                fig = px.pie(relation_df, values="件数", names="関係タイプ", title="関係タイプの分布")
                st.plotly_chart(fig, use_container_width=True)
            
            # 情報源の分布
            if sources:
                st.subheader("📊 情報源の分布")
                source_df = pd.DataFrame(list(sources.items()), columns=["情報源", "件数"])
                fig = px.bar(source_df, x="情報源", y="件数", title="情報源別の事実数")
                st.plotly_chart(fig, use_container_width=True)
            
            # エンジンの統計
            engine_stats = self.engine.get_statistics()
            if engine_stats:
                st.subheader("🔧 エンジン統計")
                st.json(engine_stats)
        
        except Exception as e:
            st.error(f"統計情報の取得中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _render_knowledge_graph(self):
        """知識グラフをレンダリング"""
        st.header("🕸️ 知識グラフ")
        
        try:
            # 知識グラフの取得
            graph = self.engine.knowledge_graph.graph
            
            if graph.number_of_nodes() == 0:
                st.info("知識グラフが空です。まず事実を追加してください。")
                return
            
            # グラフの可視化
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # ノードとエッジの情報を準備
            node_x = []
            node_y = []
            node_text = []
            
            for node in graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            edge_x = []
            edge_y = []
            
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # プロットの作成
            fig = go.Figure()
            
            # エッジの追加
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='関係'
            ))
            
            # ノードの追加
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                name='エンティティ'
            ))
            
            fig.update_layout(
                title="知識グラフ",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="ノードをクリックして詳細を表示",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # グラフの統計情報
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ノード数", graph.number_of_nodes())
            
            with col2:
                st.metric("エッジ数", graph.number_of_edges())
            
            with col3:
                density = nx.density(graph)
                st.metric("密度", f"{density:.3f}")
        
        except Exception as e:
            st.error(f"知識グラフの表示中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _render_settings(self):
        """設定画面をレンダリング"""
        st.header("⚙️ 設定")
        
        # モデル設定
        st.subheader("🤖 モデル設定")
        
        model_name = st.text_input("モデル名", value="cl-tohoku/bert-base-japanese-v3")
        max_length = st.slider("最大長", min_value=128, max_value=1024, value=512)
        batch_size = st.slider("バッチサイズ", min_value=1, max_value=32, value=16)
        use_gpu = st.checkbox("GPUを使用", value=True)
        
        if st.button("設定を保存"):
            st.success("設定を保存しました")
        
        # キャッシュ設定
        st.subheader("💾 キャッシュ設定")
        
        cache_enabled = st.checkbox("キャッシュを有効化", value=True)
        cache_size = st.slider("キャッシュサイズ", min_value=100, max_value=10000, value=1000)
        cache_ttl = st.slider("TTL (秒)", min_value=60, max_value=3600, value=3600)
        
        if st.button("キャッシュをクリア"):
            self.engine.cache_manager.clear_all()
            st.success("キャッシュをクリアしました")
        
        # ログ設定
        st.subheader("📝 ログ設定")
        
        log_level = st.selectbox("ログレベル", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_to_file = st.checkbox("ファイルにログを出力", value=True)
        
        if st.button("ログをダウンロード"):
            self._download_logs()
    
    def _clear_knowledge_base(self):
        """知識ベースをクリア"""
        try:
            # 知識ベースファイルを削除
            knowledge_file = project_root / "knowledge.json"
            if knowledge_file.exists():
                knowledge_file.unlink()
            
            # 新しい知識ベースを作成
            self.knowledge_base = KnowledgeBase(str(knowledge_file))
            self.engine = SymbolicEngine(str(knowledge_file))
            
        except Exception as e:
            st.error(f"知識ベースのクリア中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _export_knowledge_base(self):
        """知識ベースをエクスポート"""
        try:
            facts = self.knowledge_base.get_facts()
            json_data = json.dumps({"facts": facts}, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="📥 知識ベースをダウンロード",
                data=json_data,
                file_name="knowledge_base.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"エクスポート中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _import_knowledge_base(self):
        """知識ベースをインポート"""
        uploaded_file = st.file_uploader("JSONファイルを選択", type="json")
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                facts = data.get("facts", [])
                
                for fact in facts:
                    self.knowledge_base.add_fact(
                        fact["subject"],
                        fact["object"],
                        fact["relation"],
                        fact.get("confidence", 1.0),
                        fact.get("source", "import")
                    )
                
                st.success(f"{len(facts)}件の事実をインポートしました")
                st.rerun()
                
            except Exception as e:
                st.error(f"インポート中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _show_help(self):
        """ヘルプを表示"""
        help_text = """
        # NILE 使い方ガイド
        
        ## 基本的な使用方法
        
        ### 1. 事実の登録
        - "猫は動物です"
        - "犬は忠実です"
        - "鳥は空を飛びます"
        
        ### 2. 質問
        - "猫は動物ですか？"
        - "犬の特徴は何ですか？"
        - "鳥は何をしますか？"
        
        ### 3. 知識の確認
        - "一覧" または "list" で知識ベースの内容を表示
        
        ## 機能
        
        - **対話タブ**: NILEとの対話
        - **統計タブ**: 知識ベースの統計情報
        - **知識グラフタブ**: 知識の関係性を可視化
        - **設定タブ**: システム設定の変更
        
        ## 注意事項
        
        - 日本語での入力を推奨します
        - 複雑な文は短く分割してください
        - エラーが発生した場合は、別の表現で試してください
        """
        
        st.markdown(help_text)
    
    def _save_chat_history(self):
        """チャット履歴を保存"""
        try:
            chat_data = {
                "messages": st.session_state.messages,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            json_data = json.dumps(chat_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="💾 チャット履歴をダウンロード",
                data=json_data,
                file_name=f"chat_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"チャット履歴の保存中にエラーが発生しました: {error_handler.handle_error(e)}")
    
    def _download_logs(self):
        """ログをダウンロード"""
        try:
            # ログファイルの内容を取得
            log_file = project_root / "nile.log"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                st.download_button(
                    label="📥 ログをダウンロード",
                    data=log_content,
                    file_name="nile.log",
                    mime="text/plain"
                )
            else:
                st.info("ログファイルが見つかりません")
        except Exception as e:
            st.error(f"ログのダウンロード中にエラーが発生しました: {error_handler.handle_error(e)}")

def main():
    """メイン関数"""
    app = NILEWebInterface()
    app.run()

if __name__ == "__main__":
    main()
