import streamlit as st
import nflreadpy as nfl
import polars as pl

# 1. Configuração da página do navegador
st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")

st.title("🏈 Dashboard de Performance da NFL")

# 2. Carregamento de dados com Cache
@st.cache_data
def carregar_dados():
    pbp = nfl.load_pbp([2025])
    rosters = nfl.load_rosters([2025])
    
    ids_rbs = (
        rosters
        .filter(pl.col("position") == "RB")
        .select("gsis_id")
        .to_series()
        .to_list()
    )
    
    ids_wrs = (
        rosters
        .filter(pl.col("position") == "WR")
        .select("gsis_id")
        .to_series()
        .to_list()
    )
    
    return pbp, ids_rbs, ids_wrs

pbp, ids_rbs, ids_wrs = carregar_dados()

# 3. Criação das Abas do App
aba_qbs, aba_rbs, aba_wrs = st.tabs([
    "🎯 Quarterbacks (Passe)", 
    "🏃 Running Backs (Corrida)", 
    "🙌 Wide Receivers (Recepção)"
])

# --- ABA 1: QUARTERBACKS ---
with aba_qbs:
    st.header("Ranking de Quarterbacks (Por Eficiência Aérea EPA/Jogada)")
    
    min_passes = st.slider(
        "Mínimo de passes tentados na temporada:", 
        min_value=50, 
        max_value=500, 
        value=200, 
        step=25
    )
    
    ranking_qbs = (
        pbp
        .filter(
            (pl.col("play_type") == "pass") & 
            (pl.col("passer_player_name").is_not_null())
        )
        .group_by(["passer_player_name", "posteam"])
        .agg([
            pl.len().alias("total_passes"),
            # Multiplicamos por 100 para transformar decimal em porcentagem
            (pl.col("epa").gt(0).mean() * 100).alias("taxa_sucesso"),
            pl.col("epa").mean().alias("epa_medio"),
            pl.col("pass_touchdown").sum().cast(pl.UInt32).alias("total_tds"),
            pl.col("yards_gained").sum().cast(pl.Int64).alias("total_jardas")
        ])
        .filter(pl.col("total_passes") >= min_passes)
        .sort(["epa_medio", "taxa_sucesso"], descending=[True, True])
    )
    
    st.dataframe(
        ranking_qbs,
        column_config={
            "passer_player_name": "Jogador",
            "posteam": "Time",
            "total_passes": "Passes Tentados",
            "taxa_sucesso": st.column_config.NumberColumn("Taxa de Sucesso", format="%.2f%%"),
            "epa_medio": st.column_config.NumberColumn("EPA/Jogada", format="%.3f"),
            "total_tds": "TDs",
            "total_jardas": "Jardas Totais",
        },
        use_container_width=True,
        hide_index=True
    )

# --- ABA 2: RUNNING BACKS ---
with aba_rbs:
    st.header("Ranking de Running Backs (Por Eficiência - Taxa de sucesso")
    
    min_corridas = st.slider(
        "Mínimo de corridas na temporada:", 
        min_value=20, 
        max_value=250, 
        value=100, 
        step=10
    )
    
    ranking_rbs = (
        pbp
        .filter(
            (pl.col("play_type") == "run") & 
            (pl.col("rusher_player_name").is_not_null()) &
            (pl.col("rusher_player_id").is_in(ids_rbs))
        )
        .group_by(["rusher_player_name", "posteam"])
        .agg([
            pl.len().alias("total_corridas"),
            # Multiplicamos por 100 para transformar decimal em porcentagem
            (pl.col("epa").gt(0).mean() * 100).alias("taxa_sucesso"),
            pl.col("epa").mean().alias("epa_medio"),
            pl.col("rush_touchdown").sum().cast(pl.UInt32).alias("total_tds"),
            pl.col("yards_gained").sum().cast(pl.Int64).alias("total_jardas")
        ])
        .filter(pl.col("total_corridas") >= min_corridas)
        .sort(["taxa_sucesso", "total_jardas"], descending=[True, True])
    )
    
    st.dataframe(
        ranking_rbs,
        column_config={
            "rusher_player_name": "Jogador",
            "posteam": "Time",
            "total_corridas": "Corridas",
            "taxa_sucesso": st.column_config.NumberColumn("Taxa de Sucesso", format="%.2f%%"),
            "epa_medio": st.column_config.NumberColumn("EPA/Jogada", format="%.3f"),
            "total_tds": "TDs",
            "total_jardas": "Jardas Totais",
        },
        use_container_width=True,
        hide_index=True
    )

# --- ABA 3: WIDE RECEIVERS ---
with aba_wrs:
    st.header("Ranking de Wide Receivers (Por Eficiência - Jardas por Alvo")
    
    min_alvos = st.slider(
        "Mínimo de alvos (targets) na temporada:", 
        min_value=20, 
        max_value=150, 
        value=100, 
        step=5
    )
    
    ranking_wrs = (
        pbp
        .filter(
            (pl.col("play_type") == "pass") & 
            (pl.col("receiver_player_name").is_not_null()) &
            (pl.col("receiver_player_id").is_in(ids_wrs))
        )
        .group_by(["receiver_player_name", "posteam"])
        .agg([
            pl.len().alias("alvos"),
            pl.col("complete_pass").sum().cast(pl.UInt32).alias("recepcoes"),
            # Multiplicamos por 100 para transformar decimal em porcentagem
            (pl.col("complete_pass").mean() * 100).alias("taxa_captura"),
            pl.col("yards_gained").filter(pl.col("complete_pass") == 1).sum().cast(pl.Int64).alias("jardas"),
            pl.col("pass_touchdown").filter(pl.col("complete_pass") == 1).sum().cast(pl.UInt32).alias("tds"),
            pl.col("epa").mean().alias("epa_medio")
        ])
        .filter(pl.col("alvos") >= min_alvos)
        .with_columns(
            (pl.col("jardas") / pl.col("alvos")).alias("jardas_por_alvo")
        )
        .sort(["jardas_por_alvo", "jardas"], descending=[True, True])
    )
    
    st.dataframe(
        ranking_wrs,
        column_config={
            "receiver_player_name": "Jogador",
            "posteam": "Time",
            "alvos": "Alvos",
            "recepcoes": "Recepções",
            "taxa_captura": st.column_config.NumberColumn("Catch %", format="%.2f%%"),
            "jardas": "Jardas Totais",
            "tds": "TDs",
            "epa_medio": st.column_config.NumberColumn("EPA/Alvo", format="%.3f"),
            "jardas_por_alvo": st.column_config.NumberColumn("YDS/Target", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True
    )