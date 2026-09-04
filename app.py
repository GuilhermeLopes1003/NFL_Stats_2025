import streamlit as st
import nflreadpy as nfl
import polars as pl

# 1. Configuração da página do navegador
st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")

st.title("🏈 Dashboard de Performance da NFL")

# 2. Carregamento de dados com Cache (para resposta instantânea)
@st.cache_data
def carregar_dados():
    pbp = nfl.load_pbp([2025])
    rosters = nfl.load_rosters([2025])
    
    # Lista de IDs de RBs oficiais
    ids_rbs = (
        rosters
        .filter(pl.col("position") == "RB")
        .select("gsis_id")
        .to_series()
        .to_list()
    )
    return pbp, ids_rbs

pbp, ids_rbs = carregar_dados()

# 3. Criação das Abas do App
aba_qbs, aba_rbs = st.tabs(["🎯 Quarterbacks (Passo)", "🏃 Running Backs (Corrida)"])

# --- ABA 1: QUARTERBACKS ---
with aba_qbs:
    st.header("Ranking de Quarterbacks")
    
    # Barra interativa para controlar o limite de passes
    min_passes = st.slider(
        "Mínimo de passes tentados na temporada:", 
        min_value=50, 
        max_value=500, 
        value=200, 
        step=25
    )
    
    # Processamento Polars para QBs
    ranking_qbs = (
        pbp
        .filter(
            (pl.col("play_type") == "pass") & 
            (pl.col("passer_player_name").is_not_null())
        )
        .group_by(["passer_player_name", "posteam"])
        .agg([
            pl.len().alias("total_passes"),
            pl.col("epa").gt(0).mean().alias("taxa_sucesso"),
            pl.col("epa").mean().alias("epa_medio"),
            pl.col("pass_touchdown").sum().alias("total_tds"),
            pl.col("yards_gained").sum().alias("total_jardas")
        ])
        .filter(pl.col("total_passes") >= min_passes)
        .sort("epa_medio", descending=True)
    )
    
    st.dataframe(ranking_qbs, use_container_width=True)

# --- ABA 2: RUNNING BACKS ---
with aba_rbs:
    st.header("Ranking de Running Backs")
    
    # Barra interativa para controlar o limite de corridas
    min_corridas = st.slider(
        "Mínimo de corridas na temporada:", 
        min_value=20, 
        max_value=250, 
        value=100, 
        step=10
    )
    
    # Processamento Polars para RBs
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
            pl.col("epa").gt(0).mean().alias("taxa_sucesso"),
            pl.col("epa").mean().alias("epa_medio"),
            pl.col("rush_touchdown").sum().alias("total_tds"),
            pl.col("yards_gained").sum().alias("total_jardas")
        ])
        .filter(pl.col("total_corridas") >= min_corridas)
        .sort("taxa_sucesso", descending=True)
    )
    
    st.dataframe(ranking_rbs, use_container_width=True)