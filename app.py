import streamlit as st
import nflreadpy as nfl
import polars as pl
import plotly.express as px

# Configuração da página Web
st.set_page_config(page_title="NFL Analytics - Dashboard QBs", layout="wide")

st.title("🏈 Dashboard de Eficiência dos Quarterbacks")
st.markdown("Análise comparativa de **Eficiência (EPA/play)** e **Precisão (CPOE)** usando dados do `nflreadpy`.")

# --- BARRA LATERAL (FILTROS) ---
st.sidebar.header("Filtros da Análise")

# Filtro de Ano
ano = st.sidebar.selectbox("Selecione a Temporada", [2025, 2024, 2023], index=0)

# Filtro de Volume Mínimo de Passes
min_passes = st.sidebar.slider("Mínimo de passes tentados", min_value=20, max_value=400, value=100, step=10)

# --- CARREGAMENTO DE DADOS (COM CACHE) ---
@st.cache_data
def carregar_dados_pbp(temporada):
    return nfl.load_pbp([temporada])

with st.spinner("Carregando dados da NFL..."):
    pbp = carregar_dados_pbp(ano)

# --- PROCESSAMENTO DOS DADOS COM POLARS ---
ranking_qbs = (
    pbp
    .filter(
        (pl.col("play_type") == "pass") & 
        (pl.col("passer_player_name").is_not_null())
    )
    .group_by(["passer_player_name", "posteam"])
    .agg([
        pl.len().alias("total_passes"),
        pl.col("epa").mean().alias("epa_medio"),
        pl.col("cpoe").mean().alias("cpoe_medio"),
        pl.col("yards_gained").sum().alias("total_jardas")
    ])
    .filter(pl.col("total_passes") >= min_passes)
)

df_plot = ranking_qbs.to_pandas()

# --- LAYOUT DO DASHBOARD (2 COLUNAS) ---
col_grafico, col_tabela = st.columns([2, 1])

with col_grafico:
    st.subheader(f"Gráfico de Desempenho ({ano})")
    
    fig = px.scatter(
        df_plot,
        x="cpoe_medio",
        y="epa_medio",
        text="passer_player_name",
        size="total_passes",
        color="epa_medio",
        color_continuous_scale="RdYlGn",
        labels={
            "cpoe_medio": "Precisão (%) - CPOE Médio",
            "epa_medio": "Eficiência - EPA Médio por Jogada"
        }
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=df_plot["epa_medio"].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=df_plot["cpoe_medio"].mean(), line_dash="dash", line_color="gray")
    
    # Atualizado com o parâmetro 'width'
    st.plotly_chart(fig, width="stretch")

with col_tabela:
    st.subheader("Top QBs por EPA")
    tabela_exibicao = (
        ranking_qbs
        .sort("epa_medio", descending=True)
        .select(["passer_player_name", "posteam", "epa_medio", "total_passes"])
    )
    # Atualizado com o parâmetro 'width'
    st.dataframe(tabela_exibicao.to_pandas(), width="stretch")