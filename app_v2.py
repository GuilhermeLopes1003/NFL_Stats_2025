import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# Configuração da Página
st.set_page_config(
    page_title="NFL Play-Call Predictor v2", page_icon="🏈", layout="wide"
)


# 1. Carregar Modelo
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_pass_run.joblib")


# 2. Carregar e Calcular Estatísticas Reais dos Times
@st.cache_data
def carregar_stats_times():
    # Carrega dados mais recentes para calcular a média das equipes
    pbp = nfl.load_pbp([2024, 2025])
    pbp_valid = pbp.filter(
        (pl.col("play_type").is_in(["pass", "run"]))
        & (pl.col("posteam").is_not_null())
        & (pl.col("defteam").is_not_null())
        & (pl.col("epa").is_not_null())
    )

    # Taxa de Passe do Ataque
    offense = pbp_valid.group_by("posteam").agg(
        [
            (
                pl.col("play_type").filter(pl.col("play_type") == "pass").count()
                / pl.col("play_type").count()
            ).alias("offense_pass_ratio")
        ]
    )

    # Eficiência Defensiva (EPA)
    defense = pbp_valid.group_by("defteam").agg(
        [
            pl.col("epa")
            .filter(pl.col("play_type") == "pass")
            .mean()
            .alias("def_epa_against_pass"),
            pl.col("epa")
            .filter(pl.col("play_type") == "run")
            .mean()
            .alias("def_epa_against_run"),
        ]
    )

    df_offense = offense.to_pandas().set_index("posteam")
    df_defense = defense.to_pandas().set_index("defteam")

    return df_offense, df_defense


try:
    modelo = carregar_modelo()
    df_offense_stats, df_defense_stats = carregar_stats_times()
except Exception as e:
    st.error(f"Erro ao inicializar a aplicação: {e}")
    st.stop()

st.title("🏈 NFL Play-Call Predictor (v2.0)")
st.write(
    "Simulador tático baseado em **XGBoost** (72.6% de acurácia). "
    "Selecione os times para carregar as tendências reais automaticamente."
)

st.markdown("---")

# Layout de Colunas na Interface
col_sit, col_tact, col_teams = st.columns([1, 1, 1])

with col_sit:
    st.subheader("📍 Situação de Campo")
    down = st.selectbox("Descida (Down)", options=[1, 2, 3, 4], index=0)
    ydstogo = st.number_input(
        "Jardas para o First Down", min_value=1, max_value=99, value=10
    )
    yardline_100 = st.slider(
        "Distância para a End Zone (Yardline 100)", 1, 99, 75
    )
    tempo_minutos = st.number_input(
        "Minutos Restantes no Tempo", min_value=0, max_value=15, value=15
    )
    tempo_segundos = st.number_input(
        "Segundos Restantes no Minuto", min_value=0, max_value=59, value=0
    )
    half_seconds_remaining = (tempo_minutos * 60) + tempo_segundos
    score_differential = st.number_input(
        "Diferença no Placar (Ataque - Defesa)",
        min_value=-50,
        max_value=50,
        value=0,
    )

with col_tact:
    st.subheader("🛡️ Formação & Ritmo")
    formacao = st.radio(
        "Alinhamento do Quarterback",
        options=["Shotgun", "Under Center"],
        index=0,
    )
    shotgun = 1 if formacao == "Shotgun" else 0

    no_huddle_input = st.selectbox(
        "Ataque Rápido (No Huddle)?", options=["Não", "Sim"], index=0
    )
    no_huddle = 1 if no_huddle_input == "Sim" else 0

with col_teams:
    st.subheader("📊 Seleção dos Times")
    lista_times = sorted(df_offense_stats.index.unique().tolist())

    time_ataque = st.selectbox(
        "Time no Ataque (Offense)", options=lista_times, index=0
    )
    time_defesa = st.selectbox(
        "Time na Defesa (Defense)",
        options=lista_times,
        index=1 if len(lista_times) > 1 else 0,
    )

    # Extrai automaticamente os dados calculados do time escolhido
    offense_pass_ratio = float(
        df_offense_stats.loc[time_ataque, "offense_pass_ratio"]
    )
    def_epa_against_pass = float(
        df_defense_stats.loc[time_defesa, "def_epa_against_pass"]
    )
    def_epa_against_run = float(
        df_defense_stats.loc[time_defesa, "def_epa_against_run"]
    )

    st.info(
        f"**Métricas Carregadas:**\n"
        f"- Pass Ratio ({time_ataque}): `{offense_pass_ratio * 100:.1f}%`\n"
        f"- Def EPA/Passe ({time_defesa}): `{def_epa_against_pass:.3f}`\n"
        f"- Def EPA/Corrida ({time_defesa}): `{def_epa_against_run:.3f}`"
    )

st.markdown("---")

# Botão de Simulação
if st.button("🚀 Simular Chamada de Jogada", type="primary", use_container_width=True):
    novos_dados = pd.DataFrame(
        [
            {
                "down": down,
                "ydstogo": ydstogo,
                "yardline_100": yardline_100,
                "half_seconds_remaining": half_seconds_remaining,
                "score_differential": score_differential,
                "shotgun": shotgun,
                "no_huddle": no_huddle,
                "offense_pass_ratio": offense_pass_ratio,
                "def_epa_against_pass": def_epa_against_pass,
                "def_epa_against_run": def_epa_against_run,
            }
        ]
    )

    probabilidades = modelo.predict_proba(novos_dados)[0]
    # Conversão explícita para float puro para evitar o erro de float32
    prob_corrida = float(probabilidades[0] * 100)
    prob_passe = float(probabilidades[1] * 100)

    st.subheader("🎯 Resultado do Simulador")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Probabilidade de PASSE", value=f"{prob_passe:.1f}%")
        st.progress(prob_passe / 100.0)

    with res_col2:
        st.metric(label="Probabilidade de CORRIDA", value=f"{prob_corrida:.1f}%")
        st.progress(prob_corrida / 100.0)

    if prob_passe > prob_corrida:
        st.success(f"💡 **Predição Final: PASSE** (Confiança: {prob_passe:.1f}%)")
    else:
        st.info(f"💡 **Predição Final: CORRIDA** (Confiança: {prob_corrida:.1f}%)")