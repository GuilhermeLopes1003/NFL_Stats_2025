import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# Configuração da Página
st.set_page_config(
    page_title="NFL Analytics & Play-Call Predictor",
    page_icon="🏈",
    layout="wide",
)


# ==========================================
# 1. FUNÇÕES DE CARREGAMENTO DE DADOS E MODELO
# ==========================================
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_pass_run.joblib")


@st.cache_data
def carregar_dados_pbp():
    # Carrega PBP recente para rankings e estatísticas dos times
    pbp = nfl.load_pbp([2024, 2025])
    return pbp.filter(
        (pl.col("play_type").is_in(["pass", "run"]))
        & (pl.col("posteam").is_not_null())
        & (pl.col("defteam").is_not_null())
        & (pl.col("epa").is_not_null())
    )


@st.cache_data
def processar_stats_times(pbp_valid):
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

    return (
        offense.to_pandas().set_index("posteam"),
        defense.to_pandas().set_index("defteam"),
    )


# Inicialização dos Dados
try:
    modelo_ml = carregar_modelo()
    pbp_data = carregar_dados_pbp()
    df_offense_stats, df_defense_stats = processar_stats_times(pbp_data)
except Exception as e:
    st.error(f"Erro ao inicializar a aplicação: {e}")
    st.stop()


# ==========================================
# 2. CABEÇALHO E NAVEGAÇÃO POR ABAS
# ==========================================
st.title("🏈 NFL Analytics & Play-Call Predictor")
st.write(
    "Plataforma unificada para análise de desempenho de atletas e simulação tática preditiva via Machine Learning."
)

aba_simulador, aba_rankings = st.tabs(
    ["🎯 Simulador Preditivo (v2.0)", "📊 Rankings de Jogadores"]
)


# ==========================================
# ABA 1: SIMULADOR PREDITIVO V2
# ==========================================
with aba_simulador:
    st.header("🎯 Simulador de Chamada de Jogadas (XGBoost v2)")
    st.caption(
        "Acurácia de 72.61% no teste de 2025 com variáveis situacionais e formações pré-snap."
    )

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

    if st.button(
        "🚀 Simular Chamada de Jogada", type="primary", use_container_width=True
    ):
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

        probabilidades = modelo_ml.predict_proba(novos_dados)[0]
        prob_corrida = float(probabilidades[0] * 100)
        prob_passe = float(probabilidades[1] * 100)

        st.subheader("🎯 Resultado do Simulador")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric(
                label="Probabilidade de PASSE", value=f"{prob_passe:.1f}%"
            )
            st.progress(prob_passe / 100.0)

        with res_col2:
            st.metric(
                label="Probabilidade de CORRIDA", value=f"{prob_corrida:.1f}%"
            )
            st.progress(prob_corrida / 100.0)

        if prob_passe > prob_corrida:
            st.success(
                f"💡 **Predição Final: PASSE** (Confiança: {prob_passe:.1f}%)"
            )
        else:
            st.info(
                f"💡 **Predição Final: CORRIDA** (Confiança: {prob_corrida:.1f}%)"
            )


# ==========================================
# ABA 2: RANKINGS DE JOGADORES (REQUISITOS REAIS)
# ==========================================
with aba_rankings:
    st.header("📊 Rankings Dinâmicos de Jogadores")

    col_pos, col_filtro = st.columns([1, 1])

    with col_pos:
        posicao = st.selectbox(
            "Selecione a Posição para Análise:",
            options=[
                "Quarterbacks (QB)",
                "Running Backs (RB)",
                "Wide Receivers (WR)",
                "Tight Ends (TE)",
            ],
        )

    # 1. QUARTERBACKS: EPA/Jogada & Taxa de Sucesso (%)
    if "Quarterbacks" in posicao:
        with col_filtro:
            min_tentativas = st.slider(
                "Mínimo de Passes Tentados:",
                min_value=10,
                max_value=1200,
                value=250,
                step=10,
            )

        st.subheader(
            f"🏆 Ranking de QBs (Mínimo de {min_tentativas} tentativas)"
        )

        qb_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("passer_player_name").is_not_null())
            )
            .group_by("passer_player_name")
            .agg(
                [
                    pl.col("play_id").count().alias("tentativas"),
                    pl.col("epa").mean().alias("epa_por_jogada"),
                    (
                        pl.col("epa").filter(pl.col("epa") > 0).count()
                        / pl.col("play_id").count()
                    ).alias("taxa_sucesso"),
                ]
            )
            .filter(pl.col("tentativas") >= min_tentativas)
            .sort("epa_por_jogada", descending=True)
        )

        df_qb_display = qb_stats.to_pandas()
        df_qb_display["taxa_sucesso"] = df_qb_display["taxa_sucesso"] * 100
        st.dataframe(
            df_qb_display.style.format(
                {"epa_por_jogada": "{:.3f}", "taxa_sucesso": "{:.1f}%"}
            ),
            use_container_width=True,
        )

    # 2. RUNNING BACKS: Taxa de Sucesso (%) & Jardas Totais
    elif "Running Backs" in posicao:
        with col_filtro:
            min_carregadas = st.slider(
                "Mínimo de Corridas (Carregadas):",
                min_value=10,
                max_value=600,
                value=100,
                step=10,
            )

        st.subheader(
            f"🏆 Ranking de RBs (Mínimo de {min_carregadas} carregadas)"
        )

        rb_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "run")
                & (pl.col("rusher_player_name").is_not_null())
            )
            .group_by("rusher_player_name")
            .agg(
                [
                    pl.col("play_id").count().alias("carregadas"),
                    pl.col("yards_gained").sum().alias("jardas_totais"),
                    pl.col("yards_gained").mean().alias("media_jardas"),
                    (
                        pl.col("epa").filter(pl.col("epa") > 0).count()
                        / pl.col("play_id").count()
                    ).alias("taxa_sucesso"),
                    pl.col("epa").mean().alias("epa_por_corrida"),
                ]
            )
            .filter(pl.col("carregadas") >= min_carregadas)
            .sort("jardas_totais", descending=True)
        )

        df_rb_display = rb_stats.to_pandas()
        df_rb_display["taxa_sucesso"] = df_rb_display["taxa_sucesso"] * 100
        st.dataframe(
            df_rb_display.style.format(
                {
                    "media_jardas": "{:.2f}",
                    "taxa_sucesso": "{:.1f}%",
                    "epa_por_corrida": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    # 3. WIDE RECEIVERS: Jardas por Alvo (YDS/Target) & Catch %
    elif "Wide Receivers" in posicao:
        with col_filtro:
            min_alvos = st.slider(
                "Mínimo de Alvos (Targets):",
                min_value=10,
                max_value=400,
                value=60,
                step=10,
            )

        st.subheader(f"🏆 Ranking de Wide Receivers (Mínimo de {min_alvos} alvos)")

        wr_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("complete_pass").sum().alias("recepcoes"),
                    pl.col("yards_gained").sum().alias("jardas_recebidas"),
                    (
                        pl.col("yards_gained").sum() / pl.col("play_id").count()
                    ).alias("jardas_por_alvo"),
                    (
                        pl.col("complete_pass").sum() / pl.col("play_id").count()
                    ).alias("catch_percentage"),
                ]
            )
            .filter(pl.col("alvos") >= min_alvos)
            .sort("jardas_recebidas", descending=True)
        )

        df_wr_display = wr_stats.to_pandas()
        df_wr_display["catch_percentage"] = df_wr_display["catch_percentage"] * 100
        st.dataframe(
            df_wr_display.style.format(
                {
                    "jardas_por_alvo": "{:.2f}",
                    "catch_percentage": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

    # 4. TIGHT ENDS: Jardas por Alvo (YDS/Target) & EPA/Alvo
    elif "Tight Ends" in posicao:
        with col_filtro:
            min_alvos = st.slider(
                "Mínimo de Alvos (Targets):",
                min_value=10,
                max_value=300,
                value=40,
                step=10,
            )

        st.subheader(f"🏆 Ranking de Tight Ends (Mínimo de {min_alvos} alvos)")

        te_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("complete_pass").sum().alias("recepcoes"),
                    pl.col("yards_gained").sum().alias("jardas_recebidas"),
                    (
                        pl.col("yards_gained").sum() / pl.col("play_id").count()
                    ).alias("jardas_por_alvo"),
                    pl.col("epa").mean().alias("epa_por_alvo"),
                ]
            )
            .filter(pl.col("alvos") >= min_alvos)
            .sort("epa_por_alvo", descending=True)
        )

        st.dataframe(
            te_stats.to_pandas().style.format(
                {"jardas_por_alvo": "{:.2f}", "epa_por_alvo": "{:.3f}"}
            ),
            use_container_width=True,
        )