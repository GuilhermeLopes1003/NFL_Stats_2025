import joblib
import nflreadpy as nfl
import pandas as pd
import polars as pl
import streamlit as st

st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")

st.title("🏈 Dashboard de Performance da NFL (2025)")


# 1. Carregamento dos dados e modelo
@st.cache_data
def carregar_dados():
    pbp = nfl.load_pbp([2025])
    rosters = nfl.load_rosters([2025])

    ids_rbs = (
        rosters.filter(pl.col("position") == "RB")
        .select("gsis_id")
        .to_series()
        .to_list()
    )
    ids_wrs = (
        rosters.filter(pl.col("position") == "WR")
        .select("gsis_id")
        .to_series()
        .to_list()
    )
    ids_tes = (
        rosters.filter(pl.col("position") == "TE")
        .select("gsis_id")
        .to_series()
        .to_list()
    )

    return pbp, ids_rbs, ids_wrs, ids_tes


@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load("modelo_pass_run.joblib")
    except Exception:
        return None


pbp, ids_rbs, ids_wrs, ids_tes = carregar_dados()
modelo = carregar_modelo()

# 2. SEÇÃO DE FILTROS SITUACIONAIS (Topo do App)
st.subheader("⚙️ Contexto do Game Script")
col_f1, col_f2 = st.columns(2)

with col_f1:
    filtro_zona = st.selectbox(
        "Zona do Campo:", ["Campo Inteiro", "Red Zone (Últimas 20 jardas)"]
    )

with col_f2:
    filtro_descida = st.selectbox(
        "Situação de Descida (Down):",
        ["Todas as Descidas", "Momentos Decisivos (3ª e 4ª descidas)"],
    )

# Aplicação dos filtros no DataFrame PBP
pbp_filtrado = pbp

if filtro_zona == "Red Zone (Últimas 20 jardas)":
    pbp_filtrado = pbp_filtrado.filter(pl.col("yardline_100") <= 20)

if filtro_descida == "Momentos Decisivos (3ª e 4ª descidas)":
    pbp_filtrado = pbp_filtrado.filter(pl.col("down").is_in([3, 4]))

st.divider()

# 3. CRIAÇÃO DAS ABAS
aba_qbs, aba_rbs, aba_wrs, aba_tes, aba_ml = st.tabs(
    [
        "🎯 Quarterbacks (Passe)",
        "🏃 Running Backs (Corrida)",
        "🙌 Wide Receivers (Recepção)",
        "🏈 Tight Ends (Recepção)",
        "🤖 Simulador ML (Passe vs Corrida)",
    ]
)

# --- ABA 1: QUARTERBACKS ---
with aba_qbs:
    st.header("Ranking de Quarterbacks")

    min_passes = st.slider(
        "Mínimo de passes tentados:", 10, 300, 50, step=10, key="s_qb"
    )

    ranking_qbs = (
        pbp_filtrado.filter(
            (pl.col("play_type") == "pass")
            & (pl.col("passer_player_name").is_not_null())
        )
        .group_by(["passer_player_name", "posteam"])
        .agg(
            [
                pl.len().alias("total_passes"),
                (pl.col("epa").gt(0).mean() * 100).alias("taxa_sucesso"),
                pl.col("epa").mean().alias("epa_medio"),
                pl.col("pass_touchdown")
                .sum()
                .cast(pl.UInt32)
                .alias("total_tds"),
                pl.col("yards_gained")
                .sum()
                .cast(pl.Int64)
                .alias("total_jardas"),
            ]
        )
        .filter(pl.col("total_passes") >= min_passes)
        .sort(["epa_medio", "taxa_sucesso"], descending=[True, True])
    )

    st.dataframe(
        ranking_qbs,
        column_config={
            "passer_player_name": "Jogador",
            "posteam": "Time",
            "total_passes": "Passes",
            "taxa_sucesso": st.column_config.NumberColumn(
                "Taxa de Sucesso", format="%.2f%%"
            ),
            "epa_medio": st.column_config.NumberColumn(
                "EPA/Jogada", format="%.3f"
            ),
            "total_tds": "TDs",
            "total_jardas": "Jardas Totais",
        },
        use_container_width=True,
        hide_index=True,
    )

# --- ABA 2: RUNNING BACKS ---
with aba_rbs:
    st.header("Ranking de Running Backs")

    min_corridas = st.slider(
        "Mínimo de corridas:", 10, 200, 30, step=5, key="s_rb"
    )

    ranking_rbs = (
        pbp_filtrado.filter(
            (pl.col("play_type") == "run")
            & (pl.col("rusher_player_name").is_not_null())
            & (pl.col("rusher_player_id").is_in(ids_rbs))
        )
        .group_by(["rusher_player_name", "posteam"])
        .agg(
            [
                pl.len().alias("total_corridas"),
                (pl.col("epa").gt(0).mean() * 100).alias("taxa_sucesso"),
                pl.col("epa").mean().alias("epa_medio"),
                pl.col("rush_touchdown")
                .sum()
                .cast(pl.UInt32)
                .alias("total_tds"),
                pl.col("yards_gained")
                .sum()
                .cast(pl.Int64)
                .alias("total_jardas"),
            ]
        )
        .filter(pl.col("total_corridas") >= min_corridas)
        .sort(["taxa_sucesso", "total_jardas"], descending=[True, True])
    )

    st.dataframe(
        ranking_rbs,
        column_config={
            "rusher_player_name": "Jogador",
            "posteam": "Time",
            "total_corridas": "Corridas",
            "taxa_sucesso": st.column_config.NumberColumn(
                "Taxa de Sucesso", format="%.2f%%"
            ),
            "epa_medio": st.column_config.NumberColumn(
                "EPA/Jogada", format="%.3f"
            ),
            "total_tds": "TDs",
            "total_jardas": "Jardas Totais",
        },
        use_container_width=True,
        hide_index=True,
    )

# --- ABA 3: WIDE RECEIVERS ---
with aba_wrs:
    st.header("Ranking de Wide Receivers")

    min_alvos_wr = st.slider(
        "Mínimo de alvos (targets):", 5, 100, 20, step=5, key="s_wr"
    )

    ranking_wrs = (
        pbp_filtrado.filter(
            (pl.col("play_type") == "pass")
            & (pl.col("receiver_player_name").is_not_null())
            & (pl.col("receiver_player_id").is_in(ids_wrs))
        )
        .group_by(["receiver_player_name", "posteam"])
        .agg(
            [
                pl.len().alias("alvos"),
                pl.col("complete_pass")
                .sum()
                .cast(pl.UInt32)
                .alias("recepcoes"),
                (pl.col("complete_pass").mean() * 100).alias("taxa_captura"),
                pl.col("yards_gained")
                .filter(pl.col("complete_pass") == 1)
                .sum()
                .cast(pl.Int64)
                .alias("jardas"),
                pl.col("pass_touchdown")
                .filter(pl.col("complete_pass") == 1)
                .sum()
                .cast(pl.UInt32)
                .alias("tds"),
                pl.col("epa").mean().alias("epa_medio"),
            ]
        )
        .filter(pl.col("alvos") >= min_alvos_wr)
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
            "taxa_captura": st.column_config.NumberColumn(
                "Catch %", format="%.2f%%"
            ),
            "jardas": "Jardas Totais",
            "tds": "TDs",
            "epa_medio": st.column_config.NumberColumn(
                "EPA/Alvo", format="%.3f"
            ),
            "jardas_por_alvo": st.column_config.NumberColumn(
                "YDS/Target", format="%.2f"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

# --- ABA 4: TIGHT ENDS ---
with aba_tes:
    st.header("Ranking de Tight Ends")

    min_alvos_te = st.slider(
        "Mínimo de alvos (targets):", 5, 80, 15, step=5, key="s_te"
    )

    ranking_tes = (
        pbp_filtrado.filter(
            (pl.col("play_type") == "pass")
            & (pl.col("receiver_player_name").is_not_null())
            & (pl.col("receiver_player_id").is_in(ids_tes))
        )
        .group_by(["receiver_player_name", "posteam"])
        .agg(
            [
                pl.len().alias("alvos"),
                pl.col("complete_pass")
                .sum()
                .cast(pl.UInt32)
                .alias("recepcoes"),
                (pl.col("complete_pass").mean() * 100).alias("taxa_captura"),
                pl.col("yards_gained")
                .filter(pl.col("complete_pass") == 1)
                .sum()
                .cast(pl.Int64)
                .alias("jardas"),
                pl.col("pass_touchdown")
                .filter(pl.col("complete_pass") == 1)
                .sum()
                .cast(pl.UInt32)
                .alias("tds"),
                pl.col("epa").mean().alias("epa_medio"),
            ]
        )
        .filter(pl.col("alvos") >= min_alvos_te)
        .with_columns(
            (pl.col("jardas") / pl.col("alvos")).alias("jardas_por_alvo")
        )
        .sort(["jardas_por_alvo", "jardas"], descending=[True, True])
    )

    st.dataframe(
        ranking_tes,
        column_config={
            "receiver_player_name": "Jogador",
            "posteam": "Time",
            "alvos": "Alvos",
            "recepcoes": "Recepções",
            "taxa_captura": st.column_config.NumberColumn(
                "Catch %", format="%.2f%%"
            ),
            "jardas": "Jardas Totais",
            "tds": "TDs",
            "epa_medio": st.column_config.NumberColumn(
                "EPA/Alvo", format="%.3f"
            ),
            "jardas_por_alvo": st.column_config.NumberColumn(
                "YDS/Target", format="%.2f"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

# --- ABA 5: SIMULADOR ML ---
with aba_ml:
    st.header("🧠 Preditor de Chamada de Jogada (Random Forest)")
    st.write(
        "Ajuste as condições da partida abaixo para simular a decisão da comissão técnica:"
    )

    if modelo is None:
        st.error(
            "⚠️ O arquivo `modelo_pass_run.joblib` não foi encontrado. Execute `python treinar_modelo.py` no terminal antes de usar o simulador."
        )
    else:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            down = st.number_input(
                "Descida (Down):", min_value=1, max_value=4, value=1, step=1
            )
            ydstogo = st.number_input(
                "Jardas para o First Down:",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
            )
            yardline_100 = st.slider(
                "Distância para a Endzone Adversária:",
                min_value=1,
                max_value=99,
                value=75,
                help="10 = Red Zone | 50 = Meio de campo | 80 = Linha de 20 própria",
            )

        with col_m2:
            minutos = st.number_input(
                "Minutos restantes no Half:",
                min_value=0,
                max_value=30,
                value=15,
            )
            segundos = st.number_input(
                "Segundos restantes no minuto:",
                min_value=0,
                max_value=59,
                value=0,
            )
            score_differential = st.number_input(
                "Diferença de Pontos (Time com a posse):",
                min_value=-50,
                max_value=50,
                value=0,
                help="Positivo = Vencendo | Negativo = Perdendo",
            )

        half_seconds_remaining = (minutos * 60) + segundos

        # Entrada para o modelo
        dados_simulacao = pd.DataFrame(
            [
                {
                    "down": down,
                    "ydstogo": ydstogo,
                    "yardline_100": yardline_100,
                    "half_seconds_remaining": half_seconds_remaining,
                    "score_differential": score_differential,
                }
            ]
        )

        probabilidades = modelo.predict_proba(dados_simulacao)[0]
        prob_corrida = probabilidades[0] * 100
        prob_passe = probabilidades[1] * 100

        st.divider()
        st.subheader("📊 Resultado da Previsão")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric(
                label="Probabilidade de CORRIDA 🏃",
                value=f"{prob_corrida:.1f}%",
            )
            st.progress(prob_corrida / 100)

        with col_p2:
            st.metric(
                label="Probabilidade de PASSE 🏈", value=f"{prob_passe:.1f}%"
            )
            st.progress(prob_passe / 100)

        if prob_passe > prob_corrida:
            st.info(
                f"💡 **Predição Final:** Tendência clara de **PASSE** ({prob_passe:.1f}% vs {prob_corrida:.1f}%)."
            )
        else:
            st.info(
                f"💡 **Predição Final:** Tendência clara de **CORRIDA** ({prob_corrida:.1f}% vs {prob_passe:.1f}%)."
            )