import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px
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
    pbp = nfl.load_pbp([2024, 2025])
    return pbp.filter(
        (pl.col("play_type").is_in(["pass", "run"]))
        & (pl.col("posteam").is_not_null())
        & (pl.col("defteam").is_not_null())
        & (pl.col("epa").is_not_null())
    )


@st.cache_data
def processar_stats_times(pbp_valid):
    offense = pbp_valid.group_by("posteam").agg(
        [
            (
                pl.col("play_type").filter(pl.col("play_type") == "pass").count()
                / pl.col("play_type").count()
            ).alias("offense_pass_ratio")
        ]
    )

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
    "Plataforma unificada para análise de desempenho de atletas, simulação tática e visualizações avançadas."
)

aba_simulador, aba_rankings, aba_graficos = st.tabs(
    [
        "🎯 Simulador Preditivo (v2.0)",
        "📊 Rankings de Jogadores",
        "📈 Visualizações & Analytics",
    ]
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
# ABA 2: RANKINGS DE JOGADORES
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

    # 1. QUARTERBACKS
    if "Quarterbacks" in posicao:
        with col_filtro:
            min_tentativas = st.slider(
                "Mínimo de Passes Tentados:", 10, 1200, 250, 10
            )

        qb_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("passer_player_name").is_not_null())
            )
            .group_by("passer_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
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

        df_display = qb_stats.to_pandas()
        df_display["taxa_sucesso"] = df_display["taxa_sucesso"] * 100

        st.subheader(
            f"🏆 Ranking de QBs (Mínimo de {min_tentativas} tentativas)"
        )
        st.dataframe(
            df_display.style.format(
                {"epa_por_jogada": "{:.3f}", "taxa_sucesso": "{:.1f}%"}
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Detalhes do Atleta")
        lista_jogadores = df_display["passer_player_name"].tolist()
        if lista_jogadores:
            jogador_sel = st.selectbox(
                "Selecione um QB para ver o time e o resumo:",
                options=lista_jogadores,
            )
            dados_atleta = df_display[
                df_display["passer_player_name"] == jogador_sel
            ].iloc[0]
            st.info(
                f"👤 **Jogador:** {dados_atleta['passer_player_name']} | "
                f"🛡️ **Time:** {dados_atleta['time']} | "
                f"🏈 **Passes:** {dados_atleta['tentativas']} | "
                f"📈 **EPA/Jogada:** {dados_atleta['epa_por_jogada']:.3f} | "
                f"🎯 **Sucesso:** {dados_atleta['taxa_sucesso']:.1f}%"
            )

    # 2. RUNNING BACKS
    elif "Running Backs" in posicao:
        with col_filtro:
            min_carregadas = st.slider(
                "Mínimo de Corridas (Carregadas):", 10, 600, 100, 10
            )

        rb_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "run")
                & (pl.col("rusher_player_name").is_not_null())
            )
            .group_by("rusher_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("carregadas"),
                    pl.col("yards_gained")
                    .sum()
                    .cast(pl.Int64)
                    .alias("jardas_totais"),
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

        df_display = rb_stats.to_pandas()
        df_display["taxa_sucesso"] = df_display["taxa_sucesso"] * 100

        st.subheader(
            f"🏆 Ranking de RBs (Mínimo de {min_carregadas} carregadas)"
        )
        st.dataframe(
            df_display.style.format(
                {
                    "jardas_totais": "{:.0f}",
                    "media_jardas": "{:.2f}",
                    "taxa_sucesso": "{:.1f}%",
                    "epa_por_corrida": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Detalhes do Atleta")
        lista_jogadores = df_display["rusher_player_name"].tolist()
        if lista_jogadores:
            jogador_sel = st.selectbox(
                "Selecione um RB para ver o time e o resumo:",
                options=lista_jogadores,
            )
            dados_atleta = df_display[
                df_display["rusher_player_name"] == jogador_sel
            ].iloc[0]
            st.info(
                f"👤 **Jogador:** {dados_atleta['rusher_player_name']} | "
                f"🛡️ **Time:** {dados_atleta['time']} | "
                f"🏃 **Carregadas:** {dados_atleta['carregadas']} | "
                f"📏 **Jardas Totais:** {dados_atleta['jardas_totais']} | "
                f"📈 **EPA/Corrida:** {dados_atleta['epa_por_corrida']:.3f}"
            )

    # 3. WIDE RECEIVERS
    elif "Wide Receivers" in posicao:
        with col_filtro:
            min_alvos = st.slider(
                "Mínimo de Alvos (Targets):", 10, 400, 60, 10
            )

        wr_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("complete_pass")
                    .sum()
                    .cast(pl.Int64)
                    .alias("recepcoes"),
                    pl.col("yards_gained")
                    .sum()
                    .cast(pl.Int64)
                    .alias("jardas_recebidas"),
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

        df_display = wr_stats.to_pandas()
        df_display["catch_percentage"] = df_display["catch_percentage"] * 100

        st.subheader(f"🏆 Ranking de Wide Receivers (Mínimo de {min_alvos} alvos)")
        st.dataframe(
            df_display.style.format(
                {
                    "recepcoes": "{:.0f}",
                    "jardas_recebidas": "{:.0f}",
                    "jardas_por_alvo": "{:.2f}",
                    "catch_percentage": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Detalhes do Atleta")
        lista_jogadores = df_display["receiver_player_name"].tolist()
        if lista_jogadores:
            jogador_sel = st.selectbox(
                "Selecione um WR para ver o time e o resumo:",
                options=lista_jogadores,
            )
            dados_atleta = df_display[
                df_display["receiver_player_name"] == jogador_sel
            ].iloc[0]
            st.info(
                f"👤 **Jogador:** {dados_atleta['receiver_player_name']} | "
                f"🛡️ **Time:** {dados_atleta['time']} | "
                f"🎯 **Alvos:** {dados_atleta['alvos']} | "
                f"🙌 **Recepções:** {dados_atleta['recepcoes']} | "
                f"📊 **Catch %:** {dados_atleta['catch_percentage']:.1f}%"
            )

    # 4. TIGHT ENDS
    elif "Tight Ends" in posicao:
        with col_filtro:
            min_alvos = st.slider(
                "Mínimo de Alvos (Targets):", 10, 300, 40, 10
            )

        te_stats = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("complete_pass")
                    .sum()
                    .cast(pl.Int64)
                    .alias("recepcoes"),
                    pl.col("yards_gained")
                    .sum()
                    .cast(pl.Int64)
                    .alias("jardas_recebidas"),
                    (
                        pl.col("yards_gained").sum() / pl.col("play_id").count()
                    ).alias("jardas_por_alvo"),
                    pl.col("epa").mean().alias("epa_por_alvo"),
                ]
            )
            .filter(pl.col("alvos") >= min_alvos)
            .sort("epa_por_alvo", descending=True)
        )

        df_display = te_stats.to_pandas()

        st.subheader(f"🏆 Ranking de Tight Ends (Mínimo de {min_alvos} alvos)")
        st.dataframe(
            df_display.style.format(
                {
                    "recepcoes": "{:.0f}",
                    "jardas_recebidas": "{:.0f}",
                    "jardas_por_alvo": "{:.2f}",
                    "epa_por_alvo": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Detalhes do Atleta")
        lista_jogadores = df_display["receiver_player_name"].tolist()
        if lista_jogadores:
            jogador_sel = st.selectbox(
                "Selecione um TE para ver o time e o resumo:",
                options=lista_jogadores,
            )
            dados_atleta = df_display[
                df_display["receiver_player_name"] == jogador_sel
            ].iloc[0]
            st.info(
                f"👤 **Jogador:** {dados_atleta['receiver_player_name']} | "
                f"🛡️ **Time:** {dados_atleta['time']} | "
                f"🎯 **Alvos:** {dados_atleta['alvos']} | "
                f"📏 **Jardas/Alvo:** {dados_atleta['jardas_por_alvo']:.2f} | "
                f"📈 **EPA/Alvo:** {dados_atleta['epa_por_alvo']:.3f}"
            )


# ==========================================
# ABA 3: VISUALIZAÇÕES & ANALYTICS INTERATIVOS
# ==========================================
with aba_graficos:
    st.header("📈 Visualizações Avançadas de Desempenho")
    st.caption(
        "Explore relações de eficiência, volume e geração de valor (EPA) com gráficos interativos."
    )

    tipo_grafico = st.selectbox(
        "Selecione a Análise Visual:",
        options=[
            "Quarterbacks: EPA/Passes vs. Taxa de Sucesso (%)",
            "Running Backs: Top Eficiência por EPA/Corrida",
            "Wide Receivers: Alvos Totais vs. Jardas/Alvo",
            "Tight Ends: Eficiência de EPA por Alvo",
        ],
    )

    # 1. GRÁFICO DE DISPERSÃO - QUARTERBACKS
    if "Quarterbacks" in tipo_grafico:
        min_qb_passes = st.slider(
            "Filtrar QBs por mínimo de passes no gráfico:",
            10,
            1200,
            200,
            10,
            key="slider_graf_qb",
        )

        df_qb_plot = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("passer_player_name").is_not_null())
            )
            .group_by("passer_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("tentativas"),
                    pl.col("epa").mean().alias("epa_por_jogada"),
                    (
                        pl.col("epa").filter(pl.col("epa") > 0).count()
                        / pl.col("play_id").count()
                    ).alias("taxa_sucesso"),
                ]
            )
            .filter(pl.col("tentativas") >= min_qb_passes)
            .to_pandas()
        )

        df_qb_plot["taxa_sucesso"] = df_qb_plot["taxa_sucesso"] * 100

        fig_qb = px.scatter(
            df_qb_plot,
            x="taxa_sucesso",
            y="epa_por_jogada",
            size="tentativas",
            color="time",
            text="passer_player_name",
            hover_data=["passer_player_name", "time", "tentativas"],
            labels={
                "taxa_sucesso": "Taxa de Sucesso (%)",
                "epa_por_jogada": "EPA Médio por Jogada",
                "tentativas": "Passes Tentados",
            },
            title=f"Eficiência de QBs (Mínimo {min_qb_passes} passes)",
        )

        fig_qb.update_traces(textposition="top center")
        fig_qb.add_hline(
            y=df_qb_plot["epa_por_jogada"].mean(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Média EPA",
        )
        fig_qb.add_vline(
            x=df_qb_plot["taxa_sucesso"].mean(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Média Sucesso",
        )
        fig_qb.update_layout(height=600)

        st.plotly_chart(fig_qb, use_container_width=True)

    # 2. GRÁFICO DE BARRAS - RUNNING BACKS (FILTRADO SEM QBS)
    elif "Running Backs" in tipo_grafico:
        top_n = st.slider(
            "Quantidade de RBs no Top Ranking:",
            5,
            30,
            15,
            5,
            key="slider_graf_rb",
        )

        # Filtra e remove os QBs conhecidos que fazem corridas desenhadas/scrambles
        df_rb_plot = (
            pbp_data.filter(
                (pl.col("play_type") == "run")
                & (pl.col("rusher_player_name").is_not_null())
                & (pl.col("qb_scramble") == 0)  # Remove scrambles de QB
                & (
                    ~pl.col("rusher_player_name").is_in(
                        [
                            "P.Mahomes",
                            "J.Allen",
                            "B.Purdy",
                            "B.Mayfield",
                            "D.Maye",
                            "J.Daniels",
                            "D.Jones",
                            "J.Herbert",
                            "B.Nix",
                            "J.Hurts",
                            "T.Lawrence",
                            "A.Richardson",
                            "C.Stroud",
                            "L.Jackson",
                            "K.Murray",
                            "C.Williams",
                            "J.Fields",
                            "T.Hill",
                        ]
                    )
                )
            )
            .group_by("rusher_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("carregadas"),
                    pl.col("yards_gained").sum().alias("jardas_totais"),
                    pl.col("epa").mean().alias("epa_por_corrida"),
                ]
            )
            .filter(pl.col("carregadas") >= 50)
            .sort("epa_por_corrida", descending=True)
            .limit(top_n)
            .to_pandas()
        )

        fig_rb = px.bar(
            df_rb_plot,
            x="epa_por_corrida",
            y="rusher_player_name",
            orientation="h",
            color="epa_por_corrida",
            color_continuous_scale="RdYlGn",
            text_auto=".3f",
            hover_data=["time", "carregadas", "jardas_totais"],
            labels={
                "epa_por_corrida": "EPA Médio por Corrida",
                "rusher_player_name": "Running Back",
            },
            title=f"Top {top_n} Running Backs Mais Eficientes em EPA por Corrida (Mínimo 50 carregadas)",
        )

        fig_rb.update_layout(
            yaxis={"categoryorder": "total ascending"}, height=600
        )

        st.plotly_chart(fig_rb, use_container_width=True)

    # 3. GRÁFICO DE DISPERSÃO - WIDE RECEIVERS
    elif "Wide Receivers" in tipo_grafico:
        min_rec_alvos = st.slider(
            "Filtrar WRs por mínimo de alvos no gráfico:",
            10,
            400,
            50,
            10,
            key="slider_graf_wr",
        )

        df_wr_plot = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("yards_gained").sum().alias("jardas_totais"),
                    (
                        pl.col("yards_gained").sum() / pl.col("play_id").count()
                    ).alias("jardas_por_alvo"),
                    pl.col("epa").mean().alias("epa_por_alvo"),
                ]
            )
            .filter(pl.col("alvos") >= min_rec_alvos)
            .to_pandas()
        )

        fig_wr = px.scatter(
            df_wr_plot,
            x="alvos",
            y="jardas_por_alvo",
            size="jardas_totais",
            color="epa_por_alvo",
            text="receiver_player_name",
            color_continuous_scale="Viridis",
            hover_data=["receiver_player_name", "time", "jardas_totais"],
            labels={
                "alvos": "Volume Total de Alvos",
                "jardas_por_alvo": "Jardas por Alvo (YDS/Target)",
                "epa_por_alvo": "EPA / Alvo",
            },
            title=f"Volume vs Eficiência de WRs (Mínimo {min_rec_alvos} alvos)",
        )

        fig_wr.update_traces(textposition="top center")
        fig_wr.update_layout(height=600)

        st.plotly_chart(fig_wr, use_container_width=True)

    # 4. GRÁFICO DE DISPERSÃO - TIGHT ENDS
    elif "Tight Ends" in tipo_grafico:
        min_te_alvos = st.slider(
            "Filtrar TEs por mínimo de alvos no gráfico:",
            10,
            300,
            30,
            10,
            key="slider_graf_te",
        )

        df_te_plot = (
            pbp_data.filter(
                (pl.col("play_type") == "pass")
                & (pl.col("receiver_player_name").is_not_null())
            )
            .group_by("receiver_player_name")
            .agg(
                [
                    pl.col("posteam").last().alias("time"),
                    pl.col("play_id").count().alias("alvos"),
                    pl.col("complete_pass").sum().alias("recepcoes"),
                    pl.col("yards_gained").sum().alias("jardas_totais"),
                    pl.col("epa").mean().alias("epa_por_alvo"),
                ]
            )
            .filter(pl.col("alvos") >= min_te_alvos)
            .sort("epa_por_alvo", descending=True)
            .to_pandas()
        )

        fig_te = px.scatter(
            df_te_plot,
            x="alvos",
            y="epa_por_alvo",
            size="recepcoes",
            color="time",
            text="receiver_player_name",
            hover_data=[
                "receiver_player_name",
                "time",
                "recepcoes",
                "jardas_totais",
            ],
            labels={
                "alvos": "Alvos Totais (Targets)",
                "epa_por_alvo": "EPA Médio por Alvo",
                "recepcoes": "Recepções",
            },
            title=f"Eficiência de Tight Ends por EPA/Alvo (Mínimo {min_te_alvos} alvos)",
        )

        fig_te.update_traces(textposition="top center")
        fig_te.add_hline(
            y=df_te_plot["epa_por_alvo"].mean(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Média EPA",
        )
        fig_te.update_layout(height=600)

        st.plotly_chart(fig_te, use_container_width=True)