import joblib
import nflreadpy as nfl
import polars as pl
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

print("=== INICIANDO TREINAMENTO DO MODELO V2 (XGBOOST) ===")

# 1. Carregar Dados de 2022 a 2025
anos = [2022, 2023, 2024, 2025]
print(f"1. Carregando dados PBP para {anos}...")
pbp = nfl.load_pbp(anos)

# 2. Filtragem de Jogadas Válidas
print("2. Filtrando jogadas e limpando dados...")
pbp_valid = pbp.filter(
    (pl.col("play_type").is_in(["pass", "run"]))
    & (pl.col("down").is_not_null())
    & (pl.col("ydstogo").is_not_null())
    & (pl.col("yardline_100").is_not_null())
    & (pl.col("half_seconds_remaining").is_not_null())
    & (pl.col("score_differential").is_not_null())
    & (pl.col("shotgun").is_not_null())
    & (pl.col("no_huddle").is_not_null())
    & (pl.col("season").is_not_null())
    & (pl.col("posteam").is_not_null())
    & (pl.col("defteam").is_not_null())
    & (pl.col("epa").is_not_null())
)

# 3. Engenharia de Features (Agregações de Time)
print("3. Calculando métricas de tendência de Ataque e Defesa...")

offense_stats = pbp_valid.group_by(["season", "posteam"]).agg(
    [
        (
            pl.col("play_type").filter(pl.col("play_type") == "pass").count()
            / pl.col("play_type").count()
        ).alias("offense_pass_ratio")
    ]
)

defense_stats = pbp_valid.group_by(["season", "defteam"]).agg(
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

# 4. Unir Features e Criar Dataset Final
print("4. Consolidando dataset final de ML...")
df_ml = (
    pbp_valid.join(offense_stats, on=["season", "posteam"], how="left")
    .join(defense_stats, on=["season", "defteam"], how="left")
    .select(
        [
            "season",
            "down",
            "ydstogo",
            "yardline_100",
            "half_seconds_remaining",
            "score_differential",
            "shotgun",
            "no_huddle",
            "offense_pass_ratio",
            "def_epa_against_pass",
            "def_epa_against_run",
            "play_type",
        ]
    )
    .with_columns(
        pl.when(pl.col("play_type") == "pass")
        .then(1)
        .otherwise(0)
        .alias("target")
    )
    .drop("play_type")
)

df_pandas = df_ml.to_pandas()

# 5. Divisão Treino (2022-2024) e Teste (2025)
df_treino = df_pandas[df_pandas["season"] < 2025]
df_teste = df_pandas[df_pandas["season"] == 2025]

features = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
    "shotgun",
    "no_huddle",
    "offense_pass_ratio",
    "def_epa_against_pass",
    "def_epa_against_run",
]

X_train = df_treino[features]
y_train = df_treino["target"]

X_test = df_teste[features]
y_test = df_teste["target"]

print(f"   - Base de Treino: {len(X_train):,} jogadas".replace(",", "."))
print(f"   - Base de Teste (2025): {len(X_test):,} jogadas".replace(",", "."))

# 6. Otimização do XGBoost Balanceado
param_grid = {
    "n_estimators": [150, 200],
    "max_depth": [4, 5],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.5, 0.6],
    "reg_alpha": [1.0, 5.0],
    "reg_lambda": [1.0, 5.0],
}

print("\n5. Executando GridSearchCV para treinar o XGBoost...")
xgb_base = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

melhor_modelo_xgb = grid_search.best_estimator_

# 7. Avaliação
previsoes = melhor_modelo_xgb.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print("\n🏆 Melhores Hiperparâmetros:", grid_search.best_params_)
print(f"🎯 Acurácia Final no Teste (2025): {acuracia * 100:.2f}%\n")
print(
    classification_report(
        y_test, previsoes, target_names=["Corrida (0)", "Passe (1)"]
    )
)

# 8. Exportação do Modelo Treinado
print("6. Salvando o arquivo de produção modelo_pass_run.joblib...")
joblib.dump(melhor_modelo_xgb, "modelo_pass_run.joblib")
print("✅ Modelo V2 salvo com sucesso!")