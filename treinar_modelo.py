import joblib
import nflreadpy as nfl
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar Múltiplos Anos de Dados
anos = [2022, 2023, 2024, 2025]
print(f"1. Carregando dados do PBP para os anos: {anos}...")
pbp = nfl.load_pbp(anos)

# 2. Filtragem e Tratamento de Features
print("2. Filtrando e tratando jogadas...")
df_ml = (
    pbp.filter(
        (pl.col("play_type").is_in(["pass", "run"]))
        & (pl.col("down").is_not_null())
        & (pl.col("ydstogo").is_not_null())
        & (pl.col("yardline_100").is_not_null())
        & (pl.col("half_seconds_remaining").is_not_null())
        & (pl.col("score_differential").is_not_null())
        & (pl.col("season").is_not_null())
    )
    .select(
        [
            "season",
            "down",
            "ydstogo",
            "yardline_100",
            "half_seconds_remaining",
            "score_differential",
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

# 3. Divisão Temporal (Treino: 2022-2024 | Teste: 2025)
df_treino = df_pandas[df_pandas["season"] < 2025]
df_teste = df_pandas[df_pandas["season"] == 2025]

features = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
]

X_train = df_treino[features]
y_train = df_treino["target"]

X_test = df_teste[features]
y_test = df_teste["target"]

print(f"   - Treino (2022-2024): {len(X_train):,} jogadas".replace(",", "."))
print(f"   - Teste (2025): {len(X_test):,} jogadas".replace(",", "."))

# 4. Treinamento
print("3. Treinando o modelo...")
modelo = RandomForestClassifier(
    n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
)
modelo.fit(X_train, y_train)

# 5. Avaliação
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print(f"\n✅ Modelo treinado com sucesso!")
print(f"🎯 Nova Acurácia (Testado na temporada 2025): {acuracia * 100:.2f}%\n")
print(classification_report(y_test, previsoes, target_names=["Corrida (0)", "Passe (1)"]))

# 6. Salvar Modelo
joblib.dump(modelo, "modelo_pass_run.joblib")
print("💾 Modelo salvo como 'modelo_pass_run.joblib'")