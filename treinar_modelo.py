import joblib
import nflreadpy as nfl
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

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

# 4. Definição do Grid de Hiperparâmetros
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [8, 10, 12, 15],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 5],
}

print("\n3. Iniciando o GridSearchCV (isso pode levar de 1 a 3 minutos)...")
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# cv=3 realiza Validação Cruzada de 3 dobras (folds)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

# 5. Extração do Melhor Modelo Encontrado
melhor_modelo = grid_search.best_estimator_

print("\n🏆 Melhores Hiperparâmetros Encontrados:")
for param, valor in grid_search.best_params_.items():
    print(f"   - {param}: {valor}")

# 6. Avaliação com os dados de Teste (Temporada 2025)
previsoes = melhor_modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print(f"\n✅ Avaliação com o Melhor Modelo!")
print(f"🎯 Acurácia na Temporada 2025: {acuracia * 100:.2f}%\n")
print(
    classification_report(
        y_test, previsoes, target_names=["Corrida (0)", "Passe (1)"]
    )
)

# 7. Salvar o Melhor Modelo
joblib.dump(melhor_modelo, "modelo_pass_run.joblib")
print("💾 Melhor modelo salvo como 'modelo_pass_run.joblib'")