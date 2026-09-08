# 🏈 NFL Analytics Dashboard & Play-Call Predictor (v2.0)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Polars](https://img.shields.io/badge/Data%20Engine-Polars-CD7F32?style=flat-square)
![XGBoost](https://img.shields.io/badge/ML%20Model-XGBoost-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)

Dashboard interativo desenvolvido em Python para análise avançada de métricas de desempenho da NFL e simulação preditiva de chamadas de jogada (*Play-Calling*) utilizando Machine Learning.

---

## 📌 Sobre o Projeto

O objetivo deste projeto é processar dados brutos de jogada a jogada (*play-by-play*) da NFL (2022–2025) e disponibilizar:

1. **Rankings Dinâmicos de Jogadores (`app.py`):** Análise comparativa por posição (Quarterbacks, Running Backs, Wide Receivers e Tight Ends) focando em eficiência baseada em **EPA (Expected Points Added)** e **Taxa de Sucesso**.
2. **Simulador Preditivo v2 (`app_v2.py`):** Ferramenta tática baseada em **XGBoost Classifier** para prever em tempo real a probabilidade de uma jogada ser **Passe (1)** ou **Corrida (0)** combinando o contexto situacional com formações pré-snap e estatísticas dinâmicas das equipes.

---

## ⚙️ Tecnologias Utilizadas

* **[Python 3.10+](https://www.python.org/)** — Linguagem principal
* **[Streamlit](https://streamlit.io/)** — Interface web interativa (Dashboard v1 e Simulador v2)
* **[Polars](https://pypolars.org/)** — Manipulação e agregação de dados em alta performance
* **[XGBoost](https://xgboost.readthedocs.io/) & [Scikit-Learn](https://scikit-learn.org/)** — Construção, treinamento e otimização do modelo preditivo (`GridSearchCV`)
* **[nflreadpy](https://github.com/nflverse/nflreadpy)** — Obtenção dos dados oficiais de *play-by-play* da NFL
* **[Joblib](https://joblib.readthedocs.io/)** — Persistência e carregamento do modelo treinado

---

## 🤖 Evolução do Modelo de Machine Learning (Play-Calling Predictor)

O projeto passou por uma evolução técnica significativa, migrando de um modelo inicial baseado apenas em situação de jogo para um modelo tático rebalanceado de alta precisão.

### 📈 Comparativo de Evolução

| Métrica / Recurso | Versão 1.0 (Random Forest) | Versão 2.0 (XGBoost Balanceado) |
| :--- | :---: | :---: |
| **Algoritmo** | `RandomForestClassifier` | `XGBoostClassifier` |
| **Acurácia (Teste 2025)** | **69.18%** | **72.61%** |
| **Nº de Features** | 5 situacionais | 10 (Situacionais + Táticas + Times) |
| **Controle de Viés** | Sem regularização de colunas | Regularização L1/L2 + `colsample_bytree: 0.6` |
| **F1-Score (Passe)** | 0.72 | **0.76** |

### 🎯 Resultados & Hiperparâmetros da Versão 2.0

Treinado com **105.680 jogadas** (2022–2024) e avaliado em **34.502 jogadas** inéditas da temporada de 2025:

* **Hiperparâmetros Otimizados (`GridSearchCV`):**
  `{'colsample_bytree': 0.6, 'learning_rate': 0.03, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'subsample': 0.8}`

* **Matriz de Importância Rebalanceada das Features:**
  * **`shotgun` (74.73%):** Formação e alinhamento do QB pré-snap.
  * **`down` (7.80%):** Descida atual.
  * **`ydstogo` (7.49%):** Jardas necessárias para o *first down*.
  * **`half_seconds_remaining` (2.94%):** Tempo restante no meio tempo.
  * **`score_differential` (1.96%):** Diferença no placar.
  * **`offense_pass_ratio` (1.67%):** Tendência histórica do ataque.
  * **`no_huddle` (1.66%):** Ritmo de jogo (*Ataque Rápido*).
  * **`yardline_100` (1.09%):** Posição de campo em relação à *end zone*.
  * **`def_epa_against_run` (0.33%) & `def_epa_against_pass` (0.33%):** Eficiência da defesa adversária.

---

## 📊 Métricas Analisadas nos Rankings de Jogadores

| Posição | Métricas Principais | Foco da Análise |
| :--- | :--- | :--- |
| **Quarterbacks (QBs)** | EPA/Jogada & Taxa de Sucesso (%) | Eficiência e consistência no jogo aéreo. |
| **Running Backs (RBs)** | Taxa de Sucesso (%) & Jardas Totais | Eficiência e volume do jogo terrestre. |
| **Wide Receivers (WRs)** | Jardas por Alvo (`YDS/Target`) & Catch % | Eficiência ao ser visado pelo QB. |
| **Tight Ends (TEs)** | Jardas por Alvo (`YDS/Target`) & EPA/Alvo | Impacto em recepções no meio de campo e *Red Zone*. |

---

## 📁 Estrutura do Repositório

```text
NFL_Stats_2025/
├── 01_explorando_dados.py      # Script de exploração inicial do dataset
├── 02_ranking_qbs.py          # Laboratório de agregação e ranking de QBs
├── 03_grafico_qbs.py          # Testes de visualização e gráficos
├── 04_ranking_rb.py           # Laboratório de agregação e ranking de RBs
├── 05_ranking_wr.py           # Laboratório de agregação de WRs
├── 06_ranking_te.py           # Laboratório de agregação de TEs
├── 09_formation_features_ml.py# Laboratório de testes do XGBoost rebalanceado (v2)
├── treinar_modelo.py          # Pipeline de ETL, treino e exportação do XGBoost v2 (.joblib)
├── modelo_pass_run.joblib     # Modelo XGBoost otimizado e serializado (10 features)
├── app.py                     # Aplicação original (Dashboard de Rankings + Simulador v1)
├── app_v2.py                  # Interface v2 (Simulador Tático em tempo real com estatísticas de times)
└── README.md                  # Documentação do projeto
```
##  Como Executar o Projeto

### 1. Instalar as dependências:
```sh
pip install streamlit polars nflreadpy scikit-learn xgboost joblib pandas
```
### 2.Treinar e gerar o modelo de ML:
```sh
python treinar_modelo.py
```
### 3.Iniciar o Dashboard no Streamlit:
```sh
streamlit run app.py
```
### Para acessar a nova Interface do Simulador Tático (v2.0):
```sh
stramlit run app_v2.py