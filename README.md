# 🏈 NFL Analytics Dashboard & Play-Call Predictor (2025)

Dashboard interativo desenvolvido em Python para análise avançada de métricas de desempenho da NFL e simulação preditiva de chamadas de jogada (*Play-Calling*) utilizando Machine Learning.

---

## 📌 Sobre o Projeto

O objetivo deste projeto é processar dados brutos de jogada a jogada (*play-by-play*) da NFL e disponibilizar:
1. **Rankings Dinâmicos de Jogadores:** Análise comparativa por posição (Quarterbacks, Running Backs, Wide Receivers e Tight Ends) focando em eficiência baseada em **EPA (Expected Points Added)** e **Taxa de Sucesso**.
2. **Simulador Preditivo (ML):** Ferramenta baseada em modelo de aprendizado de máquina para prever em tempo real a probabilidade de uma jogada ser **Passe (1)** ou **Corrida (0)** com base no contexto situacional da partida.

---

## ⚙️ Tecnologias Utilizadas

* **[Python 3.10+](https://www.python.org/)** — Linguagem principal
* **[Streamlit](https://streamlit.io/)** — Criação da interface web interativa
* **[Polars](https://pypolars.org/)** — Manipulação e agregação de dados em alta performance
* **[Scikit-Learn](https://scikit-learn.org/)** — Construção, treinamento e otimização do modelo preditivo (`RandomForestClassifier` e `GridSearchCV`)
* **[nflreadpy](https://github.com/nflverse)** — Obtenção dos dados oficiais de play-by-play da NFL
* **[Joblib](https://joblib.readthedocs.io/)** — Persistência e carregamento do modelo treinado

---

## 🤖 Modelo de Machine Learning (Play-Calling Predictor)

O modelo foi construído utilizando um **RandomForestClassifier** treinado com mais de 100.000 jogadas históricas das temporadas de 2022 a 2024 e avaliado nos dados de 2025.

### 🎯 Resultados & Hiperparâmetros
Após otimização via **GridSearchCV** (avaliando 144 combinações com validação cruzada `cv=3`), o modelo atingiu **~69.18% de acurácia**:

* **Hiperparâmetros Otimizados:** `n_estimators: 200`, `max_depth: 12`, `min_samples_split: 5`, `min_samples_leaf: 5`
* **Features de Entrada:**
  * Descida (`down`)
  * Jardas para o *First Down* (`ydstogo`)
  * Distância para a *Endzone* adversária (`yardline_100`)
  * Tempo restante no meio tempo (`half_seconds_remaining`)
  * Diferença de pontos do time da posse (`score_differential`)

---

## 📊 Métricas Analisadas nos Rankings

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
├── 01_explorando_dados.py   # Script de exploração inicial do dataset
├── 02_ranking_qbs.py       # Laboratório de agregação e ranking de QBs
├── 03_grafico_qbs.py       # Testes de visualização e gráficos
├── 04_ranking_rb.py        # Laboratório de agregação e ranking de RBs
├── 05_ranking_wr.py        # Laboratório de agregação de WRs
├── 06_ranking_te.py        # Laboratório de agregação de TEs
├── treinar_modelo.py       # Pipeline de ETL, treinamento e otimização do modelo ML (GridSearchCV)
├── modelo_pass_run.joblib  # Modelo Random Forest otimizado e serializado
├── app.py                  # Aplicação principal Streamlit (Dashboard + Simulador ML)
├── teste.py                # Script de testes pontuais
└── README.md               # Documentação do projeto
```
##  Como Executar o Projeto

### 1. Instalar as dependências:
```sh
pip install streamlit polars nflreadpy scikit-learn joblib pandas
```
### 2.Treinar e gerar o modelo de ML:
```sh
python treinar_modelo.py
```
### 3.Iniciar o Dashboard no Streamlit:
```sh
streamlit run app.py