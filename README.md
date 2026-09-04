# 🏈 NFL Analytics Dashboard (2025)

Dashboard interativo desenvolvido em Python para análise avançada de métricas de desempenho de jogadores da NFL, focando em estatísticas de eficiência baseadas em **EPA (Expected Points Added)** e **Taxa de Sucesso**.

---

## 📌 Sobre o Projeto

O objetivo deste projeto é processar dados brutos de jogada a jogada (*play-by-play*) da NFL e disponibilizar rankings dinâmicos e comparativos entre diferentes posições (Quarterbacks e Running Backs).

A aplicação foi desenhada para processamento ultra-rápido de dados em memória e navegação simples via interface web.

---

## ⚙️ Tecnologias Utilizadas

* **[Python 3.10+](https://www.python.org/)** — Linguagem principal
* **[Streamlit](https://streamlit.io/)** — Criação da interface web interativa
* **[Polars](https://pypolars.org/)** — Manipulação e agregação de dados em alta performance
* **[nfl_data_py / nflreadpy](https://github.com/nflverse)** — Obtenção dos dados de play-by-play da NFL

---

## 📊 Métricas Analisadas

| Métrica | Definição | Métrica de Ordenação Principal |
| :--- | :--- | :--- |
| **EPA Médio** | Média de Pontos Esperados Adicionados por jogada. | **Quarterbacks (QBs)** — Mede a eficiência do jogo aéreo. |
| **Taxa de Sucesso** | Percentual de jogadas com `EPA > 0`. | **Running Backs (RBs)** — Mede a consistência do jogo terrestre. |

---

## 📁 Estrutura do Repositório

```text
nfl_analytics/
├── 01_explorando_dados.py  # Script de exploração inicial do dataset
├── 02_ranking_qbs.py       # Laboratório de agregação e ranking de QBs
├── 03_grafico_qbs.py       # Testes de visualização e gráficos de QBs
├── 04_ranking_rb.py        # Laboratório de agregação e ranking de RBs
├── app.py                  # Aplicação principal Streamlit (Dashboard)
├── teste.py                # Script de testes pontuais
└── README.md               # Documentação do repositório
