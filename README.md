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
├── app.py                 # Aplicação principal Streamlit (Interface e rankings)
├── 01_ranking_qb.py       # Script de teste/laboratório para agregação de QBs
├── 04_ranking_rb.py       # Script de teste/laboratório para agregação de RBs
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do repositório