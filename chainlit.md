# 🎓 Bem-vindo ao RAG UFG!

Este é um **Assistente Inteligente Especializado** nas normativas, regulamentos e resoluções da **Universidade Federal de Goiás (UFG)**.

O assistente foi desenvolvido como parte do projeto da disciplina de **Linguagens e Paradigmas da Programação (LPP)**, utilizando uma arquitetura multi-agente avançada (RAG - *Retrieval-Augmented Generation*).

### 🔍 Como usar:
Basta digitar sua pergunta no chat abaixo (ou clicar em um dos exemplos). O sistema irá:
1. **Interpretar** a sua dúvida (mesmo que use gírias ou termos informais).
2. **Pesquisar** em milhares de páginas de documentos oficiais da UFG.
3. **Sintetizar** uma resposta precisa baseada **exclusivamente** nas regras da universidade, sem inventar informações.

### 🤖 Como ele pensa (em tempo real):
Enquanto processa, você verá o trabalho de cada agente aparecer como um passo na tela:
* **🧭 Planner** — decompõe a pergunta e traduz a gíria para o vocabulário institucional.
* **🔎 Retriever** — busca híbrida nos documentos + reranking dos trechos mais relevantes.
* **✍️ Sintetizador** — redige a resposta usando apenas o que foi recuperado.
* **⚖️ Juiz** — audita a resposta; se faltar algo, comanda uma repescagem autônoma.

> 💡 **Dica:** Você pode visualizar as fontes exatas (textos em PDF) que o assistente utilizou para responder clicando nos painéis de auditoria após a resposta!

Fique à vontade para fazer perguntas como:
* *"Como funciona o trancamento de matrícula?"*
* *"O que acontece se eu faltar muito nas disciplinas?"*
* *"Quais são as regras para o jubilamento?"*

---
### 👥 Desenvolvedores (Créditos)
* Caio Wallace Machado Gomes
* Eduardo Fraga Pereira
* Rafael Augusto Dias Batista
* Lucas Boclin Cunha Borges
