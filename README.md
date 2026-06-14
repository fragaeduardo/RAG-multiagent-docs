# UFG RAG - Assistente de Regulamentos e Normativas 🎓

Este é um sistema inteligente de Busca e Geração de Respostas (RAG - *Retrieval-Augmented Generation*) focado exclusivamente no banco de documentos, normativas, cartilhas e regulamentos da **Universidade Federal de Goiás (UFG)**.

> **Nota Acadêmica:** Este projeto foi desenvolvido e idealizado como parte da disciplina de **Linguagens e Paradigmas de Programação (LPP)**.

---

## 🧠 Como funciona a Inteligência do Sistema?

Sem entrar em jargões profundos, o sistema funciona como um assistente de pesquisa incansável que consulta as leis da faculdade antes de te responder. O fluxo da "mente" dele funciona assim:

1. **Planner Agent:** Quando você faz uma pergunta complexa (ex: *"Como funciona o trancamento e jubilamento?"*), este agente percebe as múltiplas intenções da frase. Ele decompõe a sua pergunta em partes menores e traduz a sua linguagem informal para o vocabulário institucional da UFG.
2. **Retriever:** O sistema realiza uma busca vetorial híbrida em um banco de dados com milhares de trechos de documentos da UFG e recupera as 100 páginas mais relevantes com base em similaridade semântica.
3. **Reranker (Cross-Encoder):** Uma outra inteligência artificial especializada (modelo de reranking) lê detalhadamente essas 100 páginas e descarta as menos aderentes, garantindo que apenas o *Top 15* mais preciso seja injetado como contexto.
4. **Synthesizer Agent:** Um modelo de linguagem junta as regras exatas recuperadas e redige a resposta final. Ele é instruído sob "Zero Interação", ou seja, é proibido de alucinar ou inventar regras que não estão nos textos recuperados.
5. **Judge Agent:** O sistema avalia a própria resposta final. Se o Juiz detectar alucinações ou considerar a resposta incompleta, ele interrompe o fluxo, volta ao passo 1 e comanda uma repescagem autônoma para recuperar os dados que faltaram antes de exibir a tela para o usuário.

Para ver a especificação pesada e o design pattern dos Agentes e do motor de processamento, leia a nossa [Documentação de Arquitetura Oficial](arquitetura.md).

---

## 🚀 Como Rodar o Projeto

Como o banco de dados já foi processado e está incluso (via Snapshot Vetorial na pasta do Qdrant), você não precisará rodar scripts de extração de PDFs! Tudo está pronto para uso.

### Passo 1: Clonar o Repositório
Abra o seu terminal e baixe o código:
```bash
git clone https://github.com/seu-usuario/rag-ufg-lpp.git
cd rag-ufg-lpp
```

### Passo 2: Configurar as Chaves
O sistema precisa se comunicar com as IAs para pensar. 
1. Crie um arquivo chamado `.env` na raiz do projeto.
2. Copie o conteúdo do arquivo `.env.example` para dentro dele.
3. Preencha as suas chaves reais (`OPENROUTER_API_KEY` e `OPENAI_API_KEY`).

### Passo 3: Ligar a Máquina (Docker)
O projeto roda totalmente isolado para não sujar o seu computador. Basta subir os containers:
```bash
make start
```
*(O comando acima irá baixar as imagens do Python e do Banco de Dados Vetorial e subir o servidor na sua máquina).*

### Passo 4: Fazer a sua Pergunta!
Agora que o motor está rodando, basta você abrir o terminal e fazer qualquer pergunta sobre as normas da UFG. O sistema vai pensar, pesquisar e te responder no próprio console:

```bash
make test-rag query="Segundo as normas, o que acontece se eu faltar muito nas disciplinas? Tem como eu ser jubilado do meu curso de graduação?"
```

> **Dica de Auditoria:** Toda vez que você ou alguém fizer uma pergunta, o sistema salvará a pergunta, os documentos encontrados e a resposta final dentro do arquivo secreto de logs em `data/logs/rag_queries.log`.
