# Arquitetura e Especificações Técnicas do Sistema: RAG LPP (UFG)

Este documento centraliza as abordagens arquiteturais, bibliotecas e estratégias adotadas para garantir alta performance, resiliência na construção da base de dados (ETL) e ausência de alucinações na orquestração de resposta (RAG).

---

## 1. Pipeline de Dados (ETL)

> Todo o processamento é determinístico. Não há uso de Agentes de IA Generativa nesta camada, apenas Modelos de Visão e Embeddings pré-treinados.

### Mapeamento e Estratégias (ETL)

* **Agente Determinístico: Ingestor & Download** (`src/ETL/ingestor.py` / `src/tools/download.py`)
  * **Bypass de Proteções WAF:** Utilização do `cloudscraper` para contornar barreiras Cloudflare (Erro 403) nos servidores da UFG (Cercomp). O scraper emula assinaturas de browsers reais e resolve desafios Javascript nativamente.
  * **Coleta via Dados Abertos:** Uso da API CKAN do Portal de Dados da UFG para mapeamento automático de URLs autênticas de PDFs (evitando scraping HTML frágil).
  * **Manifesto Stateful:** Arquitetura stateful focada em retentativas usando o `to_download.json`. O manifesto cataloga cada PDF (`pending`, `downloaded`, `failed`), evitando download duplicado e garantindo economia de rede na retomada.

* **Agente Determinístico: Parser & OCR Semântico** (`src/ETL/parser.py`)
  * **Motor de Extração Docling:** Substitui ferramentas tradicionais de extração linear. Utiliza IA de Visão para interpretar layouts de duas colunas nativamente e estruturar tabelas de forma fidedigna.
  * **Multiprocessamento e Controle de VRAM:** 
    * O processamento roda paralelamente, limitado por `MAX_PARALLEL_WORKERS`. (Ex: 3 workers consomem a GPU de 6GB, pois o *TableFormer* puxa até 1.5GB cada).
    * **Contexto Spawn:** Utiliza `multiprocessing.get_context('spawn')` estritamente. O método `fork` causaria vazamento fatal de VRAM no PyTorch.
    * **Pre-flight Cleanup:** O gerenciador invoca `gc.collect()` e `torch.cuda.empty_cache()` sempre antes de instanciar novas tarefas de parsing.
  * **Normalização de Hierarquia (`text_normalizer.py`):** Limpeza textual RegEx para converter identificadores como *RESOLUÇÃO CEPEC* em marcações Markdown (`H1` a `H5`), removendo ruídos de quebra de página (cabeçalhos órfãos).

* **Agente Determinístico: Chunker** (`src/ETL/chunker.py`)
  * O fatiamento foca em retenção do raciocínio institucional, possuindo 3 estratégias modulares:
    1. **Markdown Structural Split:** Estratégia principal de corte baseada nas seções semânticas (`#`, `##`). Previne que uma norma seja rasgada no meio da frase, mantendo a unidade de sentido.
    2. **Recursive Character Split:** Rede de segurança para blocos massivos sem formatação (`RECURSIVE_CHUNK_SIZE` e `OVERLAP`).
  * **Injeção de Breadcrumbs (Contextual Retrieval):** Injeta um "rastreador de tópico" na 1ª linha do bloco (Ex: `[Contexto: edital_proec > VAGAS > Distribuição]`) para não isolar fragmentos perdidos.
  * **Table Unrolling (Prevenção de Amnésia de Cabeçalho):** Ferramenta customizada (`table_unroller.py`) que converte tabelas Markdown em texto linear chave-valor (`Campo: X. Descrição: Y`), permitindo que a busca vetorial entenda colunas mesmo puxando apenas a última linha da tabela.

* **Agente Determinístico: Vetorizador** (`src/ETL/vetorizador.py`)
  * Gera embeddings densos (BAAI/bge-m3) e esparsos (BM25 via FastEmbed) para cada chunk estruturado.
  * Indexa no Qdrant em modo **Híbrido** (Dense + Sparse) com UUIDs determinísticos.
  * Suporta "Resume-friendly" (processed_files.json) para continuar de onde parou.
  * Otimização de GPU iterativa (`torch.cuda.empty_cache`) para rodar os 30 mil chunks no BGE-M3 sem colapsar a placa doméstica.

---

## 2. Pipeline de Resposta (RAG Multi-Agente)

> Orquestrador síncrono com Loop de Memória Persistente e Incremental Reasoning.

### Fluxo de Execução

```
Pergunta → Planner → Retriever → Synthesizer → Judge
                ↑                              │
                └─── feedback (Repescagem) ────┘
```

### Mapeamento (Agentes)

* **Orquestrador Principal** (`src/RAG/orchestrator.py`)
  * Máquina de estados (`RAGState`) que trafega a Pergunta, o Plano, e o Acúmulo de Contexto.
  * **Observabilidade**: O root logger força (`force=True`) a gravação de auditoria permanente no arquivo `data/logs/rag_queries.log`.
  * **Memória Persistente**: A variável `all_raw_chunks` acumula documentos recuperados entre retentativas para evitar amnésia de contexto.

* **Agente: Planner** (`src/RAG/planner.py`) | *Modelo: openai/gpt-5.4-nano*
  * Decompõe perguntas complexas em no máximo 2 queries otimizadas tecnicamente + variações semânticas.
  * Tradução de jargões: (ex: "faltas" → "frequência mínima").
  * **Inteligência de Repescagem**: Ao receber feedback de falha do Juiz, foca **exclusivamente** na criação de queries para a parte da pergunta não respondida, ignorando o restante.

* **Agente: Retriever** (`src/RAG/orchestrator.py` → `RetrieverAgent`)
  * Acumula vetores das chamadas iterativas e deduplica todo o pool via hash de conteúdo.
  * **Reranking Global** de todas as subqueries contra a pergunta original.
  * Parâmetros brutais (`RETRIEVAL_LIMIT=100`, `PREFETCH_LIMIT=300`) garantem recuperação profunda para queries complexas.

* **Skill: Reranker Cross-Encoder** (`src/RAG/compactador.py`) | *Modelo: BAAI/bge-reranker-v2-m3*
  * Avalia a relevância entre pergunta/resposta e seleciona os `RERANKER_TOP_K` absolutos.
  * **Otimização Crítica (Anti-OOM):** Executa o predict em lotes minúsculos (`batch_size=4`) e uso intensivo de `gc.collect()` para impedir estouro da placa de vídeo ao confrontar centenas de documentos densos de uma vez.

* **Agente: Sintetizador** (`src/RAG/sintetizador.py`) | *Modelo: openai/gpt-5.4-nano*
  * Recebe os pedaços brutos do Reranker (preservando perfeitamente métricas, artigos e datas).
  * **Diretriz de Ouro**: "Zero Interação". O agente age como oráculo oficial, nunca como chatbot prestativo, sendo proibido de fazer perguntas de follow-up.
  * **Incremental Reasoning**: Quando convocado pelo Juiz para repescagem, recebe o Rascunho Anterior + Crítica + Novos Textos, e apenas complementa/corrige o rascunho em vez de escrever do zero.

* **Agente: LLM-as-a-Judge** (`src/RAG/llm_as_judge.py`) | *Modelo: anthropic/claude-sonnet-4.6*
  * Auditor final rigoroso contra alucinações, omissões de informações chave ou respostas evasivas.
  * Aprovando a resposta, o loop encerra. Se reprovar, passa o motivo cirúrgico de volta ao Planner.

---

## 3. Funil de Recuperação de Informação (Qdrant)

| Etapa | Parâmetro | Valor | Descrição |
|---|---|---|---|
| **Prefetch** | `PREFETCH_LIMIT` | 300 | Pré-seleciona candidatos do motor denso/esparso antes da fusão |
| **Fusão RRF** | — | ~600 | Agrupa candidatos das buscas Dense e Sparse |
| **Retrieval** | `RETRIEVAL_LIMIT` | 100 | Chunks híbridos resultantes que vão pra memória, por busca |
| **Pool Bruto** | — | ~600+ | Volume máximo por execução complexa |
| **Deduplicação** | — | ~400 | Chunks unificados e limpos para envio à Placa de Vídeo |
| **Reranking Global** | `RERANKER_TOP_K` | 15 | Sobreviventes finais selecionados com lupa pelo Cross-Encoder |

---

## 4. Configurações (.env) Referência 2026

| Variável | Valor | Propósito |
|---|---|---|
| `LLM_PLANNER` | `openai/gpt-5.4-nano` | Cérebro analítico veloz e barato |
| `LLM_SINTETIZADOR_FINAL` | `openai/gpt-5.4-nano` | Capacidade de redigir textos formais/jurídicos |
| `LLM_JUDGE` | `anthropic/claude-sonnet-4.6` | Alta rigidez cognitiva contra alucinações sistêmicas |
| `MAX_RETRIES` | `2` | Total de repescagens da Memória Persistente |
| `RERANKER_TOP_K` | `15` | Documentos entregues à síntese final |
| `RETRIEVAL_LIMIT` | `100` | Amplitude da malha fina de busca primária |
| `MAX_PARALLEL_WORKERS` | `2` | Processamento seguro de ETL para evitar sobrecarga de VRAM |

---

## 5. Ferramentas Compartilhadas (Tools / Skills)

> Utilitários globais invocados na pasta `src/tools/`

* **`download.py`**: Motor de extração HTTP (`cloudscraper`) imune a soft-blocks do WAF.
* **`pdf_extractor.py`**: Interface global do Docling (GPU-accelerated).
* **`text_normalizer.py`**: Sanetizador de Markdown governamental.
* **`table_unroller.py`**: Serialização linear de tabelas anti-amnésia em janelas de contexto limitadas.
* **`dbsearch.py`**: Cliente Singleton do Qdrant com fusão nativa.
