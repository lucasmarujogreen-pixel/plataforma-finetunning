# Comparativo das abordagens — PDA-716

> Documento de referência: como cada uma das quatro abordagens testadas (fine-tuning generativo, reranker, kNN, API few-shot) funciona por dentro — que dados usa, como valida, se respeita o critério de aceite da POC, e quanto custa. Complementa `poc-pda-716.md` (jornada e resultados) e `proximos-passos-melhoria-modelo.md` (hipóteses de melhoria).

## Resumo comparativo

| | Fine-tuning (Qwen3-0.6B) | Reranker (`bge-reranker-v2-m3`) | kNN copiar-vizinho | API few-shot (Sonnet 5) |
| --- | --- | --- | --- | --- |
| Treina pesos? | Sim (LoRA) | Sim (LoRA) | Não | Não |
| Dataset de treino | `train_v6.jsonl` (14.194) | `train_v5.jsonl` (14.194) | N/A (só índice) | N/A (só índice) |
| Onde valida de verdade | `test_v6.jsonl` (200 amostras fixas) | idem | idem | idem |
| Custo | GPU local, ~12h30 | GPU local, ~12h49 | Zero (minutos de indexação) | ~$1-2 / 200 chamadas |
| F1 item exato | 0.457 | 0.370 | 0.559 | **0.573** |
| Respeita o critério de aceite? | Sim (mas falha no resultado) | Sim | Sim | Sim |

Todas as quatro são avaliadas exatamente da mesma forma: JSON de condições gerado/copiado vs. JSON real do analista humano, nas mesmas 200 amostras de teste (`datasets/raw/greenlegis_condicoes_test_v6.jsonl`), split por hash de `norma_id` (nenhuma norma do teste aparece em nenhum treino/índice). Isso é o que torna os números comparáveis entre si.

---

## 1. Fine-tuning (Qwen3-0.6B causal-LM)

**Dataset**: `datasets/raw/greenlegis_condicoes_train_v6.jsonl` (14.194 exemplos) / `..._test_v6.jsonl` (200 amostras fixas). Cada exemplo tem 3 mensagens:

- **system**: prompt fixo com as instruções de formato.
- **user**: `Norma:` (tipo + número + data) + `Ementa:` + `Requisitos analisados:` (texto integral de cada requisito legal que a análise cobriu — a descoberta da Etapa 7; sem isso a tarefa é matematicamente impossível, concordância entre analistas de 0.081) +, na v6, um bloco `Candidatos da taxonomia:` com ~20–40 itens trazidos por kNN.
- **assistant**: o JSON real `{"condicoes": [{"vinculo": N, "itens": [...]}]}` que o analista humano registrou.

**Como treina**: SFT (teacher forcing) — o modelo aprende a prever o JSON do `assistant` dado o resto do contexto. LoRA r=16 em todas as camadas lineares, bf16.

**Como valida**: durante o treino, um split de `validation` interno gera `eval_loss` (usado por early stopping). A Etapa 4 da POC descobriu que essa validação estava **inflada por quase-duplicatas** entre treino e validação (o modelo "acertava" por memória de vizinhança, não aprendizado real) — por isso a decisão final nunca usa essa métrica, só o **test set fixo de 200 amostras**, separado por hash do `norma_id`.

**Respeita o critério de aceite?** Sim, no processo: base mínima preparada, testado fora do treino, comparado com análises humanas reais (inclusive medindo o teto de concordância entre analistas, 0.871 com requisitos). Falha no critério mais importante — "fine-tuning entrega ganho sobre prompt enriquecido?" — porque a resposta medida é **não** (0.457 vs. 0.500 do Haiku e 0.559 do kNN).

**Custo**: só GPU local — v5 ~7h, v6 ~12h30. Zero nuvem, mas ~12h de notebook ocupado por versão treinada.

---

## 2. Reranker (cross-encoder `bge-reranker-v2-m3`)

**Dataset**: `scripts/build_reranker_dataset.py` monta pares a partir do **mesmo `train_v5.jsonl`** (nunca toca no `test_v6`). Para cada análise, busca candidatos via kNN (e5 + TF-IDF sobre norma+ementa+requisitos) garantindo que o item certo (gold) sempre esteja entre eles, e monta um registro listwise:

```json
{"query": "<norma + requisitos>", "docs": ["<candidato 1>", "..."], "labels": [1.0, 0.0, ...]}
```

Split interno pair-train/pair-validation (hash salgado próprio, independente do split oficial) → 11.468 registros base (+4.453 de oversampling de itens raros da taxonomia = 15.921 efetivos) / 613 de validação.

**Como treina**: `CrossEncoderTrainer` com `LambdaLoss` (loss de ranking listwise) — o modelo aprende a pontuar `(query, candidato)` mais alto quando o candidato é o item certo. LoRA no classificador + camadas lineares.

**Como valida**: mesma dinâmica do fine-tuning — tem um `eval_loss` interno (613 exemplos) usado durante o treino, mas a avaliação que conta de fato é `ft evaluate-reranker`, que roda contra o **mesmo test set de 200 amostras** de todo mundo.

**Respeita o critério de aceite?** Sim, metodologicamente idêntico aos outros. Resultado final (após correções de causa-raiz — oversampling de itens raros, corte de candidatos por query): F1 0.370 — pior que todos os outros sistemas. Diagnóstico de causa raiz (ver `proximos-passos-melhoria-modelo.md` seção 2.2c): cauda longa extrema na taxonomia (53,9% dos itens vistos como positivo só 1x no treino) — o gargalo é estrutural, não de arquitetura.

**Custo**: GPU local, ~12h49 por época. Zero nuvem.

---

## 3. kNN (copiar-vizinho)

**Sem treino.** Constrói um índice (embeddings `multilingual-e5-small` + TF-IDF) sobre o **mesmo pool histórico** (`train_v5`/`v6`, análises já resolvidas por humanos). Para cada exemplo de teste, acha a análise histórica mais parecida e **copia literalmente as condições dela** como resposta.

**Dados usados**: o texto (norma+ementa+requisitos) de cada análise histórica só serve para indexação/busca — a "resposta" copiada é a condição real que o analista humano já preencheu naquele caso parecido.

**Como valida**: não existe fase de treino, então não há validação no sentido de ML — é medido direto contra o mesmo test set de 200 amostras.

**Respeita o critério de aceite?** Sim, plenamente — mesmo protocolo, mesmo split.

**Custo**: **zero GPU, zero treino, zero chamada de API**. Só indexar (minutos, uma vez) e buscar o vizinho mais próximo (milissegundos por consulta). É o "chão" contra o qual todo o resto precisa se justificar — daí ser a régua oficial da recomendação da POC (seção 8 de `poc-pda-716.md`).

---

## 4. API few-shot (testado com Haiku 4.5 e depois Sonnet 5)

**Dados usados em cada chamada**: system prompt (igual aos outros) + **3 exemplos few-shot** (análises históricas parecidas, buscadas por TF-IDF no mesmo pool `train_v6`, cada uma com o texto da norma+requisitos **e a condição real que o analista determinou**) + a mensagem do exemplo de teste (norma+ementa+requisitos — **sem a resposta**). O modelo gera o JSON de condições olhando só isso.

**Não há treino** — é in-context learning: o modelo "aprende" o padrão a partir dos 3 exemplos mostrados na hora da chamada, não de pesos ajustados.

**Os dados de teste são diferentes dos usados como referência?** O modelo nunca vê a resposta certa do exemplo de teste (isso só entra depois, para pontuar). Os exemplos few-shot vêm do mesmo pool histórico que o kNN usa — a diferença é que o kNN copia 1 vizinho inteiro e a API usa 3 vizinhos como demonstração e **gera** uma resposta nova, em vez de copiar.

**Respeita o critério de aceite?** Sim — mesmíssimo test set de 200 amostras, mesmo split por `norma_id`, mesma métrica hierárquica.

**Custo**: só chamada de API, sem GPU/infra própria — ~$1 (Haiku 4.5) a ~$1,35–2,00 (Sonnet 5) pelas 200 chamadas do teste. Em produção seria custo por análise processada, contínuo (diferente do kNN, que depois de indexado é grátis).

### Isso reflete um cenário real de produção?

Sim. O input que a API recebe (norma + ementa + requisitos analisados) é exatamente o que já existiria no momento em que um consultor precisa determinar as condições — nada do que está sendo predito (a condição) entra como dado de entrada. O fluxo simulado é: *"aqui está um caso novo (dado que já existe no sistema), aqui estão 3 casos parecidos já resolvidos (com a condição real que foi determinada), agora determine a condição do caso novo"* — que é literalmente a tarefa que o consultor faz hoje. O teste já reflete fielmente o cenário de produção, não é uma simplificação artificial.

### Como a avaliação é feita, e por que o Sonnet 5 sai na frente

Mesma métrica hierárquica de F1 (item exato + por nível da taxonomia: folha/pai/nível-2/raiz) comparando o JSON gerado contra o JSON real do analista, nas mesmas 200 amostras (`extract_item_ids`/`extract_paths` em `src/finetuning/evaluation/hierarchical_f1.py`).

O Sonnet 5 (F1 0.573) supera o Haiku (0.500) e o kNN (0.559) porque é um modelo de raciocínio mais capaz — e essa tarefa depende de entender nuances finas entre itens de taxonomia quase-idênticos (a mesma dificuldade que travou o reranker: variantes que só diferem, por exemplo, em data de licenciamento). Diferente do reranker/fine-tuned local, o Sonnet não depende de ter visto muitos exemplos daquele item específico durante um treino — ele generaliza a partir de 3 exemplos em contexto mais o conhecimento amplo já embutido no modelo, exatamente onde um modelo pequeno/local (0.6B ou o reranker de 568M) trava por escassez de dado por classe.

**Estratificação por similaridade (10/07/2026)** — ver detalhe completo na Etapa 10 de `poc-pda-716.md`: o Sonnet 5 não vence o kNN em nenhuma faixa individual de similaridade (perde nos casos de alta similaridade, onde copiar já é quase ótimo), mas é consistentemente melhor que Haiku e fine-tuned em quase todas as faixas — o suficiente para virar o resultado agregado. O ganho do Sonnet 5 não vem de "resolver a cauda difícil" (que era a única vantagem do Haiku) — vem de ser um modelo melhor de ponta a ponta. Isso sugere que ele funciona melhor como **sistema principal** do que como complemento pontual do retrieval, embora o kNN continue mais barato e ainda vença nos casos de alta similaridade — um híbrido (kNN quando há vizinho muito parecido, Sonnet 5 no resto) provavelmente supera qualquer um isolado.
