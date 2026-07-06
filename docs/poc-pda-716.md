# POC PDA-716 — Fine-tuning para extração de condições em análises normativas

> **Status:** concluída — todos os experimentos executados, comparativo final fechado e recomendação definida (seção 8).
> **Autor:** Lucas Marujo · **Período:** 02/07/2026 a 05/07/2026

## 1. Resumo executivo

Testei se fine-tuning de um LLM consegue automatizar a extração das condições de aplicabilidade em análises normativas, usando o histórico real do time de consultoria. Ao longo de 8 etapas, treinei 6 versões de modelo/dataset, construí baselines sem treino e comparei com um modelo de API (Claude Haiku 4.5) em prompt enriquecido.

Três conclusões principais:

1. **Fine-tuning não entrega ganho sobre prompt enriquecido** na escala que temos disponível: o melhor modelo treinado fez F1 0.457, a API few-shot fez 0.500 — e um baseline de busca por similaridade, **sem nenhum treino, fez 0.559**.
2. **A descoberta mais valiosa da POC não foi um modelo, foi um dado**: as condições pertencem aos *requisitos* que a análise cobre, não à norma inteira. Quando incluí os textos dos requisitos no input, a tarefa saiu de "impossível" (concordância entre analistas de 0.081) para "determinística" (0.871). Sem essa correção, nenhuma abordagem — nem API — funcionava.
3. **Recomendo produtizar um motor de sugestão por retrieval** (buscar a análise histórica mais parecida e sugerir as condições dela), com régua de confiança pela similaridade. Nos ~25% de casos quase idênticos ao histórico, ele acerta F1 0.836; nos ~30% de casos genuinamente novos, nenhum método passou de 0.26 — ali o consultor continua indispensável.

## 2. Objetivo e critérios de aceite

A task pede para avaliar se fine-tuning é viável para automatizar/apoiar a extração de condições, com os seguintes critérios de aceite, que respondo um a um na seção 7:

- Base mínima de dados preparada para treinamento;
- Modelo testado com normas fora da base de treino;
- Resultados comparados com análises humanas;
- Indicação se fine-tuning entrega ganho relevante em relação a enriquecimento de prompt;
- Decisão final considerando qualidade, custo, esforço de atualização e necessidade de revisão humana.

## 3. Setup

| Item | Valor |
| --- | --- |
| Hardware | NVIDIA RTX 4050 Laptop (6 GB VRAM), WSL2 — zero custo de nuvem |
| Modelo local | Qwen3-0.6B, LoRA r=16 all-linear, bf16 |
| Modelo de API | Claude Haiku 4.5, few-shot com 3 vizinhos |
| Dados | Export do Mongo de produção: 18.543 análises com condições (histórico completo, não amostra) |
| Plataforma | CLI própria (`ft train`/`ft eval`), configs Hydra, MLflow, 122 testes unitários |

O formato de cada exemplo é chat: system prompt fixo, a norma (título + ementa) no user e o JSON de condições (itens de uma taxonomia fechada de 2.715 itens) no assistant. O split de teste é por hash do `norma_id` — verifiquei que nenhum texto de norma aparece dos dois lados.

## 4. A jornada, etapa por etapa

### Etapa 1 — Otimização de hardware

Antes de treinar, otimizei o uso da GPU. O smoke test rodava com batch 1 e ~20% de utilização. Três aprendizados que valem para qualquer treino local:

- **QLoRA 4-bit era overhead puro num modelo de 0.6B** (pesos bf16 = 1,2 GB, cabem folgados) — troquei para LoRA bf16 e ganhei ~2x;
- **Packing + batch maior** eliminaram ~80% de compute desperdiçado em padding;
- **A "memória compartilhada" da GPU é armadilha**: com batch 16, a VRAM transbordou silenciosamente para a RAM via PCIe (~12x mais lenta) e cada step passou de ~19s para 63s. Com batch 12 tudo coube na VRAM dedicada. Regra que adotei: memória compartilhada perto de zero = treino saudável.

Resultado: GPU a 75–100%, treino do v1 em ~2h.

### Etapa 2 — Fine-tuning direto (v1): o modelo não decora 2.715 itens

Treinei o primeiro modelo com norma → condições, direto. Curva de treino saudável (eval_loss 0.405, sem overfitting) e, no test set, **F1 0.236** — contra **0.0 do modelo base sem treino** (que inventa itens de taxonomia inexistentes). O fine-tuning claramente ensinou o formato e o vocabulário real, mas 0.236 não serve para produção.

Investiguei as 100 amostras e estratifiquei o recall pela frequência do item no treino: itens vistos 50+ vezes → 31% de recall; itens vistos menos de 10 vezes → **0%**. O modelo precisava decorar a taxonomia inteira e só aprendia a cabeça da distribuição. Os erros eram vizinhos semânticos (norma de fertilizantes → ele respondia "Rural > Agrícola" em vez de "Fertilizantes e afins"), o que me indicou o caminho: dar candidatos no prompt e transformar geração em seleção.

### Etapa 3 — Retrieval de candidatos: um erro e um acerto

**Errei primeiro**: casei a ementa da norma com os textos da taxonomia via embeddings (`multilingual-e5-small`). Recall@20 de **8%** — texto de taxonomia é curto e abstrato demais. Pior: injetar candidatos ruins no prompt derrubou o modelo de 0.236 para 0.013, porque ele obedeceu à lista errada (aderência 89%). Lição: *retrieval ruim é pior que nenhum retrieval*.

**Acertei depois**: buscar as análises históricas mais parecidas (kNN e5 + TF-IDF sobre ementas) e usar as condições delas como candidatos — recall de 71% com ~37 candidatos. Desse estudo saiu um baseline que mudou minha régua: **copiar as condições da norma mais parecida do histórico, sem nenhum treino, já fazia F1 0.161** — perto do meu modelo treinado.

### Etapa 4 — Modelo seletor (v2): validação enganosa

Retreinei com os candidatos no prompt (gold garantido no treino, retrieval real no teste, lista alfabética contra atalho posicional). O treino levou 19h29 e a validação parecia impecável: eval_loss 0.104, 96% de acurácia. No test set: **F1 0.215 — pior que o v1**.

A dissecação me ensinou duas coisas: a mecânica de seleção funcionou perfeitamente (100% de aderência à lista), mas os candidatos do teste eram ruins; e minha validação estava **inflada por quase-duplicatas** entre treino e validação — o modelo "acertava" por memória de vizinhança. Passei a desconfiar de qualquer métrica de validação e a decidir só pelo test set.

### Etapa 5 — Alvo agregado por norma (v3): empate com o baseline

Descobri que a mesma norma aparecia várias vezes no dataset com condições disjuntas, então reformulei o alvo para a união de todas as análises da norma. O v3 (treino de 1h08) fez **F1 0.194, empatado com o baseline gratuito (0.201)** no mesmo alvo. Também corrigi aqui dois bugs de avaliação: truncamento de geração em 512 tokens e uma amostra de teste enviesada (ordenada por id, pegava as normas mais antigas e pesadas).

### Etapa 6 — Forense completa dos dados: o problema não era o modelo

Parei de treinar e fui entender o dado de ponta a ponta, direto no export e depois no Mongo. Encontrei, com números:

1. **46% das análises vinculam 2+ normas e o export atribuía as condições do grupo inteiro à primeira norma** (`Normas[0]`) — rótulo estruturalmente errado para metade do dataset.
2. Mesmo nas análises de norma única, **duas análises da mesma norma concordavam em apenas F1 0.081** nos itens exatos.
3. O "teto humano" nessa métrica era **F1 0.065**: pontuei cada análise humana contra a união das demais análises da mesma norma. Ou seja: meus modelos (0.19–0.24) já estavam *acima* da consistência humana — a régua é que não media o que parecia medir. Criei a avaliação hierárquica (folha/pai/nível-2/raiz da taxonomia) para ter leitura honesta.

Conclusão da etapa: com input de título+ementa, a tarefa era **matematicamente impossível** no nível de item exato. Faltava contexto — e ele existia no banco.

### Etapa 7 — O breakthrough: requisitos são o contexto que faltava

Inspecionando o schema real do Mongo, encontrei o campo que o export original descartava: **`RequisitosIds`** — os requisitos legais (obrigações extraídas da norma) que cada análise cobre, com texto integral na coleção `requisitos`. A validação foi cirúrgica no caso da NR-38: a análise que marcou "Trabalho com veículo coletor-compactador" analisava requisitos sobre *coletores-compactadores*; a que marcou "Varrição" analisava requisitos de *varrição*. As análises "contraditórias" nunca foram contraditórias — cobriam requisitos diferentes.

Refiz o export com os requisitos (16.123 análises, 78% do total) e medi:

| Métrica | Sem requisitos | Com requisitos |
| --- | --- | --- |
| Concordância entre análises com o mesmo input | 0.081 | **0.871** |
| Baseline kNN copiar-vizinho (split por norma) | 0.178 | **0.727** |

A tarefa virou determinística e aprendível. E o baseline kNN virou um sistema quase utilizável — e a nova régua a bater.

### Etapa 8 — Rodada final: fine-tuning × kNN × API

Com o dataset correto (v5: norma + requisitos → condições; v6: idem + candidatos kNN no prompt), treinei as versões finais e rodei o comparativo com API, tudo nas **mesmas 200 amostras de teste e mesma métrica**:

| Sistema | Treino | F1 item exato | F1 raiz | Match exato |
| --- | --- | --- | --- | --- |
| **kNN copiar-vizinho (TF-IDF)** | nenhum | **0.559** | — | **47,5%** |
| Híbrido (kNN + Haiku nos casos novos) | nenhum | 0.564 | — | — |
| API Claude Haiku 4.5 few-shot | nenhum | 0.500 | 0.732 | 36,0% |
| Fine-tuned v6 (0.6B, com candidatos) | 12h30 GPU | 0.457 | 0.739 | 42,5% |
| Fine-tuned v5 (0.6B, sem candidatos) | ~7h GPU | 0.391 | 0.711 | 36,5% |
| *Teto do retrieval (recall dos candidatos)* | — | *0.737* | — | — |

Estratificando pela similaridade do caso de teste com o histórico:

| Similaridade do vizinho mais próximo | n | kNN | Haiku | Fine-tuned v6 |
| --- | --- | --- | --- | --- |
| ≥ 0.8 (quase-duplicata) | 50 | **0.836** | 0.656 | 0.615 |
| 0.5–0.8 | 90 | **0.633** | 0.573 | 0.570 |
| < 0.5 (caso genuinamente novo) | 60 | 0.212 | **0.256** | 0.135 |

O que eu leio disso: o histórico com requisitos é tão repetitivo (versões estaduais da mesma lei, NRs recorrentes) que copiar a análise mais parecida é imbatível em custo-benefício. O fine-tuned encontra o ramo certo da taxonomia (raiz 0.739) mas tropeça na string exata do item — e mesmo com candidatos para selecionar, o 0.6B não supera a cópia literal. A API só ganha na cauda nova, e por pouco.

## 5. O que ficou de fora e por quê

- **Qwen3-1.7B/4B local**: 30–40h de treino nesta GPU para disputar com um baseline que custa zero. O padrão dos dados (cópia vence) me diz que o ganho não paga o custo. Descartei conscientemente.
- **Sonnet/modelos maiores de API no fluxo todo**: o Haiku já mostrou o formato da curva; um modelo maior faria sentido apenas na fatia <0.5, como experimento futuro (~US$1).
- **22% das análises sem requisitos**: ficaram fora do v5/v6; entrariam no fluxo humano.

## 6. Custo, esforço e manutenção

- **Infra da POC**: zero nuvem — tudo numa GPU de notebook de 6 GB. Treinos: v1 2h, v2 19h, v3 1h, v5 7h, v6 12h30. API: ~US$1 por rodada de 200 avaliações.
- **Da solução recomendada (retrieval)**: sem GPU, sem treino; indexação TF-IDF/embeddings de 16k análises roda em minutos; manutenção = reindexar análises novas (incremental, barato). Sem dependência de fornecedor.
- **Do fine-tuning (se fosse adiante)**: retreino periódico a cada atualização relevante do histórico ou da taxonomia, avaliação de regressão a cada versão, e hardware dedicado — esforço contínuo que os números não justificam.
- **Requisito de implantação**: o pipeline de dados precisa carregar o elo `análise → requisitos → condições` (corrigido em `scripts/export_greenlegis_raw_v2.py`).

## 7. Resposta aos critérios de aceite

1. **Base mínima preparada** ✅ — 18,5k análises exportadas; 6 versões de dataset com limpeza, dedupe e split por norma sem vazamento (verificado). A versão definitiva (v5/v6) tem 14.194 exemplos de treino e 1.702 de teste.
2. **Testado com normas fora do treino** ✅ — split por hash de `norma_id`; 200 amostras de teste na rodada final.
3. **Comparado com análises humanas** ✅ — F1 por item exato + hierárquico contra as análises reais; medi inclusive o teto de consistência entre os próprios analistas (0.871 com requisitos no input; 0.081 sem).
4. **Fine-tuning entrega ganho sobre prompt enriquecido?** ❌ — **Não.** API few-shot (0.500) > fine-tuned (0.457), e ambos perdem para busca por similaridade sem treino (0.559).
5. **Decisão considerando qualidade, custo, atualização e revisão humana** ✅ — seção 8.

## 8. Recomendação final

**Não seguir com fine-tuning. Produtizar um motor de sugestão por retrieval sobre o histórico de análises (com requisitos), com régua de confiança pela similaridade:**

- **Similaridade ≥ 0.8** (~25% dos casos): sugerir automaticamente as condições da análise vizinha (F1 0.836) — revisão humana rápida;
- **0.5–0.8** (~45%): apresentar os itens dos vizinhos como *checklist de candidatos* (top-9 cobre 73,7% dos itens corretos) — o consultor seleciona em vez de digitar;
- **< 0.5** (~30%): análise humana normal; opcionalmente sugestões de modelo de API (ganho modesto, 0.256).

A necessidade de revisão humana permanece em todos os níveis — o sistema acelera o consultor, não o substitui. O maior ativo descoberto pela POC é que **o histórico de análises, quando ligado aos requisitos, já contém a maior parte das respostas**.
