# Próximos Passos — Melhoria do Modelo (caminho Fine-tuning)

> **Premissa deste documento:** a [POC PDA-716](poc-pda-716.md) concluiu, com dados, que fine-tuning **não** supera o baseline de retrieval (kNN copiar-vizinho, F1 0.559) nem a API few-shot (F1 0.500) — o melhor fine-tuned (v6) fez F1 0.457. A recomendação oficial da POC é produtizar o motor de retrieval, não o fine-tuning.
>
> Este documento existe para responder a uma pergunta hipotética: **se, mesmo assim, o time decidir seguir pelo caminho de fine-tuning**, quais são as alavancas reais de melhoria e o roadmap para perseguir ganho de qualidade. Todo item abaixo deve ser validado contra os números da POC antes de virar investimento — nenhuma melhoria aqui tem retorno garantido, é hipótese a testar.

## 0. Escalar o modelo em nuvem resolve o problema? (leitura crítica)

Antes de qualquer investimento em nuvem, vale confrontar a hipótese "modelo maior resolve" com um dado que a própria POC já gerou e que normalmente passa despercebido: **o Claude Haiku 4.5 já é, de longe, um modelo muito mais capaz que qualquer coisa que o time treinaria localmente ou em nuvem nesta escala de projeto** — e mesmo assim ele **perdeu para o baseline gratuito de kNN** (F1 0.500 vs. 0.559) e teve **F1 raiz praticamente empatado com o fine-tuned de 0.6B** (0.732 vs. 0.739 — o modelo pequeno treinado localmente ficou até levemente à frente nesse recorte).

Isso é evidência direta de que **o gargalo não é capacidade/tamanho de modelo** — se fosse, o Haiku teria disparado na frente. O que limita todos os métodos é (a) o teto de recall do retrieval de candidatos (0.737) e (b) a natureza do problema em si: o histórico é tão repetitivo que copiar o vizinho mais parecido já é quase ótimo nos casos não-novos. Um modelo maior treinado na nuvem **herda exatamente essas duas limitações** — ele não vê candidatos melhores nem reduz a vantagem estrutural da cópia.

**Onde um modelo melhor plausivelmente ajudaria de fato**: só na fatia de casos genuinamente novos (similaridade < 0.5, ~30% dos casos), onde o Haiku (0.256) já supera o fine-tuned 0.6B (0.135) — é o único recorte da POC inteira onde "modelo mais forte" mostrou vantagem real. Fazendo as contas com os números já medidos na POC (25% × 0.836 + 45% × 0.633 + 30% × melhor-modelo-nessa-fatia), o teto de ganho de **só trocar o modelo por um maior, mantendo a mesma abordagem de geração**, fica em torno de **F1 0.57–0.60** no melhor cenário otimista — uma melhora de ~2 a 4 pontos sobre o kNN gratuito (0.559), não uma virada de jogo. E isso pressupõe que o modelo maior generativo bata o Haiku nessa fatia difícil, o que não é garantido.

**Conclusão prática:** a chance de gastar dinheiro de nuvem e não sair de um resultado "ligeiramente melhor que grátis" é alta **se a única mudança for o tamanho do modelo**. Vale a pena gastar em nuvem só se o dinheiro for para testar, ao mesmo tempo, uma mudança de abordagem que ataque o teto estrutural — não apenas escala (ver seções 2.1 e 2.3): reformular como seleção/classificação, e/ou melhorar o retrieval de candidatos. Essas duas coisas, aliás, **podem ser testadas localmente, sem nuvem** (um reranker como `bge-reranker-v2-m3` tem ~568M parâmetros — cabe tranquilamente nos 6GB da RTX 4050 atual). Recomendação: gastar cloud compute **depois** de já ter evidência local de que a reformulação ajuda, e então usá-la para escalar especificamente a fatia <0.5 — não para repetir o experimento generativo em tamanho maior.

## 1. Diagnóstico — onde o modelo atual (v6) perde pontos

Da Etapa 8 da POC:

| Sintoma | Evidência |
| --- | --- |
| Acha o ramo certo da taxonomia, erra a folha | F1 raiz 0.739 vs. F1 item exato 0.457 — a hierarquia é aprendida, a string exata não |
| Não supera cópia literal do vizinho mais parecido | kNN 0.559 > FT 0.457 mesmo com os mesmos candidatos disponíveis |
| Perde feio em casos quase-duplicados do histórico | similaridade ≥ 0.8: kNN 0.836 vs. FT 0.615 — o modelo "reinterpreta" onde deveria só copiar |
| Only ganha relativamente na cauda nova | similaridade < 0.5: FT 0.135 vs. Haiku 0.256 — nem aí o modelo pequeno se destaca |
| Teto de recall do retrieval já limita tudo | candidatos cobrem no máximo 0.737 de F1 — nenhum método consegue passar disso sem melhorar o retrieval |

Conclusão do diagnóstico: **o gargalo não é só o tamanho do modelo**. Há pelo menos três gargalos independentes — (a) o modelo generativo erra a string exata mesmo sabendo o ramo certo, (b) ele não aproveita bem o caso trivial de "copiar quando é praticamente igual", (c) o teto do retrieval (0.737) já limita o resultado antes mesmo do modelo entrar em ação. Escalar o modelo ataca só uma dessas causas.

## 2. Eixos de melhoria

### 2.1 Reformular a tarefa: de geração para seleção/ranqueamento

Hoje o v6 gera texto livre a partir de uma lista de candidatos no prompt — e erra a string exata mesmo escolhendo o ramo certo. Duas alternativas mais robustas a testar antes de qualquer scale-up:

- **Decodificação restrita (constrained decoding)**: forçar o modelo a só gerar itens que existem literalmente na lista de candidatos (grammar-based generation, ex. bibliotecas como `outlines` ou `lm-format-enforcer`, ou logit masking manual sobre os IDs válidos). Elimina por construção o erro de "vizinho semântico errado" que já apareceu na Etapa 2 e reaparece aqui.
- **Reformular como classificação multi-label sobre o conjunto de candidatos** (ex. cross-encoder/reranker: para cada par `(input, candidato)` prever prob. de pertencer ao conjunto de condições) em vez de causal LM. Modelos de reranking (ex. família `bge-reranker`, ou até um classificador leve tipo `DeBERTa` fine-tunado) tendem a ser mais baratos de treinar que um LLM generativo e mais adequados a "escolher da lista" do que a "escrever a resposta".

Isso é mudança de arquitetura de solução, não só de escala — e é o item com maior chance de fechar o gap para o kNN, porque ataca diretamente o sintoma (a) do diagnóstico.

### 2.2 Escala de modelo (Qwen3-1.7B / 4B, ou outra família)

A POC descartou conscientemente 1.7B/4B por custo de tempo local (30–40h numa GPU de 6GB para um baseline que "custa zero"). Se o fine-tuning for adiante, faz sentido reabrir esse experimento **em nuvem** (ver [pesquisa de plataformas](pesquisa-plataformas-treinamento-cloud.md)), mas com critério de corte claro (ver também seção 0 — a expectativa de ganho real com escala pura é modesta):

- Rodar 1.7B e 4B nas mesmas 200 amostras de teste, mesmo protocolo de avaliação hierárquica da POC.
- Critério de "vale a pena": só justifica manter fine-tuning em produção se o modelo maior superar o kNN (0.559) por margem que compense o custo de retreino periódico — não basta empatar.
- Testar especificamente no fatiamento por similaridade < 0.5 (a única fatia onde há espaço real de ganho, ver seção 0) — não gastar o orçamento de nuvem rodando o full test set se o objetivo é só validar essa hipótese.

**Panorama de modelos além do Qwen3 (julho/2026), caso valha testar outra família:**

| Modelo | Porte / licença | Por que consideraria |
| --- | --- | --- |
| Qwen3.5 | Denso e MoE, Apache 2.0, 201 idiomas | Já é a família usada; a atualização 3.5 mantém ecossistema de fine-tuning maduro (mesmas ferramentas HF/TRL/PEFT) |
| Gemma 3 / Gemma 4 (26B-A4B) | Denso pequeno (1B–27B) e MoE, licença Gemma | Boa relação custo/qualidade para rodar local (Gemma 3 4B cabe em ~4GB), forte em multilíngue |
| Llama 4 Scout/Maverick | MoE (109B total, 17B ativos), licença Llama | Contexto longo, mas historicamente atrás de Qwen/Gemma em benchmarks de raciocínio — menor prioridade aqui |
| DeepSeek V3.2 / R1 | MoE grande, licença MIT | Forte em raciocínio/matemática; overkill de infraestrutura para uma tarefa de classificação/extração como esta |
| GLM-5 / 5.1 | Licença MIT | Ponto forte é código, não é o diferencial que a tarefa precisa |
| Tucano (PT-BR) | Modelo pequeno pré-treinado especificamente em português | Vale um experimento de baixo custo: fluência nativa em português jurídico/normativo pode ajudar mais que escala bruta em inglês-cêntrico, mas é bem menor/menos capaz em geral — testar antes de descartar |

**Leitura**: nenhuma dessas famílias ataca o gargalo estrutural identificado na seção 0 (teto de retrieval + natureza repetitiva do histórico). A troca de família só faz sentido combinada com a reformulação da seção 2.1, ou como teste barato e pontual (ex. Tucano, que é pequeno e roda local) — não como justificativa isolada para migrar para nuvem.

### 2.2b Fine-tuning de reranker/embeddings (alternativa mais barata que escalar o LLM gerador)

Dado que o problema tem cara de "ranquear/selecionar de uma lista fechada" mais do que "gerar texto livre", vale considerar treinar um modelo dedicado a isso em vez de um LLM generativo maior — é mais barato e ataca diretamente o teto de retrieval (seção 2.3):

- **Rerankers cross-encoder multilíngues** (ex. `BGE-reranker-v2-m3`, `Jina-reranker-v2-multilingual`) são modelos pequenos (~300M–600M parâmetros) que já rodam nos 6GB da GPU local atual, sem precisar de nuvem. Fine-tunados no par `(requisito, condição-candidata)` do histórico, atacam diretamente o problema de "achar o ramo certo mas errar a string exata" — o reranker escolhe entre strings que já existem, não gera texto novo.
- Isso também melhora o teto de recall do kNN (seção 2.3), porque o mesmo modelo pode reordenar os candidatos do retrieval, não só os do fine-tuned.
- **Sequência de menor risco**: testar isso localmente primeiro (custo ≈ zero) e só depois decidir se compensa somar um LLM maior em nuvem para a fatia <0.5 que sobrar.

**Resultado real (09/07/2026)**: implementado e treinado de ponta a ponta. F1 item exato final: **0.351** — não superou kNN (0.559), Haiku (0.500) nem o fine-tuned v6 (0.457). Escrita completa da implementação, dos bugs reais encontrados no caminho e do diagnóstico qualitativo: ver **Etapa 9 em `poc-pda-716.md`**. Decisão: não investir mais GPU nessa configuração específica sem antes corrigir os pontos identificados na investigação de causa-raiz abaixo.

### 2.2c Investigação de causa-raiz: por que várias abordagens diferentes falham do mesmo jeito (09/07/2026)

18.543 análises parece muita informação, então o resultado consistentemente abaixo do baseline gratuito, em três arquiteturas completamente diferentes (LLM generativo, API few-shot, reranker cross-encoder), levanta a pergunta certa: é coincidência, ou tem uma causa estrutural comum? Testei três hipóteses concretas com medição direta nos dados reais, não especulação.

**Hipótese 1 — truncamento (REFUTADA como causa principal).** O cross-encoder usava `max_length=512` tokens; se a query (norma + requisitos, que pode ser longa) empurrasse o candidato pra fora da janela, o modelo nunca veria a parte que distingue variantes quase-idênticas da taxonomia. Tokenizei uma amostra real de 30 queries × seus candidatos (633 pares): a query sozinha chega a 524 tokens no pior caso (p50=120, p90=225), e apenas **6,3% dos pares** excedem 512 tokens no total. Como o tokenizer trunca preferencialmente a sequência mais longa (`longest_first`), é a query — não o candidato, curto (p50=25 tokens) — quem perde texto na prática. Truncamento é real mas marginal. Ação: subi `max_length` de 512→768 em `configs/reranker/training/{lora,full_ft}.yaml` (grátis, elimina a cauda, zero risco).

**Hipótese 2 — falta de negativos difíceis (REFUTADA).** Se o retrieval kNN só trouxesse negativos "fáceis" (de ramos totalmente diferentes), o modelo nunca aprenderia a discriminar entre variantes parecidas. Medi nos 11.468 pares de treino reais: **86,6%** têm pelo menos um negativo do mesmo ramo raiz que o item gold, e **62,6%** têm um negativo do mesmo sub-ramo (nível 2) — o kNN já naturalmente traz confusores difíceis na maioria dos casos, porque análises de normas parecidas tendem a ter condições do mesmo domínio regulatório. Não é aqui que está o problema.

**Hipótese 3 — cauda longa de itens raros na taxonomia (CONFIRMADA, é a causa dominante).** Contei quantas vezes cada um dos 2.715 itens da taxonomia aparece como positivo nos 11.468 pares de treino do reranker: **53,9% aparecem exatamente 1 vez**, **77,3% aparecem menos de 5 vezes**, e só 8,1% têm 20+ ocorrências. Cruzando isso com o recall real do reranker nas 200 amostras de teste, por faixa de frequência do item gold no treino:

| Frequência do item no treino | Recall do reranker |
| --- | --- |
| 0 (nunca visto) | 0.024 |
| 1–4 (raro) | 0.146 |
| 5–19 (médio) | 0.236 |
| 20+ (comum) | 0.541 |

A relação é quase linear. **Isso é exatamente a mesma descoberta da Etapa 2 da POC** (causal-LM: "itens vistos 50+ vezes → 31% recall; itens vistos <10 vezes → 0%"), agora replicada numa arquitetura completamente diferente. Conclusão: o volume total de análises (18,5k) não é o fator relevante — o que importa é como esse volume se distribui entre 2.715 classes, e essa distribução é extremamente desbalanceada (cauda longa / power-law). Nenhuma arquitetura (generativa, cross-encoder, ou kNN) escapa disso: para um item visto 1 vez na vida, não existe sinal suficiente pra aprender o que o distingue dos seus vizinhos na taxonomia, seja qual for o modelo. Isso também explica por que o F1 raiz/nível-2 é sempre bem melhor que o F1 de folha exata em todas as abordagens testadas: as categorias mais amplas têm muito mais suporte agregado (todos os itens de um mesmo ramo contribuem exemplos para aprender esse ramo), enquanto a folha exata depende só das próprias ocorrências daquele item específico.

Isso não é "resultado anormal" — é a assinatura clássica de um problema de classificação de cauda longa extrema (2.715 classes, a maioria com <5 exemplos). É consistente com a régua que a própria Etapa 8 já media por outro ângulo: a faixa de similaridade <0.5 (~30% dos casos, sem precedente parecido no histórico) é onde todo método capota (kNN 0.212, FT 0.135) — baixa similaridade com o histórico e item raro na taxonomia são, na prática, o mesmo fenômeno visto de dois jeitos.

**Correções implementadas (09/07/2026)**, ambas de baixo risco e sem precisar de mais dado novo:

1. **`max_length` 512→768** nas duas configs de treino do reranker — elimina o truncamento residual.
2. **Oversampling balanceado por classe** em `scripts/build_reranker_dataset.py` (`compute_item_frequency` + `oversample_factor`): registros cujo item positivo mais raro foi visto só 1x no pool de treino são repetidos 4x; 2–4x são repetidos 2x; 5+ ficam como estão (capado em 4x de propósito, pra reponderar sem virar memorização de uma única query). Aplicado só no split de treino — validação fica intocada para não inflar artificialmente o sinal do early stopping. Dataset real regenerado: 11.468 registros base → 15.921 registros efetivos de treino (+38,8%), 613 de validação (sem alteração).

**Opções maiores, não implementadas — decisão do usuário antes de seguir:**

- **Roteamento hierárquico (coarse-to-fine)**: já que o modelo acerta ramo/subramo bem melhor que a folha exata, uma segunda etapa poderia primeiro classificar o ramo (poucas classes, muito suporte) e só depois discriminar entre os poucos irmãos daquele ramo — em vez de rankear direto entre até 40 candidatos de ramos variados. Isso é uma mudança de arquitetura de solução (dois estágios), não uma correção pontual — maior esforço de implementação e validação.
- **Aumento sintético para itens raros**: gerar pares de treino adicionais para os 77% de itens com <5 exemplos, via paraphraseamento (LLM) do requisito/texto de norma associado a esses itens na taxonomia oficial. Ataca a causa raiz diretamente (mais sinal por classe), mas introduz risco de ruído/viés se a geração sintética não for bem calibrada, e não é uma correção "grátis" — precisa de validação cuidadosa antes de confiar nela para treino.
- **Reponderação de loss por classe** (em vez de oversampling de registros): equivalente em espírito ao oversampling implementado, mas atuando direto na loss (`LambdaLoss` do `sentence-transformers` não expõe pesos por documento nativamente até onde investigado) — precisaria de uma loss customizada. Não implementado por não ter suporte nativo na biblioteca sem um patch mais invasivo.

Nenhuma dessas três é uma correção "significativa" que eu implementaria sem confirmação, porque todas envolvem trade-offs de escopo/risco que só o usuário deve decidir — mas as duas primeiras (oversampling e `max_length`) já estão aplicadas e valem ser testadas no próximo treino real antes de ir para qualquer uma destas.

### 2.3 Melhorar o teto do retrieval de candidatos

Como o teto de recall dos candidatos (0.737) já limita qualquer modelo, subir esse teto vale tanto quanto — ou mais que — trocar o modelo:

- Testar embeddings maiores/mais recentes que o `multilingual-e5-small` atual (ex. `e5-large`, `bge-m3`, ou embeddings de API tipo `voyage-law` se houver orçamento) para o kNN sobre ementas + requisitos.
- Aumentar `K` de candidatos e medir a curva recall × ruído (candidatos demais também atrapalham, como mostrou o erro da Etapa 3).
- Indexar por requisito individual (não só por ementa da norma) já que a Etapa 7 mostrou que o texto do requisito é o sinal mais forte — o retrieval pode estar subaproveitando esse campo.

### 2.4 Estratégia híbrida focada na cauda nova

Os dados mostram que o fine-tuning só tem chance de agregar valor real na faixa de similaridade < 0.5 (~30% dos casos), onde cópia não funciona. Duas ideias:

- Treinar/especializar o modelo **só** nesse subconjunto de casos genuinamente novos, em vez de diluir capacidade aprendendo também os casos triviais que o kNN já resolve sozinho.
- Produção híbrida: usar kNN para ≥0.5 (onde ele já ganha ou empata) e reservar o modelo fine-tunado (ou a API) só para <0.5 — isso já apareceu como ideia no "Híbrido (kNN + Haiku)" da Etapa 8 (F1 0.564), mas nunca foi testado com um fine-tuned dedicado a essa fatia.

### 2.5 Hiperparâmetros e regime de treino

Os runs da POC usaram LoRA r=16, 1 época, QLoRA 4-bit nos defaults de 6GB. Com hardware maior em nuvem, dá para testar sem restrição de VRAM:

- LoRA r=32/64 ou até full fine-tuning (parâmetro cheio) — a POC nunca testou full FT.
- Mais de 1 época com early stopping real (`early_stopping.enabled` já existe em `configs/training/sft.yaml` mas está desligado).
- Context length maior que 1024 (hoje trunca; a Etapa 5 já achou um bug de truncamento em 512 tokens na geração).
- Sempre validar contra o **test set fixo de 200 amostras**, nunca só contra a métrica de validação — a Etapa 4 mostrou que a validação pode estar inflada por quase-duplicatas.

### 2.6 Cobertura dos 22% sem requisitos

A Etapa 7 deixou de fora as análises sem `RequisitosIds` (22% do total). Se o fine-tuning avançar, decidir explicitamente: (a) manter essas sempre em fluxo humano, ou (b) investir em preencher/inferir requisitos faltantes (ex. reprocessar normas antigas) para trazê-las ao pipeline treinável.

### 2.7 Pipeline de avaliação e regressão contínua

Já existe `ft compare` para comparar experimentos. Para produção, formalizar:

- Teste de regressão automático contra o test set fixo antes de qualquer novo modelo substituir o anterior.
- Congelar a métrica de referência (F1 item exato + F1 hierárquico + corte por similaridade) como critério de aceite de deploy.

### 2.8 Flywheel de dados

Se o modelo for para produção, as correções feitas pelos consultores sobre as sugestões (do retrieval e/ou do modelo) são o dataset de treino mais valioso daqui para frente — muito mais informativo que o histórico original, porque captura exatamente os casos em que o sistema errou. Vale desenhar, desde já, como essas correções são capturadas e versionadas para retreinos futuros.

## 3. Roadmap sugerido (ordem por custo/risco crescente)

| Fase | Ação | Custo aproximado | Critério de avanço |
| --- | --- | --- | --- |
| 0 | Decodificação restrita sobre o v6 atual (sem retreinar) | Horas, hardware local | Se F1 subir de forma relevante, já indica que o problema é geração livre, não capacidade do modelo |
| 1 | Melhorar retrieval (embeddings + indexação por requisito) | Dias, sem GPU de treino | Teto de recall dos candidatos deve subir de 0.737 |
| 2 | Retreinar v6 com candidatos melhores + reformulação seleção/reranking | Local ou nuvem pequena (L4/A10) | Bater o kNN (0.559) na faixa 0.5–0.8 |
| 3 | Escalar para 1.7B/4B em nuvem, focado na faixa <0.5 | Nuvem (ver pesquisa de plataformas) | Superar Haiku (0.256) e o FT 0.6B (0.135) nessa faixa |
| 4 | Produção híbrida (kNN + modelo especializado na cauda nova) | Depende da fase 3 | F1 combinado > 0.564 (híbrido kNN+Haiku já medido) |

Cada fase só deve avançar para a próxima se o ganho for medido no mesmo protocolo da POC (200 amostras de teste fixas, split por `norma_id`, métrica hierárquica). Sem isso, corre-se o risco de repetir os dois falsos-positivos já vividos na POC (validação inflada na Etapa 4, e teste enviesado por ID na Etapa 5).

## 4. Custo de manutenção se este caminho for adiante

Independente da fase alcançada, fine-tuning implica custo recorrente que o retrieval não tem: retreino a cada atualização relevante do histórico/taxonomia, avaliação de regressão a cada versão, e infraestrutura de treino (própria ou em nuvem) mantida disponível. Ver seção 6 da POC para o comparativo de esforço com a solução de retrieval recomendada.
