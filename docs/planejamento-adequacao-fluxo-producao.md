# Planejamento — Adequação das POCs (PDA-715 e PDA-716) ao fluxo real de produção

> Documento de planejamento. Cobre as duas POCs: `poc_finetunning` (PDA-716, fine-tuning/kNN/reranker) e `poc_promptanalise` (PDA-715, prompt enriquecido).

---

## 1. A descoberta que muda o problema

Até agora as duas POCs trataram "condição" como um rótulo textual a ser previsto. A descoberta sobre o modelo de dados real muda a natureza da tarefa:

1. **Condição = item de formulário.** Cada condição é um registro de `condicao_itens_formularios` (SQL Server, serviço Analises), ligado a `formularios` via `formulario_condicoes`/`condicoes`. Não é texto livre — é uma entidade com identidade (`FormularioId`) e posição numa árvore.

2. **A árvore é recursiva.** Um item de formulário pode ser pai ou filho; um pai tem N filhos, e filhos podem ser pais de outros N filhos, sem limite de profundidade. Na escrita textual:
   - `XPTO` → condição que é apenas item pai (raiz);
   - `XPTO / Xptofilho` → condição filha de `XPTO`;
   - `XPTO / Xptofilho / Xptofilhodofilho` → e assim por diante.

3. **A LLM nunca inventa — sempre seleciona.** A tarefa em produção é: dada uma análise (norma, requisito, etc.), escolher **uma OU mais condições dentre a lista disponível no banco**. E com semântica hierárquica: **decidir um item filho já é suficiente — o pai está implícito e não precisa ser decidido junto**.

4. **Análises têm tipos, com fluxos de entrada diferentes:**
   - **Aplicável com requisito** — análise em cima do requisito; o campo *assunto* é puxado do requisito.
   - **Aplicável sem requisito** — análise em cima da norma como um todo; o *assunto* é definido manualmente.

5. **Mapa de dados real (CQRS):**

   | Fonte | Papel | Tabelas/coleções relevantes |
   | --- | --- | --- |
   | SQL Server — serviço Analises | Write side | `formularios`, `formulario_categorias`, `formularios_sistemas_gestao`, `formulario_condicoes` (TPT), `condicoes` (pai), `condicao_itens_formularios`, `condicao_itens_localidades`, `formularios_ancestrais` (M2M self-ref) |
   | SQL Server — serviço ConformidadeLegal | Write side | `formularios_unidades`, `formulario_unidade_versoes` |
   | MongoDB `greenlegis` | Read side | `formularios` → `FormularioRead` |
   | MongoDB `cliente` | Read side | `formularios_unidades` → `FormularioUnidadeRead` |

**Consequência central:** o problema deixa de ser "gerar/adivinhar a condição certa" e passa a ser **"selecionar, numa lista fechada e real vinda do banco, o(s) item(ns) de formulário mais específico(s) que se aplicam à análise"**. Isso explica em parte o resultado fraco do fine-tuning: treinamos um modelo para *gerar* strings de uma taxonomia que ele não conhecia por inteiro, sem o objetivo final (seleção em lista fechada) estar definido — o alvo estava errado, não (só) o modelo.

---

## 2. Onde cada POC está hoje vs. o fluxo real

### 2.1 O que as POCs já acertaram (não refazer)

| Já validado | Onde | Evidência |
| --- | --- | --- |
| Seleção em lista fechada >> geração livre | Ambas | Único efeito robusto na ablação da 715 (n=30 e n=500); v6 > v5 na 716 |
| Requisitos são o campo-chave do input | Ambas | Concordância entre analistas F1 0.081 → 0.871 ao incluir requisitos (716, Etapa 7) |
| Hierarquia pai/filho existe e importa | Ambas | `taxonomy/tree.py` reconstrói a árvore via `PaiId`; F1 hierárquica em 4 níveis (root > l2 > parent > leaf) |
| Prompt enriquecido supera fine-tuning e kNN | 715 | item_f1 0.5985 (715) vs 0.573 (Sonnet 5 few-shot 716), 0.559 (kNN), 0.457 (fine-tuned v6) |
| Gargalo dominante = recall de candidatos, não alucinação | 715 | `candidate_recall` = 0.6705 limita o teto do F1 |

### 2.2 Premissas que conflitam com o fluxo real

| # | Premissa atual das POCs | Realidade em produção | Impacto |
| --- | --- | --- | --- |
| P1 | Candidatos = união de itens dos vizinhos kNN + top do contexto macro, **cap de 40** (715) ou candidatos kNN no prompt (716 v6) | A lista disponível é a **lista real do banco** (`condicao_itens_formularios` / `FormularioRead`), não um subconjunto derivado de retrieval | O `candidate_recall` de 0.67 é um teto **artificial** que o fluxo real não tem. Se a lista real entra completa (ou corretamente escopada), o recall de candidatos vai a ~1.0 e o F1 sobe de graça |
| P2 | Prever **todos** os itens que o analista marcou (pai e filho contam como itens independentes na métrica) | **Filho decidido ⇒ pai implícito.** Prever só o filho mais específico já é resposta correta | Métrica atual **pune predições corretas** (filho sem pai = falso negativo do pai) e **premia redundância**. Ground truth e predições precisam ser normalizados para o "conjunto mínimo" antes de comparar |
| P3 | Análises sem `RequisitosIds` (22%) foram **descartadas** como problema de qualidade de dados (716 v5/v6) | São um **tipo legítimo de análise** ("aplicável sem requisito"), com input diferente (assunto manual + norma como um todo) | As POCs só cobrem 1 dos 2 fluxos. O fluxo "sem requisito" precisa de caminho de prompt próprio, não de exclusão |
| P4 | Identidade do item = string `"PAI > Filho (47) [Sim]"` vinda pré-formatada do Mongo, separador `" > "` | Identidade = `FormularioId`; a notação de produção usa `" / "`; a string é só apresentação | Casar por string é frágil. Canonicalizar por `FormularioId` em tudo (candidatos, saída da LLM, ground truth, métrica) |
| P5 | Taxonomia = coleção `formularios` inteira (2.715 itens), igual para toda análise | Existem eixos de escopo: `formulario_categorias`, `formularios_sistemas_gestao`, `formularios_unidades` (por unidade/cliente), `formulario_unidade_versoes` (versionamento) | Se a lista disponível para uma análise for escopada (por categoria, sistema de gestão ou unidade), o espaço de busca real pode ser **muito menor** que 2.715 — o que ataca diretamente o problema da cauda longa (53,9% dos itens vistos 1x no treino) |
| P6 | Flag `[Sim]`/`[Não]` (`Marcado`) tratada como parte da string do item | Semântica real a confirmar (resposta do item de formulário?) | Pode ser um segundo output (decidir o item **e** a marcação) ou um filtro — precisa definição antes de re-treinar/re-avaliar |

---

## 3. Perguntas a validar no banco antes de implementar

Estas respostas determinam o desenho das fases seguintes. Validar com queries diretas (read side Mongo já acessível nas POCs) e/ou com o time do serviço Analises:

1. **Escopo da lista disponível:** para uma análise específica, a lista de condições selecionáveis é a taxonomia global de `formularios`, ou é filtrada por categoria (`formulario_categorias`), sistema de gestão (`formularios_sistemas_gestao`) ou unidade do cliente (`formularios_unidades`)? *(Hipótese: no momento da análise a lista é global, e `formularios_unidades` só entra na aplicação ao cliente — confirmar.)*
2. **Papel de `formularios_ancestrais`:** a M2M self-ref é a mesma relação do `PaiId` do read model ou representa outra coisa (ex.: múltiplos pais)? Se um item puder ter mais de um pai, a normalização "filho cobre pai" muda.
3. **Semântica de `Marcado` / `[Sim]`/`[Não]`:** é parte da decisão (a LLM precisa prever) ou metadado do formulário?
4. **`VinculoId`:** o inteiro `vinculo` presente no ground truth precisa ser previsto pela LLM ou é derivável do contexto da análise (norma vinculada)?
5. **Ground truth histórico e a regra pai/filho:** nos registros históricos de `condicoes_analises`, quando o analista marcou um filho, o pai também aparece marcado? (Define se a normalização é "remover ancestrais" ou se o dado já vem mínimo.)
6. **Como identificar o tipo da análise** (com/sem requisito) nos dados: campo `tipo_analise` do export v1 da 716? Presença de `RequisitosIds`? Onde fica o *assunto* manual das análises sem requisito?
7. **Itens excluídos/versões:** `Excluido` no read model e `formulario_unidade_versoes` — a lista disponível deve refletir a versão vigente; como filtrar histórico treinado com itens hoje excluídos?

**Entregável:** seção "Modelo de dados validado" adicionada a este documento, com as respostas e as queries usadas.

### 3.1 Modelo de dados validado (respostas — validadas em 2026-07-13 nos containers `mongo` e `mysql` locais)

1. **Escopo da lista:** a lista disponível é a **taxonomia global** — o filtro por `SistemasGestao` (via cadeia requisito → assunto → sistemas) cobre só 96,8% dos itens do ground truth e mal reduz o espaço (mediana 2.802 de 3.956 ativos), então não vale a perda de recall. Lista real = `TipoId=2 e Excluido=false` → **3.640 itens selecionáveis** (`formularios.TipoId=1` são grupos/seções: 329 pais com apenas 14 marcações em todo o histórico — ruído). `formularios_unidades` (lado cliente) só entra depois, na aplicação ao cliente.
2. **`formularios_ancestrais`:** é uma **closure table** (`AncestralId`,`DescendenteId`; 9.973 linhas p/ 4.395 nós) — materialização da mesma árvore de `PaiId` único do read model. Sem múltiplos pais.
3. **`Marcado`:** faz parte da decisão do analista — `true` = "[Sim]", `false` = "[Não]" (682 condições contêm itens "[Não]", ~3%). A métrica por id trata ambos como item decidido (igual às rodadas anteriores).
4. **`VinculoId`:** enum smallint do write side (`condicoes.Vinculo`), valores 1/2 (87% = 2), sem tabela de lookup. Não derivável dos dados — segue sendo previsto pela LLM como metadado, sem impacto na métrica de itens.
5. **Pai+filho juntos no GT:** só **0,9%** das condições históricas (181/19.976). O histórico já é praticamente conjunto mínimo — confirma a regra "filho basta". Re-pontuar a rodada final da 715 com a métrica normalizada moveu o item_f1 de 0.5985 → **0.6017** (a métrica antiga punia pouco; o ganho grande está nos candidatos).
6. **Tipo da análise:** campo `analises.Tipo` — 1 = com requisito (16.766 com `RequisitosIds`), 2/3 = sem requisito com `AssuntosIds` (4.054). **Atenção:** `AssuntosIds` referencia a coleção **`assuntos`** (hierárquica via `PaiId`; 190/190 ids do histórico resolvem lá), NÃO `assuntos_analises_requisitos`, que tem ids numéricos coincidentes.
7. **Itens excluídos:** 439/4.395 formulários com `Excluido=true`; apenas 0,16% dos itens do GT apontam para itens hoje excluídos — a lista nova (só ativos) não perde recall relevante.

**Status de execução:** Fases A–E executadas na `poc_promptanalise` em 2026-07-13 (candidatos = taxonomia real completa em bloco de system com `cache_control` ~90k tokens; saída por `FormularioId` com validação dura contra a lista; conjunto mínimo hierárquico no GT, predição, few-shot e métricas; universo com análises sem requisito — 18.045 análises, sendo 1.933 tipo 2/3 — e quebra `by_tipo` no relatório; comando `pa rescore`; `max_tokens` 4096 + retry para o thinking adaptativo do Sonnet 5).

### 3.2 Resultado da Fase E (test, n=200, 2026-07-13 — run `20260713_095525`)

| Métrica | Antes (final PDA-715) | Depois (fluxo real) |
| --- | --- | --- |
| `candidate_recall` | 0.6705 | **1.0** |
| Itens inventados | não medido | **0** |
| `json_valid_rate` | 0.98 | **1.0** |
| item_f1 overall | 0.5985 (0.6017 na régua mínima) | 0.5605 (universo novo, inclui tipo 2/3) |
| item_f1 tipo 1 (com requisito) | — | 0.5876 (n=170) |
| item_f1 tipo 2 (sem requisito) | nunca medido (fluxo descartado) | 0.3571 (n=26) |

**Comparação pareada** (170 análises presentes nos dois test runs): micro-F1 0.5927 (antigo) vs 0.5876 (novo) — estatisticamente iguais; no pareado por amostra o novo vence em 30, perde em 19, empata em 121.

**Leitura:** o ganho da adequação é **estrutural, não de F1 agregado** — o teto de candidatos sumiu, a garantia "nunca inventar" é dura, e o fluxo sem requisito passou a ser coberto. A conclusão anterior de que "o gargalo dominante era o teto de recall do retrieval" estava **errada**: com recall de candidatos 1.0 o F1 ficou no mesmo patamar, ou seja, o gargalo real é a **capacidade de seleção nos casos genéricos/difíceis** (Resolução 0.37, Portaria 0.35, NR 0.40, IN 0.13 — normas federais amplas onde a condição depende de contexto que não está no texto). O critério de F1 ≥ 0.75 não foi atingido; antes de acionar a Fase G (fine-tuning), as alavancas de prompt pendentes são: (a) reintroduzir os itens dos vizinhos kNN como *bloco de dica* ao lado da taxonomia completa (dica sem teto); (b) roteador híbrido — copiar vizinho quando similaridade ≥ 0.8 (kNN fazia 0.836 nessa faixa vs 0.54 da LLM) e LLM no resto (~+0.06 no agregado, estimado); (c) enriquecer o input do tipo 2 com o `Complemento` da norma na análise.

**Status (2026-07-13):** alavancas (a) e (b) implementadas na POC 715 — bloco de dica kNN no turno do alvo (`render_knn_hint_block`, desligável com `--no-knn-hint`) e roteador híbrido no use case (`router_threshold = sim_high_threshold = 0.8`, desligável com `--no-router`; predições marcadas com `route` = `llm`/`knn_copy` e relatório com quebra `by_route`).

**Validação dev n=150 (run `20260713_103108`, pareada com `20260713_093406` — mesmas 150 amostras):** overall 0.6566 → 0.6612. Separando por alavanca: (a) dica kNN, nas 130 amostras que ficaram na LLM, 0.6409 → 0.6513 (pareado 14 melhores / 10 piores / 106 iguais) — ganho pequeno mas consistente, **mantida**; (b) roteador, nas 20 amostras roteadas, a cópia kNN fez 0.7229 contra 0.7595 da LLM nas mesmas amostras (pareado 2/4/14) — **a premissa não transferiu**: o "kNN 0.836 vs LLM 0.54 na faixa alta" veio da régua antiga, em que a LLM operava com o teto de candidatos; com a taxonomia completa a LLM já é forte na faixa alta e a cópia só subtrai. Sem gradiente por score dentro da faixa (perdas em 0.815–0.822, ganho em 0.809) — subir o limiar não salva. **Recomendação: roteador desligado daqui em diante**; o único benefício residual é ~13% menos chamadas de API. Próxima alavanca com maior headroom: (c) `Complemento` no input do tipo 2 (0.2892 no dev, vs 0.7192 do tipo 1).

**Status (2026-07-13, tarde):** roteador **desligado por padrão** no CLI (religável com `--router`, para ablação) e alavanca (c) implementada — `Normas[0].Complemento` (validado no banco: presente em 1.631 de 4.310 análises tipo 2/3, ~38%; HTML de rich-text, limpo na projeção) entra como "Nota do consultor sobre a aplicabilidade" no bloco sem-requisito de `build_query_text`, alimentando o embedding de retrieval, o few-shot e o caso-alvo; fluxo com requisito intocado (comparabilidade). O `evaluate` também passou a imprimir a tabela de assertividade em % (acerto exato, recall e precisão de itens — geral, por tipo e por espécie).

**Resultado test n=200 com dica kNN + Complemento (run `20260713_111437`, pareado com `20260713_095525` — mesmas 200 amostras):** item_f1 0.5605 → **0.5960** (pareado 29 melhores / 20 piores / 151 iguais); tipo 1 0.5876 → **0.6216** (21/15/134); tipo 2/3 0.3617 → **0.4124** (8/5/17). Exact match 0.47 → 0.485. `candidate_recall` 1.0, zero item inventado, `json_valid` 1.0 mantidos. Nas mesmas 170 amostras tipo 1 compartilhadas com o test da PDA-715 original, a linha evoluiu: 0.5927 (original, com teto de candidatos) → 0.5876 (fluxo real, sem alavancas) → **0.6216** (fluxo real + dica kNN) — ou seja, o fluxo real agora **supera** o resultado original com todas as garantias estruturais. Espécies genéricas seguem sendo o limite (NR 0.29, IN 0.18, Portaria 0.34) — é onde a condição depende de contexto fora do texto da norma, limite que se aplica igualmente a fine-tuning (gatilho da Fase G continua não recomendado).

**Status (2026-07-13, iteração 3):** duas novas alavancas implementadas mirando as espécies genéricas: (d) **órgão emissor** no input (`normas.OrgaosIds` → `orgaos.NomeCompleto`, presente em 99,98% das normas) — entra como linha "Órgão emissor:" no texto de retrieval/few-shot/alvo; para NR/IN/Portaria é o sinal que distingue o domínio da condição (Ministério do Trabalho → SST, IBAMA → ambiental). O texto integral da norma foi descartado como alavanca: não existe no Mongo (`normas` só tem título/resumo; o conteúdo fica em arquivo via `PrimeiroArquivoId`). (e) **pool da dica kNN ampliado para 10 vizinhos** (`knn_hint_k`, CLI `--hint-k`), mantendo 3 como demonstrações few-shot — a agregação da dica foi o mecanismo que mediu ganho, não o número de demos. **Validação dev n=150 (run `20260713_114407`, pareado com as rodadas anteriores — mesmas amostras):** overall 0.6566 (sem alavancas) → 0.6612 (dica-3) → **0.6995** (pareado vs sem-alavancas: 23 melhores / 10 piores / 117 iguais). Tipo 1: 0.7118 → **0.7301** (14/8/102). Tipo 2/3: 0.30 → **0.5319** (9/2/15) — maior salto da POC, soma de Complemento + órgão + dica-10. Nenhuma degradação nas espécies fortes (Lei Municipal/Complementar ≥ 83% de acerto exato). Aprovado para o test n=200.

**Status (2026-07-13, iteração 4 — em validação):** (f) **órgão emissor fora do embedding** — continua no prompt (alvo e few-shot, via `render_target_block`) mas saiu do texto de busca (`build_query_text(include_orgao=False)` por padrão); corrige a regressão de Resolução (0.4112 → 0.3495) preservando os ganhos de NR/IN, e o retrieval volta à configuração da rodada 0.596. (g) **segunda passada hierárquica** (`--no-refine` para ablação): a anatomia do erro mostra F1 raiz ~0.75 vs folha ~0.62 — o modelo acerta o ramo e erra a folha; a segunda chamada continua a mesma conversa (taxonomia lida do cache), reapresenta expandidos só os ramos da 1ª resposta + dos vizinhos e pede reavaliação item a item (fallback: mantém a 1ª resposta se a 2ª não parsear; amostras marcadas com `refined`). Custo por análise ~2x chamadas (2ª chamada é cache-read + ramos).

**Validação dev da v1 (run `20260713_153755`): regressão — 0.6995 → 0.6522.** Diagnóstico nos dados: itens previstos saltaram 275 → 313 (+14%), precisão 77,5% → 67,4% com recall estável; nos 26 casos piorados a 2ª passada adicionou 39 itens (removeu 21) — a instrução "adicione itens aplicáveis que faltaram" induziu superpredição. **Correção (v2):** a revisão passou a ser SÓ de especificidade — troca o item pela variante mais específica do próprio ramo, com proibição explícita de adicionar/remover ("na dúvida, mantenha") — e só dispara quando algum item da 1ª resposta tem descendente selecionável (`has_selectable_descendants`; itens-folha pulam a 2ª chamada, cortando custo). Ramos apresentados: apenas os da 1ª resposta (vizinhos saíram do escopo da revisão).

**Validação dev da v2 (run `20260713_161153`): 0.6807** — a superpredição da v1 sumiu (itens previstos 275 → 283, contra 313 na v1), mas ainda -0.019 vs a referência 0.6995. O pareado por amostra isolou as duas alavancas: (g) **refine v2 é neutro-a-levemente-positivo** — nas 75 amostras com `refined=True` o F1 ficou 0.7260 → 0.7259 (6 melhores / 6 piores), enquanto o grupo SEM refine caiu -0.016 por efeito puro do retrieval, sugerindo que o refine compensou essa perda no grupo dele; (f) **tirar o órgão do embedding é o que custa os -0.019** — os vizinhos mudaram em 149/150 amostras e, no dev, Resolução PIOROU sem o órgão (0.5626 → 0.4926), o contrário do que o test sugeria: a atribuição da regressão de Resolução ao embedding estava confundida com a mudança simultânea do pool da dica (3 → 10 vizinhos), feita na mesma rodada. **Decisão: alavanca (f) revertida** (`include_orgao=True` de volta como padrão em `build_query_text`); a próxima rodada dev isola o refine v2 puro — única diferença vs o run 0.6995 (esperado ≥ 0.70 se o refine agrega; ~0.6995 se neutro, e nesse caso o refine sai do default por custar ~2x chamadas sem ganho).

**Isolamento do refine v2 (run `20260713_165406`, órgão de volta no embedding): micro-F1 idêntico à referência — 0.6995 vs 0.6995** (precisão 0.7745 e recall 0.6377 idênticos até a 4ª casa). Por amostra há leve viés positivo (11 melhores / 5 piores; dentro do grupo refinado, 7/2), mas os ganhos e perdas de itens se anulam no micro. Vizinhos divergiram em 12/150 amostras por não-determinismo do embedding da OpenAI (mesmo conjunto com ordem trocada em 8 delas, scores divergindo na 4ª casa) — ruído, não mudança de configuração. **Decisão: alavanca (g) fora do default** (`--refine` opt-in no CLI; `refine=False` em `execute`/`RunEvaluation`) — não paga as ~2x chamadas em metade das amostras. Encerra a iteração 4: configuração vigente volta a ser exatamente a da iteração 3 (dev 0.6995 / test 0.6171), com o refine disponível para retomar caso uma alavanca futura aumente a taxa de erro de folha.

**PDA-717 executada (2026-07-13):** os três sistemas foram comparados nas **mesmas 200 amostras** do test da 715 e na mesma régua (a Fase F prevista no §5 foi cumprida por essa via): prompt enriquecido **0.6198**, kNN 0.4625, fine-tuned v6 0.4524 — o prompt vence todos os cortes (tipo, espécie). Entregável completo com matriz de custo/manutenção/riscos e recomendação: `poc_promptanalise/comparacao_pocs_717/comparativo-pda-717.md`.

**Resultado test n=200 (run `20260713_120625`, pareado nas mesmas 200 amostras):** overall 0.5605 (base) → 0.5960 (dica-3+Complemento) → **0.6171** (pareado vs base: 47 melhores / 20 piores / 133 iguais; vs rodada anterior: 30/22/148). Tipo 1: **0.6450**; tipo 2/3: **0.42**. Acerto exato 47% → 51%. Linha histórica nas 170 tipo 1 compartilhadas: 0.5927 (PDA-715 original) → 0.5876 → 0.6216 → **0.6450**. Por espécie vs rodada anterior: o órgão emissor moveu **NR 0.2857 → 0.4167** (5/2) e **IN 0.1778 → 0.2692** (3/0), Lei Estadual 0.59 → 0.68; porém **Resolução regrediu 0.4112 → 0.3495** (2 melhores / 8 piores) e Portaria ficou neutra (2/6) — hipótese: emissor de Resolução é heterogêneo (CONAMA/ANVISA/ANTT/agências) e a mudança no embedding piorou os vizinhos recuperados dessas amostras. Próximo passo natural: investigar as 8 Resoluções que pioraram (vizinhos antes/depois) antes de nova alavanca.

---

## 4. Plano — POC prompt enriquecido (PDA-715) · prioridade 1

É a abordagem vencedora (F1 0.5985) e a que mais se beneficia da descoberta: o fluxo real é exatamente "identificar as informações da análise e escolher da lista do banco". Adequações em ordem de execução:

### Fase A — Camada de candidatos real (ataca P1 e P5)

- Substituir `_collect_candidates` (união vizinhos+macro, cap 40, truncamento alfabético) por um **provider de lista real**: carregar de `FormularioRead` (Mongo `greenlegis.formularios`) a árvore vigente (filtrando `Excluido`), aplicando o escopo confirmado na pergunta 1.
- Se a lista escopada couber no contexto (poucas centenas de itens), enviar **inteira** no prompt — elimina o teto de recall. Se não couber, manter retrieval apenas como **pré-ranqueador** da lista real (nunca como fonte), com cap bem maior e priorização por relevância (corrige de quebra o bug documentado do truncamento alfabético).
- Medir `candidate_recall` antes/depois — meta: ≥ 0.95 (hoje 0.6705).

### Fase B — Canonicalização por `FormularioId` (ataca P4)

- Toda a cadeia (candidatos no prompt, saída da LLM, ground truth, métricas) passa a usar `FormularioId` como chave; o caminho textual (`"XPTO / Filho"`) vira apresentação. A saída da LLM passa a ser validada contra o conjunto de IDs da lista enviada — rejeição/retry se vier ID fora da lista (garantia dura do "nunca inventar").

### Fase C — Normalização hierárquica pai/filho (ataca P2)

- Implementar `minimal_set(itens)`: dado um conjunto de itens, **remover todo item que seja ancestral de outro item do conjunto** (usando a árvore de `PaiId`). Aplicar ao ground truth **e** à predição antes de qualquer métrica.
- Instruir a LLM no system prompt: "selecione sempre o item mais específico aplicável; não selecione o pai quando um filho dele já foi selecionado".
- Recalcular as métricas da rodada final atual (n=200) com essa normalização **sem mudar mais nada**, para separar o ganho da métrica corrigida do ganho das outras fases.

### Fase D — Dois caminhos por tipo de análise (ataca P3)

- **Com requisito:** caminho atual (norma + ementa + requisitos), assunto derivado do requisito.
- **Sem requisito:** novo template — norma como um todo (espécie + título + ementa) + campo *assunto* (manual). Reincorporar as análises hoje descartadas ao universo de avaliação, com relatório de métricas **separado por tipo** (as bases são diferentes; não misturar).

### Fase E — Re-avaliação e critérios

- Repetir o protocolo da 715 (split dev/test por hash, test só no final) com as fases A–D aplicadas.
- Métricas: item_f1 por `FormularioId` sobre conjuntos mínimos + F1 hierárquica (mantida, agora como diagnóstico) + `candidate_recall` + quebra por espécie e por tipo de análise.
- **Critério de sucesso sugerido:** item_f1 (conjunto mínimo) ≥ 0.75 no agregado com `candidate_recall` ≥ 0.95, e nenhuma espécie relevante abaixo de 0.5. *(O 0.5985 atual foi obtido com teto de candidatos 0.67 e métrica que pune a semântica correta — há espaço real para esse salto.)*

---

## 5. Plano — POC fine-tuning (PDA-716) · prioridade 2, condicional

A recomendação da 716 (não seguir com fine-tuning) **continua válida**, mas os experimentos foram feitos com o alvo errado (P1–P3). Adequar o mínimo para manter a comparação honesta e a opção viva:

### Fase F — Dataset v7 (correção do alvo, mesmo sem re-treinar)

- Reconstruir o dataset com: candidatos = lista real escopada (não kNN), targets normalizados para conjunto mínimo por `FormularioId`, separação por tipo de análise (incluindo o fluxo "sem requisito"), notação `" / "` apenas como display.
- Reprocessar as **avaliações existentes** (kNN, API few-shot, fine-tuned v6, reranker) contra o v7 com a métrica normalizada — atualiza o comparativo `comparativo-abordagens-pda-716.md` na mesma régua da 715 adequada.

### Fase G — Re-treino, somente se disparado

- **Gatilho:** a PDA-715 adequada (Fase E) não atingir o critério de sucesso, **ou** custo/latência da API inviabilizar produção.
- Se disparado, a forma correta agora é clara: **não é geração** — é seleção/ranqueamento sobre a lista real:
  - reranker (`bge-reranker-v2-m3`, trilho já existente em `configs/reranker/`) pontuando os itens da **lista real escopada** (não candidatos kNN — remove o `top_k` fixo e os falsos positivos documentados);
  - ou constrained decoding sobre `FormularioId` (já listado em `proximos-passos-melhoria-modelo.md` §2.1, nunca testado).
- A cauda longa (53,9% dos itens vistos 1x) continua sendo o limite estrutural de qualquer modelo treinado — é o argumento para a 715 permanecer como caminho principal.

---

## 6. Ordem de execução consolidada

| Ordem | Item | POC | Depende de |
| --- | --- | --- | --- |
| 1 | Validar as 7 perguntas do §3 no banco | ambas | — |
| 2 | Fase C isolada (re-métrica da rodada atual com conjunto mínimo) | 715 | §3 pergunta 5 |
| 3 | Fases A + B (candidatos reais + `FormularioId`) | 715 | §3 perguntas 1, 2, 7 |
| 4 | Fase D (fluxo sem requisito) | 715 | §3 pergunta 6 |
| 5 | Fase E (re-avaliação final) | 715 | 2–4 |
| 6 | Fase F (dataset v7 + comparativo re-medido) | 716 | 2–3 |
| 7 | Fase G (re-treino) | 716 | somente se gatilho da Fase E |

O passo 2 vem antes por ser barato (só re-métrica, sem chamadas de API) e por responder rápido "quanto do resultado atual era artefato da métrica errada".

---

## 7. Riscos

- **Escopo da lista maior que o contexto:** se a resposta da pergunta 1 for "lista global de 2.715 itens sempre", enviar tudo no prompt pode custar caro/estourar contexto — o fallback é o pré-ranqueador da Fase A, que reintroduz (mitigado) o teto de recall.
- **Ground truth histórico inconsistente com a regra pai/filho** (pergunta 5): se analistas marcavam ora pai+filho, ora só filho, a normalização resolve a métrica, mas os few-shots precisam ser normalizados também para não ensinar o padrão redundante.
- **Read models defasados do write side:** as POCs leem o Mongo (read side CQRS); qualquer campo que exista só no SQL Server (ex.: detalhes de `formularios_ancestrais`, versões) exigirá acesso novo — hoje nenhuma das POCs conecta em SQL Server.
- **Comparabilidade histórica:** após Fases B–D as métricas novas não são comparáveis aos números antigos (0.5985 etc.). O passo 2 do §6 cria a ponte (mesma rodada, duas métricas).
- **Análises "sem requisito" podem ter assunto manual não disponível no histórico** — se o campo não existir no read side, esse fluxo fica bloqueado na pergunta 6.
