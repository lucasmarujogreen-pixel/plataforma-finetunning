# Pesquisa — Plataformas de Nuvem para Fine-tuning de Modelos Maiores

> **Data da pesquisa:** 08/07/2026 · **Contexto:** hardware local atual é uma RTX 4050 Laptop (6 GB VRAM); a POC (ver [poc-pda-716.md](poc-pda-716.md)) treinou Qwen3-0.6B localmente. Este documento mapeia opções de nuvem para treinar versões maiores (1.7B/4B/8B) caso o time decida perseguir esse caminho (ver [próximos passos](proximos-passos-melhoria-modelo.md), seção 2.2).
>
> **Nota de metodologia:** esta pesquisa foi feita com um harness de busca multi-fonte com verificação cruzada (cada afirmação checada por até 3 buscas independentes contra a página oficial). O processo foi interrompido duas vezes por limites de conta (spend limit e depois session limit) antes de terminar 100% das verificações. Os números abaixo marcados como **confirmado** foram checados ao vivo contra a página oficial citada; os marcados como **não verificado de forma independente** vêm de uma única fonte (geralmente um agregador/blog) e devem ser tratados como estimativa, não cotação — antes de comprometer orçamento, confira o preço na página oficial linkada.

## 1. Metodologia para estimar tempo e custo de treino

Fórmula geral:

```
tempo_treino (h) = (n_epochs × tokens_totais_do_dataset) / (tokens_por_segundo × 3600)
custo (US$) = tempo_treino (h) × preço_por_hora_da_GPU
```

Como referência calibrada para o próprio caso do time (modelo pequeno da família Qwen, GPU de VRAM limitada), a fonte mais comparável encontrada foi um estudo acadêmico (arXiv, set/2025) que mediu LoRA em **Qwen2.5-1.5B-Instruct numa RTX 4060 8GB**: até **628 tokens/s** com otimizador paginado (PagedAdamW), viável em contexto de até 2048 tokens, usando 6,2–8,1 GB de VRAM — a mesma faixa de hardware e família de modelo do time. Esse artigo **não compara diretamente com GPUs de datacenter** (A100/H100), então a extrapolação para nuvem abaixo é uma estimativa de ordem de grandeza, não um dado medido.

Com o dataset da POC (14.194 exemplos de treino, contexto de até 1024–2048 tokens ⇒ ordem de 15–20M tokens por época) e assumindo 1 época (como o `training.num_epochs=1.0` já usado no projeto):

| Modelo | Throughput estimado (consumer GPU, extrapolado por tamanho) | Throughput estimado em A100 80GB (multiplicador 5–10x não verificado) | Tempo estimado em A100 | Custo estimado (A100 ~US$1,4–3,4/h) |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | ~550 tok/s | ~2.750–5.500 tok/s | ~1–2h | US$ 2–7 |
| Qwen3-4B | ~235 tok/s | ~1.200–2.350 tok/s | ~2–4h | US$ 4–14 |
| Qwen3-8B | ~120 tok/s | ~590–1.200 tok/s | ~4–8h | US$ 8–27 |

Isso é consistente, em ordem de grandeza, com os poucos benchmarks públicos encontrados (não oficiais, de blogs de provedores): "13B QLoRA em A100 40GB: 3–6h, US$2,28–4,56" e "70B QLoRA em H100 80GB: 8–12h, US$10,64–15,96" — ambos escalando razoavelmente para modelos menores. **Recomendação prática:** como a própria POC aprendeu na Etapa 1 (o smoke test revelou 20% de utilização de GPU antes de otimizar), não confie nesse cálculo teórico — rode um smoke test de poucos steps na máquina de nuvem escolhida, meça tokens/s real, e só então estime o custo do treino completo.

## 2. Hyperscalers

### 2.1 AWS

| Item | Dado | Status |
| --- | --- | --- |
| Instâncias GPU para treino | P4d/P4de (8× A100), P5 (8× H100), P5e/P5en (H200), nova série P6 (B200/B300) | **Confirmado** (aws.amazon.com/ec2/capacityblocks/pricing) |
| Spot | Desconto de até 90% vs. on-demand; preço flutua e é ajustado pela AWS a cada ~5 min conforme oferta/demanda | **Confirmado** (aws.amazon.com/ec2/spot/pricing) |
| H100 pós-corte de preço (jun/2025) | AWS reduziu preço de H100 em 44%, chegando a ~US$3,90/GPU-hora no P5 | Não verificado de forma independente (fonte única, blog) |
| EC2 Capacity Blocks for ML | Modelo de reserva antecipada: taxa de reserva paga na compra + taxa de SO cobrada por hora de instância (não é spot nem on-demand tradicional) | **Confirmado** |
| Disponibilidade regional | Capacity Blocks cobrem América do Sul, além de US East/West, Ásia-Pacífico, Europa, Austrália e GovCloud | **Confirmado** — é o único hyperscaler com essa confirmação explícita para América do Sul nesta pesquisa |
| Página geral de pricing on-demand | Não lista preços por instância GPU — é preciso ir à AWS Pricing Calculator ou página dedicada por família | **Confirmado** |

Sem exemplo de preço puramente on-demand por hora do P5.48xlarge (8×H100) que tenha sido confirmado como atual — o valor de US$31,464/hora encontrado é rotulado pela própria AWS como "exemplo ilustrativo" datado de março/2025, não um preço vigente garantido. Dividido por 8 GPUs (~US$3,93/GPU-hora), é consistente com o número pós-corte de 44% acima, então serve como referência de ordem de grandeza.

### 2.2 Azure

| Item | Dado | Status |
| --- | --- | --- |
| Instância de referência | ND96isr H100 v5: 8× H100, 96 vCPUs, 1.900 GiB RAM | Não verificado de forma independente (fonte secundária; a busca de verificação foi interrompida pelo limite de sessão antes de checar contra a página oficial da Azure) |
| Preço on-demand | ~US$98,32/hora (≈ US$12,29/GPU-hora) | Mesma ressalva acima |
| Preço spot/low-priority | ~US$18,17/hora (≈ US$2,27/GPU-hora, ~82% de desconto) | Mesma ressalva acima |
| Regiões | 26–50+ regiões globais citadas nas fontes secundárias, nenhuma menção a Brasil/América do Sul | Mesma ressalva acima |

Esta é a plataforma com **menor confiança nos números** desta pesquisa, porque a etapa de verificação cruzada para Azure foi interrompida pelo limite de sessão antes de confirmar contra `azure.microsoft.com/pricing`. Antes de decidir por Azure, confirme os valores diretamente na calculadora de preços oficial.

### 2.3 Google Cloud (Vertex AI)

Nota: a página `cloud.google.com/vertex-ai/pricing` foi renomeada/redirecionada para `cloud.google.com/products/gemini-enterprise-agent-platform/pricing` durante a pesquisa — o produto de custom training segue existindo sob essa nova URL.

| Acelerador | Preço de computação | Taxa de gerenciamento (management fee) | Status |
| --- | --- | --- | --- |
| A100 40GB | US$2,933908/hora | +US$0,4400862/hora | **Confirmado** (fetch ao vivo da página oficial) |
| A100 80GB | ~US$4,548324/hora | +US$0,682248/hora | Não corroborado de forma independente (2 tentativas de verificação falharam em confirmar o valor exato) |
| H100 80GB | US$11,34343/hora | +US$1,7015139/hora | **Confirmado** (fetch ao vivo da página oficial) |
| H100 MEGA 80GB | US$13,7742824/hora | não listada | Não verificado |

A Vertex AI cobra **infraestrutura + management fee separadamente** em cima do preço do acelerador — é o único hyperscaler nesta pesquisa com esse modelo de cobrança em duas camadas explicitado. Suporta Spot VMs e reservas do Compute Engine, mas a management fee incide mesmo assim.

## 3. Provedores de GPU cloud especializados (mais baratos)

Preços on-demand por GPU, cruzando múltiplas fontes independentes (a maioria confirmada ao vivo contra a página oficial do provedor):

| Provedor | A100 80GB | H100 (PCIe/SXM) | L4/entrada | Modelo de cobrança | Managed vs. infra crua | Regiões BR/AmSul |
| --- | --- | --- | --- | --- | --- | --- |
| **RunPod** | US$1,39/h (**confirmado**) | US$2,79–2,89/h on-demand, ~US$1,99/h spot-like (**confirmado**) | L4 US$0,39/h | Por segundo, sem mínimo, sem taxa de reserva | Dashboard/CLI/API + Docker; sem suporte nativo a Jupyter explicitado na página de produto | Não — 31 regiões em EUA/Europa/Ásia/Austrália |
| **Lambda Labs** | US$1,79/h | US$2,49/h (PCIe) a US$3,29/h (SXM single), ~US$2,99/h em config 8×GPU (**confirmado** em múltiplas fontes) | — | Por minuto, zero taxa de egress | JupyterLab pré-instalado em toda instância (**confirmado**) | Não — 13 regiões (EUA, Europa, Ásia, Israel) |
| **Vast.ai** | ~US$1,27/h | US$1,49–4,69/h (varia por host, marketplace peer-to-peer) | RTX 4090 desde US$0,50/h; RTX 3090 desde US$0,16/h | Por hora, preço dinâmico por host | Marketplace — confiabilidade variável, sem SLA, não recomendado para produção | Não mencionado |
| **CoreWeave** | ~US$2,50/h (normalizado de nó 8×GPU) | ~US$2,70/h (normalizado de nó 8×GPU) | — | GPU + vCPU + RAM cobrados separadamente; desconto de até 60% em uso reservado | Requer conhecimento de Kubernetes/orquestração — maior atrito para time pequeno | Não mencionado |
| **Paperspace** (DigitalOcean) | US$3,09/h on-demand (US$1,15/h só com compromisso de 3 anos) | US$5,95/h on-demand (US$2,24/h só com compromisso de 3 anos) | — | Por hora; plano pago (US$39/mês) necessário para GPUs premium | UI amigável para devs, notebooks gerenciados (Gradient) | Apenas 3 regiões no total, nenhuma citada como BR/AmSul |

**Leitura geral:** RunPod e Lambda Labs saem na frente em relação custo/praticidade para um time pequeno — preços confirmados, cobrança granular (segundo/minuto), sem necessidade de reserva. Vast.ai é o mais barato em termos absolutos mas com o trade-off de confiabilidade de marketplace (adequado a experimentos, não a produção). CoreWeave e Paperspace são mais caros e/ou mais burocráticos para o perfil do time. **Nenhum dos provedores especializados confirma cobertura no Brasil/América do Sul** — a latência de acesso via internet residencial/corporativa do Brasil até EUA costuma ser aceitável para treino (não é workload sensível a latência de rede como inferência online).

## 4. Serviços gerenciados de fine-tuning (abstraem a infraestrutura)

| Serviço | Modelo de cobrança | Suporta modelos abertos tipo Qwen? | Managed? | Observação |
| --- | --- | --- | --- | --- |
| **Together AI** | Por token processado: `total_tokens = (n_epochs × tokens_treino) + (n_evals × tokens_validação)`; mínimo US$4/job (**confirmado**, fórmula e mínimo verificados na doc oficial) | Página de marketing diz "qualquer modelo do HF Hub, sem lock-in" — mas essa alegação **não resistiu à verificação adversarial** (foi contestada em 3 votos como overreach de marketing sem confirmação técnica independente) | Sim — API única, sem gerenciar infraestrutura (**confirmado**) | Preço exato em US$/milhão de tokens não foi extraído (está só na página de pricing, não na doc); jobs cancelados só cobram os steps completados, jobs que falham são reembolsados integralmente |
| **Hugging Face AutoTrain** | Pay-as-you-go por minuto conforme hardware, se rodar em HF Spaces (**confirmado**); grátis se rodar localmente na própria infra | Sim, é a stack HF nativa | Sim (Spaces) ou não (local) | Sem lock-in — modelo treinado pode ser baixado e usado onde quiser |
| **Modal** | Serverless, cobrança por segundo real de uso, sem custo de ociosidade (**confirmado**). A100 80GB ≈ US$2,50/h, H100 ≈ US$3,95/h, L4 ≈ US$0,80/h, A10 ≈ US$1,10/h (todos **confirmados** contra `modal.com/pricing`) | Sim — é infraestrutura Python genérica, roda qualquer script com Transformers/TRL | Meio-termo: você escreve o script de treino, a Modal só gerencia o provisionamento/scaling da GPU | Créditos grátis: US$30/mês no plano Starter (até 10 GPUs concorrentes), US$100/mês adicional no plano Team (US$250/mês); bolsas acadêmicas de até US$10.000 |

## 5. Tabela comparativa final

| Plataforma | GPU sugerida | Preço/hora | Tempo estimado 1.7B / 4B / 8B (LoRA, 1 época, ~14k exemplos) | Custo estimado total | Managed vs. infra crua | Atende BR/AmSul? |
| --- | --- | --- | --- | --- | --- | --- |
| RunPod | A100 80GB | US$1,39/h | ~1–2h / ~2–4h / ~4–8h | US$2–11 | Infra crua (containers) | Não confirmado |
| RunPod | H100 | US$2,79–2,89/h | ~0,5–1h / ~1–2h / ~2–4h | US$1–12 | Infra crua | Não confirmado |
| Lambda Labs | A100 80GB | US$1,79/h | ~1–2h / ~2–4h / ~4–8h | US$2–14 | Infra crua + Jupyter incluso | Não confirmado |
| Vast.ai | A100 80GB | ~US$1,27/h | ~1–2h / ~2–4h / ~4–8h | US$1–10 | Marketplace (variável) | Não confirmado |
| Modal | A100 80GB | US$2,50/h | ~1–2h / ~2–4h / ~4–8h | US$2,5–20 | Meio-termo (serverless, script próprio) | Não confirmado |
| Google Vertex AI | A100 40GB | US$2,93/h + US$0,44/h taxa | ~1–3h / ~2–5h / ~5–10h | US$3–34 | Managed completo | Não confirmado |
| AWS (Capacity Blocks) | P4d (8×A100, reserva) | Modelo de reserva, não comparável a on-demand direto | — | Overkill (8 GPUs) para um modelo só | Infra crua | **Sim** (única confirmação explícita) |
| Azure | ND96isr H100 v5 | ~US$12,29/GPU-h on-demand (não verificado) | — | Overkill (8 GPUs) para um modelo só; dado de baixa confiança | Infra crua | Não confirmado |
| Together AI | (abstraído) | Por token, valor exato não obtido | — | Mínimo US$4/job | Managed completo (API) | N/A (API) |
| HF AutoTrain (Spaces) | Conforme hardware | Por minuto | — | Baixo para modelos pequenos | Managed completo | N/A (API) |

Para os três tamanhos de modelo do interesse do time (1.7B/4B/8B), **as instâncias multi-GPU de Azure (8×H100) e o P4d/P5 completo da AWS são superdimensionadas** — fazem sentido para treinar vários experimentos em paralelo ou modelos muito maiores, não para um fine-tuning LoRA pontual de um modelo único. As opções de GPU única sob demanda (RunPod, Lambda Labs, Vast.ai, Modal) são mais adequadas ao perfil "experimentos pontuais, não treino contínuo" do time.

## 6. Recomendação

Para um time pequeno que já usa Python/Transformers/TRL/PEFT/Hydra e quer o menor atrito operacional com custo controlado, rodando **experimentos pontuais** (não treino contínuo):

1. **Primeira escolha: RunPod ou Lambda Labs**, com GPU única A100 80GB (~US$1,4–1,8/hora) ou H100 se o tempo total valer a pena pagar mais por hora para terminar mais rápido. Ambos têm cobrança granular (segundo/minuto), sem necessidade de reserva de capacidade, setup simples (containers Docker + o script `ft train` do projeto rodaria praticamente sem modificação, só trocando o `configs/model/*.yaml` e ajustando `lora`/`training` para VRAM maior). Lambda Labs tem a vantagem adicional de JupyterLab pré-instalado, útil para debugging interativo.
2. **Se preferir não gerenciar infraestrutura nenhuma**: Modal — mesma faixa de preço, cobrança por segundo real, mas o time escreve/roda o script Python normalmente (compatível com o pipeline `ft` existente) sem lidar com provisionamento manual de instância.
3. **Evitar para este caso de uso**: Azure e AWS puros (setup mais burocrático, cotas de GPU exigem aprovação prévia, instâncias padrão são multi-GPU e superdimensionadas para um fine-tuning de modelo único) e CoreWeave (exige Kubernetes). Vertex AI é utilizável mas tem a camada extra de management fee.
4. **Vast.ai** é a opção mais barata em teoria, mas justamente por ser marketplace peer-to-peer tem confiabilidade variável — cogitar só se o objetivo for reduzir custo ao mínimo absoluto em experimentos totalmente tolerantes a falha/reinício.
5. Nenhum provedor confirma explicitamente datacenters no Brasil/América do Sul (exceto os Capacity Blocks da AWS, que são overkill para este caso) — mas para treino em batch (não inferência online), a latência de rede do Brasil até EUA não é um fator limitante relevante.
6. **Antes de comprometer orçamento**: rodar um smoke test curto (algumas centenas de steps) na plataforma escolhida para medir tokens/s real e recalcular o custo total — a estimativa da seção 1 é de ordem de grandeza, não uma cotação.

## 7. Quanto o projeto precisaria mudar para treinar na nuvem

Pouco, em termos de código — o projeto já foi construído pensando em portabilidade:

- **`docker/Dockerfile`** já parte de `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, usa `uv` para instalar as mesmas dependências (PyTorch cu128, Transformers, TRL, PEFT) e expõe o mesmo `ft` CLI como entrypoint — é a mesma imagem que rodaria localmente ou em qualquer VM com Docker + NVIDIA Container Toolkit.
- **`docker/docker-compose.yml`** já declara a reserva de GPU no formato padrão do Compose (`deploy.resources.reservations.devices` com `driver: nvidia`) — é exatamente o que provedores como RunPod/Lambda Labs esperam quando oferecem templates com Docker pronto.
- A detecção de hardware (`precision: auto`, `attention: auto` em `configs/model/*.yaml`, ver [README.md](../README.md)) já ajusta sozinha para bf16/flash-attention conforme a GPU disponível — não precisa hardcodar nada por ambiente.

O que de fato muda são **arquivos de configuração, não código**:

1. Criar `configs/model/<novo-modelo>.yaml` apontando para o repositório HF do modelo maior (o README já documenta esse fluxo como "Estendendo").
2. Ajustar `configs/lora/*.yaml` e `configs/training/sft.yaml` para aproveitar a VRAM maior — hoje os defaults (QLoRA 4-bit, `micro_batch_size=1`, `gradient_accumulation_steps=8`, `context_length=1024`) existem só por causa do limite de 6GB local; numa A100/H100 provavelmente valeria trocar para LoRA bf16 (mais rápido, como a própria POC descobriu na Etapa 1 ao sair do QLoRA no modelo pequeno), aumentar batch e context length.
3. Levar os dados: `datasets/processed/` (ou os artefatos brutos + rodar `ft prepare-dataset` de novo na máquina de nuvem) e o cache de embeddings/taxonomia — via volume, bucket (S3/GCS) ou simplesmente `rsync`/`scp` para a instância.
4. Configurar credenciais (HF token só é necessário se o modelo escolhido for gated — Qwen3 não é).

**O que não muda:** a lógica de treino (`training/strategies.py`), o pipeline de dataset, a avaliação, a exportação para GGUF/Ollama — nada disso é específico de hardware local. Ou seja, o esforço de migração é majoritariamente **operacional** (provisionar a instância, sincronizar dados, ajustar YAML) e não uma reescrita do projeto.

## 8. Limitações desta pesquisa

- A pesquisa foi interrompida duas vezes por limites de conta antes de completar 100% da verificação cruzada planejada; os itens marcados "não verificado de forma independente" acima vieram de fonte única (geralmente um agregador/blog, não a página oficial do provedor).
- Os dados de Azure especificamente não chegaram a ser confirmados contra a página oficial (`azure.microsoft.com/pricing`) — recomenda-se checar diretamente antes de decidir por essa opção.
- Os valores de preço/hora de GPU mudam com frequência (vários provedores comentam isso explicitamente) — trate as tabelas acima como um retrato de julho/2026, não como preço garantido no momento da decisão.
- As estimativas de tempo/custo de treino (seção 1) são extrapolações a partir de um único benchmark acadêmico diretamente comparável (Qwen2.5-1.5B em RTX 4060) e de exemplos de blogs de provedores (não fontes oficiais) — validar com smoke test antes de rodar o treino completo.

## 8. Fontes principais (confirmadas ao vivo durante a pesquisa)

- AWS EC2 Spot Pricing — https://aws.amazon.com/ec2/spot/pricing/
- AWS EC2 Capacity Blocks for ML Pricing — https://aws.amazon.com/ec2/capacityblocks/pricing/
- AWS EC2 On-Demand Pricing — https://aws.amazon.com/ec2/pricing/on-demand/
- Google Cloud Vertex AI / Gemini Enterprise Agent Platform Pricing — https://cloud.google.com/vertex-ai/pricing (redireciona para `cloud.google.com/products/gemini-enterprise-agent-platform/pricing`)
- RunPod Cloud GPUs — https://www.runpod.io/product/cloud-gpus
- Lambda Labs On-Demand Cloud — documentação oficial de produto (GPUs e regiões)
- Modal Pricing — https://modal.com/pricing
- Together AI Fine-tuning Pricing (docs) — https://docs.together.ai/docs/fine-tuning-pricing
- Together AI Fine-tuning (produto) — https://www.together.ai/fine-tuning
- Hugging Face AutoTrain Cost — https://huggingface.co/docs/autotrain/en/cost
- Profiling LoRA/QLoRA Fine-Tuning Efficiency on Consumer GPUs: An RTX 4060 Case Study (arXiv, set/2025) — base do benchmark de throughput usado na seção 1

Fontes secundárias (blogs/agregadores, não oficiais, usadas apenas para contexto/cruzamento de ordem de grandeza): Spheron Network blog, artigos comparativos de GPU cloud pricing de 2025–2026 sobre Vast.ai/CoreWeave/Paperspace/Azure.
