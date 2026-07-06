# Plataforma Local de Fine-Tuning de LLMs

Plataforma profissional para fine-tuning **100% local** de LLMs: modelos baixados do Hugging Face, treino via **CUDA** na sua GPU, experimentos versionados e exportação para **GGUF + Modelfile (Ollama)**.

- Métodos: **LoRA**, **QLoRA**, **SFT completo** (extensível via Strategy Pattern)
- Configuração 100% via **YAML** (Hydra) validada com **Pydantic**
- Histórico completo de experimentos com manifest de reprodutibilidade
- Métricas em **MLflow**, **TensorBoard**, JSONL e CSV
- Pipeline de dataset com validação, limpeza, dedup, estatísticas e cache

## Arquitetura

Clean Architecture em camadas, dentro de `src/finetuning/`:

| Camada | Pacote | Responsabilidade |
|---|---|---|
| Domain | `domain/` | Entidades (`DatasetMetadata`) e ports (interfaces) |
| Application | `application/` | Use cases: `TrainModel`, `PrepareDataset`, `EvaluateModel`, `ExportModel`, `CompareExperiments`, `ResumeTraining`, `DownloadModel` |
| Infrastructure | `infrastructure/` | HF Hub, gerenciador de experimentos, snapshot de ambiente |
| Especializados | `training/`, `preprocessing/`, `tokenization/`, `evaluation/`, `exporters/`, `monitoring/` | Strategies de treino, pipeline de dataset, perplexity/benchmark, GGUF/Modelfile, hardware/telemetria |
| Presentation | `cli/` | CLI Typer (`ft`) |
| Core | `core/` | Config (Hydra→Pydantic), enums, exceções, logging, hashing |

### Fluxo de treinamento

```
ft train [overrides]
  → Hydra compõe configs/*.yaml → Pydantic valida → AppConfig
  → Detecção de hardware (CUDA, VRAM, bf16, flash-attn) e resolução de precision/attention
  → Criação do run em runs/<timestamp>_<modelo>/ com manifest (UUID, git commit, hashes, versões, GPU)
  → Pipeline de dataset: load → validate → clean → normalize → dedup → stats → tokenize → split → cache
  → Strategy (LoRA/QLoRA/FullSFT) monta o modelo (BitsAndBytes 4-bit p/ QLoRA)
  → TRL SFTTrainer + callbacks (loss, LR, GPU util, VRAM, RAM, temp, tokens/s → MLflow/TensorBoard/JSONL/CSV)
  → Checkpoints (best/last/steps) com resume
  → Avaliação: perplexity, benchmark, comparação com o modelo base, plots
  → Export: adapter LoRA, merge, safetensors, GGUF, Modelfile
```

## Requisitos

- **Python 3.12+** e [`uv`](https://docs.astral.sh/uv/)
- **GPU NVIDIA** com driver atualizado (CUDA 12.8+ via wheels do PyTorch — não precisa instalar o CUDA Toolkit)
- WSL2 (Windows) ou Linux
- Para GGUF: `git`, `cmake`, `build-essential` (usados por `scripts/setup_llamacpp.sh`)
- [Ollama](https://ollama.com) para servir o modelo exportado (pode ser no Windows)

Verifique o driver: `nvidia-smi` deve funcionar dentro do WSL2.

## Instalação

```bash
uv sync                         # cria .venv e instala tudo (PyTorch cu128 incluso)
uv run ft system-info           # confirma CUDA, bf16, VRAM
uv run ft gpu-info
```

## Configuração

Tudo é YAML em `configs/`, um grupo por domínio:

```
configs/
├── config.yaml        # defaults raiz
├── model/             # qwen3-0.6b.yaml (adicione outros modelos aqui)
├── training/          # sft.yaml: método, épocas, batch, context length...
├── optimizer/         # paged_adamw_8bit.yaml, adamw.yaml
├── scheduler/         # cosine.yaml, linear.yaml
├── dataset/           # example-chat.yaml
├── lora/              # qlora.yaml (4-bit), lora.yaml
├── logging/ evaluation/ experiment/ hardware/ tokenizer/ export/
```

Qualquer valor pode ser sobrescrito na linha de comando (sintaxe Hydra):

```bash
uv run ft train model=qwen3-0.6b lora=qlora optimizer.learning_rate=1e-4 lora.r=8
```

Para um modelo novo, crie `configs/model/<nome>.yaml`:

```yaml
name: Qwen/Qwen3-1.7B     # qualquer modelo compatível com Transformers
revision: main
cache_dir: models
trust_remote_code: false
precision: auto            # auto → bf16 se a GPU suportar
attention: auto            # auto → flash-attn se instalado, senão SDPA
```

## Datasets

Formatos aceitos: **JSON, JSONL, CSV, Parquet, Arrow, HF Hub** (e streaming para treino direto).

Schemas de registro (`dataset.record_schema`):

- `chat` — `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}`
- `alpaca` — `{"instruction": ..., "input": ..., "output": ...}`
- `text` — `{"text": "..."}` (continual pretraining)

Novos layouts entram como novos `SchemaAdapter` em `preprocessing/schema_adapters.py`, sem tocar no resto do pipeline.

```bash
uv run python scripts/generate_example_dataset.py   # dataset de exemplo (PT-BR)
uv run ft validate                                  # valida os registros
uv run ft prepare-dataset                           # pipeline completo + cache
uv run ft tokenize                                  # só estatísticas de tokens
```

Metadata automática por dataset: SHA256 da fonte, nº de registros/tokens, média e máximo de tokens, idioma, duplicatas removidas, splits e datas — salva em `datasets/processed/<nome>-<versão>/metadata.json`.

## Treinamento

```bash
uv run ft download-model                    # baixa o modelo base (idempotente)
uv run ft train                             # treino com os defaults (QLoRA 4-bit)
uv run ft train training.max_steps=100      # overrides pontuais
uv run ft resume --run runs/<run>           # retoma do último checkpoint
```

Cada treino cria `runs/<timestamp>_<modelo>/`:

```
runs/2026-07-02_15-30-18_qwen3-0.6b/
├── manifest.json      # UUID, git commit, hashes (config/dataset), versões, GPU, status, métricas finais
├── configs/resolved.yaml
├── checkpoints/       # checkpoint-N (best/last conforme eval_loss)
├── metrics/           # training_log.jsonl + training_log.csv (loss, LR, VRAM, tokens/s...)
├── logs/tensorboard/
├── plots/             # loss.png, perplexity.png, tokens_per_second.png
├── evaluation/evaluation.json
├── model/             # adapter (ou modelo completo) + tokenizer
└── exported/          # lora/, merged/, model-*.gguf, Modelfile
```

Dashboards:

```bash
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db   # http://localhost:5000
uv run tensorboard --logdir artifacts/tensorboard            # http://localhost:6006
```

### GPU de 6GB (defaults do projeto)

Os defaults já assumem pouca VRAM: QLoRA 4-bit NF4 + double quant, `paged_adamw_8bit`, gradient checkpointing, micro-batch 1 com gradient accumulation 8, context length 1024. Para GPUs maiores, aumente `training.micro_batch_size` e `training.context_length`.

## Avaliação e comparação

```bash
uv run ft evaluate --run runs/<run>      # perplexity + benchmark + comparação com o base + plots
uv run ft benchmark --run runs/<run>     # só velocidade de inferência
uv run ft list-experiments               # tabela de todos os runs
uv run ft show-experiment --run runs/<run>
uv run ft compare                        # compara todos e gera runs/reports/comparison_*.md
uv run ft compare <run-a> <run-b>        # compara runs específicos
```

## Exportação para Ollama

```bash
bash scripts/setup_llamacpp.sh           # 1x: clona e builda llama.cpp (conversão GGUF)
uv run ft export --run runs/<run>        # lora/ + merged/ + model-q4_k_m.gguf + Modelfile
uv run ft merge-lora --run runs/<run>    # só o merge
```

Depois (no Windows ou onde o Ollama estiver):

```bash
ollama create meu-modelo -f runs/<run>/exported/Modelfile
ollama run meu-modelo
```

Quantização GGUF configurável em `configs/export/gguf.yaml` (`f16`, `q8_0`, `q5_k_m`, `q4_k_m`). Parâmetros do Modelfile (temperature, top_p, num_ctx, system prompt, nome) em `export.ollama`.

## Docker

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm train train training.max_steps=100
docker compose -f docker/docker-compose.yml up mlflow tensorboard   # dashboards
```

Requer [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). No WSL2, o Docker Desktop já expõe a GPU.

## Testes e qualidade

```bash
uv run pytest                    # unit (rápidos, sem GPU/rede)
uv run pytest -m integration     # baixa modelo real
uv run ruff check src tests scripts
uv run black --check src tests scripts
uv run mypy
```

## Troubleshooting

| Problema | Causa provável | Solução |
|---|---|---|
| `CUDA was requested but is not available` | Driver/WSL sem GPU | `nvidia-smi` no WSL; atualize o driver NVIDIA no Windows |
| OOM (CUDA out of memory) | VRAM insuficiente | Reduza `training.context_length`, use `lora=qlora`, `training.micro_batch_size=1` |
| Download do modelo lento/interrompido | Rede | Rode `ft download-model` de novo — o download é retomado |
| `llama.cpp convert script not found` | GGUF sem setup | `bash scripts/setup_llamacpp.sh` |
| Modelo gated no HF (Llama, Gemma) | Licença | `uv run huggingface-cli login` e aceite a licença na página do modelo |
| Treino em CPU | CUDA indisponível | Apenas para debug; verifique instalação do driver |

## Estendendo

- **Novo método de treino**: nova classe em `training/strategies.py` registrada em `_STRATEGIES`.
- **Novo schema de dataset**: novo `SchemaAdapter` + case em `get_schema_adapter`.
- **Novo formato de export**: novo módulo em `exporters/` + case em `application/export_model.py`.
- **Novo modelo**: só um YAML novo em `configs/model/`.

Documentação de arquitetura detalhada em [`docs/architecture.md`](docs/architecture.md).
