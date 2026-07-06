# Arquitetura

## Camadas (Clean Architecture)

```
cli (Typer)  →  application (use cases)  →  domain (entidades + ports)
                        ↓
        infrastructure / training / preprocessing / tokenization /
        evaluation / exporters / monitoring   (implementações)
                        ↓
                core (config, enums, exceptions, logging, hashing)
```

Regra de dependência: camadas externas dependem das internas; `domain` e `core` não importam nada das demais.

## Módulos

| Módulo | Conteúdo | Pontos de extensão |
|---|---|---|
| `core/config` | Schemas Pydantic (1 por grupo Hydra) + `load_config` (compose → validate) | Novo grupo = novo schema + pasta em `configs/` |
| `core` | `enums`, `exceptions` (hierarquia `PlatformError`), `logging` (loguru), `hashing` (SHA256 de texto/arquivo/diretório) | — |
| `domain` | `DatasetMetadata`, `ports.ModelStorePort` | Novos ports para novas integrações |
| `application` | `DownloadModel`, `PrepareDataset`, `TrainModel`, `ResumeTraining`, `EvaluateModel`, `ExportModel`, `CompareExperiments` | 1 use case por operação de negócio |
| `infrastructure` | `HuggingFaceModelStore`, `ExperimentManager` (runs + manifests), `environment` (versões/git) | Outros stores/trackers |
| `preprocessing` | `loaders` (json/jsonl/csv/parquet/arrow/hub/streaming), `schema_adapters` (chat/alpaca/text), `stages` (validate→clean→dedup→stats→tokenize→split), `pipeline` (orquestra + cache por hash) | Novo `SchemaAdapter`; novo estágio é uma função `(Dataset, PipelineContext) -> Dataset` |
| `tokenization` | `load_tokenizer` (padding, chat template, pad token) | — |
| `training` | `strategies` (LoRA/QLoRA/FullSFT + registry), `quantization` (BnB), `trainer_factory` (SFTConfig), `callbacks` (métricas → JSONL/CSV/MLflow) | Nova estratégia = subclasse de `TrainingStrategy` registrada em `_STRATEGIES` |
| `evaluation` | `perplexity`, `benchmark`, `plots`, `model_loading` (adapter/full) | Novas métricas |
| `exporters` | `merge` (LoRA→full), `gguf` (llama.cpp), `modelfile` (Ollama) | Novo exportador + case no use case |
| `monitoring` | `hardware` (detecção CUDA/VRAM/bf16/attention), `system_metrics` (NVML/psutil) | — |
| `cli` | 1 módulo de comandos por área, registrados em `app.py` | Novo comando = função + `app.command()` |

## Decisões

- **Hydra + Pydantic**: Hydra dá composição/overrides de YAML; Pydantic (`extra="forbid"`) rejeita chaves desconhecidas e valores inválidos antes de qualquer código de treino rodar. Validações cruzadas (ex.: `qlora` exige `load_in_4bit`) vivem em `AppConfig`.
- **Compose API, não `@hydra.main`**: convive com Typer e é testável.
- **Strategy + registry para métodos de treino**: adicionar método novo não altera o use case.
- **`snapshot_download` sempre**: idempotente e incremental; evita cache parcial ser tratado como completo.
- **Pipeline de dataset como funções puras + contexto**: cada estágio é testável isolado; cache keyed por SHA256(fonte + config + seed).
- **Merge de LoRA em CPU**: evita OOM em GPUs pequenas; modelos maiores dependem de RAM, não de VRAM.
- **Métricas em arquivo por run (JSONL/CSV) além de MLflow/TensorBoard**: o run é autocontido e comparável offline.
- **Reprodutibilidade**: manifest por run com UUID, git commit, hash da config, hash do dataset, seed, versões de bibliotecas, CUDA/driver e GPU.

## Ciclo de vida de um experimento

1. `create_run` — pasta `runs/<ts>_<modelo>/`, `resolved.yaml`, `manifest.json` (`status=running`).
2. Treino — checkpoints em `checkpoints/`, métricas em `metrics/` e trackers.
3. `finalize_run` — `status=completed|failed`, tempo total e métricas finais no manifest.
4. `evaluate` — `evaluation/evaluation.json` + `plots/*.png`.
5. `export` — `exported/` com adapter, merged, GGUF e Modelfile.
6. `compare` — relatório markdown em `runs/reports/`.
