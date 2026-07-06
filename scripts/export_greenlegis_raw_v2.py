"""PDA-716: export v2 — analyses with the requisito texts that determine conditions.

Forensics showed that each analysis extracts conditions for a specific set of
requisitos (legal requirements), and analyses of the same norma with disjoint
conditions simply cover different requisitos. The original export dropped
RequisitosIds entirely, making the task under-determined. This export joins
the requisito texts, turning norma + requisitos -> condicoes into a well-posed
mapping (validated on NR-38: compactor requisitos -> item 2261, street-sweeping
requisitos -> item 2262).

Usage:
  uv run --with pymongo python scripts/export_greenlegis_raw_v2.py
Environment:
  GREENLEGIS_MONGO_URI (default mongodb://localhost:32768/?directConnection=true)
"""

import json
import os
from pathlib import Path

from pymongo import MongoClient

OUTPUT_PATH = Path("datasets/raw/greenlegis_raw_v2.jsonl")
DEFAULT_URI = "mongodb://localhost:32768/?directConnection=true"

PIPELINE = [
    {"$match": {"Normas.0": {"$exists": True}, "RequisitosIds.0": {"$exists": True}}},
    {
        "$lookup": {
            "from": "condicoes_analises",
            "localField": "_id",
            "foreignField": "AnaliseId",
            "as": "conds",
        }
    },
    {"$match": {"conds.0": {"$exists": True}}},
    {"$addFields": {"primeiraNorma": {"$arrayElemAt": ["$Normas", 0]}}},
    {
        "$lookup": {
            "from": "normas",
            "localField": "primeiraNorma.NormaId",
            "foreignField": "_id",
            "as": "norma",
        }
    },
    {"$unwind": "$norma"},
    {
        "$lookup": {
            "from": "requisitos",
            "localField": "RequisitosIds",
            "foreignField": "_id",
            "as": "reqs",
            "pipeline": [{"$project": {"TextoPuro": 1}}],
        }
    },
    {
        "$project": {
            "_id": 1,
            "tipo_analise": "$Tipo",
            "num_normas": {"$size": "$Normas"},
            "norma_id": "$norma._id",
            "titulo": {"$ifNull": ["$norma.TituloPuro", ""]},
            "resumo": {"$ifNull": ["$norma.ResumoPuro", ""]},
            "especie": {"$ifNull": ["$norma.Especie.Descricao", ""]},
            "requisitos": {
                "$map": {
                    "input": "$reqs",
                    "as": "r",
                    "in": {
                        "id": "$$r._id",
                        "texto": {"$ifNull": ["$$r.TextoPuro", ""]},
                    },
                }
            },
            "condicoes": {
                "$map": {
                    "input": "$conds",
                    "as": "c",
                    "in": {
                        "sequencia": "$$c.Sequencia",
                        "tipo": "$$c.TipoId",
                        "vinculo": "$$c.VinculoId",
                        "itens": {
                            "$map": {
                                "input": {"$ifNull": ["$$c.ItensFormulario", []]},
                                "as": "i",
                                "in": {
                                    "descricao": "$$i.Descricao",
                                    "marcado": "$$i.Marcado",
                                    "formulario_id": "$$i.FormularioId",
                                },
                            }
                        },
                    },
                }
            },
        }
    },
]


def main() -> None:
    uri = os.environ.get("GREENLEGIS_MONGO_URI", DEFAULT_URI)
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    collection = client["Greenlegis"]["analises"]
    count = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for document in collection.aggregate(PIPELINE, allowDiskUse=True):
            handle.write(json.dumps(document, ensure_ascii=False, default=str) + "\n")
            count += 1
    print(f"exportadas {count} analises para {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
