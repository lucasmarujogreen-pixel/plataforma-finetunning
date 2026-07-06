const database = db.getSiblingDB("Greenlegis");

const cursor = database.analises.aggregate(
  [
    { $match: { "Normas.0": { $exists: true } } },
    {
      $lookup: {
        from: "condicoes_analises",
        localField: "_id",
        foreignField: "AnaliseId",
        as: "conds",
      },
    },
    { $match: { "conds.0": { $exists: true } } },
    { $addFields: { primeiraNorma: { $arrayElemAt: ["$Normas", 0] } } },
    {
      $lookup: {
        from: "normas",
        localField: "primeiraNorma.NormaId",
        foreignField: "_id",
        as: "norma",
      },
    },
    { $unwind: "$norma" },
    {
      $project: {
        _id: 1,
        tipo_analise: "$Tipo",
        texto_analise: { $ifNull: ["$TextoPuro", ""] },
        complemento: { $ifNull: ["$primeiraNorma.ComplementoPuro", ""] },
        num_normas: { $size: "$Normas" },
        norma_id: "$norma._id",
        titulo: { $ifNull: ["$norma.TituloPuro", ""] },
        resumo: { $ifNull: ["$norma.ResumoPuro", ""] },
        especie: { $ifNull: ["$norma.Especie.Descricao", ""] },
        condicoes: {
          $map: {
            input: "$conds",
            as: "c",
            in: {
              sequencia: "$$c.Sequencia",
              tipo: "$$c.TipoId",
              vinculo: "$$c.VinculoId",
              itens: {
                $map: {
                  input: { $ifNull: ["$$c.ItensFormulario", []] },
                  as: "i",
                  in: {
                    descricao: "$$i.Descricao",
                    marcado: "$$i.Marcado",
                    formulario_id: "$$i.FormularioId",
                  },
                },
              },
              localidades: { $size: { $ifNull: ["$$c.ItensLocalidade", []] } },
            },
          },
        },
      },
    },
  ],
  { allowDiskUse: true }
);

cursor.forEach((doc) => print(JSON.stringify(doc)));
