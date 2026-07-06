const database = db.getSiblingDB("Greenlegis");

const analise = database.analises.findOne({ "Normas.0": { $exists: true } });
print("=== campos de analises ===");
printjson(Object.keys(analise));

const condicao = database.condicoes_analises.findOne({});
print("=== campos de condicoes_analises ===");
printjson(Object.keys(condicao));

const norma = database.normas.findOne({});
print("=== campos de normas ===");
printjson(Object.keys(norma));

print("=== exemplo de analise (sem TextoPuro) ===");
delete analise.TextoPuro;
printjson(analise);
