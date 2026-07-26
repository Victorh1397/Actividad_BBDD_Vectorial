# ADR-004 · Umbral de relevancia para Recall@10 y MRR@10

- **Estado:** aceptada
- **Requisitos afectados:** RF-19
- **Principios:** [P-05](../00_constitution.md)

## Contexto

El enunciado fija el mapeo de relevancia graduada para nDCG:

| Etiqueta ESCI | Significado | Relevancia |
|---|---|---|
| `E` · Exact | El producto satisface la consulta | 3 |
| `S` · Substitute | Producto distinto pero aceptable como sustituto | 2 |
| `C` · Complement | Producto complementario o accesorio | 1 |
| `I` · Irrelevant | No relacionado | 0 |

nDCG usa la escala completa, pero **Recall@10 y MRR@10 son métricas binarias**:
necesitan una frontera que separe "relevante" de "no relevante". El enunciado deja
la elección abierta pero impone dos condiciones: declararla con claridad y **no
cambiarla silenciosamente entre experimentos**.

## Decisión

Se considera relevante `relevancia >= 2`, es decir **Exact y Substitute**.
Complement e Irrelevant cuentan como no relevantes.

El valor se declara en `config/final.yaml` como `relevance_threshold: 2.0`, se
escribe en `metricas_desarrollo.json` junto a las métricas, y se pasa
explícitamente a cada llamada de evaluación — nunca por defecto implícito.

## Justificación

La frontera debe reflejar la intención de negocio de Aurum Market: alguien que
busca *"funda ipad air 4 sin tapa"* quiere una funda. Un **Substitute** —otra
funda de características algo distintas— resuelve la necesidad y es un resultado
legítimo en el top-10. Un **Complement** —un limpiador de pantalla, un stylus— es
un producto que *acompaña* a la compra pero no la satisface.

Contar los Complement como aciertos inflaría Recall@10 sin que la persona
encuentre lo que buscaba: la métrica mejoraría mientras la experiencia empeora.
Ese es exactamente el tipo de métrica decorativa que el enunciado penaliza
("La evaluación debe servir para decidir, no para decorar el informe").

El umbral en 2 también mantiene coherencia con nDCG: los Complement siguen
aportando ganancia (relevancia 1) en la métrica graduada, así que el sistema no
es ciego a ellos — simplemente no se les concede el estatus de acierto en las
métricas binarias.

## Consecuencias

- Recall@10 y MRR@10 serán **más bajos** que con umbral 1. Es el precio de medir
  algo útil, y se advierte en el informe para que las cifras no se comparen con
  publicaciones que usen otra frontera.
- El denominador de Recall@10 es el número de documentos juzgados con
  `relevancia >= 2` para esa consulta. Si una consulta no tuviera ninguno, su
  recall es 0 por definición y se señala en el análisis por consulta.
- El valor queda congelado: cambiarlo obliga a re-ejecutar **todos** los
  experimentos, no solo el afectado ([P-05](../00_constitution.md)).

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| `>= 1` (E, S y C relevantes) | Infla el recall contando accesorios como aciertos. Mide cobertura temática, no satisfacción de la intención. |
| `>= 3` (solo Exact) | Demasiado estricto para un catálogo de marketplace donde el sustituto es un resultado comercialmente válido; además muchas consultas tendrían muy pocos positivos, haciendo la métrica inestable con solo 8 consultas. |
| Reportar ambos umbrales | Duplica la tabla de métricas y diluye la decisión. El enunciado pide declarar una elección y sostenerla, no evitar elegir. |
