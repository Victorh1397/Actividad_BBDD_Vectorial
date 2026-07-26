# ADR-002 · Qdrant como motor vectorial

- **Estado:** aceptada
- **Requisitos afectados:** RF-07, RF-08, RF-14
- **Principios:** [P-02](../00_constitution.md), [P-09](../00_constitution.md)

## Contexto

El enunciado permite Chroma, Weaviate, Milvus, Qdrant, Pinecone u otra base
vectorial, siempre que la elección sea coherente y el núcleo use su **SDK
nativo**. El bloque "Índice y base de datos" pesa un 25 % y evalúa
específicamente la configuración ANN, el esquema, la ingesta, la persistencia,
los filtros y las mutaciones.

Restricciones del entorno: portátil Windows 11, sin presupuesto cloud, y la
solución debe poder evaluarse sin que el corrector herede costes.

## Decisión

**Qdrant** en local vía Docker Compose, con `qdrant-client` como SDK nativo.

## Justificación

| Criterio | Qdrant |
|---|---|
| Configuración ANN explícita | `HnswConfigDiff(m, ef_construct)` en la colección y `hnsw_ef` en consulta — los tres parámetros que gobiernan el grafo |
| Filtro como operación de BD | `Filter`/`FieldCondition`/`MatchValue` + `create_payload_index` sobre `brand`; el filtro viaja dentro de la consulta |
| Mutaciones | `upsert` sobre UUID, `delete`, `retrieve` por ID — todo lo que exige RF-16 |
| Observabilidad | `points_count` e `indexed_vectors_count` separados, lo que permite verificar el estado de indexación (RF-10) |
| Coste | Cero: imagen local con healthcheck |
| Continuidad con el curso | La sesión 03 ya trae compose y notebook con `m=24, ef_construct=120, hnsw_ef=128` |

El punto decisivo es el segundo: `create_payload_index` hace que el filtrado por
marca sea genuinamente una operación del motor, lo que permite defender RF-14 sin
matices.

## Consecuencias

- Se **pierde** la posibilidad de comparar familias de índices: Qdrant implementa
  solo HNSW para vectores densos, sin equivalente a IVF o IVF-PQ. Esta pérdida se
  documenta explícitamente en [02_plan.md](../02_plan.md) §4, como pide el
  enunciado ("si el proveedor oculta esa decisión, explicad qué control se pierde").
- La palanca de ajuste queda reducida a `hnsw_ef`, que a cambio tiene la ventaja
  de ser **ajustable sin reindexar**. Es el parámetro que barremos para trazar la
  curva fidelidad/latencia (RF-20).
- La evaluación exige Docker levantado. `aurum doctor` lo detecta y lo informa.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| **Chroma persistente** | Cero infraestructura, pero expone la configuración HNSW solo como metadatos de colección y su filtrado es menos explícito. Techo más bajo en el bloque del 25 %. |
| **Milvus Lite** | Es la única que permitiría comparar IVF_FLAT frente a HNSW, lo que conectaría muy bien con la sesión 02. Descartada por madurez en Windows y por un SDK más verboso que añadiría riesgo sin cerrar ningún RF adicional. |
| **Pinecone Cloud** | Requiere credenciales y podría trasladar coste al corrector, que el enunciado pide evitar. El enunciado además declara que no hay ventaja por usar cloud. |
| **Weaviate** | Capaz y con HNSW configurable, pero su modelo de esquema añade conceptos que este caso no necesita ([P-13](../00_constitution.md)). |
