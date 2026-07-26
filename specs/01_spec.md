# Especificación funcional · Aurum Market

**Qué** debe hacer el sistema y **cómo se sabe** que lo hace. El *cómo* técnico
vive en [02_plan.md](02_plan.md); las tareas ejecutables, en [03_tasks.md](03_tasks.md).

Cada requisito tiene un identificador estable `RF-NN`. Ningún fichero de
`src/` existe si no cierra al menos un RF (principio [P-13](00_constitution.md)).

**Estados:** `pendiente` · `en curso` · `cerrado` (con evidencia ejecutable).

---

## 0. El problema

Aurum Market opera un catálogo de 15.000 referencias en español aportadas por
vendedores distintos. La búsqueda actual depende de la coincidencia literal, de
modo que una persona que describe lo que necesita sin usar las palabras del
título no encuentra el producto. Además, llegan fichas nuevas con títulos
reordenados o marcas omitidas que duplican productos ya publicados.

El sistema debe resolver **dos recorridos**:

- **R1 · Descubrimiento semántico.** Dada una consulta en lenguaje natural,
  devolver un top-k ordenado, opcionalmente restringido por metadatos.
- **R2 · Control de altas.** Dada una ficha nueva, recuperar el producto más
  parecido y decidir si es un duplicado que debe revisarse antes de publicar.

**Fuera de alcance:** RAG, generación de respuestas, LLM en tiempo de ejecución,
interfaz web.

---

## 1. Vocabulario

| Término | Significado en este proyecto |
|---|---|
| `record_id` | UUIDv5 estable del catálogo. **Es el ID del punto en la base vectorial.** |
| `product_id` | Identificador comercial (ASIN). **Es el ID que se reporta en los artefactos y con el que se calculan las métricas.** |
| Perfil `sample` | `catalogo_muestra.csv`, 1.500 registros. Desarrollo y depuración. |
| Perfil `full` | `catalogo_productos.csv.gz`, 15.000 registros. **Es el recorrido evaluado.** |
| Oráculo exacto | Búsqueda por producto matricial NumPy sobre todos los vectores, sin ANN. Sirve para separar el error del índice del error del modelo. |
| Score nativo | El valor que devuelve Qdrant, sin transformar. Con COSINE es una similitud en `[-1, 1]`, donde más alto es mejor. |

---

## 2. Requisitos

### Bloque A · Problema y baseline · 10 %

#### RF-01 · Interfaz común de recuperación
El sistema expone una única función que recibe una consulta y devuelve
resultados normalizados.

- **Firma:** `search(query_text: str, *, top_k: int = 10, brand: str | None = None) -> list[SearchHit]`
- Cada `SearchHit` contiene como mínimo: `product_id`, `rank` (1..k), `title`,
  `brand`, `native_score`, `score_kind`, `higher_is_better`.
- Los `rank` son consecutivos desde 1 y los `product_id` no se repiten.
- **Verificación:** `tests/test_search.py::test_search_hit_contract`

#### RF-02 · Baseline léxico comparable
Existe un baseline TF-IDF evaluado con **las mismas métricas, las mismas 8
consultas de desarrollo y el mismo texto** que el sistema denso.

- El baseline no usa prefijos E5 (no le corresponden: son contrato del modelo, no lenguaje).
- **Verificación:** `aurum experiment` incluye la fila `E0` con sus métricas.

---

### Bloque B · Representación vectorial · 20 %

#### RF-03 · Composición del texto documentada
Se documenta qué campos forman el texto que se codifica y cómo se sanean los
valores ausentes.

- Existen al menos **dos** estrategias de composición implementadas y nombradas.
- Un valor ausente produce cadena vacía, nunca `"nan"`, `"None"` ni `"null"`.
- El saneado ocurre **en un solo lugar** (`data.py`).
- **Verificación:** `tests/test_data.py::test_missing_values_never_render_as_nan`

#### RF-04 · Contrato del modelo de embeddings
Los textos se codifican respetando el contrato del modelo elegido.

- Para la familia E5: prefijo `query: ` en consultas y `passage: ` en documentos,
  aplicado automáticamente y **solo** a modelos E5.
- Vectores L2-normalizados y `float32`.
- **Verificación:** `tests/test_embeddings.py::test_e5_prefixes_and_normalization`

#### RF-05 · Métrica, normalización y significado del score
La documentación explica por qué, con vectores L2-normalizados, la similitud
coseno y el producto interno inducen el mismo orden, y qué significa exactamente
el número que devuelve la base de datos.

- **Verificación:** sección "Representación" en [02_plan.md](02_plan.md) y en el informe.

#### RF-06 · Al menos dos configuraciones comparadas con análisis
Se comparan configuraciones que aíslan **una variable cada vez**, y se explica
el resultado. Cambiar el nombre del modelo sin analizar no cuenta.

| Exp | Texto | Modelo | Aísla |
|---|---|---|---|
| E0 | campo `text` | TF-IDF | baseline léxico |
| E1 | campo `text` | e5-small | denso vs léxico |
| E2 | `title + brand + color` | e5-small | efecto de la composición del texto |
| E3 | ganador E1/E2 | e5-base | efecto del tamaño del modelo |

- Todos los experimentos se ejecutan con el **oráculo exacto**, para que la
  comparación mida representación y no índice.
- **Verificación:** `.artifacts/experiments/*.json` con config + métricas + IDs.

---

### Bloque C · Índice y base de datos · 25 %

#### RF-07 · Esquema explícito de la colección
Quedan definidos y documentados: dimensión, métrica, ID, metadatos y política de nulos.

- Dimensión 384 · métrica `COSINE` · ID del punto = `record_id`.
- Payload: `product_id`, `title`, `brand`, `color`, `locale`, `catalog_version`, `active`.
- Los metadatos ausentes se almacenan como cadena vacía, de forma uniforme.
- **Verificación:** `tests/test_store.py::test_collection_schema`

#### RF-08 · Configuración ANN explícita y justificada
La configuración HNSW se fija en código y se relaciona con lo aprendido sobre
familias de índices.

- `m=24`, `ef_construct=120` en la colección; `hnsw_ef=128` en la consulta.
- Se documenta por qué Qdrant no ofrece una familia IVF equivalente y qué
  implica esa restricción.
- Se mide el efecto de `hnsw_ef` sobre fidelidad y latencia (barrido).
- **Verificación:** `aurum evaluate --sweep-ef` produce la curva.

#### RF-09 · Ingesta por lotes e idempotente
Repetir la ingesta completa **no aumenta el recuento**.

- Lotes configurables (por defecto 256).
- Idempotencia por diseño: `upsert` sobre `record_id` ([P-08](00_constitution.md)).
- **Verificación:** `tests/test_ingest.py::test_double_ingest_keeps_count` *(punto 1 de "Antes de entregar")*

#### RF-10 · Verificación previa a aceptar consultas
Antes de servir búsquedas se verifica el recuento y el estado de indexación.

- Se comprueba `points_count` esperado e `indexed_vectors_count`, esperando con
  plazo acotado.
- **Verificación:** `aurum verify` falla con mensaje accionable si no cuadra.

#### RF-11 · Persistencia y reconstrucción
La colección persiste entre reinicios del proceso y puede reconstruirse desde
cero con un comando documentado.

- **Verificación:** procedimiento en el README, probado en la Fase 9.

#### RF-12 · Semántica del score preservada
El score nativo llega hasta el artefacto sin transformaciones ocultas
([P-03](00_constitution.md)).

- **Verificación:** `tests/test_search.py::test_score_semantics_are_declared`

---

### Bloque D · Recuperación y operaciones

#### RF-13 · Búsqueda semántica global
`top_k` configurable; por defecto 10.
- **Verificación:** `tests/test_search.py::test_top_k_is_honoured`

#### RF-14 · Filtro de marca ejecutado por la base de datos
El filtro viaja dentro de la consulta al SDK, con índice de payload sobre la
marca ([P-09](00_constitution.md)).

- Las 4 consultas de `consultas_filtradas.csv` devuelven **exclusivamente** la
  marca pedida.
- **Verificación:** `tests/test_search.py::test_filtered_queries_never_leak_other_brands` *(punto 2 de "Antes de entregar")*

#### RF-15 · Casos límite tratados explícitamente
| Situación | Comportamiento exigido |
|---|---|
| Colección vacía o inexistente | Excepción propia con mensaje accionable, no lista vacía silenciosa |
| Filtro sin resultados | Lista vacía **documentada**, no excepción |
| Motor no disponible | Excepción propia que nombra la URL y sugiere `make up` |

- **Verificación:** `tests/test_search.py::test_edge_cases`

#### RF-16 · Eventos de catálogo
Los 24 eventos de `eventos_catalogo.csv` se aplican **en orden de `sequence`**,
distinguiendo altas, actualizaciones y eliminaciones.

- Reaplicar el fichero completo deja **exactamente** el mismo estado final.
- Se mide la visibilidad de al menos una operación de cada tipo, por lectura por
  ID **y** por consulta vectorial, con espera acotada ([P-10](00_constitution.md)).
- **Verificación:** `tests/test_events.py::test_events_are_idempotent` *(punto 3 de "Antes de entregar")*

#### RF-17 · Regla de decisión de duplicados
La base vectorial es **siempre** el mecanismo de generación de candidatos.

- La regla puede combinar score, margen respecto al segundo candidato, marca o
  comprobación léxica.
- Una predicción positiva **debe** señalar el `product_id` concreto.
- El umbral se calibra con `altas_desarrollo.csv` y se congela antes de tocar
  la evaluación ([P-04](00_constitution.md)).
- **Verificación:** `tests/test_duplicates.py::test_positive_prediction_names_a_candidate` *(punto 5 de "Antes de entregar")*

#### RF-18 · Seguridad de las operaciones
Sin credenciales en el repositorio; limpieza desactivada por defecto y limitada
al prefijo `aurum-market` ([P-11](00_constitution.md)).

- **Verificación:** `tests/test_safety.py::test_cleanup_is_disabled_by_default` *(punto 7 de "Antes de entregar")*

---

### Bloque E · Evaluación y análisis · 30 %

#### RF-19 · Calidad del ranking
`nDCG@10`, `Recall@10` y `MRR@10` sobre las 8 consultas de desarrollo.

- Relevancia graduada `E=3, S=2, C=1, I=0`.
- **Decisión declarada:** para Recall@10 y MRR@10 se considera relevante
  `relevancia >= 2` (Exact y Substitute). Justificación: un *Complement* es un
  producto accesorio, no una respuesta a la intención de compra; contarlo como
  acierto inflaría el recall sin mejorar la experiencia. Esta elección **no
  cambia entre experimentos** ([P-05](00_constitution.md)).
- Métricas macro-promediadas por consulta.
- **Verificación:** `tests/test_metrics.py` con casos de referencia calculados a mano.

#### RF-20 · Fidelidad ANN
Se comparan los IDs devueltos por Qdrant con los del oráculo exacto sobre una
muestra de consultas.

- Se reporta el solapamiento @10 y se separa la pérdida del índice del error del modelo.
- **Verificación:** `aurum evaluate` incluye `ann_fidelity_at_10`.

#### RF-21 · Latencia
`p50` y `p95` declarando entorno, calentamiento y número de repeticiones.

- Describe **esta** ejecución; no se usa para comparar proveedores en
  infraestructuras distintas.
- **Verificación:** `metricas_desarrollo.json` contiene `latency_p50_ms` y `latency_p95_ms`.

#### RF-22 · Filtros
Las 4 consultas filtradas cumplen la marca en el 100 % de los resultados.
Cubierto por RF-14; aquí se reporta como evidencia en el informe.

#### RF-23 · Duplicados
`precision`, `recall` y `F1` sobre desarrollo, con el umbral documentado.

- Se analizan **por separado** falsos positivos y falsos negativos, explicando
  su coste de negocio: un FP bloquea una publicación legítima y genera trabajo
  de revisión manual; un FN publica un duplicado que degrada el catálogo y
  divide las señales de venta entre dos fichas.
- **Verificación:** `aurum duplicates calibrate` produce la tabla de barrido.

#### RF-24 · Atribución de errores
Al menos **tres** fallos representativos atribuidos a una capa con evidencia,
siguiendo la tabla de [P-12](00_constitution.md).

- **Verificación:** `.artifacts/attribution.json` + sección del informe.

---

### Bloque F · Ingeniería y comunicación · 15 %

#### RF-25 · Artefactos de resultados
| Fichero | Contenido |
|---|---|
| `resultados/resultados_busqueda.csv` | `evaluation_id`, `rank` (1..10), `product_id`, `score` — 12 consultas × 10 filas |
| `resultados/resultados_duplicados.csv` | `incoming_id`, `predicted_duplicate`, `matched_product_id`, `score` — 14 filas |
| `resultados/metricas_desarrollo.json` | `ndcg_at_10`, `recall_at_10`, `mrr_at_10`, `latency_p50_ms`, `latency_p95_ms` |
| `docs/arquitectura.md` | Diagrama de arquitectura |
| `config/final.yaml` | Configuración exacta de la ejecución final |

- Cada artefacto se valida contra su esquema en `specs/contracts/` **al escribirse**.
- Los rankings ciegos contienen 10 `product_id` únicos y válidos.
- **Verificación:** `tests/test_entrega.py` *(punto 4 de "Antes de entregar")*

#### RF-26 · README ejecutable en entorno limpio
Requisitos, instalación, variables, comandos, **tiempos aproximados** y solución
de los fallos previsibles.

#### RF-27 · Pruebas mínimas
Cobertura obligatoria: IDs, batching, filtros, mutaciones y formato de resultados.

#### RF-28 · Regeneración con un comando
`uv run aurum deliver` regenera los tres artefactos y las métricas.
- **Verificación:** `tests/test_entrega.py::test_deliver_is_the_single_entry_point` *(punto 6 de "Antes de entregar")*

#### RF-29 · Informe
PDF de hasta 10 páginas sin anexos: problema, alternativas probadas,
arquitectura final, resultados y recomendación. No es una transcripción del código.

---

## 3. Trazabilidad · checklist "Antes de entregar"

Los siete puntos del enunciado, cada uno con el test que lo cierra:

| # | Punto del enunciado | RF | Test |
|---|---|---|---|
| 1 | La ingesta completa puede repetirse sin aumentar el recuento | RF-09 | `test_double_ingest_keeps_count` |
| 2 | Las consultas filtradas nunca devuelven otra marca | RF-14 | `test_filtered_queries_never_leak_other_brands` |
| 3 | Los eventos dejan exactamente el estado esperado | RF-16 | `test_events_are_idempotent` |
| 4 | Los rankings ciegos contienen diez IDs únicos y válidos | RF-25 | `test_blind_rankings_have_ten_unique_valid_ids` |
| 5 | La detección de duplicados señala un candidato cuando predice positivo | RF-17 | `test_positive_prediction_names_a_candidate` |
| 6 | Las métricas pueden regenerarse desde un único comando | RF-28 | `test_deliver_is_the_single_entry_point` |
| 7 | El repositorio no contiene claves, volúmenes ni datos reservados | RF-18 | `test_repository_has_no_secrets_or_reserved_data` |

---

## 4. Estado de los requisitos

| RF | Estado | Evidencia |
|---|---|---|
| RF-01 … RF-29 | `pendiente` | — |

> Esta tabla se actualiza al cerrar cada requisito. Un RF solo pasa a `cerrado`
> cuando existe evidencia ejecutable, no cuando "el código está escrito".
