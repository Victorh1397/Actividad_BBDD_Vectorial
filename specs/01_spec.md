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

Un RF pasa a `cerrado` cuando existe evidencia ejecutable, no cuando "el código
está escrito". `parcial` significa que parte del criterio de aceptación ya está
cubierta con evidencia y el resto depende de una fase posterior.

**Última actualización:** cierre de la Fase 5 · 303 tests en verde.

| RF | Estado | Evidencia |
|---|---|---|
| RF-01 | **`cerrado`** | `Retriever` es un `Protocol` que cumplen TF-IDF, el oráculo exacto y Qdrant indistintamente; `aurum search` expone la interfaz con `product_id`, posición, título, metadatos y score con su semántica. |
| RF-02 | **`cerrado`** | Baseline TF-IDF evaluado con las mismas métricas, consultas y texto que el sistema denso: `E0` con nDCG@10 0,6198. Pliega acentos, imprescindible en español. |
| RF-03 | **`cerrado`** | Tres estrategias implementadas y documentadas en `text.py`; saneado en un único punto con 9 casos en `test_data_integrity.py` y las 44 marcas / 549 colores ausentes verificados. |
| RF-04 | **`cerrado`** | Prefijos `query:`/`passage:` aplicados solo a modelos E5, idempotentes; vectores L2 y `float32` verificados contra el modelo real (`test_embeddings.py`). |
| RF-05 | `parcial` | Razonado en [02_plan.md](02_plan.md) §3 y comprobado en código: `ExactVectorStore` rechaza vectores sin normalizar porque el producto interno dejaría de ser el coseno. Falta llevarlo al informe. |
| RF-06 | **`cerrado`** | Cuatro configuraciones que aíslan una variable cada una, con análisis en [ADR-005](decisiones/ADR-005-modelo-de-embeddings.md). Artefactos en `.artifacts/experiments/E{0..3}.json`, validados contra su contrato. |
| RF-07 | **`cerrado`** | Colección verificada en vivo: 15.000 puntos, dimensión 768, distancia `Cosine`, ID = `record_id`, payload uniforme con índice `KEYWORD` sobre `brand`. |
| RF-08 | **`cerrado`** | `m=24` y `ef_construct=120` **leídos de vuelta del motor**, umbrales de indexación explícitos ([ADR-006](decisiones/ADR-006-umbrales-de-indexacion.md)) y barrido de `ef_search` sobre el catálogo completo: 0,85 → 1,00 de fidelidad entre 16 y 256 ([ADR-007](decisiones/ADR-007-ef-search.md)). |
| RF-09 | **`cerrado`** | Segunda ingesta sobre el perfil `sample`: `variación del recuento: +0`. Cubierto además por `test_double_ingest_keeps_count`. *(Punto 1 de la checklist)* |
| RF-10 | **`cerrado`** | `aurum verify` exige recuento, dimensión y distancia, y reporta `indexed_vectors_count`: 15.000 de 15.000 en 4 segmentos. |
| RF-11 | **`cerrado`** | Verificado destruyendo el contenedor (`docker compose down` → `Removed`) y recreándolo: los 15.000 puntos y su índice sobreviven en el volumen. |
| RF-12 | `parcial` | `SearchHit` rechaza `score_kind="distance"` junto a `higher_is_better=True`, y los tres backends declaran `similarity`. Falta propagarlo al artefacto (Fase 8). |
| RF-13 | **`cerrado`** | `top_k` configurable en los tres backends y en `aurum search`. |
| RF-14 | **`cerrado`** | El filtro viaja como `Filter/FieldCondition` dentro de la consulta, con índice `KEYWORD` sobre `brand`. Las 4 consultas reales devuelven **10/10** de la marca pedida, sin intrusos. *(Punto 2 de la checklist)* |
| RF-15 | **`cerrado`** | Colección inexistente o vacía → `CollectionEmptyError` con instrucción; filtro sin resultados → lista vacía documentada; motor caído → `ProviderUnavailableError` que nombra la URL y sugiere `make up`. |
| RF-16 | `parcial` | Los 24 eventos cargan ordenados y sin huecos; `CatalogEvent` modela que un `DELETE` opera sobre un ID. Falta aplicarlos (Fase 6). |
| RF-17 | `parcial` | `DuplicateDecision` hace imposible un positivo sin `matched_product_id`. Falta la regla y su calibración (Fase 7). |
| RF-18 | **`cerrado`** | `aurum reset` exige `AURUM_ALLOW_RESET` **y** `AURUM_CONFIRM_CLEANUP` con el nombre exacto —probado en vivo: sin ambas, se bloquea y explica qué falta—, y ningún recurso fuera del prefijo `aurum-market` es alcanzable. |
| RF-19 | **`cerrado`** | nDCG@10, Recall@10 y MRR@10 graduadas, contrastadas contra valores calculados a mano en `test_metrics.py`. Umbral ≥ 2 declarado en [ADR-004](decisiones/ADR-004-umbral-de-relevancia.md) y pasado explícitamente en cada llamada. |
| RF-20 | **`cerrado`** | Fidelidad **1,0000** con orden idéntico sobre el catálogo completo: el índice no pierde ni un candidato frente al oráculo. Cualquier fallo de ranking queda por tanto atribuido a la representación. |
| RF-21 | **`cerrado`** | p50 y p95 con calentamiento y repeticiones declarados, separando codificación (0,50 ms) de recorrido completo (81,77 ms), y con el entorno adjunto en `.artifacts/evaluation.json`. |
| RF-22 | **`cerrado`** | Las 4 consultas filtradas: 10/10 de la marca en todas. |
| RF-23 … RF-24 | `pendiente` | — |
| RF-25 | `parcial` | Los seis contratos JSON existen; `experiment_run` ya valida artefactos reales al escribirse. Falta escribir los tres de entrega (Fase 8). |
| RF-26 | `pendiente` | — |
| RF-27 | `parcial` | 254 tests cubren IDs, saneado, contratos, métricas, texto, embeddings y seguridad. Faltan batching, filtros nativos y mutaciones. |
| RF-28 … RF-29 | `pendiente` | — |

### Hallazgos de la Fase 5

11. **El sistema se degrada con las consultas semánticas.** Con *"taladro 24v
    batería"* devuelve taladros; con *"quiero una herramienta inalámbrica
    potente para perforar sin depender de un enchufe"* devuelve guantes y
    estantes de cocina. **El oráculo exacto falla igual**, así que es un fallo
    de **representación**, no del índice. El enunciado plantea las tres
    formulaciones de cada consulta *"para comprobar si el comportamiento se
    mantiene cuando cambia la superficie léxica"*: la respuesta medida es que
    no se mantiene. Probablemente el texto `title_brand_color` (140 caracteres)
    deja al modelo sin contexto para conectar *"perforar"* con *"taladro"*.
    La decisión **no se revisa**: hacerlo mirando el conjunto de evaluación
    violaría [P-04](00_constitution.md). Va al informe como principal
    recomendación de mejora.
12. **A esta escala el índice ANN no compensa.** El oráculo por fuerza bruta en
    NumPy (18,43 ms) es más rápido que Qdrant (40,74 ms) sobre los mismos
    15.000 vectores. HNSW empieza a rentar con órdenes de magnitud más datos.
13. **La latencia medida está dominada por el transporte.** Una operación que
    no busca nada tarda 32,64 ms contra Docker Desktop sobre WSL2, así que de
    los ~81 ms del recorrido completo la búsqueda propiamente dicha son unos
    8 ms. Por eso subir `ef_search` de 16 a 256 apenas cuesta 3 ms y compra
    fidelidad perfecta ([ADR-007](decisiones/ADR-007-ef-search.md)).
14. **Las cifras de la muestra eran optimistas, como se advirtió.** nDCG@10 cae
    de 0,7072 sobre 1.500 productos a 0,5422 sobre 15.000: diez veces más
    distractores compitiendo por las mismas diez posiciones.

### Hallazgos de la Fase 4

8. **Una colección puede responder sin usar su índice.** Tras la primera
   ingesta, Qdrant informaba `points_count=1500`, `status=green` e
   `indexed_vectors_count=0`: contestaba por fuerza bruta con el grafo HNSW sin
   construir. Los umbrales que lo deciden están **en kilobytes y por segmento**,
   no en número de vectores. Sin corregirlo, RF-08 sería decoración, RF-20
   saldría trivialmente 1,0 y RF-21 mediría un escaneo lineal. Ver
   [ADR-006](decisiones/ADR-006-umbrales-de-indexacion.md).
9. **`green` no significa indexado.** Qdrant confirma la escritura de inmediato
   y construye el grafo después, en segundo plano. Cualquier medición que
   dependa del índice debe esperar a `indexed_vectors_count`, no a `status`.
10. **La verificación tiene que leer del motor, no de la configuración.**
    Declarar `m=24` y comprobar que Qdrant lo aplicó son cosas distintas, y el
    enunciado pide la segunda. `aurum verify` muestra los valores tal y como
    los reporta el motor.

### Hallazgos de la Fase 3

5. **La composición del texto pesa más que el modelo.** De E0 a E3 el nDCG@10
   sube +0,087, repartido así: pasar de léxico a denso aporta +0,023 (26 %),
   **cambiar qué texto se codifica aporta +0,045 (52 %)** y duplicar el tamaño
   del modelo aporta +0,019 (22 %). La causa está medida: el campo `text`
   promedia 1.309 caracteres y el **27,2 %** de los productos supera los 512
   tokens del modelo, gastando su presupuesto en relleno de palabras clave.
6. **Adoptar embeddings sin arreglar el texto empeoró el sistema.** `E1` queda
   por debajo del baseline TF-IDF en Recall (0,283 vs 0,303) y en MRR (0,813 vs
   0,917). Solo `E3` supera al baseline en las tres métricas.
7. **El conjunto de desarrollo favorece al baseline léxico.** Sus ocho consultas
   son todas `customer_query`, literales como *"botines marrones mujer tacon
   medio"*. Las doce de evaluación sí traen variantes `semantic` y `context`, de
   modo que la ventaja del sistema denso medida aquí es probablemente un suelo,
   no un techo. Afirmar lo contrario sería extrapolar.

### Hallazgos de la Fase 2

La Fase 2 destapó tres cosas que la especificación no había anticipado. Quedan
registradas aquí porque cambian decisiones posteriores:

1. **Un `DELETE` no lleva ficha.** Los ocho eventos de borrado traen solo
   `record_id`, `product_id`, `locale`, `catalog_version` y `active=False`.
   `CatalogEvent` se rediseñó para que `record` sea opcional: modelar la
   asimetría impide que el aplicador indexe una ficha vacía por accidente.
2. **La distinción alta / actualización no está en los datos.** El fichero solo
   dice `UPSERT` o `DELETE`; los 16 upserts son 8 altas y 8 actualizaciones
   según exista o no el `record_id` en la colección. Esa clasificación es
   responsabilidad del aplicador, y RF-16 la exige explícitamente.
3. **Recall@10 tiene un techo estructural de 0,519.** Medido, no estimado: hay
   198 productos relevantes con umbral ≥ 2 repartidos entre 8 consultas y solo
   diez posiciones disponibles. El techo macro es 0,519 y su dispersión es
   enorme —de **0,256** en `DEV-28703` (39 relevantes) a **1,000** en
   `DEV-33633` (4 relevantes)—, así que la media de Recall@10 dice tanto sobre
   el reparto de los juicios como sobre el sistema. Debe explicarse en el
   informe, y conviene acompañar el macro con el detalle por consulta.
4. **Los eventos no pertenecen al recorrido de medición.** Los 24 eventos están
   construidos para tocar los productos que las otras pruebas usan como
   respuesta correcta: los 8 `DELETE` borran **6 de las 7** referencias de
   `altas_evaluacion.csv`, y ninguno de los 8 `AURUM-NEW-*` que añaden figura en
   los 248 juicios de relevancia. Aplicarlos antes de predecir duplicados haría
   imposible señalar el candidato, incumpliendo el punto 5 de la checklist de
   entrega. Esto obligó a **revisar [ADR-001](decisiones/ADR-001-orden-canonico-de-ejecucion.md)**:
   los eventos pasan a ser el último paso, una prueba operativa aislada.
