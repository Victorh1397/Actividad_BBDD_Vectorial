# Tareas · Aurum Market

Tablero de trabajo. Cada tarea está trazada a un requisito de
[01_spec.md](01_spec.md) y se cierra con un test o un comando que produce
evidencia observable.

**Estados:** `[ ]` pendiente · `[~]` en curso · `[x]` cerrada

---

## Fase 0 · Fundación

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-001 | Copiar los 8 CSV + `manifest.json` + `README_DATOS.md` | — | `datos/` | `aurum doctor` valida checksums | `[x]` |
| T-002 | `pyproject.toml`, `.python-version`, `.env.example`, `Makefile` | — | raíz | `make setup` | `[x]` |
| T-003 | Compose de Qdrant con prefijo `aurum-market` | RF-18 | `deploy/qdrant/compose.yaml` | `make up` con healthcheck | `[x]` |
| T-004 | `.gitignore`: `.artifacts/`, `datos/profesorado/`, volúmenes | RF-18 | `.gitignore` | `test_repository_has_no_secrets_or_reserved_data` | `[x]` |
| T-005 | `config.py` con carga y validación de `.env` | RF-18 | `src/aurum_market/config.py` | `test_config_rejects_invalid_settings` | `[ ]` |
| T-006 | `aurum doctor`: Python, `.env`, datos, checksums, Qdrant | RF-10, RF-26 | `cli.py` | `uv run aurum doctor` en verde | `[ ]` |

## Fase 1 · Especificación

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-010 | Constitución del proyecto | — | `specs/00_constitution.md` | revisión | `[x]` |
| T-011 | Especificación RF-01..RF-29 | — | `specs/01_spec.md` | los 7 puntos de entrega trazados | `[x]` |
| T-012 | Contratos JSON | RF-25 | `specs/contracts/*.json` | `test_artifacts_validate_against_contracts` | `[x]` |
| T-013 | Plan técnico | — | `specs/02_plan.md` | revisión | `[x]` |
| T-014 | Tablero de tareas | — | `specs/03_tasks.md` | este fichero | `[x]` |
| T-015 | ADR-001..004 | — | `specs/decisiones/` | revisión | `[x]` |

## Fase 2 · Datos y contratos

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-020 | Dataclasses del dominio con invariantes forzadas | RF-01, RF-12, RF-17 | `contracts.py` | `test_contracts.py` (39 tests) | `[x]` |
| T-021 | Carga validada del catálogo contra `manifest.json` | RF-07 | `data.py` | `test_manifest_counts_match_the_statement` | `[x]` |
| T-022 | Saneado uniforme de nulos | RF-03 | `data.py` | `test_missing_values_never_render_as_nan` | `[x]` |
| T-023 | Verificar contrato UUIDv5 del `record_id` | RF-07 | `data.py` | `test_record_id_follows_the_uuid5_contract` | `[x]` |
| T-024 | Resolver `workload_id` ↔ `query_id` ↔ `product_id` en qrels | RF-19 | `data.py` | `test_qrels_join_has_no_orphans` | `[x]` |
| T-025 | Perfiles `sample` / `full` | — | `data.py` | `test_sample_profile_loads_the_expected_count` | `[x]` |
| T-026 | Carga de workloads, altas y eventos de catálogo | RF-16, RF-17, RF-19 | `data.py` | `test_workloads.py` (25 tests) | `[x]` |
| T-027 | Modelar el `DELETE` como operación sobre un ID | RF-16 | `contracts.py` | `test_a_deletion_operates_on_an_id_alone` | `[x]` |

## Fase 3 · Representación y baseline

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-030 | Estrategias de composición del texto | RF-03, RF-06 | `text.py` | `test_text_and_baseline.py::TestTextStrategies` | `[x]` |
| T-031 | Encoder E5 con prefijos y L2 | RF-04 | `embeddings.py` | `test_embeddings.py::TestPrefixes` | `[x]` |
| T-032 | Caché con checksum del texto y metadatos | RF-04 | `embeddings.py` | `test_the_cache_round_trips` | `[x]` |
| T-033 | Métricas nDCG@10 / Recall@10 / MRR@10 graduadas | RF-19 | `evaluation/metrics.py` | `test_metrics.py` (32 tests, valores a mano) | `[x]` |
| T-034 | Baseline TF-IDF | RF-02 | `baselines.py` | fila `E0`: nDCG@10 0,6198 | `[x]` |
| T-035 | Oráculo exacto NumPy | RF-20 | `store/exact_store.py` | `test_text_and_baseline.py::TestExactVectorStore` | `[x]` |
| T-036 | Matriz de experimentos E0–E3 | RF-06 | `experiments.py`, `cli.py` | `.artifacts/experiments/E{0..3}.json` válidos | `[x]` |
| T-037 | Interfaz común `Retriever` | RF-01, RF-13 | `search.py` | `test_satisfies_the_retriever_protocol` | `[x]` |
| T-038 | Techo estructural de Recall@10 | RF-19 | `evaluation/metrics.py` | `test_recall_at_10_has_a_structural_ceiling` | `[x]` |
| T-039 | ADR-005 y `config/final.yaml` congelado | RF-06, RF-25 | `specs/decisiones/`, `config/` | revisión | `[x]` |

## Fase 4 · Almacén vectorial e ingesta

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-040 | `Protocol VectorStore` + excepciones propias | RF-01, RF-15 | `store/base.py` | `test_searching_a_missing_collection_fails_loudly` | `[x]` |
| T-041 | Colección Qdrant: esquema, HNSW, payload index | RF-07, RF-08 | `store/qdrant_store.py` | `test_collection_schema_is_explicit` | `[x]` |
| T-042 | Ingesta por lotes con `upsert` sobre `record_id` | RF-09 | `ingest.py` | **`test_double_ingest_keeps_count`** | `[x]` |
| T-043 | Verificación de recuento e indexación con espera acotada | RF-10 | `ingest.py`, `cli.py` | `aurum verify` + `test_waiting_gives_up_with_an_explanation` | `[x]` |
| T-044 | `validate_resource_name` + `reset` con doble confirmación | RF-18 | `store/qdrant_store.py` | **`test_reset_is_blocked_by_default`** | `[x]` |
| T-045 | Umbrales de indexación explícitos | RF-08, RF-20 | `config.py`, `store/qdrant_store.py` | `test_the_index_is_actually_built` | `[x]` |
| T-046 | Leer la configuración HNSW de vuelta desde el motor | RF-08 | `store/qdrant_store.py` | `test_the_declared_hnsw_configuration_is_applied` | `[x]` |
| T-047 | ADR-006 sobre los umbrales | RF-08 | `specs/decisiones/` | revisión | `[x]` |

## Fase 5 · Recuperación y filtros

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-050 | Interfaz común sobre cualquier almacén + `aurum search` | RF-01, RF-13 | `search.py`, `cli.py` | `test_satisfies_the_retriever_protocol` | `[x]` |
| T-051 | Filtro de marca nativo con `Filter`/`FieldCondition` | RF-14 | `store/qdrant_store.py` | **10/10 en las 4 consultas reales** | `[x]` |
| T-052 | Casos límite: colección vacía, filtro sin resultados, motor caído | RF-15 | `store/` | `test_searching_a_missing_collection_fails_loudly` | `[x]` |
| T-053 | Fidelidad ANN vs oráculo + barrido de `ef_search` | RF-08, RF-20 | `evaluation/fidelity.py` | fidelidad **1,0000** con `ef_search=256` | `[x]` |
| T-054 | Latencia p50/p95 con calentamiento y entorno | RF-21 | `evaluation/latency.py` | `.artifacts/evaluation.json` | `[x]` |
| T-055 | Comando `aurum evaluate` con toda la evidencia | RF-19 … RF-22 | `cli.py` | ejecución sobre el catálogo completo | `[x]` |
| T-056 | ADR-007 sobre `ef_search` | RF-08 | `specs/decisiones/` | revisión | `[x]` |

> **Sobre el orden de estas tres fases.** Siguen el orden de ejecución fijado en
> [ADR-001](decisiones/ADR-001-orden-canonico-de-ejecucion.md), no el orden en
> que se planificaron. Los eventos van **al final** porque sus ocho borrados
> eliminan seis de las siete referencias de `altas_evaluacion.csv`: aplicarlos
> antes destruiría el estado base que necesitan los duplicados y los artefactos.
>
> Los identificadores se renumeraron para que `T-0X0` siga perteneciendo a la
> fase X, aprovechando que ninguna de estas tareas se había ejecutado todavía.
> **A partir de aquí quedan congelados**: si el orden volviera a cambiar, se
> reordenarían las secciones pero no los IDs, porque los commits ya los citan y
> un identificador no puede pasar a significar otra cosa.

## Fase 6 · Duplicados

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-060 | Generación de candidatos top-2 vía base vectorial | RF-17 | `duplicates.py` | `test_candidates_come_from_the_vector_store` | `[x]` |
| T-061 | Regla de decisión y formato de la consulta | RF-17 | `duplicates.py` | `test_the_rule_is_deterministic` | `[x]` |
| T-062 | Barrido de umbral sobre desarrollo + precision/recall/F1 | RF-23 | `duplicates.py` | F1 = 1,0 con TP=7 FP=0 TN=7 FN=0 | `[x]` |
| T-063 | Congelar umbral en `config/final.yaml` | RF-17 | `config/final.yaml` | 0,9191 fijado antes de predecir | `[x]` |
| T-064 | Análisis separado de FP y FN con coste de negocio | RF-23 | `duplicates.py`, [ADR-008](decisiones/ADR-008-umbral-de-duplicados.md) | `test_explains_what_each_error_costs` | `[x]` |
| T-065 | Predicción sobre `altas_evaluacion.csv` | RF-17 | `duplicates.py` | **`test_positive_prediction_names_a_candidate`** | `[x]` |
| T-066 | ADR-008 con la regla y su justificación | RF-17, RF-23 | `specs/decisiones/` | revisión | `[x]` |

## Fase 7 · Evaluación y artefactos

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-070 | Escritura + validación de los 3 artefactos | RF-25 | `artifacts.py` | `test_artifacts_validate_against_contracts` | `[ ]` |
| T-071 | Rankings ciegos: 10 IDs únicos y válidos | RF-25 | `artifacts.py` | **`test_blind_rankings_have_ten_unique_valid_ids`** | `[ ]` |
| T-072 | Atribución de ≥3 fallos a capa | RF-24 | `evaluation/attribution.py` | `.artifacts/attribution.json` | `[ ]` |
| T-073 | `aurum deliver` como comando único | RF-28 | `cli.py` | **`test_deliver_is_the_single_entry_point`** | `[ ]` |
| T-074 | `config/final.yaml` con la configuración de la ejecución final | RF-25 | `config/final.yaml` | revisión | `[ ]` |

## Fase 8 · Operaciones de catálogo

*Prueba operativa aislada. Se ejecuta **después** de `aurum deliver`, cuando los
tres artefactos ya están escritos: modifica la colección de forma irreversible y
volver al estado base exige reingerir.*

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-080 | Aplicar 24 eventos por `sequence`, distinguiendo tipos | RF-16 | `events.py` | `test_events_apply_in_sequence_order` | `[ ]` |
| T-081 | Idempotencia de la reaplicación | RF-16 | `events.py` | **`test_events_are_idempotent`** | `[ ]` |
| T-082 | Visibilidad por ID y por búsqueda, con espera acotada | RF-16 | `events.py` | `test_visibility_after_each_operation_type` | `[ ]` |

## Fase 9 · Informe y cierre

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-090 | Diagrama de arquitectura | RF-25 | `docs/arquitectura.md` | revisión | `[ ]` |
| T-091 | README con tiempos y fallos previsibles | RF-26 | `README.md` | ejecución en clon limpio | `[ ]` |
| T-092 | Informe PDF ≤10 páginas | RF-29 | `docs/informe/` | revisión | `[ ]` |
| T-093 | Auditoría final: sin claves, volúmenes ni datos reservados | RF-18 | — | **`test_repository_has_no_secrets_or_reserved_data`** | `[ ]` |
| T-094 | Actualizar la tabla de estado de `01_spec.md` | — | `specs/01_spec.md` | los 29 RF en `cerrado` | `[ ]` |

---

## Tareas críticas

Las **negritas** de la columna "Cierra con" son los siete puntos de la checklist
"Antes de entregar" del enunciado. Ninguna entrega sale sin esos siete en verde:

| Punto de la checklist | Tarea | Estado |
|---|---|---|
| La ingesta repetida no aumenta el recuento | T-042 | `[x]` |
| Las consultas filtradas nunca devuelven otra marca | T-051 | `[x]` |
| Los eventos dejan exactamente el estado esperado | T-081 | `[ ]` |
| Los rankings ciegos: diez IDs únicos y válidos | T-071 | `[ ]` |
| Un positivo de duplicados señala su candidato | T-065 | `[ ]` |
| Las métricas se regeneran con un único comando | T-073 | `[ ]` |
| Sin claves, volúmenes ni datos reservados | T-044, T-093 | `[x]` |
