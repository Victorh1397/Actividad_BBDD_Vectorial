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
| T-015 | ADR-001..004 | — | `specs/decisiones/` | revisión | `[ ]` |

## Fase 2 · Datos y contratos

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-020 | Dataclasses `CatalogRecord`, `SearchHit`, `CatalogEvent`, `DuplicateDecision` | RF-01, RF-12 | `contracts.py` | `test_search_hit_contract` | `[ ]` |
| T-021 | Carga validada de los 8 CSV contra `manifest.json` | RF-07 | `data.py` | `test_manifest_counts_match` | `[ ]` |
| T-022 | Saneado uniforme de nulos | RF-03 | `data.py` | `test_missing_values_never_render_as_nan` | `[ ]` |
| T-023 | Verificar contrato UUIDv5 del `record_id` | RF-07 | `data.py` | `test_record_id_follows_uuid5_contract` | `[ ]` |
| T-024 | Resolver `workload_id` ↔ `query_id` ↔ `product_id` en qrels | RF-19 | `data.py` | `test_qrels_join_has_no_orphans` | `[ ]` |
| T-025 | Perfiles `sample` / `full` | — | `data.py` | `test_profiles_load_expected_counts` | `[ ]` |

## Fase 3 · Representación y baseline

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-030 | Estrategias de composición del texto | RF-03, RF-06 | `text.py` | `test_text_strategies` | `[ ]` |
| T-031 | Encoder E5 con prefijos y L2 | RF-04 | `embeddings.py` | `test_e5_prefixes_and_normalization` | `[ ]` |
| T-032 | Caché `.npy` con checksum y metadatos | RF-04 | `embeddings.py` | `test_embedding_cache_roundtrip` | `[ ]` |
| T-033 | Métricas nDCG@10 / Recall@10 / MRR@10 graduadas | RF-19 | `evaluation/metrics.py` | `test_metrics_against_hand_computed_values` | `[ ]` |
| T-034 | Baseline TF-IDF | RF-02 | `baselines.py` | fila `E0` de `aurum experiment` | `[ ]` |
| T-035 | Oráculo exacto NumPy | RF-20 | `store/exact_store.py` | `test_exact_store_matches_bruteforce` | `[ ]` |
| T-036 | Matriz de experimentos E0–E3 | RF-06 | `cli.py`, `evaluation/` | `.artifacts/experiments/*.json` válidos | `[ ]` |

## Fase 4 · Almacén vectorial e ingesta

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-040 | `Protocol VectorStore` + excepciones propias | RF-01, RF-15 | `store/base.py` | `test_edge_cases` | `[ ]` |
| T-041 | Colección Qdrant: esquema, HNSW, payload index | RF-07, RF-08 | `store/qdrant_store.py` | `test_collection_schema` | `[ ]` |
| T-042 | Ingesta por lotes con `upsert` sobre `record_id` | RF-09 | `ingest.py` | **`test_double_ingest_keeps_count`** | `[ ]` |
| T-043 | Verificación de recuento e indexación con espera acotada | RF-10 | `ingest.py` | `aurum verify` | `[ ]` |
| T-044 | `validate_resource_name` + `reset` con doble confirmación | RF-18 | `store/qdrant_store.py` | **`test_cleanup_is_disabled_by_default`** | `[ ]` |

## Fase 5 · Recuperación y filtros

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-050 | `search()` global con `top_k` | RF-01, RF-13 | `search.py` | `test_top_k_is_honoured` | `[ ]` |
| T-051 | Filtro de marca nativo con `Filter`/`FieldCondition` | RF-14 | `search.py` | **`test_filtered_queries_never_leak_other_brands`** | `[ ]` |
| T-052 | Casos límite: colección vacía, filtro sin resultados, motor caído | RF-15 | `search.py` | `test_edge_cases` | `[ ]` |
| T-053 | Fidelidad ANN vs oráculo + barrido de `hnsw_ef` | RF-20 | `evaluation/fidelity.py` | `ann_fidelity_at_10` en métricas | `[ ]` |
| T-054 | Latencia p50/p95 con calentamiento | RF-21 | `evaluation/latency.py` | `latency_p50_ms`, `latency_p95_ms` | `[ ]` |

## Fase 6 · Operaciones de catálogo

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-060 | Aplicar 24 eventos por `sequence`, distinguiendo tipos | RF-16 | `events.py` | `test_events_apply_in_sequence_order` | `[ ]` |
| T-061 | Idempotencia de la reaplicación | RF-16 | `events.py` | **`test_events_are_idempotent`** | `[ ]` |
| T-062 | Visibilidad por ID y por búsqueda, con espera acotada | RF-16 | `events.py` | `test_visibility_after_each_operation_type` | `[ ]` |

## Fase 7 · Duplicados

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-070 | Generación de candidatos top-2 vía base vectorial | RF-17 | `duplicates.py` | `test_candidates_come_from_the_vector_store` | `[ ]` |
| T-071 | Regla score + margen (+ marca / léxico si aporta) | RF-17 | `duplicates.py` | `test_rule_is_deterministic` | `[ ]` |
| T-072 | Barrido de umbral sobre desarrollo + precision/recall/F1 | RF-23 | `duplicates.py` | `aurum duplicates calibrate` | `[ ]` |
| T-073 | Congelar umbral en `config/final.yaml` | RF-17 | `config/final.yaml` | revisión antes del paso 7 | `[ ]` |
| T-074 | Análisis separado de FP y FN con coste de negocio | RF-23 | informe | sección del informe | `[ ]` |
| T-075 | Predicción sobre `altas_evaluacion.csv` | RF-17 | `duplicates.py` | **`test_positive_prediction_names_a_candidate`** | `[ ]` |

## Fase 8 · Evaluación y artefactos

| ID | Tarea | RF | Ficheros | Cierra con | Estado |
|---|---|---|---|---|---|
| T-080 | Escritura + validación de los 3 artefactos | RF-25 | `artifacts.py` | `test_artifacts_validate_against_contracts` | `[ ]` |
| T-081 | Rankings ciegos: 10 IDs únicos y válidos | RF-25 | `artifacts.py` | **`test_blind_rankings_have_ten_unique_valid_ids`** | `[ ]` |
| T-082 | Atribución de ≥3 fallos a capa | RF-24 | `evaluation/attribution.py` | `.artifacts/attribution.json` | `[ ]` |
| T-083 | `aurum deliver` como comando único | RF-28 | `cli.py` | **`test_deliver_is_the_single_entry_point`** | `[ ]` |
| T-084 | `config/final.yaml` con la configuración de la ejecución final | RF-25 | `config/final.yaml` | revisión | `[ ]` |

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
T-042, T-044, T-051, T-061, T-075, T-081, T-083, T-093.
