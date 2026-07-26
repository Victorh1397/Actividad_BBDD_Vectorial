# ADR-003 · El oráculo exacto es NumPy, no FAISS

- **Estado:** aceptada
- **Requisitos afectados:** RF-20
- **Principios:** [P-13](../00_constitution.md)

## Contexto

RF-20 exige comparar los IDs devueltos por el motor ANN con los de un **oráculo
exacto** para separar la pérdida del índice del error del modelo. La sesión 02
trabajó esto con FAISS (`IndexFlatIP` como ground truth frente a `IndexHNSWFlat`,
`IndexIVFFlat` e `IndexIVFPQ`), así que la opción evidente sería reutilizar
`faiss-cpu`.

## Decisión

El oráculo exacto se implementa como un **producto matricial NumPy** en
`store/exact_store.py`, sin dependencia de FAISS.

## Justificación

Con vectores L2-normalizados, la búsqueda exacta por coseno **es** un producto
matriz-vector seguido de un `argpartition`:

```python
scores = embeddings @ query          # (15000, 384) @ (384,) → (15000,)
top = np.argpartition(-scores, k)[:k]
```

Dimensiones del problema: 15.000 × 384 en `float32` son **23 MB** y el producto
son ~5,8 millones de operaciones — del orden de milisegundos. FAISS resolvería lo
mismo, pero añadiría una dependencia binaria pesada y sensible a la plataforma
(las ruedas de `faiss-cpu` en Windows han sido históricamente frágiles) para
cerrar exactamente cero requisitos adicionales.

El enunciado es explícito: *"No se puntúan funcionalidades ajenas al caso por el
mero hecho de añadir complejidad."*

## Consecuencias

- El oráculo es **auditable de un vistazo**: cinco líneas de álgebra lineal que
  nadie puede acusar de aproximar nada. Eso refuerza su papel de ground truth.
- Se pierde la comparación entre familias de índices que ofrecía FAISS, pero esa
  pérdida ya está asumida y documentada en [ADR-002](ADR-002-motor-vectorial-qdrant.md):
  Qdrant solo implementa HNSW, así que no habría con qué comparar.
- Se reutiliza el patrón de `exact_top_k` de la sesión 03, que ya incorpora el
  filtro por marca sobre el subconjunto de candidatos.
- Si el catálogo creciera un orden de magnitud, el oráculo pasaría a evaluarse
  sobre una muestra de consultas en lugar de sobre todas — que es precisamente
  lo que el enunciado pide ("comparar IDs con un oráculo exacto sobre una muestra").

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| `faiss-cpu` con `IndexFlatIP` | Dependencia binaria pesada, frágil en Windows, y no cierra ningún RF que NumPy no cierre. |
| Búsqueda exacta contra Qdrant desactivando HNSW | Confunde ground truth con sistema evaluado: el oráculo debe ser independiente del motor que se está auditando. |
| `sklearn.neighbors.NearestNeighbors` | Envoltorio innecesario sobre la misma álgebra, con menos control sobre el desempate. |
