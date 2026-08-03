# Aurum Market

Motor de descubrimiento de productos sobre un catálogo de **15.000 referencias
en español**. Resuelve dos recorridos de negocio con una única base vectorial:

- **Búsqueda semántica** por intención, con filtro por marca resuelto dentro de
  la consulta.
- **Control de altas duplicadas**, decidido con un umbral calibrado y congelado.

Entregable evaluable del Módulo 10 de Pontia. **No usa ningún LLM en tiempo de
ejecución**: los modelos de embeddings son parte del problema, no un atajo para
resolverlo.

---

## Requisitos

| | Mínimo | Comprobación |
|---|---|---|
| **Docker** | Desktop arrancado | `docker version` responde con servidor |
| **uv** | 0.11+ | `uv --version` |
| **Python** | 3.12 *(uv lo descarga)* | fijado en `.python-version` |
| **Disco** | ~2,5 GB | modelo 1,1 GB · volumen 400 MB · caché 50 MB |
| **RAM** | 8 GB | el modelo ocupa ~1,5 GB al cargar |

No hace falta GPU. Todas las cifras de este README se midieron en CPU, en
Windows 11 con 8 núcleos.

---

## Puesta en marcha

```bash
git clone <url> && cd Actividad_BBDD_Vectorial

make setup          # uv sync + crea .env desde la plantilla
make up             # levanta Qdrant y espera a su healthcheck
uv run aurum doctor # entorno, .env, checksums de los datos y conectividad
```

`doctor` no modifica nada. Si sale todo en verde, el resto funcionará.

### El recorrido completo

⚠️ **El orden importa y no es libre.** Los pasos 1–7 los encadena un solo
comando; el paso 8 va aparte porque modifica la colección de forma irreversible.

```bash
uv run aurum ingest --profile full    # 1  los 15.000 productos
uv run aurum deliver                  # 2  pasos 2 a 7: los tres artefactos
uv run aurum events --apply --twice   # 3  los 24 eventos + idempotencia
```

Para probar sin esperar, `--profile sample` trabaja con 1.500 productos.

---

## Por qué los eventos van al final

Es la decisión menos evidente del proyecto y merece leerse antes de ejecutar
nada.

Los 24 eventos de `datos/eventos_catalogo.csv` están construidos para tocar
**exactamente los productos que las demás pruebas usan como respuesta
correcta**:

| Grupo | A quién afecta |
|---|---|
| 8 `UPSERT` de fichas existentes | las 7 referencias de `altas_desarrollo.csv` |
| 8 `DELETE` | **5 de los 6** candidatos que `resultados_duplicados.csv` señala |
| 8 `UPSERT` nuevos (`AURUM-NEW-*`) | ninguna métrica: no figuran en los juicios |

Aplicarlos antes de `deliver` haría que la entrega apuntara a productos que ya
no existen. Por eso `aurum deliver` **nunca** los aplica, y hay un test que lo
verifica leyendo su código fuente — ninguna salida del programa delataría lo
contrario.

**Consecuencia práctica:** una vez aplicados, volver al estado base exige
reingerir. El camino es borrar los 8 `AURUM-NEW-*` y ejecutar `aurum ingest`
otra vez: la ingesta devuelve los 8 productos borrados y deshace los títulos
revisados.

Justificación completa en [ADR-001](specs/decisiones/ADR-001-orden-canonico-de-ejecucion.md).

---

## Tiempos

En la máquina descrita arriba. `medido` son cronometrajes reales; `estimado`
son extrapolaciones a partir de ellos.

| Operación | Coste | |
|---|---|---|
| Descarga del modelo e5-base | ~1,1 GB, una vez, según conexión | — |
| Carga del modelo en memoria | **16 s**, una vez por proceso | `medido` |
| Codificación de textos | **21 textos/s** en CPU | `medido` |
| `aurum doctor` | **3,5 s** | `medido` |
| `aurum verify` | **2,3 s** | `medido` |
| `make test` | **58 s** · 379 pruebas | `medido` |
| `aurum ingest --profile full` · 1ª vez | ~13 min, 12 de ellos codificando | `estimado` |
| `aurum ingest --profile full` · con caché | ~1 min | `estimado` |
| `aurum deliver` | ~3 min | `estimado` |
| `aurum events --apply --twice` | ~30 s | `estimado` |

La codificación es el término dominante de la primera ingesta. Los vectores se
cachean en `.artifacts/embeddings/` con clave por modelo, rol y digest del
texto, así que una segunda ingesta no vuelve a codificar nada — de ahí que pase
de trece minutos a uno.

---

## Comandos

| Comando | Qué hace |
|---|---|
| `aurum doctor` | Verifica entorno, `.env`, checksums SHA-256 y conectividad. No modifica nada |
| `aurum ingest --profile {sample,full}` | Ingesta por lotes. Repetirla **no** aumenta el recuento |
| `aurum verify` | Recuento, dimensión, distancia y estado real de indexación |
| `aurum search "consulta" [--brand M] [--top-k N]` | Búsqueda ad-hoc para inspección manual |
| `aurum experiment [--profile]` | Matriz E0–E3 de representaciones, con el oráculo exacto |
| `aurum evaluate [--sweep-ef]` | Métricas, fidelidad ANN y latencia |
| `aurum duplicates calibrate` | Barrido de umbral sobre `altas_desarrollo.csv` |
| `aurum duplicates predict` | Decisiones sobre el conjunto ciego, con el umbral congelado |
| **`aurum deliver`** | **Comando único que regenera los tres artefactos** |
| `aurum events [--apply] [--twice]` | Sin `--apply` solo clasifica. Con él, aplica los 24 eventos |
| `aurum reset` | Destructivo. Requiere doble confirmación explícita |

Atajos en el `Makefile`: `setup`, `up`, `down`, `doctor`, `test`, `lint`,
`ingest`, `evaluate`, `deliver`, `clean`, `reset`.

---

## Configuración

Todo vive en `.env`, creado por `make setup` desde `.env.example`. Lo que más
importa:

```bash
AURUM_EMBEDDING_MODEL=intfloat/multilingual-e5-base   # 768 dim
QDRANT_COLLECTION=aurum-market-catalogo

AURUM_HNSW_M=24                 # grafo: exige reindexar si cambia
AURUM_HNSW_EF_CONSTRUCT=120
AURUM_HNSW_EF_SEARCH=256        # por consulta: se puede barrer sin reconstruir

AURUM_INDEXING_THRESHOLD=1000   # KILOBYTES por segmento, no vectores
AURUM_FULL_SCAN_THRESHOLD=1000

AURUM_ALLOW_RESET=false         # ambas hacen falta para borrar algo
AURUM_CONFIRM_CLEANUP=
```

La configuración **de la ejecución final** está congelada aparte, en
[`config/final.yaml`](config/final.yaml). Ese fichero es un artefacto de
entrega: declara con qué parámetros exactos se produjeron los resultados, y
cambiar un valor obliga a regenerarlos.

---

## Cuando algo falla

### `Qdrant en http://localhost:6333 — ConnectionError`

Docker Desktop no está arrancado, o el contenedor no está levantado.

```bash
docker version   # ¿responde el servidor?
make up
```

Qdrant es cliente-servidor, no una base embebida: **tiene que estar corriendo**
para cualquier operación sobre la colección.

### El puerto 6333 ya está ocupado

Otro Qdrant vive ahí. Párralo, o cambie el puerto publicado en
`deploy/qdrant/compose.yaml` y ajuste `QDRANT_URL` en el `.env`.

### `La colección 'aurum-market-catalogo' no está lista`

No se ha ingerido todavía. `uv run aurum ingest --profile full`.

Es un error deliberado y no una lista vacía: una colección sin ingerir y una
consulta sin resultados son indistinguibles a simple vista, y confundirlas
llevaría números sin sentido hasta las métricas.

### La primera consulta tarda 16 segundos

Es la carga del modelo en memoria, una vez por proceso. Las consultas
siguientes van a ~60 ms.

### `aurum verify` dice que hay 0 vectores indexados

Qdrant **responde igual**, por fuerza bruta, y los indicadores salen en verde.
Pero `m` y `ef_construct` no estarían interviniendo y cualquier medida de
fidelidad ANN sería trivialmente 1,0.

La causa suele ser `AURUM_INDEXING_THRESHOLD` con el valor por defecto de
Qdrant. Está en **kilobytes y por segmento**, no en número de vectores: 1.500
vectores de 768 dimensiones son 4.500 KB que, repartidos en tres segmentos, no
llegan al límite de 20.000. Ver [ADR-006](specs/decisiones/ADR-006-umbrales-de-indexacion.md).

### `aurum reset` se niega a borrar

Es lo esperado. Hacen falta **las dos** variables, y el nombre escrito a mano:

```bash
AURUM_ALLOW_RESET=true
AURUM_CONFIRM_CLEANUP=aurum-market-catalogo
```

Ninguna operación destructiva está habilitada por defecto, y ningún recurso
fuera del prefijo `aurum-market` es alcanzable.

### `UnicodeEncodeError: 'charmap' codec can't encode…`

No debería ocurrir: la CLI reconfigura su salida para degradar un carácter no
representable a `?` en vez de abortar. Si aparece en una consola antigua,
`chcp 65001` la pasa a UTF-8.

### Los checksums de `datos/` no coinciden

Casi siempre es `core.autocrlf` reescribiendo los CSV con finales de línea de
Windows. El repositorio lo previene con `datos/** -text` en `.gitattributes`;
si ese fichero se perdiera, los checksums dejarían de coincidir en cualquier
clon.

---

## Resultados

Sobre el conjunto de desarrollo, con la configuración congelada:

| Métrica | Valor | Lectura |
|---|---|---|
| nDCG@10 | **0,5422** | calidad del orden, con relevancia graduada |
| Recall@10 | **0,2481** | sobre un **techo de 0,5188**: es el 48 % de lo alcanzable |
| MRR@10 | **0,9167** | el primer acierto llega casi siempre en la posición 1 |
| Fidelidad ANN | **1,0000** | el índice no pierde ni un candidato frente al oráculo exacto |
| Latencia p50 / p95 | **59 / 85 ms** | 2 calentamientos, 5 repeticiones, 8 consultas |
| Duplicados P/R/F1 | **1,0 / 1,0 / 1,0** | sobre desarrollo, umbral 0,9191 |
| Consultas filtradas | **10/10** de la marca | en las 4, sin intrusos |

**El techo de Recall no es un tecnicismo.** Con ~25 productos relevantes por
consulta y solo 10 posiciones, ningún sistema puede pasar de 0,5188. Un 0,2481
sin ese contexto parece un fracaso; con él, es el 48 % del máximo posible.

**Que la fidelidad sea 1,0 tiene una consecuencia fuerte:** el índice no es
responsable de ningún fallo de ranking. Todo el error observado es de la
representación, y eso se demuestra en `.artifacts/attribution.json` en lugar de
suponerse.

---

## Estructura

```
specs/           constitución, requisitos, plan, tareas, contratos y ADR
datos/           los 10 ficheros + manifest.json con sus SHA-256
src/aurum_market/
  contracts.py   los tipos del dominio: hacen ilegal el estado inválido
  data.py        carga validada · saneado de nulos en un único punto
  text.py        composición del texto · embeddings.py  encoder E5 con caché
  store/         Protocol + Qdrant (motor) + NumPy (oráculo exacto)
  search.py      la interfaz común de recuperación
  duplicates.py  regla calibrada · events.py  mutaciones ordenadas
  evaluation/    métricas · fidelidad · latencia · atribución de fallos
  artifacts.py   valida contra el contrato ANTES de escribir
tests/           379 pruebas · marcadores `integration` y `slow`
resultados/      los tres artefactos de entrega
docs/            arquitectura e informe
```

Las pruebas sin marcador corren sin Docker. Las `integration` necesitan
`make up` y trabajan sobre colecciones propias, nunca sobre la de la entrega.

```bash
make test                                  # todo
uv run pytest -m "not integration"         # sin Docker
uv run pytest tests/test_entrega.py        # los 7 puntos de la checklist
```

---

## Para entender las decisiones

| Documento | Qué contiene |
|---|---|
| [docs/arquitectura.md](docs/arquitectura.md) | Diagramas y por qué cada frontera existe |
| [specs/00_constitution.md](specs/00_constitution.md) | Los 14 principios que resuelven los conflictos de criterio |
| [specs/01_spec.md](specs/01_spec.md) | Los 29 requisitos, su estado y los hallazgos del desarrollo |
| [specs/decisiones/](specs/decisiones/) | 8 ADR con las alternativas descartadas y por qué |
| [docs/informe/](docs/informe/) | El informe de la actividad |

El desarrollo siguió **Spec Driven Development**: ningún código existe sin un
requisito que lo justifique, y ningún requisito se cierra sin evidencia
ejecutable.
