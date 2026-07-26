# Aurum Market · guía de trabajo

Entregable evaluable del Módulo 10 (bases de datos vectoriales). Busca productos
por intención sobre un catálogo de 15.000 referencias, filtra por metadatos y
detecta altas duplicadas, apoyado en Qdrant.

El proyecto se desarrolla bajo **Spec Driven Development**. La regla que gobierna
todo lo demás: **no se escribe código que no cierre un requisito de
[specs/01_spec.md](specs/01_spec.md), y no se cierra un requisito sin evidencia
ejecutable.**

## Antes de escribir código, lee

| Documento | Para qué |
|---|---|
| [specs/00_constitution.md](specs/00_constitution.md) | Principios P-01..P-14. Ante un conflicto de criterio, gana el principio. |
| [specs/01_spec.md](specs/01_spec.md) | Requisitos RF-01..RF-29 y sus criterios de aceptación. |
| [specs/02_plan.md](specs/02_plan.md) | Arquitectura, esquema de la colección y **orden canónico de ejecución**. |
| [specs/03_tasks.md](specs/03_tasks.md) | Tablero de tareas T-0NN. |
| [specs/decisiones/](specs/decisiones/) | ADR con las alternativas descartadas y por qué. |

## Comandos

```bash
make setup          # uv sync + copia .env.example a .env
make up             # levanta Qdrant y espera al healthcheck
make down           # para el contenedor. El volumen y los datos sobreviven
uv run aurum doctor # entorno, configuración, checksums y conectividad
make test           # pytest
make lint           # ruff check + ruff format --check
make deliver        # comando único que regenera los tres artefactos (RF-28)
```

Los tests marcados `integration` necesitan `make up` previo. Los marcados `slow`
recorren el catálogo completo o descargan modelos.

## Convenciones

- **Idioma:** documentación, specs y mensajes de usuario en español; nombres de
  código y docstrings en inglés. Es el patrón del material del curso.
- **Commits:** convencionales, en inglés, con trailer de trazabilidad:
  ```
  Refs: RF-07
  Tasks: T-021
  ```
- **Ramas:** las crea el usuario, siempre. Sugiere el nombre y espera.
- **Python 3.12** fijado en `.python-version`. Ejecuta siempre vía `uv run`.
- Estilo `ruff` con `line-length = 88`. Pasa `make lint` antes de dar algo por
  terminado.

## Trampas de este repositorio

Cosas que ya han costado tiempo una vez:

- **Los datos no se editan a mano.** Su suciedad —marcas ausentes, títulos con
  keyword-stuffing— es parte del problema a resolver, y `datos/README_DATOS.md`
  lo prohíbe explícitamente.
- **`.gitattributes` marca `datos/** -text` y eso no se toca.** Con
  `core.autocrlf=true`, git reescribiría los CSV con CRLF y sus checksums SHA-256
  dejarían de coincidir con `datos/manifest.json` en cualquier clon, incluido el
  del corrector.
- **Orden canónico de ejecución.** `eventos_catalogo.csv` hace UPSERT sobre los
  mismos `product_id` que son referencia en `altas_desarrollo.csv`, así que el
  orden calibrar → eventos → predecir cambia los scores. Está fijado en
  [ADR-001](specs/decisiones/ADR-001-orden-canonico-de-ejecucion.md) y no se altera.
- **Un umbral calibrado se congela** en `config/final.yaml` antes de tocar
  cualquier conjunto de evaluación (P-04).
- **Qdrant tiene que estar corriendo** para cualquier operación sobre la
  colección: es cliente-servidor, no una base embebida.

## Qué no hacer

- Ningún LLM en tiempo de ejecución del sistema. Ni RAG, ni reordenación, ni
  etiquetado (P-01). Los modelos de embeddings sí son parte del problema.
- No sustituir el SDK nativo de Qdrant por una capa de abstracción (P-02).
- No comparar ni promediar un score de distancia con uno de similitud (P-03).
- No añadir dependencias ni funcionalidades que no cierren un RF (P-13).
- No habilitar operaciones destructivas por defecto (P-11).

## Material de apoyo

`../Recursos sesiones live/sesion_0{1,2,3}/` contiene el código de las sesiones
del curso, **fuera** de este repositorio. Hay piezas directamente reutilizables:
métricas de ranking graduadas y medición de latencia p50/p95 en la sesión 01,
recall ANN contra oráculo en la sesión 02, y contratos, `wait_until` y el patrón
de oráculo exacto en la sesión 03. Adáptalas al esquema de Aurum; no las
reinventes.
