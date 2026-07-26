# Datos del caso Aurum Market

Aurum Market recibe consultas breves, descripciones funcionales y peticiones con restricciones. El catálogo contiene 15.000 productos y conserva deliberadamente parte de la suciedad habitual de un sistema real: metadatos ausentes, títulos largos, marcas escritas de formas distintas y descripciones con una calidad desigual. No debéis corregir el dataset a mano. Esa variabilidad forma parte del problema que tiene que resolver el sistema.

## El catálogo

`catalogo_productos.csv.gz` contiene la colección completa. `catalogo_muestra.csv` conserva el mismo esquema y reúne 1.500 registros para desarrollar el pipeline, comprobar la conexión o ejecutar pruebas rápidas antes de ingerir el catálogo completo.

| Campo | Significado |
|---|---|
| `record_id` | UUIDv5 estable que debe utilizarse como ID del registro vectorial. |
| `product_id` | Identificador comercial del producto en el dataset de origen. |
| `title` | Título visible del producto. |
| `brand`, `color`, `locale` | Metadatos disponibles para filtrar o presentar resultados. |
| `text` | Representación textual de partida. Puede modificarse si se justifica y se conserva la trazabilidad del experimento. |
| `catalog_version` | Versión de la ficha. |
| `active` | Indica si el producto forma parte del catálogo consultable. |

Los valores vacíos son información ausente, no la cadena literal `"nan"`. La ingesta debe tratarlos siempre de la misma forma.

## Desarrollo y evaluación de la búsqueda

`consultas_desarrollo.csv` contiene ocho consultas y `relevancias_desarrollo.csv` aporta los juicios conocidos para medir el ranking. Las etiquetas ESCI se convierten en relevancia graduada de esta forma:

- `E` · Exact: 3
- `S` · Substitute: 2
- `C` · Complement: 1
- `I` · Irrelevant: 0

`consultas_evaluacion.csv` contiene doce consultas sin relevancias visibles. Para cada `evaluation_id` se entregará un top-10 en `resultados_busqueda.csv`. Las tres formulaciones de una misma intención están pensadas para comprobar si el comportamiento se mantiene cuando cambia la superficie léxica.

`consultas_filtradas.csv` define cuatro búsquedas con una restricción de marca. La consulta vectorial y el filtro forman una sola operación de recuperación: no basta con recuperar globalmente y borrar después los resultados que no cumplen la condición.

## Cambios en el catálogo

`eventos_catalogo.csv` contiene 24 operaciones ordenadas:

- ocho actualizaciones de registros existentes;
- ocho eliminaciones;
- ocho altas.

El fichero representa el estado que debe quedar después de cada evento. El proceso debe poder repetirse sin duplicar registros ni alterar el resultado final. Tras aplicarlo, se comprobará por ID y mediante búsqueda que las altas aparecen, las actualizaciones muestran la versión nueva y las bajas dejan de ser recuperables.

## Detección de duplicados

`altas_desarrollo.csv` ofrece catorce ejemplos etiquetados para construir una regla de decisión. Un duplicado no tiene por qué compartir exactamente el mismo título: puede cambiar el orden, perder parte de la marca o llegar con una descripción ligeramente distinta.

`altas_evaluacion.csv` contiene otros catorce casos sin etiqueta. Para cada alta se debe recuperar el candidato más próximo del catálogo y decidir si es un duplicado. El umbral, los filtros y cualquier transformación deben obtenerse a partir de los datos de desarrollo, no de una inspección manual del conjunto de evaluación.

## Formato de los resultados

La entrega incluirá estos tres artefactos:

### `resultados_busqueda.csv`

| Campo | Contenido |
|---|---|
| `evaluation_id` | ID de `consultas_evaluacion.csv`. |
| `rank` | Posición entera de 1 a 10. |
| `product_id` | Producto recuperado. No puede repetirse dentro de una consulta. |
| `score` | Score nativo o transformado que utilice el sistema. |

### `resultados_duplicados.csv`

| Campo | Contenido |
|---|---|
| `incoming_id` | ID de `altas_evaluacion.csv`. |
| `predicted_duplicate` | `true` o `false`. |
| `matched_product_id` | Producto propuesto como duplicado; vacío si la predicción es negativa. |
| `score` | Score utilizado para tomar la decisión. |

### `metricas_desarrollo.json`

Debe contener, como mínimo, `ndcg_at_10`, `recall_at_10`, `mrr_at_10`, `latency_p50_ms` y `latency_p95_ms`. Las latencias describirán una ejecución reproducible; no se utilizarán para comparar proveedores ejecutados en infraestructuras diferentes.

## Procedencia

El catálogo es una selección española derivada del [Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data), publicado bajo licencia Apache-2.0. El fichero `manifest.json` conserva el tamaño del snapshot, el contrato de IDs, el mapeo de relevancia y los checksums SHA-256.

