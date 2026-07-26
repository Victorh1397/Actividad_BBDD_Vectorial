# Constitución del proyecto Aurum Market

Este documento recoge los principios que **no se negocian** durante el desarrollo.
Su función es resolver discusiones antes de que ocurran: cuando una decisión de
implementación entre en conflicto con un principio, gana el principio.

Cada principio nace de una exigencia explícita del enunciado, que se cita entre
paréntesis para poder auditarlo.

---

## P-01 · El sistema no razona, recupera

No se construye un RAG, no se generan respuestas y **ningún LLM interviene en
tiempo de ejecución** para resolver, etiquetar o reordenar consultas. Los modelos
de embeddings sí forman parte del problema.

> *"No se construirá un RAG, no se generarán respuestas y no se utilizará un LLM
> para resolver, etiquetar o reordenar consultas en tiempo de ejecución."*

**Cómo se comprueba:** no hay dependencias de proveedores generativos en
`pyproject.toml`, y ninguna ruta de código llama a un modelo de chat.

---

## P-02 · El SDK nativo es el núcleo

La configuración de la colección, la observación de errores y las operaciones
administrativas se hacen con `qdrant-client`. Una capa de abstracción como
LangChain puede aparecer como anexo demostrativo, pero **nunca** sustituye a la
configuración explícita del motor.

> *"el núcleo de la solución utilice su SDK nativo. Una capa como LangChain puede
> aparecer como integración adicional, pero no sustituye la configuración, la
> observación de errores ni las operaciones administrativas del motor."*

---

## P-03 · Un score nunca cambia de significado en silencio

Todo resultado viaja con su score **nativo** acompañado de `score_kind`
(`similarity` | `distance`) y `higher_is_better`. Está prohibido comparar,
promediar u ordenar conjuntamente valores cuya semántica difiere, y está
prohibido convertir una distancia en similitud sin dejarlo escrito.

> *"Conservad la semántica del score nativo. No comparéis como si fueran
> equivalentes una distancia y una similitud."*

---

## P-04 · Los umbrales se congelan antes de ver la evaluación

Cualquier parámetro de decisión —umbral de duplicados, margen, `top_k`— se
calibra **exclusivamente** con datos de desarrollo y se escribe en
`config/final.yaml` antes de ejecutar sobre los conjuntos ciegos. Inspeccionar la
evaluación para ajustar un umbral invalida el experimento.

> *"Justificad el umbral con los casos de desarrollo; no lo fijéis después de
> inspeccionar manualmente la evaluación."*

---

## P-05 · Las decisiones sobre relevancia se declaran y no cambian

El mapeo de relevancia graduada es `E=3, S=2, C=1, I=0`. El umbral que define
"relevante" para Recall@10 y MRR@10 se declara una vez, se justifica y **no
cambia entre experimentos**. Si se cambia, se re-ejecutan todos los experimentos.

> *"La elección puede discutirse, pero no puede cambiar silenciosamente entre
> experimentos."*

---

## P-06 · Ningún experimento sin configuración, métricas e IDs

Un experimento que no conserva los tres elementos no es evidencia, es una
anécdota. Cambiar el nombre de un modelo y reportar una cifra distinta no
constituye un experimento: hay que analizar **por qué** cambió.

> *"Cada experimento conservará la configuración, las métricas y los IDs
> recuperados."*

---

## P-07 · La ausencia de dato es ausencia, nunca la cadena `"nan"`

Los valores vacíos del catálogo se sanean de una única forma en un único lugar
(`data.py`). Ninguna capa posterior vuelve a interpretarlos. Está prohibido
"arreglar" el dataset a mano: su suciedad es parte del problema.

> *"Los valores vacíos son información ausente, no la cadena literal `"nan"`.
> La ingesta debe tratarlos siempre de la misma forma."*

---

## P-08 · La idempotencia es de diseño, no de comprobación

La ingesta usa `upsert` sobre un ID estable (`record_id`, UUIDv5 del catálogo).
Repetirla no puede aumentar el recuento **por construcción**, no porque después
se borren duplicados.

> *"La ingesta completa puede repetirse sin aumentar el recuento."*

---

## P-09 · El filtro es parte de la consulta

Una búsqueda restringida por marca se ejecuta **en la base de datos**, con su
mecanismo nativo de filtrado y su índice de payload. Recuperar globalmente y
descartar después los resultados que no cumplen la condición es una violación
de este principio, aunque produzca la misma lista.

> *"La consulta vectorial y el filtro forman una sola operación de recuperación:
> no basta con recuperar globalmente y borrar después los resultados que no
> cumplen la condición."*

---

## P-10 · Una escritura confirmada debe volverse observable, o el sistema lo dice

Tras aplicar una mutación, el sistema espera la visibilidad con un plazo
acotado y, si no llega, **falla o informa**. Nunca continúa asumiendo que el
estado es el esperado.

> *"Se busca verificar que una escritura confirmada acaba siendo observable por
> las rutas de lectura utilizadas y que el sistema sabe esperar, fallar o
> informar cuando no ocurre."*

---

## P-11 · Lo destructivo está apagado por defecto

Borrar o recrear un recurso exige dos condiciones simultáneas: el permiso
(`AURUM_ALLOW_RESET=true`) y el nombre exacto del recurso escrito a mano
(`AURUM_CONFIRM_CLEANUP`). Además, la operación solo puede afectar a recursos
cuyo nombre empieza por `aurum-market`. El repositorio no contiene credenciales.

> *"Cualquier limpieza debe limitarse a recursos creados para la actividad y
> estar desactivada por defecto. No se incluirán credenciales en el repositorio."*

---

## P-12 · Los fallos se atribuyen a una capa con evidencia

Ante un mal resultado, la pregunta no es "¿cómo lo arreglo?" sino "¿qué capa lo
explica?". El procedimiento es mecánico:

| Observación | Capa responsable |
|---|---|
| El vecino exacto ya es semánticamente malo | **Representación** |
| El oráculo exacto lo recupera y el ANN no | **Índice** |
| Falta el metadato, es inconsistente o el filtro excluye el producto | **Datos o filtros** |
| El estado leído no coincide con la escritura aplicada | **Persistencia o consistencia** |

---

## P-13 · No se añade complejidad ajena al caso

No se puntúan funcionalidades extra por el hecho de existir. Sin interfaz web,
sin servicios que el caso no pida, sin dependencias que no cierren un requisito.
Si algo no está trazado a un RF de [01_spec.md](01_spec.md), no se escribe.

> *"No se puntúan funcionalidades ajenas al caso por el mero hecho de añadir
> complejidad."*

---

## P-14 · Todo se regenera con un comando

Un entorno limpio debe poder reproducir los artefactos siguiendo el README. Las
métricas salen de `uv run aurum deliver`, no de copiar cifras a mano en el
informe.

> *"Las métricas pueden regenerarse desde un único comando."*
