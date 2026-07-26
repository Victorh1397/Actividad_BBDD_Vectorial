# ADR-001 · Orden canónico de ejecución y calibración de duplicados

- **Estado:** aceptada
- **Requisitos afectados:** RF-16, RF-17, RF-23
- **Principios:** [P-04](../00_constitution.md), [P-06](../00_constitution.md)

## Contexto

Al inspeccionar los datos aparece un solapamiento que no es casual.
`eventos_catalogo.csv` aplica `UPSERT` sobre estos `product_id`:

```
B000G3T55M · B07NV4L2W5 · B00BEFAR80 · B076HKFZ8N · B07JYHSK27 · B07N379P73 · B08JCQP3JW …
```

Y esos son **exactamente** los `reference_product_id` de los primeros casos de
`altas_desarrollo.csv`. Tras aplicar los eventos, los títulos de esos productos
llevan el sufijo `- ficha revisada`, el `text` incorpora
`Estado del catálogo: ficha revisada` y `catalog_version` pasa de 1 a 2.

Consecuencia directa: **la similitud entre un alta entrante y su duplicado de
referencia cambia según se hayan aplicado o no los eventos.** Un umbral calibrado
sobre el catálogo base no es el mismo umbral óptimo sobre el catálogo
post-eventos.

Esto crea un riesgo silencioso: si la calibración y la predicción ocurren a
distintos lados de los eventos sin que nadie lo note, la regla se valida contra
una distribución de scores que no es la que verá en producción, y el informe
reporta una precisión que no se sostiene.

## Decisión

Se fija un **orden canónico obligatorio**, idéntico en desarrollo y en la
ejecución final:

```
1. ingest                     catálogo base
2. verify                     recuento + estado de indexación
3. experiment                 representación (oráculo exacto)
4. duplicates calibrate       umbral sobre altas_desarrollo.csv → congela config/final.yaml
5. events --apply --verify    24 eventos + visibilidad
6. evaluate                   métricas, fidelidad ANN, latencia
7. duplicates predict         altas_evaluacion.csv con el umbral congelado
8. deliver                    artefactos
```

La calibración (paso 4) se hace sobre el **catálogo base** y la predicción
(paso 7) sobre el **catálogo post-eventos**.

## Justificación

El paso 7 después del 5 refleja el estado real de operación: en producción, el
control de altas se ejecuta contra el catálogo vivo, que ya ha recibido
actualizaciones. Calibrar en el paso 4 mantiene la disciplina de
[P-04](../00_constitution.md): el umbral se fija sin haber visto nunca el
conjunto ciego ni el catálogo en el que se aplicará.

Se asume explícitamente que esto introduce un **desplazamiento de distribución**
entre calibración y predicción. Es el mismo desplazamiento que sufriría el
sistema real, y por tanto medirlo es más honesto que eliminarlo artificialmente.

## Consecuencias

- El informe debe reportar la **sensibilidad** de la regla a este
  desplazamiento: se recalculan las métricas de desarrollo también sobre el
  catálogo post-eventos, y se compara. Si la degradación es material, es un
  hallazgo del trabajo, no un error a esconder.
- El orden queda codificado en `aurum deliver`, no en la memoria de quien ejecuta.
- Cualquier cambio de orden invalida los artefactos y exige re-ejecutar desde el paso 1.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Calibrar y predecir ambos sobre el catálogo base | Ignora que los eventos forman parte del recorrido exigido; la predicción no reflejaría el estado real del catálogo. |
| Calibrar y predecir ambos post-eventos | La calibración vería un catálogo ya mutado por el fichero de eventos, acercándose a ajustar el umbral con información del recorrido de evaluación. |
| Recalibrar el umbral después de los eventos | Violación directa de [P-04](../00_constitution.md): el umbral dejaría de estar congelado. |
