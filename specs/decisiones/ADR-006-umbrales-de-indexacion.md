# ADR-006 · Bajar los umbrales de indexación de Qdrant

- **Estado:** aceptada
- **Requisitos afectados:** RF-08, RF-10, RF-20, RF-21
- **Principios:** [P-06](../00_constitution.md)

## Contexto

Tras la primera ingesta con el perfil `sample`, la colección respondía consultas
con normalidad pero informaba de algo raro:

```
puntos en la colección: 1500
vectores indexados    : 0
estado                : green
```

Es decir: **Qdrant estaba contestando por fuerza bruta, sin usar el índice
HNSW**. La configuración `m=24, ef_construct=120` estaba aplicada pero no
intervenía en ninguna búsqueda.

La causa son dos umbrales con valores por defecto que no encajan con el tamaño
de este caso:

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `optimizers_config.indexing_threshold` | 20.000 | Tamaño mínimo para construir el grafo |
| `hnsw_config.full_scan_threshold` | 10.000 | Por debajo, escanea aunque exista grafo |

Dos detalles que cuesta leer en la documentación y que se midieron sobre la
colección real:

1. **Están en kilobytes, no en número de vectores.** La muestra de 1.500
   productos ocupa 4.500 KB a 768 dimensiones, pero solo 2.250 KB a 384.
2. **Se aplican por segmento, no por colección.** Qdrant repartió los 1.500
   puntos en 3 segmentos de ~1.500 KB cada uno.

Por eso 1.500 puntos sí se indexaron con el umbral bajado a 1.000 KB, mientras
que 500 puntos —500 KB por segmento— seguían sin indexarse.

## Decisión

Fijar ambos umbrales en **1.000 KB**, declarados en `.env.example` y en
`config/final.yaml`, y aplicarlos explícitamente al crear la colección.

## Justificación

Dejar los valores por defecto haría que **tres requisitos midiesen algo
distinto de lo que dicen medir**:

- **RF-08** exige configuración ANN explícita y relacionada con lo aprendido
  sobre HNSW. Con el índice sin construir, `m` y `ef_construct` serían
  decoración: se declararían sin tener efecto sobre ninguna búsqueda.
- **RF-20** exige comparar los IDs del motor con los del oráculo exacto para
  *"separar pérdida del índice de error del modelo"*. Si el motor responde por
  fuerza bruta, la fidelidad sale trivialmente 1,0 y no separa nada: estaríamos
  midiendo un índice que no se usa.
- **RF-21** exige medir latencia p50 y p95. Un escaneo lineal y un recorrido de
  grafo tienen perfiles distintos, así que el número describiría otra cosa.

El catálogo completo (15.000 × 768 × 4 = 45.000 KB) supera el umbral por
defecto de todas formas, pero la muestra no. Bajarlo mantiene **el mismo
comportamiento en ambos perfiles**, que es lo que hace comparables los
experimentos de desarrollo con la ejecución final.

## Consecuencias

- `CollectionStatus` lee de vuelta `m`, `ef_construct` y ambos umbrales desde
  el motor. Declarar una configuración y comprobar que se aplicó son cosas
  distintas, y `aurum verify` ahora muestra la segunda.
- `wait_until_indexed` acepta `require_indexed=True`. Qdrant marca `green` en
  cuanto almacena y construye el grafo **después**, en segundo plano, así que
  esperar solo a `green` demuestra que el dato está pero no que el índice
  exista. Una latencia medida antes del grafo mide un escaneo.
- Con estos umbrales el índice se construye antes, lo que hace la ingesta algo
  más lenta a cambio de que las consultas usen realmente HNSW.
- Este comportamiento va al informe: es un ejemplo concreto de por qué el
  enunciado pide *"verificad el recuento final **y el estado de indexación**
  antes de aceptar consultas"*. El recuento correcto no basta.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Dejar los valores por defecto | El índice no se construiría sobre la muestra, y RF-08, RF-20 y RF-21 medirían un motor que no usa su índice. |
| Bajarlos solo para el perfil `full` | Los experimentos de desarrollo dejarían de ser representativos de la ejecución final, que es justo lo que deben anticipar. |
| Forzar un solo segmento | Eliminaría el paralelismo de Qdrant para ganar previsibilidad en una cifra. Complejidad ajena al caso ([P-13](../00_constitution.md)). |
| Bajarlos a 0 | Indexaría hasta un segmento de un vector, penalizando la ingesta sin ganar nada observable. |
