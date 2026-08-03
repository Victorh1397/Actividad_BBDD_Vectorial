# ADR-008 · Regla y umbral de detección de duplicados

- **Estado:** aceptada · **umbral congelado el 2026-08-02**
- **Requisitos afectados:** RF-17, RF-23
- **Principios:** [P-04](../00_constitution.md), [P-06](../00_constitution.md)

## Contexto

El enunciado permite una regla que combine *"score vectorial, margen respecto al
segundo candidato, filtros de metadatos o una comprobación léxica"*, con dos
condiciones: que la base vectorial siga siendo el mecanismo de generación de
candidatos, y que el umbral se justifique con los casos de desarrollo **antes**
de mirar la evaluación.

## La decisión que resultó importar

No fue la regla, sino **qué texto de la ficha entrante se usa para buscar**.

El catálogo está indexado con `title + Marca: X + Color: Y`. Comparar contra él
el campo `text` en bruto de la ficha entrante —que además llega sucio, con
restos como `"Marca: . Color: B - distribuido por meross"`— compara dos formatos
distintos. Medido sobre los 14 casos etiquetados:

| Texto de la consulta | Separación entre clases | Candidato correcto |
|---|---|---|
| Campo `text` en bruto | +0,0032 | 6/7 |
| Solo `title` | +0,0392 | 6/7 |
| **`title + marca + color`** | **+0,0586** | **7/7** |

Alinear ambos lados **multiplica por 18 la separación** y resuelve el único caso
que fallaba. No es ajuste de curvas: es coherencia de formato, y habría sido la
elección correcta aunque no mejorase el número.

### El caso que lo destapó

`DEV-DUP-001` es un legging NIKE. El catálogo contiene **dos fichas casi
idénticas del mismo pantalón**, `B00MG5Q8TE` y `B000G3T55M`, separadas por
0,0002. Con el texto en bruto ganaba la equivocada; con el formato alineado gana
la correcta. Que el propio catálogo contenga duplicados internos es un hallazgo
que va al informe: el problema que Aurum Market quiere resolver **ya está dentro
de sus datos**.

## Decisión

- **Candidatos:** los dos mejores, vía base vectorial (Qdrant con `ef_search=256`).
- **Texto de la consulta:** `title + marca + color`, el mismo formato del catálogo.
- **Regla:** `predicted_duplicate = score_del_mejor >= 0,9191`.
- **Margen:** se registra pero **no interviene en la regla**.

## Justificación

### Por qué el margen no decide

Se midió y no discrimina: los duplicados abarcan de 0,0002 a 0,1162 y los
productos nuevos de 0,0004 a 0,0200. Se solapan por completo. Un `DEV-DUP-001`
con margen 0,0019 es duplicado y un `DEV-NEW-002` con margen 0,0200 no lo es.

Meterlo en la regla habría añadido un parámetro que no aporta señal, y con solo
14 casos eso es una invitación a sobreajustar. Se conserva porque sí es útil
**para reportar**: dice cuán ajustada fue una decisión, y un margen minúsculo
señala productos con variantes que merecen revisión humana.

### Por qué 0,9191 y no otro valor del hueco

Las clases no se solapan:

```
duplicados   [0,9484 … 0,9662]
nuevos       [0,8521 … 0,8898]
                    hueco: 0,0586
```

Cualquier umbral dentro del hueco clasifica el desarrollo **sin un solo error**,
así que F1 no puede elegir: vale 1,0 en todo el rango. El barrido decide *el
rango*; la robustez decide *el punto*. Se toma el punto medio porque deja el
margen más ancho a ambos lados para un caso no visto.

### Los dos errores no cuestan lo mismo (RF-23)

- **Falso positivo:** bloquea una publicación legítima. Genera fricción con el
  vendedor y trabajo de revisión manual. Es molesto, **visible y recuperable**.
- **Falso negativo:** publica un duplicado. El catálogo se degrada en silencio,
  el producto compite consigo mismo y la señal de venta se parte entre dos
  fichas. Más barato de cometer y **mucho más caro de arrastrar**.

Con separación perfecta no hubo que elegir entre ambos en desarrollo. Si la
hubiera, esta asimetría manda: ante la duda, **preferir revisar de más**. Queda
codificado en el desempate del barrido, que a igual F1 escoge el umbral con
menos falsos negativos.

## Consecuencias

- El umbral queda **congelado** en `config/final.yaml`. Reajustarlo tras ver
  `altas_evaluacion.csv` invalidaría la evaluación ([P-04](../00_constitution.md)).
- Un F1 de 1,0 sobre 14 casos **no es una promesa de generalización**. La
  muestra es diminuta y el hueco, aunque amplio en términos relativos, se apoya
  en 7 positivos. El informe debe presentarlo como lo que es: la regla separa
  limpiamente los casos disponibles.
- La predicción sobre el conjunto ciego se ejecuta con el catálogo **base**, sin
  eventos aplicados, por el motivo documentado en
  [ADR-001](ADR-001-orden-canonico-de-ejecucion.md).

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Regla con score **y** margen | El margen no discrimina: las dos clases se solapan en él por completo. Añadiría un parámetro sin señal sobre 14 casos. |
| Filtrar candidatos por marca | Probado en el caso NIKE: no ayuda, porque las dos fichas rivales comparten marca. Y excluiría duplicados que omiten la marca, que es justo uno de los patrones que el enunciado describe. |
| Comprobación léxica adicional | Con las clases ya separadas por completo, no queda error que corregir. Complejidad ajena al caso ([P-13](../00_constitution.md)). |
| Umbral en el borde del hueco | Elegir 0,9484 o 0,8898 daría el mismo F1 en desarrollo pero dejaría margen cero a un lado. |
