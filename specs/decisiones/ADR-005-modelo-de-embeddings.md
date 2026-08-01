# ADR-005 · Modelo de embeddings y composición del texto

- **Estado:** aceptada
- **Requisitos afectados:** RF-04, RF-06, RF-07
- **Principios:** [P-06](../00_constitution.md)

## Contexto

Había que elegir dos cosas a la vez: **qué texto se codifica** y **con qué
modelo**. Elegirlas juntas impide saber a cuál se debe una diferencia, así que
se midieron por separado sobre las ocho consultas de desarrollo, siempre con el
oráculo exacto para que el índice no contaminase la comparación.

### Resultados

| Exp | Texto | Modelo | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| E0 | `raw_text` | TF-IDF | 0,6198 | 0,3031 | 0,9167 |
| E1 | `raw_text` | e5-small | 0,6427 | 0,2830 | 0,8125 |
| E2 | `title_brand_color` | e5-small | 0,6881 | 0,3277 | 0,8542 |
| **E3** | `title_brand_color` | **e5-base** | **0,7072** | **0,3648** | **0,9375** |

Techo estructural de Recall@10: **0,5188**. E3 alcanza el **70 %** de lo
máximo alcanzable.

### Cómo se reparte la mejora

De E0 a E3 el nDCG sube +0,087, y el reparto es el hallazgo principal:

| Cambio | Ganancia | Peso |
|---|---|---|
| Léxico → denso (E0→E1) | +0,023 | 26 % |
| **Qué texto se codifica** (E1→E2) | **+0,045** | **52 %** |
| Modelo pequeño → base (E2→E3) | +0,019 | 22 % |

**Decidir qué texto codificar aportó el doble que cambiar de tecnología de
búsqueda, y más que duplicar el tamaño del modelo.**

La causa está medida: el campo `text` promedia 1.309 caracteres y llega a 3.000,
mientras el modelo trunca a 512 tokens (~2.048 caracteres). El **27,2 %** de los
productos pierde su cola, y lo que ocupa ese presupuesto suele ser relleno de
palabras clave. En el producto más largo, el corte cae en pleno *"…fiesta largos
de noche para bodas elegantes dolores promesas vestidos baratos desigual…"*.

## Decisión

- **Texto:** `title_brand_color` — título más marca y color, con etiquetas
  explícitas, omitiendo los campos ausentes.
- **Modelo:** `intfloat/multilingual-e5-base`, 768 dimensiones, con prefijos
  `query:` / `passage:` y normalización L2.

Ambos quedan congelados en `config/final.yaml`.

## Justificación

E3 gana en las **tres** métricas, no en una a costa de otra. Y hay un argumento
que pesa más que el 2,8 % de nDCG: **es la única configuración densa cuyo MRR
supera al baseline léxico** (0,9375 frente a 0,9167). E2 se queda en 0,8542,
por debajo de TF-IDF.

Eso importa porque MRR mide en qué posición aparece el primer resultado útil,
que es lo que percibe quien busca. Entregar un sistema vectorial que coloca el
primer acierto peor que una búsqueda por palabras sería difícil de defender ante
Aurum Market, por muy bueno que fuese su nDCG.

El coste es asumible para este catálogo:

| | e5-small | e5-base |
|---|---|---|
| Dimensión | 384 | 768 |
| Colección (15.000 productos) | 23 MB | 46 MB |
| Descarga del modelo | ~120 MB | ~1,1 GB |

46 MB no compromete el requisito de *"ejecutarse con recursos razonables"*. El
impacto en latencia se medirá en la Fase 5 y se reportará.

## Consecuencias

- La colección Qdrant se crea con **dimensión 768**. Cambiar de modelo obliga a
  reconstruirla desde cero: la dimensión es parte del esquema.
- `AURUM_EMBEDDING_MODEL` pasa a `intfloat/multilingual-e5-base` en
  `.env.example`. Las sesiones del curso usan e5-small, así que la diferencia se
  señala en el README.
- La primera ejecución en un entorno limpio descarga 1,1 GB. Debe advertirse en
  los tiempos aproximados del README.
- Se conserva `raw_text` implementado: es el punto de comparación que sostiene
  el hallazgo y permite reproducir E1.

## Un resultado que no conviene esconder

**E1 es peor que el baseline TF-IDF** en Recall (0,283 vs 0,303) y en MRR
(0,813 vs 0,917). Es decir, adoptar embeddings sin arreglar el texto **empeoró
el sistema**.

Hay además un sesgo del conjunto de desarrollo: sus ocho consultas son todas de
tipo `customer_query`, literales como *"botines marrones mujer tacon medio"*,
que es el terreno natural del emparejamiento léxico. Las doce consultas de
evaluación sí incluyen variantes `semantic` y `context` —*"quiero una
herramienta inalámbrica potente para perforar sin depender de un enchufe"*—
donde el sistema denso debería sacar ventaja.

Conclusión honesta: **sobre este conjunto de desarrollo la ventaja del denso es
modesta**, y afirmar lo contrario sería extrapolar. Va al informe tal cual.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| **e5-small con `title_brand_color`** (E2) | Ahorra la mitad de memoria y sería coherente con las sesiones del curso, pero su MRR queda por debajo del baseline léxico. |
| **e5-small con `raw_text`** (E1) | Peor que TF-IDF en dos de tres métricas. Es la evidencia del problema, no una opción. |
| **e5-large** | 1024 dimensiones y ~2,2 GB. El salto small→base ya rinde decreciente (+0,019 tras +0,045); no hay motivo para esperar que compense el coste. |
| **Texto híbrido** (título + primeros 300 caracteres) | Atacaría el fallo de *"cámaras bridge baratas"*, donde la información de gama vive en el texto largo. Descartada por alcance: con cuatro experimentos analizados el requisito está cubierto, y el caso se explica en la atribución de errores. |
