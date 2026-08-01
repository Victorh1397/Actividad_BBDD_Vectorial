# ADR-007 · `ef_search` a 256

- **Estado:** aceptada
- **Requisitos afectados:** RF-08, RF-20, RF-21
- **Principios:** [P-06](../00_constitution.md)

## Contexto

`ef_search` es el único parámetro HNSW ajustable **sin reconstruir el grafo**,
así que es el que un sistema en producción realmente tocaría. Heredamos 128 de
la sesión 03; el enunciado pide justificar la configuración con evidencia, no
con la herencia.

### El barrido

Fidelidad medida contra el oráculo exacto sobre las 8 consultas de desarrollo,
catálogo completo de 15.000 productos:

| `ef_search` | Fidelidad@10 | Orden idéntico |
|---|---|---|
| 16 | 0,8500 | 0,7000 |
| 32 | 0,8875 | 0,7250 |
| 64 | 0,9750 | 0,8750 |
| 128 *(heredado)* | 0,9875 | 0,9250 |
| **256** | **1,0000** | **1,0000** |

### El coste, medido

| Operación | p50 |
|---|---|
| Qdrant, `ef_search=16` | 44,00 ms |
| Qdrant, `ef_search=256` | 47,16 ms |
| **Qdrant, solo contar puntos** | **32,64 ms** |
| Oráculo NumPy, fuerza bruta | 18,43 ms |
| Codificar la consulta | 2,56 ms |

Esa tercera fila es la que explica todo: una operación que **no busca nada**
tarda 32,64 ms. El transporte HTTP contra Docker Desktop sobre WSL2 domina la
medición, y la búsqueda propiamente dicha son unos 8 ms.

## Decisión

`ef_search = 256`.

## Justificación

Pasar de 128 a 256 cuesta ~3 ms —dentro del ruido del transporte— y compra
**fidelidad y orden perfectos**: el índice deja de perder candidatos frente a
la búsqueda exacta.

Eso importa más allá del número: con fidelidad 1,0, cualquier fallo de ranking
que observemos es atribuible **con certeza** a la representación y no al
índice. Simplifica la atribución de errores que exige RF-24, porque elimina una
de las cuatro capas sospechosas.

En un despliegue con latencia de red baja y millones de vectores el cálculo
sería distinto y 256 podría ser caro. Aquí no lo es, y conviene decir por qué:
**no es que HNSW sea barato, es que el transporte lo eclipsa**.

## Consecuencias

- Fidelidad ANN = 1,0 en la ejecución final. La comparación con el oráculo deja
  de ser una fuente de error y pasa a ser una confirmación.
- La latencia reportada describe **este** montaje —Windows, Docker Desktop,
  WSL2, localhost— y no debe compararse con otra infraestructura, que es
  justamente lo que el enunciado advierte.
- Al informe va un hallazgo que no esperábamos: **a esta escala el índice ANN
  no compensa**. La fuerza bruta en NumPy (18,43 ms) es más rápida que Qdrant
  (40,74 ms). Con 15.000 vectores de 768 dimensiones —46 MB— recorrer un grafo
  y cruzar la red cuesta más que comparar contra todo. HNSW empieza a rentar
  con órdenes de magnitud más datos, y eso responde directamente a la pregunta
  del enunciado sobre *"qué cambiaría al crecer el catálogo"*.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Mantener 128 | Pierde un candidato de 80 sin ahorrar tiempo observable. |
| Subir a 512 o más | Ya hay fidelidad perfecta en 256: no queda nada que ganar. |
| Bajar a 64 para ganar latencia | La latencia no mejora —está dominada por la red— y la fidelidad cae a 0,975. |
| Aumentar `m` o `ef_construct` | Exigen reconstruir el grafo y reingerir. `ef_search` logra el objetivo sin tocar el índice, que es precisamente su razón de ser. |
