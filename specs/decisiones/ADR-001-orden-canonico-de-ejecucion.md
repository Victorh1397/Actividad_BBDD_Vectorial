# ADR-001 · Orden canónico de ejecución

- **Estado:** aceptada · **revisada durante la Fase 2 con evidencia de los datos**
- **Requisitos afectados:** RF-16, RF-17, RF-19, RF-23, RF-25
- **Principios:** [P-04](../00_constitution.md), [P-06](../00_constitution.md)

## Contexto

El enunciado plantea tres pruebas sobre la misma colección: búsqueda semántica,
control de altas duplicadas y operaciones de catálogo. Nada dice en qué orden
ejecutarlas, así que hay que decidirlo. Al inspeccionar los datos aparece que la
decisión no es inocua: **los 24 eventos están construidos a mano para tocar
exactamente los productos que las otras dos pruebas utilizan como respuesta
correcta.**

### La evidencia

`eventos_catalogo.csv` se reparte en tres grupos de ocho, y cada grupo apunta a
algo concreto:

| Grupo | Qué hace | A quién afecta |
|---|---|---|
| 8 `UPSERT` sobre productos existentes | Añaden el sufijo `- ficha revisada` y suben a `catalog_version` 2 | **7 de las 7** referencias de `altas_desarrollo.csv` |
| 8 `DELETE` | Retiran productos del catálogo | **6 de las 7** referencias de `altas_evaluacion.csv` |
| 8 `UPSERT` de productos nuevos (`AURUM-NEW-001..008`) | Introducen fichas cuyos títulos responden a consultas conocidas | Ninguna métrica: **no aparecen en los juicios de relevancia** |

Si los eventos tocaran productos al azar, la probabilidad de afectar a una
referencia concreta sería de 24/15.000 = 0,16 %. Que acierten 7 de 7 y 6 de 7 no
es casualidad: está diseñado para que el orden importe.

Dos comprobaciones cierran la cuestión:

- **Ningún `AURUM-NEW-*` aparece en `relevancias_desarrollo.csv`**, y los 248
  productos juzgados existen todos en el catálogo base, sin un solo huérfano. Las
  métricas de ranking, por tanto, se calculan sobre el catálogo **base**.
- Los productos que los eventos añaden responden a consultas del workload:
  `AURUM-NEW-001` "Taladro inalámbrico compacto 24 V con dos baterías" frente a
  `EVAL-100455-direct` "taladro 24v batería"; `AURUM-NEW-008` "Base tapizada
  160 x 200 sin patas" frente a `DEV-13357` "base tapizada 160x200 sin patas".
  No están para puntuar, sino para que la **visibilidad de un alta pueda
  comprobarse con una búsqueda semántica real** y no solo con una lectura por ID,
  que es literalmente lo que pide RF-16.

## Decisión

Las tres pruebas parten del **mismo estado** —el catálogo completo recién
ingerido— y los eventos se aplican **al final**, cuando los artefactos de entrega
ya están escritos:

```
1. ingest                     catálogo completo
2. verify                     recuento + estado de indexación
3. experiment                 representación, con oráculo exacto
4. duplicates calibrate       umbral sobre altas_desarrollo.csv → congela config/final.yaml
5. evaluate                   nDCG/Recall/MRR, fidelidad ANN, latencia
6. duplicates predict         altas_evaluacion.csv con el umbral congelado
7. deliver                    los tres artefactos
8. events --apply --verify    prueba operativa aislada: idempotencia y visibilidad
```

Los pasos 4 a 7 operan sobre el catálogo base. El paso 8 lo modifica, y por eso
va después: para no contaminar a los demás.

## Justificación

El enunciado clasifica los ficheros por su **uso esperado**, y la distinción es
explícita en la tabla de §2:

| Fichero | Uso esperado, textual |
|---|---|
| `eventos_catalogo.csv` | *"**Probar** idempotencia, mutaciones y visibilidad"* |
| `altas_evaluacion.csv` | *"**Entregar** la decisión y el candidato recuperado"* |
| `consultas_evaluacion.csv` | *"**Entregar** un top-10 reproducible"* |
| `catalogo_productos.csv.gz` | *"Colección final sobre la que se realizarán las búsquedas"* |

Unos ficheros existen para **probar** el sistema y otros para **producir la
entrega**. Los eventos pertenecen al primer grupo y en ningún momento se les
atribuye el papel de alterar el estado sobre el que se entrega. La colección de
búsqueda es `catalogo_productos.csv.gz`, sin cualificar.

Coherentemente, §5 lista las mutaciones como una evidencia independiente cuya
interpretación es *"demostrar idempotencia y visibilidad"*, y el recuadro de §4.1
se titula *"qué demuestra esta prueba"* y habla solo de observabilidad.

**El argumento decisivo** es que el orden contrario haría imposible cumplir §4.2,
que exige que *"la base vectorial siga siendo el mecanismo de generación de
candidatos"* y que *"una predicción positiva debe señalar el product_id concreto
que se considera duplicado"*. Con los eventos aplicados, seis de los siete
candidatos han sido borrados: la base vectorial no puede generarlos y por tanto
es imposible señalarlos. El recall de duplicados caería a 1/7 ≈ 0,14, y el punto
5 de la checklist de entrega quedaría incumplido por construcción.

## Consecuencias

- Calibración y predicción de duplicados ocurren sobre el **mismo** catálogo, así
  que el umbral se aplica en las condiciones en que se eligió. Desaparece el
  desplazamiento de distribución que la versión anterior de este ADR daba por
  inevitable.
- Las métricas de ranking son reproducibles con solo declarar el perfil de
  catálogo: no dependen de cuántos eventos se hubieran aplicado antes.
- La prueba de mutaciones gana valor propio: al ejecutarse sobre un estado
  conocido y ya medido, sus tres comprobaciones son limpias. Las altas deben
  aparecer al buscarlas, las actualizaciones deben mostrar `catalog_version` 2 y
  las bajas deben dejar de ser recuperables.
- **La reproducibilidad exige poder volver al estado base.** Como el paso 8
  modifica la colección de forma irreversible, repetir el recorrido completo pasa
  por reingerir desde cero. El README debe decirlo con claridad.
- `aurum deliver` codifica los pasos 1 a 7 y **no** aplica eventos. Aplicarlos es
  un comando explícito y separado.

## Revisión · qué decía antes este documento

La primera versión situaba los eventos en el paso 5, entre la calibración y la
predicción de duplicados, razonando que *"en producción el control de altas se
ejecuta contra el catálogo vivo, que ya ha recibido actualizaciones"*, y asumía
el desplazamiento de scores como el precio honesto de esa realidad.

El razonamiento era plausible pero introducía una dependencia que **el enunciado
nunca establece**, y la evidencia lo refuta: con ese orden, seis de los siete
duplicados de evaluación son indetectables porque su candidato ya no existe. Se
optimizaba el realismo del escenario a costa de destruir la prueba.

Queda registrado porque el error es instructivo: al modelar un caso de estudio,
el criterio no es "qué haría un sistema real" sino "qué mide realmente cada
prueba". Ambas cosas coinciden a menudo, pero no siempre.

## Alternativas descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Eventos entre calibración y predicción *(versión anterior)* | Borra 6 de 7 candidatos de evaluación: hace inalcanzable §4.2 y el punto 5 de la checklist. |
| Eventos al principio, antes de medir | Los juicios de relevancia no conocen los `AURUM-NEW-*` y los 248 productos juzgados son todos del catálogo base: mediríamos contra una colección que la verdad de referencia ignora. |
| Recalibrar el umbral después de los eventos | Violación directa de [P-04](../00_constitution.md): el umbral dejaría de estar congelado. |
| Dos colecciones, una por estado | Duplica la ingesta de 15.000 vectores para evitar un problema que el orden ya resuelve. Complejidad ajena al caso ([P-13](../00_constitution.md)). |
