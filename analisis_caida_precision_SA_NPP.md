# Análisis de la caída de precisión en `SA4NPP_phase_diagram.jl` para N ≳ 27

## 1. Resumen ejecutivo

La caída de `P_sat` observada a partir de N≈27 en zonas donde, según la teoría
(κ bien por debajo de κ_c(N)), debería seguir siendo fácil encontrar una
partición perfecta, **no se debe a un problema de precisión de punto
flotante**. La aritmética que determina si una partición es perfecta (la
energía) se realiza íntegramente en `Int128` de forma exacta, y los rangos de
N y b presentes en los datos (N≤35, b≤34) están muy por debajo del umbral en
que `Float64` deja de representar enteros de forma exacta.

La causa raíz identificada, y confirmada directamente en los CSV adjuntos, es
un **error de diseño en el cómputo del presupuesto de Monte Carlo por etapa de
temperatura** (`MCS_per_T`) en `simulated_annealing_geometric` y
`simulated_annealing_logarithmic`: para `regime="linear"`, este presupuesto
es **constante e igual a 1** para todo N, en lugar de crecer con N como el
nombre del régimen sugiere. A esto se suma un **schedule de enfriamiento mal
calibrado respecto del presupuesto total** (`cooling_factor` efectivamente
≈1 en el experimento geométrico, y β demasiado pequeño en el logarítmico),
que impide que el sistema efectivamente "se enfríe" dentro del número de
etapas disponibles. El resultado neto es que, a medida que N crece, el
algoritmo deja de hacer un *annealing* real y se convierte, en la práctica,
en una caminata aleatoria a temperatura casi constante (≈T0) durante todo el
presupuesto —incapaz de concentrarse en la cuenca del óptimo global antes de
agotar el límite de pasos—.

## 2. Evidencia utilizada

- `SA4NPP_phase_diagram.jl` (código fuente completo).
- `npp_20261010_085941_{data,params}.json/csv` — experimento `geometric`,
  `cooling_factor = 0.9999999999999999`.
- `npp_20261012_093339_{data,params,aggregated}.csv` — experimento
  `logarithmic`, `cooling_factor` (β) `= 0.001`.
- `npp_multi_20261012_114351_k_vs_N_pcrit_0_5.png` — κ_c(N) medido vs. teórico.

Verificación directa sobre los CSV de ambos experimentos:

```
n_MCS (= MCS_per_T) observado para TODAS las filas, ambos experimentos: {1}
cooling_factor (geométrico) almacenado:  1.000000   (es decir, ≈ 1)
cooling_factor (logarítmico, β):         0.001000
N máximo = 35, b máximo = 34  (en ambos experimentos)
```

## 3. Hipótesis descartada: precisión de punto flotante

### 3.1 Dónde entra el punto flotante en el código

- `generate_instance`: genera los números con `Int128` (o `BigInt` si
  `b > 100`). **Exacto**.
- `signed_sum` / `residue`: la suma firmada y el valor absoluto se calculan
  enteramente sobre el tipo entero de los números (`T = Int128`). **Exacto**.
- El criterio de éxito, `current_energy <= SUCCESS_THRESHOLD` (línea ~266,
  ~371, ~491), compara un **entero exacto** (`Int128`) con la constante
  `1.1`. Julia promueve la comparación sin pérdida de información del lado
  entero. **Exacto**: nunca hay un falso negativo/positivo en la detección de
  "partición perfecta" por redondeo.
- `delta_E = proposed_energy - current_energy` (línea ~280, ~383, ~504): se
  calcula también en `Int128`, de forma exacta. Solo el valor ya calculado se
  convierte a `Float64` (`Float64(delta_E)`) para evaluarlo dentro de
  `exp(-ΔE/T)` en la regla de Metropolis. Es decir: **la decisión de aceptar
  automáticamente una mejora (`delta_E <= 0`) se toma en dominio entero, no en
  punto flotante** — solo el término probabilístico de aceptar un peor estado
  usa `Float64`, donde una pequeña imprecisión solo perturba levemente una
  probabilidad, sin generar errores sistemáticos.

### 3.2 Margen de seguridad numérico en los datos actuales

Un `Float64` representa enteros de forma exacta hasta 2^53 ≈ 9.007×10^15.
Las magnitudes involucradas en los experimentos compartidos:

- Máximo valor de un número de la instancia: 2^b − 1, con b_max = 34 →
  ≈1.7×10^10.
- Suma total máxima (y por tanto el techo de `T0_used` antes de dividir por
  4N y de cualquier `delta_signed`): ~ N·2^b ≈ 35 × 1.7×10^10 ≈ 6×10^11.

6×10^11 está **cuatro órdenes de magnitud por debajo** del límite de
representación exacta de enteros en `Float64` (9×10^15). Por lo tanto, para
el rango N∈[10,35], b∈[6,34] de los datos entregados, **no existe pérdida de
precisión numérica** en ningún paso del algoritmo.

### 3.3 Dónde SÍ podría convertirse en un problema real (advertencia a futuro)

El bloque `if __name__ == ...` del propio script (líneas 886-907) ya contiene
un ejemplo con `N=50`, `b ∈ [N-10, N-6] = [40,44]`. Ahí, N·2^b ≈ 50×2^44 ≈
8.8×10^14, que ya se acerca al límite de 9×10^15 (menos de dos órdenes de
magnitud de margen). Si en el futuro se escalan N y b más allá de este punto
(sin llegar aún a b>100, donde el código ya conmuta a `BigInt`), el casteo
`Float64(delta_E)` empezará a perder bits de precisión en la cola, afectando
la exactitud de la probabilidad de Metropolis (no la detección de éxito, que
seguirá siendo exacta). **Recomendación preventiva**: bajar
`BIGINT_THRESHOLD_BITS` o, mejor, condicionar el cambio a `BigInt`/`BigFloat`
al producto `N·2^b` en vez de solo a `b`, ya que es ese producto el que
determina el riesgo real de desbordar la mantisa de 53 bits.

**Conclusión de esta sección: el punto flotante queda descartado como causa
de la caída observada en N≈27–35.**

## 4. Causa raíz identificada: el presupuesto por etapa no escala con N

### 4.1 El bug

En `simulated_annealing_geometric` (línea 359) y
`simulated_annealing_logarithmic` (línea 479):

```julia
z = max(0, regime_exponent(regime) - 1)   # geometric
z = regime_exponent(regime) - 1           # logarithmic
MCS_per_T = max(1, round(Int, A * (N^z)))
```

Para `regime = "linear"` (el régimen usado en ambos experimentos
compartidos), `regime_exponent("linear") = 1`, por lo que **z = 0** y
`N^z = 1` para cualquier N. Con `A = 1.0`:

```
MCS_per_T = max(1, round(1.0 * N^0)) = 1      ∀ N
```

Esto está confirmado empíricamente: la columna `n_MCS` (que registra
`MCS_per_T`) vale **exactamente 1 en las 11 400 filas** de ambos CSV, sin
excepción, para N desde 10 hasta 35.

### 4.2 Por qué esto importa

El docstring del archivo (líneas 40-46) describe `MCS_per_T` como "el número
de barridos de Monte Carlo realizados EN CADA etapa de temperatura" y afirma
que `regime`/`A` lo determinan de forma análoga al régimen fijo. Pero la
resta de un grado al exponente (`z - 1`) hace que, específicamente para
`regime="linear"`, el presupuesto por etapa **no dependa de N en absoluto**:
cada nivel de temperatura recibe un único barrido (N intentos de flip) antes
de enfriar, sin importar si N=10 o N=35.

El espacio de configuraciones crece como 2^N. El tiempo de mezcla
(*mixing time*) de una cadena de Metropolis a temperatura fija crece, en el
peor caso razonable, al menos linealmente con N (y típicamente más rápido
cerca de la transición de fase). Mantener el número de intentos por etapa
constante en 1 mientras el espacio de búsqueda se duplica en cada incremento
de N implica una pérdida sistemática y creciente de capacidad de
equilibración relativa, que se vuelve crítica justo en la zona donde
empiezan a aparecer valores de b grandes (b≈N−4 a N−1), es decir, cerca de
κ_c(N) — y, por el error mencionado en 4.3, también lejos de κ_c(N).

### 4.3 Un segundo problema que amplifica el primero: el schedule no enfría dentro del presupuesto

**Experimento geométrico** (`npp_20261010_085941`):
`cooling_factor = 0.9999999999999999`. Este valor es, en `Float64`,
`prevfloat(1.0)` — el flotante representable inmediatamente anterior a 1.0
(1 − 2⁻⁵³). Con `Temp *= cooling_factor` por etapa:

```
ln(cooling_factor) ≈ -1.11×10⁻¹⁶
Nº de etapas para reducir T a la mitad ≈ ln(2)/1.11×10⁻¹⁶ ≈ 6.2×10¹⁵
```

Esto es ~8 órdenes de magnitud mayor que `n_MCS_limit` (≤ 3.5×10⁷ para
N=35, ya multiplicado por N). **En la práctica, la temperatura nunca baja de
forma apreciable durante todo el presupuesto**: el algoritmo opera como una
caminata de Metropolis a T≈T0 (temperatura inicial, alta) durante todas las
etapas disponibles, sin llegar jamás a la fase de "congelamiento" que
permite que el *annealing* explote los mínimos encontrados. Dado que
`T0_used = mean(numbers)/4` es del orden de la magnitud típica de un único
número (y por tanto comparable o mayor a las diferencias de energía
relevantes), a T≈T0 casi cualquier movimiento, incluso uno que empeore la
energía, es aceptado con probabilidad apreciable — el algoritmo nunca
estabiliza una solución encontrada por azar.

**Experimento logarítmico** (`npp_20261012_093339`): β = `cooling_factor` =
0.001. Con `T(stage) = T0/(1 + β·ln(1+stage))`, y dado que (por el bug de
4.1) cada etapa solo consume 1 MCS, alcanzar, por ejemplo, 9×10⁵ etapas
(valor observado en los datos cerca de κ_c para N=24-28, columna
`mean_final_energy`/MCS usadas) da:

```
T(9×10⁵) / T0 = 1 / (1 + 0.001 · ln(9×10⁵+1)) = 1 / (1 + 0.001·13.7) ≈ 0.986
```

Es decir, tras casi un millón de etapas (y por ende, dado el bug 4.1, casi un
millón de barridos totales) la temperatura solo ha caído ~1.4% respecto de
T0. El schedule logarítmico, con este β y combinado con `MCS_per_T=1`, es
demasiado lento para la ventana de presupuesto (`n_MCS_limit = 10⁶·N`)
disponible: nunca llega a explorar el régimen de baja temperatura donde el
*annealing* realmente concentra la búsqueda.

### 4.4 Por qué el efecto se concentra "a partir de N≈27"

Ninguno de los dos problemas anteriores depende explícitamente de un umbral
mágico en N=27; son efectos continuos que empeoran con N. Lo que sí cambia
gradualmente —y de forma coherente con lo observado en el gráfico— es que:

1. El espacio de búsqueda 2^N crece exponencialmente con N, mientras que el
   presupuesto efectivo de equilibración por etapa permanece fijo (4.1): la
   brecha entre "dificultad del problema" y "esfuerzo de muestreo por etapa"
   se ensancha con cada incremento de N.
2. Cerca de κ_c(N) el número de particiones perfectas decae rápidamente
   (dificultad real, esperada por la teoría), por lo que la ya escasa
   capacidad de equilibración del punto 1 dejar de alcanzar para encontrar
   el óptimo primero afecta a los κ cercanos a κ_c, y luego, al seguir
   creciendo N, también a κ progresivamente más alejados de κ_c — lo cual es
   consistente con el "esparcimiento" observado en el gráfico de κ_c(N)
   medido en la zona N≈27 en adelante (mayor dispersión de los puntos
   alrededor de la curva teórica, incluyendo puntos que caen tanto por
   encima como por debajo de ella).
3. Con solo 100 muestras por punto (`samples=100`) y una resolución en b de
   Δκ=1/N, la estimación de κ_c por interpolación al 50% de éxito es
   sensible al ruido estadístico cuando la curva P_sat(κ) se vuelve más
   empinada (como ocurre cerca de la transición real) o cuando el fallo del
   *schedule* introduce un "piso" artificial de fallos que no es puramente
   función de κ. Esto puede producir el salto de dispersión visualmente
   abrupto que se observa a partir de cierto N, sin que exista un umbral
   discreto único en el código.

## 5. Otras causas descartadas o de impacto menor

- **Desbordamiento de `Int128`**: con b≤34 y N≤35, el rango utilizado está
  muy lejos del límite de 127 bits de `Int128` (que soporta sin problema b
  hasta ~100 antes de conmutar a `BigInt`). Descartado.
- **`SUCCESS_THRESHOLD = 1.1`**: correcto para ambos casos de paridad de
  `sum(numbers)` (energía mínima 0 o 1). No introduce falsos negativos.
- **Semilla / paralelismo (`Threads.@threads` + RNG por tarea)**: el esquema
  de semillas por tarea (`seed + hash((N,b,s)) % 1e9`) es determinista y no
  depende del número de hilos; no debería introducir sesgos dependientes de
  N. No se encontró evidencia de problema aquí.
- **`T0_used = sum(numbers)/(4N)`**: heurística razonable y dimensionalmente
  consistente con la escala de energía del problema (∝2^b); no es la causa
  del efecto observado, aunque su idoneidad exacta como punto de partida del
  *annealing* no fue evaluada cuantitativamente en este análisis.

## 6. Recomendaciones concretas

1. **Corregir el exponente de `MCS_per_T`.** Usar
   `z = regime_exponent(regime)` (sin el `-1`) en
   `simulated_annealing_geometric`/`logarithmic`, de modo que
   `regime="linear"` efectivamente otorgue `MCS_per_T ∝ N`, como indica la
   documentación del propio archivo. Alternativamente, si se desea
   preservar la semántica actual (presupuesto total ≈ `MCS_per_T × n_etapas`
   comparable entre regímenes), documentarlo explícitamente y compensar con
   un número de etapas que sí escale con N.
2. **Calibrar `cooling_factor`/β en función del presupuesto disponible**, no
   como constantes fijas e independientes de `n_MCS_limit`. Por ejemplo,
   fijar el número de etapas objetivo `S = n_MCS_limit / MCS_per_T` y
   despejar:
   - Geométrico: `cooling_factor = (min_T / T0)^(1/S)`.
   - Logarítmico: `β` tal que `T0/(1+β·ln(1+S)) ≈ min_T`.
   Esto garantiza que el *annealing* realmente recorra todo el rango
   [T0, min_T] dentro del presupuesto fijado, en vez de quedar "congelado"
   en T0 (caso geométrico) o enfriarse de forma imperceptible (caso
   logarítmico).
3. **Evitar valores de `cooling_factor` extremadamente cercanos a 1.0 por
   construcción flotante** (p. ej. generados por fórmulas que restan un
   número muy pequeño a 1). Verificar explícitamente, tras calcular
   `cooling_factor`, que `cooling_factor < 1 - δ` para algún δ con sentido
   físico respecto al número de etapas esperado, y lanzar un *warning* si
   no se cumple.
4. **Escalar `n_MCS_limit` de forma más agresiva que lineal en N**, o
   hacerlo depender de κ (mayor presupuesto cerca de κ_c(N)), dado que la
   dificultad real del muestreo crece más rápido que N cerca de la
   transición.
5. **Aumentar `samples` y/o la resolución en b** cerca de κ_c(N) para N
   grandes, para reducir el ruido estadístico en la estimación de κ_c por
   interpolación al 50%, independientemente de las correcciones anteriores.
6. **Validación cruzada en N pequeños**: para un subconjunto de instancias
   con N≤25 donde el presupuesto actual sí basta, comparar el resultado de
   la SA contra un resolutor exacto (branch & bound o DP sobre residuos) para
   confirmar que los "fallos" reportados corresponden a verdaderos negativos
   del algoritmo y no a instancias sin partición perfecta.
7. **(Preventivo, no urgente con los rangos actuales)** Para escalados
   futuros de N/b, condicionar el cambio a aritmética de mayor precisión
   (`BigInt`/`BigFloat`) al valor de `N·2^b` en vez de solo a `b`, dado que
   es ese producto el que determina el riesgo real de exceder los 2^53 bits
   exactos de `Float64` en la evaluación del factor de Boltzmann.

## 7. Conclusión (preliminar — ver revisión en la Sección 8)

La degradación de `P_sat` a partir de N≈27 es consistente con un problema
**algorítmico de calibración del *schedule* de enfriamiento respecto del
presupuesto de cómputo disponible**, agravado por un error puntual en el
cálculo de `MCS_per_T` que lo deja fijo en 1 para `regime="linear"`
independientemente de N. No hay evidencia, en el código ni en los datos
entregados (N≤35, b≤34), de pérdida de precisión en punto flotante: toda la
aritmética que determina el éxito/fracaso de una partición se realiza de
forma exacta en enteros. El punto flotante solo interviene en el término
probabilístico de Metropolis, donde su precisión es más que suficiente en el
rango actual, aunque conviene vigilarlo si N y b se escalan considerablemente
más allá de lo ya ensayado.

**Nota (añadida tras la Sección 8):** la recomendación de "acelerar el
enfriamiento" formulada en la Sección 6 (puntos 2 y 3) se contradice con
evidencia experimental adicional del propio usuario y queda **matizada y en
parte retractada** más abajo. El diagnóstico sobre el bug de `MCS_per_T=1`
y sobre la ausencia de problemas de punto flotante se mantiene sin cambios.

---

## 8. Revisión crítica a la luz de nueva evidencia experimental y de la bibliografía

### 8.1 La observación que motiva esta revisión

El usuario reporta una observación empírica reproducible: **enfriar más
lento (cooling_factor más cercano a 1) mejora las soluciones (κ_c medido se
acerca más al valor teórico); enfriar más rápido las empeora
dramáticamente (κ_c cae incluso por debajo de 0.8)**. Esto es, en principio,
inconsistente con la Sección 4.3 del análisis original, donde se calificó
`cooling_factor = 0.9999999999999999` como un *bug* que "impide que el
sistema se enfríe" y se recomendó, en la Sección 6, calibrar el *schedule*
para que sí llegue a congelar dentro del presupuesto.

El experimento `npp_20261007_120607` (geometric, `cooling_factor =
0.9999999999999999`, `min_T=1e-20`, `n_MCS_limit=5×10⁷`, `samples=200`)
comparte exactamente el mismo valor "casi-1" de `cooling_factor` analizado
en la Sección 4.3, y exactamente el mismo bug de `MCS_per_T=1` (confirmado:
la columna `n_MCS` vale 1 en las 17 600 filas de este experimento también).
Y sin embargo, **según el propio usuario, este tipo de configuración es la
que ha dado los mejores resultados** (los que más se acercan a κ_c(N)
teórico) entre los experimentos realizados. Esto exige revisar si la
explicación de la Sección 4.3 — "el algoritmo nunca se enfría, por lo tanto
no funciona bien" — es la correcta, o si, en este problema particular, es
precisamente lo contrario de lo que cabría esperar de un *annealing*
clásico.

### 8.2 Por qué la intuición "enfriar lento es malo" no aplica aquí: la geometría del espacio de soluciones del NPP

La simulated annealing estándar asume implícitamente que el **vecindario de
búsqueda tiene estructura**: que la energía de una configuración está
correlacionada con la de sus vecinos (moverse "cuesta abajo" localmente
tiende a acercar al óptimo global), de modo que enfriar gradualmente permite
"seguir" esa estructura y converger. Esta suposición es razonable en
problemas como el *graph bipartitioning* o el TSP, pero **falla de forma
documentada y específica en el NPP cuando el vecindario es el de
single-spin-flip** (voltear el signo de un único elemento), que es
exactamente el vecindario usado en `SA4NPP_phase_diagram.jl`
(`delta_signed = -2*state[i]*numbers[i]`, líneas 277, 380, 501).

La razón, descrita ya en 1986 por **Fu y Anderson** y cuantificada
rigurosamente por **Ferreira y Fontanari (2001)**, es la siguiente: bajo un
único flip, el cambio de energía es típicamente del orden de un término
individual a_j (en nuestra notación, del orden de 2^b, la escala de un solo
número de la instancia), mientras que la energía del óptimo global es
muchísimo menor —de orden √N·2^(−N) en la escala normalizada de Ferreira y
Fontanari, o, en términos combinatorios, una fracción extremadamente
pequeña (del orden de 2^(−N) relativo al rango total de sumas posibles)—.
En palabras de ese trabajo (traducido): *el interés en este problema
proviene precisamente del notorio fracaso de la heurística estocástica de
"simulated annealing" para encontrar buenas soluciones, en comparación con
las heurísticas determinísticas; la razón de ese fracaso es que la
estrategia habitual de explorar el espacio de configuraciones mediante
flips de un único spin produce cambios de energía típicamente de orden
1/N, mientras que un análisis teórico indica que la energía del mínimo
global es de orden √N·2^(−N)* [Ferreira & Fontanari, 2001].

Esto describe exactamente un paisaje tipo **"golf course"** (o, en el
lenguaje de la mecánica estadística de sistemas desordenados, tipo
**Modelo de Energía Aleatoria / REM**, Derrida 1981; Mertens 1998, 2001,
2003): las energías de configuraciones vecinas bajo single-spin-flip **no
están correlacionadas entre sí** a la escala fina en que vive el óptimo. No
existe, por tanto, un "embudo" (*funnel*) que un *annealing* clásico pueda
seguir bajando gradualmente la temperatura para explotar. En un paisaje así:

- **Enfriar rápido congela al caminante en el primer mínimo local
  razonablemente bueno que encuentra por azar** (un estado metaestable,
  estable frente a todos los N flips individuales, pero con energía E≫1).
  Una vez que T cae lo suficiente, prácticamente ningún flip cuesta-arriba
  se acepta, y como el paisaje es incorrelacionado, tampoco hay un camino
  "cuesta abajo" de un solo flip desde ese mínimo local hacia el mínimo
  global: el caminante queda atrapado indefinidamente. Esto explica
  cuantitativamente la caída de κ_c por debajo de 0.8 que el usuario reporta
  al acelerar el enfriamiento.
- **Mantener T alta (o descender casi imperceptiblemente, como en los
  experimentos con `cooling_factor≈1`) mantiene al caminante efectuando,
  en la práctica, un muestreo amplio y poco sesgado del espacio de
  configuraciones** durante todo el presupuesto disponible. Dado que, en la
  fase κ<κ_c, el número de particiones perfectas crece exponencialmente con
  N (Mertens, 1998: para κ̃<κ_c el número esperado de particiones perfectas
  es exponencial en N; por debajo del umbral, "el número de particiones
  perfectas aumenta" según crece N para κ fijo), un recorrido suficientemente
  largo y suficientemente poco sesgado tiene más oportunidades de topar por
  azar con uno de esos óptimos (aunque sigan siendo raros frente al total de
  2^N estados) que un recorrido que se compromete tempranamente (se
  "congela") con una única cuenca de atracción.

Esta es precisamente la razón empírica, documentada también por **Johnson,
Aragon, McGeoch y Schevon (1991)**, por la que el *simulated annealing* con
la representación "directa" (asignación de signos, idéntica a la usada en
este código) **no es competitiva frente al algoritmo de diferencias de
Karmarkar y Karp (1982)**, y por la que, en sus palabras, "si se tiene en
cuenta el tiempo de cómputo, los esquemas de *annealing* convencionales ni
siquiera logran superar a múltiples corridas aleatorias independientes del
algoritmo de optimización local en el que se basan" — es decir, el
*annealing* clásico (con enfriamiento efectivo) rinde, en el mejor de los
casos, igual que un muestreo amplio sin estructura, y en el peor, peor.
**Ruml, Ngo, Marks y Shieber (1996)** muestran, en la misma línea, que lo
que realmente permite que una búsqueda estocástica supere al método de
diferencias no es ajustar el *cooling schedule*, sino cambiar la
**representación/vecindario** de búsqueda (p. ej. moverse sobre particiones
"pre-agrupadas" en vez de sobre asignaciones de signo individuales).

### 8.3 Reconciliación con el resto del análisis: dos efectos distintos, no uno solo

Con esta nueva evidencia, el análisis original debe matizarse así:

1. **El hallazgo de la Sección 4.1 (`MCS_per_T=1` para `regime="linear"`,
   verificado también en este tercer experimento) sigue siendo un hecho
   objetivo del código**, pero su interpretación cambia: no es "un error que
   limita indebidamente la equilibración térmica en cada etapa", sino,
   dado el paisaje tipo REM del NPP, un parámetro que en la práctica
   convierte a toda la fase de "etapas de temperatura" en una sucesión muy
   fina y casi continua de micro-pasos sobre un `Temp` que apenas cambia —
   es decir, el algoritmo termina pareciéndose a una cadena de Metropolis a
   temperatura efectivamente constante (o muy lentamente variable) durante
   todo el presupuesto, que es, según la Sección 8.2, la estrategia
   **correcta** para este tipo de paisaje, aunque probablemente no por
   diseño deliberado.
2. **El hallazgo de la Sección 4.3 (`cooling_factor≈1` para el experimento
   geométrico) debe reinterpretarse de la misma manera**: no es un *bug*
   que "impide enfriar", sino, de facto, el régimen que la evidencia
   empírica del usuario (y la literatura) indica que **funciona mejor**
   para este problema, precisamente porque evita el congelamiento prematuro
   en mínimos locales no correlacionados con el óptimo.
3. **Lo que NO cambia es la explicación última de por qué P_sat sigue
   cayendo a partir de N≈27 incluso con el *schedule* más lento posible.**
   Según la Sección 8.2, incluso el muestreo amplio y poco sesgado más
   favorable posible para este paisaje tiene que, en algún momento,
   "tropezar" por azar con una de las exponencialmente numerosas (pero
   exponencialmente diluidas dentro de 2^N) particiones perfectas. El
   número de pasos de Monte Carlo necesario para que esto ocurra con
   probabilidad razonable crece, en el peor caso, de forma **exponencial en
   N** (la razón entre el número de óptimos, ~2^(N(1−κ)) según Mertens, y el
   tamaño total del espacio, 2^N, decrece exponencialmente conforme κ
   aumenta hacia κ_c), mientras que el presupuesto de cómputo disponible en
   estos experimentos, `n_MCS_limit`, crece únicamente de forma **lineal en
   N** (`n_MCS_limit = 5×10⁷ · N`, o `1×10⁶ · N` en los experimentos
   anteriores). Un presupuesto que crece polinomialmente nunca puede
   sostener indefinidamente una necesidad que crece exponencialmente: por
   más que se optimice el *schedule* dentro de esa clase de algoritmos
   (single-spin-flip + Metropolis), existe necesariamente un N a partir del
   cual el presupuesto deja de alcanzar. Esto es coherente con que el
   quiebre se observe en una región de N moderado (≈27 en adelante) y no en
   N muy pequeño ni de forma asintótica muy distante.

En síntesis: **la Sección 4 original identificó correctamente los hechos
del código (MCS_per_T=1, cooling_factor≈1), pero su interpretación como
"errores que impiden el enfriamiento y por tanto perjudican la búsqueda" es
incorrecta a la luz de la bibliografía y de la evidencia experimental del
propio usuario.** Para el vecindario de single-spin-flip del NPP, ambos
"defectos" se comportan, en la práctica, como una aproximación razonable
(aunque probablemente no intencionada) a la estrategia que la literatura
recomienda: maximizar el tiempo de exploración a temperatura alta/casi
constante en vez de perseguir un enfriamiento agresivo. Lo que si sigue
siendo válido, y es la causa de fondo que no depende del *schedule*, es el
**desajuste entre el crecimiento exponencial de la dificultad de muestreo
cerca de κ_c y el crecimiento solo polinomial del presupuesto de cómputo
asignado**, agravado por la ausencia de una regla que indique de forma
confiable, para cada N y κ, cuánto presupuesto haría falta.

### 8.4 Recomendaciones revisadas

Se **retiran o matizan** las recomendaciones 2 y 3 de la Sección 6
(calibrar `cooling_factor`/β para que el sistema sí llegue a congelar
dentro del presupuesto, y evitar valores cercanos a 1): a la luz de 8.1–8.3,
seguirlas literalmente degradaría el rendimiento, tal como el usuario ya
observó. Se mantienen vigentes, y se refuerzan, las siguientes:

1. **No usar la velocidad de enfriamiento como palanca principal de mejora.**
   La evidencia (propia y de la literatura) indica que, para el vecindario
   de single-spin-flip, acercarse a un régimen de temperatura casi
   constante y sostenida es preferible a un enfriamiento agresivo. Si se
   desea seguir explorando esta familia de algoritmos, es razonable fijar
   `cooling_factor`/β en un valor muy cercano al "sin enfriamiento" de forma
   **explícita e intencional** (no como un artefacto accidental de haber
   escrito muchos nueves), documentando que se trata de una elección
   deliberada de favorecer la exploración, no una fase de "temple" en el
   sentido clásico.
2. **Corregir igualmente el cálculo de `MCS_per_T` (Sección 4.1/6.1)**, pero
   con otro objetivo: no para "dar más tiempo de equilibración por etapa",
   sino para que el número total de intentos de flip (`n_MCS_used × N`)
   sea el parámetro que se controla y reporta explícitamente, en vez de ser
   un efecto colateral confuso de un exponente mal definido. Mantener la
   semántica (qué significa "linear"/"quadratic"/"cubic" aquí) clara y
   documentada evitará interpretaciones erróneas como la de la Sección 4
   original.
3. **Dado que la causa de fondo es el desajuste polinomio-vs-exponencial de
   8.3, la palanca más efectiva no es el *schedule*, sino (en orden de
   impacto esperado según la bibliografía):**
   a. **Cambiar la representación/vecindario de búsqueda**, siguiendo a
      Ruml et al. (1996) y Johnson et al. (1991): en vez de single-spin-flip
      sobre asignaciones de signo, usar movimientos sobre una
      representación "pre-particionada" (p. ej. inicializar/combinar con el
      algoritmo de diferencias de Karmarkar y Karp, o usar un esquema de
      *swap* de elementos entre los dos subconjuntos en vez de voltear un
      signo) — esto ataca directamente la causa raíz (falta de correlación
      local) en vez de intentar compensarla con más tiempo de cómputo.
   b. **Usar SA como fase de refinamiento sobre una solución inicial ya
      buena** obtenida por el método de diferencias de Karmarkar-Karp (KK),
      en vez de partir de un estado aleatorio: KK típicamente entrega
      energías de orden 1/N^(α·log N), muy por debajo del punto de partida
      aleatorio, reduciendo drásticamente la distancia (en número de flips)
      que el *annealing* necesita recorrer.
   c. **Reiniciar (random restarts) en vez de, o además de, alargar una
      única corrida**: dado que el paisaje es tipo REM, muchas corridas
      cortas e independientes pueden ser más eficientes por unidad de
      cómputo que una sola corrida muy larga a temperatura constante,
      porque reducen la correlación entre pasos consecutivos (en una cadena
      de Metropolis a T fija, pasos consecutivos están correlacionados y
      "desperdician" presupuesto revisitando regiones cercanas).
   d. Si se desea seguir cuantificando κ_c(N) específicamente con SA de
      single-spin-flip (p. ej. para comparar con la cota teórica de Mertens
      como ejercicio de caracterización, no como heurística de producción),
      **reportar explícitamente el régimen de T utilizado (casi-isotérmico)
      y escalar `n_MCS_limit` de forma supra-lineal en N** —idealmente
      informada por la propia fórmula de Mertens (la razón
      2^(N(1−κ))/2^N), para que el presupuesto y la dificultad escalen de
      forma comparable en vez de divergir como ocurre actualmente.
4. Las recomendaciones 1, 4, 5, 6 y 7 de la Sección 6 (descarte de punto
   flotante, muestras/resolución en κ, validación cruzada con resolutor
   exacto, guardas para instancias futuras con N/b mayores) **se mantienen
   sin cambios**: no están relacionadas con la velocidad de enfriamiento y
   siguen siendo válidas.

### 8.5 Bibliografía

- Y. Fu and P. W. Anderson, "Application of statistical mechanics to
  NP-complete problems in combinatorial optimisation," *J. Phys. A: Math.
  Gen.* 19, 1605–1620 (1986).
- D. S. Johnson, C. R. Aragon, L. A. McGeoch, and C. Schevon, "Optimization
  by Simulated Annealing: An Experimental Evaluation; Part II, Graph
  Coloring and Number Partitioning," *Operations Research* 39(3), 378–406
  (1991).
- D. S. Johnson, C. R. Aragon, L. A. McGeoch, and C. Schevon, "Optimization
  by Simulated Annealing: An Experimental Evaluation; Part I, Graph
  Partitioning," *Operations Research* 37(6), 865–892 (1989).
- W. Ruml, J. T. Ngo, J. Marks, and S. M. Shieber, "Easily Searched
  Encodings for Number Partitioning," *Journal of Optimization Theory and
  Applications* 89(2), 251–291 (1996).
- N. Karmarkar and R. M. Karp, "The Differencing Method of Set
  Partitioning," Report UCB/CSD 82/113, Computer Science Division, EECS,
  University of California, Berkeley (1982).
- F. F. Ferreira and J. F. Fontanari, "Probabilistic Analysis of the Number
  Partitioning Problem," cond-mat/adap-org 9801002; ver también la versión
  publicada en *J. Phys. A: Math. Gen.* (1998).
- S. Mertens, "Phase Transition in the Number Partitioning Problem,"
  *Phys. Rev. Lett.* 81, 4281 (1998).
- S. Mertens, "A complete anytime algorithm for number partitioning,"
  *Artificial Intelligence* 139, 275–328 (2002).
- S. Mertens, "A physicist's approach to number partitioning," cond-mat/0009230;
  *Theoretical Computer Science* 265, 79–108 (2001); también publicado como
  capítulo introductorio en *Computational Complexity and Statistical
  Physics* (Oxford Univ. Press, 2003).
- B. Derrida, "Random-Energy Model: An exactly solvable model of disordered
  systems," *Phys. Rev. B* 24, 2613 (1981).
- S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by Simulated
  Annealing," *Science* 220(4598), 671–680 (1983).

## 9. Conclusión final (actualizada)

La causa de fondo de la caída de P_sat/κ_c a partir de N≈27 **no es un
problema de precisión de punto flotante** (confirmado en la Sección 3) ni,
como se afirmó preliminarmente en la Sección 4, un simple error de
calibración del *cooling schedule* que bastaría con "corregir" acelerando
el enfriamiento. La evidencia experimental adicional aportada por el
usuario, contrastada con la bibliografía clásica sobre SA aplicado al NPP
(Fu & Anderson 1986; Johnson et al. 1991; Ferreira & Fontanari 1998; Mertens
1998–2003), indica que el **vecindario de single-spin-flip produce un
paisaje de energía sin correlación local (tipo "golf course"/REM)** en el
que un enfriamiento rápido es activamente contraproducente, y en el que el
régimen casi-isotérmico que el código ejecuta hoy (por una combinación de
`MCS_per_T=1` y `cooling_factor≈1`) se aproxima, probablemente sin que
fuera intencional, al comportamiento que la literatura recomienda para este
problema. El límite real que explica la caída hacia N≈27 y en adelante es
estructural: el número de pasos de Monte Carlo necesario para "tropezar"
con una de las particiones perfectas (exponencialmente numerosas pero
exponencialmente diluidas en el espacio 2^N) crece exponencialmente con N,
mientras que el presupuesto de cómputo asignado (`n_MCS_limit`) solo crece
linealmente con N. Ninguna recalibración del *schedule* dentro de la
familia single-spin-flip + Metropolis puede revertir ese desajuste de
manera sostenible; las vías con respaldo bibliográfico para mejorar el
rendimiento son, en cambio, cambiar la representación/vecindario de
búsqueda (Ruml et al. 1996), inicializar desde una solución de
Karmarkar-Karp, o usar reinicios múltiples en vez de una única corrida muy
larga.
