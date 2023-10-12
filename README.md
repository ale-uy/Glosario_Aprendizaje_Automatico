# Fundamentos de Machine Learning

### Activation functions (Función de activación)
Una función que permite a las redes neuronales aprender relaciones no lineales (complejas) entre las características y la etiqueta.

Las funciones de activación populares incluyen:

* ReLU
* Sigmoid
* Softmax

Las gráficas de funciones de activación nunca son líneas rectas simples.

### AUC (Área bajo la curva ROC)
Un número entre 0,0 y 1,0 que representa la capacidad de un modelo de clasificación binaria para separar clases positivas de clases negativas. Cuanto más cerca esté el AUC de 1,0, mejor será la capacidad del modelo para separar clases entre sí.

La mayoría de los modelos se encuentran en algún punto entre los dos extremos.
AUC ignora cualquier valor que establezca para el umbral de clasificación. En cambio, las AUC consideran todos los umbrales de clasificación posibles.

### Backpropagation (Propagación hacia atras)
El algoritmo que implementa el descenso de gradiente en redes neuronales.

Entrenar una red neuronal implica muchas iteraciones del siguiente ciclo de dos pasos:

1. Durante el avance, el sistema procesa un lote de ejemplos para generar predicciones. El sistema compara cada predicción con cada valor de etiqueta. La diferencia entre la predicción y el valor de la etiqueta es la pérdida para ese ejemplo. El sistema agrega las pérdidas de todos los ejemplos para calcular la pérdida total del lote actual.
2. Durante el paso hacia atrás (propagación hacia atrás), el sistema reduce la pérdida ajustando los pesos de todas las neuronas en todas las capas ocultas.

Las redes neuronales suelen contener muchas neuronas en muchas capas ocultas. Cada una de esas neuronas contribuye a la pérdida general de diferentes maneras. La retropropagación determina si se deben aumentar o disminuir los pesos aplicados a neuronas particulares.

La tasa de aprendizaje es un multiplicador que controla el grado en que cada pase hacia atrás aumenta o disminuye cada peso. Una tasa de aprendizaje alta aumentará o disminuirá cada peso más que una tasa de aprendizaje pequeña.

En términos de cálculo, la retropropagación implementa la regla de la cadena del cálculo. Es decir, la retropropagación calcula la derivada parcial del error con respecto a cada parámetro. Para obtener más detalles, consulte este tutorial en el curso intensivo de aprendizaje automático.

Hace años, los profesionales del aprendizaje automático tenían que escribir código para implementar la retropropagación. Las API de aprendizaje automático modernas, como TensorFlow, ahora implementan la retropropagación por usted.

### Batch (Lote)
El conjunto de ejemplos utilizados en una iteración de entrenamiento. El tamaño del lote determina la cantidad de ejemplos en un lote.

Consulte **epoch** para obtener una explicación de cómo se relaciona un lote con una época.

### Batch Size (Tamaño del lote)
El número de ejemplos en un lote. Por ejemplo, si el tamaño del lote es 100, entonces el modelo procesa 100 ejemplos por iteración.

Las siguientes son estrategias populares de tamaño de lote:

* Descenso de gradiente estocástico (SGD), en el que el tamaño del lote es 1.
* lote completo, en el que el tamaño del lote es el número de ejemplos en todo el conjunto de entrenamiento. Por ejemplo, si el conjunto de entrenamiento contiene un millón de ejemplos, entonces el tamaño del lote sería de un millón de ejemplos. El lote completo suele ser una estrategia ineficiente.
* minilote en el que el tamaño del lote suele estar entre 10 y 1000. El minilote suele ser la estrategia más eficiente.

### Bias (Sesgo)
Una intersección o desplazamiento desde un origen. El sesgo es un parámetro en los modelos de aprendizaje automático, que se simboliza mediante **b** o **w0**

En una línea bidimensional simple, el sesgo simplemente significa "intercepción y".
Existe sesgo porque no todos los modelos parten del origen (0,0). Por ejemplo, supongamos que la entrada a un parque de atracciones cuesta 2 euros y 0,5 euros adicionales por cada hora de estancia de un cliente. Por tanto, un modelo que mapea el coste total tiene un sesgo de 2 porque el coste más bajo es 2 euros.

El sesgo no debe confundirse con el sesgo en ética y equidad o sesgo de predicción.

### Bucketing ("Cubos", se interpreta como segmentación)
Convertir una única característica en múltiples características binarias llamadas buckets o bins, generalmente en función de un rango de valores. La característica cortada suele ser una característica continua.

Por ejemplo, en lugar de representar la temperatura como una única característica continua de punto flotante, podría dividir rangos de temperaturas en grupos discretos, como por ejemplo:

><= 10 grados centígrados sería el nivel "frío".
11 - 24 grados centígrados sería el nivel "templado".
\>= 25 grados Celsius sería el nivel "cálido".

El modelo tratará todos los valores del mismo grupo de forma idéntica. Por ejemplo, los valores 13 y 22 están ambos en el grupo templado, por lo que el modelo trata los dos valores de manera idéntica.

### Classification threshold (Límite o Umbral de clasificación)

En una clasificación binaria, un número entre 0 y 1 que convierte el resultado bruto de un modelo de regresión logística en una predicción de la clase positiva o de la clase negativa. Tenga en cuenta que el umbral de clasificación es un valor que elige un ser humano, no un valor elegido mediante el entrenamiento del modelo.

Un modelo de regresión logística genera un valor bruto entre 0 y 1. Entonces:

Si este valor bruto es mayor que el umbral de clasificación, entonces se predice la clase positiva.
Si este valor bruto es menor que el umbral de clasificación, entonces se predice la clase negativa.
Por ejemplo, supongamos que el umbral de clasificación es 0,8. Si el valor bruto es 0,9, entonces el modelo predice la clase positiva. Si el valor bruto es 0,7, entonces el modelo predice la clase negativa.

La elección del umbral de clasificación influye fuertemente en el número de falsos positivos y falsos negativos.

### Clipping (Recorte)
Una técnica para manejar valores atípicos realizando una o ambas de las siguientes acciones:

* Reducir los valores de características que son mayores que un umbral máximo hasta ese umbral máximo.
* Aumentar los valores de características que son inferiores a un umbral mínimo hasta ese umbral mínimo.

Por ejemplo, supongamos que <0,5% de los valores de una característica particular están fuera del rango 40-60. En este caso, podrías hacer lo siguiente:

* Recorte todos los valores superiores a 60 (el umbral máximo) para que sean exactamente 60.
* Recorte todos los valores por debajo de 40 (el umbral mínimo) para que sean exactamente 40.

Los valores atípicos pueden dañar los modelos y, en ocasiones, provocar que los pesos se desborden durante el entrenamiento. Algunos valores atípicos también pueden arruinar drásticamente métricas como la precisión. El recorte es una técnica común para limitar el daño.

El recorte de gradiente fuerza los valores de gradiente dentro de un rango designado durante el entrenamiento.

### Dense feature (Característica Densa)
Característica en la que la mayoría o todos los valores son distintos de cero, normalmente un tensor de valores de punto flotante.

### Depth (Profundidad)
La suma de lo siguiente en una red neuronal:

* el número de capas ocultas
* el número de capas de salida, que normalmente es 1
* el número de capas de incrustación

Por ejemplo, una red neuronal con cinco capas ocultas y una capa de salida tiene una profundidad de 6.

Observe que la capa de entrada no influye en la profundidad.

### Dynamic (Dinámica)
Algo que se hace frecuente o continuamente. Los términos dinámico y online son sinónimos en el aprendizaje automático. Los siguientes son usos comunes de dinámico y en línea en el aprendizaje automático:

* Un modelo dinámico (o modelo en línea) es un modelo que se reentrena frecuente o continuamente.
* La formación dinámica (o formación online) es el proceso de formación frecuente o continua.
* La inferencia dinámica (o inferencia en línea) es el proceso de generar predicciones bajo demanda.

### Dynamic model (Modelo Dinámico)
Un modelo que se reentrena con frecuencia (tal vez incluso continuamente). Un modelo dinámico es un "aprendizaje permanente" que se adapta constantemente a los datos en evolución. Un modelo dinámico también se conoce como modelo en línea.

Contraste con el modelo estático.

### Early stopping (Parada anticipada)
Un método de regularización que implica finalizar el entrenamiento antes de que la pérdida de entrenamiento termine de disminuir. Al detenerse anticipadamente, se deja de entrenar el modelo intencionalmente cuando la pérdida en un conjunto de datos de validación comienza a aumentar; es decir, cuando el rendimiento de la generalización empeora.

### Embedding layer (Capa incrustada)
Una capa oculta especial que se entrena en una característica categórica de alta dimensión para aprender gradualmente un vector de incrustación de menor dimensión. Una capa de incrustación permite que una red neuronal se entrene de manera mucho más eficiente que entrenar solo en la característica categórica de alta dimensión.

Por ejemplo, la Tierra actualmente alberga alrededor de 73.000 especies de árboles. Supongamos que las especies de árboles son una característica de su modelo, por lo que la capa de entrada de su modelo incluye un vector único de 73 000 elementos de largo.

Una matriz de 73.000 elementos es muy larga. Si no agrega una capa de incrustación al modelo, el entrenamiento llevará mucho tiempo debido a la multiplicación de 72,999 ceros. Quizás elija que la capa de incrustación tenga 12 dimensiones. En consecuencia, la capa de incrustación aprenderá gradualmente un nuevo vector de incrustación para cada especie de árbol.

En determinadas situaciones, el hashing es una alternativa razonable a una capa de incrustación.

### Epoch ("Época", conjunto de iteracciónes)
Un pase de entrenamiento completo sobre todo el conjunto de entrenamiento de modo que cada ejemplo se haya procesado una vez.

Una época representa N iteraciones de entrenamiento de tamaño de lote, donde N es el número total de ejemplos.

Por ejemplo, supongamos lo siguiente:

El conjunto de datos consta de 1.000 ejemplos.
El tamaño del lote es de 50 ejemplos.
Por tanto, una sola época requiere 20 iteraciones

### Feature cross (Caracteristica cruzada)
Una característica sintética formada "cruzando" características categóricas o agrupadas.

Por ejemplo, considere un modelo de "previsión del estado de ánimo" que representa la temperatura en uno de los cuatro grupos siguientes:

> *congelación*, *frío*, *templado*, *cálido*.

Y representa la velocidad del viento en uno de los siguientes dos grupos:

>*luz*, *ventoso*.

Sin cruces de características, el modelo lineal se entrena de forma independiente en cada uno de los seis grupos anteriores. Por lo tanto, el modelo entrena, por ejemplo, en congelación independientemente del entrenamiento en, por ejemplo, viento.

Alternativamente, puede crear una característica cruzada de temperatura y velocidad del viento. Esta característica sintética tendría los siguientes valores posibles:

>luz helada
viento helado
luz fría
frío y ventoso
luz-templada
templado-ventoso
luz calida
viento cálido

Gracias a los cruces de funciones, el modelo puede aprender las diferencias de humor entre un día helado y ventoso y un día helado.

Si crea una característica sintética a partir de dos características que tienen cada una muchos grupos diferentes, el cruce de características resultante tendrá una gran cantidad de combinaciones posibles. Por ejemplo, si una entidad tiene 1000 depósitos y la otra entidad tiene 2000 depósitos, el cruce de funciones resultante tiene 2 000 000 de depósitos.

Formalmente, una cruz es un producto cartesiano.

Los cruces de características se usan principalmente con modelos lineales y rara vez se usan con redes neuronales.

### Feature engineering (Ingenieria de Caracteristicas)
Un proceso que involucra los siguientes pasos:

* Determinar qué características podrían ser útiles al entrenar un modelo.
* Convertir datos sin procesar del conjunto de datos en versiones eficientes de esas funciones.

Por ejemplo, podría determinar que la temperatura podría ser una característica útil. Luego, podría experimentar con el agrupamiento para optimizar lo que el modelo puede aprender de diferentes rangos de temperatura.

La ingeniería de características a veces se denomina extracción de características (feature extraction).

### Generalización (Generalización)
La capacidad de un modelo para hacer predicciones correctas sobre datos nuevos, nunca antes vistos. Un modelo que puede generalizar es lo opuesto a un modelo que se sobreajusta.

### Generalization curve (Curva de Generalización)
Un gráfico de la pérdida de entrenamiento y de la pérdida de validación en función del número de iteraciones.

Una curva de generalización puede ayudarle a detectar un posible sobreajuste. Por ejemplo, si la pérdida de validación se vuelve significativamente mayor que la pérdida de entrenamiento, sugiere un sobreajuste.

### Gradient Descent (Descenso de Gradiente)
Una técnica matemática para minimizar las pérdidas. El descenso de gradiente ajusta de forma iterativa pesos y sesgos, encontrando gradualmente la mejor combinación para minimizar la pérdida.

El descenso de gradientes es más antiguo (mucho, mucho más antiguo) que el aprendizaje automático.

### Hyperparameters (Hiperparametros)
Las variables que usted o un servicio de ajuste de hiperparámetros ajustan durante ejecuciones sucesivas de entrenamiento de un modelo. Por ejemplo, la tasa de aprendizaje es un hiperparámetro. Puedes establecer la tasa de aprendizaje en 0,01 antes de una sesión de entrenamiento. Si determina que 0,01 es demasiado alto, quizás pueda establecer la tasa de aprendizaje en 0,003 para la siguiente sesión de entrenamiento.

Por el contrario, los parámetros son los distintos pesos y sesgos que el modelo aprende durante el entrenamiento.

### Interpretabilidad
La capacidad de explicar o presentar el razonamiento de un modelo de ML en términos comprensibles para un ser humano.

La mayoría de los modelos de regresión lineal, por ejemplo, son altamente interpretables. (Simplemente necesita observar los pesos entrenados para cada característica). Los bosques de decisiones también son altamente interpretables. Algunos modelos, sin embargo, requieren una visualización sofisticada para ser interpretables.

Puede utilizar la herramienta de interpretación de aprendizaje ([LIT](https://developers.google.com/machine-learning/glossary#Learning-Interpretability-Tool)) para interpretar modelos de ML.

### Iteration (Iteracion)
Una única actualización de los parámetros de un modelo (las ponderaciones y los sesgos del modelo) durante el entrenamiento. El tamaño del lote determina cuántos ejemplos procesa el modelo en una sola iteración. Por ejemplo, si el tamaño del lote es 20, entonces el modelo procesa 20 ejemplos antes de ajustar los parámetros.

Al entrenar una red neuronal, una única iteración implica los dos pasos siguientes:

* Un pase directo para evaluar la pérdida en un solo lote.
* Un paso hacia atrás (backpropagation) para ajustar los parámetros del modelo en función de la pérdida y la tasa de aprendizaje.

### L0 Regularization (Regularización L0 o de norma)
Un tipo de regularización que penaliza el número total de pesos distintos de cero en un modelo. Por ejemplo, un modelo que tenga 11 pesos distintos de cero sería penalizado más que un modelo similar que tenga 10 pesos distintos de cero.

La regularización L0 a veces se denomina regularización de norma L0.

### L1 Loss (Pérdida L1)
Una función de pérdida que calcula el valor absoluto de la diferencia entre los valores reales de las etiquetas y los valores que predice un modelo.
La pérdida L1 es menos sensible a los valores atípicos que la pérdida L2.

El error absoluto medio es la pérdida L1 promedio por ejemplo.

### L1 regularization (Regularización L1 o por escazes)
Un tipo de regularización que penaliza los pesos en proporción a la suma del valor absoluto de los pesos. La regularización L1 ayuda a llevar los pesos de características irrelevantes o apenas relevantes a exactamente 0. Una característica con un peso de 0 se elimina efectivamente del modelo.

Contraste con la regularización L2.

### L2 Loss (Pérdida L2)
Una función de pérdida que calcula el cuadrado de la diferencia entre los valores reales de las etiquetas y los valores que predice un modelo.
Debido a la elevación al cuadrado, la pérdida de L2 amplifica la influencia de los valores atípicos. Es decir, la pérdida L2 reacciona más fuertemente a las malas predicciones que la pérdida L1. Por ejemplo, la pérdida L1 para el lote anterior sería 8 en lugar de 16. Observe que un solo valor atípico representa 9 de los 16.

Los modelos de regresión suelen utilizar la pérdida L2 como función de pérdida.

El error cuadrático medio es la pérdida L2 promedio por ejemplo. La pérdida al cuadrado es otro nombre para la pérdida L2.

### L2 regularization (Regularización L2 o por simplicidad)
Un tipo de regularización que penaliza los pesos en proporción a la suma de los cuadrados de los pesos. La regularización L2 ayuda a acercar los pesos de los valores atípicos (aquellos con valores positivos altos o negativos bajos) a 0, pero no del todo a 0. Las características con valores muy cercanos a 0 permanecen en el modelo, pero no influyen mucho en la predicción del modelo.

La regularización L2 siempre mejora la generalización en modelos lineales.

Contraste con la regularización L1.

### Learning Rate (Tasa de Aprendizaje)
Un número de punto flotante que le indica al algoritmo de descenso de gradiente con qué fuerza ajustar los pesos y los sesgos en cada iteración. Por ejemplo, una tasa de aprendizaje de 0,3 ajustaría las ponderaciones y los sesgos tres veces más poderosamente que una tasa de aprendizaje de 0,1.

La tasa de aprendizaje es un hiperparámetro clave. Si establece una tasa de aprendizaje demasiado baja, la capacitación llevará demasiado tiempo. Si establece una tasa de aprendizaje demasiado alta, el descenso de gradiente a menudo tendrá problemas para alcanzar la convergencia.

### Linear Model (Modelo Lineal)
Un modelo que asigna un peso por característica para hacer predicciones. (Los modelos lineales también incorporan un sesgo). Por el contrario, la relación de las características con las predicciones en los modelos profundos generalmente no es lineal.

Los modelos lineales suelen ser más fáciles de entrenar y más interpretables que los modelos profundos. Sin embargo, los modelos profundos pueden aprender relaciones complejas entre características.

La regresión lineal y la regresión logística son dos tipos de modelos lineales.

### Regresión Lineal 
Un tipo de modelo de aprendizaje automático en el que se cumplen las dos condiciones siguientes:

* El modelo es un modelo lineal.
* La predicción es un valor de punto flotante. (Esta es la parte de regresión de la regresión lineal).

Contraste la regresión lineal con la regresión logística. Además, compare la regresión con la clasificación.

### Regresión Logística
Un tipo de modelo de regresión que predice una probabilidad. Los modelos de regresión logística tienen las siguientes características:

* La etiqueta es categórica. El término regresión logística suele referirse a la regresión logística binaria, es decir, a un modelo que calcula probabilidades para etiquetas con dos valores posibles. Una variante menos común, la regresión logística multinomial, calcula probabilidades para etiquetas con más de dos valores posibles.
* La función de pérdida durante el entrenamiento es Log Loss. (Se pueden colocar varias unidades de pérdida de registros en paralelo para etiquetas con más de dos valores posibles).
* El modelo tiene una arquitectura lineal, no una red neuronal profunda. Sin embargo, el resto de esta definición también se aplica a modelos profundos que predicen probabilidades de etiquetas categóricas.

Por ejemplo, considere un modelo de regresión logística que calcula la probabilidad de que un correo electrónico entrante sea spam o no. Durante la inferencia, supongamos que el modelo predice 0,72. Por tanto, el modelo está estimando:

* Un 72% de posibilidades de que el correo electrónico sea spam.
* Un 28% de posibilidades de que el correo electrónico no sea spam.

Un modelo de regresión logística utiliza la siguiente arquitectura de dos pasos:

* El modelo genera una predicción bruta (y') aplicando una función lineal de características de entrada.
* El modelo utiliza esa predicción sin procesar como entrada para una función sigmoidea, que convierte la predicción sin procesar a un valor entre 0 y 1, exclusivo.

Como cualquier modelo de regresión, un modelo de regresión logística predice un número. Sin embargo, este número normalmente pasa a formar parte de un modelo de clasificación binaria de la siguiente manera:

* Si el número predicho es mayor que el umbral de clasificación, el modelo de clasificación binaria predice la clase positiva.
* Si el número predicho es menor que el umbral de clasificación, el modelo de clasificación binaria predice la clase negativa.

### Log Loss (Logarítmo de Pérdida)
La función de pérdida utilizada en la regresión logística binaria.

### Loss Function (Función de pérdida)
Durante el entrenamiento o las pruebas, una función matemática que calcula la pérdida en un lote de ejemplos. Una función de pérdida devuelve una pérdida menor para los modelos que hacen buenas predicciones que para los modelos que hacen malas predicciones.

El objetivo del entrenamiento suele ser minimizar la pérdida que devuelve una función de pérdida.

Existen muchos tipos diferentes de funciones de pérdida. Elija la función de pérdida adecuada para el tipo de modelo que está construyendo. Por ejemplo:

* La pérdida L2 (o error cuadrático medio) es la función de pérdida para la regresión lineal.
* Log Loss es la función de pérdida para la regresión logística.

### Model (Modelo)
En general, cualquier construcción matemática que procese datos de entrada y devuelva resultados. Dicho de otra manera, un modelo es el conjunto de parámetros y estructura necesarios para que un sistema haga predicciones. En el aprendizaje automático supervisado, un modelo toma un ejemplo como entrada e infiere una predicción como salida. Dentro del aprendizaje automático supervisado, los modelos difieren algo. Por ejemplo:

* Un modelo de regresión lineal consta de un conjunto de ponderaciones y un sesgo.
* Un modelo de red neuronal consta de:
    * Un conjunto de capas ocultas, cada una de las cuales contiene una o más neuronas.
    * Los pesos y sesgos asociados con cada neurona.
* Un modelo de árbol de decisión consta de:
    * La forma del árbol; es decir, el patrón en el que se conectan las condiciones y las hojas.
    * Las condiciones y las hojas.

Puede guardar, restaurar o hacer copias de un modelo.

El aprendizaje automático no supervisado también genera modelos, normalmente una función que puede asignar un ejemplo de entrada al clúster más apropiado.

### Multi-class Classification (Clasificación multi clase)

En el aprendizaje supervisado, un problema de clasificación en el que el conjunto de datos contiene más de dos clases de etiquetas. Por ejemplo, las etiquetas del conjunto de datos Iris deben ser una de las tres clases siguientes:

>Iris setosa
iris virginica
iris versicolor

Un modelo entrenado en el conjunto de datos de Iris que predice el tipo de Iris en nuevos ejemplos está realizando una clasificación de clases múltiples.

Por el contrario, los problemas de clasificación que distinguen exactamente dos clases son modelos de clasificación binaria. Por ejemplo, un modelo de correo electrónico que predice spam o no spam es un modelo de clasificación binaria.

En los problemas de agrupamiento, la clasificación de clases múltiples se refiere a más de dos grupos.

### Neurona
En aprendizaje automático, una unidad distinta dentro de una capa oculta de una red neuronal. Cada neurona realiza la siguiente acción de dos pasos:

* Calcula la suma ponderada de los valores de entrada multiplicada por sus pesos correspondientes.
* Pasa la suma ponderada como entrada a una función de activación.

Una neurona en la primera capa oculta acepta entradas de los valores de las características en la capa de entrada. Una neurona en cualquier capa oculta más allá de la primera acepta entradas de las neuronas de la capa oculta anterior. Por ejemplo, una neurona de la segunda capa oculta acepta entradas de las neuronas de la primera capa oculta.

Una neurona en una red neuronal imita el comportamiento de las neuronas del cerebro y otras partes del sistema nervioso.

### Nonstationarity (No estacional)
Característica cuyos valores cambian en una o más dimensiones, generalmente el tiempo. Por ejemplo, considere los siguientes ejemplos de no estacionariedad:

* La cantidad de trajes de baño vendidos en una tienda en particular varía según la temporada.
* La cantidad de una fruta particular cosechada en una región particular es cero durante gran parte del año, pero grande durante un breve período.
* Debido al cambio climático, las temperaturas medias anuales están cambiando.

### Normalization (Normalización)
En términos generales, el proceso de convertir el rango de valores real de una variable en un rango de valores estándar, como por ejemplo:

>-1 a +1 | 0 a 1 | la distribución normal

Por ejemplo, supongamos que el rango real de valores de una determinada característica es de 800 a 2400. Como parte de la ingeniería de funciones, puede normalizar los valores reales hasta un rango estándar, como -1 a +1.

La normalización es una tarea común en la ingeniería de características. Los modelos normalmente se entrenan más rápido (y producen mejores predicciones) cuando cada característica numérica en el vector de características tiene aproximadamente el mismo rango.

### One-Hot encode (Codificación one-hot)
Representar datos categóricos como un vector en el que:

* Un elemento se establece en 1.
* Todos los demás elementos se establecen en 0.

La codificación one-hot se usa comúnmente para representar cadenas o identificadores que tienen un conjunto finito de valores posibles.

### Overfitting (Sobreentrenamiento)
Crear un modelo que coincida tanto con los datos de entrenamiento que el modelo no pueda hacer predicciones correctas sobre datos nuevos.

La regularización puede reducir el sobreajuste. Entrenar en un conjunto de entrenamiento grande y diverso también puede reducir el sobreajuste.

### Post-Procesamiento
Ajustar la salida de un modelo después de ejecutarlo. El posprocesamiento se puede utilizar para imponer restricciones de equidad sin modificar los modelos en sí.

Por ejemplo, se podría aplicar el posprocesamiento a un clasificador binario estableciendo un umbral de clasificación tal que se mantenga la igualdad de oportunidades para algún atributo verificando que la tasa de verdaderos positivos sea la misma para todos los valores de ese atributo.

### Proxy labels (Variables o características proxy)
Datos utilizados para aproximar etiquetas que no están disponibles directamente en un conjunto de datos.

Por ejemplo, suponga que debe entrenar un modelo para predecir el nivel de estrés de los empleados. Su conjunto de datos contiene muchas características predictivas, pero no contiene una etiqueta denominada nivel de estrés. Sin desanimarse, elige "accidentes laborales" como etiqueta representativa del nivel de estrés. Después de todo, los empleados sometidos a mucho estrés sufren más accidentes que los empleados tranquilos. O quizás los accidentes laborales en realidad aumentan y disminuyen por múltiples razones.

Como segundo ejemplo, supongamos que quiere saber: ¿está lloviendo? ser una etiqueta booleana para su conjunto de datos, pero su conjunto de datos no contiene datos de lluvia. Si hay fotografías disponibles, podría establecer imágenes de personas llevando paraguas como una etiqueta representativa de ¿está lloviendo? ¿Es esa una buena etiqueta proxy? Posiblemente, pero es más probable que las personas en algunas culturas lleven paraguas para protegerse del sol que de la lluvia.

Las etiquetas proxy suelen ser imperfectas. Cuando sea posible, elija etiquetas reales en lugar de etiquetas proxy. Dicho esto, cuando no haya una etiqueta real, elija la etiqueta proxy con mucho cuidado y elija la candidata a etiqueta proxy menos horrible.

### Rectified Linear Unit (ReLU) (Unidad lineal rectificada)
Una función de activación con el siguiente comportamiento:

* Si la entrada es negativa o cero, entonces la salida es 0.
* Si la entrada es positiva, entonces la salida es igual a la entrada.

Por ejemplo:

* Si la entrada es -3, entonces la salida es 0.
* Si la entrada es +3, entonces la salida es 3,0.

ReLU es una función de activación muy popular. A pesar de su comportamiento simple, ReLU aún permite que una red neuronal aprenda relaciones no lineales entre las características y la etiqueta.

### Regression Model (Modelo de regresión)
Informalmente, un modelo que genera una predicción numérica. (Por el contrario, un modelo de clasificación genera una predicción de clase). Por ejemplo, los siguientes son todos modelos de regresión:

* Un modelo que predice el valor de una determinada casa, como por ejemplo 423.000 euros.
* Un modelo que predice la esperanza de vida de un determinado árbol, por ejemplo 23,2 años.
* Un modelo que predice la cantidad de lluvia que caerá en una determinada ciudad durante las próximas seis horas, por ejemplo 0,18 pulgadas.

Dos tipos comunes de modelos de regresión son:

* Regresión lineal, que encuentra la línea que mejor ajusta los valores de las etiquetas a las características.
* Regresión logística, que genera una probabilidad entre 0,0 y 1,0 de que un sistema normalmente se asigne a una predicción de clase.

No todos los modelos que generan predicciones numéricas son modelos de regresión. En algunos casos, una predicción numérica es en realidad solo un modelo de clasificación que tiene nombres de clases numéricos. Por ejemplo, un modelo que predice un código postal numérico es un modelo de clasificación, no un modelo de regresión.

### Regularization rate (Tasa de Regularizacion)
Un número que especifica la importancia relativa de la regularización durante el entrenamiento. El aumento de la tasa de regularización reduce el sobreajuste, pero puede reducir el poder predictivo del modelo. Por el contrario, reducir u omitir la tasa de regularización aumenta el sobreajuste.

### ROC Curve, receiver operating characteristic curve (Curva característica de funcionamiento del receptor)
Un gráfico de la tasa de verdaderos positivos frente a la tasa de falsos positivos para diferentes umbrales de clasificación en la clasificación binaria.

La forma de una curva ROC sugiere la capacidad de un modelo de clasificación binaria para separar clases positivas de clases negativas.

El punto de una curva ROC más cercano al vertice izquierdo, identifica teóricamente el umbral de clasificación ideal. Sin embargo, varias otras cuestiones del mundo real influyen en la selección del umbral de clasificación ideal. Por ejemplo, quizás los falsos negativos causen mucho más dolor que los falsos positivos.

Una métrica numérica llamada AUC resume la curva ROC en un único valor de punto flotante.

### Sigmoid Function (Función Sigmoidea)
Una función matemática que "comprime" un valor de entrada en un rango restringido, normalmente de [0 a 1] o de [-1 a +1]. Es decir, puede pasar cualquier número (dos, un millón, mil millones negativos, lo que sea) a un sigmoide y la salida seguirá estando en el rango restringido.

La función sigmoidea tiene varios usos en el aprendizaje automático, que incluyen:

* Convertir el resultado bruto de un modelo de regresión logística o de regresión multinomial en una probabilidad.
* Actuando como función de activación en algunas redes neuronales.

### Softmax (Máximo débil)
Función que determina probabilidades para cada clase posible en un modelo de clasificación de clases múltiples. Las probabilidades suman exactamente 1,0.

### Sparse Feature (Característica escasa)
Una característica cuyos valores son predominantemente cero o vacíos. Por ejemplo, una característica que contiene un único valor 1 y un millón de valores 0 es escasa. Por el contrario, una característica densa tiene valores que predominantemente no son cero o están vacíos.

En el aprendizaje automático, una sorprendente cantidad de funciones son 'sparse'. Las características categóricas suelen ser características escasas. Por ejemplo, de las 300 especies de árboles posibles en un bosque, un solo ejemplo podría identificar solo un arce. O, de los millones de vídeos posibles en una videoteca, un solo ejemplo podría identificar simplemente "Casablanca".

En un modelo, normalmente se representan características dispersas con codificación one-hot. Si la codificación one-hot es grande, puede colocar una capa de incrustación encima de la codificación one-hot para una mayor eficiencia.

### SGD, Stochastic gradient descent (Descenso de Gradiente estocástico)
Un algoritmo de descenso de gradiente en el que el tamaño del lote es uno. En otras palabras, SGD entrena con un único ejemplo elegido uniformemente al azar de un conjunto de entrenamiento.

### Synthetic feature ( Característica Sintética o artificial)
Característica que no está presente entre las características de entrada, pero que se ensambla a partir de una o más de ellas. Los métodos para crear características sintéticas incluyen los siguientes:

* Dividir una característica continua en contenedores de rango.
* Creando una cruz de características.
* Multiplicar (o dividir) el valor de una característica por otros valores de característica o por sí mismo. Por ejemplo, si a y b son características de entrada, los siguientes son ejemplos de características sintéticas:
    * a*b
    * a**2
* Aplicar una función trascendental a un valor de característica. Por ejemplo, si c es una característica de entrada, los siguientes son ejemplos de características sintéticas:
    * sin(c)
    * ln(c)

Las características creadas únicamente mediante la normalización o el escalado no se consideran características sintéticas.

### Training Loss (Pérdida en entrenamiento)
Una métrica que representa la pérdida de un modelo durante una iteración de entrenamiento particular. Por ejemplo, supongamos que la función de pérdida es el error cuadrático medio. Quizás la pérdida de entrenamiento (el error cuadrático medio) para la décima iteración sea 2,2 y la pérdida de entrenamiento para la centésima iteración sea 1,9.

Una curva de pérdida traza la pérdida de entrenamiento frente al número de iteraciones. Una curva de pérdidas proporciona las siguientes sugerencias sobre el entrenamiento:

* Una pendiente descendente implica que el modelo está mejorando.
* Una pendiente ascendente implica que el modelo está empeorando.
* Una pendiente plana implica que el modelo ha alcanzado la convergencia.

### Underfitting (Subentrenamiento o Falta de adaptación)
Producir un modelo con poca capacidad de predicción porque el modelo no ha capturado completamente la complejidad de los datos de entrenamiento. Muchos problemas pueden causar un ajuste insuficiente, entre ellos:

* Capacitación en el conjunto incorrecto de funciones.
* Entrenamiento para muy pocas épocas o a una tasa de aprendizaje demasiado baja.
* Entrenamiento con una tasa de regularización demasiado alta.
* Proporcionar muy pocas capas ocultas en una red neuronal profunda.

### Unlabeled examples (Ejemplos sin etiquetar)
Un ejemplo que contiene características pero no etiqueta, es decir que posee matriz de datos pero no tiene una salida 'y' o 'target'.

En el aprendizaje automático supervisado, los modelos se entrenan con ejemplos etiquetados y hacen predicciones sobre ejemplos sin etiquetar.

En el aprendizaje semisupervisado y no supervisado, se utilizan ejemplos sin etiquetar durante el entrenamiento.

### Unsupervised Machine Learning (Aprendizaje automático sin supervisión)
Entrenar un modelo para encontrar patrones en un conjunto de datos, generalmente un conjunto de datos sin etiquetar.

El uso más común del aprendizaje automático no supervisado es agrupar datos en grupos de ejemplos similares. Por ejemplo, un algoritmo de aprendizaje automático no supervisado puede agrupar canciones en función de diversas propiedades de la música. Los grupos resultantes pueden convertirse en una entrada para otros algoritmos de aprendizaje automático (por ejemplo, para un servicio de recomendación musical). La agrupación puede ayudar cuando las etiquetas útiles son escasas o no existen. Por ejemplo, en ámbitos como la lucha contra el abuso y el fraude, los clústeres pueden ayudar a los humanos a comprender mejor los datos.

### Z-score normalization (Normalización z-score)
Una técnica de escala que reemplaza un valor de característica sin procesar con un valor de punto flotante que representa el número de desviaciones estándar de la media de esa característica. Por ejemplo, considere una característica cuya media es 800 y cuya desviación estándar es 100.

Luego, el modelo de aprendizaje automático se entrena con las puntuaciones Z para esa característica en lugar de con los valores brutos.
