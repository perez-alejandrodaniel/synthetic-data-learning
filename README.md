# synthetic-data-learning

Mi experiencia de aprendizaje explorando el arte de generar datos sintéticos realistas.

Esto no es una librería ni un producto: es más bien un cuaderno de laboratorio. Tomo un dataset bancario de un dataset abierto, pruebo varias formas de generar datos "parecidos" y después reviso si los datos inventados sirven para entrenar modelos útiles sin copiar los originales.

## ¿Qué hay por aquí?

- `datasources/bank_full_dataset/`: el dataset original (`bank-full.csv`, 45.211 filas y 17 columnas) y un notebook para entenderlo antes de tocar nada: distribuciones, desbalance de clases, correlaciones, valores `unknown`, etc.
- `notebooks/copula/`, `notebooks/ctgan/` y `notebooks/tvae/`: experimentos con SDV (Gaussian Copula, CTGAN y TVAE). Cada carpeta tiene un notebook para generar datos, otro para validarlos, el CSV sintético y el modelo guardado en `.pkl` (formato pickle).
- `notebooks/be_great/`: experimentos con modelos de lenguaje pequeños mediante GReaT (DistilGPT-2 y Qwen3-0.3B). También hay generación, validación y modelos/checkpoints locales. Puede que hayan quedado algo viejos los modelos para el momento que estes leyendo esto.
- `requirements.txt`: dependencias con versiones fijadas, probadas con Python 3.12.

## El dataset

`bank-full.csv` viene del conocido dataset *Bank Marketing* de UCI. Puedes ver la [página del dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) o ir directo a la [descarga del zip](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip); dentro viene `bank-full.csv` junto con `bank.csv`, `bank-additional.csv` y `bank-additional-full.csv`.

Cada fila es un cliente y la columna `y` dice si contrató (`yes`) o no (`no`) un depósito a plazo. Es clasificación binaria con un desbalance bastante fuerte: aproximadamente 88 % de `no` y 12 % de `yes`.

El notebook de análisis ya deja claro que el dataset tiene algunas particularidades incómodas: columnas como `balance` y `previous` tienen colas muy largas, y varias categóricas usan `unknown` como valor desconocido. Justo ahí empieza lo interesante: ver qué tan bien aguanta cada método esas rarezas.

En este link hay un analisis muy interesante sobre este mismo dataset. Un experimento pendiente es reproducir el
mismo analisis usando datos sintenticos para revisar si puedo llegar a las mismas conclusiones.
El trabajo es este https://rpubs.com/isabellaabril23/907846, Analysis Bank Marketing Data Set
de Isabella Hernandez

## Cómo está montado el experimento

La rutina se repite para cada sintetizador:

1. Entrenar con los datos reales.
2. Generar un CSV sintético (45.211 filas en los métodos SDV; 1.000 en los experimentos con GReaT, que son más bien una prueba de concepto).
3. Validar en otro notebook: comparar distribuciones y correlaciones, pedir un score de calidad con `sdmetrics`, entrenar un Random Forest con datos sintéticos y probarlo contra datos reales (TSTR), y revisar privacidad para detectar copias casi idénticas.

Los modelos SDV quedan guardados como `.pkl`, así que puedes generar más datos sin volver a entrenar. En la parte de GReaT los modelos y checkpoints son archivos grandes y están ignorados por Git: se descargan o se entrenan desde los propios notebooks.

## Resultados que salieron (resumen rápido)

- **TVAE** fue el que mejor score global dio y el que mejor conservó la utilidad para machine learning (alrededor del 94 % de retención TSTR).
- **CTGAN** quedó muy cerca (también ~94 % de retención) y conservó bien las correlaciones entre columnas.
- **Gaussian Copula** es el más simple y rápido, y copia casi perfectamente las proporciones categóricas, pero le cuesta más con la clase minoritaria.
- **GReaT + DistilGPT-2** fue el más débil: de vez en cuando genera valores mal formados y casi no conserva señal predictiva.
- **GReaT + Qwen3-0.3B** mejoró bastante respecto a DistilGPT-2, aunque la validación de utilidad se quedó a medias por un valor inesperado (`admin.` en vez de `admin`).

En todos los casos el chequeo de privacidad dijo lo mismo: no aparecieron filas sintéticas casi idénticas a las reales.

## Cómo arrancar

Con Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Después abre los notebooks con Jupyter. Un orden razonable sería: primero `datasources/bank_full_dataset/analisis_dataset_bank.ipynb`, luego el notebook de generación de la carpeta que te interese y, al final, su `check_synthetic_data.ipynb`. Los notebooks ya intentan instalar requirements al arrancar, así que tampoco pasa nada si el entorno está medio preparado.

## Cosas a tener en cuenta

- Los modelos de lenguaje pesan varios GB y no versionan bien. Si clonas el repo, la primera ejecución implica descarga y entrenamiento.
- Algunas salidas guardadas dentro de los notebooks pueden estar desactualizadas frente al código. Si un número no cuadra, vuelve a correr las celdas.
- Esto es un experimento de aprendizaje, no una receta: los resultados sirven para comparar enfoques, no para proclamar un ganador definitivo.
