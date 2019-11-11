# PASO 2: Entrar el modelo de reconocimiento facial

En este punto, hemos extraido incrustaciones por cada cara. Pero, como reconoceremos realmente a una persona basada en estas incrustaciones?

La respuesta es que necesitamos entrenar un modelo de aprendizaje automatico "estandar" (como un SVM, un clasificador k-NN Random Forest, etc.) sobre las incrustaciones.

Vamos a construir un clasificador mas potente sobre las incrustaciones.

Vamos a abrir el archivo **train_model.py** e insertamos el siguiente codigo:

```python
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())
```

Necesitaremos **scikit-learn**, una biblioteca de machine learning.

Se instala de la siguiente manera:

```console
pip install scikit-learn
```

Importamos nuestros paquetes y modulos en las primera lineas.

Y luego usaremos la implementacion de **scikit-learn** de Support Vecotr Machines (SVM), un modelo comun de aprendizaje automentico.


Parseamos los argumentos de la linea de comandos:

- **--embeddings**: la ruta a las incrustaciones serializadas (lo exportamos ejecutando el script extract_embeddings.py anterior).
- **--reconocidor**: Este será nuestro modelo de salida que reconoce caras. Está basado en SVM. Lo guardaremos* para poder usarlo en los siguientes dos scripts de reconocimiento.
- **--le**: nuestra ruta de archivo de salida del codificador de etiquetas. Vamos a serializar nuestro codificador de etiquetas en el disco para poder usarlo y el modelo de reconocimiento en nuestras secuencias de comandos de reconocimiento de imagen / video.

Cada uno de estos argumentos es requeridos.

```python
# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
 
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
```

Aqui cargamos las incrustaciones que obtuvimos en el primer paso. No genereraremos incrustaciones en este script de capacitacion modelo; utilizaremos las incrustaciones generadas y serializadas previamente.

Luego inicializaremos nuestro **LabelEncoder** de **scikit-learn** y codificamos neustras etiquetas de nombre.

Ahora es el momento de entrenar nuestro modelo **SVM** para reconocer caras:

```python
# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
```

Inicializamos nuestro **modelo SVM** y ademas ajustamos el modelo (tambien conocido como "modelo de entrenamiento").

Aqui es donde usamos el **SVM** pero pueden intentar experimentar con otro modelo de aprendizaje automantico.

Despues de entrenar el modelo nosotros guardamos en el disco como archivo **pickle** el output del modelo y su etiqueta(label).

```python
# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
```

Creamos dos archivos pickle en el disco en esta parte del codigo (el modelo reconocedor facial y la etiqueta encodificada). 

Ahora que hemos acabado de programarlo vamos a correr el script **train_model.py**.

```console
python train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```

![segundo script](/utilities/segundo-script.png)

Y devuelve lo siguiente:

```console
ls output/
```

![output 2do](/utilities/output-2.png)

Y podemos ver que nuestro modelo **SVM** ha sido entreado en las incrustaciones y tanto el (1) SVM en si como (2) la codificacion de la etiqueta se han escrito en el disco, lo que nos permite aplicarlos a las imagenes de entrada y al de video.









