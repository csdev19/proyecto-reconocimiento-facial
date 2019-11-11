# PASO 3: Reconociendo caras con OpenCV

Ya estamos listos para implementar el reconocimiento facial con OpenCV.

Empezaremos en el archivo **recognize.py** en nuestro proyecto:

```python
#import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

Hacemos nuestras respectivas importaciones en las primeras lineas.

Y luego hacemos el parseador de los parametros que pasamos por consola:

- **--image**: la ruta a la imagen de entrada. Intentaremos reconocer las caras en esta imagen.
- **--detector**: la ruta al detector de caras de aprendizaje profundo de OpenCV. Utilizaremos este modelo para detectar en qué parte de la imagen se encuentran los ROI de la cara.
- **--embedding**-model: el camino hacia el modelo de incrustación de caras de aprendizaje profundo de OpenCV. Utilizaremos este modelo para extraer la incrustación de la cara 128-D desde el ROI de la cara; alimentaremos los datos en el reconocedor.
- **--recognizer**: el camino hacia nuestro modelo reconocedor. Entrenamos a nuestro reconocedor SVM en el Paso 2. Esto es lo que realmente determinará quién es una cara.
- **--le**: la ruta a nuestro codificador de etiquetas. Contiene nuestras etiquetas faciales como 'adrian' o 'trisha'.
- **--confidence**: el umbral opcional para filtrar las detecciones faciales débiles.

Y luego cargamos nuestros 3 modelos desde disco a la memoria.

```python
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
 
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
 
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
```

Cargamos 3 modelos en este bloqueo de codigo. Vamos a aclarar las diferencias entre estos modelos:

- **detector**: Un modelo Caffe DL pre-entrenado para detectar en qué parte de la imagen están las caras.
- **embedder**: un modelo Torch DL pre-entrenado para calcular nuestras incrustaciones de cara 128-D 
- **reconocedor**: Nuestro modelo de reconocimiento facial lineal SVM . Entrenamos este modelo en el Paso 2.

Los dos primeros que ya han sido pre-entrenados. Lo que significa que OpenCV te los proporciona tal cual. 

Tambien cargamos nuestro codificador de etiquetas que contiene los nombres de las personas que nuestro modelo puede reconocer.

Ahora cargamos nuestra imagen y detectamos caras:

```python
# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
 
# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)
 
# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()
```

Aqui nosotros:

- Cargamos la imagen en la memoria y construimos un BLOB. Obtenemos informacion sobre cv2.dnn.blobFromImage aqui. 
- Localizamos caras en la imagen a traves de nuestro detector.

Dadas nuestras nuevas detecciones reconozcamos los rostros en la imagen. Pero primero tenemos que filtrar las detecciones debiles y extraer el ROI de lo cara:

```python
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
 
	# filter out weak detections
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]
 
		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue
```

Recorremos las detecciones y extraemos la confianza de cada. Luego comparamos la confianza con el umbral de deteccion de probabilidad minima contenido en nuestro diccionario de argumentos de linea de comando, asegurando que la probabilidad calculada sea mayor que la probabilidad minima.

A partir de ahi extraemos el ROI de la cara y aseguramos que sus dimensiones espaciales sean suficientemente grandes.

Reconocer el nombre del ROI de la cara requiere solo unos pocos pasos:

```python
		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()
 
		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]
```

Primero, construimos un **faceBlob** (desde el ROI de la cara) yy los pasamos a traves del integrador para generar un vector 128-D que describe la cara.

Luego, pasamos el vec a traves de nuestra modelo de reconocimiento SVM, cuyo resultado son nuestras predicciones sobre quien esta en la cara ROI.

Tomamos el indice de probabilidad mas alto y consultamos nuestro codificador de etiquetas para encontrar el nombre. En el medio, extraigo la probabilidad.

Nota: Puede filtarr aun mas los reconocimientos de cara debiles aplicando una prueba de umbral adicional sobre la probabilidad. Por ejemplo, insertar **if proba < T** (donde T es una variable que usted define) puede proporcionar una capa adicional de filtrado para garantizar que haya menos reconocimientos de rostros falsos positivos.

Ahora vamso a mostrar los resultados del reconocimiento facial de OpenCV:

----------------------------------------------------------------------------------------------------------

```python
		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
```

Para cada cara que reconocemos en el bucle (incluidas las desconocidas):

- Construimos una variable **text** de tipo string conteniendo el nombre y la probabilidad en la **Linea 93**
.
- Y luego dibujamos un rectagulo al rededor de la cara y ponemos un texto cerca a la caja.

Y luego vizualizamos los resultados hasta que una tecla sea presionada.

Y ahora vamos a ver los resultados:

```console
python recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--image images/cristian2.jpg
```

