# Esta documentacion sera para poder analizar los rostros mediante videos en vivo

Vamos a empezar a codear en el archivo **recognize_video.py**

```python
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
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

Primero importamos los paquetes necesarios y luego seteamos los parametros que enviaremos por consola.

Nuestras importaciones son las mismas que en la seccion anterior, a excepcion que ahora usamos **imutils.video**.

Utilizaremos **VideoSteam** para capturas fotogramas de nuestras cama y FPS para calcular estadisticas de fotogramas por segundo.

Los argumentos de la linea de comandos tambien son los mismos, excepto que no estamos pasando una ruta a una imagen estatica a traves de la linea de comando. En cambio, tomaremos una referencia a nuestra camara de video.

Luego se cargan los 3 modelos y el codificador de etiquetas.

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

Aqui cargamos el detector de rostros, el modelo de embebido de rostros, el modelo de reconocimiento de rostros (SVM lineal) y el codificador de etiquetas.

Inicialicemos nuestra transmicion de video y procesamos cuadros:

```python
# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
 
# start the FPS throughput estimator
fps = FPS().start()
 
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
 
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
 
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
 
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
```

Nuestro objeto **VideoStream** se inicializa y se inicia. Esperamos a que el sensor de la cama se caliente.

Tambien inicializamos nuestros cuadros por segundo contador y comenzamos a recorrer los cuadros.

```python
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
 
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
```

Al igual que en la seccion anterior, comenzamos a recorrer las detecciones y filtramos la debiles. Luego extraemos el ROI de la cara y aseguramos que las dimensiones espaciales sean lo suficientemente grandes para los siguientes pasos.


Ahora usaremos OpenCV:

```python
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
 
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
 
			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
	# update the FPS counter
	fps.update()
```

Aqui nosotros:

- Construimos el **faceBlob** y calculamos las incrustaciones faciales mediante aprendizaje profundo.
- Reconocemos el nombre mas probable de la cara mientras se calcula la probabilidad.
- Dibujamos un cuadro delimitador alrededor de la cara y el nombre de la persona + su probabilidad.
- Nuestro contador FPS se actualiza


Mostramos los resultados y limpiamos:

```python
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

Para cerrar el guion nosotros:

- Mostramos el cuadro anotado y esperamos a que se presione la tecla "Q" en cuyo punto saldremos del bucle.
- Detenemos nuestro contador FPS e imprime estadisticas en el terminal.
- Limpie cerrando ventanas y liberando punteros.

Para ejecutar nuestro codigo hacemos lo siguiente:

```console
python recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```
