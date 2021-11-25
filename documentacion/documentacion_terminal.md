# Documentacion de los comandos que usaremos en la terminal


## 1ro Este script es para calcular las coincidencias faciales con respecto a las imagenes que tenemos

```console
python3 extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7
```

## 2do para entrenas el modelo

```console
python3 train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```

## 3ro para comparar una imagen con los modelos ya creados

```console
python3 recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--image images/cristian2.jpeg
```

## 4to para ver la comparacion con la camara

```console
python3 recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```
