# Estructura del proyecto y dataset

## Primero necesitamos un DATASET

En este caso usaremos un dataset de 3 tipos de fotos. El primer dataset sera de mis propias fotos (la cantidad todavia no la tengo estimada con exactitud pero digamos que 4 son suficientes), el segundo de una persona X en especifico. Y el tercero de personas desconocidas dispersas llamadas como **Unknown**.

## La estructura del proyecto es la siguiente:

![estructura del proyecto](/utilities/estructura_proyecto.png)

- **dataset/**: Contiene nuestras imágenes faciales organizadas en subcarpetas por nombre.

- **images/**: Contiene tres imágenes de prueba que utilizaremos para verificar el funcionamiento de nuestro modelo.

- **face_detection_model/**: Contiene un modelo de aprendizaje profundo Caffe previamente entrenado proporcionado por OpenCV para detectar caras. Este modelo detecta y localiza caras en una imagen.

- **output/**: Contiene mis archivos de salida encurtidos. Si está trabajando con su propio conjunto de datos, también puede almacenar sus archivos de salida aquí. Los archivos de salida incluyen:

    - **embeddings.pickle**: A serialized facial embeddings file. Embeddings have been computed for every face in the dataset and are stored in this file.
    - **le.pickle**: Our label encoder. Contains the name labels for the people that our model can recognize.
    - **recognizer.pickle** : Our Linear Support Vector Machine (SVM) model. This is a machine learning model rather than a deep learning model and it is responsible for actually recognizing faces.

- **extract_embeddings.py**: Revisaremos este archivo en el Paso 1, que es responsable del uso de un extractor de funciones de aprendizaje profundo para generar un vector 128-D que describa una cara. Todas las caras de nuestro conjunto de datos se pasarán a través de la red neuronal para generar.

- **openface_nn4.small2.v1.t7**: Antorcha modelo de aprendizaje profundo que produce las incrustaciones faciales 128-D. Usaremos este modelo de aprendizaje profundo en los Pasos 1, 2 y 3, así como en la sección de Bonificación.

- **train_model.py**: Nuestro modelo Linear SVM será entrenado por este script en el Paso # 2. Detectaremos caras, extraeremos incrustaciones y ajustaremos nuestro modelo SVM a los datos de incrustaciones.

- **recognize.py**: En el paso 3 y reconoceremos caras en imágenes. Detectaremos rostros, extraeremos incrustaciones y consultaremos nuestro modelo SVM para determinar quién está en una imagen. Dibujaremos cuadros alrededor de las caras y anotaremos cada cuadro con un nombre.

- **recognize_video.py**: Describe cómo reconocer quién está en cuadros de una transmisión de video tal como lo hicimos en el Paso # 3 en imágenes estáticas.


