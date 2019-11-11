# Documentacion de como funciona 

Vamos a usar dos librerias externas muy importantes:

- dlib
- face_recognition

Obviamente usaremos tambien **OpenCV** como herramienta principal para facilitar el reconocimiento facial. Por sus herramientas.

## Como funciona el reconocimiento facial de OpenCV:

1. Primero tomamos una imagen de entrada
2. La libreria detecta el rostro
3. Transforma la silueta de la cara 
4. Recorta el contorno del rostro hasta tener algo procesable
5. Se pasa a la red neuronal
6. Y la representamos

## Vamos a aplicar DEEP LEARNING en dos pasos clave

1. Para aplicar la detección de rostros, que detecta la presencia y ubicación de un rostro en una imagen, pero no la identifica
2. Para extraer los vectores de características de 128-d (llamados "incrustaciones") que cuantifican cada cara en una imagen


## Los enlaces que me ayudaron

- [para ambiente virtual](https://github.com/pystudent1913/twetclone-django1.10/blob/master/documentacion/setup.md)
-[virtualenv doc](https://docs.python-guide.org/dev/virtualenvs/)
- [set environment](https://github.com/pystudent1913/learning-django/blob/master/00_Instalacion_and_basics/01Set_environment.md)
- [face recognition](https://github.com/ageitgey/face_recognition)
- [tutorial - omg](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
 







