import cv2
import os
import argparse
import numpy as np
from skimage.feature import hog
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Yolo live')
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args 

args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

# Captura el video y su resolucion
cap = cv2.VideoCapture(0)                           # Se puede sustituir el dispositivo de entrada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)    # Alto
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)      # Ancho

# Exporta el modelo
model = YOLO('Model/best.onnx')
m = ONNXPlateRecognizer('argentinian-plates-cnn-model')

# Carpeta donde se guardarán las imágenes detectadas
output_folder = 'Imagenes Detectadas'
os.makedirs(output_folder, exist_ok=True)

def save_detected_image(image, text):
    filename = os.path.join(output_folder, f"{text}.jpg")
    cv2.imwrite(filename, image)

while True:
    ret, frame = cap.read()

    results = model.predict(frame)

    for result in results:
        for box in result.boxes:
            if box.conf > 0.40:  # Solo extrae las matriculas detectadas con una confianza superior al 30%
                x1, y1, x2, y2 = box.xyxy[0].tolist()           # Extrae las coordenadas de la bounding box
                conf = box.conf[0]                              # Extrae la confianza de la predicción
                cls = box.cls[0]                                # Extrae la clase de la predicción

                # Recorta la imagen usando las coordenadas de la bounding box
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                gray_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

                # Realiza OCR en el recorte de la matrícula
                plate_text = m.run(gray_cropped_frame)

                # Guarda la imagen recortada con el texto detectado como nombre
                save_detected_image(cropped_frame, plate_text)

                # Muestra el frame recortado en una nueva ventana
                cv2.imshow('Recorte de Matricula', cropped_frame)

                label = f'{plate_text} {conf:.2f}'

                # Dibuja la bounding box en el frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('TRABAJO INTEGRADOR', frame)  # Titulo de la ventana

    if cv2.waitKey(1) & 0xFF == ord("q"):   # Presionar q para salir
        break

cap.release()
cv2.destroyAllWindows()

# Función para preprocesar y extraer características HOG de una imagen
def extract_hog_features(image):
    image_resized = cv2.resize(image, (128, 64))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    return features

# Función para cargar imágenes desde una carpeta
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Ruta de la carpeta de imágenes
input_folder = output_folder
images, filenames = load_images_from_folder(input_folder)

# Extraer características de las imágenes
features_list = [extract_hog_features(image) for image in images]
from sklearn.cluster import KMeans

# Convertir la lista de características en un array numpy
features_array = np.array(features_list)

# Definir el número de clústeres
num_clusters = 2

# Aplicar K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features_array)

# Obtener las etiquetas de clúster para cada imagen
cluster_labels = kmeans.labels_
# Función para crear carpetas de salida si no existen
def create_output_folders(base_folder, num_clusters):
    for i in range(num_clusters):
        cluster_folder = os.path.join(base_folder, f'cluster_{i}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

# Ruta de la carpeta de salida
output_folder = 'Imagenes Clusterizadas'
create_output_folders(output_folder, num_clusters)

# Guardar imágenes en las carpetas correspondientes a sus clústeres
for img, label, filename in zip(images, cluster_labels, filenames):
    cluster_folder = os.path.join(output_folder, f'cluster_{label}')
    output_path = os.path.join(cluster_folder, filename)
    cv2.imwrite(output_path, img)
