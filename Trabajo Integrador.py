import cv2
import argparse
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Yolo live')
    parser.add_argument(
        "--webcam-resolution",
        default = [1280, 720],
        nargs = 2,
        type = int
    )
    args = parser.parse_args()
    return args 

args = parse_arguments()
frame_width, frame_heigth = args.webcam_resolution

# Captura el video y su resolucion
cap = cv2.VideoCapture(2)                           # Se puede sustituir el dispositivo de entrada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_heigth)    # Alto
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)      # Ancho

# Exporta el modelo
model = YOLO('yolov8n.pt')

# Crea 'yolov8n_openvino_model/'
model.export(format="openvino")  

# Carga el modelo de OpenVINO
ov_model = YOLO("yolov8n_openvino_model/")

vehicle_classes = [2, 3, 5, 7] # 2 = car, 3 = motorcycle, 5 = bus, 7 = truck

while True:
        ret,frame = cap.read()

        results = ov_model(frame)

        for result in results:
            for box in result.boxes:
                if box.cls in vehicle_classes and box.conf > 0.30:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extrae las coordenadas de la bounding box
                    conf = box.conf[0]  # Extrae la confianza de la predicción
                    cls = box.cls[0]  # Extrae la clase de la predicción
                    
                    label = f'{ov_model.names[int(cls)]} {conf:.2f}'
                    
                    # Dibuja la bounding box en el frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('TRABAJO INTEGRADOR',frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
