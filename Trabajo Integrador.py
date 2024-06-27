import cv2
import argparse
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

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
cap = cv2.VideoCapture(0)                           # Se puede sustituir el dispositivo de entrada
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_heigth)    # Alto
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)      # Ancho

# Exporta el modelo
model = YOLO('best.onnx')
m = ONNXPlateRecognizer('argentinian-plates-cnn-model')

while True:
        ret,frame = cap.read()

        results = model(frame)

        for result in results:
            for box in result.boxes:
                if box.conf > 0.60:  # Solo extrae las matriculas detectadas con una confianza superior al 30%
                    x1, y1, x2, y2 = box.xyxy[0].tolist()           # Extrae las coordenadas de la bounding box
                    conf = box.conf[0]                              # Extrae la confianza de la predicción
                    cls = box.cls[0]                                # Extrae la clase de la predicción
                    
                    # Recorta la imagen usando las coordenadas de la bounding box
                    cropped_frame = frame[int(y1):int(y2),int(x1):int(x2)]
                    gray_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

                    # Realiza OCR en el recorte de la matrícula
                    plate_text = m.run(gray_cropped_frame)

                    # Muestra el frame recortado en una nueva ventana
                    cv2.imshow('Recorte de Matricula', cropped_frame)

                    label = f'{plate_text} {conf:.2f}'
                    
                    # Dibuja la bounding box en el frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('TRABAJO INTEGRADOR',frame)  # Titulo de la ventana

        if cv2.waitKey(1) & 0xFF == ord("q"):   # Precionar q para salir
            break
