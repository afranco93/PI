import cv2
import pytesseract
import numpy as np
import re
import mysql.connector
import webbrowser
import time

# Cargar la cascada
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Configurar la cámara web
cap = cv2.VideoCapture(1)

# Crear el objeto ORB
orb = cv2.ORB_create()

# Patrón de expresión regular para validar la matrícula
patron = re.compile(r'^\d{4} [A-Z]{3}$')

# Limpia y formatea la matrícula
def clean_matricula(matricula):
    matricula = re.sub('[^a-zA-Z0-9]', '', matricula)
    numeros = matricula[:4]
    letras = matricula[4:]
    letras = re.sub('[^bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', '', letras)
    letras = letras[:3].ljust(3, 'X')
    numeros = re.sub('[^0-9]', '', numeros)
    numeros = numeros[:4].rjust(4, '0')
    cleaned_matricula = f"{numeros} {letras}"
    
    if not patron.match(cleaned_matricula):
        cleaned_matricula = 'Matrícula inválida'

    return cleaned_matricula

# Realizar la consulta a la base de datos
def consultar_base_de_datos(matricula):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Asir2005",
            database="Empleados"
        )
        cursor = connection.cursor()
        query = "SELECT * FROM parking WHERE matricula = %s"
        cursor.execute(query, (matricula,))
        result = cursor.fetchone()
        
        if result:
            return True  # Matrícula encontrada en la base de datos
        else:
            return False  # Matrícula no encontrada en la base de datos
        
    except mysql.connector.Error as error:
        print("Error al conectar a la base de datos:", error)
        return False

is_page_loaded = False
last_page_load_time = time.time()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    max_matches = 0
    max_matches_matricula = ''

    plates = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in plates:
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=0.1*w, maxLineGap=10)

        if lines is not None and len(lines) >= 3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            kp, des = orb.detectAndCompute(roi, None)
            img_keypoints = cv2.drawKeypoints(roi, kp, None, color=(0, 255, 0), flags=0)
            cv2.imshow('Keypoints', img_keypoints)

            matricula = pytesseract.image_to_string(roi, lang='eng', config='--psm 7')
            if matricula:
                cleaned_matricula = clean_matricula(matricula)
                print("Matrícula detectada:", cleaned_matricula)
                
                # Consultar la base de datos
                matricula_en_bd = consultar_base_de_datos(cleaned_matricula)

                if is_page_loaded:
                    # Esperar durante 5 segundos después de cargar la página
                    if time.time() - last_page_load_time >= 5:
                        is_page_loaded = False
                else:
                    # Esperar durante 5 segundos antes de cargar una nueva página
                    if time.time() - last_page_load_time >= 5:
                        if matricula_en_bd:
                            print("Matrícula encontrada en la base de datos")
                            permitido_url = "http://localhost/permitido.php?matricula=" + cleaned_matricula
                            # Abrir la página permitido.php en el navegador
                            webbrowser.open(permitido_url)
                        else:
                            print("Matrícula no encontrada en la base de datos")
                            denegado_url = "http://localhost/denegado.php?matricula=" + cleaned_matricula
                            # Abrir la página denegado.php en el navegador
                            webbrowser.open(denegado_url)
                        is_page_loaded = True
                        last_page_load_time = time.time()

    cv2.imshow('Detector de matrículas', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
