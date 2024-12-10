from flask import Flask, request, render_template_string, make_response # Servicio web
import os # Biblioteca para manipulación de archivos y directorios
import base64 # Biblioteca para convierte datos binarios en cadenas ASCII
import pytesseract #Biblioteca para el reconocimiento de texto en imágenes (visión por computadora)
import cv2 #Blblioteca OpenCV, para procesamiento de imágenes (visión por computadora)
from PIL import Image # Blblioteca para manipulación de imágenes y procesamiento básico
import re # Blblioteca para trabajar con expresiones regulares (bùsqueda)
import numpy as np # Biblioteca para el procesamiento de datos

# Configuración de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Carpeta donde se guardarán las fotos
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegurarse de que la carpeta de subida exista
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    html_content = open('upload.html').read()
    response = make_response(render_template_string(html_content))
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/upload', methods=['POST'])
def upload():
    photo_data = request.form.get('photo_data')

    if photo_data:
        # Eliminar el encabezado de la imagen en base64 (data:image/jpeg;base64,)
        photo_data = photo_data.split(',')[1]
        img_data = base64.b64decode(photo_data)
        
        # Guardar la imagen como un archivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg')
        with open(file_path, 'wb') as f:
            f.write(img_data)

        # Procesar la imagen
        extracted_text = extract_text(file_path)
        even_check = is_last_number_even(extracted_text)
        even_check, last_digit = is_last_number_even(extracted_text)
       

        # Preparar los resultados
        placa = extracted_text
        tipo_num = "par" if even_check else "impar" if even_check is not None else "no encontrado"
        Ultimo = last_digit     

        # Pasar los resultados a la plantilla result.html
        return render_template_string(open('result.html').read(), placa=placa, tipo_num=tipo_num, Ultimo=Ultimo)

    return 'No se ha recibido la foto', 400

#Prueba de función
def detectar_color_placa(imagen):
    """
    Detecta si la placa es amarilla (vehículo particular).
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir rangos de color amarillo
    amarillo_bajo = np.array([20, 100, 100])
    amarillo_alto = np.array([30, 255, 255])

    # Crear máscara
    mascara_amarilla = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)

    # Calcular porcentaje de píxeles amarillos
    total_pixeles = imagen.shape[0] * imagen.shape[1]
    pixeles_amarillos = cv2.countNonZero(mascara_amarilla)
    porcentaje_amarillo = (pixeles_amarillos / total_pixeles) * 100

    return porcentaje_amarillo > 30  # Si más del 30% es amarillo

def extract_text(image_path):
    img = cv2.imread(image_path)  # Cargar la imagen
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Aumentar la resolución
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises

    custom_config = r'--oem 3 --psm 11'  # Configuración de Tesseract 
    text = pytesseract.image_to_string(gray, config=custom_config)  # Usar la imagen mejorada
    return text.strip()  # Limpiar y devolver el texto


def is_last_number_even(text):
    # Buscar todos los números en el texto
    numbers = re.findall(r'\d+', text)  # Encuentra todos los números en el texto
    if numbers:
        last_number = int(numbers[-1])  # Obtiene el último número
        last_digit = last_number % 10  # Obtiene el último dígito del número
        return last_number % 2 == 0, last_digit # Retorna True si es par, False si es impar
    return None, None  # Retorna None si no hay números



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8080, ssl_context=('server.cer', 'server.key'))







