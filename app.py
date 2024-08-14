import easyocr
import cv2
import numpy as np

# Créer un lecteur easyocr pour la langue française
reader = easyocr.Reader(['fr'])

# Chemin vers votre image JPEG
image_path = 'img3.jpg'

# Lire l'image avec OpenCV
image = cv2.imread(image_path)

# Prétraiter l'image
# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enhanced preprocessing
gray = cv2.fastNlMeansDenoising(gray, h=10)  # Non-local means denoising
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Binariser l'image
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Sauvegarder l'image prétraitée (optionnel, pour vérification)
preprocessed_image_path = 'preprocessed_image.jpeg'
cv2.imwrite(preprocessed_image_path, binary_image)

# Utiliser easyocr sur l'image prétraitée
result = reader.readtext(preprocessed_image_path, detail=0, allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789éèàç')

# Post-processing (example)
corrected_text = []
for text in result:
    # Apply character and word correction logic here
    corrected_text.append(text)  # Corrected text should be appended

# Chemin vers le fichier de sortie
output_file_path = 'texte_extrait.txt'

# Écrire le texte extrait dans un fichier
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("Texte extrait :\n")
    for text in corrected_text:
         file.write(text + "\n")

print(f"Texte extrait écrit dans le fichier {output_file_path}")
