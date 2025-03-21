# Maria Regina Orduño Lopez - A01252959
# Mariana Islas Mondragon - A01253435
# Mariana Carrillo Holguin - A01253358
# Zuleyca Guadalupe Balles Soto - A01741687
# 20/03/2025
# Este programa aplica una convolucion con un kernel de Sobel.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image_path, filtro):

    # Obtener dimensiones
    img_height, img_width = image.shape
    filtro_height, filtro_width = filtro.shape

    # Dimensiones de la imagen resultante (más pequeña)
    output = np.zeros((img_height - filtro_height + 1, img_width - filtro_width + 1), dtype=np.float32)

    # Aplicar convolución
    for i in range(img_height - filtro_height + 1):
        for j in range(img_width - filtro_width + 1):
            region = image[i:i + filtro_height, j:j + filtro_width]
            output[i, j] = np.sum(region * filtro)

    output = np.clip(output, 0, 255).astype(np.uint8)

    
    return output # Devuelve la imagen original y la filtrada

# Ejemplo de uso
if __name__ == "__main__":
    image_path = "dog.jpeg" 

    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen")
        exit()

    # Filtro Sobel en X (detección de bordes)
    filtro1 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    filtro2 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    filtro3 = np.array([    
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Aplicar convolución
    resultado1 = convolution(image, filtro1)
    resultado2 = convolution(resultado1, filtro2)
    resultado3 = convolution(resultado2, filtro3)

    if resultado1 is not None:
        # Mostrar las imágenes con Matplotlib
        plt.figure(figsize=(10, 5))

        # Imagen original
        plt.subplot(1, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Imagen Original")
        plt.axis("off")

        # Filtro 1
        plt.subplot(1, 4, 2)
        plt.imshow(resultado1, cmap='gray')
        plt.title("Imagen con Convolución ")
        plt.axis("off")

        # Filtro 2
        plt.subplot(1, 4, 3)
        plt.imshow(resultado2, cmap='gray')

        # Filtro 3
        plt.subplot(1, 4, 4)
        plt.imshow(resultado3, cmap='gray')
        
        # Mostrar la figura
        plt.show()