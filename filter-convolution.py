# Maria Regina Orduño Lopez - A01252959
# Mariana Islas Mondragon - A01253435
# Mariana Carrillo Holguin - A01253358
# Zuleyca Guadalupe Balles Soto - A01741687
# 20/03/2025
# Este programa aplica a una imagen una convolución con un kernel de aumento de ruido, 
# subsecuentemente a la imagen resultante con ruido se le aplica un filtro Sobel.

# Importar librerías necesarias:
# numpy para operaciones numéricas, cv2 para procesamiento de imágenes y matplotlib para graficar.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Definición de la función de convolución que aplica un filtro (kernel) a una imagen.
def convolution(image, filtro):
    # Obtener las dimensiones (altura y anchura) de la imagen de entrada.
    img_height, img_width = image.shape
    # Obtener las dimensiones (altura y anchura) del filtro.
    filtro_height, filtro_width = filtro.shape

    # Crear una matriz de salida (imagen resultante) con dimensiones reducidas:
    # La imagen resultante tendrá dimensiones: (img_height - filtro_height + 1) x (img_width - filtro_width + 1)
    output = np.zeros((img_height - filtro_height + 1, img_width - filtro_width + 1), dtype=np.float32)

    # Aplicar la convolución sin padding recorriendo la imagen:
    for i in range(img_height - filtro_height + 1):
        for j in range(img_width - filtro_width + 1):
            # Extraer una región de la imagen con el mismo tamaño que el filtro.
            region = image[i:i + filtro_height, j:j + filtro_width]
            # Multiplicar elemento a elemento la región y el filtro, y luego sumar todos los valores.
            output[i, j] = np.sum(region * filtro)

    # Limitar los valores del resultado al rango [0, 255] y convertir a uint8 para visualización.
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Retornar la imagen resultante de la convolución.
    return output

# Bloque principal del programa.
if __name__ == "__main__":
    # Definir la ruta de la imagen a procesar.
    image_path = "semana-tec-convolucion/dog.jpeg" 

    # Cargar la imagen en escala de grises usando OpenCV.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Si la imagen no se cargó correctamente, se muestra un mensaje de error y se finaliza el programa.
    if image is None:
        print("Error: No se pudo cargar la imagen")
        exit()

    # Convertir la imagen a tipo float32 para realizar operaciones de convolución sin pérdida de precisión.
    image = image.astype(np.float32)

    # Definir el filtro 1: Kernel para aumento de ruido.
    filtro1 = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    # Definir el filtro 2: Kernel Sobel en X para detección de bordes horizontales.
    filtro2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Definir el filtro 3: Kernel Laplaciano para realce de bordes.
    filtro3 = np.array([    
        [0, 1, 0],
        [1, -16, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Aplicar el filtro 1 (aumento de ruido) a la imagen original.
    resultado1 = convolution(image, filtro1)
    # Aplicar el filtro 2 (Sobel) al resultado obtenido del filtro 1.
    resultado2 = convolution(resultado1, filtro2)
    # Aplicar el filtro 3 (Laplaciano) al resultado obtenido del filtro 2.
    resultado3 = convolution(resultado2, filtro3)
    
    # Agregar padding circular al resultado final.
    # Se agrega un padding de 15 píxeles alrededor de la imagen, este valor es ajustable.
    pad = 15
    resultadopad = np.pad(resultado3, ((pad, pad), (pad, pad)), mode='wrap').astype(np.uint8)

    # Si el resultado del primer filtro no es None, continuar con la visualización.
    if resultado1 is not None:
        # Crear una figura con tamaño 15x5 pulgadas para mostrar las imágenes.
        # Se crean 5 subplots para mostrar cada etapa del proceso de convolución.
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 5, 1)  # Imagen Original.
        plt.imshow(image, cmap='gray')
        plt.title("Imagen Original")
        plt.axis("off")

        plt.subplot(1, 5, 2)  # Primer filtro (Aumento de Ruido).
        plt.imshow(resultado1.astype(np.uint8), cmap='gray')
        plt.title("Aumento de Ruido")
        plt.axis("off")

        plt.subplot(1, 5, 3)  # Segundo filtro (Filtro Sobel).
        plt.imshow(resultado2.astype(np.uint8), cmap='gray')
        plt.title("Filtro Sobel")
        plt.axis("off")

        plt.subplot(1, 5, 4)  # Tercer filtro (Filtro Laplaciano).
        plt.imshow(resultado3.astype(np.uint8), cmap='gray')
        plt.title("Filtro Laplaciano")
        plt.axis("off")

        plt.subplot(1, 5, 5)  # Resultado final con padding.
        plt.imshow(resultadopad, cmap='gray')
        plt.title("Con Padding Final")
        plt.axis("off")
        
        plt.show()