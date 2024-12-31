# DIO - Algoritmos de Processamento de Imagens Digitais para Detecção de Bordas em Objetos 2D

## Filtro de Bordas com OpenCV

### Filtro Sobel - Exemplo de Código usando OpenCV

O filtro de Sobel é amplamente usado para a detecção de bordas em imagens ao calcular as derivadas em relação às direções X e Y. A seguir está um exemplo de implementação do filtro de Sobel usando a biblioteca OpenCV em Python:

### Código

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# Aplicar o filtro de Sobel na direção X e Y
sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente na direção X
sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente na direção Y

# Combinar os gradientes para obter a magnitude da borda
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Normalizar para escala de 0 a 255
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# Exibir as imagens
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Imagem Original")
plt.imshow(imagem, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobel - Gradiente X")
plt.imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel - Gradiente Y")
plt.imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.title("Sobel - Magnitude das Bordas")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.show()
```

### Explicação

1. **Carregamento da Imagem**: A imagem é carregada em escala de cinza para simplificar o processamento.

2. **Filtro Sobel**:

   - `cv2.Sobel`: Calcula o gradiente da imagem em uma direção específica (X ou Y).

   - Os parâmetros `1, 0` indicam o cálculo na direção X, e `0, 1` na direção Y.

3. **Combinação de Gradientes**:

   - Usa `cv2.magnitude` para calcular a magnitude combinada dos gradientes em X e Y, representando as bordas finais.

4. **Normalização**:

   - As imagens processadas são normalizadas para valores entre 0 e 255 para facilitar a visualização.

5. **Visualização**:

   - As imagens intermediárias (gradientes X e Y) e a magnitude combinada são exibidas lado a lado para comparação.

### Dica

Para melhores resultados, você pode aplicar um filtro GaussianBlur antes do filtro Sobel para suavizar a imagem e reduzir ruídos. 

Adicione esta linha antes de aplicar o Sobel:

```python
imagem = cv2.GaussianBlur(imagem, (5, 5), 0)
```

## Programando um Filtro de Segmentação no OpenCV

### Segmentação por cores - Exemplo de Código usando OpenCV

A segmentação por cores é um método comum para isolar objetos ou regiões específicas em uma imagem com base em seus valores de cor. Abaixo, você encontrará um exemplo de código que utiliza a biblioteca OpenCV para segmentação por cores:

### Código

```python
import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('imagem.jpg')

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# Converter a imagem para o espaço de cores HSV
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Definir o intervalo de cor para segmentação (exemplo: cor vermelha)
cor_min = np.array([0, 120, 70])  # Limite inferior (H, S, V)
cor_max = np.array([10, 255, 255])  # Limite superior (H, S, V)

# Criar uma máscara baseada no intervalo de cor
mascara = cv2.inRange(imagem_hsv, cor_min, cor_max)

# Aplicar a máscara na imagem original
imagem_segmentada = cv2.bitwise_and(imagem, imagem, mask=mascara)

# Exibir as imagens
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Máscara", mascara)
cv2.imshow("Imagem Segmentada", imagem_segmentada)

# Aguardar tecla para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Explicação

1. **Carregamento da Imagem**:

   - A imagem é carregada no formato BGR (padrão do OpenCV).

2. **Conversão para o Espaço de Cores HSV**:

   - O espaço HSV (Hue, Saturation, Value) facilita a segmentação de cores, já que o matiz (Hue) é independente de brilho e saturação.

3. **Definição de Intervalo de Cor**:

   - Os limites `cor_min` e `cor_max` definem a faixa de valores de cor a ser segmentada no espaço HSV.

   - O exemplo dado segmenta tons de vermelho. Para outras cores, você pode ajustar os valores de matiz, saturação e valor.

4. **Criação da Máscara**:

   - A função `cv2.inRange` gera uma máscara binária onde pixels dentro do intervalo de cor recebem o valor 255 (branco), e os demais 0 (preto).

5. **Aplicação da Máscara**:

   - A função `cv2.bitwise_and` aplica a máscara à imagem original, preservando apenas os pixels correspondentes à cor segmentada.

6. **Exibição**:

   - As janelas mostram a imagem original, a máscara e a imagem segmentada.

### Personalização para Outras Cores

Para ajustar a segmentação a outras cores:

- **Verde**: `cor_min = np.array([36, 100, 100])`, `cor_max = np.array([86, 255, 255])`

- **Azul**: `cor_min = np.array([94, 80, 2])`, `cor_max = np.array([126, 255, 255])`

### Dica

Se a segmentação não funcionar bem devido a iluminação ou ruído, você pode:

1. **Suavizar a Imagem**: Use `cv2.GaussianBlur` ou `cv2.medianBlur` antes de converter para HSV.

2. **Combinar Intervalos de Cor**:

   - Alguns tons, como o vermelho, exigem dois intervalos (ex.: `[0-10]` e `[170-180]` em Hue).

## Segmentação Semântica com Deep Learning

### Rede DeepLab - Exemplo de Código

A segmentação semântica com redes de Deep Learning como a DeepLab é uma abordagem avançada para classificar cada pixel de uma imagem em diferentes categorias. 

O modelo DeepLab é amplamente utilizado devido à sua capacidade de detectar bordas precisas e segmentar objetos em imagens. 

Aqui está um exemplo de como usar a rede DeepLab pré-treinada com a biblioteca TensorFlow:

### Código

```python
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Baixar o modelo DeepLab pré-treinado
model = tf.keras.applications.DeepLabV3Plus(
    weights="pascal_voc",  # Dataset pré-treinado
    input_shape=(513, 513, 3),
    include_top=False
)

# Carregar e preprocessar a imagem
def preprocess_image(image_path):
    imagem = cv2.imread(image_path)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    imagem_resized = cv2.resize(imagem, (513, 513))
    imagem_preprocessed = tf.keras.applications.resnet50.preprocess_input(imagem_resized)
    return imagem_preprocessed, imagem_resized

# Função para aplicar a segmentação
def segment_image(model, image):
    image = np.expand_dims(image, axis=0)  # Adicionar batch dimension
    pred = model.predict(image)
    segmentation_map = np.argmax(pred[0], axis=-1)  # Mapear a classe mais provável
    return segmentation_map

# Caminho da imagem
image_path = "imagem.jpg"

# Pré-processar e segmentar
input_image, original_image = preprocess_image(image_path)
segmentation = segment_image(model, input_image)

# Visualizar os resultados
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmentação Semântica")
plt.imshow(segmentation, cmap="jet")
plt.axis("off")

plt.show()
```

### Explicação do Código

1. **Modelo DeepLabV3+**:

   - Utiliza a arquitetura DeepLabV3+, um modelo pré-treinado no dataset PASCAL VOC. Ele é eficiente para segmentação semântica.

2. **Pré-processamento da Imagem**:

   - A imagem é redimensionada para 513x513 pixels (requisito do modelo).

   - O `preprocess_input` é aplicado para ajustar os valores da imagem conforme esperado pelo modelo.

3. **Segmentação**:

   - O modelo retorna um tensor de dimensões `(altura, largura, classes)` para cada pixel.

   - A classe mais provável para cada pixel é selecionada usando `np.argmax`.

4. **Visualização**:

   - A segmentação é exibida como um mapa de cores (`cmap="jet"`), sobrepondo diferentes categorias na imagem.

### Dependências

Certifique-se de instalar as bibliotecas necessárias:

```bash
pip install tensorflow opencv-python matplotlib
```

### Dicas

1. **Outros Modelos**:
   - Você pode usar modelos diferentes para tarefas específicas, como `ADE20K` ou `Cityscapes`, dependendo do domínio da aplicação.

2. **Sobreposição na Imagem Original**:
   - Para sobrepor a segmentação diretamente na imagem, use uma combinação como:
     ```python
     overlay = cv2.addWeighted(original_image, 0.6, segmentation_colored, 0.4, 0)
     ```
