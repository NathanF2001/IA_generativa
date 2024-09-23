from PIL import Image
from ImagesLoader import get_paths_original_data
import os

def cortar_imagem_em_matriz(imagem_caminho, linhas, colunas, j_largura, j_altura ):
    imagem = Image.open(imagem_caminho)
    largura, altura = imagem.size
    largura = largura - j_largura
    altura = altura - j_altura

    bloco_largura = largura // colunas
    bloco_altura = altura // linhas

    blocos = []

    for j in range(colunas):
        for i in range(linhas):
            caixa = (j * bloco_largura + j_largura/2, i * bloco_altura + j_altura/2,
                     (j + 1) * bloco_largura + j_largura/2, (i + 1) * bloco_altura + j_altura/2)
            bloco = imagem.crop(caixa)
            blocos.append(bloco)

    return blocos

paths_original_data = get_paths_original_data()

result_path = os.path.join(os.getcwd(), "images", "unique_eggs")

if (not os.path.isdir(os.path.join(os.getcwd(), 'images'))):
    os.makedirs(os.path.join(os.getcwd(), 'images'))
if (not os.path.isdir(os.path.join(os.getcwd(), 'images' , 'unique_eggs'))):
    os.makedirs(os.path.join(os.getcwd(), 'images' , 'unique_eggs'))

index_image = 1

for p, imagem_caminho in enumerate(paths_original_data):
    if p % 2 == 0:
        blocos = cortar_imagem_em_matriz(imagem_caminho, 3, 5, 400, 300)
    else: 
        blocos = cortar_imagem_em_matriz(imagem_caminho, 3, 5, 250, 200)

    if p%14 == 13:
        blocos = [blocos[0], blocos[1], blocos[2], blocos[3], blocos[4]]

    for idx, bloco in enumerate(blocos):

        bloco.save(os.path.join(result_path, f"ovo_{index_image}.jpg"))

        index_image += 1