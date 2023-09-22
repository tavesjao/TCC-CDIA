# BEGIN: 3f5a6d8b5c7a
import os

def id_images(pasta='../data/frames/images'):
    contador = 1  

    arquivo_controle = os.path.join(pasta, 'contador.txt')
    if os.path.exists(arquivo_controle):
        with open(arquivo_controle, 'r') as arquivo:
            contador = int(arquivo.read())

    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith('.jpg'):
            partes_nome = nome_arquivo.split('_')
            if len(partes_nome) == 2 and partes_nome[0].isdigit():
                # O arquivo já começa com um número e tem um '_'
                continue  # Pula para o próximo arquivo

            novo_nome = f'{contador}_original.jpg'
            caminho_antigo = os.path.join(pasta, nome_arquivo)
            caminho_novo = os.path.join(pasta, novo_nome)
            os.rename(caminho_antigo, caminho_novo)
            contador += 1

    # Salva o novo valor do contador no arquivo contador.txt
    with open(arquivo_controle, 'w') as arquivo:
        arquivo.write(str(contador))
# END: 3f5a6d8b5c7a

