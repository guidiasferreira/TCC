import os
import numpy as np
import keras.utils

from config_modelos import SEED
from dados import baixar_dataset, copiar_pastas, coletar_caminhos_rotulos
from validar_modelo import rodar_kfold
from relatorios import salvar_resultados_csv, salvar_resultados_excel

os.environ["PYTHONHASHSEED"] = str(SEED)

keras.utils.set_random_seed(SEED)

if (__name__ == "__main__"):
    dataset = baixar_dataset()
    pasta_destino = copiar_pastas(dataset)

    paths, labels = coletar_caminhos_rotulos(pasta_destino)
    classe_unica = np.unique(labels)

    df_folds, df_resumo = rodar_kfold(paths, labels, classe_unica)

    salvar_resultados_csv(df_folds, df_resumo)
    salvar_resultados_excel(df_folds = df_folds, df_resumo = df_resumo, quantidade_total_imagens = len(paths)) 