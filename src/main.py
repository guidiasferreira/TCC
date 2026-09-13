import numpy as np

from dados import baixar_dataset, copiar_pastas, coletar_caminhos_rotulos, separar_test_set
from treino import rodar_kfold
from avaliar_modelo import treinar_modelo_final, avaliar_modelo
from relatorios import salvar_resultados_csv, salvar_resultados_excel


if (__name__ == "__main__"):
    dataset_path = baixar_dataset()
    pasta_destino = copiar_pastas(dataset_path)

    paths, labels = coletar_caminhos_rotulos(pasta_destino)
    classe_unica = np.unique(labels)

    train_val_paths, test_paths, train_val_labels, test_labels = separar_test_set(
        paths, labels
    )

    df_folds, df_resumo = rodar_kfold(
        train_val_paths, train_val_labels, classe_unica
    )

    modelo_final, tempo_final, epocas_finais = treinar_modelo_final(
        train_val_paths, train_val_labels, classe_unica
    )

    df_teste, df_metricas_classe, matriz = avaliar_modelo(
        modelo_final, test_paths, test_labels, classe_unica
    )

    salvar_resultados_csv(
        df_folds,
        df_resumo,
        df_teste
    )

    salvar_resultados_excel(
        df_folds = df_folds,
        df_resumo = df_resumo,
        df_teste = df_teste,
        df_metricas_classe = df_metricas_classe,
        matriz = matriz,
        modelo = modelo_final,
        tempo_final = tempo_final,
        epocas_finais = epocas_finais,
        quantidade_desenvolvimento = len(train_val_paths),
        quantidade_teste = len(test_paths)
    )