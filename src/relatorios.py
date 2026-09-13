import numpy as np
import pandas as pd

from config import CURRENT_MODEL, IMG_SIZE, BATCH_SIZE, EPOCHS, FOLDS, TEST_SIZE, SEED, PATIENCE, EVALUATION_FOLDER


def preparar_resultados_csv(df_folds, df_resumo, df_teste):
    df_folds_csv = df_folds.copy()
    df_folds_csv["etapa"] = "fold"

    df_resumo_csv = df_resumo.copy()
    df_resumo_csv["etapa"] = "validacao_media"

    df_teste_csv = df_teste.copy()
    df_teste_csv["etapa"] = "teste"

    return pd.concat([df_folds_csv, df_resumo_csv, df_teste_csv], ignore_index = True, sort = False)


def salvar_resultados_csv(df_folds, df_resumo, df_teste):
    resultados_csv = preparar_resultados_csv(df_folds, df_resumo, df_teste)
    caminho_resultados = (EVALUATION_FOLDER / "evaluation" / f"resultados_{CURRENT_MODEL.lower()}_rgb_" f"{IMG_SIZE[0]}x{IMG_SIZE[1]}.csv")
    caminho_resultados.parent.mkdir(parents = True, exist_ok = True)
    resultados_csv.to_csv(caminho_resultados, index = False)

    print(f"\nResultados numéricos salvos em: {caminho_resultados}")

    return caminho_resultados


def salvar_resultados_excel(df_folds, df_resumo, df_teste, df_metricas_classe, matriz, modelo, tempo_final, epocas_finais, quantidade_desenvolvimento, quantidade_teste):
    caminho_excel = EVALUATION_FOLDER / "resultados_experimentos.xlsx"

    caminho_excel.parent.mkdir(parents = True, exist_ok = True)

    configuracoes = pd.DataFrame({
        "parametro": [
            "modelo",
            "entrada",
            "tamanho_imagem",
            "batch_size",
            "epocas_maximas",
            "folds",
            "percentual_teste",
            "seed",
            "patience_early_stopping",
            "quantidade_desenvolvimento",
            "quantidade_teste",
            "epocas_modelo_final",
            "tempo_treino_modelo_final_seg"
        ],
        "valor": [
            CURRENT_MODEL,
            "RGB",
            f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            BATCH_SIZE,
            EPOCHS,
            FOLDS,
            TEST_SIZE,
            SEED,
            PATIENCE,
            quantidade_desenvolvimento,
            quantidade_teste,
            epocas_finais,
            tempo_final
        ]
    })

    custo_computacional = pd.DataFrame([{
        "modelo": CURRENT_MODEL,
        "parametros_totais": modelo.count_params(),
        "parametros_treinaveis": sum(
            np.prod(peso.shape)
            for peso in modelo.trainable_weights
        ),
        "tempo_treino_modelo_final_seg": tempo_final,
        "epocas_modelo_final": epocas_finais,
        "tempo_inferencia_seg": (
            df_teste.iloc[0]["tempo_inferencia_seg"]
        ),
        "tempo_medio_por_imagem_seg": (
            df_teste.iloc[0]["tempo_medio_por_imagem_seg"]
        )
    }])

    with pd.ExcelWriter(
        caminho_excel,
        engine="openpyxl"

    ) as escritor:
        df_resumo.to_excel(
            escritor,
            sheet_name="resumo",
            index=False
        )

        df_folds.to_excel(
            escritor,
            sheet_name="folds",
            index=False
        )

        df_teste.to_excel(
            escritor,
            sheet_name="teste",
            index=False
        )

        df_metricas_classe.to_excel(
            escritor,
            sheet_name="metricas_classe",
            index=False
        )

        matriz.to_excel(
            escritor,
            sheet_name="matriz_confusao",
            index=True
        )

        configuracoes.to_excel(
            escritor,
            sheet_name="configuracao",
            index=False
        )

        custo_computacional.to_excel(
            escritor,
            sheet_name="custo_computacional",
            index=False
        )

    print(f"\nResultados organizados no Excel: {caminho_excel}")

    return caminho_excel