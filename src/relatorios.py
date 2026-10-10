import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config_modelos import CURRENT_MODEL, IMG_SIZE, BATCH_SIZE, EPOCHS, FOLDS, SEED, PATIENCE, EVALUATION_FOLDER


def preparar_resultados_csv(df_folds, df_resumo):
    df_folds_csv = df_folds.copy()
    df_folds_csv["etapa"] = "fold"

    df_resumo_csv = df_resumo.copy()
    df_resumo_csv["etapa"] = "validacao_media"

    return pd.concat([df_folds_csv, df_resumo_csv], ignore_index = True, sort = False)


def salvar_resultados_csv(df_folds, df_resumo):
    resultados_csv = preparar_resultados_csv(df_folds, df_resumo)
    caminho_resultados = (EVALUATION_FOLDER / "csv" / f"resultados_{CURRENT_MODEL.lower()}_rgb_" f"{IMG_SIZE[0]}x{IMG_SIZE[1]}.csv")
    caminho_resultados.parent.mkdir(parents = True, exist_ok = True)
    resultados_csv.to_csv(caminho_resultados, index = False)

    print(f"\nResultados numéricos salvos em: {caminho_resultados}")

    return caminho_resultados


def salvar_resultados_excel(df_folds, df_resumo, quantidade_total_imagens):
    caminho_excel = (EVALUATION_FOLDER / "excel" / "resultados_experimentos.xlsx")

    caminho_excel.parent.mkdir(parents = True, exist_ok = True)

    configuracoes = pd.DataFrame({
        "parametro": [
            "modelo",
            "entrada",
            "tamanho_imagem",
            "batch_size",
            "epocas_maximas",
            "folds",
            "seed",
            "patience_early_stopping",
            "quantidade_total_imagens"
        ],
        "valor": [
            CURRENT_MODEL,
            "RGB",
            f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            BATCH_SIZE,
            EPOCHS,
            FOLDS,
            SEED,
            PATIENCE,
            quantidade_total_imagens
        ]
    })

    abas_anteriores = {}
    
    if (caminho_excel.exists()):
        try:
            abas_anteriores = pd.read_excel(caminho_excel, sheet_name = None)

        except Exception:
            abas_anteriores = {}

    def acumular(nome_aba, df_novo):
        if (nome_aba in abas_anteriores):
            return pd.concat([abas_anteriores[nome_aba], df_novo], ignore_index = True)
        
        return df_novo

    with pd.ExcelWriter(
        caminho_excel,
        engine = "openpyxl"
    ) as escritor:
        
        acumular("FOLDS", df_folds).to_excel(
            escritor,
            sheet_name = "FOLDS",
            index = False
        )

        acumular("MEDIA_FOLD", df_resumo).to_excel(
            escritor,
            sheet_name = "MEDIA_FOLD",
            index = False
        )

        acumular("CONFIGURACAO", configuracoes).to_excel(
            escritor,
            sheet_name = "CONFIGURACAO",
            index = False
        )

    return caminho_excel


def gerar_matriz_confusao(matriz, classes):
    plt.figure(figsize = (12, 10))
    sns.heatmap(matriz, annot = True, fmt = "d", cmap = "Blues", xticklabels = classes, yticklabels = classes)

    plt.title("Matriz de Confusão Global (K-Fold)")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.tight_layout()

    caminho = EVALUATION_FOLDER / "matrizes" / f"matriz_{CURRENT_MODEL.lower()}_rgb_{IMG_SIZE[0]}x{IMG_SIZE[1]}.png"
    caminho.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(caminho)
    
    plt.close()