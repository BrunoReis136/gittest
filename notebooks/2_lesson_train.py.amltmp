import argparse
import os
import pandas as pd
from azureml.core import Run

def main():
    # Analisar argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help="Name of input dataset")
    args = parser.parse_args()

    # Pegar contexto de execução do Azure ML
    run = Run.get_context()

    # Pegar o dataset pelo seu nome passado através do --input_data
    print(f'Loading dataset named:{args.input_data}')
    input_dataset = run.input_datasets[args.input_data]

    # Converte para dataframe pandas
    df = input_dataset.to_pandas_dataframe()

    # Procede com a preparação de dados e treinamento
    print(f'Dataset loaded successfully with shape{df.shape}')
    
        