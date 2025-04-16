import os
import opendatasets as od
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def main():
    print("=== Configurando Kaggle JSON ===")
    # Cria a pasta ~/.kaggle se ainda não existir
    os.system('mkdir -p ~/.kaggle')

    # Copia o kaggle.json local para ~/.kaggle
    if not os.path.isfile('kaggle.json'):
        print("ERRO: kaggle.json não encontrado na pasta atual!")
        return
    else:
        os.system('cp kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        print("kaggle.json copiado para ~/.kaggle e permissões ajustadas.\n")

    # 1. Baixar dataset do Kaggle (arquivo data.csv) usando opendatasets
    print("=== Baixando dataset do Kaggle (Ecommerce Data) ===")
    od.download("https://www.kaggle.com/datasets/carrie1/ecommerce-data")

    # O dataset será salvo em ./ecommerce-data/data.csv
    file_path = "ecommerce-data/data.csv"

    # 2. Ler o CSV
    print("=== Lendo CSV ===")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("Dimensão inicial:", df.shape)
    
    # 3. Remover valores nulos
    df.dropna(inplace=True)
    
    # 4. Remover devoluções (InvoiceNo começando com 'C')
    df = df[df['InvoiceNo'].str.startswith('C') == False]
    print("Dimensão após limpeza:", df.shape)
    
    # 5. Criar variável alvo 'Abandoned'
    user_purchase_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
    def only_purchased_once(cid):
        return 1 if user_purchase_counts[cid] == 1 else 0
    df['Abandoned'] = df['CustomerID'].apply(only_purchased_once)
    
    # 6. Criar 'TotalPrice'
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # 7. Definir features
    features = df[['Quantity', 'UnitPrice', 'TotalPrice', 'Country']].copy()
    target = df['Abandoned']
    
    # 8. Label Encoding da coluna 'Country'
    le = LabelEncoder()
    features['Country'] = le.fit_transform(features['Country'])
    
    # 9. Escalonar
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target
    
    # 10. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 11. Definir modelos
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    
    # 12. Treinar e avaliar cada modelo
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': report_dict['weighted avg']['precision'],
            'Recall': report_dict['weighted avg']['recall'],
            'F1-Score': report_dict['weighted avg']['f1-score']
        })
        
        print(f"\n--- {name} ---")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # 13. Comparar resultados
    results_df = pd.DataFrame(results)
    print("\n===== Resumo das Métricas =====")
    print(results_df)
    
    # 14. Plotar gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Comparação de Acurácia dos Modelos')
    plt.ylim(0, 1)
    plt.show()

    print("\nPronto! O código executou a comparação dos 5 modelos (incluindo KNN).")

if __name__ == "__main__":
    main()
