import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph

# Carregar os dados
data = pd.read_csv('creditcard.csv')

# Tratamento de valores ausentes (NaN) nos dados
data.dropna(inplace=True)

# Separar as features (X)
X = data.drop(columns=['Class'])
y = data['Class']  # Alvo real para cálculo de AUC-ROC e precisão para detecção de anomalias

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Lista para armazenar os resultados
results = []

# Função para treinar e avaliar modelos não supervisionados
def train_and_evaluate_unsupervised(model, model_name):
    accuracy, auc_roc = None, None
    
    if model_name == "DBSCAN":
        y_pred = model.fit_predict(X_pca)
    elif model_name == "Autoencoder":
        model.fit(X_pca, X_pca)
        mse = np.mean(np.power(X_pca - model.predict(X_pca), 2), axis=1)
        threshold = np.percentile(mse, 95)  # Limite para detectar anomalias
        y_pred = (mse > threshold).astype(int)
        accuracy = accuracy_score(y, y_pred)
        auc_roc = roc_auc_score(y, mse)
        silhouette = 'N/A'
    else:
        model.fit(X_pca)
        y_pred = model.predict(X_pca)
        if model_name == "Isolation Forest":
            accuracy = accuracy_score(y, y_pred)
            auc_roc = roc_auc_score(y, model.decision_function(X_pca))
        else:
            accuracy = 'N/A'
            auc_roc = 'N/A'
    
    silhouette = silhouette_score(X_pca, y_pred) if model_name != 'Autoencoder' else 'N/A'
    
    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Silhouette Score": silhouette,
        "AUC-ROC": auc_roc
    })

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Silhouette Score: {silhouette}")
    print(f"AUC-ROC: {auc_roc}")
    print("\n" + "-"*50 + "\n")

# Lista de modelos a serem treinados
models = [
    (KMeans(n_clusters=2, random_state=42), "K-Means"),
    (IsolationForest(contamination=0.01, random_state=42), "Isolation Forest"),
    (DBSCAN(eps=0.5, min_samples=5), "DBSCAN"),
    (MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=200), "Autoencoder")
]

# Treinar e avaliar cada modelo
for model, name in models:
    train_and_evaluate_unsupervised(model, name)

# Criar um DataFrame com os resultados
results_df = pd.DataFrame(results)

# Exibir a tabela de resultados
print("\nTabela de Resultados Gerais:")
print(results_df)

# Função para gerar e salvar o relatório em PDF
def save_results_to_pdf(results_df, filename='unsupervised_results.pdf'):
    # Criar um documento PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Converter o DataFrame em uma lista de listas para a tabela
    data = [list(results_df.columns)] + results_df.values.tolist()

    # Criar a tabela e definir estilos
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    # Adicionar a tabela ao documento
    elements = [Paragraph("Relatório de Resultados dos Métodos de Aprendizado Não Supervisionado", styles['Title'])]
    elements.append(table)

    # Salvar o PDF
    doc.build(elements)

# Salvar os resultados em um arquivo PDF
save_results_to_pdf(results_df, filename='unsupervised_model_evaluation_results.pdf')
