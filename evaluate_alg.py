import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# Carregar os dados
data = pd.read_csv('creditcard.csv')

# Verificar se há NaNs nos rótulos de classe (y)
print("Valores ausentes em y antes do tratamento:")
print(data['Class'].isnull().sum())  # Verifique se há NaNs em 'Class'

# Remover amostras com NaN em 'Class' (se houver)
data.dropna(subset=['Class'], inplace=True)

# Verificar novamente se há NaNs após a limpeza
print("Valores ausentes em y após o tratamento:")
print(data['Class'].isnull().sum())  # Deve retornar 0 se não houver NaNs em 'Class'

# Separar as features (X) e o alvo (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Função para treinar e avaliar um modelo
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": auc_roc
    })

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "-"*50 + "\n")

# Lista de modelos a serem treinados
models = [
    (LogisticRegression(max_iter=2000), "Logistic Regression"),
    (DecisionTreeClassifier(), "Decision Tree"),
    (RandomForestClassifier(n_estimators=50, n_jobs=-1), "Random Forest"),
    (SVC(kernel='linear', probability=True), "SVM"),
    (GradientBoostingClassifier(), "Gradient Boosting Machine (GBM)")
]

# Lista para armazenar os resultados
results = []

# Treinar e avaliar cada modelo
for model, name in models:
    train_and_evaluate_model(model, name)

# Criar um DataFrame com os resultados
results_df = pd.DataFrame(results)

# Exibir a tabela de resultados
print("\nTabela de Resultados Gerais:")
print(results_df)

# Função para gerar e salvar o relatório em PDF
def save_results_to_pdf(results_df, filename='results.pdf'):
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
    elements = []
    elements.append(table)

    # Salvar o PDF
    doc.build(elements)

# Salvar os resultados em um arquivo PDF
save_results_to_pdf(results_df, filename='model_evaluation_results.pdf')
