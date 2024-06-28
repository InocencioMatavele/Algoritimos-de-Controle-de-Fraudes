import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph

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

# Função para criar, treinar e avaliar um modelo de rede neural
def create_and_evaluate_nn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model

# Função para criar, treinar e avaliar um modelo de CNN
def create_and_evaluate_cnn():
    model = Sequential([
        Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model

# Função para criar, treinar e avaliar um modelo de RNN/LSTM
def create_and_evaluate_rnn():
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model

# Função para criar, treinar e avaliar um modelo de Autoencoder
def create_and_evaluate_autoencoder():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(X_train.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0, callbacks=[EarlyStopping(patience=3)])
    return model

# Lista de modelos a serem treinados
models = [
    (create_and_evaluate_nn, "Neural Network"),
    (create_and_evaluate_cnn, "CNN"),
    (create_and_evaluate_rnn, "RNN/LSTM"),
    (create_and_evaluate_autoencoder, "Autoencoder")
]

# Lista para armazenar os resultados
results = []

# Função para avaliar os modelos
def evaluate_model(model, model_name):
    if model_name in ["CNN", "RNN/LSTM"]:
        X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
        y_pred = model.predict(X_test_reshaped)
    else:
        y_pred = model.predict(X_test)
    
    if model_name == "Autoencoder":
        mse = np.mean(np.power(X_test - y_pred, 2), axis=1)
        y_pred = (mse > np.percentile(mse, 95)).astype(int)
    else:
        y_pred = (y_pred > 0.5).astype(int).flatten()
    
    y_pred_proba = y_pred.flatten() if model_name != "Autoencoder" else mse
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

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

# Treinar e avaliar cada modelo
for create_model, name in models:
    model = create_model()
    evaluate_model(model, name)

# Criar um DataFrame com os resultados
results_df = pd.DataFrame(results)

# Exibir a tabela de resultados
print("\nTabela de Resultados Gerais:")
print(results_df)

# Função para gerar e salvar o relatório em PDF
def save_results_to_pdf(results_df, filename='deep_learning_results.pdf'):
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
    elements = [Paragraph("Relatório de Resultados dos Modelos de Deep Learning", styles['Title'])]
    elements.append(table)

    # Salvar o PDF
    doc.build(elements)

# Salvar os resultados em um arquivo PDF
save_results_to_pdf(results_df, filename='deep_learning_model_evaluation_results.pdf')
