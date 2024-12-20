# 필요한 라이브러리 임포트
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 데이터 로드
data = pd.read_csv('flood_data.csv')

# 데이터 전처리
data.dropna(inplace=True)  # 결측치 제거
X = data[['Rainfall', 'Temperature', 'Humidity', 'SoilMoisture', 'RainDuration']]
y = data['Flood']

# 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 초기화
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

# 모델 훈련 및 평가
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # 모델 훈련
    preds = model.predict(X_test)  # 예측
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1, 'AUC': auc}

# 결과 출력
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['Accuracy']:.2f}, F1 Score: {metrics['F1 Score']:.2f}, AUC: {metrics['AUC']:.2f}")
