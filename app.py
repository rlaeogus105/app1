import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title('기계학습을 이용한 예측 분석')

# 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    
    # 데이터 전처리
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 모델 학습
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 결과 출력
    st.write(f"정확도: {accuracy_score(y_test, y_pred)}")
    st.text("분류 보고서:")
    st.text(classification_report(y_test, y_pred))
    
    # 중요 피처 시각화
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    feature_importances.plot(kind='bar', ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
