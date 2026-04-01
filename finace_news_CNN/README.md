# Financial News Sentiment Analysis (NLP)

## Overview
금융 뉴스 텍스트의 감성을 **Positive / Neutral / Negative** 3개 클래스로 분류하는 딥러닝 모델 비교 프로젝트

## Dataset
- **FinancialPhraseBank** (Kaggle)
- 4,846개 금융 뉴스 문장
- 3개 클래스: Neutral(2,879) / Positive(1,363) / Negative(604)
- Train: 3,876 / Test: 970 (80:20, Stratified)

## Models & Results

| Model | Test Accuracy | Test Loss | 비고 |
|-------|:---:|:---:|------|
| **MLP** | 68.35% | 0.7393 | 베이스라인, 단어 순서 무시 |
| **CNN** | **73.09%** | **0.7194** | **최고 성능 - 채택** |
| GRU | 59.38% | 0.9261 | 학습 실패 |
| LSTM | 59.38% | 0.9262 | 학습 실패 |

## Pipeline

```
1. 데이터 로드 및 EDA
2. 전처리 (Tokenizing, Padding, Label Encoding)
3. 4개 모델 학습 (EarlyStopping 적용)
4. 모델 비교 → CNN 채택
5. 실제 Yahoo Finance 뉴스 기사로 예측
```

## Key Findings

- **CNN이 금융 텍스트 감성분석에 가장 적합** — 핵심 키워드 패턴("profit increased", "loss reported") 추출에 강점
- **GRU/LSTM 학습 실패** — 데이터 규모(4,846건)가 작고 클래스 불균형(neutral 59%)으로 RNN 계열이 다수 클래스만 예측
- **EarlyStopping**으로 과적합 방지 — 모델별 개별 callback 생성 필요
- 실제 Yahoo Finance 뉴스 4건에 대해 본문 전체 / 앞 3문장 예측 비교 수행

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- scikit-learn
- NLTK
- newspaper3k (뉴스 크롤링)

