# Toxic-Comment-Classification
Multi-label classification system for detecting toxic comments across six categories using machine learning and deep learning approaches.

## Dataset
- **Source**: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Size**: 31,911 samples (after preprocessing)
- **Categories**: toxic, severe_toxic, obscene, threat, insult, identity_hate
- **Split**: 80% training, 20% testing with stratification

## Models

### Machine Learning
- **Logistic Regression**: Binary relevance strategy with class balancing
- **Naive Bayes**: MultinomialNB with oversampling and threshold optimization
- **Features**: TF-IDF vectors (unigrams + bigrams, max 10k vocabulary)

### Deep Learning
- **BERT**: Fine-tuned bert-base-uncased with classification head
- **Architecture**: 12-layer transformer with sigmoid output for multi-label prediction
- **Optimizer**: AdamW (lr=2e-5)

## Key Features
- Multi-label classification handling label correlations
- Class imbalance mitigation using RandomOverSampler
- Per-label threshold optimization for F1-score maximization
- Interactive GUI using Gradio for real-time toxicity detection

## Results
- **Best Performance**: BERT model
  - Toxic: F1=0.82
  - Obscene: F1=0.81  
  - Insult: F1=0.78
- **Logistic Regression**: Strong baseline with F1=0.72 for toxic comments
- **Challenge**: Lower performance on rare classes (threat, identity_hate)

## Preprocessing
- Text cleaning (URLs, HTML, punctuation removal)
- Lowercase normalization and lemmatization
- Stopword removal and tokenization
- Handled class imbalance with oversampling

## Technologies
- Python, scikit-learn, transformers, PyTorch
- NLTK for preprocessing
- Pandas, NumPy for data handling
- Gradio for GUI interface
