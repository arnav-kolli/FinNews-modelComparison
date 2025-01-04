# Investigating the Impact of Long-Form Financial News Articles on Stock Market Fluctuations
**CS 485: Applications of Natural Language Processing**  
**Authors**: Arnav Kolli, Anvesh Sunkara, Rahul Vedula

---

## Abstract
This paper investigates the relationship between financial news and stock market fluctuations by attempting to gauge the impact of an article’s content and sentiment on the opening price of a stock the following day. To do this, we developed a multitude of NLP models trained on a dataset comprising article content, stock tickers, and stock prices. The results of our investigation showed that an article’s impact on a stock, while tangible, is hard to predict (as evidenced by low accuracies in our models) due to the extremely nuanced nature of the stock market.

---

## Introduction
Attempting to predict stock market fluctuations is something that, if done correctly, can have huge fiscal upside. This has led to the creation and development of models, companies, and even industries dedicated to predicting and anticipating these fluctuations. These predictions are usually based on either:
- **Technical Analysis**, which analyzes historical data and chart patterns to identify suitable investment points, or
- **Fundamental Analysis**, which focuses on measuring an asset’s intrinsic value through economic, financial, and other qualitative/quantitative factors.

Given the rise of algorithmic trading and the vast amount of fast-disseminating information, this paper explores the potential of **qualitative fundamental analysis**—through news content and sentiment—in short-term trading. The key research question:  
- **Does a tangible relationship exist between the content and sentiment of a news article and the opening price of the associated stock the day after the article was published?**

Additional research goals include:
- **Determining whether the sentiment expressed in financial articles can accurately forecast stock price trends for major companies.**
- **Comparing the accuracy of various NLP models in predicting stock movements.**
- **Identifying keywords or phrases in articles that correlate strongly with stock price shifts.**

---

## Related Work
Two foundational studies influenced our research:
1. **Schumaker and Chen**: "A Quantitative Stock Prediction System Based on Financial News." This study used linguistic, financial, and statistical techniques to create the Arizona Financial Text System.
2. **Bollen, Mao, and Zeng**: "Twitter Mood Predicts the Stock Market." This study analyzed mood states derived from large-scale Twitter feeds to predict movements in the Dow Jones Industrial Average.

---

## Data Collection

### Financial News
We initially relied on a Kaggle dataset with over a million article headlines but found it insufficient due to outdated content. We filtered headlines to focus on Fortune 50 companies and later integrated long-form article content using the **Benzinga Financial News API**.

### Stock Prices and Annotations
Stock prices were fetched using an unofficial Yahoo Finance API. Adjustments were made for weekends:
- If **day one** was a weekend, the price for the preceding Friday was fetched.
- If **day two** was a weekend, the price for the subsequent Monday was fetched.

**Final Dataset**: 17,021 unique article bodies across 60 different tickers.

---

## Methodology

### Classification Model
We implemented a multi-class classification model with three labels:
- **Increase (I)**: Price increase.
- **Decrease (D)**: Price decrease.
- **Neutral (N)**: Movement within ±1% of day one's opening price.

### Data Engineering and Feature Engineering
Preprocessing steps included tokenization, stopword removal, and stemming. Numerical data within texts was ignored to focus solely on the impact of textual content and sentiment.

### Models Tested
We tested the following models:
1. **Support Vector Machine (SVM)**:
    - Used TF-IDF vectorization (max 5000 features).
    - Default hyperparameter settings.
2. **Convolutional Neural Network (CNN)**:
    - Standard Keras tokenizer with padding.
    - Layers: Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout.
3. **Random Forest**:
    - Default hyperparameters.
4. **BERT**:
    - Used Hugging Face’s `TFBertForSequenceClassification` with:
      - Learning rate: 5e-5.
      - Loss function: Categorical cross-entropy.
      - Trained for 3 epochs with batch size 8.
5. **NLTK Sentiment Analysis (Control)**:
    - Baseline model.

### Experimental Setup
All experiments were conducted on Google Colab with a T4 GPU hardware accelerator.

---

## Evaluation

### Metrics
We evaluated model performance using:
- **Accuracy**
- **Precision**
- **Recall**
- **F-1 Score**

### Results
1. **Accuracy**: Models achieved ~40%, indicating limited success in predicting stock price movements.
2. **Precision and Recall**: SVM and Random Forest outperformed others. BERT struggled due to the shorter nature of articles.
3. **F-1 Score**: Indicated room for improvement, especially for BERT.

---

## Challenges
1. **Computational Constraints**:
    - Google Colab timeouts required careful resource allocation.
2. **Data Acquisition**:
    - Anti-scraping measures and API request limits hindered progress.
3. **Overfitting**:
    - CNN models overfitted initially, mitigated through early stopping.
4. **Data Cleaning**:
    - Considerable effort required to handle diverse formats.

---

## Future Improvements
- Use more authoritative sources like Bloomberg.  
- Fine-tune hyperparameters for better performance.  
- Invest in advanced GPU resources for scalability.  
- Clean data more rigorously to remove noise and promotional content.

---

## Conclusion
While our models consistently achieved modest accuracies (~40%), this research highlights the complex, multifaceted nature of stock market dynamics. Financial articles alone may not significantly influence stock prices, suggesting a need to integrate additional data sources (e.g., social media sentiment, economic indicators). This study serves as a foundation for future exploration into NLP applications in financial contexts.

---

## Citations and References
- Schumaker, R. P., & Chen, H. (2009). *A Quantitative Stock Prediction System Based on Financial News.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306457309000478#aep-section-id14)  
- Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter Mood Predicts the Stock Market.* [Journal of Computational Science](https://doi.org/10.1016/j.jocs.2010.12.007)  
