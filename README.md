# Fake News Classifier using ML and DL
<img width="1262" height="629" alt="image" src="https://github.com/user-attachments/assets/adfcd535-d606-411e-961d-fc7b0bbd3983" />

## Introduction

The rapid spread of misinformation through online news platforms has made fake news detection an important challenge in today’s digital world. This project aims to build a robust classifier that can accurately distinguish between fake and real news articles using both machine learning and deep learning approaches. The dataset includes thousands of news articles with labels, allowing for thorough experimentation with various models, preprocessing techniques, and evaluation metrics. By leveraging natural language processing and neural networks, the project provides an effective solution for automated fake news detection, supporting efforts to promote reliable information in society.

## Dataset

This project uses the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data), which is widely recognized for fake news detection research. The dataset consists of two CSV files: **True.csv** (real news articles from Reuters.com) and **Fake.csv** (fake news articles collected from unreliable sources flagged by Politifact and Wikipedia). Each file contains over 12,600 articles with details such as the article title, full text, subject, and publication date.

The dataset covers a wide range of topics, with a particular emphasis on political and world news from 2016 to 2017. While the data has been cleaned and processed, original punctuation and errors present in the fake news articles have been preserved to reflect real-world characteristics. For more details and to access the data, visit the [Kaggle dataset page](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data).

## Exploratory Data Analysis (EDA)

To gain an initial understanding of the dataset, several NLP techniques were employed. Common English stopwords and uninformative words were removed from both the titles and texts. Word clouds were generated to visually identify the most prominent and critical words distinguishing fake and true news articles.

<p align="center">
  <img width="671" height="444" alt="Screenshot 2025-08-03 at 1 38 17 PM" src="https://github.com/user-attachments/assets/afc4d1bd-918a-4722-8d07-ef9169ef7983" />
</p>
<p align="center"><b>Figure: Linguistic Landscape of Fake News Articles</b></p>

This word cloud captures the most prominent terms found in the text bodies of fake news articles. The visualization highlights the frequent appearance of political names such as “trump,” “donald,” “american,” “republican,” and “hillary.” Notably, emotionally charged and action-oriented words like “believe,” “make,” “even,” and “call” are also prevalent, reflecting the persuasive and provocative nature often associated with fabricated news. This representation demonstrates how fake news content tends to focus heavily on political figures, national identity, and sensational claims, providing insight into linguistic strategies that may be used to attract attention or provoke strong reader reactions.

---

<p align="center">
  <img width="671" height="444" alt="Screenshot 2025-08-03 at 1 24 43 PM" src="https://github.com/user-attachments/assets/a1f9e3c5-10db-4b78-b716-e622e7972fce" />
</p>
<p align="center"><b>Figure: Semantic Patterns in Authentic News Reporting</b></p>

This plot illustrates the key words most often used in the text of true news articles. Terms such as “united,” “state,” “government,” “white house,” “official,” “including,” and “north korea” dominate, highlighting a focus on formal institutions, geopolitical events, and objective reporting. The absence of highly sensationalized language suggests an emphasis on information delivery over emotional persuasion. The prevalence of “government,” “official,” and place names further supports the idea that genuine news sources rely on facts and credible attributions. This word cloud serves as a linguistic fingerprint of authentic journalism, providing a useful contrast to patterns seen in fabricated news content.

---

<p align="center">
 <img width="671" height="444" alt="image" src="https://github.com/user-attachments/assets/e3478f94-a70b-4831-a45d-4da45123e2ec" />
</p>
<p align="center"><b>Figure: Distinctive Lexicon of Fake News Headlines</b></p>

This plot visualizes the vocabulary most frequently found in the titles of fake news stories. Words such as “trump,” “hillary,” “obama,” “republican,” “democrat,” “supporter,” and “watch” dominate the landscape, revealing a heavy reliance on polarizing figures and urgent calls to action. The prevalence of emotionally loaded terms like “black,” “attack,” “lie,” and “racist” indicates an intent to provoke and capture attention. This headline word cloud demonstrates how fake news outlets employ sensationalism, controversy, and personalization in headline writing, differentiating themselves from mainstream journalistic norms.

---

<p align="center">
  <img width="671" height="444" alt="Screenshot 2025-08-03 at 1 41 25 PM" src="https://github.com/user-attachments/assets/b812c9a9-4521-43ea-afc1-9cf6759292e0" />
</p>
<p align="center"><b>Figure: Headline Vocabulary of True News Sources</b></p>

This visualization presents the most common words in the titles of true news articles. Noteworthy terms include “says,” “trump,” “white house,” “north korea,” “government,” “official,” and “congress.” These words emphasize the focus on authoritative sources and important global events. The frequent appearance of action words like “call,” “vote,” and “plan” suggests coverage of political decisions, legislative actions, and government statements. This pattern reflects journalistic priorities in headline writing—clarity, credibility, and event-centric reporting—making this word cloud an effective summary of mainstream news headline strategies.

---

<div align="center"> <img width="671" height="444" alt="Top Words in Fake News" src="https://github.com/user-attachments/assets/b6de51eb-0597-407e-a108-2c84f5b26e5d" /> <br> <b>1. Dissecting Deceptive Narratives: Top Words in Fake News</b> </div>
The first bar chart spotlights the 20 most frequent words in fake news articles. Dominance of names like “trump,” “president,” and “obama” highlights the political focus of much fake content. Terms such as “like,” “even,” and “make” illustrate how sensational and informal language is employed to maximize emotional impact. This uneven distribution suggests that fake news creators intentionally center on specific personalities and polarizing events, potentially to amplify engagement and manipulate opinions. The prominence of “clinton,” “hillary,” and “media” further underscores how these narratives are constructed around divisive topics. Ultimately, this analysis reveals not just the linguistic preferences of fake news articles, but also their underlying strategy—leveraging repetition and charged language to influence public perception and sow doubt about real-world events.

---

<div align="center"> <img width="671" height="444" alt="Top Words in True News" src="https://github.com/user-attachments/assets/20454cff-52ca-44ea-bb65-2d29c3b439fd" /> <br> <b>2. Lexical Signature of Authentic News Coverage</b> </div>
The second bar chart presents the most common words found in true news articles, offering a contrast to the language of fake news. Here, “trump” and “president” remain frequent, reflecting real-world relevance, but are joined by terms like “state,” “united,” “government,” and “official.” These words point to a more formal, fact-based reporting style that prioritizes governance, structure, and credible attribution. Words such as “republican,” “told,” “washington,” and “former” further ground the articles in established sources and institutional authority. This lexical profile suggests that authentic news stories emphasize accuracy, context, and verified statements over sensationalism. The analysis underscores how true news anchors itself in credible sources and measured language, reflecting an intent to inform rather than provoke, which is essential for reliable journalism.

---

<div align="center"> <img width="670" height="590" alt="Fake News Subject Distribution" src="https://github.com/user-attachments/assets/78a02224-4645-4f1a-ab03-3aba00af0052" /> <br> <b>3. Unequal Landscape: Fake News Subject Distribution</b> </div>
The pie chart for fake news subject distribution uncovers a notably uneven allocation of topics. “News” and “politics” overwhelmingly dominate, together comprising over two-thirds of the articles. “Left-news,” “Government News,” “Middle-east,” and “US_News” constitute much smaller segments, indicating a sharp thematic imbalance. This skew suggests that fake news articles are strategically concentrated in political discourse, where they can have the greatest societal impact. The underrepresentation of other subjects hints that the creators of fake news selectively target audiences most receptive to political manipulation or controversy. This visualization is crucial for understanding how misinformation exploits specific domains—mainly politics—to shape narratives, influence attitudes, and destabilize public trust. Recognizing these imbalances helps researchers and practitioners focus their detection and intervention efforts.

---

<div align="center"> <img width="670" height="590" alt="True News Subject Distribution" src="https://github.com/user-attachments/assets/34ea0a87-88e4-454f-87cd-85753790db86" /> <br> <b>4. Bipolar Domains: True News Subject Spectrum</b> </div>
The final pie chart highlights the subject distribution within true news, revealing a sharply binary structure. “PoliticsNews” and “worldnews” account for nearly equal proportions of all true news articles, with little room for other topics. This balanced yet narrow focus illustrates how legitimate news sources emphasize coverage of global events and political developments, ensuring comprehensive and factual reporting. The division underscores the journalistic commitment to transparency and the prioritization of significant, far-reaching issues. However, the lack of diversity in subject matter may also indicate a media tendency to gravitate toward stories with broad relevance or higher stakes. Understanding this distribution provides insight into news priorities and can guide the development of more robust fake news detection models by highlighting the differences in topical emphasis.

---

<div align="center"> <img width="671" height="444" alt="Monthly Fake vs True News Trends" src="https://github.com/user-attachments/assets/03bad1c8-ea35-4c9c-8938-3abf7a38cd02" /> <br> <b>5. Rhythms of Reality and Fabrication: Monthly News Publication Trends</b> </div>
This line plot presents the month-by-month publication counts for both fake and true news articles. Distinct trends emerge, with fake news consistently outnumbering true news in the early months, suggesting a strategic surge in misinformation. A dramatic rise in true news volume near the end of 2017 signals a response to heightened global or political events, possibly reflecting journalistic efforts to counteract misinformation or increased public scrutiny. The graph also highlights cyclical and event-driven fluctuations in news reporting. This temporal analysis is vital for understanding how fake and true news propagate over time and how external events or media interventions can influence the scale and dynamics of news dissemination in the digital era.

---

## Data Preprocessing

To prepare the data for modeling, several preprocessing steps were applied:
- A new column, `full_text`, was created by merging the `title` and `text` fields for each article.
- Text cleaning included converting to lowercase, removing punctuation, numeric characters, and stopwords, followed by lemmatization to reduce words to their base forms.
- A histogram was used to analyze the distribution of text lengths, leading to the selection of a maximum input length of 300 tokens and a vocabulary size of 20,000 most frequent words.

---

## Data Splitting and Tokenization

Text data was tokenized and converted to padded integer sequences suitable for input into machine learning and deep learning models. The process included:
- Initializing a Keras tokenizer with a maximum vocabulary size of 20,000 and an out-of-vocabulary token.
- Fitting the tokenizer on the cleaned text and converting texts to sequences.
- Padding sequences to a uniform length of 300.
- Splitting the data into training and testing sets with a 70:30 ratio.

---

## Model Development

A robust pipeline was created for both machine learning and deep learning approaches:

### Machine Learning Models

The following classifiers were implemented as part of a comparative evaluation pipeline:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors
- XGBoost Classifier
- CatBoost Classifier
- LightGBM Classifier

### Deep Learning Models

For deep learning, both LSTM and Bidirectional LSTM architectures were utilized to capture sequential dependencies and contextual information within the news articles.

---

## Results and Evaluations

<p align="center"><b>Table 1: Comparative Evaluation of Machine Learning and Deep Learning Models for Fake News Detection</b></p>
<p align="center"><b>Unveiling Model Performance: Accuracy and Class-wise Evaluation Across All Algorithms</b></p>

| Model                  | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | Precision (True) | Recall (True) | F1-Score (True) | Macro Avg | Weighted Avg |
|------------------------|:--------:|:---------------:|:-------------:|:---------------:|:----------------:|:-------------:|:---------------:|:---------:|:------------:|
| Logistic Regression    | 0.61     | 0.61            | 0.75          | 0.67            | 0.62             | 0.46          | 0.53            | 0.60      | 0.61         |
| Random Forest          | 0.90     | 0.90            | 0.91          | 0.90            | 0.89             | 0.88          | 0.89            | 0.89      | 0.90         |
| KNeighbors Classifier  | 0.58     | 0.67            | 0.42          | 0.51            | 0.54             | 0.77          | 0.64            | 0.58      | 0.57         |
| XGBoost Classifier     | 0.99     | 0.99            | 1.00          | 0.99            | 1.00             | 0.98          | 0.99            | 0.99      | 0.99         |
| CatBoost Classifier    | 0.99     | 0.99            | 1.00          | 0.99            | 1.00             | 0.99          | 0.99            | 0.99      | 0.99         |
| LightGBM Classifier    | 0.99     | 0.99            | 1.00          | 0.99            | 1.00             | 0.98          | 0.99            | 0.99      | 0.99         |
| LSTM                   | 0.98     | 0.98            | 0.98          | 0.98            | 0.98             | 0.98          | 0.98            | 0.98      | 0.98         |
| Bidirectional LSTM     | 1.00     | 1.00            | 1.00          | 1.00            | 1.00             | 1.00          | 1.00            | 1.00      | 1.00         |
