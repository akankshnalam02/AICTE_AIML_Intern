
# 🎓 AICTE AIML Internship Projects (2025)

Welcome to my submission for the **AICT Pvt. Ltd. - AI/ML Internship**!  
This repository includes two practical machine learning projects designed to demonstrate core concepts in **recommendation systems** and **time series prediction**.

---

## 📁 Repository Structure
```
AICTE_AIML_Intern/
│
├── 1_Movie Recommender System/
│ ├── Task_1_ Movie Recommender System.ipynb
│ ├── movie_recommender_app/
│ │ ├── app.py
│ │ └── movie_recommender.py
│ └── movie_lens_data/ ❌ (Excluded in GitHub repo – see dataset note below)
│ ├── genome_scores.csv
│ ├── genome_tags.csv
│ ├── link.csv
│ ├── movie.csv
│ ├── rating.csv
│ └── tag.csv
│
├── 2_Stock Market Price Predictor/
│ ├── Stocks_Prediction_using_LSTM.ipynb
│ └── AMAZON_2006_to_2018_stocks.csv
│
└── README.md
```
## 🎬 Task 1: Movie Recommender System

### 📌 Objective
Build a **content-based movie recommender system** using genres and user-assigned tags to suggest similar movies.

### 💡 Key Features
- **Data Source**: [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- **Preprocessing**: Merge genres with top genome tags per movie
- **Feature Extraction**: TF-IDF vectorization of combined movie profiles
- **Similarity Measurement**: Cosine similarity between movie profiles
- **Recommendation Engine**: Returns Top 5 similar movies
- **Web App**: Built using **Streamlit** for real-time recommendations with a fuzzy search bar

### 🧪 How to Run
1. 📥 Download the MovieLens dataset from [here](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).
2. Place the CSV files inside:
    `1_Movie Recommender System/movie_lens_data/`

3. Run the Streamlit app:
```bash
cd "1_Movie Recommender System/movie_recommender_app"
streamlit run app.py
```
Access app at: `http://localhost:8501`

### 🧰 Technologies Used
- Pandas, scikit-learn
- TF-IDF, Cosine Similarity
- Streamlit for UI
- difflib for fuzzy matching in search

### 📸 Output Example
🎥 Content-Based Movie Recommender

🔍 Search for a movie: Toy Story (1995)

✅ Did you mean: Toy Story (1995)?

🎯 Top 5 Recommended Movies:

1.Antz (1998)

2.Toy Story 2 (1999)

3.Adventures of Rocky and Bullwinkle, The (2000)

4.Emperor's New Groove, The (2000)

5.Monsters, Inc. (2001)

---

## 📈 Task 2: Stock Market Price Predictor

### 📌 Objective
Predict the closing price of a stock using historical data and a deep learning model.

### 📂 Dataset
`AMAZON_2006_to_2018_stocks.csv`  
Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 🧠 Model Used
- LSTM (Long Short-Term Memory) neural network
- Historical window-based prediction
- Train/test split and RMSE evaluation

### 📊 Output
- Actual vs Predicted Close Price Plot
- Model training and validation loss graphs

### 🧰 Technologies Used
- Keras, TensorFlow
- Matplotlib, Seaborn
- scikit-learn, MinMaxScaler
- Pandas, NumPy

### 🔁 Steps Performed
- Load & visualize stock price trends
- Normalize closing price
- Create time series data for LSTM
- Build and train LSTM model
- Plot actual vs predicted prices

---

## ⚠️ Note on Dataset Size
This repository does not contain the full MovieLens dataset due to GitHub’s 100MB file size limit.  
Please download the data manually from Kaggle and place it in the appropriate directory.

---

## 📌 Final Thoughts
These two mini-projects demonstrate essential AI/ML skills:

🔍 Content-based filtering  
📉 Deep learning for time series  
🧠 Hands-on coding + model explanation  
🖥️ Interactive visualization using Streamlit

---

## 🧑‍💻 Author
**Akanksh Nalam**  
B.Tech – Artificial Intelligence and Machine Learning  
SRKR Engineering College (2022–2026)  
🔗 [GitHub](https://github.com/akankshnalam02) | [LinkedIn](https://www.linkedin.com/akankshnalam)

📌 Feel free to fork, explore, and contribute!
