# 🎬 MCA E-Consultation Sentiment Analyzer  

An AI-powered web application built with **Streamlit** that analyzes and summarizes public comments.  
The project demonstrates **sentiment analysis, text summarization, and visualization** using NLP techniques.  
It was developed as a **Hackathon Project** to analyze IMDB reviews and MCA E-Consultation comments.  

---

## ✨ Features  
- 📝 **Comment Input Options**:  
  - Manual input  
  - Pre-loaded IMDB dataset (`imdb_sample.csv`)  

- 🔍 **Sentiment Analysis**:  
  - Classifies comments into **Positive, Negative, Neutral** using **TextBlob**  

- 🧾 **Summarization**:  
  - Generates concise summaries using **HuggingFace Transformers (BART & DistilBART)**  

- ☁️ **Data Storage**:  
  - Saves analysis history into **MongoDB**  

- 📊 **Visualization**:  
  - Word Clouds  
  - Sentiment Distribution Pie Charts  
  - Interactive history filtering by date  

- 🎨 **Custom Styling**:  
  - Uses `style.css` for a polished UI  

---

## 🛠️ Tech Stack  
- **Frontend**: Streamlit + Custom CSS  
- **NLP**: TextBlob, HuggingFace Transformers, WordCloud, NLTK  
- **Database**: MongoDB  
- **Visualization**: Matplotlib  

---

## 📂 Project Structure  
├── Final_WeScrapIMDB.ipynb # Jupyter notebook for IMDB data scraping
├── Sentimental_Analysis.py # Main Streamlit app
├── imdb_sample.csv # Sample dataset
├── style.css # Custom styling
└── requirements.txt # Dependencies

---

## ⚡ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2️⃣ Create Virtual Environment (Recommended)  
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```
### 3️⃣ Install Dependencies
```1bash
pip install -r requirements.txt
```
### 4️⃣ Start MongoDB
```bash
MONGO_URI = "mongodb://localhost:27017"
```
### 5️⃣ Run the App
```bash
streamlit run Sentimental_Analysis.py
```
