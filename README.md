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

```text
├── Final_WeScrapIMDB.ipynb   # Jupyter notebook for IMDB data scraping  
├── Sentimental_Analysis.py   # Main Streamlit app  
├── imdb_sample.csv           # Sample dataset  
├── style.css                 # Custom styling  
└── requirements.txt          # Dependencies  


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
---

### 📸 Screenshots 

**🔹 Home Page**

![WhatsApp Image 2025-09-14 at 18 03 20_5a290444](https://github.com/user-attachments/assets/ac909e3a-d257-499a-bae3-ebe5f7f85619)

**🔹 Analysis Page**

![WhatsApp Image 2025-09-14 at 18 04 58_195e0629](https://github.com/user-attachments/assets/882ac8c8-b2ea-4fdb-a817-4ea7fae71d6e)
![WhatsApp Image 2025-09-14 at 18 09 10_b7e98ee3](https://github.com/user-attachments/assets/21251036-9699-4f4e-9103-1326166c99d2)

**🔹 Visualizations**

![WhatsApp Image 2025-09-14 at 18 09 31_460bc88f](https://github.com/user-attachments/assets/f88ed3c5-31e2-468a-9c83-15eab9024f99)
![WhatsApp Image 2025-09-14 at 18 09 51_809adc95](https://github.com/user-attachments/assets/0a2e659a-22bf-45f5-a6b2-88355ce824df)

**🔹 History Page**

![WhatsApp Image 2025-09-14 at 18 10 18_e19a3a78](https://github.com/user-attachments/assets/d647ff9c-b9bd-4762-8054-eaddcfe3adff)

### 👩‍💻 Team / Authors
**Team Name: TEAM MUFASA**

**Hackathon Team Members:**  
1. K.P.N.S.Rama Karthik  
2. J. Venkata Shailesh  
3. R. Sathvika  
4. A. Nikhitha  
5. B. Thrisha  
6. A. Hema Sri Satya Latha
   
## 🚀 Live Demo

Check out the deployed app here: [sih-hackathon-mufasa.streamlit.app](https://sih-hackathon-mufasa.streamlit.app)

