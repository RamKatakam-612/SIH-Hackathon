# Install dependencies first:
# pip install -r requirements.txt

import streamlit as st
import matplotlib.pyplot as plt
import os
import os
import nltk
from wordcloud import WordCloud
from textblob import TextBlob
from transformers import pipeline
from datetime import datetime
import pymongo
import pandas as pd
import pathlib
import re
import string
from bs4 import BeautifulSoup

# -------------------------------
# MongoDB Connection
# -------------------------------
MONGO_URI = "mongodb://localhost:27017"   # replace with Atlas URI if cloud
client = pymongo.MongoClient(MONGO_URI)
db = client["sentiment_dashboard"]
collection = db["analyses"]

# -------------------------------
# Load CSS
# -------------------------------
css_path = pathlib.Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="MCA E-Consultation Sentiment Analyzer", layout="wide", page_icon="mca.png")
# -------------------------------
# Preprocessing function
# -------------------------------
def clean_review(text: str) -> str:
    """Preprocess review text: strip HTML, remove special chars, keep only clean text."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions (@user) and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Navbar
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown("""
<div style="background-color:#003366;padding:10px;border-radius:8px;">
    <a href="?page=Home" style="color:white;margin-right:20px;text-decoration:none;font-weight:bold;">🏠 Home</a>
    <a href="?page=Analysis" style="color:white;margin-right:20px;text-decoration:none;font-weight:bold;">📝 Analysis</a>
    <a href="?page=History" style="color:white;margin-right:20px;text-decoration:none;font-weight:bold;">📂 History</a>
    <a href="?page=About" style="color:white;text-decoration:none;font-weight:bold;">ℹ About</a>
</div>
""", unsafe_allow_html=True)

query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

# -------------------------------
# HOME
# -------------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div style="background-color:#e6f2ff;padding:20px;border-radius:10px;margin-top:20px;">
        <h2 style="color:#003366;">📊 AI-powered Comment Analysis</h2>
        <p>Welcome to the <b>MCA E-Consultation Sentiment Analyzer</b>.<br>
        This platform provides insights into public opinions expressed through the e-consultation 
        module of the Ministry of Corporate Affairs (MCA) website. By analyzing comments submitted by stakeholders, the tool classifies sentiments as positive, 
        neutral, or negative, helping policymakers, researchers, and businesses understand trends, concerns, and feedback in real-time.</p>
        <img src="https://cdn-icons-png.flaticon.com/512/3209/3209265.png" width="200">
    </div>
    """, unsafe_allow_html=True)
# Utility function to clean reviews (assumed defined somewhere)
def clean_review(text):
    return text.strip()

# -------------------------------
# ANALYSIS
# -------------------------------
if st.session_state.page == "Analysis":
    st.markdown("""
    <div style="background-color:#e6f2ff;padding:20px;border-radius:10px;margin-top:20px;">
        <h2 style="color:#003366;">📝 Run Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    analysis_mode = st.radio("Select Data Source:", ["Manual Input", "IMDB Data"])

    comments = []
    summaries = []
    date_input = datetime.today()

    if analysis_mode == "Manual Input":
        user_input = st.text_area("Paste comments (one per line):",
                                  "This amendment will simplify compliance and is very useful.\n"
                                  "The new rule increases burden on small firms, not acceptable.\n"
                                  "The change affects filing deadlines but overall seems fine.\n"
                                  "Great initiative, will reduce paperwork.\n"
                                  "Implementation details are unclear and confusing.")
        date_input = st.date_input("Select Date", datetime.today())
        comments = [clean_review(c) for c in user_input.split("\n") if c.strip()]
    elif analysis_mode == "IMDB Data":
        st.info("📥 Using IMDB scraped data from notebook...")

        try:
            import os
            base_path = os.path.dirname(__file__)
            csv_path = os.path.join(base_path, "imdb_sample.csv")
            imdb_df = pd.read_csv(csv_path)
            #imdb_df = pd.read_csv("imdb_sample.csv")

            if "date" not in imdb_df.columns:
                st.error("❌ 'date' column not found in IMDB dataset.")
            else:
                imdb_df["date"] = pd.to_datetime(imdb_df["date"], errors="coerce")
                imdb_df = imdb_df.dropna(subset=["date"])

                min_date = imdb_df["date"].min().date()
                max_date = imdb_df["date"].max().date()

                st.markdown("### 🔍 Filter by Date Range")

                # Use session_state to persist dates
                start_date = st.date_input(
                    "📅 Start Date",
                    min_value=min_date,
                    max_value=max_date,
                    value=st.session_state.get("start_date", min_date),
                    key="start_date"
                )

                end_date = st.date_input(
                    "📅 End Date",
                    min_value=min_date,
                    max_value=max_date,
                    value=st.session_state.get("end_date", max_date),
                    key="end_date"
                )

                if "comments" not in st.session_state:
                    st.session_state.comments = []

                if "date_filter_applied" not in st.session_state:
                    st.session_state.date_filter_applied = False

                if st.button("✅ Apply Filter"):
                    if start_date > end_date:
                        st.warning("⚠ Start date cannot be after end date.")
                        st.session_state.comments = []
                        st.session_state.date_filter_applied = False
                    else:
                        filtered_df = imdb_df[
                            (imdb_df["date"].dt.date >= start_date) &
                            (imdb_df["date"].dt.date <= end_date)
                        ]

                        st.markdown("### 🧾 Filtered Comments Preview")
                        st.dataframe(filtered_df[["text", "date"]].reset_index(drop=True), use_container_width=True)

                        raw_comments = filtered_df["text"].dropna().tolist()
                        st.session_state.comments = [clean_review(c) for c in raw_comments if len(c.split()) > 2]
                        st.session_state.date_filter_applied = True

                        if not filtered_df.empty:
                            date_input = filtered_df["date"].iloc[0].date()

                # Use filtered comments or show instruction
                if st.session_state.date_filter_applied:
                    comments = st.session_state.comments
                else:
                    st.info("👈 Select date range and click *Apply Filter* to load comments.")
                    comments = []

        except Exception as e:
            st.error(f"❌ Error loading or processing IMDB data: {e}")






        # -------------------------------
        # Sentiment Analysis
        # -------------------------------
    if st.button("🚀 Analyze") and comments:

        sentiments, pos, neg, neu = [], 0, 0, 0
        for c in comments:
            polarity = TextBlob(c).sentiment.polarity
            if polarity > 0:
                sentiments.append("Positive")
                pos += 1
            elif polarity < 0:
                sentiments.append("Negative")
                neg += 1
            else:
                sentiments.append("Neutral")
                neu += 1
                
        # -------------------------------
        # Sentiment Count Display
        # -------------------------------
        st.markdown(f"""
        <div style="background-color:#dff0d8;padding:10px;border-radius:6px;margin-top:20px;">
            <b>Sentiment Counts:</b>
            <span style="color:green;">Positive: {pos}</span>, 
            <span style="color:red;">Negative: {neg}</span>, 
            <span style="color:gray;">Neutral: {neu}</span>
        </div>
        """, unsafe_allow_html=True)
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import streamlit as st

        # ------------------------
        # Load Summarizers Once
        # ------------------------
        @st.cache_resource
        def load_summarizers():
            primary_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            primary_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            summarizer_primary = pipeline("summarization", model=primary_model, tokenizer=primary_tokenizer)

            fallback_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            fallback_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            summarizer_fallback = pipeline("summarization", model=fallback_model, tokenizer=fallback_tokenizer)

            return summarizer_primary, summarizer_fallback

        summarizer_primary, summarizer_fallback = load_summarizers()

        # ------------------------
        # Helper Functions
        # ------------------------
        def split_text_into_chunks(text, max_chunk_chars=900):
            """Split long text into smaller chunks based on sentence boundaries."""
            sentences = text.split('. ')
            chunks, current_chunk = [], ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_chars:
                    current_chunk += sentence + ". "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        def dynamic_summary_length(word_count):
            """Determine summary length bounds dynamically."""
            if word_count < 30:
                return None
            elif word_count < 50:
                return (20, 40)
            elif word_count < 100:
                return (40, 60)
            elif word_count < 250:
                return (60, 100)
            elif word_count < 500:
                return (100, 160)
            else:
                return (120, 200)

        def generate_summary(comment, primary=True, min_len=40, max_len=80):
            """Summarize given text using the specified model."""
            summarizer = summarizer_primary if primary else summarizer_fallback
            result = summarizer(comment, min_length=min_len, max_length=max_len, do_sample=False, truncation=True)
            return result[0]["summary_text"]

        # ------------------------
        # Summarization Loop
        # ------------------------

        summaries = []

        for comment in comments:
            comment = comment.strip()
            if not comment:
                summaries.append("[Empty comment]")
                continue

            try:
                word_count = len(comment.split())

                # Token count might fail if tokenizer is not exposed, catch it
                try:
                    token_count = summarizer_primary.tokenizer(comment, return_tensors="pt")["input_ids"].shape[1]
                except:
                    token_count = 0

                # Determine length range for summarization
                length_range = dynamic_summary_length(word_count)

                # Too short to summarize? Return as-is
                if length_range is None and token_count <= 1024:
                    summaries.append(comment)
                    continue

                # If short, assign a fallback range
                if length_range is None:
                    length_range = (20, 40)

                min_len, max_len = length_range

                if token_count <= 1024:
                    summary = generate_summary(comment, primary=True, min_len=min_len, max_len=max_len)
                    summaries.append(summary)
                else:
                    # Too long: split → summarize chunks → re-summarize
                    chunks = split_text_into_chunks(comment)
                    chunk_summaries = [
                        generate_summary(chunk, primary=True, min_len=40, max_len=80)
                        for chunk in chunks
                    ]
                    combined_summary = " ".join(chunk_summaries)
                    combined_word_count = len(combined_summary.split())
                    combined_range = dynamic_summary_length(combined_word_count) or (60, 100)
                    final_summary = generate_summary(combined_summary, primary=True,
                                                    min_len=combined_range[0], max_len=combined_range[1])
                    summaries.append(final_summary)

            except Exception as e1:
                st.warning(f"⚠ Primary summarizer failed: {e1}")
                try:
                    chunks = split_text_into_chunks(comment)
                    chunk_summaries = [
                        generate_summary(chunk, primary=False, min_len=40, max_len=80)
                        for chunk in chunks
                    ]
                    combined_summary = " ".join(chunk_summaries)
                    combined_word_count = len(combined_summary.split())
                    combined_range = dynamic_summary_length(combined_word_count) or (60, 100)
                    final_summary = generate_summary(combined_summary, primary=False,
                                                    min_len=combined_range[0], max_len=combined_range[1])
                    summaries.append(final_summary)

                except Exception as e2:
                    st.error(f"❌ Fallback summarizer also failed: {e2}")
                    summaries.append("[Error summarizing comment]")


        # -------------------------------
        # ✅ Display Summaries
        # -------------------------------
        st.markdown("### 📝 Summarized Comments")

        for i, (c, summ) in enumerate(zip(comments, summaries)):
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-bottom: 15px;">
                <div style="flex:1; max-height:150px; overflow-y:auto; background:#f0f0f0; padding:10px; border-radius:6px;">
                    <b>Comment #{i+1}:</b><br> {c}
                </div>
                <div style="flex:1; max-height:150px; overflow-y:auto; background:#e8f4ff; padding:10px; border-radius:6px;">
                    <b>Summary:</b><br> {summ}
                </div>
            </div>
            """, unsafe_allow_html=True)


        # -------------------------------
        # Word Cloud
        # -------------------------------
        all_text = " ".join(comments)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # -------------------------------
        # Pie Chart
        # -------------------------------
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie([pos, neu, neg],
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75
        )
        plt.setp(autotexts, size=10, weight="bold", color="white")
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        fig.gca().add_artist(centre_circle)
        ax.set_title("Overall Sentiment Distribution", fontsize=14, weight="bold")
        ax.legend(wedges, ["Positive", "Neutral", "Negative"], title="Sentiments",
                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig)

        # -------------------------------
        # Save to DB (use 'summaries' key, not 'summary')
        # -------------------------------
        doc = {
            "timestamp": date_input.strftime("%Y-%m-%d"),
            "comments": comments,
            "sentiments": sentiments,
            "summaries": summaries,
            "source": analysis_mode
        }
        collection.insert_one(doc)
        st.success("✅ Analysis saved!")
# -------------------------------
# HISTORY
# -------------------------------
elif st.session_state.page == "History":
    st.markdown("## 📂 Analysis History")

    # 🔴 DELETE OPTION
    with st.expander("🗑 Delete History", expanded=False):
        st.warning("This will permanently delete all saved analyses from the database.")
        if st.button("❌ Confirm Delete All History"):
            try:
                delete_result = collection.delete_many({})
                st.success(f"✅ Deleted {delete_result.deleted_count} record(s). Please refresh to see changes.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting history: {e}")

    # Fetch stored data
    data = list(collection.find({}, {"_id": 0}))  # exclude MongoDB _id field

    if not data:
        st.warning("No stored analyses found.")
    else:
        df = pd.DataFrame(data)

        flat_data = []
        for _, row in df.iterrows():
            comments = row.get('comments', [])
            sentiments = row.get('sentiments', [])
            summaries = row.get('summaries', [])
            timestamp = row.get('timestamp', None)

            if isinstance(comments, list) and isinstance(sentiments, list) and isinstance(summaries, list):
                if len(comments) == len(sentiments) == len(summaries):
                    for c, s, summ in zip(comments, sentiments, summaries):
                        flat_data.append({
                            "Date of Analysis": timestamp,
                            "Comment": c,
                            "Sentiment": s,
                            "Summary": summ
                        })

        df_flat = pd.DataFrame(flat_data)

        if df_flat.empty:
            st.warning("No valid analysis entries to display.")
        else:
            # Convert date column
            df_flat["Date of Analysis"] = pd.to_datetime(df_flat["Date of Analysis"], errors="coerce")
            df_flat = df_flat.dropna(subset=["Date of Analysis"])

            st.markdown("### 🔹 Filter by Date Range")

            default_start = df_flat["Date of Analysis"].min().date()
            default_end = df_flat["Date of Analysis"].max().date()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", default_start)
            with col2:
                end_date = st.date_input("End Date", default_end)

            # Filter
            df_filtered = df_flat[
                (df_flat["Date of Analysis"] >= pd.to_datetime(start_date)) &
                (df_flat["Date of Analysis"] <= pd.to_datetime(end_date))
            ]

            if df_filtered.empty:
                st.warning("No analyses found in the selected date range.")
            else:
                st.success(f"✅ {len(df_filtered)} comment(s) found in selected date range.")

                df_filtered = df_filtered.sort_values(by="Date of Analysis", ascending=False)
                df_filtered.insert(0, "No.", range(1, len(df_filtered) + 1))

                # Scrollable style for long fields
                st.markdown("""
                    <style>
                    .scrollable-cell {
                        max-height: 100px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }
                    table td {
                        vertical-align: top;
                    }
                    </style>
                """, unsafe_allow_html=True)

                df_filtered["Comment"] = df_filtered["Comment"].apply(
                    lambda x: f'<div class="scrollable-cell">{x}</div>'
                )
                df_filtered["Summary"] = df_filtered["Summary"].apply(
                    lambda x: f'<div class="scrollable-cell">{x}</div>'
                )

                st.markdown("### 🧾 Analysis Table")
                st.markdown(df_filtered.to_html(escape=False, index=False), unsafe_allow_html=True)

                # CSV download
                csv = df_filtered.drop(columns=["Comment", "Summary"]).to_csv(index=False).encode("utf-8")
                file_name = f"MCA_Comments_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"

                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=file_name,
                    mime="text/csv"
                )


# -------------------------------
# ABOUT
# -------------------------------
elif st.session_state.page == "About":
    st.markdown("""
    <div style="background-color:#e6f2ff;padding:20px;border-radius:10px;margin-top:20px;">
        <h2 style="color:#003366;">ℹ About This App</h2>
        <ul>
            <li>Built with <b>Streamlit</b></li>
            <li>NLP: <b>TextBlob, HuggingFace Transformers, WordCloud</b></li>
            <li>Storage: <b>MongoDB</b></li>
            <li>Styled with <b>custom HTML & CSS</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


