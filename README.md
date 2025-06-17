# Controllable Playlist Generator

**Python • pandas • SentenceTransformers • PyTorch • scikit-learn**

## 1. Introduction

This project implements a content-based music recommendation system that generates personalized playlists from natural language prompts by fusing semantic text embeddings with acoustic and categorical features for transparent, controllable retrieval :contentReference[oaicite:12]{index=12}.

## 2. Data

- **Source:** SpotGen dataset from Kaggle (~90,000 tracks with metadata, lyrics, and audio features) :contentReference[oaicite:13]{index=13}.
- **Features:**  
  - **Categorical:** genre, language, release year, popularity  
  - **Text:** lyrics and generated descriptive sentences  
  - **Numerical:** 14 audio features (tempo, energy, loudness, etc.; min–max scaled)

## 3. Preprocessing

1. **Table Merging & Cleaning:** merged track, artist, lyrics, and album tables; corrected misencoded lyrics and dropped irrelevant columns :contentReference[oaicite:14]{index=14}.  
2. **Language Annotation:** applied `papluca/xlm-roberta-base-language-detection` for initial labels, with manual correction for English-heavy misclassifications :contentReference[oaicite:15]{index=15}.  
3. **Feature Scaling & Bucketing:** min–max scaled each audio feature to [0,1] and defined five buckets (“very low”–“very high”) based on empirical distributions :contentReference[oaicite:16]{index=16}.  
4. **Descriptive Sentence Construction:** concatenated bucket labels with metadata (title, artist, genre, lyrics) to form natural-language summaries for SBERT encoding :contentReference[oaicite:17]{index=17}.

## 4. Architecture

- **Text Branch:** Encodes descriptive sentences with the all-MiniLM-L6-v2 SentenceTransformer into 384-dim vectors :contentReference[oaicite:18]{index=18}.  
- **Numeric Branch:** Projects 14-dim audio feature vectors through a 2-layer MLP followed by L2 normalization :contentReference[oaicite:19]{index=19}.  
- **Fusion Layer:** Concatenates text and numeric embeddings, applies Linear→ReLU→LayerNorm→normalize.  
- **Retrieval:** Ranks songs by cosine similarity between prompt and song embeddings in the shared space.

## 5. Retrieval Pipeline Versions

- **v0:** Semantic-only retrieval (cosine similarity).  
- **v1:** + Explicit genre filtering.  
- **v2:** + Popularity & release-year pre-filters with weighted scoring.  
- **v3:** + Contrastive fine-tuning on prompt–song pairs from curated playlists.  
- **v4:** + Language-aware matching.  
- **v5:** + Prompt-driven audio feature boosting.  
- **v6:** + Fine-tuning on ~98K synthetic playlists for improved semantic alignment :contentReference[oaicite:20]{index=20}.

## 6. Results

- **User Study:** 57 participants; 84.2% preferred the final model over the baseline (15.8%) :contentReference[oaicite:21]{index=21}.  
- **Qualitative Improvements:** Synthetic playlist fine-tuning significantly enhanced alignment with desired genres, languages, and audio characteristics :contentReference[oaicite:22]{index=22}.
