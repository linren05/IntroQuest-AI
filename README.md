🌟 IntroQuest AI — Premium Transcript Evaluation System

A next-generation scoring engine combining rule-based logic + NLP semantic intelligence.

🚀 Overview

IntroQuest AI is an advanced transcript-evaluation platform designed to assess spoken or written introductions with high precision.
The system combines rule-based scoring, semantic similarity using transformer embeddings, and rubric-driven weighting to generate a professional, actionable evaluation.

Users can paste a transcript or upload a text file, attach a rubric Excel file, and instantly receive a detailed breakdown with analytics, visualizations, performance badges, and improvement suggestions.

This AI-powered evaluator is tailored for interviews, pitch training, communication skill assessment, and academic use.

✨ Key Features
🔍 1. Multi-Layered Scoring Engine

Rule-based checks: keyword detection, word-count validation, exact matches.

NLP model scoring: semantic similarity via Sentence-Transformers.

Weighted aggregation: final score normalised to a 0–100 scale.

📊 2. Detailed Evaluation Dashboard

Overall performance score

Radar chart visualization

Per-criterion scoring cards

Keyword match statistics

Word count and length adherence

Automatic achievement badges

📁 3. Flexible Rubric Input

Upload your own rubric as .xlsx

Auto-parser detects messy or non-standard formats

Falls back to a default rubric if needed

🎨 4. Ultra-Premium UI

Centered layout

Bright theme option

Glassmorphism design

Gradient headers

Minimal top-aligned interface

Modern score-circle animation

📥 5. Export & Sharing

Download results as CSV

Streamlit-based deployment (local or cloud)

Optimized for GitHub sharing and public access

🛠 Tech Stack

Streamlit (UI)

Python

Sentence-Transformers

Torch

Plotly

Pandas / OpenPyXL

📦 Installation
git clone https://github.com/linren05/introquest-ai.git
cd introquest-ai
pip install -r requirements.txt
streamlit run introquest_app.py

📘 How to Use

Launch the app

Upload rubric (optional — default included)

Paste your transcript

Click Score Transcript

Review the visual dashboard & download results

📝 License

MIT License — free to use, modify, and extend.

🤝 Contributions

PRs are welcome! If you’d like to improve UI, rubric parsing, or scoring logic, feel free to contribute.
