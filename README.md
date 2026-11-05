# analystic-answer-analystics-
ANSLYTIC is an NLP-powered answer evaluation platform that uses Sentence Transformers, semantic similarity, and keyword analysis to assess student answers. It evaluates relevance, completeness, and semantic similarity, generates an overall score, and provides automated feedback through an interactive Gradio interface.

# ANSLYTIC – AI-Powered Answer Evaluation Platform

**Beyond Words. Into Meaning.**

ANSLYTIC is an NLP-based Question Answer Quality Analyzer that evaluates student answers using transformer-based sentence embeddings and semantic similarity.

## Project Overview

Traditional answer evaluation often depends on manual checking and exact keyword matching. ANSLYTIC explores an automated approach to evaluating answers by analyzing semantic similarity, relevance, and completeness.

## Key Features

- Automatic question matching using semantic similarity.
- Transformer-based sentence embeddings.
- Answer relevance evaluation.
- Completeness estimation using keyword coverage.
- Semantic similarity scoring.
- Weighted overall score.
- Automated feedback generation.
- Graphical evaluation report.
- Interactive Gradio web interface.

## Technologies Used

- Python
- Natural Language Processing
- Sentence Transformers
- MiniLM
- SQuAD Dataset
- Cosine Similarity
- Scikit-learn
- Gradio
- Matplotlib

## Dataset

The project uses the SQuAD (Stanford Question Answering Dataset).

Dataset: https://huggingface.co/datasets/rajpurkar/squad

## Model

sentence-transformers/all-MiniLM-L6-v2

The model converts sentences into dense vector representations, allowing semantic similarity comparisons.

## Evaluation Methodology

The system calculates:

1. Relevance Score – Measures how closely the answer relates to the question and reference information.
2. Completeness Score – Estimates coverage of important reference keywords.
3. Semantic Similarity Score – Measures semantic similarity between the student's answer and reference sentence.
4. Overall Score – Combines the three metrics using weighted scoring.

Overall Score:

30% Relevance + 30% Completeness + 40% Semantic Similarity

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ANSLYTIC.git
cd ANSLYTIC
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

The Gradio application will start locally.

## Future Enhancements

- Support for custom datasets.
- Advanced semantic completeness evaluation.
- Multiple language support.
- BERT-based and domain-specific model comparisons.
- Teacher dashboard and student performance tracking.
- Downloadable evaluation reports.

## Disclaimer

ANSLYTIC is an academic prototype. Its heuristic scores are estimates and are not validated substitutes for human grading.

## Author

Developed as an academic NLP project.
