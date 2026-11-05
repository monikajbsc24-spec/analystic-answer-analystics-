
import re
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

APP_NAME = "ANSLYTIC"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = "rajpurkar/squad"
MAX_SAMPLES = 5000

print("Loading ANSLYTIC...")

# =========================================================
# 2. LOAD DATASET
# =========================================================

dataset = load_dataset(DATASET_NAME, split="validation")
dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

questions = []
answers = []
contexts = []

for item in dataset:
    if item["answers"]["text"]:
        questions.append(item["question"])
        answers.append(item["answers"]["text"][0])
        contexts.append(item["context"])

print("Dataset loaded:", len(questions), "records")


# =========================================================
# 3. LOAD TRANSFORMER MODEL
# =========================================================

model = SentenceTransformer(MODEL_NAME)

print("Sentence Transformer model loaded.")


# =========================================================
# 4. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()

    stopwords = {
        "the", "is", "are", "was", "were", "a", "an",
        "and", "or", "of", "to", "in", "on", "for",
        "with", "by", "as", "at", "it", "this", "that"
    }

    words = [word for word in words if word not in stopwords]

    return words


# =========================================================
# 5. GENERATE QUESTION EMBEDDINGS
# =========================================================

print("Generating question embeddings...")

question_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    show_progress_bar=True
)


# =========================================================
# 6. FIND MOST RELEVANT QUESTION
# =========================================================

def find_best_question(user_question):

    user_embedding = model.encode(
        [user_question],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )[0]

    best_index = int(np.argmax(similarities))

    return best_index, float(similarities[best_index])


# =========================================================
# 7. EXTRACT REFERENCE INFORMATION
# =========================================================

def extract_reference_sentence(context, expected_answer):

    sentences = re.split(r"(?<=[.!?])\s+", context)

    for sentence in sentences:
        if expected_answer.lower() in sentence.lower():
            return sentence

    return expected_answer


# =========================================================
# 8. ANSWER EVALUATION
# =========================================================

def evaluate_answer(student_answer, reference, question):

    student_answer = student_answer.strip()

    if not student_answer:
        return 0.0, 0.0, 0.0, 0.0

    # Semantic similarity
    student_embedding = model.encode([student_answer])
    reference_embedding = model.encode([reference])

    similarity_score = cosine_similarity(
        student_embedding,
        reference_embedding
    )[0][0]

    similarity_score = max(0, float(similarity_score)) * 100

    # Relevance score
    combined_text = question + " " + reference

    combined_embedding = model.encode([combined_text])

    relevance_score = cosine_similarity(
        student_embedding,
        combined_embedding
    )[0][0]

    relevance_score = max(0, float(relevance_score)) * 100

    # Completeness score using keyword coverage
    reference_words = set(preprocess_text(reference))
    student_words = set(preprocess_text(student_answer))

    if reference_words:
        completeness_score = (
            len(reference_words.intersection(student_words))
            / len(reference_words)
        ) * 100
    else:
        completeness_score = 0

    # Overall weighted score
    overall_score = (
        0.30 * relevance_score
        + 0.30 * completeness_score
        + 0.40 * similarity_score
    )

    return (
        round(relevance_score, 2),
        round(completeness_score, 2),
        round(similarity_score, 2),
        round(overall_score, 2)
    )


# =========================================================
# 9. GENERATE FEEDBACK
# =========================================================

def generate_feedback(score):

    if score >= 85:
        return "Excellent! Your answer is highly relevant and complete."

    elif score >= 70:
        return "Good work! Your answer covers most of the important concepts."

    elif score >= 50:
        return "Fair attempt. Add more relevant details and key concepts."

    elif score >= 30:
        return "Your answer needs improvement. Review the topic and include important points."

    else:
        return "Your answer needs significant improvement. Try writing a more relevant and complete response."


# =========================================================
# 10. EVALUATION REPORT
# =========================================================

def create_report(question, student_answer, reference, scores):

    relevance, completeness, similarity, overall = scores

    report = f"""
# ANSLYTIC - Answer Evaluation Report

### Question
{question}

### Student Answer
{student_answer}

### Reference Answer
{reference}

---

## Evaluation Results

| Evaluation Metric | Score |
|---|---:|
| Relevance | {relevance}% |
| Completeness | {completeness}% |
| Semantic Similarity | {similarity}% |
| Overall Score | {overall}% |

### Feedback

{generate_feedback(overall)}

---

### Evaluation Method

- Transformer-based sentence embeddings
- Cosine similarity
- Semantic relevance analysis
- Keyword coverage
- Weighted scoring mechanism

**Note:** Scores are heuristic estimates and should not be treated as validated academic grades.
"""

    return report


# =========================================================
# 11. VISUALIZATION
# =========================================================

def create_chart(scores):

    labels = [
        "Relevance",
        "Completeness",
        "Similarity",
        "Overall"
    ]

    values = list(scores)

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("ANSLYTIC - Answer Evaluation Results")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    return fig


# =========================================================
# 12. MAIN EVALUATION FUNCTION
# =========================================================

def analyze_answer(question, student_answer):

    if not question.strip():
        return "Please enter a question.", None

    if not student_answer.strip():
        return "Please enter your answer.", None

    # Find the closest dataset question
    best_index, match_score = find_best_question(question)

    matched_question = questions[best_index]
    expected_answer = answers[best_index]
    context = contexts[best_index]

    reference = extract_reference_sentence(
        context,
        expected_answer
    )

    # Evaluate answer
    scores = evaluate_answer(
        student_answer,
        reference,
        matched_question
    )

    report = create_report(
        matched_question,
        student_answer,
        reference,
        scores
    )

    chart = create_chart(scores)

    report += f"\n\n### Dataset Question Match\n{match_score * 100:.2f}%"

    return report, chart


# =========================================================
# 13. GRADIO INTERFACE
# =========================================================

custom_css = """
body {
    background: #101827;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

.description {
    text-align: center;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="ANSLYTIC - AI Answer Evaluator"
) as demo:

    gr.Markdown(
        """
        # ANSLYTIC
        ### Beyond Words. Into Meaning.

        An NLP-powered platform that evaluates student answers
        using transformer-based semantic analysis.
        """
    )

    gr.Markdown("## Answer Evaluation")

    question_input = gr.Textbox(
        label="Enter Your Question",
        placeholder="Example: What is Natural Language Processing?",
        lines=2
    )

    answer_input = gr.Textbox(
        label="Enter Your Answer",
        placeholder="Write your answer here...",
        lines=5
    )

    analyze_button = gr.Button(
        "Analyze My Answer",
        variant="primary"
    )

    report_output = gr.Markdown(
        label="Evaluation Report"
    )

    chart_output = gr.Plot(
        label="Performance Visualization"
    )

    analyze_button.click(
        fn=analyze_answer,
        inputs=[question_input, answer_input],
        outputs=[report_output, chart_output]
    )

    gr.Markdown(
        """
        ---
        **ANSLYTIC | NLP-Based Question Answer Quality Analyzer**

        Powered by Sentence Transformers and the SQuAD dataset.
        """
    )


# =========================================================
# 14. LAUNCH APPLICATION
# =========================================================

if __name__ == "__main__":
    demo.launch()
