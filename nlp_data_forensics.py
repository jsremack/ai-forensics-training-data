import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import hashlib
import json
from datetime import datetime
from transformers import pipeline
from common.utils import document_analysis
from common.metrics import text_complexity_metrics

# Load NLP resources
nlp = spacy.load("en_core_web_md")
synthetic_text_detector = pipeline("text-classification", model="roberta-base-openai-detector")

# 1. Text corpus analysis
def analyze_text_corpus(texts):
    """Analyze basic properties of the text corpus"""
    corpus_metrics = {
        "document_count": len(texts),
        "total_tokens": 0,
        "vocabulary_size": 0,
        "avg_document_length": 0,
        "language_statistics": {}
    }
    
    # Token and vocabulary analysis
    all_tokens = []
    doc_lengths = []
    languages = []
    
    for text in texts:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        all_tokens.extend(tokens)
        doc_lengths.append(len(tokens))
        
        # Detect language
        # ... implementation for language detection
    
    vocabulary = set(all_tokens)
    corpus_metrics["total_tokens"] = len(all_tokens)
    corpus_metrics["vocabulary_size"] = len(vocabulary)
    corpus_metrics["avg_document_length"] = np.mean(doc_lengths)
    
    # Token frequency distribution
    token_counts = Counter(all_tokens)
    corpus_metrics["top_100_tokens"] = token_counts.most_common(100)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.hist(doc_lengths, bins=50)
    plt.title("Distribution of Document Lengths")
    plt.xlabel("Tokens per Document")
    plt.ylabel("Count")
    plt.savefig("document_length_distribution.png")
    
    return corpus_metrics

# 2. Synthetic text detection
def detect_synthetic_content(texts, threshold=0.8):
    """Detect potentially synthetic or AI-generated content"""
    synthetic_analysis = {
        "potentially_synthetic": [],
        "natural": [],
        "confidence_scores": [],
        "statistical_markers": {}
    }
    
    for i, text in enumerate(texts):
        # Use model-based detection
        result = synthetic_text_detector(text)
        score = result[0]["score"] if result[0]["label"] == "LABEL_1" else 1 - result[0]["score"]
        synthetic_analysis["confidence_scores"].append(score)
        
        if score > threshold:
            synthetic_analysis["potentially_synthetic"].append(i)
        else:
            synthetic_analysis["natural"].append(i)
        
        # Statistical markers for synthetic text
        # ... implementation for statistical markers
    
    # Analyze patterns in potentially synthetic content
    # ... implementation for pattern analysis
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.hist(synthetic_analysis["confidence_scores"], bins=20)
    plt.title("Distribution of Synthetic Content Confidence Scores")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.axvline(x=threshold, color='r', linestyle='--')
    plt.savefig("synthetic_content_distribution.png")
    
    return synthetic_analysis

# 3. Text bias analysis
def analyze_text_biases(texts, metadata=None):
    """Analyze potential biases in text corpus"""
    bias_analysis = {
        "entity_representation": {},
        "sentiment_bias": {},
        "topic_distribution": {},
        "word_association_biases": {}
    }
    
    # Entity representation analysis
    entity_counts = Counter()
    gender_terms = {
        "male": ["he", "him", "his", "himself", "man", "men", "boy", "boys", "male", "males"],
        "female": ["she", "her", "hers", "herself", "woman", "women", "girl", "girls", "female", "females"]
    }
    gender_counts = {"male": 0, "female": 0}
    
    for text in texts:
        doc = nlp(text)
        
        # Count named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]:
                entity_counts[f"{ent.label_}:{ent.text}"] += 1
        
        # Count gender terms
        text_lower = text.lower()
        for gender, terms in gender_terms.items():
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                gender_counts[gender] += len(re.findall(pattern, text_lower))
    
    bias_analysis["entity_representation"]["top_entities"] = entity_counts.most_common(100)
    bias_analysis["entity_representation"]["gender_distribution"] = gender_counts
    
    # Sentiment analysis by demographic references
    # ... implementation for sentiment bias analysis
    
    # Topic modeling and distribution
    # ... implementation for topic analysis
    
    # Word association bias detection
    # ... implementation for association bias
    
    # Create visualizations
    gender_df = pd.DataFrame([gender_counts])
    plt.figure(figsize=(8, 6))
    sns.barplot(data=gender_df)
    plt.title("Gender Term Distribution")
    plt.ylabel("Count")
    plt.savefig("gender_distribution.png")
    
    return bias_analysis

# 4. Source and lineage analysis
def analyze_text_lineage(texts, metadata=None):
    """Analyze potential sources and lineage of text content"""
    lineage_analysis = {
        "source_identification": {},
        "content_clustering": {},
        "duplication_analysis": {},
        "stylometric_analysis": {}
    }
    
    # Text fingerprinting and hashing
    text_hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    lineage_analysis["source_identification"]["text_hashes"] = text_hashes
    
    # Detect duplicates and near-duplicates
    # ... implementation for duplication detection
    
    # Content clustering
    # ... implementation for content clustering
    
    # Stylometric analysis for authorship
    # ... implementation for stylometric analysis
    
    return lineage_analysis

# 5. Quality and consistency assessment
def assess_text_quality(texts):
    """Assess quality of text data for training purposes"""
    quality_assessment = {
        "complexity_metrics": [],
        "language_quality": [],
        "consistency_issues": [],
        "anomaly_detection": {}
    }
    
    # Calculate text complexity metrics
    for text in texts:
        metrics = text_complexity_metrics(text)
        quality_assessment["complexity_metrics"].append(metrics)
    
    # Language quality assessment
    # ... implementation for language quality
    
    # Consistency analysis
    # ... implementation for consistency checks
    
    # Anomaly detection
    # ... implementation for anomaly detection
    
    # Create quality distribution visualization
    complexity_df = pd.DataFrame(quality_assessment["complexity_metrics"])
    plt.figure(figsize=(12, 8))
    sns.pairplot(complexity_df[["flesch_reading_ease", "avg_word_length", "lexical_diversity"]])
    plt.savefig("text_quality_distributions.png")
    
    return quality_assessment

# Main analysis workflow
def main():
    # Load dataset
    df = pd.read_csv("datasets/synthetic/nlp_training_data.csv")
    texts = df["text"].tolist()
    
    # Run analyses
    corpus_metrics = analyze_text_corpus(texts)
    synthetic_analysis = detect_synthetic_content(texts)
    bias_analysis = analyze_text_biases(texts, metadata=df)
    lineage_analysis = analyze_text_lineage(texts, metadata=df)
    quality_assessment = assess_text_quality(texts)
    
    # Compile comprehensive report
    forensic_report = {
        "corpus_metrics": corpus_metrics,
        "synthetic_content_analysis": synthetic_analysis,
        "bias_analysis": bias_analysis,
        "lineage_analysis": lineage_analysis,
        "quality_assessment": quality_assessment,
        "timestamp": datetime.now().isoformat(),
        "analysis_version": "1.0.0"
    }
    
    # Save report
    with open("nlp_data_forensic_report.json", "w") as f:
        json.dump(forensic_report, f, indent=2)
    
    # Generate visualization summary
    # ... implementation for report visualization

if __name__ == "__main__":
    main()