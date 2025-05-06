import numpy as np
import pandas as pd
from collections import Counter
import re
import scipy.stats as stats
import cv2
from textstat import textstat

def calculate_bias_metrics(data, sensitive_cols, target_col):
    """
    Calculate bias metrics for classification dataset
    
    Parameters:
    -----------
    data: pandas.DataFrame
        Dataset to analyze
    sensitive_cols: list
        Column names of sensitive attributes
    target_col: str
        Column name of target variable
        
    Returns:
    --------
    dict
        Dictionary of bias metrics
    """
    metrics = {}
    
    for col in sensitive_cols:
        groups = data[col].unique()
        group_metrics = {}
        
        for group in groups:
            group_data = data[data[col] == group]
            
            # Calculate basic metrics
            if pd.api.types.is_numeric_dtype(data[target_col]):
                # For regression
                mean = group_data[target_col].mean()
                median = group_data[target_col].median()
                std = group_data[target_col].std()
                
                group_metrics[group] = {
                    "count": len(group_data),
                    "mean": mean,
                    "median": median,
                    "std": std
                }
            else:
                # For classification
                value_counts = group_data[target_col].value_counts(normalize=True)
                
                group_metrics[group] = {
                    "count": len(group_data),
                    "distribution": value_counts.to_dict()
                }
        
        # Calculate statistical disparities
        if pd.api.types.is_numeric_dtype(data[target_col]):
            # For regression: calculate disparity between groups
            group_means = {g: metrics["mean"] for g, metrics in group_metrics.items()}
            max_group = max(group_means, key=group_means.get)
            min_group = min(group_means, key=group_means.get)
            
            metrics[col] = {
                "group_metrics": group_metrics,
                "max_disparity": group_means[max_group] - group_means[min_group],
                "max_ratio": group_means[max_group] / group_means[min_group] if group_means[min_group] > 0 else float('inf'),
                "disparate_impact": group_means[min_group] / group_means[max_group] if group_means[max_group] > 0 else float('inf')
            }
        else:
            # For classification: calculate disparate impact for positive class
            positive_class = data[target_col].value_counts().index[0]  # Assume most common is positive
            group_rates = {g: m["distribution"].get(positive_class, 0) for g, m in group_metrics.items()}
            
            max_group = max(group_rates, key=group_rates.get)
            min_group = min(group_rates, key=group_rates.get)
            
            metrics[col] = {
                "group_metrics": group_metrics,
                "positive_class": positive_class,
                "max_disparity": group_rates[max_group] - group_rates[min_group],
                "max_ratio": group_rates[max_group] / group_rates[min_group] if group_rates[min_group] > 0 else float('inf'),
                "disparate_impact": group_rates[min_group] / group_rates[max_group] if group_rates[max_group] > 0 else float('inf')
            }
    
    return metrics

def detect_outliers(data, method="zscore", threshold=3.0):
    """
    Detect outliers in numeric data
    
    Parameters:
    -----------
    data: array-like
        Numeric data to analyze
    method: str
        Method to use ('zscore', 'iqr', or 'isolation_forest')
    threshold: float
        Threshold for outlier detection
        
    Returns:
    --------
    array
        Indices of outliers
    """
    if method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        return np.where(z_scores > threshold)[0]
    
    elif method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    
    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.05)
        preds = model.fit_predict(data.reshape(-1, 1))
        return np.where(preds == -1)[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")

def text_complexity_metrics(text):
    """
    Calculate complexity metrics for text data
    
    Parameters:
    -----------
    text: str
        Text to analyze
        
    Returns:
    --------
    dict
        Dictionary of complexity metrics
    """
    # Basic length metrics
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Average lengths
    avg_word_length = char_count / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Vocabulary richness
    unique_words = len(set(re.findall(r'\b\w+\b', text.lower())))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade
    }

def image_quality_metrics(image):
    """
    Calculate quality metrics for image data
    
    Parameters:
    -----------
    image: numpy.ndarray
        Image data to analyze
        
    Returns:
    --------
    dict
        Dictionary of quality metrics
    """
    # Ensure image is grayscale for some metrics
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Basic properties
    height, width = gray.shape[:2]
    aspect_ratio = width / height
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Contrast (standard deviation of pixel values)
    contrast = gray.std()
    
    # Brightness (mean pixel value)
    brightness = gray.mean()
    
    # Noise estimation (using median filter difference)
    median_filtered = cv2.medianBlur(gray, 3)
    noise_diff = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
    noise_level = noise_diff.mean()
    
    # Compression artifacts estimation
    # High frequency components after JPEG compression
    # This is a simplified approximation
    _, jpeg_data = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpeg_gray = cv2.imdecode(jpeg_data, 0)
    compression_diff = np.abs(gray.astype(np.float32) - jpeg_gray.astype(np.float32))
    compression_artifacts = compression_diff.mean()
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "noise_level": noise_level,
        "compression_artifacts": compression_artifacts
    }