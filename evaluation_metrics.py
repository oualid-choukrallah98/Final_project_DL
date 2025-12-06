"""
Evaluation Metrics for Medical Image Captioning
Includes: BLEU, CIDEr, METEOR, RadGraph F1
"""
import evaluate
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import numpy as np
from typing import List, Dict, Tuple
import re


def calculate_bleu(predictions: List[str], references: List[str]) -> Dict:
    """Calculate BLEU score"""
    bleu_metric = evaluate.load("bleu")
    # Format references as lists of lists
    references_formatted = [[ref] for ref in references]
    results = bleu_metric.compute(
        predictions=predictions,
        references=references_formatted,
        max_order=4
    )
    return results


def calculate_cider(predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
    """Calculate CIDEr score
    
    CIDEr expects raw strings (not tokenized). The library will handle tokenization internally.
    """
    # Format data into dictionaries
    gts = {}
    res = {}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        img_id = str(i)
        # CIDEr expects strings, not tokenized lists
        # The library will tokenize internally using its own method
        gts[img_id] = [ref]  # List of reference strings
        res[img_id] = [pred]  # List of prediction strings
    
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def calculate_meteor(predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
    """Calculate METEOR score
    
    METEOR expects raw strings. The library will handle tokenization internally.
    """
    # Format data into dictionaries
    gts = {}
    res = {}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        img_id = str(i)
        # METEOR expects strings, not tokenized lists
        gts[img_id] = [ref]  # List of reference strings
        res[img_id] = [pred]  # List of prediction strings
    
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def extract_entities(text: str) -> set:
    """
    Extract clinical entities from text (simplified version for RadGraph F1)
    This is a simplified implementation. For production, use the actual RadGraph model.
    """
    # Common medical entities patterns
    entities = set()
    
    # Extract anatomical structures
    anatomical = [
        'heart', 'lung', 'lungs', 'chest', 'mediastinum', 'pleura', 'pleural',
        'cardiac', 'pulmonary', 'aorta', 'spine', 'rib', 'ribs', 'diaphragm',
        'hilum', 'hila', 'cardiomediastinal', 'silhouette'
    ]
    
    # Extract findings
    findings = [
        'cardiomegaly', 'atelectasis', 'pneumonia', 'edema', 'effusion',
        'consolidation', 'pneumothorax', 'fibrosis', 'emphysema', 'opacity',
        'opacities', 'infiltrate', 'infiltrates', 'mass', 'nodule', 'nodules',
        'normal', 'abnormal', 'enlarged', 'clear', 'stable', 'acute', 'chronic'
    ]
    
    text_lower = text.lower()
    
    # Extract anatomical entities
    for term in anatomical:
        if term in text_lower:
            entities.add(f"ANATOMICAL:{term}")
    
    # Extract finding entities
    for term in findings:
        if term in text_lower:
            entities.add(f"FINDING:{term}")
    
    # Extract negation patterns
    negation_patterns = [
        r'no\s+(\w+)', r'without\s+(\w+)', r'absence\s+of\s+(\w+)',
        r'negative\s+for\s+(\w+)', r'free\s+of\s+(\w+)'
    ]
    
    for pattern in negation_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.add(f"NEGATION:{match}")
    
    return entities


def calculate_radgraph_f1(
    predictions: List[str], 
    references: List[str]
) -> Tuple[float, List[float]]:
    """
    Calculate RadGraph F1 score
    This is a simplified implementation. For production, use the actual RadGraph model.
    """
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_entities = extract_entities(pred)
        ref_entities = extract_entities(ref)
        
        if len(pred_entities) == 0 and len(ref_entities) == 0:
            f1 = 1.0
        elif len(pred_entities) == 0 or len(ref_entities) == 0:
            f1 = 0.0
        else:
            # Calculate precision, recall, F1
            intersection = pred_entities & ref_entities
            precision = len(intersection) / len(pred_entities) if len(pred_entities) > 0 else 0.0
            recall = len(intersection) / len(ref_entities) if len(ref_entities) > 0 else 0.0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_scores.append(f1)
    
    average_f1 = np.mean(f1_scores)
    return average_f1, f1_scores


def evaluate_all_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """Calculate all evaluation metrics"""
    results = {}
    
    # BLEU
    bleu_results = calculate_bleu(predictions, references)
    results['BLEU'] = bleu_results.get('bleu', 0.0)
    
    # CIDEr
    cider_score, _ = calculate_cider(predictions, references)
    results['CIDEr'] = float(cider_score)
    
    # METEOR
    meteor_score, _ = calculate_meteor(predictions, references)
    results['METEOR'] = float(meteor_score)
    
    # RadGraph F1
    radgraph_f1, _ = calculate_radgraph_f1(predictions, references)
    results['RadGraph F1'] = float(radgraph_f1)
    
    return results

