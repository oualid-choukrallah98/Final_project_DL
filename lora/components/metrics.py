from pycocoevalcap.cider.cider import Cider
import evaluate
from radgraph import F1RadGraph


def calculate_cider(generated_captions, ground_truth_captions):
    """
    generated_captions: List of strings
    ground_truth_captions: List of strings
    """
    
    # 1. Format data into dictionaries expected by the library
    # Structure: {image_id: [caption_string]}
    gts = {}
    res = {}
    
    for i, (gen, true) in enumerate(zip(generated_captions, ground_truth_captions)):
        img_id = str(i) # Create a dummy ID for the image
        
        # Ground Truth (Reference)
        # Note: In standard datasets, one image might have 5 captions.
        # In medical data, it usually has 1. We wrap it in a list [].
        gts[img_id] = [true] 
        
        # Generated (Hypothesis)
        res[img_id] = [gen]

    # 2. Initialize Scorer
    scorer = Cider()
    
    # 3. Compute Score
    # score: The average CIDEr score for the whole dataset
    # scores: A list of CIDEr scores for each individual image
    score, _ = scorer.compute_score(gts, res)
    
    return score


def calculate_bleu(generated_captions, ground_truth_captions):
    bleu_metric = evaluate.load("bleu")
    results = bleu_metric.compute(predictions=generated_captions, references=ground_truth_captions, max_order=4)
    return results['bleu']


def calculate_meteor(generated_captions, ground_truth_captions):
    meteor_metric = evaluate.load('meteor')
    meteor_results = meteor_metric.compute(predictions=generated_captions, references=ground_truth_captions)
    return meteor_results['meteor']


def calculate_radgraph_f1(generated_captions, ground_truth_captions):
    radgraph_scorer = F1RadGraph(reward_level="partial")
    preds_list = generated_captions.tolist()
    refs_list = ground_truth_captions.tolist()
    result = radgraph_scorer(hyps=preds_list, refs=refs_list)
    return result[0]
