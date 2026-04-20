"""
Metric computation: ROUGE, BLEU, METEOR, token-level F1, BERTScore.

Token-level F1 is the standard SQuAD-style metric: overlap of
bag-of-words tokens between prediction and reference.

BERTScore tip: loading the transformer model takes ~10 seconds.
Always pass all predictions for a single model's run as one batch
via compute_bertscore_batch() to pay the load cost once.
"""

from typing import List, Optional

import nltk
from rouge_score import rouge_scorer
from .audio_quality import compute_audio_clarity

nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4",  quiet=True)


def _token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 (SQuAD-style).

    Tokenise by whitespace after lower-casing; measure overlap of the
    multisets (bags) of tokens between prediction and reference.
    Returns 0.0 when either string is empty.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Bag-of-words intersection
    pred_bag = {}
    for t in pred_tokens:
        pred_bag[t] = pred_bag.get(t, 0) + 1
    ref_bag = {}
    for t in ref_tokens:
        ref_bag[t] = ref_bag.get(t, 0) + 1

    common = sum(
        min(pred_bag.get(t, 0), ref_bag.get(t, 0)) for t in ref_bag
    )
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class Evaluator:
    def __init__(
        self,
        bertscore_lang: str = "en",
        bertscore_model: Optional[str] = None,
    ):
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self._bs_lang = bertscore_lang
        self._bs_model = bertscore_model  # None = auto-select per language

    def compute_rouge_bleu_f1(self, prediction: str, reference: str) -> dict:
        """Compute ROUGE-1/2/L (with P/R/F), BLEU, METEOR, token-level F1, and response length."""
        rouge = self._rouge.score(reference, prediction)

        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from nltk.translate.meteor_score import meteor_score

        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(prediction.lower())

        bleu = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=SmoothingFunction().method1,
        )

        meteor = meteor_score([ref_tokens], hyp_tokens) if ref_tokens and hyp_tokens else 0.0

        return {
            "rouge1":          rouge["rouge1"].fmeasure,
            "rouge1_p":        rouge["rouge1"].precision,
            "rouge1_r":        rouge["rouge1"].recall,
            "rouge2":          rouge["rouge2"].fmeasure,
            "rougeL":          rouge["rougeL"].fmeasure,
            "bleu":            bleu,
            "meteor":          meteor,
            "f1":              _token_f1(prediction, reference),
            "response_length": len(prediction.split()),
        }

    def compute_audio_quality(
        self,
        audio_path: str,
        whisper_confidence: Optional[float] = None,
    ) -> dict:
        """
        Compute audio clarity metrics for a single audio file.

        Delegates to audio_quality.compute_audio_clarity().
        Returns: {snr_db, speech_ratio, clarity_score}
        """
        return compute_audio_clarity(audio_path, whisper_confidence=whisper_confidence)

    def compute_bertscore_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> List[dict]:
        """
        Compute BERTScore for a batch of (prediction, reference) pairs.
        Pass all predictions for one model's run as a single batch.
        """
        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            raise ImportError(
                "bert_score is not installed. Install it with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install bert-score>=0.3.13\n"
                "Or uncheck 'Skip BERTScore' in the UI to skip it."
            ) from None

        P, R, F1 = bert_score_fn(
            predictions,
            references,
            lang=self._bs_lang,
            model_type=self._bs_model,
            verbose=False,
        )
        return [
            {
                "bertscore_p": p.item(),
                "bertscore_r": r.item(),
                "bertscore_f1": f.item(),
            }
            for p, r, f in zip(P, R, F1)
        ]
