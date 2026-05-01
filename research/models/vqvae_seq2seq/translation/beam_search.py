"""Beam search with CTC prefix scoring for decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BeamHypothesis:
    """Single hypothesis in beam search."""

    tokens: List[int]
    score: float
    ctc_score: float
    attn_score: float
    finished: bool = False

    def __lt__(self, other: "BeamHypothesis") -> bool:
        return self.score < other.score


class CTCPrefixScorer:
    """
    Computes CTC prefix scores for beam search.

    Used to rescore hypotheses with CTC alignment probabilities.
    """

    def __init__(
        self,
        ctc_log_probs: torch.Tensor,
        blank_idx: int = 0,
        eos_idx: int = 2,
    ):
        """
        Args:
            ctc_log_probs: (T, vocab_size) CTC log probabilities
            blank_idx: Index of blank token
            eos_idx: Index of EOS token
        """
        self.log_probs = ctc_log_probs
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.T = ctc_log_probs.shape[0]

        # Initialize prefix scores
        self.prefix_scores: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_prefix_score(
        self,
        prefix: Tuple[int, ...],
    ) -> float:
        """
        Compute CTC prefix score for a given prefix.

        Uses forward algorithm to compute probability of all alignments
        that result in the given prefix.
        """
        if len(prefix) == 0:
            return 0.0

        T = self.T
        prefix_tensor = torch.tensor(prefix, device=self.log_probs.device)
        L = len(prefix)

        # Alpha table: (T, 2*L+1)
        # States: blank, c1, blank, c2, blank, ..., cL, blank
        n_states = 2 * L + 1
        alpha = torch.full((T, n_states), float("-inf"), device=self.log_probs.device)

        # Initialize
        alpha[0, 0] = self.log_probs[0, self.blank_idx]
        if L > 0:
            alpha[0, 1] = self.log_probs[0, prefix[0]]

        # Forward pass
        for t in range(1, T):
            for s in range(n_states):
                if s % 2 == 0:  # Blank state
                    char_idx = self.blank_idx
                else:  # Character state
                    char_idx = prefix[s // 2]

                # Stay in current state
                alpha[t, s] = alpha[t - 1, s]

                # From previous state
                if s > 0:
                    alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t - 1, s - 1])

                # Skip blank for repeated characters
                if s > 1 and s % 2 == 1:
                    prev_char = prefix[(s - 2) // 2] if (s - 2) // 2 < L else -1
                    curr_char = prefix[s // 2]
                    if prev_char != curr_char:
                        alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t - 1, s - 2])

                alpha[t, s] = alpha[t, s] + self.log_probs[t, char_idx]

        # Final score: sum of last blank and last character
        final_score = torch.logaddexp(
            alpha[-1, -1], alpha[-1, -2] if n_states > 1 else alpha[-1, -1]
        )

        return final_score.item()

    def score_hypothesis(
        self,
        hypothesis: List[int],
    ) -> float:
        """
        Score a hypothesis using CTC prefix scoring.

        Args:
            hypothesis: List of token indices

        Returns:
            CTC prefix score
        """
        prefix = tuple(hypothesis)
        return self._get_prefix_score(prefix)


class BeamSearch:
    """
    Beam search decoder with CTC prefix scoring.

    Combines attention scores with CTC alignment probabilities.
    """

    def __init__(
        self,
        beam_size: int = 5,
        max_len: int = 50,
        eos_idx: int = 2,
        bos_idx: int = 1,
        pad_idx: int = 0,
        length_penalty: float = 0.6,
        ctc_weight: float = 0.4,
    ):
        self.beam_size = beam_size
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.length_penalty = length_penalty
        self.ctc_weight = ctc_weight

    def _length_normalize(self, score: float, length: int) -> float:
        """Apply length penalty."""
        return score / (length**self.length_penalty)

    def search(
        self,
        encoder_output: torch.Tensor,
        decoder: nn.Module,
        ctc_log_probs: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Perform beam search decoding.

        Args:
            encoder_output: (1, T, D) single encoder output
            decoder: Attention decoder module
            ctc_log_probs: Optional (T, vocab_size) CTC log probs for rescoring

        Returns:
            List of decoded sequences (best to worst)
        """
        device = encoder_output.device

        # Initialize CTC scorer
        ctc_scorer = None
        if ctc_log_probs is not None and self.ctc_weight > 0:
            ctc_scorer = CTCPrefixScorer(ctc_log_probs[0], self.pad_idx, self.eos_idx)

        # Initialize beams
        beams = [
            BeamHypothesis(
                tokens=[self.bos_idx],
                score=0.0,
                ctc_score=0.0,
                attn_score=0.0,
            )
        ]

        finished = []

        for step in range(self.max_len):
            all_candidates = []

            for beam in beams:
                if beam.finished:
                    finished.append(beam)
                    continue

                # Get next token probabilities
                input_ids = torch.tensor([beam.tokens], device=device)
                logits = decoder(input_ids, encoder_output)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)

                # Get top-k tokens
                topk_log_probs, topk_ids = log_probs.topk(self.beam_size * 2)

                for log_prob, token_id in zip(
                    topk_log_probs.tolist(), topk_ids.tolist()
                ):
                    new_tokens = beam.tokens + [token_id]
                    new_attn_score = beam.attn_score + log_prob

                    # CTC score
                    if ctc_scorer is not None:
                        # Score without BOS token
                        new_ctc_score = ctc_scorer.score_hypothesis(new_tokens[1:])
                    else:
                        new_ctc_score = 0.0

                    # Combined score
                    combined_score = (
                        1 - self.ctc_weight
                    ) * new_attn_score + self.ctc_weight * new_ctc_score

                    all_candidates.append(
                        BeamHypothesis(
                            tokens=new_tokens,
                            score=combined_score,
                            ctc_score=new_ctc_score,
                            attn_score=new_attn_score,
                            finished=(token_id == self.eos_idx),
                        )
                    )

            # Select top beams
            all_candidates.sort(
                key=lambda x: -self._length_normalize(x.score, len(x.tokens))
            )
            beams = all_candidates[: self.beam_size]

            # Check if all beams finished
            if all(beam.finished for beam in beams):
                break

        # Add remaining beams to finished
        finished.extend([b for b in beams if b.finished])
        if not finished:
            finished = beams

        # Sort by normalized score
        finished.sort(key=lambda x: -self._length_normalize(x.score, len(x.tokens)))

        # Return token sequences (without BOS/EOS)
        results = []
        for beam in finished[: self.beam_size]:
            tokens = beam.tokens[1:]  # Remove BOS
            if tokens and tokens[-1] == self.eos_idx:
                tokens = tokens[:-1]  # Remove EOS
            results.append(tokens)

        return results


class GreedyDecoder:
    """Simple greedy decoding (for fast inference)."""

    def __init__(
        self,
        max_len: int = 50,
        eos_idx: int = 2,
        bos_idx: int = 1,
    ):
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

    def decode(
        self,
        encoder_output: torch.Tensor,
        decoder: nn.Module,
    ) -> List[int]:
        """
        Greedy decode a single sequence.

        Args:
            encoder_output: (1, T, D) encoder output
            decoder: Attention decoder

        Returns:
            List of token indices
        """
        device = encoder_output.device
        tokens = [self.bos_idx]

        for _ in range(self.max_len):
            input_ids = torch.tensor([tokens], device=device)
            logits = decoder(input_ids, encoder_output)
            next_token = logits[0, -1].argmax().item()

            if next_token == self.eos_idx:
                break

            tokens.append(next_token)

        return tokens[1:]  # Remove BOS


class CTCGreedyDecoder:
    """Greedy CTC decoding."""

    def __init__(self, blank_idx: int = 0):
        self.blank_idx = blank_idx

    def decode(self, log_probs: torch.Tensor) -> List[int]:
        """
        Greedy decode CTC output.

        Args:
            log_probs: (T, vocab_size) log probabilities

        Returns:
            List of token indices (collapsed)
        """
        # Get best path
        best_path = log_probs.argmax(dim=-1).tolist()

        # Collapse repeated and remove blanks
        result = []
        prev_token = None

        for token in best_path:
            if token != self.blank_idx and token != prev_token:
                result.append(token)
            prev_token = token

        return result
