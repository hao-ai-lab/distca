"""
Token and document monitoring utilities for distributed training.

This module provides tools to inspect what tokens and documents are being
processed during training iterations.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from .logging import get_logger

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TokenMonitorConfig:
    """Configuration for token monitoring."""
    enabled: bool = True
    max_tokens_to_decode: int = 200  # Max tokens to decode per sample for text preview
    max_samples_to_log: int = 2  # Max samples to log detailed info per batch
    max_docs_to_preview: int = 10  # Max documents to show previews for
    doc_preview_chars: int = 50  # Characters to show per document preview


# Global config instance
_config = TokenMonitorConfig()

# Mutable counter for tracking iterations
_iteration_counter = [0]


def get_token_monitor_config() -> TokenMonitorConfig:
    """Get the current token monitor configuration."""
    return _config


def set_token_monitor_config(
    enabled: Optional[bool] = None,
    max_tokens_to_decode: Optional[int] = None,
    max_samples_to_log: Optional[int] = None,
    max_docs_to_preview: Optional[int] = None,
    doc_preview_chars: Optional[int] = None,
) -> None:
    """Update token monitor configuration."""
    global _config
    if enabled is not None:
        _config.enabled = enabled
    if max_tokens_to_decode is not None:
        _config.max_tokens_to_decode = max_tokens_to_decode
    if max_samples_to_log is not None:
        _config.max_samples_to_log = max_samples_to_log
    if max_docs_to_preview is not None:
        _config.max_docs_to_preview = max_docs_to_preview
    if doc_preview_chars is not None:
        _config.doc_preview_chars = doc_preview_chars


def reset_iteration_counter() -> None:
    """Reset the iteration counter to 0."""
    _iteration_counter[0] = 0


def get_iteration_count() -> int:
    """Get the current iteration count."""
    return _iteration_counter[0]


def increment_iteration_counter() -> int:
    """Increment and return the new iteration count."""
    _iteration_counter[0] += 1
    return _iteration_counter[0]


# =============================================================================
# Document Analysis
# =============================================================================

@dataclass
class DocumentInfo:
    """Information about documents within a token sequence."""
    num_documents: int
    tokens_per_doc: List[int]
    eod_positions: List[int]


def analyze_documents_in_sequence(token_ids: list, eod_token_id: int) -> DocumentInfo:
    """Analyze document structure within a token sequence.
    
    Documents are separated by EOD (end-of-document) tokens. This function
    finds document boundaries and calculates per-document token counts.
    
    Args:
        token_ids: List of token IDs
        eod_token_id: The end-of-document token ID
        
    Returns:
        DocumentInfo with num_documents, tokens_per_doc, and eod_positions
    """
    eod_positions = [i for i, tok in enumerate(token_ids) if tok == eod_token_id]
    
    if not eod_positions:
        # No EOD found - treat entire sequence as one document
        return DocumentInfo(
            num_documents=1,
            tokens_per_doc=[len(token_ids)],
            eod_positions=[],
        )
    
    # Calculate tokens per document
    # Documents are: [0:eod_0+1], [eod_0+1:eod_1+1], ...
    tokens_per_doc = []
    prev_end = 0
    for eod_pos in eod_positions:
        doc_length = eod_pos - prev_end + 1  # Include the EOD token
        tokens_per_doc.append(doc_length)
        prev_end = eod_pos + 1
    
    # If there are tokens after the last EOD, count them as a partial document
    if prev_end < len(token_ids):
        tokens_per_doc.append(len(token_ids) - prev_end)
    
    num_docs = len(eod_positions) + (1 if prev_end < len(token_ids) else 0)
    
    return DocumentInfo(
        num_documents=num_docs,
        tokens_per_doc=tokens_per_doc,
        eod_positions=eod_positions,
    )


# =============================================================================
# Token Logging
# =============================================================================

def log_tokens_text(
    tokens: torch.Tensor, 
    tokenizer, 
    iteration: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[TokenMonitorConfig] = None,
) -> None:
    """Log decoded text from token IDs for monitoring.
    
    Logs document structure, token previews, and text for each sample in the batch.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_length]
        tokenizer: The tokenizer to decode tokens (must have .eod and .detokenize())
        iteration: Current training iteration. If None, uses internal counter.
        logger: Logger to use. If None, uses the module logger.
        config: TokenMonitorConfig. If None, uses global config.
    """
    if tokens is None:
        return
    
    cfg = config or _config
    if not cfg.enabled:
        return
    
    log = logger or get_logger()
    iter_num = iteration if iteration is not None else _iteration_counter[0]
    
    batch_size = tokens.shape[0]
    seq_length = tokens.shape[1] if len(tokens.shape) > 1 else len(tokens)
    eod_token_id = tokenizer.eod
    
    log.info(f"=== Iteration {iter_num} Token Monitor (batch_size={batch_size}, seq_len={seq_length}, eod_id={eod_token_id}) ===")
    
    # Aggregate document stats across all samples in batch
    total_docs = 0
    all_doc_lengths = []
    
    for sample_idx in range(batch_size):
        sample_tokens = tokens[sample_idx] if len(tokens.shape) > 1 else tokens
        full_sample_tokens = sample_tokens.tolist()
        
        # Analyze document structure for the full sequence
        doc_info = analyze_documents_in_sequence(full_sample_tokens, eod_token_id)
        total_docs += doc_info.num_documents
        all_doc_lengths.extend(doc_info.tokens_per_doc)
        
        # Only log detailed text for first few samples
        if sample_idx < cfg.max_samples_to_log:
            truncated_tokens = full_sample_tokens[:cfg.max_tokens_to_decode]
            
            try:
                # Decode tokens to text
                decoded_text = tokenizer.detokenize(truncated_tokens)
                # Truncate if too long and add ellipsis
                if len(decoded_text) > 500:
                    decoded_text = decoded_text[:500] + "..."
                
                log.info(f"  Sample {sample_idx}:")
                log.info(f"    Documents in sample: {doc_info.num_documents}")
                log.info(f"    Tokens per doc: {doc_info.tokens_per_doc[:10]}{'...' if len(doc_info.tokens_per_doc) > 10 else ''}")
                log.info(f"    EOD positions: {doc_info.eod_positions[:10]}{'...' if len(doc_info.eod_positions) > 10 else ''}")
                log.info(f"    Token IDs (first 20): {truncated_tokens[:20]}{'...' if len(truncated_tokens) > 20 else ''}")
                log.info(f"    Text (first {len(truncated_tokens)} tokens): {repr(decoded_text)}")
                
                # Print first N chars of each document for identification
                doc_starts = [0] + [pos + 1 for pos in doc_info.eod_positions if pos + 1 < len(full_sample_tokens)]
                log.info(f"    --- Document Previews (first {cfg.doc_preview_chars} chars each) ---")
                for doc_idx, start_pos in enumerate(doc_starts[:cfg.max_docs_to_preview]):
                    # Extract ~50 tokens from doc start (enough to get 50+ chars usually)
                    doc_tokens = full_sample_tokens[start_pos:start_pos + 50]
                    try:
                        doc_text = tokenizer.detokenize(doc_tokens)
                        doc_preview = doc_text[:cfg.doc_preview_chars].replace('\n', '\\n')
                        log.info(f"      Doc {doc_idx} (@pos {start_pos}): {repr(doc_preview)}...")
                    except Exception:
                        log.info(f"      Doc {doc_idx} (@pos {start_pos}): [decode failed]")
                if len(doc_starts) > cfg.max_docs_to_preview:
                    log.info(f"      ... and {len(doc_starts) - cfg.max_docs_to_preview} more documents")
            except Exception as e:
                log.warning(f"  Sample {sample_idx}: Failed to decode tokens: {e}")
    
    # Summary stats for entire microbatch
    if all_doc_lengths:
        avg_doc_len = sum(all_doc_lengths) / len(all_doc_lengths)
        min_doc_len = min(all_doc_lengths)
        max_doc_len = max(all_doc_lengths)
        log.info(f"  --- Microbatch Summary ---")
        log.info(f"    Total documents in microbatch: {total_docs}")
        log.info(f"    Avg tokens/doc: {avg_doc_len:.1f}, Min: {min_doc_len}, Max: {max_doc_len}")
    
    log.info("=" * 60)


def monitor_batch_tokens(
    tokens: torch.Tensor,
    tokenizer,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Convenience function to monitor tokens in a batch.
    
    Automatically increments the iteration counter and logs token info.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_length]
        tokenizer: The tokenizer to decode tokens
        logger: Logger to use. If None, uses the module logger.
    """
    if not _config.enabled:
        return
    
    iteration = increment_iteration_counter()
    log_tokens_text(tokens, tokenizer, iteration=iteration, logger=logger)

