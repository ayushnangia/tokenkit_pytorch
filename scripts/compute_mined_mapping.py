#!/usr/bin/env python
"""
Compute MinED Token Mapping Script (PyTorch Implementation)

This script computes a mapping between tokens in different tokenizers
based on minimum edit distance (MinED). The mapping is useful for
transferring knowledge between models with different tokenizers.

Example Usage:
    python scripts/compute_mined_mapping.py \
        teacher_tokenizer_name=google/gemma-2-2b-it:source=Gemma2 \
        target_tokenizer_name=Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2 \
        output='outputs/tokenizer_data/gemma2_to_qwen2_mined' \
        num_workers=8
"""

import json
from pathlib import Path
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from tokenkit.baseline_utils import compute_mined_mapping
from tokenkit.byteify import load_byteify_tokenizer


@hydra.main(config_path="../configs", config_name="compute_mined_mapping")
def main(args: DictConfig) -> None:
    """Main function.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load tokenizers
    tokenizer_teacher = load_byteify_tokenizer(args.teacher_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    print(f"Computing MinED mapping from {args.teacher_tokenizer_name} to {args.target_tokenizer_name}")
    print(f"Teacher vocabulary size: {len(tokenizer_teacher)}")
    print(f"Target vocabulary size: {len(target_tokenizer)}")

    # Compute mapping
    mined_mapping, mined_distances = compute_mined_mapping(
        tokenizer_teacher, target_tokenizer, 
        num_workers=args.num_workers
    )

    # Save mapping and distances
    np.save(output_dir / "mined_mapping.npy", mined_mapping)
    json.dump(
        mined_distances,
        open(output_dir / "mined_distances.json", "w"),
        indent=4,
    )
    
    # Compute statistics
    avg_distance = np.mean(list(mined_distances.values()))
    max_distance = max(mined_distances.values())
    zero_distance = sum(1 for d in mined_distances.values() if d == 0)
    
    print(f"Mapping statistics:")
    print(f"  Average edit distance: {avg_distance:.2f}")
    print(f"  Maximum edit distance: {max_distance}")
    print(f"  Exact matches: {zero_distance} ({zero_distance/len(mined_distances)*100:.1f}%)")
    print(f"Successfully saved mapping to {output_dir}")
    
    # For convenience, also save a summary
    summary = {
        "teacher_tokenizer": args.teacher_tokenizer_name,
        "target_tokenizer": args.target_tokenizer_name,
        "teacher_vocab_size": len(tokenizer_teacher),
        "target_vocab_size": len(target_tokenizer),
        "avg_edit_distance": float(avg_distance),
        "max_edit_distance": int(max_distance),
        "exact_matches": int(zero_distance),
        "exact_match_percent": float(zero_distance/len(mined_distances)*100)
    }
    
    json.dump(
        summary,
        open(output_dir / "mapping_summary.json", "w"),
        indent=4,
    )


if __name__ == "__main__":
    main()