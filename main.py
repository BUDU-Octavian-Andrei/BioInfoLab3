#!/usr/bin/env python3
"""
Codon Frequency Analyzer
Reads a FASTA file (viral / bacterial genome) and counts all 64 codons
using a sliding window of exactly 3 nucleotides. Prints the top N codons.

Usage:
    python codon_freq.py genome.fasta            # sliding window, step 1
    python codon_freq.py genome.fna.gz           # gzipped FASTA works too
    python codon_freq.py genome.fasta --step 3   # in-frame (non-overlapping)
    python codon_freq.py genome.fasta --top 10 --all
    python codon_freq.py genome.fna.gz --ask "What host does this codon bias suggest?"
"""
import argparse
import gzip
import sys
from itertools import product

BASES = "ACGT"
WINDOW = 3  # window size is fixed at exactly 3 positions (one codon)


def read_fasta(path):
    """Return a list of (header, sequence) records from a FASTA file."""
    records, header, chunks = [], None, []
    opener = gzip.open if path.endswith(".gz") else open  # .fna.gz / .fasta.gz read directly
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.upper().replace("U", "T"))  # RNA viruses -> DNA alphabet
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def count_codons(sequence, step=1):
    """Slide a 3-nt window across the sequence and add one to the counter
    of the codon currently inside the window. Windows containing
    non-ACGT characters (N, R, Y, ...) are skipped."""
    counts = {"".join(c): 0 for c in product(BASES, repeat=3)}  # all 64 codons
    skipped = 0
    for i in range(0, len(sequence) - WINDOW + 1, step):
        codon = sequence[i:i + WINDOW]
        if codon in counts:
            counts[codon] += 1
        else:
            skipped += 1
    return counts, skipped


def report(counts, skipped, top, show_all):
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"Windows counted: {total:,}   (skipped ambiguous: {skipped:,})")
    print(f"\nTop {top} most frequent codons")
    print(f"{'Rank':>4}  {'Codon':<5}  {'Count':>10}  {'Freq (%)':>9}")
    print("-" * 34)
    for rank, (codon, n) in enumerate(ranked[:top], 1):
        pct = 100 * n / total if total else 0.0
        print(f"{rank:>4}  {codon:<5}  {n:>10,}  {pct:>9.3f}")
    if show_all:
        print("\nAll 64 codons")
        for codon, n in ranked:
            pct = 100 * n / total if total else 0.0
            print(f"  {codon}  {n:>10,}  {pct:>7.3f}%")


def main():
    p = argparse.ArgumentParser(description="Codon frequency analyzer for FASTA genomes")
    p.add_argument("fasta", help="Path to FASTA file")
    p.add_argument("--top", type=int, default=10, help="How many top codons to show (default 10)")
    p.add_argument("--step", type=int, default=1, choices=[1, 3],
                   help="Window step: 1 = sliding every position (default), 3 = reading frame")
    p.add_argument("--all", action="store_true", help="Also print all 64 codon counts")
    p.add_argument("--ask", nargs="?", const="", metavar="QUESTION",
                   help="Ask the AI layer a question about this genome. "
                        "With no text, uses USER_PROMPT in genome_ai.py")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama model (default $OLLAMA_MODEL or gemma4)")
    p.add_argument("--ollama-host", default=None, metavar="URL",
                   help="Ollama endpoint (default $OLLAMA_HOST or "
                        "http://localhost:11434; unix:///path/to.sock works)")
    p.add_argument("--no-retrieval", action="store_true",
                   help="Skip NCBI/PubMed/web lookups; use the counts alone")
    args = p.parse_args()

    try:
        records = read_fasta(args.fasta)
    except (OSError, ValueError) as e:
        sys.exit(f"Error: {e}")

    total_counts = {"".join(c): 0 for c in product(BASES, repeat=3)}
    total_skipped = 0
    print(f"File: {args.fasta}")
    for header, seq in records:
        print(f"  > {header[:70]}  ({len(seq):,} nt)")
        counts, skipped = count_codons(seq, args.step)  # count per record: no windows across boundaries
        for k, v in counts.items():
            total_counts[k] += v
        total_skipped += skipped
    print(f"Window size: {WINDOW}   Step: {args.step}\n")
    report(total_counts, total_skipped, args.top, args.all)

    if args.ask is not None:
        import genome_ai
        question = args.ask.strip() or genome_ai.USER_PROMPT
        print()
        genome_ai.run(records, total_counts, total_skipped, args.step,
                      question, top_n=args.top, model=args.model,
                      host=args.ollama_host, no_retrieval=args.no_retrieval)


if __name__ == "__main__":
    main()
