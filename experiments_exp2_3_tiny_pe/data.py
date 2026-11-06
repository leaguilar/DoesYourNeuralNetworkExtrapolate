"""CDS-vs-nonCDS read dataset, tokenised for DNABERT-6.

Generates ReadSeeker-style synthetic reads:
  - **CDS reads** (label=1): a read of length L_bp whose center falls
    inside a plus-strand CDS segment annotated in the supplied GFF3.
  - **nonCDS reads** (label=0): a read of length L_bp whose entire span
    lies outside any CDS segment.

Reads are tokenised with the DNABERT-6 6-mer tokenizer (stride 1), so an
L_bp read maps to (L_bp - 5) k-mer tokens plus [CLS] and [SEP] for a
total of (L_bp - 3) input ids.

Chromosome splits (default, yeast):
  train     -> chr I..IX
  val       -> chr X
  test_id   -> chr XI         (in-distribution: same chroms style as train)
  ood_chrom -> chr XIII..XVI  (held-out chromosomes)

Length-OOD is implemented by *constructing the dataset at a different
read length*; the same chromosome split can be sampled at multiple lengths
to give matched ID/OOD comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_fasta(path: str | Path) -> dict[str, bytes]:
    seqs: dict[str, list[bytes]] = {}
    name: str | None = None
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                name = line[1:].split()[0].decode()
                seqs[name] = []
            else:
                if name is None:
                    continue
                seqs[name].append(line.strip())
    return {n: b"".join(parts) for n, parts in seqs.items()}


def parse_gff3_cds(gff_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Plus-strand CDS segments per chromosome, 0-based half-open [start, end).

    Kept for backward compatibility. New code should call
    `parse_gff3_cds_both_strands` and use any-strand labels (the field
    standard for gene prediction).
    """
    both = parse_gff3_cds_both_strands(gff_path)
    return {chrom: segs["+"] for chrom, segs in both.items()
            if segs["+"]}


def parse_gff3_cds_both_strands(
        gff_path: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """CDS segments per chromosome × strand, 0-based half-open [start, end).

    Returns `{chrom: {'+': [...], '-': [...]}}`. Plus and minus strand are
    populated independently; chromosomes with no CDS on either strand are
    omitted.
    """
    cds: dict[str, dict[str, list[tuple[int, int]]]] = {}
    with open(gff_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _, ftype, start_s, end_s, _, strand, _, _ = parts
            if ftype != "CDS" or strand not in ("+", "-"):
                continue
            try:
                start = int(start_s) - 1
                end = int(end_s)
            except ValueError:
                continue
            cds.setdefault(seqid, {"+": [], "-": []})[strand].append((start, end))
    return cds


def cds_mask(seq_len: int, segments: list[tuple[int, int]]) -> np.ndarray:
    """Bool array, True at positions inside any CDS segment."""
    m = np.zeros(seq_len, dtype=bool)
    for s, e in segments:
        m[max(0, s): min(seq_len, e)] = True
    return m


@dataclass
class GenomeData:
    chrom_seq: dict[str, bytes]
    # Per-position mask of plus-strand CDS regions (the original mask used
    # by the v0 pipeline). Kept under the same name for backward-compat
    # with existing experiment code that imports `chrom_cds_mask`.
    chrom_cds_mask: dict[str, np.ndarray]
    # Per-position mask of minus-strand CDS regions. Populated by
    # `load_genome` for use by the clean-label sampler. May be empty
    # dict for genomes loaded by code paths that bypass the new loader,
    # in which case the clean sampler will treat the minus mask as all
    # zeros (only the original plus-strand label semantics are available).
    chrom_cds_mask_minus: dict[str, np.ndarray] | None = None


def load_yeast(data_dir: Path | None = None) -> GenomeData:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "genome"
    return load_genome(data_dir / "scerevisiae_R64.fa",
                       data_dir / "scerevisiae_R64.gff3")


def load_genome(fasta_path: Path, gff_path: Path) -> GenomeData:
    """Generic loader: build GenomeData from any FASTA + GFF3 pair.

    Builds BOTH the plus-strand and minus-strand CDS masks. The plus-strand
    mask is exposed as `chrom_cds_mask` for backward compatibility with the
    v0 sampler; the minus-strand mask is on `chrom_cds_mask_minus` and is
    used by `sample_reads_clean` to restrict negatives to true intergenic
    and to detect minus-strand-CDS positives for RC handling.
    """
    seqs = read_fasta(fasta_path)
    cds = parse_gff3_cds_both_strands(gff_path)
    masks_plus = {
        c: cds_mask(len(s), cds.get(c, {"+": [], "-": []})["+"])
        for c, s in seqs.items()
    }
    masks_minus = {
        c: cds_mask(len(s), cds.get(c, {"+": [], "-": []})["-"])
        for c, s in seqs.items()
    }
    return GenomeData(chrom_seq=seqs,
                      chrom_cds_mask=masks_plus,
                      chrom_cds_mask_minus=masks_minus)


def load_organism(name: str, data_dir: Path | None = None) -> GenomeData:
    """Load a registered organism by short name.

    Names map to subdirectories of `src/data/genome/`:
      yeast      -> scerevisiae_R64.fa + .gff3 (top-level files)
      ecoli      -> ecoli/{genome.fa, anno.gff3}
      bsubtilis  -> bsubtilis/{genome.fa, anno.gff3}
      spombe     -> spombe/{genome.fa, anno.gff3}
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "genome"
    if name == "yeast":
        return load_yeast(data_dir)
    sub = data_dir / name
    return load_genome(sub / "genome.fa", sub / "anno.gff3")


def sample_reads(genome: GenomeData,
                 chroms: list[str],
                 read_bp: int,
                 n_per_class: int,
                 seed: int,
                 phase_align: int | None = None) -> list[tuple[bytes, int]]:
    """Sample `n_per_class` CDS reads (label=1) and `n_per_class` nonCDS
    reads (label=0) of length `read_bp` from the given chromosomes.

    A CDS read is one whose **center** is inside a CDS segment AND whose
    full span has no 'N'. A nonCDS read is one whose **entire span** is
    outside any CDS segment AND has no 'N'.

    `phase_align` (None, 0, 1, 2): if set, only accept reads whose absolute
    genomic start position is congruent to phase_align mod 3. This locks
    the read's first nucleotide to a fixed residue class relative to the
    chromosome coordinate system; the codon-period signal in the hidden
    states then sits at a consistent absolute position within the
    transformer's input. Used to manufacture phase-clean training data so
    we can measure phase-OOD failure at evaluation.

    Sampling is rejection-based with a hard retry cap. If the genome is
    too small we draw fewer reads (caller's responsibility).
    """
    rng = np.random.default_rng(seed)
    half = read_bp // 2
    cds_reads: list[tuple[bytes, int]] = []
    non_reads: list[tuple[bytes, int]] = []
    chroms_avail = [c for c in chroms if c in genome.chrom_seq
                    and len(genome.chrom_seq[c]) >= read_bp + 2]
    if not chroms_avail:
        return []
    weights = np.array([len(genome.chrom_seq[c]) for c in chroms_avail],
                       dtype=np.float64)
    weights /= weights.sum()
    # Phase constraint cuts yield by ~3x; bump the attempt budget.
    base_attempts = max(n_per_class * 50, 5000)
    max_attempts = base_attempts * (3 if phase_align is not None else 1)

    def _try_sample(want_cds: bool, target: list):
        attempts = 0
        while len(target) < n_per_class and attempts < max_attempts:
            attempts += 1
            c = rng.choice(chroms_avail, p=weights)
            seq = genome.chrom_seq[c]
            mask = genome.chrom_cds_mask[c]
            n = len(seq)
            center = int(rng.integers(half, n - half))
            start = center - half
            end = start + read_bp
            if phase_align is not None and (start % 3) != phase_align:
                continue
            sub = seq[start:end]
            if b"N" in sub or b"n" in sub:
                continue
            sub_mask = mask[start:end]
            if want_cds:
                if not sub_mask[half]:
                    continue
            else:
                if sub_mask.any():
                    continue
            target.append((sub, 1 if want_cds else 0))

    _try_sample(True, cds_reads)
    _try_sample(False, non_reads)
    out = cds_reads + non_reads
    rng.shuffle(out)
    return out


# Static reverse-complement table for clean-label minus-strand positives.
# Lowercase covered for robustness; non-IUPAC characters (incl. N) map to
# themselves so the read survives downstream tokenization.
_RC_TABLE = bytes.maketrans(
    b"ACGTacgtNn",
    b"TGCAtgcaNn",
)


def _rc(seq: bytes) -> bytes:
    """Reverse complement a DNA sequence (bytes), preserving case."""
    return seq.translate(_RC_TABLE)[::-1]


def sample_reads_clean(genome: GenomeData,
                       chroms: list[str],
                       read_bp: int,
                       n_per_class: int,
                       seed: int,
                       rc_minus_positives: bool = True
                       ) -> tuple[list[tuple[bytes, int]], dict[str, int]]:
    """Sample CDS-vs-nonCDS reads with the field-standard any-strand label.

    Positive reads: read center is in CDS on EITHER strand. Reads centered
    on minus-strand CDS are reverse-complemented (when `rc_minus_positives`
    is True) so the model always sees CDS in 5'->3' coding orientation.
    Reads where the center is in CDS on BOTH strands (overlapping genes)
    are rejected as ambiguous-orientation.

    Negative reads: read SPAN has no CDS on either strand (true intergenic).

    Reads with `N`/`n` are rejected as in `sample_reads`.

    Returns (reads, stats) where `reads` is a list of (sequence, label) and
    `stats` reports per-class accept counts and minus-strand positive count
    (so callers can log how much RC happened, which matters for downstream
    sanity checks).
    """
    rng = np.random.default_rng(seed)
    half = read_bp // 2
    if genome.chrom_cds_mask_minus is None:
        raise RuntimeError(
            "sample_reads_clean requires GenomeData built via load_genome "
            "with both strand masks. The provided genome only has the "
            "plus-strand mask (legacy loader path).")
    chroms_avail = [c for c in chroms if c in genome.chrom_seq
                    and len(genome.chrom_seq[c]) >= read_bp + 2]
    if not chroms_avail:
        return [], {"cds_plus": 0, "cds_minus": 0, "non": 0,
                    "rejected_overlap": 0, "rejected_n": 0,
                    "rejected_pos_attempts": 0, "rejected_neg_attempts": 0}
    weights = np.array([len(genome.chrom_seq[c]) for c in chroms_avail],
                       dtype=np.float64)
    weights /= weights.sum()
    # Negatives are scarcer in compact bacterial genomes (10% intergenic);
    # double the attempt budget compared to sample_reads.
    base_attempts = max(n_per_class * 100, 10000)

    cds_reads: list[tuple[bytes, int]] = []
    non_reads: list[tuple[bytes, int]] = []
    stats = {"cds_plus": 0, "cds_minus": 0, "non": 0,
             "rejected_overlap": 0, "rejected_n": 0,
             "rejected_pos_attempts": 0, "rejected_neg_attempts": 0}

    # Sample positives.
    attempts = 0
    while len(cds_reads) < n_per_class and attempts < base_attempts:
        attempts += 1
        c = rng.choice(chroms_avail, p=weights)
        seq = genome.chrom_seq[c]
        m_plus = genome.chrom_cds_mask[c]
        m_minus = genome.chrom_cds_mask_minus[c]
        n = len(seq)
        center = int(rng.integers(half, n - half))
        is_plus = bool(m_plus[center])
        is_minus = bool(m_minus[center])
        if not (is_plus or is_minus):
            stats["rejected_pos_attempts"] += 1
            continue
        if is_plus and is_minus:
            stats["rejected_overlap"] += 1
            continue
        start = center - half
        end = start + read_bp
        sub = seq[start:end]
        if b"N" in sub or b"n" in sub:
            stats["rejected_n"] += 1
            continue
        if is_minus and rc_minus_positives:
            sub = _rc(sub)
            stats["cds_minus"] += 1
        elif is_plus:
            stats["cds_plus"] += 1
        else:
            stats["cds_minus"] += 1
        cds_reads.append((sub, 1))

    # Sample negatives: span has NO CDS on either strand.
    attempts = 0
    while len(non_reads) < n_per_class and attempts < base_attempts:
        attempts += 1
        c = rng.choice(chroms_avail, p=weights)
        seq = genome.chrom_seq[c]
        m_plus = genome.chrom_cds_mask[c]
        m_minus = genome.chrom_cds_mask_minus[c]
        n = len(seq)
        center = int(rng.integers(half, n - half))
        start = center - half
        end = start + read_bp
        if (m_plus[start:end].any() or m_minus[start:end].any()):
            stats["rejected_neg_attempts"] += 1
            continue
        sub = seq[start:end]
        if b"N" in sub or b"n" in sub:
            stats["rejected_n"] += 1
            continue
        non_reads.append((sub, 0))
        stats["non"] += 1

    out = cds_reads + non_reads
    rng.shuffle(out)
    return out, stats


def seq_to_kmer_string(seq: bytes, k: int = 6) -> str:
    """Bytes -> space-separated upper-case k-mers (DNABERT input format)."""
    s = seq.decode("ascii", errors="replace").upper()
    return " ".join(s[i:i + k] for i in range(len(s) - k + 1))


class CDSReadDataset(Dataset):
    """Tokenised CDS-vs-nonCDS reads.

    Stores reads as raw bytes and tokenises lazily in `__getitem__` with
    a HuggingFace tokenizer. This keeps memory low and allows the same
    `reads` list to be re-tokenised with different max-length settings.
    """

    def __init__(self, reads: list[tuple[bytes, int]], tokenizer,
                 max_length: int, k: int = 6):
        self.reads = reads
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k = k

    def __len__(self) -> int:
        return len(self.reads)

    def __getitem__(self, idx: int):
        seq, label = self.reads[idx]
        kmers = seq_to_kmer_string(seq, self.k)
        enc = self.tokenizer(kmers, truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return dict(
            input_ids=enc["input_ids"][0],
            attention_mask=enc["attention_mask"][0],
            label=torch.tensor(label, dtype=torch.long),
        )


YEAST_SPLIT = {
    "train":     ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"],
    "val":       ["X"],
    "test_id":   ["XI"],
    "ood_chrom": ["XIII", "XIV", "XV", "XVI"],
}


def gc_content_array(reads: list[tuple[bytes, int]]) -> np.ndarray:
    """Per-read GC fraction for a list of (seq, label) tuples."""
    if not reads:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(len(reads), dtype=np.float32)
    for i, (s, _) in enumerate(reads):
        u = s.upper()
        out[i] = (u.count(b"G") + u.count(b"C")) / max(1, len(u))
    return out


def stratify_by_gc(reads: list[tuple[bytes, int]],
                   low_q: float = 0.33,
                   high_q: float = 0.67,
                   thresholds: tuple[float, float] | None = None
                   ) -> tuple[list, list, float, float]:
    """Split reads into low-GC and high-GC strata.

    If `thresholds=(lo, hi)` is given, those absolute GC fractions are used.
    Otherwise empirical quantiles `low_q` and `high_q` are computed from
    the input pool. Returns (low_reads, high_reads, lo_threshold, hi_threshold).
    Reads in the middle band (lo < GC < hi) are dropped, ensuring a clean
    gap between the two strata.
    """
    gcs = gc_content_array(reads)
    if thresholds is not None:
        lo, hi = thresholds
    elif len(gcs) == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.quantile(gcs, [low_q, high_q]).tolist()
    low = [r for r, g in zip(reads, gcs) if g <= lo]
    high = [r for r, g in zip(reads, gcs) if g >= hi]
    return low, high, float(lo), float(hi)


def sample_reads_multi_organism(organisms: list[GenomeData],
                                names: list[str],
                                read_bp: int,
                                n_per_class_per_org: int,
                                seed: int) -> list[tuple[bytes, int]]:
    """Sample CDS/nonCDS reads from multiple organisms and shuffle the union.

    Each organism contributes `n_per_class_per_org` reads of each class.
    Used by leave-one-species-out training.
    """
    out: list[tuple[bytes, int]] = []
    for i, (g, name) in enumerate(zip(organisms, names)):
        chroms = list(g.chrom_seq.keys())
        reads = sample_reads(g, chroms, read_bp,
                             n_per_class=n_per_class_per_org,
                             seed=seed + 1000 * i + hash(name) % 100)
        out.extend(reads)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    return [out[i] for i in perm]


def build_organism_eval_dataset(genome: GenomeData, tokenizer,
                                read_bp: int = 300,
                                max_length: int | None = None,
                                n_per_class: int = 1000,
                                seed: int = 42) -> CDSReadDataset:
    """Sample CDS-vs-nonCDS reads from ALL chromosomes of an organism, for
    cross-organism eval. No train/val/test split — the whole genome is
    eval-only, since the organism itself is the held-out axis.
    """
    if max_length is None:
        max_length = read_bp - 3
    chroms = list(genome.chrom_seq.keys())
    reads = sample_reads(genome, chroms, read_bp,
                         n_per_class=n_per_class, seed=seed)
    return CDSReadDataset(reads, tokenizer, max_length=max_length)


def build_yeast_datasets(tokenizer,
                         read_bp: int = 300,
                         max_length: int | None = None,
                         n_train_per_class: int = 5000,
                         n_eval_per_class: int = 1000,
                         seed: int = 42,
                         train_phase_align: int | None = None,
                         eval_phase_aligns: tuple[int | None, ...] = (None,),
                         ) -> dict[str, CDSReadDataset]:
    """Build {train, val, test_id, ood_chrom} read datasets from yeast.

    For length-OOD evaluation, call this twice with different `read_bp`
    values and use the eval-only splits from the second call.

    `train_phase_align` (None or 0/1/2): forces training-read start positions
    to one residue class mod 3 (see `sample_reads`). When set, val keeps the
    same alignment (so we can confirm the model didn't lose its trained
    distribution). The eval splits are duplicated across `eval_phase_aligns`:
    for each phase value, an additional split `<name>_p{phase}` is built. Use
    e.g. `eval_phase_aligns=(0, 1, 2, None)` to compare matched-phase ID,
    each shifted phase, and a random-phase mixture.
    """
    if max_length is None:
        # 6-mer tokenisation: read_bp gives read_bp-5 kmers, plus [CLS]+[SEP].
        max_length = read_bp - 3
    genome = load_yeast()

    counts = {"train": n_train_per_class,
              "val": n_eval_per_class,
              "test_id": n_eval_per_class,
              "ood_chrom": n_eval_per_class}

    out: dict[str, CDSReadDataset] = {}
    for split, chroms in YEAST_SPLIT.items():
        if split in ("train", "val"):
            # Train and val use the configured training phase alignment so
            # val tracks training-distribution performance.
            phase = train_phase_align
            reads = sample_reads(genome, chroms, read_bp,
                                 n_per_class=counts[split],
                                 seed=seed + hash(split) % 1000,
                                 phase_align=phase)
            out[split] = CDSReadDataset(reads, tokenizer,
                                        max_length=max_length)
        else:
            # Eval splits get duplicated across requested phase alignments
            # (None = random phase, 0/1/2 = fixed phase).
            for phase in eval_phase_aligns:
                tag = f"{split}_p{phase}" if phase is not None else f"{split}_prand"
                reads = sample_reads(genome, chroms, read_bp,
                                     n_per_class=counts[split],
                                     seed=seed + hash(tag) % 1000,
                                     phase_align=phase)
                out[tag] = CDSReadDataset(reads, tokenizer,
                                          max_length=max_length)
            # Back-compat: also expose the random-phase variant under the
            # bare split name so callers expecting "test_id" / "ood_chrom"
            # still find them.
            if None in eval_phase_aligns:
                out[split] = out[f"{split}_prand"]
    return out
