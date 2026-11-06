"""Shared CDS PE / backbone display constants used by the CDS-frame tables."""

PE_DISPLAY_CDS = {
    "codon_pe":            r"\textbf{Codon} ($P{=}3$, oracle)",
    "multi_3_resmlp":      r"Codon $\oplus$ $\{10, 40\}$ + zero-init MLP",
    "dual_pe":             r"Codon $\oplus$ Helical (sum)",
    "learned_ape":         r"Learned APE",
    "nucleosome_pe":       r"Helical ($P{=}10.4$, wrong)",
    "standard_sinusoidal": r"Sinusoidal (multi-freq)",
    "no_pe":               r"No PE",
}
PE_ORDER_CDS_TINY = ["codon_pe", "multi_3_resmlp", "dual_pe", "learned_ape",
                     "nucleosome_pe", "standard_sinusoidal", "no_pe"]
PE_ORDER_CDS_FM = ["codon_pe", "no_pe"]
BACKBONE_DISPLAY_CDS = {
    "tiny_transformer": r"\emph{Tiny transformer (random init)}",
    "hyenadna":         r"\emph{HyenaDNA-small (frozen, pretrained)}",
    "caduceus":         r"\emph{Caduceus-PS (frozen, pretrained)}",
}
