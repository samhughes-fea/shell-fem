# Roadmap — per-plate outer / mid / inner stations (**closed**)

**Status:** complete. Folded into the production contract in [`README.md`](README.md#contract-production).

`ShellTertiaryResultsOrchestrator` recovers in-plane stress at each plate’s **outer / mid / inner** faces. Stored shape: `(n_elem, n_ply, 3, 3)`.

| Face index | Slot |
|------------|------|
| 0 | outer (\(x_3^{k+}\)) |
| 1 | mid |
| 2 | inner (\(x_3^{k-}\)) |

Quaternary complementarity (not a tertiary gap):

- Hashin / Von Mises use the **mid** face only (standard criteria).
- Extreme-fibre track uses laminate free surfaces \(\pm h/2\) with ultimate strengths.

See [cascade index](../../docs/README.md#mid-face-criteria-vs-extreme-fibre).
