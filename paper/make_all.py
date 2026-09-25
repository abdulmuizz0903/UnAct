"""Rebuild every figure and table in the results section.

    python paper/make_all.py

Outputs land in paper/figures/ (PDF + PNG) and paper/tables/ (.tex, including
tables/numbers.tex with one macro per number quoted in the prose). No number in
the paper is hand-copied: every output is aggregated from results/*.csv.

The method figure (Figure 1) is drawn by hand and is not generated here.
"""
from __future__ import annotations

import fig_cost
import fig_data_efficiency
import fig_evidence
import fig_mechanism
import fig_multiclass
import fig_sensitivity
import make_rebuttal_tables
import make_tables
from common import set_style


def main():
    set_style()
    print("main-text figures:")
    fig_data_efficiency.fig_forget_size_main()   # Fig. 2: forget-set size
    fig_multiclass.fig_sequential()               # Fig. 3: sequential requests
    fig_cost.fig_cost_k()                         # Fig. 4: cost vs rounds k
    print("appendix figures:")
    fig_data_efficiency.fig_forget_size_full()
    fig_sensitivity.fig_sensitivity_profiles()
    fig_sensitivity.fig_tuning_robustness()
    fig_sensitivity.fig_grids()
    fig_mechanism.fig_selectivity()
    fig_mechanism.fig_coverage()
    fig_evidence.fig_relearn()
    fig_evidence.fig_vit_relearn()
    fig_data_efficiency.fig_vit_forget_size()
    print("tables:")
    make_tables.main()
    make_rebuttal_tables.main()


if __name__ == "__main__":
    main()
