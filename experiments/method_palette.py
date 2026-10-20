# -*- coding: utf-8 -*-
"""Single source of truth for method colors across ALL publication figures.

The canonical palette uses a modern, publication-grade Tableau-like family:

    GBLUP         slate blue   #4E79A7
    RR-BLUP       soft teal    #76B7B2
    BayesA        green        #59A14F
    BayesCpi      light green  #8CD17D   (legend label "BayesC")
    MSB-GP        coral red    #E15759   (our method)
    MSB-GP-Add    amber        #F28E2B   (additive-only ablation)
    RandomForest  neutral gray #BAB0AC   (legend label "RF")
    DNNGP         muted purple #B07AA1

The GMKL trio (MeanGMKL / FHGMKL / CKAGMKL) does not appear in the real-data
figures, so it is assigned a coherent teal / brown / pink family that does not
collide with any of the colors above.
"""

# --- Canonical mapping (used across all publication figures) ------------ #
METHOD_COLORS = {
    "GBLUP":        "#4E79A7",  # slate blue
    "RR-BLUP":      "#76B7B2",  # soft teal-blue
    "BayesA":       "#59A14F",  # green
    "BayesCpi":     "#8CD17D",  # light green
    "MSB-GP":       "#E15759",  # coral red
    "MSB-GP-Add":   "#F28E2B",  # amber
    "RandomForest": "#BAB0AC",  # neutral gray
    "DNNGP":        "#B07AA1",  # muted purple
    # GMKL competitors (not in the real-data palette -> non-colliding family)
    "MeanGMKL":     "#86BCB6",  # soft teal
    "FHGMKL":       "#9C755F",  # taupe brown
    "CKAGMKL":      "#D37295",  # rose pink
}

# Short labels used in figure legends / tick labels
SHORT_LABEL = {
    "RandomForest": "RF",
    "BayesCpi": "BayesC",
    "MeanGMKL": "MeanGMKL",
    "FHGMKL": "FHGMKL",
    "CKAGMKL": "CKAGMKL",
}

# Fixed display order (used in every figure)
METHOD_ORDER = [
    "GBLUP", "RR-BLUP", "BayesA", "BayesCpi", "MSB-GP", "MSB-GP-Add",
    "RandomForest", "DNNGP", "MeanGMKL", "FHGMKL", "CKAGMKL",
]


def color_for(method, default="#7F7F7F"):
    return METHOD_COLORS.get(method, default)


def label_for(method):
    return SHORT_LABEL.get(method, method)




