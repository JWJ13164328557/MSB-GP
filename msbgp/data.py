"""Simulated Genomic Selection data.

A structured genotype-to-phenotype simulator that lets a user control the
underlying genetic architecture.  This is essential for benchmarking: by
varying the architecture we can show WHEN a flexible multi-kernel method beats
a purely additive model (GBLUP) and when they are equivalent.

Supported architectures
-----------------------
  * "polygenic"  -- many small-effect QTL (infinitesimal model, GBLUP's home turf)
  * "oligogenic" -- few large-effect QTL (Bayes-type methods win)
  * "epistatic"  -- additive + pairwise interaction (epistasis) effects,
                    which linear additive models cannot capture (kernels win)

Genotypes are simulated as biallelic SNPs under Hardy-Weinberg equilibrium
with a user-specified allele-frequency distribution.
"""

from pathlib import Path

import numpy as np

__all__ = ["simulate_gs_data", "save_genotype_memmap", "load_genotype_memmap", "validate_genotype_memmap", "genotype_memmap_to_csr"]


def _genotypes(n, m, rng, min_maf=0.05, max_maf=0.5):
    """Simulate {0,1,2} SNP calls under HWE with random allele frequencies."""
    p = rng.uniform(min_maf, max_maf, size=m)
    a = rng.binomial(1, p, size=(n, m)).astype(float)
    b = rng.binomial(1, p, size=(n, m)).astype(float)
    X = a + b  # counts of reference allele => {0,1,2}
    return X, p


def simulate_gs_data(n=500, m=1000, architecture="polygenic", h2=0.5,
                     n_qtl=None, seed=0, interaction_degree=2):
    """Simulate a genomic selection dataset.

    Parameters
    ----------
    n : int
        Number of individuals.
    m : int
        Number of markers.
    architecture : {"polygenic", "oligogenic", "epistatic"}
        Underlying genetic architecture.
    h2 : float
        Narrow-sense heritability of the additive (or total) genetic variance.
    n_qtl : int, optional
        Number of markers with non-zero effect.  Defaults to m/10 for
        polygenic, 20 for oligogenic, and m/10 for epistatic.
    interaction_degree : int
        Degree of epistasis for the "epistatic" architecture (2 = pairwise).

    Returns
    -------
    dict with keys: X (genotypes), y (phenotype), g (true genetic value),
                    qtl_idx, h2, architecture, maf
    """
    rng = np.random.default_rng(seed)
    X, p = _genotypes(n, m, rng)

    if n_qtl is None:
        if architecture == "oligogenic":
            n_qtl = 20
        else:
            n_qtl = max(10, m // 10)

    qtl_idx = rng.choice(m, size=n_qtl, replace=False)
    Xq = X[:, qtl_idx]

    # standardise markers so effect sizes are comparable
    Xs = (Xq - Xq.mean(axis=0)) / (Xq.std(axis=0) + 1e-12)

    if architecture == "polygenic":
        # many small effects
        beta = rng.normal(0, 1, size=n_qtl)
        g = Xs @ beta
    elif architecture == "oligogenic":
        # few large effects ~ geometric decay
        beta = np.zeros(n_qtl)
        for i in range(n_qtl):
            beta[i] = rng.normal(0, 1)
        # make first QTL dominate (capsule of the decay pattern)
        beta = beta / (np.arange(1, n_qtl + 1) ** 0.5)
        g = Xs @ beta
    elif architecture == "epistatic":
        # additive part
        beta = rng.normal(0, 1, size=n_qtl)
        g = Xs @ beta
        # pairwise interactions among a subset of QTL
        n_int = min(n_qtl, 30)
        if n_int >= 2:
            idx = rng.choice(n_qtl, size=n_int, replace=False)
            g_int = np.zeros(n)
            for i in range(0, n_int - 1):
                for j in range(i + 1, n_int):
                    g_int += rng.normal(0, 1) * (Xs[:, idx[i]] * Xs[:, idx[j]])
            g = 0.6 * g + 0.4 * g_int
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    # scale genetic variance to requested heritability
    g = g - g.mean()
    var_g = np.var(g)
    var_e = var_g * (1.0 - h2) / h2
    e = rng.normal(0, np.sqrt(var_e), size=n)
    y = g + e
    y = y - y.mean()

    return {
        "X": X,
        "y": y,
        "g": g,
        "qtl_idx": qtl_idx,
        "h2": h2,
        "architecture": architecture,
        "maf": p,
    }


# ---------------------------------------------------------------------------
# Genotype memory-mapped I/O helpers
# ---------------------------------------------------------------------------

def _memmap_sidecar(path):
    """Return the JSON sidecar path used by the .dat memmap format."""
    p = Path(path)
    return p.with_suffix(p.suffix + ".json") if p.suffix else p.with_suffix(".json")


def validate_genotype_memmap(X, *, require_integer=True, allow_missing=False,
                             name="genotypes", verbose=False):
    """Validate a memory-mapped or dense genotype dosage matrix.

    The public MSB-GP genotype contract is biallelic dosage coded as 0/1/2.
    Integer ``int8`` memmaps are the preferred on-disk representation for
    large cohorts because they reduce raw genotype storage by 8x relative to
    float64 without changing the exact additive/dominance kernel path.

    Parameters
    ----------
    X : ndarray or numpy.memmap
        Genotype matrix with samples in rows and markers in columns.
    require_integer : bool
        If True, all finite values must be in {0, 1, 2}.
    allow_missing : bool
        If False, NaN values raise ``ValueError``.  If True, NaN values are
        allowed and downstream preprocessing must impute them.
    name : str
        Human-readable object name used in error messages.
    verbose : bool
        Print a concise validation summary.

    Returns
    -------
    bool
        True when validation passes.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2-D sample x marker matrix")
    values = X.data if isinstance(X, np.memmap) else X
    if not allow_missing:
        has_missing = False
        if np.issubdtype(values.dtype, np.floating):
            # Chunked NaN check keeps memory bounded for memmaps.
            step = 65536
            flat = values.reshape(-1)
            for start in range(0, flat.size, step):
                if np.isnan(np.asarray(flat[start:start + step])).any():
                    has_missing = True
                    break
        if has_missing:
            raise ValueError(f"{name} contains NaN values; impute them before fitting")
    if require_integer:
        step = 65536
        flat = values.reshape(-1)
        ok = True
        for start in range(0, flat.size, step):
            block = np.asarray(flat[start:start + step])
            if np.issubdtype(block.dtype, np.floating):
                finite = np.isfinite(block)
                if not np.all(np.isin(block[finite], (0.0, 1.0, 2.0))):
                    ok = False
                    break
            else:
                if not np.all((block >= 0) & (block <= 2)):
                    ok = False
                    break
        if not ok:
            raise ValueError(
                f"{name} must use 0/1/2 biallelic dosage coding for "
                "dominance/epistasis kernels"
            )
    if verbose:
        print(f"{name}: shape={X.shape}, dtype={X.dtype}, memmap={isinstance(X, np.memmap)}")
    return True


def save_genotype_memmap(X, path, dtype=np.int8, *, overwrite=True,
                         validate=True, verbose=False):
    """Save genotypes as a memory-mappable ``.npy`` or ``.dat`` file.

    ``.npy`` files can be reopened with ``np.load(..., mmap_mode='r')``.
    Other suffixes use a raw binary ``.dat`` plus a JSON sidecar that records
    shape and dtype.  For biallelic dosages, ``dtype=np.int8`` is recommended.

    Returns
    -------
    numpy.memmap
        Read-only memmap over the saved file.
    """
    X = np.asarray(X)
    if validate:
        validate_genotype_memmap(X, require_integer=True, allow_missing=False,
                                 name="X", verbose=verbose)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists")
    arr = X.astype(dtype, copy=False)
    if p.suffix.lower() == ".npy":
        np.save(p, arr, allow_pickle=False)
        return np.load(p, mmap_mode="r")
    sidecar = _memmap_sidecar(p)
    if p.exists():
        p.unlink()
    mm = np.memmap(p, dtype=np.dtype(dtype), mode="w+", shape=X.shape)
    step = max(1, int(16_000_000 // max(1, X.shape[1])))
    for start in range(0, X.shape[0], step):
        stop = min(X.shape[0], start + step)
        mm[start:stop, :] = arr[start:stop, :]
    mm.flush()
    sidecar.write_text(
        '{"shape": [' + ','.join(str(int(v)) for v in X.shape) +
        f'], "dtype": "{np.dtype(dtype).str}"}}',
        encoding="utf-8",
    )
    if verbose:
        print(f"saved genotype memmap: {p} {X.shape} {np.dtype(dtype)}")
    return load_genotype_memmap(p, mode="r")


def load_genotype_memmap(path, mode="r"):
    """Load a genotype memmap written by :func:`save_genotype_memmap`.

    Parameters
    ----------
    path : str or pathlib.Path
        ``.npy`` file or raw ``.dat`` file with a JSON sidecar.
    mode : {"r", "r+", "c"}
        NumPy memmap mode.  Use ``"r"`` for read-only analysis.

    Returns
    -------
    numpy.memmap
        Memory-mapped genotype matrix.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        return np.load(p, mmap_mode=mode, allow_pickle=False)
    sidecar = _memmap_sidecar(p)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing memmap sidecar: {sidecar}")
    import json
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["dtype"])
    return np.memmap(p, dtype=dtype, mode=mode, shape=shape)


def genotype_memmap_to_csr(X, *, chunk_size=2048, dtype=np.float32):
    """Convert a dense or memory-mapped 0/1/2 genotype matrix to CSR.

    Chunking avoids allocating a second full dense float64 genotype matrix.
    Use this for very sparse cohorts or when SciPy sparse kernels are desired.
    """
    from scipy import sparse as sp
    X = np.asarray(X)
    validate_genotype_memmap(X, require_integer=True, allow_missing=False,
                             name="X")
    blocks = []
    for start in range(0, X.shape[0], int(chunk_size)):
        stop = min(X.shape[0], start + int(chunk_size))
        blocks.append(sp.csr_matrix(np.asarray(X[start:stop], dtype=dtype)))
    return sp.vstack(blocks, format="csr")
