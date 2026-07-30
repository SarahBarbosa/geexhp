import os, re, sys, glob
import numpy as np
import pandas as pd

ERAS = ["modern", "proterozoic", "archean", "random"]

# Bin edges chosen to straddle the underflow cliff at ~136 deg.
PHASE_BINS = [0, 60, 90, 120, 130, 133, 135, 136, 137, 138, 141]

# Same threshold the training set uses (03-load_read_TFRecords.ipynb):
# a planet is kept only if SNR_pctl > 3 in all six bands.
SNR_PCTL_MIN = 3.0


def era_of(fname):
    b = os.path.basename(fname)
    for e in ERAS:
        if b.startswith(e + "_"):
            return e
    return "unknown"


def find_folders(args):
    if args:
        return [a.rstrip("/") for a in args if os.path.isdir(a)]
    return sorted(
        [
            d
            for d in glob.glob("data*")
            if os.path.isdir(d) and re.match(r"^data\d+$", os.path.basename(d))
        ]
    )


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df else None


def snr_pctl(sig, noise, p_low=10, p_high=90):
    """
    (P_high - P_low) / (sqrt(2) * RMS(noise)), row-wise.

    Mirrors _calculate_feature_snr_percentile in
    geexhp/modelfuncs/tfrecord_conversion.py, which is what fills
    SNR_FEATURE_PCTL_* in the TFRecords.  The signal is the noiseless ALBEDO,
    so a fully zeroed spectrum has zero amplitude and scores exactly 0.
    """
    amp = np.percentile(sig, p_high, axis=1) - np.percentile(sig, p_low, axis=1)
    rms = np.sqrt(np.mean(noise**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return amp / (np.sqrt(2.0) * rms)


def zeros_vs_phase(df, insts):
    """
    Zeroed albedo channels binned by phase angle, and the effect of the
    SNR_pctl > 3 cut.  Returns per-era totals for the global rollup, or None
    if the batch predates the OBJECT-PHASE-ANGLE column.
    """
    phase = num(df, "OBJECT-PHASE-ANGLE")
    if phase is None or not phase.notna().any():
        print("  no OBJECT-PHASE-ANGLE column -- skipping phase/SNR report")
        return None
    phase = phase.to_numpy()

    nzero = np.zeros(len(df), dtype=int)
    peak = np.zeros(len(df))  # brightest channel over all bands
    nchan = 0
    snr = np.full((len(df), len(insts)), np.nan)
    for j, inst in enumerate(insts):
        a = np.vstack(df[f"ALBEDO_{inst}"].apply(np.array)).astype(np.float64)
        n = np.vstack(df[f"NOISE_{inst}"].apply(np.array)).astype(np.float64)
        nzero += (a == 0).sum(1)
        peak = np.maximum(peak, a.max(1))
        nchan += a.shape[1]
        snr[:, j] = snr_pctl(a, n)

    dirty = nzero > 0  # any zeroed channel
    gone = nzero > 0.5 * nchan  # spectrum destroyed
    keep = np.isfinite(snr).all(1) & (snr > SNR_PCTL_MIN).all(1)

    print(
        f"\n  zeroed albedo channels vs phase angle "
        f"({nchan} channels/planet over {len(insts)} bands)"
    )
    print(
        f"    {'phase [deg]':>13} {'planets':>8} {'any zero':>9} "
        f"{'>50% zero':>10} {'med peak alb':>13} {'pass SNR>3':>11} {'dirty kept':>11}"
    )
    for lo, hi in zip(PHASE_BINS[:-1], PHASE_BINS[1:]):
        m = (phase >= lo) & (phase < hi)
        if not m.any():
            continue
        print(
            f"    {f'{lo}-{hi}':>13} {int(m.sum()):>8} "
            f"{int(dirty[m].sum()):>9} {int(gone[m].sum()):>10} "
            f"{np.median(peak[m]):>13.3e} {int(keep[m].sum()):>11} "
            f"{int((dirty & keep)[m].sum()):>11}"
        )

    clean_max = phase[~dirty].max() if (~dirty).any() else float("nan")
    dirty_min = phase[dirty].min() if dirty.any() else float("nan")
    print(
        f"    cliff: highest clean phase {clean_max:.2f} deg, "
        f"lowest contaminated {dirty_min:.2f} deg"
    )

    print(
        f"\n  SNR_pctl > {SNR_PCTL_MIN:g} cut (all {len(insts)} bands, "
        f"as applied in 03-load_read_TFRecords.ipynb)"
    )
    print(
        f"    kept                       : {int(keep.sum())}/{len(df)} "
        f"({100 * keep.mean():.2f}%)"
    )
    print(f"    with >=1 zeroed channel    : {int(dirty.sum())}")
    print(
        f"    ... of those, kept         : {int((dirty & keep).sum())}"
        f"{'   <-- LEAKING INTO TRAINING SET' if (dirty & keep).any() else ''}"
    )
    if keep.any():
        print(f"    max phase among kept       : {phase[keep].max():.2f} deg")

    return {
        "n": len(df),
        "dirty": int(dirty.sum()),
        "gone": int(gone.sum()),
        "keep": int(keep.sum()),
        "leak": int((dirty & keep).sum()),
        "keep_phase_max": phase[keep].max() if keep.any() else float("nan"),
    }


def check_era(era, files):
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(
        f"\n{'='*70}\n{era.upper()}  ({len(df)} planets from {len(files)} batch file(s))\n{'='*70}"
    )

    for c in [
        "GEOMETRY-OBS-ALTITUDE",
        "OBJECT-GRAVITY",
        "ATMOSPHERE-PRESSURE",
        "OBJECT-STAR-TEMPERATURE",
        "OBJECT-DIAMETER",
        "OBJECT-SOLAR-LATITUDE",
        "OBJECT-OBS-LATITUDE",
        "ATMOSPHERE-WEIGHT",
    ]:
        v = num(df, c)
        if v is not None and v.notna().any():
            u = v.nunique()
            print(
                f"  {c:26s} min={v.min():.4g}  max={v.max():.4g}  unique={u if u<6 else '>5'}"
            )
    if "OBJECT-STAR-TYPE" in df:
        print(f"  star types: {df['OBJECT-STAR-TYPE'].value_counts().to_dict()}")
    if "ATMOSPHERE-GAS" in df:
        gs = df["ATMOSPHERE-GAS"].apply(lambda s: tuple(s.split(","))).value_counts()
        print(
            f"  gas sets: {len(gs)} distinct; most common ({gs.iloc[0]}x): {','.join(gs.index[0])}"
        )

    insts = sorted({c.split("_", 1)[1] for c in df.columns if c.startswith("ALBEDO_")})
    for inst in insts:
        w = np.array(df[f"WAVELENGTH_{inst}"].iloc[0])
        a = np.vstack(df[f"ALBEDO_{inst}"].apply(np.array))
        n = np.vstack(df[f"NOISE_{inst}"].apply(np.array))
        snr = a / np.where(n == 0, np.nan, n)
        print(
            f"  {inst:7s} {w.min():.3f}-{w.max():.3f}um  nchan={len(w):3d}  "
            f"alb[{a.min():.2e},{a.max():.2e}]  medSNR={np.nanmedian(snr):.2f}  "
            f"nonfin={int((~np.isfinite(a)).sum())}  neg={int((a<0).sum())}  "
            f"zeros={int((a==0).sum())}"
        )

    for inst in [i for i in insts if "NIR" in i]:
        w = np.array(df[f"WAVELENGTH_{inst}"].iloc[0])
        a = np.vstack(df[f"ALBEDO_{inst}"].apply(np.array))
        band = (w > 1.34) & (w < 1.45)
        cont = (w > 1.20) & (w < 1.30)
        if band.any() and cont.any():
            r = np.median(a[:, band]) / np.median(a[:, cont])
            tag = "H2O band PRESENT" if r < 0.7 else "NO band -- CHECK opacity tables"
            print(f"  {inst}: 1.4um band/continuum = {r:.3f}  ({tag})")

    return zeros_vs_phase(df, insts)


def main():
    folders = find_folders(sys.argv[1:])
    if not folders:
        sys.exit("No data<N> folders found.")
    print(f"Scanning folders: {', '.join(folders)}")
    by_era = {}
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, "*.parquet"))):
            by_era.setdefault(era_of(f), []).append(f)
    stats = {}
    for era in ERAS:
        if by_era.get(era):
            s = check_era(era, by_era[era])
            if s:
                stats[era] = s
    if by_era.get("unknown"):
        print(
            f"\n{len(by_era['unknown'])} file(s) with unrecognized era prefix (ignored)."
        )

    if stats:
        tot = {
            k: sum(s[k] for s in stats.values())
            for k in ("n", "dirty", "gone", "keep", "leak")
        }
        print(f"\n{'='*70}\nALL ERAS\n{'='*70}")
        print(
            f"  {'era':<14}{'planets':>9}{'any zero':>10}{'>50% zero':>11}"
            f"{'kept':>9}{'dirty kept':>12}"
        )
        for era, s in stats.items():
            print(
                f"  {era:<14}{s['n']:>9}{s['dirty']:>10}{s['gone']:>11}"
                f"{s['keep']:>9}{s['leak']:>12}"
            )
        print(
            f"  {'TOTAL':<14}{tot['n']:>9}{tot['dirty']:>10}{tot['gone']:>11}"
            f"{tot['keep']:>9}{tot['leak']:>12}"
        )
        phase_max = max(s["keep_phase_max"] for s in stats.values())
        print(
            f"\n  {100 * tot['dirty'] / tot['n']:.2f}% of generated planets have at "
            f"least one zeroed channel; all of them sit above ~135 deg phase."
        )
        if tot["leak"]:
            print(
                f"  WARNING: {tot['leak']} contaminated planet(s) pass the "
                f"SNR_pctl > {SNR_PCTL_MIN:g} cut and would reach the training set."
            )
        else:
            print(
                f"  The SNR_pctl > {SNR_PCTL_MIN:g} cut removes every one of them: "
                f"{tot['keep']} planets kept, max phase {phase_max:.2f} deg."
            )
            print(
                f"  Note this is incidental, not by design -- the cut works because "
                f"a zeroed\n  spectrum has zero P90-P10 amplitude. Lowering the "
                f"threshold would let them back in."
            )


if __name__ == "__main__":
    main()
