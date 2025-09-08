#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch: register many PETs to each subject's MRI using PoR (por_pet_mri.py).

Inputs
------
--mri-csv : CSV with columns: subject,t1w_folder
            (the folder must contain *_T1w_128iso2.nii.gz; we will pick the largest if multiple)
--pet-csv : CSV with columns: subject,nifti_path
--mask4d  : Either a fixed 4D mask path (in MR space), or a template with {subject} and/or {session}.
            Example template: "/path/ADNI_OUT/{subject}/registered/{session}/mask4d_44voi.nii.gz"
--out-root: Output root; per-PET results go into <out-root>/<subject>/<pet_stem>/

Options: --petpvc, --fwhm, --iters, --skip-rbv, --dry-run

Notes
-----
- Requires the file por_pet_mri.py to be importable.
- Dependencies: SimpleITK, pandas, nibabel.
"""

import os, sys, csv, shutil
from pathlib import Path
import pandas as pd

# ---- import PoR ----
def import_por(por_path=None):
    if por_path is not None:
        por_path = Path(por_path)
        sys.path.insert(0, str(por_path.parent))
    try:
        import por_pet_mri as por
        return por
    except Exception as e:
        raise SystemExit(f"[ERROR] Cannot import por_pet_mri.py: {e}")

def find_t1w(t1_folder: Path):
    cands = sorted(t1_folder.glob("*_T1w_128iso2.nii*"))
    if not cands:
        # fallback: any .nii* inside that looks like T1
        cands = [p for p in t1_folder.glob("*.nii*") if "t1" in p.name.lower() or "mprage" in p.name.lower()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return cands[0]

def resolve_mask(mask_pattern: str, subject: str, session: str, t1_folder: Path):
    # If a file exists at mask_pattern, use it
    mp = Path(mask_pattern)
    if mp.exists():
        return mp
    # If pattern contains placeholders
    try:
        templ = mask_pattern.format(subject=subject, session=session)
        pt = Path(templ)
        if pt.exists():
            return pt
    except KeyError:
        pass
    # Heuristic: look for a 4D mask within the T1 folder or its parent
    for root in [t1_folder, t1_folder.parent, t1_folder.parent.parent]:
        if root is None or not root.exists():
            continue
        cands = list(root.glob("*4D*44.nii*")) + list(root.glob("mask4d*44.nii*")) + list(root.glob("*.4d.nii*"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_size, reverse=True)
            return cands[0]
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch PoR: PET->MRI for all subjects")
    ap.add_argument("--mri-csv", required=True, type=Path, help="CSV with columns: subject,t1w_folder")
    ap.add_argument("--pet-csv", required=True, type=Path, help="CSV with columns: subject,nifti_path")
    ap.add_argument("--mask4d", required=True, help="4D mask path or template with {subject},{session}")
    ap.add_argument("--out-root", required=True, type=Path, help="Output root directory")
    ap.add_argument("--por", default=None, help="Path to por_pet_mri.py (if not importable)")
    ap.add_argument("--petpvc", default="petpvc", help="PETPVC binary path")
    ap.add_argument("--fwhm", nargs=3, type=float, default=[8.0,8.0,8.0], help="PSF FWHM mm (x y z)")
    ap.add_argument("--iters", type=int, default=5, help="Max iterations")
    ap.add_argument("--tol-t", type=float, default=0.1, help="Translation tolerance (mm)")
    ap.add_argument("--tol-r", type=float, default=0.1, help="Rotation tolerance (deg)")
    ap.add_argument("--skip-rbv", action="store_true", help="Skip final RBV PVC")
    ap.add_argument("--dry-run", action="store_true", help="List actions but do not execute")
    ap.add_argument("--csv", type=Path, default=None, help="Optional summary CSV")
    args = ap.parse_args()

    por = import_por(args.por)

    df_mri = pd.read_csv(args.mri_csv)
    df_pet = pd.read_csv(args.pet_csv)

    # Build subject -> t1 folder map
    mri_map = {}
    for _, r in df_mri.iterrows():
        subj = str(r["subject"]).strip()
        t1_folder = Path(str(r["t1w_folder"]).strip())
        if not t1_folder.exists():
            print(f"[WARN] T1 folder missing: {t1_folder}")
            continue
        mri_map[subj] = t1_folder

    rows = []
    for _, r in df_pet.iterrows():
        subj = str(r["subject"]).strip()
        pet_path = Path(str(r["nifti_path"]).strip())
        if not pet_path.exists():
            print(f"[WARN] PET missing: {pet_path}")
            rows.append({"subject":subj,"pet":str(pet_path),"status":"PET_NOT_FOUND"})
            continue
        if subj not in mri_map:
            print(f"[WARN] No MRI for subject {subj}")
            rows.append({"subject":subj,"pet":str(pet_path),"status":"MRI_NOT_FOUND"})
            continue

        t1_folder = mri_map[subj]
        t1_img = find_t1w(t1_folder)
        if t1_img is None:
            print(f"[WARN] No T1 NIfTI in {t1_folder}")
            rows.append({"subject":subj,"pet":str(pet_path),"status":"T1_NOT_FOUND"})
            continue

        session = t1_folder.name
        mask4d = resolve_mask(args.mask4d, subj, session, t1_folder)
        if mask4d is None or not mask4d.exists():
            print(f"[WARN] mask4d not found for {subj} session {session}")
            rows.append({"subject":subj,"pet":str(pet_path),"mr":str(t1_img),"status":"MASK4D_NOT_FOUND"})
            continue

        outdir = args.out_root / subj / pet_path.stem
        outdir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            print(f"[DRY-RUN] {subj}: PET={pet_path.name} -> MR={t1_img.name} (mask={mask4d.name})  out={outdir}")
            rows.append({"subject":subj,"pet":str(pet_path),"mr":str(t1_img),"mask4d":str(mask4d),
                        "out":str(outdir),"status":"DRY_RUN"})
            continue

        try:
            outs = por.por_register_and_pvc(
                pet_path=str(pet_path),
                mr_path=str(t1_img),
                mask4d_path=str(mask4d),
                out_dir=str(outdir),
                fwhm_xyz_mm=tuple(args.fwhm),
                petpvc_bin=args.petpvc,
                max_iters=int(args.iters),
                tol_trans_mm=float(args.tol_t),
                tol_rot_deg=float(args.tol_r),
                do_rbv=(not args.skip_rbv),
            )
            rows.append({
                "subject": subj,
                "pet": str(pet_path),
                "mr": str(t1_img),
                "mask4d": str(mask4d),
                "out": str(outdir),
                "pet_in_mr_final": outs.get("pet_in_mr_final",""),
                "transform_path": outs.get("transform_path",""),
                "rbv_pvc_path": outs.get("rbv_pvc_path",""),
                "gtm_means_csv_final": outs.get("gtm_means_csv_final",""),
                "status": "OK"
            })
            print(f"[OK] {subj}: {pet_path.name} -> {outdir}")
        except Exception as e:
            rows.append({"subject":subj,"pet":str(pet_path),"mr":str(t1_img),"mask4d":str(mask4d),
                        "out":str(outdir),"status":f"ERROR: {e}"})
            print(f"[ERROR] {subj}: {e}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"[OK] Summary CSV: {args.csv}")

if __name__ == "__main__":
    main()
