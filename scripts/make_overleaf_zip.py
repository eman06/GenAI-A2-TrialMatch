"""Package report/ for Overleaf (New Project -> Upload Project -> report_overleaf.zip, compiler pdfLaTeX)."""
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "report"
out = R.parent / "report_overleaf.zip"
missing = [n for n in ["main.tex", "refs.bib", "numbers.tex", "results_section.tex", "error_section.tex",
                       "kv_section.tex", "attn_section.tex", "spec_section.tex"] if not (R / n).exists()]
if missing:
    raise SystemExit(f"missing report parts: {missing}")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(R.rglob("*")):
        if p.is_file() and p.suffix in {".tex", ".bib", ".pdf", ".png"} and p.name != "main.pdf":
            z.write(p, p.relative_to(R).as_posix())
print(out, sum(1 for _ in zipfile.ZipFile(out).namelist()), "files")
