"""Static checks for the report (no TeX installation needed): inputs exist, figures exist,
every \\ref has a \\label, every \\cite key is in refs.bib, every custom macro is defined,
no non-ASCII characters, balanced braces per file."""
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "report"
texs = {p.name: p.read_text(encoding="utf-8") for p in R.glob("*.tex")}
tables = {p.relative_to(R).as_posix(): p.read_text(encoding="utf-8") for p in (R / "tables").glob("*.tex")}
allsrc = "\n".join(list(texs.values()) + list(tables.values()))
problems = []
for name, s in list(texs.items()) + list(tables.items()):
    s_nc = re.sub(r"(?<!\\)%.*", "", s)
    for i, ch in enumerate(s):
        if ord(ch) > 127:
            problems.append(f"{name}: non-ASCII {ch!r} near {s[max(0, i-30):i+10]!r}")
            break
    if s_nc.count("{") - s_nc.count("\\{") != s_nc.count("}") - s_nc.count("\\}"):
        problems.append(f"{name}: unbalanced braces")
    for inc in re.findall(r"\\input\{([^}]+)\}", s_nc):
        f = R / (inc if inc.endswith(".tex") else inc + ".tex")
        if not f.exists():
            problems.append(f"{name}: missing \\input {inc}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s_nc):
        if not (R / g).exists():
            problems.append(f"{name}: missing figure {g}")
labels = set(re.findall(r"\\label\{([^}]+)\}", allsrc))
for ref in set(re.findall(r"\\ref\{([^}]+)\}", allsrc)):
    if ref not in labels:
        problems.append(f"undefined ref {ref}")
bib = (R / "refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
for group in re.findall(r"\\cite\{([^}]+)\}", allsrc):
    for k in group.split(","):
        if k.strip() not in keys:
            problems.append(f"missing bib key {k}")
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", allsrc))
for m in ["baseTestAcc", "ftTestAcc", "baseTestFone", "ftTestFone", "baseTestEvF", "ftTestEvF", "calTestAcc",
          "calTestFone", "kvMaxSpeedup", "tiledMaxErr", "promptMean"]:
    if f"\\{m}" in allsrc and m not in defined:
        problems.append(f"macro {m} used but not defined")
nums = (R / "numbers.tex").read_text() if (R / "numbers.tex").exists() else ""
if "TBD" in nums:
    problems.append("numbers.tex still contains TBD")
print("\n".join(problems) if problems else "LaTeX static checks passed")
cited = {k.strip() for g in re.findall(r'cite\{([^}]+)\}', allsrc) for k in g.split(',')}
print(len(labels), 'labels,', len(keys), 'bib entries,', len(cited), 'cited')
