Manuscript sources for CMAME. Figures live in `figures/`.

To compile (TeX Live + elsarticle):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If figure paths in `main.tex` assume the flat paper folder, either copy `figures/*` up one level or adjust `\includegraphics` paths.
