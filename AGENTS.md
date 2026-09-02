# AGENTS.md

This is a public student repository for AMAT5315-2026Fall: Modern Scientific Computing at HKUST(GZ). The owner is jlei, a student in the course.

Course materials: https://giggleliu.github.io/AMAT5315-2026Fall/index.html

This repo holds weekly learning records and exercises. There is no weekly hand-in; work is shown in class. Graded work is the midterm and final projects.

## Layout

Weekly work lives in `week1/`, `week2/`, `week3/`, and so on. Week 3's work goes in `week3/`. Generated figures go in that week's directory, not at the repo root.

Do not commit lecture PDFs.

## Environment

Use the conda environment `modernSC`. Run tests with pytest, for example:

```bash
conda activate modernSC
python -m pytest week1/
```

`setup.txt` records coding-agent and environment setup information (agent identity/version and the Python version in `modernSC`).

## Git

Write clear, specific commit messages. Commit each step separately when the week sheet asks for it.

Memory probe: W1-MEMORY-5315
