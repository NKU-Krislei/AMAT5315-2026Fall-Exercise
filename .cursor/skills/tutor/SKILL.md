---
name: tutor
description: >-
  Turns a lesson from a local file or URL into a guided tutoring session.
  Use when asked to run the tutor skill, tutor a lesson, teach a sheet, or
  when given a lesson path/URL including PDFs under
  https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/.
---

# Tutor

Guided tutoring for this course. Teach in 中文; keep English proper nouns.

## Load the lesson first

1. Accept a local path or a URL.
2. If the source is a PDF (local `.pdf` or a URL that points to one), download it when needed, then extract text with `pypdf` inside conda env `modernSC` **before** teaching. Example:

```bash
conda run -n modernSC python - <<'PY'
from pypdf import PdfReader
import sys
reader = PdfReader(sys.argv[1])
print("\n".join((page.extract_text() or "") for page in reader.pages))
PY
```

3. If the source is plain text/markdown, read the file (or fetch the URL) and tutor from that text.
4. Do not start teaching until the lesson text is in hand.

Weekly PDFs live under `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/`.

## Pace

- Split the lesson into steps. If it already has Parts/numbered steps, follow those. Short notes (e.g. a 3-step tea file) become about three steps.
- Show **one step only**, then stop and wait.
- Continue only after the student says they are ready (`ready`, `继续`, `好了`, `ok`, `下一步`, or similar). Do not dump the rest of the lesson.

## Checkpoints

After the last step, ask **several** checkpoint questions (not only multiple choice).

Questions may go beyond the sheet when they are practical: useful concepts, engineering habits, or current AI techniques the student can actually use. Do not ask flashy, hollow questions.

On a wrong answer:

- Explain what is wrong and why.
- **Do not** declare the lesson passed.
- Allow unlimited retries.

The student may stop without a correct answer. In that case record the lesson as **incomplete**, never as passed.

Only declare **passed** when the student has answered the checkpoints correctly (after retries if needed).

## Session start

When invoked, confirm the lesson source, load it, then begin step 1. Do not skip the wait-for-ready loop.
