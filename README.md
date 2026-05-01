# Medical Reasoning LLM

Fine-tune a small open-weight LLM on `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2`
and compare with-reasoning vs answer-only training.
**Learning artefact — not a clinical product, no deployment.**

## Project documents

- **Spec (full implementation):** [docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md](docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md)
- **Phase-1 design doc (assignment deliverable):** [design_doc.md](design_doc.md)
- **Plan 1 (current — bootstrap + Notebook 01):** [docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md](docs/superpowers/plans/2026-05-02-plan1-bootstrap-and-notebook01.md)

## Quick start (local — `src/` tests)

```powershell
pip install transformers pytest
python -m pytest tests/ -v
```

You don't need the full `requirements.txt` locally — only the GPU notebooks
on Kaggle do.

## Quick start (Kaggle — training notebooks)

Inside any Kaggle notebook (T4 GPU, internet on):

```python
!git clone https://github.com/abhishek1998s/medical-reasoning-llm.git
%cd medical-reasoning-llm
!pip install -q -r requirements.txt
```

Then run cells. Adapters and large CSVs go under `outputs/` (gitignored).

## Layout

```
.
├── configs/experiment_config.yaml   # single source of truth
├── src/                              # reusable utilities (formatters, etc.)
├── tests/                            # pytest suite for src/
├── notebooks/                        # 6 Jupyter notebooks (added day-by-day)
├── docs/superpowers/                 # spec + plans
├── design_doc.md                     # short Phase-1 assignment deliverable
├── train_sft.py                      # consolidated at end of project
├── llm_judge.py                      # consolidated at end of project
└── outputs/                          # gitignored: adapters, predictions, audits
```

## Status

- Day 0 (bootstrap) — complete
- Day 1 (Notebook 01: setup + data exploration) — in progress
- Days 2–7 — pending
