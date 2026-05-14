# Medical Reasoning LLM

Fine-tune a small open-weight LLM on
`OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B-V2` and compare visible short
clinical reasoning against answer-only training.

This is a learning artifact, not a clinical product, and it is not intended for
deployment or medical decision-making.

## Project Documents

- Full implementation spec:
  [docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md](docs/superpowers/specs/2026-05-02-medical-reasoning-llm-design.md)
- Phase-1 assignment design doc:
  [design_doc.md](design_doc.md)
- Assignment PDF:
  [Assignment_ Fine-Tune a LLM for Reasoning.pdf](Assignment_%20Fine-Tune%20a%20LLM%20for%20Reasoning.pdf)

## Quick Start: Local Tests

```powershell
pip install pytest transformers datasets
# Run pure-Python tests (no GPU required)
python -m pytest tests/test_metrics.py tests/test_safety_rubric.py -v --basetemp=outputs/pytest-tmp
# Run all tests (requires transformers + datasets installed)
python -m pytest tests/ -v --basetemp=outputs/pytest-tmp
```

The `--basetemp` flag avoids a Windows permission error with pytest's default
temp directory. The heavyweight GPU stack in `requirements.txt` is needed for
Kaggle/Colab training and evaluation notebooks, not for most local source-code
checks.

## Quick Start: Kaggle Training

Use a Kaggle notebook with GPU enabled and internet on:

```python
!git clone https://github.com/abhishek1998s/medical-reasoning-llm.git
%cd medical-reasoning-llm
!pip install -q -r requirements.txt
```

Run notebooks in order:

1. `notebooks/01_setup_and_data_exploration.ipynb`
2. `notebooks/02_train_trackB_answer_only.ipynb`
3. `notebooks/03_train_trackA_short_cot.ipynb`
4. `notebooks/04_inference_and_metrics.ipynb`
5. `notebooks/05_llm_judge_and_safety_review.ipynb`
6. `notebooks/06_report_and_comparison.ipynb`

Large artifacts go under `outputs/`, which is gitignored.

## Layout

```text
.
|-- configs/experiment_config.yaml
|-- docs/superpowers/
|-- notebooks/
|   |-- 01_setup_and_data_exploration.ipynb
|   |-- 02_train_trackB_answer_only.ipynb
|   |-- 03_train_trackA_short_cot.ipynb
|   |-- 04_inference_and_metrics.ipynb
|   |-- 05_llm_judge_and_safety_review.ipynb
|   `-- 06_report_and_comparison.ipynb
|-- src/
|   |-- data_formatting.py
|   |-- inference.py
|   |-- metrics.py
|   |-- safety_rubric.py
|   `-- splits.py
|-- tests/
|-- train_sft.py
|-- llm_judge.py
`-- outputs/
```

## Current Status

- Phase 1 design document: present.
- Track A/B formatting and deterministic splitting: implemented and tested.
- Training notebooks for Track B and Track A: present.
- Inference, metrics, judge, audit, and report notebooks: present.
- Remaining work requires GPU/API execution: train adapters, generate
  predictions, run judges, complete the manual audit, and fill final results.
