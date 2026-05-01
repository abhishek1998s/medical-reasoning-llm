"""Medical Reasoning LLM — reusable utilities.

Contents
--------
data_formatting   Track A / Track B formatters and short-CoT truncation.
inference         Greedy generation with latency/token logging.
metrics           EM, ROUGE-L, BERTScore, sacreBLEU wrappers.
safety_rubric     Manual-audit data structures and CSV writer.
"""
