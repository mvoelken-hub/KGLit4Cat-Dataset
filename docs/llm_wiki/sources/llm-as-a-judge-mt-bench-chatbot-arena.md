---
type: source
title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
created: 2026-07-06
updated: 2026-07-06
sources: [llm-as-a-judge-mt-bench-chatbot-arena]
tags: [llm, evaluation, benchmark, llm-judge]
---

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica  
**Year:** 2023  
**Venue:** NeurIPS 2023 Datasets and Benchmarks Track / arXiv:2306.05685v4  
**File:** `docs/literature/judging-llm-as-a-judge-with-mt-bench-and-chatbot-arena-2306.05685.pdf`

## Abstract
Systematic study of using strong LLMs as judges for open-ended chatbot evaluation. The paper introduces MT-Bench, an 80-question multi-turn benchmark, and Chatbot Arena, a crowdsourced pairwise comparison platform. It compares LLM judges with expert and crowd human preferences, while also analyzing position bias, verbosity bias, possible self-enhancement bias, and limited reasoning ability.

## Key Findings
- MT-Bench targets multi-turn conversation and instruction following across categories such as writing, roleplay, extraction, reasoning, math, coding, knowledge, and humanities/social-science prompts.
- Chatbot Arena collects anonymous pairwise human preferences from real user interactions rather than fixed benchmark prompts.
- GPT-4 as a judge reaches more than 80% agreement with human preferences in the reported controlled and crowdsourced settings, roughly matching reported human-human agreement.
- LLM judges can suffer from position bias, especially when two answers are close in quality; swapping answer order or randomizing positions can reduce this risk.
- LLM judges can favor longer answers in verbosity-bias tests, even when added text is repetitive rather than substantively better.
- The paper also reports possible self-enhancement effects and examples where judges make wrong decisions on reasoning questions.
- LLM-as-a-judge is framed as scalable and explainable, but not as a replacement for all human evaluation.

## Methodology
- Uses pairwise comparison and single-answer grading prompts for LLM judges.
- Compares LLM judgments with expert labels on MT-Bench and crowd votes from Chatbot Arena.
- Uses bias probes for position effects and verbosity attacks.
- Releases MT-Bench questions, expert votes, and Chatbot Arena conversations through the FastChat judge resources.

## Limitations
- The main task is chatbot preference evaluation, not scientific fact checking or domain metadata curation.
- High agreement with aggregate human preferences does not imply that an LLM judge can prove scientific correctness.
- Biases and reasoning failures remain relevant even when strong models perform well on average.
- Reported mitigation strategies reduce some failure modes but do not remove the need for human review in high-stakes or domain-specialized evaluation.

## Relevance
Important source for SIMONE's evaluation framing. It supports the cautious claim that model-generated semantic requirement checks can be useful review aids, but should not be treated as scalar proof of curator-level correctness. It also gives concrete failure modes to mention when discussing LLM-based review artifacts: order sensitivity, preference for verbose output, and limited reasoning on difficult cases.
