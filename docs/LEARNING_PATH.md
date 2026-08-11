# Learning path

## Module 1 — Define the task

Turn contract review into supervised clause classification. Establish the unit
of prediction, label taxonomy, baseline, and evaluation criteria.

## Module 2 — Know the data

Inspect provenance, licensing, label balance, duplicates, document groups, and
text length. Learn why examples from one contract must not cross data splits.

## Module 3 — Represent text

Understand subword tokenization, token IDs, attention masks, truncation, padding,
and the model's maximum sequence length.

## Module 4 — Fine-tune an encoder

Load pretrained weights, attach a classification head, calculate cross-entropy,
backpropagate gradients, update parameters, and validate without gradients.

## Module 5 — Train reliably

Use reproducible seeds, mixed precision, gradient accumulation, clipping,
learning-rate scheduling, checkpointing, and experiment tracking.

## Module 6 — Evaluate honestly

Use macro F1 for imbalanced classes, inspect per-class recall, plot a confusion
matrix, and read individual errors. Do not tune against the test set.

## Module 7 — Deploy into the assistant

Load the best checkpoint, segment a new contract, classify its passages, retain
source spans, and distinguish model output from heuristic rules.

## Module 8 — Extend the system

Add extraction, LoRA-based explanation, playbooks, and retrieval-backed question
answering as distinct, testable components rather than one opaque chatbot.
