from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lesson:
    slug: str
    title: str
    outcome: str
    overview: str
    concepts: tuple[str, ...]
    mathematics: str
    code_focus: str
    experiment: str
    steps: tuple[str, ...]
    checks: tuple[str, ...]
    deliverable: str


LESSONS = [
    Lesson(
        "task",
        "1. Frame clause classification",
        "Define one prediction unit, a label taxonomy, and a measurable objective.",
        "A model cannot rescue an ambiguous task definition. Decide what one input represents, "
        "whether labels are mutually exclusive, which users rely on the result, and which errors "
        "matter before choosing an architecture.",
        ("Supervised learning", "Multiclass classification", "Baselines", "Scope and limitations"),
        r"The model estimates $p(y=k\mid x)$ for each passage $x$ and label $k$.",
        "contractlab/data.py",
        "Read ten passages and decide whether one label is sufficient for each. Record ambiguous and out-of-taxonomy cases.",
        (
            "Write the prediction unit in one sentence.",
            "List the allowed labels and an explicit unknown or review path.",
            "Choose a simple majority-class or keyword baseline.",
            "Describe one costly false positive and one costly false negative.",
        ),
        (
            "Why is a classifier prediction a hypothesis rather than a legal conclusion?",
            "When would this task need multi-label classification instead?",
        ),
        "A one-paragraph task specification with inputs, outputs, exclusions, baseline, and review boundary.",
    ),
    Lesson(
        "data",
        "2. Inspect data and prevent leakage",
        "Validate provenance, class balance, duplicates, and document-level grouping.",
        "Legal passages from one agreement share names, drafting style, and repeated language. Random "
        "row splits can leak those signals across partitions and inflate the apparent ability to generalize.",
        ("Licensing", "Class imbalance", "Duplicate detection", "Group-aware splits"),
        r"A split estimates generalization only when $contracts_{train}\cap contracts_{test}=\varnothing$.",
        "contractlab/data.py:split_by_contract",
        "Change the random seed and compare label counts while proving that no contract crosses partitions.",
        (
            "Inspect required columns, blank values, label counts, and provenance.",
            "Find exact and normalized text/label duplicates.",
            "Group by contract or template before splitting.",
            "Seal the test partition before making model decisions.",
        ),
        (
            "Why can two different clauses from the same contract still leak information?",
            "Which partition may influence checkpoint selection?",
        ),
        "A split audit showing row counts, label counts, and zero group overlap.",
    ),
    Lesson(
        "tokens",
        "3. Turn text into tokens",
        "Understand subword IDs, attention masks, padding, truncation, and sequence length.",
        "Transformers receive token IDs, not raw text. Contract terms, citations, defined names, and "
        "long provisos may fragment or be truncated, so preprocessing choices affect what evidence the model can see.",
        ("Vocabulary", "Subword tokenization", "Attention mask", "Dynamic padding"),
        r"A tokenizer maps text to IDs $[t_1,\dots,t_n]$ that index learned embedding vectors.",
        "contractlab/training.py:ClauseDataset",
        "Compare token counts and retained endings for short clauses and long limitation-of-liability clauses at two maximum lengths.",
        (
            "Tokenize representative short, median, and long passages.",
            "Measure the percentage truncated at each candidate maximum length.",
            "Inspect which words split into subwords.",
            "Choose a length using both coverage and memory evidence.",
        ),
        (
            "What does an attention-mask zero mean?",
            "Why can increasing maximum length reduce feasible batch size?",
        ),
        "A token-length table and a justified maximum sequence length.",
    ),
    Lesson(
        "model",
        "4. Load a pretrained encoder",
        "Attach a classification head to contextual representations and identify trainable parameters.",
        "Pretraining supplies language representations; fine-tuning adapts them to a closed label "
        "taxonomy. Domain pretraining may help, but it does not create legal authority or fix poor labels.",
        ("Transfer learning", "Encoder", "Classification head", "Trainable parameters"),
        r"The head produces logits $z=W h_{[CLS]}+b$ from a pooled passage representation.",
        "contractlab/training.py:run_training",
        "Compare full fine-tuning with a frozen encoder and explain the trainable-parameter and memory differences.",
        (
            "Read the upstream model card and license.",
            "Match the tokenizer to the checkpoint.",
            "Set the classifier head to the exact label count.",
            "Record model ID, revision, label map, and trainable parameters.",
        ),
        (
            "Which parameters begin randomly initialized?",
            "Why must label IDs remain stable between training and inference?",
        ),
        "A model-selection note covering task fit, license, size, and expected trade-offs.",
    ),
    Lesson(
        "loss",
        "5. Calculate loss and gradients",
        "Connect cross-entropy, class weighting, backpropagation, and optimizer updates.",
        "The loss converts incorrect logits into a training signal. Imbalance strategies change which "
        "errors matter most, so they must be evaluated on the same held-out split rather than assumed to help.",
        ("Logits", "Softmax", "Cross-entropy", "Class weights", "Gradient", "AdamW"),
        r"$L=-\frac{1}{N}\sum_i \log\frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}}$ and $\theta\leftarrow\theta-\eta\nabla_\theta L$.",
        "contractlab/training.py:run_training",
        "Run two learning rates or compare unweighted and class-weighted loss; inspect stability and rare-label behavior.",
        (
            "Trace logits, labels, scalar loss, backward pass, clipping, and optimizer step.",
            "Verify gradients are cleared at the intended time.",
            "Change one loss-related variable at a time.",
            "Compare macro-F1 and per-label recall, not training loss alone.",
        ),
        (
            "Why are logits passed directly to cross-entropy?",
            "How can aggressive class weights harm common-label precision?",
        ),
        "A controlled comparison with a prediction, observed metrics, and conclusion.",
    ),
    Lesson(
        "memory",
        "6. Fit training into GPU memory",
        "Use precision, accumulation, and clipping without changing the task definition.",
        "Model weights, optimizer state, gradients, and activations all consume memory. A 12 GB GPU can "
        "train useful encoders when batch size, sequence length, precision, and accumulation are chosen deliberately.",
        ("VRAM", "FP32, FP16, BF16", "Gradient accumulation", "Gradient clipping"),
        r"Effective batch size is $B_{device}\times steps_{accumulation}$ for one GPU.",
        "contractlab/modeling.py",
        "Hold effective batch size constant while changing device batch size and accumulation steps.",
        (
            "Estimate memory before starting the run.",
            "Choose precision supported by the device.",
            "Reduce device batch size before reducing evidence length blindly.",
            "Record peak memory, throughput, and effective batch size.",
        ),
        (
            "Does accumulation reduce activation memory for one forward pass?",
            "Why is gradient clipping different from gradient accumulation?",
        ),
        "A stable configuration for the target hardware with measured peak memory.",
    ),
    Lesson(
        "validation",
        "7. Validate and checkpoint",
        "Select a checkpoint without consulting the sealed test partition.",
        "Validation supports decisions during development. The test set estimates the final selected "
        "pipeline only when it remains untouched by tuning, checkpoint selection, and prompt changes.",
        ("Epoch", "Validation", "Scheduler", "Checkpoint", "Early selection", "Reproducibility"),
        r"The test set remains sealed while $\arg\max_{checkpoint}\ F1_{validation}$ selects the model.",
        "contractlab/training.py:evaluate_model",
        "Compare the last checkpoint with the best validation-macro-F1 checkpoint and explain which is eligible for final testing.",
        (
            "Evaluate after each epoch on the development split.",
            "Record checkpoint metric, seed, config, model revision, and taxonomy version.",
            "Save the best eligible checkpoint using a predeclared rule.",
            "Load that checkpoint once for final holdout evaluation.",
        ),
        (
            "Why is the last epoch not automatically the best checkpoint?",
            "What becomes invalid if test results drive another training change?",
        ),
        "A checkpoint-selection record that never uses test outcomes as a tuning signal.",
    ),
    Lesson(
        "metrics",
        "8. Evaluate beyond accuracy",
        "Interpret macro-F1, per-label behavior, confusion patterns, and uncertainty.",
        "Imbalanced legal taxonomies can produce attractive accuracy while rare doctrines fail entirely. "
        "Macro-F1 weights every label equally and must be paired with per-label counts and error inspection.",
        ("Precision", "Recall", "Macro-F1", "Confusion matrix", "Zero-F1 labels", "Error analysis"),
        r"$F1_k=2\frac{precision_k\cdot recall_k}{precision_k+recall_k}$; macro-F1 averages labels equally.",
        "contractlab/evaluation.py",
        "Find the weakest and most-confused labels, read representative passages, and classify each error cause.",
        (
            "Compare accuracy with macro precision, recall, and F1.",
            "Inspect support and zero-F1 labels.",
            "Read false positives and false negatives before proposing a fix.",
            "Separate data, taxonomy, truncation, and model-capacity failures.",
        ),
        (
            "Why can accuracy rise while macro-F1 falls?",
            "What does zero F1 mean when a label has test support?",
        ),
        "An error-analysis memo with evidence and one testable next experiment.",
    ),
    Lesson(
        "tracking",
        "9. Track reproducible experiments",
        "Record hypotheses, configuration, aggregate metrics, and artifacts without leaking text.",
        "A useful run is more than a score. Reproduction needs code, configuration, data lineage, "
        "versions, seed, hardware, logs, checkpoint rule, and a clear statement of what changed.",
        ("Run configuration", "W&B", "Local artifacts", "Privacy", "Experiment comparison"),
        r"A useful experiment changes one hypothesis-relevant variable while holding others fixed.",
        "contractlab/jobs.py",
        "Write a hypothesis before changing maximum length, run the comparison, and decide whether the result supports it.",
        (
            "Assign a descriptive run name and write the hypothesis first.",
            "Persist configuration and environment metadata.",
            "Log aggregate metrics without contract text or filenames.",
            "Compare runs on the same split and selection rule.",
        ),
        (
            "Which artifacts are necessary to reproduce a run?",
            "Why can online tracking be a confidentiality boundary?",
        ),
        "A reproducible run record with a one-variable comparison and privacy review.",
    ),
    Lesson(
        "assistant",
        "10. Integrate the checkpoint responsibly",
        "Route classifier evidence into bounded analysis while preserving provenance and human review.",
        "The encoder ranks candidates; it does not perform long-form legal reasoning. The application "
        "must reconcile those candidates with independent issue spotting, jurisdiction routing, source "
        "evidence, structured generation, abstention, and explicit uncertainty.",
        ("Inference", "Top-k routing", "Abstention", "Jurisdiction", "Structured output", "Human review"),
        r"Model confidence is $\max_k softmax(z)_k$; it is not legal certainty or calibrated risk.",
        "contractlab/legal_ai/orchestrator.py",
        "Compare a high-confidence route, a review-required route, and an abstention; verify that all preserve evidence and uncertainty.",
        (
            "Preserve top-k predictions, confidence, model version, and taxonomy version.",
            "Run independent issue spotting and determine the likely jurisdiction.",
            "Generate the required issue, elements, facts, counterarguments, caveat, and conclusion fields.",
            "Refuse fabricated authority and require human review of analysis and drafts.",
        ),
        (
            "Why must analysis confidence remain separate from classifier confidence?",
            "What should happen when jurisdiction-specific authority is unavailable?",
        ),
        "A traced request showing classifier hypotheses, route decision, structured result, caveats, and audit metadata.",
    ),
]