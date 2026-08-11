---
name: legal-issue-spotting
description: Identify legal issues in facts or documents and reconcile them with BERT taxonomy predictions.
---

# Legal Issue Spotting Skill

Use when raw facts, pleadings, correspondence, or document text must be converted into candidate legal issues.

Workflow:
1. Extract legally significant facts.
2. Review BERT top-k predictions if supplied.
3. Independently identify issues.
4. Reconcile classifier and independent candidates.
5. Map to taxonomy labels.
6. Preserve multiple plausible issues.
7. Identify defenses and missing facts.
8. Return confidence without deciding the merits.

Never treat classification as legal disposition.
