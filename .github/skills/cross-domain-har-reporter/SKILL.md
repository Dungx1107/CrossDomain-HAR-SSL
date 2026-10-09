---
name: cross-domain-har-reporter
description: Scientific Reporter for Cross-Domain K-shot Human Activity Recognition results.
---

# Cross-Domain HAR Reporter

Use this skill when the user asks for an academic report from a Cross-Domain HAR
result tree. The input is normally `outputs_evaluation/cross_k_shot/`, whose
records follow:

`{method}/{backbone}/{source}_to_{target}/{shot}/{protocol}/summary_*.json`

## Execution contract

1. Locate the result root supplied by the user; do not assume that missing
   directories or files contain zero-valued results.
2. Run:

   ```bash
   python scripts/04_results/cross_domain_har_reporter.py \
     --input <result-root> \
     --output section/Bao_Cao_Cross_Domain_HAR.md
   ```

3. Read every `summary_*.json` and `detailed_all_seeds.json`. The summary is
   authoritative for aggregate metrics; detailed seed files are used for seed
   distributions, validation, and fallback when an aggregate is absent.
4. Preserve the report's `N/A` values. Never impute, interpolate, or treat a
   missing transfer pair as zero.
5. Review the generated Markdown for unsupported claims. State counts of
   available pairs in every multi-method comparison and retain the report's
   warnings.

## Scientific rules

- The default metric is macro F1 in percent, reported as mean ± standard
  deviation across seeds.
- A transfer pair is complete only when all methods being compared have a
  record for the same source, target, shot, and protocol. Aggregate comparisons
  use only that intersection and explicitly report `used/available` pairs.
- Targets containing `hhar_watch` form an independent physical-bottleneck
  cluster. Exclude that cluster from overall pocket-level aggregates. Explain
  that wrist-worn and pocket/hip sensors have materially different kinematics;
  this is a physical domain shift, not merely an algorithmic deficit.
- Cover all three axes: method (`crosshar`, `prototype`, `contrastive`), backbone
  (`standard`, `cnn_transformer`), and target-domain bottleneck. Emphasize
  linear probing as the frozen-feature representation test.
- Warn when LP exceeds FT for a matched configuration and when any reported
  standard deviation is greater than 5 percentage points.
- Use past tense and academic, data-first language. Do not use emotional
  adjectives.

## Required report structure

The output must contain:

1. Executive summary (3–5 sentences).
2. Method × backbone × shot comparison table with protocol and `mean ± std`.
3. Three-axis analyses, each with a data table and interpretation.
4. A separate `*_to_hhar_watch` physical-bottleneck table and pocket-level
   comparison.
5. Confusion-matrix analysis based on `aggregated_confusion_matrix.png` and its
   JSON companion; name the most confused class pairs where available.
6. Conclusion and recommendations: best accuracy, best accuracy/compute
   trade-off (use complexity metadata when present and otherwise say that
   compute was not available), and future work.
7. At least four visualization specifications. The first three are mandatory:
   100-shot grouped bars with error bars, a shot learning curve, and a
   best-method source × target heatmap. The fourth is a seed-level boxplot.
   For each, state title, chart type, x/y axes, grouping/hue, message, and
   report placement.

Do not silently replace an unavailable visualization or metric with a proxy.
Label approximations and unavailable data explicitly.
