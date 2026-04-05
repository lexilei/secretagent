# Induction Pipeline Research Plan

**Date**: 2026-04-05
**Branch**: `autoresearch/apr4`
**Goal**: Maximize val accuracy on MUSR murder mysteries via induction pipeline improvements, then validate cross-benchmark.

## Current State

| Run | Config | Train Acc | Val Acc | Notes |
|-----|--------|-----------|---------|-------|
| Prior baseline | actions_structured_noname | 82.7% (iter 1) | 76.0% | Old pipeline, all 4 ptools used for val |
| Prior baseline | thoughts_structured_noname | 74.7% (iter 1) | 69.3% | Old pipeline |
| exp0 rerun | actions_structured_noname | 74.7% (iter 0) | 69.3% | High variance: same config, different ptool synthesized → 68% iter1 |

## Root Cause Analysis

The pipeline has **three core decisions** at each iteration:

| Decision | Current approach | Problem |
|----------|-----------------|---------|
| **What pattern to induce** | Most frequent action/thought category | Frequency != impact. "Summarize evidence" is frequent but redundant with what ReAct already does |
| **Whether to accept it** | Embedding orthogonality > 0.75 | Orthogonality != helpfulness. A ptool can be novel but useless (or harmful) |
| **When to stop** | Fixed iterations or convergence | No feedback loop — doesn't check if the ptool actually helped |

**Root cause of variance**: selecting by frequency, not by impact. Accepting based on novelty, not on empirical improvement.

## Hypotheses (ordered by expected impact x simplicity)

### H1: Residual/failure-based induction
Instead of inducing from ALL traces, analyze failure cases. Ask: "what reasoning step would fix the cases we're getting wrong?" This targets the actual error modes rather than reinforcing what the agent already does well.

### H2: Impact-based pattern selection
Rank patterns by success correlation, not raw frequency. Compute `freq_in_correct / freq_in_all` for each pattern category. Patterns that appear disproportionately in correct traces are more likely to be useful ptools.

### H3: Empirical probe validation
After synthesizing a candidate ptool, run a quick probe (15-20 cases) comparing `{current ptools}` vs `{current ptools + candidate}`. Only accept if probe accuracy >= baseline. Prevents bad ptools from entering the library.

### H4: Max ptools = 2
Hard cap at 2 ptools. Data consistently shows degradation past 2-3 ptools. ReAct struggles with >3 tools available.

### H5: Cross-benchmark generalization
Test the winning pipeline on MUSR object placement and True Detective to validate generalizability.

## Evaluation Protocol

- **Primary metric**: val_acc (75 held-out MUSR murder cases)
- **Secondary**: train_acc stability (low variance = good)
- **Control**: always compare against no-ptool baseline on same cases
- **Cross-benchmark**: once pipeline is validated, test on object placement + true detective

## Experiment Schedule

| # | Hypothesis | Change | Expected outcome |
|---|-----------|--------|-----------------|
| 1 | H2 | Impact-based selection (success correlation ranking) | More targeted ptools, less variance |
| 2 | H1 | Failure-based induction (analyze wrong cases only) | Ptools that fix actual error modes |
| 3 | H4 | Max ptools = 2 | Prevent over-investigation |
| 4 | H3 | Probe validation gate (15-case accuracy check) | Reject harmful ptools before full iteration |
| 5 | Best combo | Combine winning changes | Best overall val_acc |
| 6 | H5 | Cross-benchmark (object placement) | Validate generalizability |
| 7 | H5 | Cross-benchmark (true detective) | Validate generalizability |

## Success Criteria

- val_acc > 76% (beat prior best baseline) on murder mysteries
- Pipeline works without benchmark-specific tuning on at least one other benchmark
- Low variance: re-running same config gives similar results
