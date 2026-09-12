# Recommended model and benchmark matrix

This document distinguishes three things that should not be pooled into one
analysis: pretraining revisions, post-training variants, and benchmark reward
families. The main outcome remains terminal task reward; model state and the
amount of generated text observed are experimental factors.

The identifiers below were checked against the public Hugging Face repositories
on 2026-09-12. OLMo publishes intermediate base-model weights as repository
branches, so the string after `@` is a Hugging Face **revision**, not a separate
model repository.

## Models and revisions to run

Run the matrix in two panels. The primary panel is the cleanest test of how the
same underlying model family changes across post-training. The secondary panel
asks when reward-relevant information becomes detectable during pretraining.

### Panel A: primary Base to RLVR comparison

| Analysis label | Model specification | Why include it |
|---|---|---|
| `base_final` | `allenai/Olmo-3-1025-7B@main` | Released pretrained base model and anchor for the post-training comparison |
| `think_sft` | `allenai/Olmo-3-7B-Think-SFT@main` | Effect of supervised reasoning demonstrations |
| `think_dpo` | `allenai/Olmo-3-7B-Think-DPO@main` | Effect of preference optimization after SFT |
| `think_rlvr` | `allenai/Olmo-3-7B-Think@main` | Effect of outcome-based RL with verifiable rewards |

These are the four variants already used in Experiment 1. The next run should
increase prompt diversity and rollout replication before adding more models.

### Panel B: sparse pretraining trajectory

| Analysis label | Model specification | Interpretation |
|---|---|---|
| `pretrain_early` | `allenai/Olmo-3-1025-7B@stage1-step10000` | Very early control; instruction reward may be nearly degenerate |
| `pretrain_middle` | `allenai/Olmo-3-1025-7B@stage1-step700000` | Midway through broad pretraining |
| `pretrain_final` | `allenai/Olmo-3-1025-7B@stage1-step1413814` | End of broad stage-1 pretraining |
| `midtrain_final` | `allenai/Olmo-3-1025-7B@stage2-step47684` | End of targeted midtraining |
| `base_final` | `allenai/Olmo-3-1025-7B@main` | Released final base model after long-context training |

This five-point panel is preferable to running every 1,000-step branch. It
separates broad pretraining, midtraining, and the released base model while
keeping the number of comparisons manageable. If a change appears between
stage 2 and `main`, add `stage3-step1000`, `stage3-step6000`, and
`stage3-step11921` in a follow-up to localize it. If a change appears within
stage 1, add `stage1-step350000` and `stage1-step1050000`.

Run the core eight unique weight states with:

```bash
python run_exp1.py \
  --model-specs \
    pretrain_early=allenai/Olmo-3-1025-7B@stage1-step10000 \
    pretrain_middle=allenai/Olmo-3-1025-7B@stage1-step700000 \
    pretrain_final=allenai/Olmo-3-1025-7B@stage1-step1413814 \
    midtrain_final=allenai/Olmo-3-1025-7B@stage2-step47684 \
    base_final=allenai/Olmo-3-1025-7B@main \
    think_sft=allenai/Olmo-3-7B-Think-SFT@main \
    think_dpo=allenai/Olmo-3-7B-Think-DPO@main \
    think_rlvr=allenai/Olmo-3-7B-Think@main \
  --output-dir outputs/exp1-model-trajectory
```

Do not interpret a near-zero reward at an early pretraining revision as absence
of a latent value signal. A very early language model may simply be unable to
produce instruction-following rollouts. For a fair mechanistic comparison,
also replay identical generated-token sequences through each compatible base
revision and compare their internal states without changing the visible text.

The full current branch list can be obtained without downloading weights:

```python
from huggingface_hub import HfApi

refs = HfApi().list_repo_refs("allenai/Olmo-3-1025-7B")
print("\n".join(ref.name for ref in refs.branches))
```

## Benchmarks in `Dolci-Think-RL-7B`

The pinned Dolci revision contains 102,014 training prompts from 15 named
sources. These are useful development distributions, but they are not all
scorable by the current code. `rewards.py` presently implements only the
IFEvalG constraint verifier.

| Reward family | Exact `original_dataset` value | Rows | Current status |
|---|---|---:|---|
| Instruction following | `hamishivi/IF_multi_constraints_upto5_filtered` | 29,813 | **Runnable now** with the current IFEvalG reward |
| Math | `hamishivi/omega-combined-no-boxed_filtered` | 15,000 | Add a math answer extractor and verifier |
| Math | `hamishivi/AceReason-Math_filtered` | 6,598 | Add a math answer extractor and verifier |
| Math | `hamishivi/rlvr_orz_math_57k_collected_filtered` | 2,999 | Add a math answer extractor and verifier |
| Math | `hamishivi/MathSub-30K_filtered` | 2,999 | Add a math answer extractor and verifier |
| Math | `hamishivi/DAPO-Math-17k-Processed_filtered` | 2,584 | Add a math answer extractor and verifier |
| Code | `hamishivi/rlvr_acecoder_filtered_filtered` | 10,107 | Add sandboxed unit-test execution |
| Code | `hamishivi/klear-code-rlvr_filtered` | 6,272 | Add sandboxed unit-test execution |
| Code | `hamishivi/synthetic2-rlvr-code-compressed_filtered` | 3,000 | Add sandboxed unit-test execution |
| Multi-subject | `hamishivi/virtuoussy_multi_subject_rlvr_filtered` | 7,106 | Add task-specific exact-match verifiers |
| General | `hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered` | 7,109 | Add its reference/string-F1 scorer |
| General | `hamishivi/new-wildchat-english-general_filtered` | 6,421 | Requires a judge or explicit rubric; defer for the main claim |
| General | `hamishivi/llama-nemotron-rlvr-difficulty-6_filtered` | 1,121 | Add the source benchmark's verifier |
| General | `hamishivi/llama-nemotron-rlvr-difficulty-7_filtered` | 657 | Add the source benchmark's verifier |
| General | `hamishivi/llama-nemotron-rlvr-difficulty-8_filtered` | 228 | Add the source benchmark's verifier |

Recommended order:

1. Expand the existing instruction-following sample first, stratifying by
   constraint family and difficulty rather than taking the first streamed rows.
2. Add OMEGA Math next. Its objective terminal answer gives a substantially
   different reward family without introducing an LLM judge.
3. Add AceCoder or KlearReasoner Code after a safe execution-based verifier is
   available.
4. Use the multi-subject mixture as a breadth check only after its heterogeneous
   scorers are validated.
5. Defer WildChat and other judge-scored general prompts. A noisy judge reward
   would make it harder to tell whether the model internal represents value or
   merely predicts evaluator preferences.

Each reward family should be analyzed separately. A pooled score across binary
unit tests, fractional instruction constraints, and math exact match has no
single behavioral meaning.

## Held-out OLMo evaluations

The Dolci sources above were part of post-training data, so they cannot by
themselves establish out-of-distribution recovery of terminal reward. After
developing the method, use held-out suites reported for OLMo 3:

- instruction following: IFEval and IFBench;
- math: MATH, AIME 2024/2025, and OMEGA held-out splits;
- code: HumanEval+, MBPP+, and LiveCodeBench;
- broader reasoning: BBH and ZebraLogic.

For the strongest claim, choose one benchmark from each of instruction
following, math, and code, preserve each benchmark's native verifier, and split
by problem or constraint family. MemoryArena is a useful later agentic-memory
extension, but it is not part of the Dolci mixture and requires step-level state,
action, and reward logging.

Sources: [OLMo 3 base model card](https://huggingface.co/allenai/Olmo-3-1025-7B),
[OLMo 3 release and evaluation overview](https://allenai.org/blog/olmo3), and
[Dolci-Think-RL-7B dataset card](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B).
