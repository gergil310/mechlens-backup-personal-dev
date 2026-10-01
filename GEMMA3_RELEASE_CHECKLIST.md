# Gemma-3 Release Checklist

Minimum gates before tagging any release that includes Gemma-3 support, so that a
numerical mismatch of the v2.16.0 kind (RMSNorm epsilon placement, 1.3e-4 against a 1e-4
threshold, visible only on a longer prompt) does not ship. These gates supplement the
monthly release checklist; they do not replace it.

Status key: **[implemented]** exists on the branch; **[proposed]** not yet part of any
process or workflow.

---

## Gate 1 — Gemma-3 regression coverage is green on the tagged commit  [implemented]

- [ ] `tests/integration/test_gemma3_norm_eps_regression.py` runs under the integration
      job on the exact commit being tagged, and passes.
- [ ] It is not skipped, not marked `slow`, and not deselected.
- [ ] Its five cases are all present: RMSNorm component vs `Gemma3RMSNorm` on
      eps-dominated inputs; logits at three prompt lengths at 1e-4; per-layer residual
      stream vs HF hidden states at 1e-4.

What this proves: the conversion path's arithmetic matches HF on a tiny Gemma-3 with no
Hub access. It fails on all five cases when eps is moved outside the square root.

What it does not prove: anything about real weights or the real loader's config entries.

> Known state: the test passes locally and has been pushed. **No CI run has been observed
> yet.** The first green run is part of this gate, not a formality.

## Gate 2 — Real-checkpoint HuggingFace comparison within 1e-4  [proposed; cell implemented but unexecuted]

- [ ] Load `google/gemma-3-270m` through `HookedTransformer.from_pretrained_no_processing`
      in **fp32 on CPU**.
- [ ] Load the HF reference in **fp32 with eager attention**.
- [ ] Run a short prompt and a prompt of at least two hundred tokens through both, using
      the same token ids.
- [ ] Record the maximum absolute logit difference for each prompt.
- [ ] Both differences are **below 1e-4**.
- [ ] The release-morning report names the measured numbers, not just "green".

Rationale, from the incident record: the threshold is an fp32 number (a bf16 run does not
count); the v2.16.0 test passed on a short prompt and failed on a longer one; the
post-hotfix result was about 2.9e-5.

> Known state: the parity cell in `demos/Gemma3_Multimodal.ipynb` implements exactly this
> and defaults to the 270m checkpoint, but has been **syntax-checked only and never run**.
> It needs `HF_TOKEN` for the gated checkpoint. Watch the transformers 5 `dtype` keyword on
> first run. The notebook is not in the CI notebook matrix, so this gate is a manual run
> until that changes.

## Gate 3 — The comparison covers the path users hit  [proposed]

- [ ] Gate 2 is run through the HookedTransformer loader, because that is where the eps
      bug lived and where the norm is computed by TransformerLens rather than delegated to
      HF.
- [ ] A Bridge comparison in no-processing mode is **not** accepted as a substitute: it
      delegates the norm to HF's own module and passes trivially.

## Gate 4 — Forward-path changes reset Gates 1 and 2  [proposed]

- [ ] If any merged change since the last green run touches the forward path, hooks,
      normalization, attention, or the conversion path, Gates 1 and 2 are rerun on top of
      it. A result from before the change does not count.
- [ ] The same rule applies to any `transformers` pin bump.
- [ ] For the next cut specifically: confirm whether the gradient-hook refactor named as
      the v2.17.0 headline has merged, and rerun Gates 1 and 2 after it.

---

## Explicitly not required

- `verify_models` as a gate for this failure mode. Its Phase 3 logits-equivalence tolerance
  is 3e-2 and would have passed the original 1.3e-4 regression. Run it for breadth if
  desired, but it does not satisfy Gate 2.
- Reviving the skipped HookedTransformer acceptance suite. Worth doing, but a larger
  project than these gates.

---

## Open items carried alongside the gates (not blocking, must be recorded)

- [ ] Locate the strengthened Gemma-3 acceptance test described in the v2.16.1 hotfix
      record; it is not in this backup.
- [ ] Reproduce the `vocab_size` mismatch report from the HuggingFace Space discussion
      against the two Gemma-3 vocab sizes hard-coded in the loader.
- [ ] Add a real-checkpoint check with a prompt longer than the sliding window for the
      per-size local/global attention tables.
- [ ] Decide whether the Phase 3 tolerance in `verify_models` should be tightened.
- [ ] Update the Drive copy of `Gemma3_Multimodal.ipynb`, which is still the pre-change
      version.

---

## Release-morning summary line (fill in before tagging)

```
Gemma-3 gates — commit: ________  Gate 1 CI run: ________ (green/red)
Gate 2 gemma-3-270m fp32 eager: short ____ tokens max|dlogit| = ______ ; long ____ tokens max|dlogit| = ______  (threshold 1e-4)
Forward-path changes since last green run: ________  Gates rerun after them: yes / no
```
