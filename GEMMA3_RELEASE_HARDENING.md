# Gemma-3 Release Hardening Audit

Scope: the v2.16.0 Gemma-3 numerical mismatch, the v2.16.1 hotfix, and the risk of a
similar mismatch reaching the next release. Sources of record are the project's own
materials (Drive notes, release tracker, release notes, maintainer and GitHub threads) and
the repository backup examined in this review. Upstream TransformerLensOrg release pages
are not a source for this audit.

Everything below is labelled as one of:

- **Established** — stated in the project materials or observed directly in the backup.
- **Verified in this review** — executed and observed during this review, offline.
- **Unverified** — implemented but not executed, or not yet checked at all.

---

## 1. Incident record

### v2.16.0 release (2026-06-25)

- Tagged on the last-Thursday cadence. Headline change: the Gemma-3 family adapter
  (PR #1112, merged 2026-06-17, authored by Bram). *Established.*
- Cut followed the release checklist: green demo-notebook sweep the night before; Yusuke
  reported green CI on 3.10 / 3.11 / 3.12 and a green acceptance suite across the matrix
  before the tag. *Established.*

### The mismatch (issue #1121, filed 2026-06-26)

- A user reported Gemma-3 forward-pass logits not matching the HF reference, with other
  adapters fine. Filed inside the 48-hour monitoring window. *Established.*
- Triage recorded the divergence at about 1.3e-4 maximum absolute on the smallest
  checkpoint, against the 1e-4 acceptance threshold. *Established.*
- The v2.16.0 acceptance test had passed on a prompt where the error stayed just under
  threshold. A longer prompt exposed it. *Established.*
- **Two magnitudes are on record and were not reconciled.** The reporter cited a mismatch
  above 1e-2 on the 2B checkpoint; triage measured 1.3e-4 on the smallest checkpoint. They
  were measured on different checkpoints. This audit does not reconcile them. *Established.*

### Root cause

- Norm-epsilon placement in the Gemma-3 conversion table. Gemma applies RMSNorm as
  `x * rsqrt(mean(x^2) + eps) * (1 + weight)` with eps inside the rsqrt. The v2.16.0 table
  placed eps outside (in Bram's words: "eps applied after the scale instead of before").
  *Established.*
- The architecture note names the per-layer local/global attention flag as the other easy
  trap and states it did not cause this miss. *Established.*
- In the backup examined here, eps lives in the RMSNorm forward rather than in
  `convert_gemma_weights`, so the hotfix diff could not be mapped line for line. The record
  describes the location as "the conversion table" and this audit leaves it there.
  *Established (limitation).*

### v2.16.1 hotfix (2026-06-27)

- Out-of-band hotfix, two days after v2.16.0, two PRs merged. Moved eps inside the rsqrt and
  strengthened the acceptance test with a longer prompt and an intermediate-activation
  check. *Established.*
- Yusuke confirmed the acceptance suite green with the updated Gemma-3 test before the tag.
  PyPI published the same morning. The release note states logits match HF to 1e-4.
  *Established.*
- Post-hotfix maximum delta recorded at about 2.9e-5. The July 7 sync records
  post-release monitoring as clean. *Established.*
- **The 2.9e-5 figure comes from the June notes.** Nothing in this review ran against real
  Gemma-3 weights. *Unverified on current code.*

### Unanswered follow-ups

Issues #1119 (is 2.16.1 safe to pin), #1120 (docs upgrade note), #1123 (version to pin
for a paper), and a HuggingFace Space discussion asking whether 2.16.1 resolves a Gemma-3
`vocab_size` mismatch have no recorded response. The `vocab_size` question is the only one
not plainly covered by the eps fix. *Established.*

---

## 2. Regression protection as found

### What the backup contained before this review

- No Gemma-3 numeric test at any tier. The Gemma-3 unit tests check adapter wiring,
  conversion classes, and mocked config dicts. The weight-conversion test checks shapes and
  the plus-one offset against `nn.LayerNorm` stand-ins and never runs a forward pass.
  *Verified in this review.*
- No unit test for the HookedTransformer `RMSNorm` component. *Verified in this review.*
- The only HF-parity integration test is parametrized on `gpt2` (LayerNorm), so the RMSNorm
  path is never compared to HF. The Gemma-2 forward and hook tests are skipped under `CI`.
  *Verified in this review.*
- The HookedTransformer acceptance module is skipped at module level ("CI test
  pollution"). Nothing in acceptance names Gemma-3. *Verified in this review.*
- The strengthened acceptance test described in the hotfix record is not in the backup.
  Its current location is unconfirmed. *Unverified.*
- `verify_models` is not a CI gate, and its Phase 3 logits-equivalence benchmark uses a
  3e-2 absolute and relative tolerance. It would have reported the 1.3e-4 regression as a
  pass. *Verified in this review (source inspection).*

### Mutation result

The full unit suite was run offline twice: once unchanged, once with eps moved outside the
square root at runtime in both the HookedTransformer RMSNorm and the Bridge's
self-computed norm path. Results were identical (2868 passed; the same 23 failures and 135
errors, all blocked Hub downloads). **No pre-existing test detects the bug.**
*Verified in this review.*

---

## 3. Hardening work: status

### Implemented and verified

- **`tests/integration/test_gemma3_norm_eps_regression.py`** (commit `755615f`).
  Builds a tiny random `Gemma3ForCausalLM` with non-trivial norm weights, loads it through
  the real conversion path (`convert_gemma_weights` plus the `from_pretrained_no_processing`
  load flags), and compares against HF with eager attention in fp32:
  - RMSNorm component vs `Gemma3RMSNorm` on eps-dominated inputs, at 1e-6;
  - logits at three prompt lengths (8, 32, and four short of `n_ctx`), at 1e-4;
  - per-layer residual stream vs HF hidden states, at 1e-4.

  Passes on current code. All five cases fail with the eps bug injected (logit divergence
  0.13 to 0.31). Needs no Hub access. Format, isort, pycln, and mypy clean.
  *Verified in this review.*

### Implemented, not executed

- **Parity cell in `demos/Gemma3_Multimodal.ipynb`** (commit `690b187`). Loads the real
  `google/gemma-3-270m` checkpoint through `from_pretrained_no_processing` in fp32 with
  eager attention, compares against HF on a short prompt and a prompt of roughly two
  hundred tokens, prints the max logit difference, and asserts the 1e-4 threshold.
  Syntax-checked only; never run (gated checkpoint, Hub blocked in the review
  environment). First-run risks: the transformers 5 `dtype` keyword and token access. The
  notebook is not in the CI notebook matrix. The Drive copy of the notebook is still the
  pre-change version. *Unverified.*
- **CI on the branch.** Both commits are pushed; no workflow run has been observed. That
  the new test executes under the integration job is an expectation from the tier layout,
  not an observed fact. *Unverified.*

### Proposed only

- The release gates in `GEMMA3_RELEASE_CHECKLIST.md`. Not in the release checklist, the
  workflow, or any process document.
- A `verify_models` run on gemma-3-270m and gemma-3-1b-it before the next cut. The
  registry still carries March results for those entries.
- Any change to the Phase 3 tolerance or to gating `verify_models` in CI.
- Real-checkpoint coverage of the per-size local/global attention tables in the loader.
- Investigation of the `vocab_size` report against the two Gemma-3 vocab sizes hard-coded
  in the loader.
- Locating the original strengthened acceptance test.

---

## 4. Remaining risks, in priority order

1. **The new regression coverage is unproven in CI.** The committed test is the only
   hardening shown to catch the bug, and only locally. *Needs: one green CI run on the
   tagged commit.*
2. **Nothing numeric runs against a real Gemma-3 checkpoint before a tag.** The committed
   test uses a tiny random model and a hand-mirrored config; a drift that depends on the
   real loader's config entries would pass it. *Needs: one gated run of the notebook cell
   or an equivalent fp32, eager, long-prompt comparison on gemma-3-270m.*
3. **The gates that let v2.16.0 through are unchanged.** CI, the acceptance suite, and the
   demo sweep were all green at tag time. *Needs: a process decision, not a run.*
4. **`verify_models` cannot see a regression of this size** (3e-2 tolerance, not a CI
   gate). *Needs: a decision on tolerance and gating.*
5. **The local/global attention flag has partial coverage only.** The committed test
   exercises a mixed sliding/global pattern on the tiny model with a prompt longer than the
   window, but not the real per-size layer tables. *Needs: a real-checkpoint comparison
   with a prompt longer than the sliding window.*
6. **v2.17.0 is headlined by a cross-cutting gradient-hook refactor** (July 7 sync). The
   materials stop on July 9; whether it landed and what it touched is unknown. *Needs:
   confirmation, then parity rerun on top of it.*
7. **The June user threads are unanswered**, and the `vocab_size` report is not explained
   by the fix. *Needs: reproduction against the loader's two vocab sizes.*

---

## 5. Statements this audit deliberately does not make

- That the two reported mismatch magnitudes agree.
- That the hotfix's strengthened acceptance test still exists.
- That the post-hotfix 2.9e-5 result has been observed on the current code.
- That the fix lives in a specific file of this repository.
- That monitoring was the gate that caught the bug. A user filed within a day; the gates
  intended to catch it were green.
- That any Gemma-3 parity has been measured on real weights during this review.
