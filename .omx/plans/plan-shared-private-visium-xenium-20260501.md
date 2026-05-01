# Shared/Private Visium-Xenium implementation plan

## Requirements Summary

Goal: extend the existing GLIP BRCA Visium + Xenium pseudospot joint-training line with a shared/private latent split that reduces platform distribution gap without erasing platform-specific signal.

Current codebase facts:
- `train_joint_brca_naive.py` already combines Visium and Xenium pseudospot datasets, wraps `platform_id`, trains a shared image-expression retrieval model, and evaluates Visium/Xenium retrieval metrics.
- `PlatformConditionedModel` already exposes `encode_images(...)`, `encode_spots(...)`, contrastive loss, image/gene OT loss, and platform-token support.
- `module_shared_private` already exists as a CLI/config toggle but is explicitly blocked as reserved/not implemented.
- Visium samples provide image, shared-gene expression, barcode, sample id, and spatial coordinates.
- Xenium pseudospots provide image, aligned encoder expression, pseudospot id, centroid coordinates, fold id, and cell count.
- `tools/compare_joint_embedding_umap.py` already computes platform separation diagnostics such as silhouette, centroid distance, and cross-domain cosine.

Primary implementation direction:
- Treat shared/private as Stage 3B, after the smaller Stage 3A platform-token module.
- First implement pseudospot-level shared/private alignment, not raw cell-level alignment.
- Preserve existing train/eval contracts so older checkpoints and Stage 2 baselines remain comparable.

## Decision

Implement an MVP named `shared_private_v1` inside the existing joint training path:

```text
image_base = image_projection(image_encoder(image))
gene_base  = spot_projection(reduced_expression)

image_shared, image_private = split(image_base, platform_id)
gene_shared,  gene_private  = split(gene_base, platform_id)

retrieval_embedding = normalize(shared + private_gate * private)
alignment_embedding = normalize(shared)
```

Use `shared` for cross-platform alignment diagnostics/losses. Use a controlled fused embedding for prediction retrieval so platform-specific useful signal is not thrown away.

## Why This Shape

This is the lowest-risk way to test the hypothesis. The current system is contrastive retrieval, not an autoencoder, so adding full decoders, cross-decoders, NB reconstruction, and adversarial classifiers in the first pass would change too many variables at once. The MVP should answer one question first: does an explicit shared/private bottleneck improve Visium/Xenium prediction and reduce platform separability compared with naive joint, platform token, and OT/UOT?

## Acceptance Criteria

1. `--module-shared-private true` runs without `NotImplementedError` in `train_joint_brca_naive.py`.
2. Existing Stage 2 behavior is unchanged when `--module-shared-private false`.
3. `joint_config.json`, `split_manifest.json`, `history.json`, and `metrics.json` record shared/private dimensions, loss weights, and active component losses.
4. A smoke run with small spot limits completes one epoch on CPU or GPU.
5. Full or pilot 5-fold runs produce Visium and Xenium metrics comparable to current launcher outputs.
6. UMAP/domain-gap tooling can compare at least two embedding views: `shared` and `fused`.
7. Success requires no catastrophic prediction regression and at least one domain-gap metric improvement in `shared` space versus the selected baseline.

## Implementation Steps

1. Freeze the comparator.

Use the current best available Stage 2/3A runs as immutable baselines:
- naive joint
- platform token
- image OT/UOT if already accepted as an exploratory comparator

Record run directories, config, commit, fold count, average Visium Pearson, average Xenium Pearson, and domain-gap metrics in a new ledger row before changing model behavior.

2. Add model parameters and config fields.

Add CLI args in `train_joint_brca_naive.py`:
- `--shared-private-dim`, default `256`
- `--private-dim`, default `64`
- `--private-gate`, default `0.25`
- `--shared-align-weight`, default `0.05`
- `--orth-weight`, default `0.01`
- `--private-domain-weight`, default `0.0` for MVP
- `--shared-domain-adv-weight`, default `0.0` for later adversarial phase

Keep defaults inert unless `--module-shared-private true`.

3. Implement the shared/private wrapper.

Create either `glip/joint/shared_private.py` or keep a small class near `PlatformConditionedModel` initially, then move it once stable.

Required modules:
- shared head: maps base projection dim to `shared_dim`
- private head: maps base projection dim plus optional platform token to `private_dim`
- private-to-shared adapter or fused projection: maps private dim back into shared retrieval dim
- optional platform-specific private heads via `ModuleList`, only if one shared private head underfits

Forward outputs should include:
- `image_shared`, `image_private`, `image_fused`
- `gene_shared`, `gene_private`, `gene_fused`
- component losses

4. Keep retrieval contract stable.

Update:
- `encode_images(...)`
- `encode_spots(...)`

Default return should remain the embedding used by existing retrieval code. For shared/private enabled, return `fused` by default. Add a keyword such as `embedding_view="fused"` or helper methods for diagnostics:
- `encode_images(..., embedding_view="shared")`
- `encode_spots(..., embedding_view="shared")`
- `embedding_view="private"` only for analysis, not prediction.

This avoids rewriting `collect_spot_bank(...)`, `collect_image_queries(...)`, and retrieval prediction in the first patch.

5. Add losses in small increments.

MVP losses:
- existing image-gene contrastive loss on `fused`
- shared image-gene contrastive loss with small weight, or reuse main contrastive on `shared` if prediction does not regress
- cross-platform shared alignment loss using existing OT/UOT helper on `image_shared` and `gene_shared`
- shared/private covariance penalty within each modality and platform

Do not add reconstruction, cross-decoder, or adversarial loss in MVP. Those should be Stage 3C after the split proves useful.

6. Update training metrics.

Extend `train_epoch(...)` meters and history records with:
- `shared_private_main_loss`
- `shared_align_loss`
- `orth_loss` or `cross_cov_loss`
- `shared_image_ot_loss`
- `shared_gene_ot_loss`
- active pair counts

The main risk is silent no-op training, so every enabled loss must be visible in `history.json`.

7. Extend launcher support.

Add passthrough args to `run_joint_5fold.py`:
- `--module-shared-private`
- dimension args
- loss-weight args
- `--private-gate`

Keep the default launcher behavior identical to current runs.

8. Extend embedding diagnostics.

Update `tools/compare_joint_embedding_umap.py` to collect:
- current/fused image embedding
- current/fused gene embedding
- shared image embedding
- shared gene embedding

Report platform separation separately for `shared` and `fused`. The expected result is: `shared` has lower platform separation; `fused` may retain slightly more platform separability if it improves prediction.

9. Add tests.

Add lightweight tests under `tests/` or a smoke script if the project does not currently use joint-training unit tests:
- shape test for shared/private heads
- no-op compatibility test when disabled
- forward pass with mixed `platform_id`
- loss component keys exist and are finite
- `encode_*` returns expected dimension for `fused` and `shared`

10. Run experiments in gated order.

Experiment order:
- `sp_v1_smoke`: 1 epoch, small Visium/Xenium limits, CPU/GPU sanity
- `sp_v1_fold00`: one fold, compare against same fold baseline
- `sp_v1_5fold`: full random5fold only if fold00 is not catastrophically worse
- `sp_v1_plus_uot`: combine shared/private with existing UOT only after standalone shared/private is understood

## Initial Hyperparameter Grid

Start narrow:

```text
shared_dim: 256
private_dim: 32, 64
private_gate: 0.0, 0.1, 0.25
shared_align_weight: 0.01, 0.05
orth_weight: 0.001, 0.01
```

Interpretation:
- `private_gate=0.0` tests pure shared prediction.
- `private_gate=0.1/0.25` tests whether preserving platform information improves retrieval.
- If `shared` domain gap improves but prediction drops, keep fused retrieval and report shared as alignment space.

## Risks and Mitigations

Risk: private branch absorbs all biological signal.
Mitigation: keep private dim small, add covariance penalty, and monitor whether `shared` still predicts expression.

Risk: shared branch collapses to platform-invariant but biologically weak embeddings.
Mitigation: keep image-gene contrastive pressure on shared and measure prediction from shared-only retrieval.

Risk: minibatches often contain only one platform, making cross-platform losses inactive.
Mitigation: add a balanced sampler or batch construction later if active-pair fraction is low; first measure active fraction.

Risk: adding decoders too early obscures whether shared/private helped retrieval.
Mitigation: defer reconstruction/cross-decoder/adversarial phases until MVP is benchmarked.

Risk: platform-token and shared/private effects are confounded.
Mitigation: test standalone shared/private first, then shared/private plus platform token as a separate ablation.

## Verification Steps

Run syntax/import checks:

```bash
python -m py_compile train_joint_brca_naive.py run_joint_5fold.py tools/compare_joint_embedding_umap.py
```

Run smoke training:

```bash
python train_joint_brca_naive.py \
  --run-dir runs_joint/sp_v1_smoke \
  --visium-fold-manifest configs/brca_visium_random5fold_seed42.json \
  --visium-fold-index 0 \
  --xenium-test-fold 0 \
  --model resnet50 \
  --pretrained false \
  --batch-size 4 \
  --epochs 1 \
  --max-visium-train-spots 32 \
  --max-visium-test-spots 16 \
  --max-xenium-train-spots 32 \
  --max-xenium-test-spots 16 \
  --module-shared-private true
```

Run one-fold benchmark:

```bash
python train_joint_brca_naive.py \
  --run-dir runs_joint/sp_v1_fold00 \
  --visium-fold-manifest configs/brca_visium_random5fold_seed42.json \
  --visium-fold-index 0 \
  --xenium-test-fold 0 \
  --module-shared-private true \
  --shared-align-weight 0.05 \
  --orth-weight 0.01
```

Run embedding comparison after fold00:

```bash
python tools/compare_joint_embedding_umap.py \
  --output-dir runs/sp_v1_embedding_compare_fold00 \
  --fold-index 0
```

## Stage Gate

Proceed to full 5-fold only if:
- smoke run passes with finite losses,
- fold00 Visium and Xenium Pearson do not show a catastrophic drop versus baseline,
- `shared` embedding has reduced platform separability by at least one metric,
- history confirms shared alignment/covariance losses are active when enabled.

Proceed to Stage 3C only if shared/private MVP is useful:
- add adversarial shared domain invariance,
- add private domain classifier,
- add gene reconstruction decoder,
- add cross-decoder translation only after reconstruction is stable.

## ADR

Decision: implement shared/private as an incremental wrapper around the existing contrastive retrieval model, not as a full autoencoder initially.

Drivers:
- existing code already supports joint Visium/Xenium pseudospot training and retrieval evaluation;
- `module_shared_private` is already reserved;
- experiment validity requires one core change at a time;
- prediction quality must remain the primary target, with distribution-gap reduction as supporting evidence.

Alternatives considered:
- Full shared/private autoencoder with reconstruction and cross-decoder: rejected for MVP because it changes model objective and evaluation too broadly.
- Cell-level Xenium private branch immediately: rejected because current joint path is pseudospot-based and cell-level alignment would add a second unresolved problem.
- Pure shared embedding for all prediction: kept as an ablation, but not the default because platform-private signal may be useful for platform-specific expression retrieval.

Consequences:
- The MVP is easier to compare to Stage 2 and Stage 3A.
- The plan may underuse the full theoretical shared/private design at first.
- If MVP succeeds, decoder/adversarial/cell-level extensions have a cleaner justification.

Follow-ups:
- Decide after fold00 whether private heads should be platform-specific.
- Decide after diagnostics whether alignment loss should apply to image_shared, gene_shared, or both.
- If minibatch cross-platform pairing is sparse, add a platform-balanced sampler before tuning loss weights.
