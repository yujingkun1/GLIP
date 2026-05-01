# Stage 3 next module decision

## Candidate order from plan
1. platform-conditioned decoder / token
2. shared/private latent split
3. anti-collapse
4. OT alignment

## Decision
Proceed next with:
- **Stage 3A = platform-conditioned token / platform-conditioned decoding signal**

## Why this first
- smallest conceptual step beyond naive joint training
- lowest coupling compared with shared/private or OT
- directly tests whether explicit modality identity helps the joint space
- preserves the one-core-change rule

## Execution consequence
The next implementation lane should add platform-conditioning only, without introducing shared/private, OT, gene completion, or cell-level refinement at the same time.
