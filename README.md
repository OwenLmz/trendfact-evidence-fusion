# Evidence Fusion for Chinese Fact-Checking

The frozen pipeline retrieves 20 candidates with character-bigram BM25, scores
them with Qwen2.5-7B-Instruct, and uses a learned ranking model (Gate) to select
five candidates for a three-class evidence aggregator. The final classifier
averages the evidence probabilities with a three-seed Chinese RoBERTa ensemble
at equal weight. The task is SUPPORT, REFUTE, or NEI (not enough information).

The project uses a deterministic, group-aware **derived** split of
[TrendFact (Zhang et al., ACL 2026)](https://aclanthology.org/2026.acl-long.1219/)
(5,345 train, 1,147 development, 1,150 test). The original test produced:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Claim-only RoBERTa ensemble | 0.5191 | 0.4648 |
| Evidence branch | 0.5200 | 0.4515 |
| Frozen fusion | 0.5783 | 0.5076 |

Fusion improved Macro-F1 by 0.0428 over the claim-only ensemble
(original paired-bootstrap 95% CI: [0.0133, 0.0711]).
The preselected no-Gate fusion scored 0.5032 Macro-F1. The difference from the
full fusion was statistically inconclusive, and NEI F1 decreased from 0.2779
to 0.2645 relative to the claim model. In a later development-only diagnostic
using matched candidate text, the MiniCheck adaptation (0.5279 Macro-F1)
outperformed the Qwen adaptation (0.4453). These are specific to the tested
development recipe. See [frozen results](results/frozen_test.json) and
[reproduction instructions](REPRODUCE.md).

This archive covers the adopted TrendFact pipeline and its original component
checks. Source, configuration, tests, aggregate results and the Gate's numeric
parameters are included. The data and base models are obtained separately from
their original publishers. The original fine-tuned RoBERTa checkpoints have
individual file hashes for checkpoint-exact replay. The reproduction guide
also provides a public-input rebuild route that trains replacement
checkpoints and regenerates the evidence training cache.
