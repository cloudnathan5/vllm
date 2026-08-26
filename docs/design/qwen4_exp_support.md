# Qwen4Exp (Qwen3.8-Flash-Next) vLLM support — design notes

Target checkpoint: `Inferact/Qwen3.8-Flash-Next-NVFP4`
(quant of `Qwen/Qwen3.8-Flash-Next`, `modelopt` NVFP4)

`architectures: ["Qwen4ExpForConditionalGeneration"]`, `model_type: qwen4_exp`.
Not present in vLLM as of upstream `main` @ `8d301f0`.

Reference implementation: `transformers>=5.16.1`,
`src/transformers/models/qwen4_exp/` (`modular_qwen4_exp.py`, 1186 lines).

## Weight budget

| Component | Size | Placement |
| --- | ---: | --- |
| PLE n-gram table (`layers.1.ple.ple_embedding`) | 102.40 GB | host (offload) |
| MoE experts (NVFP4, 16 shards) | 67.99 GB | device |
| Dense / attention / embed / lm_head | 10.80 GB | device |
| MTP experts | 1.60 GB | device |
| **Total** | **182.78 GB** | |
| **Device resident (n-gram offloaded)** | **80.38 GB** | fits 96 GB w/ ~15.6 GB headroom |

The n-gram table is BF16 (embeddings are in the quant config `ignore` list), which is
why it dominates despite the checkpoint being NVFP4.

## Why the n-gram table offloads cleanly

`ngram_size=3`, `heads_per_ngram=8` -> 16 heads; `ple_embed_dim=2560` -> 160 dims/head.
`ngram_vocab_size_base=20e6` -> 320M rows x 160 x 2B = 102.40 GB (matches the shard exactly).

It is a single `nn.Embedding` gather, used by exactly one PLE layer at decoder layer 1,
and nowhere else. Per token the gather touches 16 rows = 5 KB:

| Case | Bytes moved | PCIe5 x16 |
| --- | ---: | ---: |
| decode bs=1 | 0.01 MB | 0.1 us |
| decode bs=256 | 1.31 MB | 20.5 us |
| prefill 8192 tok | 41.94 MB | 655 us |

The reference implementation already anticipates this placement:

```python
# We need explicit device placement here, as the embedding may be skipped from device_map completely
return self.ngram_embedding(ngram_ids.to(self.ngram_embedding.weight.device)).to(ngram_ids.device).flatten(-2)
```

Requires ~110 GB host RAM to hold the table pinned. Falling back to `mmap` off disk
works but pays NVMe latency on 16 random ~320 B reads per token.

## Port surface

Most of the architecture subclasses modules vLLM already implements
(`qwen3_5.py`, `qwen3_next.py` — `Qwen3_5Moe*`, `Qwen3Next*`). Genuinely new:

| Module | Ref lines | Notes |
| --- | ---: | --- |
| `Qwen4ExpTextQSAIndexer` | ~110 | Qwen Sparse Attention, micro-block top-k selection |
| `Qwen4ExpTextNGramEmbedding` | ~100 | hashed n-gram gather; offload target |
| `Qwen4ExpTextPLELayer` | ~75 | injects n-gram features into hyper-connection streams |
| `Qwen4ExpTextGatedResidual` | ~38 | per-branch read/write gating |

Reused as-is: Gated DeltaNet, MoE (512 experts, top-10, `moe_intermediate_size=640`),
RMSNorm(Gated), rotary, vision tower, MTP.

## Open work

- **QSA is the hard part.** vLLM's `qwen3_5.py` has no indexer or sparse-attention path;
  Qwen3.5 there uses dense attention. A block-sparse backend honoring the indexer's
  selection must be integrated with paged KV cache + attention metadata. Comparable in
  scale to the DeepSeek sparse-attention work.
- PLE offload needs a host-resident weight-loading path; vLLM's loader assumes device
  placement for `nn.Embedding`-like weights.
- 302,616 tensors in the checkpoint — loader throughput should be checked.
- Nothing here is validated: no Blackwell GPU was available in the authoring
  environment, so none of this has been executed.
