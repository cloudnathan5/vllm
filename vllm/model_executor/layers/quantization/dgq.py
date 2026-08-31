"""vLLM out-of-tree quantization plugin for the dgq int2-expert format.

v1 scope: 2-bit experts only (88% of params). Attention/dense-MLP/vision are
exported dequantized to bf16 by dgq.export_vllm. Registering "dgq" makes
`LLM(model=<exported dir>, quantization="dgq")` work on stock vLLM.

Import this module before creating the engine (e.g. via --quantization dgq
with VLLM_PLUGINS, or `import dgq.vllm_plugin` in the launcher script).
"""
from __future__ import annotations

import torch

try:
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts
except ImportError:  # vllm <= 0.25.x
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

GROUP = 64


def _unpack2(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 4,
                      dtype=torch.uint8, device=packed.device)
    v = out.view(*packed.shape[:-1], -1, 4)
    v[..., 0] = packed & 3
    v[..., 1] = (packed >> 2) & 3
    v[..., 2] = (packed >> 4) & 3
    v[..., 3] = packed >> 6
    return out


def _dq2(codes: torch.Tensor, scales: torch.Tensor, cb: torch.Tensor,
         dtype: torch.dtype) -> torch.Tensor:
    c = _unpack2(codes)
    w = torch.zeros(c.shape, dtype=dtype, device=c.device)
    for k in range(4):
        w += cb[:, k].to(dtype).view(-1, 1, 1) * (c == k).to(dtype)
    return w * scales.to(dtype).repeat_interleave(GROUP, dim=-1)


_dq2_c = torch.compile(_dq2, dynamic=False)


@register_quantization_config("dgq")
class DgqConfig(QuantizationConfig):
    """int2 codebook+group-scale experts; everything else unquantized."""

    @classmethod
    def get_name(cls) -> str:
        return "dgq"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        inst = cls()
        inst.format = (config or {}).get("format", "int2")
        return inst

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            # int4 for the decoder text stack's attention + dense MLP;
            # everything else (vision, router, embed projections) unquantized.
            if prefix.startswith("model.layers.") and (
                    ".self_attn." in prefix or ".mlp." in prefix):
                return DgqInt4LinearMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            if getattr(self, "format", "int2") == "vq4x256":
                return DgqVQMoEMethod(layer.moe_config)
            return DgqMoEMethod(layer.moe_config)
        return None


class DgqMoEMethod(UnquantizedFusedMoEMethod):
    """int2 experts: packed codes + fp16 group scales + per-expert codebook.

    create_weights registers packed buffers; apply dequantizes into transient
    bf16 w13/w2 and delegates to the parent's kernel path.
    """

    def create_weights(self, layer, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        self._shapes = (E, H, I)
        self._act_dtype = params_dtype

        def reg(name, shape, dtype):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype),
                                   requires_grad=False)
            layer.register_parameter(name, p)
            # no FusedMoE weight_loader attrs: checkpoint stores these tensors
            # whole under the exact param name, so the direct-match fallback
            # with default_weight_loader must handle them.

        reg("w13_codes", (E, 2 * I, H // 4), torch.uint8)
        reg("w13_scales", (E, 2 * I, H // GROUP), torch.float16)
        reg("w13_cb", (E, 4), torch.float16)
        reg("w2_codes", (E, H, I // 4), torch.uint8)
        reg("w2_scales", (E, H, I // GROUP), torch.float16)
        reg("w2_cb", (E, 4), torch.float16)

    def process_weights_after_loading(self, layer) -> None:
        # Build the moe kernel once with a dequantized copy. In resident mode
        # (DGQ_RESIDENT=1) the dense weights stay registered — correctness
        # reference / for backends that prepack at setup. Otherwise shrink
        # them to placeholders and rely on per-forward rebinding.
        import os
        from vllm.model_executor.utils import replace_parameter
        dt = torch.bfloat16
        if os.environ.get("DGQ_DEBUG") == "1":
            print(f"[dgq] {getattr(layer, 'layer_name', '?')} w13_cb[0]="
                  f"{layer.w13_cb[0].tolist()} codes[0,0,:4]={layer.w13_codes[0,0,:4].tolist()} "
                  f"scales mean={layer.w13_scales.float().mean().item():.6f}", flush=True)
        w13 = _dq2_c(layer.w13_codes, layer.w13_scales, layer.w13_cb, dt)
        w2 = _dq2_c(layer.w2_codes, layer.w2_scales, layer.w2_cb, dt)
        self._setup_kernel(layer, w13, w2)
        del w13, w2
        if os.environ.get("DGQ_RESIDENT") != "1":
            replace_parameter(layer, "w13_weight",
                              torch.empty(0, dtype=dt, device=layer.w13_codes.device))
            replace_parameter(layer, "w2_weight",
                              torch.empty(0, dtype=dt, device=layer.w2_codes.device))
        torch.cuda.empty_cache()

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        import os
        if os.environ.get("DGQ_RESIDENT") == "1":
            return super().apply(layer, x, topk_weights, topk_ids,
                                 shared_experts, shared_experts_input)
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            convert_to_unquantized_kernel_format,
        )
        dt = x.dtype
        w13 = _dq2_c(layer.w13_codes, layer.w13_scales, layer.w13_cb, dt)
        w2 = _dq2_c(layer.w2_codes, layer.w2_scales, layer.w2_cb, dt)
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend, moe_config=layer.moe_config,
            w13_weight=w13, w2_weight=w2)
        had13, hadw2 = layer.w13_weight, layer.w2_weight
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        try:
            return super().apply(layer, x, topk_weights, topk_ids,
                                 shared_experts, shared_experts_input)
        finally:
            layer.w13_weight, layer.w2_weight = had13, hadw2


def _vq_dq(codes, scales, cb, group, dtype):
    E, O, IV = codes.shape
    flat = codes.view(E, -1).long()
    vals = cb.to(dtype).gather(1, flat.unsqueeze(-1).expand(-1, -1, 4))
    w = vals.view(E, O, IV * 4)
    return w * scales.to(dtype).repeat_interleave(group, dim=-1)


_vq_dq_c = torch.compile(_vq_dq, dynamic=False)


class DgqVQMoEMethod(UnquantizedFusedMoEMethod):
    """Scaled vector quantization: dim-4/K-256 codebooks, group scales
    (gate_up g128, down g64)."""

    GU_GROUP, DN_GROUP = 128, 64

    def create_weights(self, layer, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition

        def reg(name, shape, dtype):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype),
                                   requires_grad=False)
            layer.register_parameter(name, p)

        reg("w13_vq", (E, 2 * I, H // 4), torch.uint8)
        reg("w13_scales", (E, 2 * I, H // self.GU_GROUP), torch.float16)
        reg("w13_cb", (E, 256, 4), torch.float16)
        reg("w2_vq", (E, H, I // 4), torch.uint8)
        reg("w2_scales", (E, H, I // self.DN_GROUP), torch.float16)
        reg("w2_cb", (E, 256, 4), torch.float16)

    def process_weights_after_loading(self, layer) -> None:
        import os
        from vllm.model_executor.utils import replace_parameter
        dt = torch.bfloat16
        w13 = _vq_dq_c(layer.w13_vq, layer.w13_scales, layer.w13_cb, self.GU_GROUP, dt)
        w2 = _vq_dq_c(layer.w2_vq, layer.w2_scales, layer.w2_cb, self.DN_GROUP, dt)
        self._setup_kernel(layer, w13, w2)
        del w13, w2
        if os.environ.get("DGQ_RESIDENT") != "1":
            replace_parameter(layer, "w13_weight",
                              torch.empty(0, dtype=dt, device=layer.w13_vq.device))
            replace_parameter(layer, "w2_weight",
                              torch.empty(0, dtype=dt, device=layer.w2_vq.device))
        torch.cuda.empty_cache()

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        import os
        if os.environ.get("DGQ_RESIDENT") == "1":
            return super().apply(layer, x, topk_weights, topk_ids,
                                 shared_experts, shared_experts_input)
        if os.environ.get("DGQ_FUSED", "1") == "1" and shared_experts is None:
            from vllm.model_executor.layers.quantization.dgq_kernels import (
                vq_moe_forward,
            )
            act = torch.nn.functional.gelu
            return vq_moe_forward(
                x, topk_weights, topk_ids,
                layer.w13_vq, layer.w13_cb.float(), layer.w13_scales.float(), self.GU_GROUP,
                layer.w2_vq, layer.w2_cb.float(), layer.w2_scales.float(), self.DN_GROUP,
                lambda t: torch.nn.functional.gelu(t, approximate="tanh"))
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            convert_to_unquantized_kernel_format,
        )
        dt = x.dtype
        w13 = _vq_dq_c(layer.w13_vq, layer.w13_scales, layer.w13_cb, self.GU_GROUP, dt)
        w2 = _vq_dq_c(layer.w2_vq, layer.w2_scales, layer.w2_cb, self.DN_GROUP, dt)
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend, moe_config=layer.moe_config,
            w13_weight=w13, w2_weight=w2)
        had13, hadw2 = layer.w13_weight, layer.w2_weight
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        try:
            return super().apply(layer, x, topk_weights, topk_ids,
                                 shared_experts, shared_experts_input)
        finally:
            layer.w13_weight, layer.w2_weight = had13, hadw2


GROUP4 = 64


def _dq4(codes, scales, zeros, dtype):
    q = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2,
                    dtype=torch.uint8, device=codes.device)
    v = q.view(*codes.shape[:-1], -1, 2)
    v[..., 0] = codes & 15
    v[..., 1] = codes >> 4
    s = scales.to(dtype).repeat_interleave(GROUP4, dim=-1)
    z = zeros.to(dtype).repeat_interleave(GROUP4, dim=-1)
    return s * (q.to(dtype) - z)


_dq4_c = torch.compile(_dq4, dynamic=False)


class DgqInt4LinearMethod(torch.nn.Module):
    """Weight-only int4 (asymmetric, group-64 along input) for nn.Linear-class
    vLLM layers. Packed along the input dim, so merged output-dim shard
    loading (qkv_proj, gate_up_proj) works with stock weight loaders."""

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        from vllm.model_executor.utils import set_weight_attrs
        out_total = sum(output_partition_sizes)
        I = input_size_per_partition
        codes = torch.nn.Parameter(
            torch.empty(out_total, I // 2, dtype=torch.uint8), requires_grad=False)
        scales = torch.nn.Parameter(
            torch.empty(out_total, I // GROUP4, dtype=torch.float16), requires_grad=False)
        zeros = torch.nn.Parameter(
            torch.empty(out_total, I // GROUP4, dtype=torch.float16), requires_grad=False)
        for name, p, pack in (("codes", codes, 2), ("scales", scales, GROUP4),
                              ("zeros", zeros, GROUP4)):
            layer.register_parameter(name, p)
            set_weight_attrs(p, {"output_dim": 0, "input_dim": 1,
                                 "packed_dim": 1, "pack_factor": pack,
                                 **extra_weight_attrs})

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        w = _dq4_c(layer.codes, layer.scales, layer.zeros, x.dtype)
        return torch.nn.functional.linear(x, w, bias)
