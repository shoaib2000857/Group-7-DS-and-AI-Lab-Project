"""Reference kernels and upstream CLIP compatibility; isolated worker environment only."""
import torch
import torch.nn.functional as F

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """mamba_ssm.ops.selective_scan_interface.selective_scan_ref (real A, B/C: b n l)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B, C = B.float(), C.float()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)
    return (out, x) if return_last_state else out

def causal_conv1d_ref(x, weight, bias=None, activation=None):
    L = x.shape[-1]
    y = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1] - 1, groups=x.shape[1])[..., :L]
    return F.silu(y) if activation in ("silu", "swish") else y

def install_patches(device, shim=True):
    """shim=False keeps mamba_ssm/causal_conv1d kernels (CUDA parity run)."""
    import transformers.models.mamba.modeling_mamba as mm
    if shim:
        mm.selective_scan_fn = selective_scan_ref
        mm.causal_conv1d_fn = causal_conv1d_ref
    elif mm.selective_scan_fn is None or mm.causal_conv1d_fn is None:
        raise RuntimeError("mamba_ssm / causal_conv1d kernels not importable")
    for name in ("causal_conv1d_update", "selective_state_update", "mamba_inner_fn"):
        if not hasattr(mm, name) or getattr(mm, name) is None:
            setattr(mm, name, None)

    import clip.model as cm

    def vit_forward(self, x):  # README / PromptCC issue #3: also return patch tokens before ln_post
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        featuremap = x[:, 1:, :]
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x, featuremap

    cm.VisionTransformer.forward = vit_forward
    torch.Tensor.cuda = lambda t, *a, **k: t.to(device)
    torch.nn.Module.cuda = lambda m, *a, **k: m.to(device)
