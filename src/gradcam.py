"""Grad-CAM generico por gancho numa camada, para qualquer CNN do torchvision (ResNet-50, ConvNeXt-Tiny)."""
import torch
import torch.nn.functional as F


def gradcam_camada(modelo, camada, x, classe=None):
    """Grad-CAM generico para qualquer CNN do torchvision (usado no RoCoLe).

    modelo: rede com saida (1, n_classes); camada: modulo cuja saida e (1, C, h, w), por exemplo
    resnet50.layer4 ou convnext_tiny.features[-1]; x: tensor (1, 3, H, W).
    classe=None usa a classe predita. Devolve (mapa (H, W) em [0, 1], classe usada).
    Roda em fp32, com o modelo em eval(), mesmo se chamado dentro de torch.no_grad().
    """
    modelo.eval()
    ativacao = {}
    gancho = camada.register_forward_hook(lambda _m, _e, saida: ativacao.__setitem__("a", saida))
    try:
        with torch.enable_grad(), torch.autocast(x.device.type, enabled=False):
            # requires_grad na entrada garante o grafo mesmo com a rede congelada
            logits = modelo(x.detach().float().requires_grad_(True))
            if classe is None:
                classe = int(logits.argmax(1)[0])
            mapa = ativacao["a"]
            grad = torch.autograd.grad(logits[0, classe], mapa)[0]
    finally:
        gancho.remove()
    pesos = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * mapa).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), classe
