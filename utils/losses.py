
import torch
import torch.nn.functional as F


def nca(similarities, targets, scale=1, margin=0.6):
    
    targets = targets.long()
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)
    
    similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

    disable_pos = torch.zeros_like(similarities)
    disable_pos[torch.arange(len(similarities)),
                targets] = similarities[torch.arange(len(similarities)), targets]

    numerator = similarities[torch.arange(similarities.shape[0]), targets]
    denominator = similarities - disable_pos

    losses = numerator - torch.log(torch.exp(denominator).sum(-1))
    losses = -losses
    
    loss = torch.mean(losses)
    return loss

def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)

def distil_feature(feat1, feat2):
    loss = torch.tensor(0.).cuda()
    for a,b in zip(feat1, feat2):
        
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)
        
        a_h = a.sum(dim=3).view(a.shape[0], -1)
        b_h = b.sum(dim=3).view(b.shape[0], -1)
        a_w = a.sum(dim=2).view(a.shape[0], -1)
        b_w = b.sum(dim=2).view(b.shape[0], -1)
        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)
        print('cat', a.shape, b.shape)
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
        print(a.shape, b.shape)
        
        dis = torch.frobenius_norm(a - b, dim=-1).mean()
        loss += dis
    return loss/ len(feat1)
    
    
def diff_feature(feat1, feat2):
    loss = torch.tensor(0.).cuda()
    
    for layer, (f1, f2) in enumerate(zip(feat1, feat2), 0):
        # f1 : (b, c, h, w)
        # f2 : (b, c, h, w)
        
        f1 = f1.pow(2)
        f2 = f2.pow(2)
        f1 = f1.view(f1.shape[0], -1)
        f2 = f2.view(f2.shape[0], -1)
        
        f1 = F.normalize(f1, dim=1, p=2)
        f2 = F.normalize(f2, dim=1, p=2)

        dis = torch.frobenius_norm(f1 - f2, dim=-1).mean()
        loss += dis

    return loss/ len(feat1)


if __name__ == "__main__":
    feat1 = [torch.rand((2,3,16,16)), torch.rand((2,6,4,4))]
    feat2 = [torch.rand((2,3,16,16)), torch.rand((2,6,4,4))]
    
    lss1 = diff_feature(feat1, feat2)
    lss2 = distil_feature(feat1, feat2)
    print(lss1, lss2)
