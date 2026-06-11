import copy

@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
  ema_params=dict(ema_model.named_parameters())
  model_params=dict(model.named_parameters())

  for k in ema_params.keys():
    ema_params[k].data.mul_(decay).add_(model_params[k].data, alpha=1-decay)
