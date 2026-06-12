from accelerate import Accelerator
import random
import torch
from torch import nn

accelerator=Accelerator(mixed_precision="fp16")

def train(
    ddpm,
    trainloader,
    alpha_bar,
    ddpm_model_name,
    ema_model_name=False,
    ema_model=False
):
    
    if ema_model is not False:
        ema_model.to(accelerator.device)
        ema_model.eval()
        
    optimizer=torch.optim.AdamW(params=ddpm.parameters(), lr=2e-4)
    loss_fn=nn.MSELoss()
    
    losses=[]
    
    
    epochs=500

    ddpm, optimizer, trainloader=accelerator.prepare(ddpm, optimizer, trainloader)
    alpha_bar=alpha_bar.to(accelerator.device)
    
    ddpm.train()
    for epoch in range(epochs):
      total_loss=0
    
      for X, y in trainloader:
    
        t=torch.randint(0, 1000, (X.shape[0],), device=accelerator.device)
    
        e=torch.randn_like(X)
        xt=torch.sqrt(alpha_bar[t])[:,None,None,None]*X+torch.sqrt(1-alpha_bar[t])[:,None,None,None]*e
        e_pred=ddpm(xt, t)
    
        optimizer.zero_grad()

        loss=loss_fn(e, e_pred)
    
        accelerator.backward(loss)
        optimizer.step()

        if ema_model is not False:
            update_ema(ema_model, ddpm)
        total_loss+=loss.item()
    
      avg_loss=total_loss/len(trainloader)
        
      losses.append(avg_loss)

      torch.save(ddpm.state_dict(), ddpm_model_name)  
      if ema_model is not False:  
          torch.save(ema_model.state_dict(), ema_model_name)
          
      print(f"epoch={epoch}; avg_loss={avg_loss}")

    return losses
