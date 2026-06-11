import matplotlib.pyplot as plt

def sample_model(ddpm, alpha, alpha_bar, posterior_variance, ema_model=False):
    if ema_model is False:
        ema_model=ddpm

    ddpm.to(accelerator.device)
    
    ema_model.eval()
    
    alpha=alpha.to(accelerator.device)
    alpha_bar=alpha_bar.to(accelerator.device)
    posterior_variance=posterior_variance.to(accelerator.device)
    
    t=999
    
    xt=torch.randn((1,3,32,32), device=accelerator.device)
    
    samples=[]
    
    with torch.no_grad():
      while t>=0:
        t_tensor=torch.tensor([t], device=accelerator.device)
        e_pred=ema_model(xt, t_tensor)
    
        if t>0:
          z=torch.randn_like(xt)
        else:
          z=torch.zeros_like(xt)
    
        xt=(1/torch.sqrt(alpha[t]))*(xt-((1-alpha[t])/torch.sqrt(1-alpha_bar[t]))*e_pred)+torch.sqrt(posterior_variance[t])*z
    
        if t%100==0:
          samples.append(xt[0].detach().cpu())
    
        t=t-1
    
    fig, axes=plt.subplots(1, len(samples), figsize=(20,3))
    
    for i, sample in enumerate(samples):
      img=sample.permute(1,2,0)
      img=(img+1)/2
      img=img.clamp(0,1)
      axes[i].imshow(img.numpy())
      axes[i].axis("off")
      axes[i].set_title(f"step {i}")
    
    plt.show()
