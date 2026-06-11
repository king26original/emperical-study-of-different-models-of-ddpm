import torch
from torch import nn

class model_multi(nn.Module):
  def __init__(
      self, 
      img_channels=3,
      base_channels=64,
      time_emb_dim=128,
      time_emb_scale=512,
      num_timesteps=1000,
      num_groups=8,
      num_heads=8
  ):
    super().__init__()

    self.num_heads=num_heads
    
    c1=base_channels
    c2=base_channels*2
    c3=base_channels*4
    c4=base_channels*8
    
    self.time_embedding=nn.Sequential(
        nn.Embedding(embedding_dim=time_emb_dim, num_embeddings=num_timesteps),
        nn.Linear(in_features=time_emb_dim, out_features=time_emb_scale),
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=time_emb_scale),
    )

    self.conv1=nn.Conv2d(in_channels=img_channels, out_channels=c1, kernel_size=3, stride=1, padding=1)

    #resblock1
    self.res1_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c1),
        nn.SiLU(),
        nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed1=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c1),
    )

    self.res1_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c1),
        nn.SiLU(),
        nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
    )

    #down sample 1
    self.down1=nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=2, padding=1)

    #resblock2
    self.res2_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed2=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c2),
    )

    self.res2_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1),
    )

    #down sample 2
    self.down2=nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=2, padding=1)

    #resblock3
    self.res3_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c3),
        nn.SiLU(),
        nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed3=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c3),
    )

    self.res3_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c3),
        nn.SiLU(),
        nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, stride=1, padding=1),
    )

    #down sample 3
    self.down3=nn.Conv2d(in_channels=c3, out_channels=c4, kernel_size=3, stride=2, padding=1)

    #resblock before attention
    self.res_att_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c4),
        nn.SiLU(),
        nn.Conv2d(in_channels=c4, out_channels=c4, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed_att=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c4),
    )

    self.res_att_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c4),
        nn.SiLU(),
        nn.Conv2d(in_channels=c4, out_channels=c4, kernel_size=3, stride=1, padding=1),
    )

    #attention
    self.att_norm=nn.GroupNorm(num_groups, c4)

    self.wq=nn.Linear(in_features=c4, out_features=c4)
    self.wk=nn.Linear(in_features=c4, out_features=c4)
    self.wv=nn.Linear(in_features=c4, out_features=c4)
    self.wo=nn.Linear(in_features=c4, out_features=c4)

    #upconv 1
    self.up1=nn.ConvTranspose2d(in_channels=c4, out_channels=c3, kernel_size=3, stride=2, padding=1, output_padding=1)

    #resblock 1 (in_channels doubled due to skip connection concat)
    self.res4_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c3 * 2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c3 * 2, out_channels=c3, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed4=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c3),
    )

    self.res4_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c3),
        nn.SiLU(),
        nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, stride=1, padding=1),
    )

    self.resx1=nn.Conv2d(in_channels=c3 * 2, out_channels=c3, kernel_size=1)

    #upconv 2
    self.up2=nn.ConvTranspose2d(in_channels=c3, out_channels=c2, kernel_size=3, stride=2, padding=1, output_padding=1)

    #resblock 2
    self.res5_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c2 * 2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c2 * 2, out_channels=c2, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed5=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c2),
    )

    self.res5_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1),
    )

    self.resx2=nn.Conv2d(in_channels=c2 * 2, out_channels=c2, kernel_size=1)

    #upconv 3
    self.up3=nn.ConvTranspose2d(in_channels=c2, out_channels=c1, kernel_size=3, stride=2, padding=1, output_padding=1)

    #resblock 3
    self.res6_1=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c1 * 2),
        nn.SiLU(),
        nn.Conv2d(in_channels=c1 * 2, out_channels=c1, kernel_size=3, stride=1, padding=1),
    )

    self.res_time_embed6=nn.Sequential(
        nn.SiLU(),
        nn.Linear(in_features=time_emb_scale, out_features=c1),
    )

    self.res6_2=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=c1),
        nn.SiLU(),
        nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
    )

    self.resx3=nn.Conv2d(in_channels=c1 * 2, out_channels=c1, kernel_size=1)

    self.conv_f=nn.ConvTranspose2d(in_channels=c1, out_channels=img_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x, t):
    # Initial Inputs
    # x shape: (B, 1, 32, 32)
    # t shape: (B,)
    
    t=self.time_embedding(t) 
    # t shape: (B, 512)

    xt=self.conv1(x) 
    # xt shape: (B, 64, 32, 32)

    # layer 1
    skip1=self.res1_2(self.res1_1(xt)+self.res_time_embed1(t)[:,:,None,None])+xt 
    # skip1 shape: (B, 64, 32, 32)
    xt=self.down1(skip1) 
    # xt shape: (B, 128, 16, 16)

    # layer 2
    skip2=self.res2_2(self.res2_1(xt)+self.res_time_embed2(t)[:,:,None,None])+xt 
    # skip2 shape: (B, 128, 16, 16)
    xt=self.down2(skip2) 
    # xt shape: (B, 256, 8, 8)

    # layer 3
    skip3=self.res3_2(self.res3_1(xt)+self.res_time_embed3(t)[:,:,None,None])+xt 
    # skip3 shape: (B, 256, 8, 8)
    xt=self.down3(skip3) 
    # xt shape: (B, 512, 4, 4)

    # bottleneck pre-attention
    xt = self.res_att_2(self.res_att_1(xt) + self.res_time_embed_att(t)[:,:,None,None])+xt
    # xt shape: (B, 512, 4, 4)

    # attention prep
    xt=self.att_norm(xt) 
    # xt shape: (B, 512, 4, 4)
    x_res=xt 
    # x_res shape: (B, 512, 4, 4)
    
    B, C, H_b, W_b=xt.shape
    seq_len=H_b*W_b # 4 * 4 = 16
    head_dim=C//self.num_heads # 512 // 8 = 64

    xt=xt.flatten(start_dim=2) 
    # xt shape: (B, 512, 16)
    xt=xt.permute(0,2,1) 
    # xt shape: (B, 16, 512)

    q=self.wq(xt) # q shape: (B, 16, 512)
    k=self.wk(xt) # k shape: (B, 16, 512)
    v=self.wv(xt) # v shape: (B, 16, 512)

    # reshape for multi-head setup
    q=q.reshape(B, seq_len, self.num_heads, head_dim) # q shape: (B, 16, 8, 64)
    k=k.reshape(B, seq_len, self.num_heads, head_dim) # k shape: (B, 16, 8, 64)
    v=v.reshape(B, seq_len, self.num_heads, head_dim) # v shape: (B, 16, 8, 64)

    # permute for scaled dot-product attention mapping
    q=q.permute(0, 2, 1, 3) # q shape: (B, 8, 16, 64)
    k=k.permute(0, 2, 1, 3) # k shape: (B, 8, 16, 64)
    v=v.permute(0, 2, 1, 3) # v shape: (B, 8, 16, 64)

    # compute attention matrix
    scale_factor=head_dim**0.5 
    att=torch.einsum('b h q d, b h k d -> b h q k', q, k) / scale_factor 
    # att shape: (B, 8, 16, 16)
    att=torch.softmax(att, dim=-1) 
    # att shape: (B, 8, 16, 16)

    # apply attention to values
    xt=torch.einsum('b h q k, b h k d -> b h q d', att, v) 
    # xt shape: (B, 8, 16, 64)
    
    # collapse heads back down
    xt=xt.permute(0, 2, 1, 3) 
    # xt shape: (B, 16, 8, 64)
    xt=xt.reshape(B, seq_len, C) 
    # xt shape: (B, 16, 512)
    
    xt=self.wo(xt) 
    # xt shape: (B, 16, 512)
    
    # restore spatial dimensions
    xt=xt.permute(0,2,1) 
    # xt shape: (B, 512, 16)
    xt=xt.unflatten(dim=-1, sizes=(H_b, W_b)) + x_res 
    # xt shape: (B, 512, 4, 4)

    # layer 1 (up)
    xt=self.up1(xt) 
    # xt shape: (B, 256, 8, 8)
    xt=torch.cat((xt, skip3), dim=1) 
    # xt shape: (B, 512, 8, 8) -- (256 from up1 + 256 from skip3)
    xt=self.res4_2(self.res4_1(xt)+self.res_time_embed4(t)[:,:,None,None])+self.resx1(xt) 
    # xt shape: (B, 256, 8, 8)

    # layer 2 (up)
    xt=self.up2(xt) 
    # xt shape: (B, 128, 16, 16)
    xt=torch.cat((xt, skip2), dim=1) 
    # xt shape: (B, 256, 16, 16) -- (128 from up2 + 128 from skip2)
    xt=self.res5_2(self.res5_1(xt)+self.res_time_embed5(t)[:,:,None,None])+self.resx2(xt) 
    # xt shape: (B, 128, 16, 16)

    # layer 3 (up)
    xt=self.up3(xt) 
    # xt shape: (B, 64, 32, 32)
    xt=torch.cat((xt, skip1), dim=1) 
    # xt shape: (B, 128, 32, 32) -- (64 from up3 + 64 from skip1)
    xt=self.res6_2(self.res6_1(xt)+self.res_time_embed6(t)[:,:,None,None])+self.resx3(xt) 
    # xt shape: (B, 64, 32, 32)

    # output
    out=self.conv_f(xt) 
    # out shape: (B, 1, 32, 32)

    return out
