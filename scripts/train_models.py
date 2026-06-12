from src.model.multi_head_attention import model_multi
from src.model.single_head_attention import model

model1=model()
model2=model()
model3=model()
model4=model_multi()

import copy
ema_model3=copy.deepcopy(model3)
ema_model4=copy.deepcopy(model4)

from src.data.dataloader import trainloader
from src.diffusion.linear_beta_schedule import linear_alpha_bar
from src.training.train import train

losses1=train(
    model1,
    trainloader=trainloader,
    alpha_bar=linear_alpha_bar,
    ddpm_model_name='model1.pth',
    ema_model_name=False,
    ema_model=False
)

losses2=train(
    model2,
    trainloader,
    alpha_bar=cosine_alpha_bar,
    ddpm_model_name='model2.pth',
    ema_model_name=False,
    ema_model=False
)

from src.diffusion.cosine_beta_schedule import cosine_alpha_bar

losses3=train(
    model3,
    trainloader,
    alpha_bar=cosine_alpha_bar,
    ddpm_model_name=False,
    ema_model_name='ema_model3.pth',
    ema_model=ema_model3
)

losses4=train(
    model4,
    trainloader,
    alpha_bar=cosine_alpha_bar,
    ddpm_model_name=False,
    ema_model_name='ema_model4.pth',
    ema_model=ema_model4
)

torch.save(model1.state_dict(), 'model1.pth')
torch.save(model2.state_dict(), 'model2.pth')
torch.save(ema_model3.state_dict(), 'ema_model3.pth')
torch.save(ema_model4.state_dict(), 'ema_model4.pth')
