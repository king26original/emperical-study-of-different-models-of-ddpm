'''run this file only after downloading the checkpoints or after training the models'''

import torch
from src.model.single_head_attention import model
from src.model.multi_head_attention import model_multi

model1=model()
model2=model()
model3=model()
model4=model_multi()

import copy
ema_model3=copy.deepcopy(model3)
ema_model4=copy.deepcopy(model4)

device="cuda" if torch.cuda.is_available() else "cpu"

model1=model1.load_state_dict(torch.load('checkpoints/model1', map_location=device))
model2=model2.load_state_dict(torch.load('checkpoints/model2', map_location=device))
ema_model3=ema_model3.load_state_dict(torch.load('checkpoints/ema_model3', map_location=device))
ema_model4=ema_model4.load_state_dict(torch.load('checkpoints/ema_model4', map_location=device))
