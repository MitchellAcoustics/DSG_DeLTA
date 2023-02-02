#%%
import sys, os, argparse
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.models_pytorch import TinyCNN
from framework.config import event_labels
from framework.data_generator import DataGenerator
from framework.processing import evaluate_asc_aec
from pathlib import Path
import torch

import torch.optim as optim

#%%

# Loading TinyCNN model

# Saved model path
model_path = Path().cwd().parent.joinpath('application', 'sys_1e05_b64', 'model.pth')

# Need to instantiate the model structure. Pull this from Yuanbo's code, models_pytorch.py
model = TinyCNN(len(event_labels), batchnormal=True)

# Load dict of model state
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Also need to instantiate the optimizer. Pull this from Yuanbo's code, processing.py
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
# optimizer.load_state_dict(checkpoint['optimizer'])


#%%
# Evaluate model
generator = DataGenerator(batch_size=64, normalization=True)
generate_func = generator.generate_validate(data_type='validate')
va_rate_mse_loss, va_event_auc = evaluate_asc_aec(model=model,
                                                    generator=generator,
                                                    data_type='train',
                                                    cuda=0)


# %%

import numpy as np
from framework.models_pytorch import move_data_to_gpu
from torch import sigmoid
import pandas as pd

def predict(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    for data in generate_func:
        (batch_x, batch_y, batch_y_event) = data # will need to replace when I have new data without y values
        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)
            batch_rate, batch_output_event = all_output[0], all_output[1]

            batch_output_event = sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())#
            outputs_event.append(batch_output_event.data.cpu().numpy())

    dictionary = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dictionary['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dictionary['output_event'] = outputs_event

    res_df = pd.DataFrame(dictionary['output_event'], columns=event_labels)
    res_df['annoyance'] = dictionary['output'][:,0]


    return res_df




# %%
if __name__ == "__main__":

    generate_func = generator.generate_validate(data_type='validate')
    res = predict(model, generate_func, 0)
    res
# %%

