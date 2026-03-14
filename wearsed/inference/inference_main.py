'''
Inference code for the Apnea Detection Model on CFS dataset recordings
'''

from wearsed.inference.inference_post_process import post_process
from wearsed.inference.inference_load_data import load_datapoint, create_batch, SEQ_LENGTH, OVERLAP_WINDOW

from wearsed.models.attention_unet.AttentionUNet import AttentionUNet

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file = 'wearsed/training/attention_unet/output/final_default/f-0/model_final.pth'
input_edf = '/vol/sleepstudy/datasets/cfs/polysomnography/edfs/cfs-visit5-800002.edf'
input_hypnogram = '/vol/sleepstudy/datasets/cfs/predicted_hypnogram/cfs-800002-1.csv'

# Load Model
model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location=DEVICE))
model.eval()

# Inference
with torch.no_grad():

    # Load data
    hypnogram, spo2, ppg = load_datapoint(input_edf, input_hypnogram)
    x = create_batch(hypnogram, spo2, ppg).to(DEVICE)

    # Run model
    y_hat = model(x)
    prediction = torch.sigmoid(y_hat)
    prediction = prediction.cpu()[:, OVERLAP_WINDOW:SEQ_LENGTH-OVERLAP_WINDOW].flatten()

ahi, tst, event_list = post_process(prediction.numpy(), hypnogram.numpy())

print(f'AHI: {ahi:.2f} events/hour')
print(f'TST: {tst:.2f} hours')
