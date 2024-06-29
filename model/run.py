from Unet import build_model

import torch
import os
import cv2
import numpy as np
import base64
import sys
import os

torch.cuda.empty_cache()
def run():
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] >500:
        scale = 1/int(img.shape[0]/500)
        img = cv2.resize(img,None,fx = scale,fy = scale,interpolation=cv2.INTER_AREA)
    
    m,n = img.shape
    final_res = torch.zeros(img.shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = build_model().to(device)
    model_filename = "./Model/Model/Model.pt"
    state = torch.load(model_filename,map_location=torch.device(device))
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    batch = []
    skip = 24

    for i in range(0,m,skip):
        for j in range(0,n,skip):
            if(i+48 > m) or (j+48 > n):
                continue

            cimg = img[i:i+48,j:j+48]

            if(sum(sum(cimg==0) == 48**2)):
                continue
            with torch.no_grad():
                out = model(torch.Tensor(cimg).unsqueeze(0).unsqueeze(0).to(device))
            

            final_res[i:i+48,j:j+48] = final_res[i:i+48,j:j+48]+out[0][0].cpu().detach().numpy()

    final_res = ((final_res/(final_res.view(-1).max() + 1e-5))*255).numpy().astype('uint8')

    _, buffer = cv2.imencode('.jpg', final_res)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    print(encoded_image)
if __name__ == "__main__":
    run()