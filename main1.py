# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import cv2
import numpy as np
import base64


app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEVICE = "cpu"
MODEL = None

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8*8*8,32)
        self.fc2 = nn.Linear(32,2)
    def forward(self,x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)),2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
        out = out.view(-1,8*8*8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


def predict(image, model):
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformations_test = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)
                                      ])
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(32,32))
    img_as_tensor = transformations_test(img)
    s = nn.Softmax(dim=1)
    batch = img_as_tensor.unsqueeze(0)
    out = model(batch)
    print(model)
    fresh_percent = s(out)

    return int(fresh_percent[0][0].item()*100)


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            MODEL = Net()
            MODEL.load_state_dict(torch.load("FreshnessDetector.pt", map_location=torch.device(DEVICE)))
            pred = predict(image, MODEL)
            # In memory
            image_content = cv2.imencode('.jpg', image)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            return render_template("index.html", prediction=pred, image_loc=to_send)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    app.run()
