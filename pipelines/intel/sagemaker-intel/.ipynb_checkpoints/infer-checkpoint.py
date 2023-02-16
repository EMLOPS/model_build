import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io
from PIL import Image
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"


transforms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")
    print("Eval Start")
    model.to(device).eval()
    print("Eval Complete")
    
    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    # assert request_content_type == "application/json"
    img_data = json.loads(request_body)["inputs"]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    data = transforms(np.array(img).astype(np.uint8)).unsqueeze(0).to(device)
    print("Data Preprocessing Complete")

    return data



# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
        prediction = F.softmax(prediction, dim=1)

    print(f"Generated the predictions Complete {prediction}")
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    print(f"Output is generated {res} , Json Dumps Output {json.dumps(res)}")
    return json.dumps(res)

