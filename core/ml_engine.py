import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from PIL import Image

RESIZED_VALUE = 224
MEAN_VALUE = [0.485, 0.456, 0.406]
STD_VALUE = [0.229, 0.224, 0.225]
IMG_WEIGHT = 0.60
TEXT_WEIGHT = 0.40
MINIMUM_SCORE = 85.0

model_resnet = models.resnet50(pretrained=True)
model_resnet.fc = nn.Identity()
model_resnet.eval()

model_nlp = SentenceTransformer("all-MiniLM-L6-v2")

transformation_img = transforms.Compose([
    transforms.Resize((RESIZED_VALUE, RESIZED_VALUE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE)
])

def get_image_similarity(img_lost: Image.Image, img_found: Image.Image) -> float:
    tensor_lost = transformation_img(img_lost).unsqueeze(0)
    tensor_found = transformation_img(img_found).unsqueeze(0)
    
    with torch.no_grad():
        feature_lost = model_resnet(tensor_lost)
        feature_found = model_resnet(tensor_found)
        sim_score = F.cosine_similarity(feature_lost, feature_found)
        score_persentage = (sim_score.item() + 1) / 2 * 100
        
    return score_persentage

def get_text_similarity(text_lost: str, text_found: str) -> float:
    vector_lost = model_nlp.encode(text_lost, convert_to_tensor=True)
    vector_found = model_nlp.encode(text_found, convert_to_tensor=True)
    
    sim_score = util.cos_sim(vector_lost, vector_found)
    return sim_score.item() * 100

def get_final_score(img_score: float, text_score: float) -> float:
    return (img_score * IMG_WEIGHT) + (text_score * TEXT_WEIGHT)