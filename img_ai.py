# ResNet 모델 사용하기

# 아래는 코드 일부만 떼온 것 (--> 이미지 특징 검출하기)
# 전체 resnet 코드를 돌려보기 -> 기본 구조 파악하고 필요한 부분 가져오기


'''
깊이가 매우 큰 네트워크를 효율적으로 학습할 수 있게 해주는 잔차연결(residual connection)을 도입한 모델
ResNet-18 : 18개의 층으로 이루어진 ResNet
'''
import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Model load
## pretrain된 Resnet model을 사용하기 위한 Model load
model = models.resnet18(pretrained=True) # pretrain(사전 훈련된) 모델을 불러온 후, eval모드로 바꾼 후 feature extraction 수행
layer = model._modules.get('avgpool') # (모델 객체 사용하여 원하는 layer 선택) avg pool layer 선택, Use the model object to select the desired layer
model.eval() # 평가 모드로 설정

# Image prep
## 정규화(Normalize) : image size 맞춰줌
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0,224, 0.225])
to_tensor = transforms.ToTensor() # PIL 이미지를 PyTorch tensor(다차원 배열)로 변환

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image 
    t_img = Variable(normalize(to_tensor(scaler(img)).unsqueeze(0)))
    # 3. Create a vector of zeros that will hold our feature vector 
    #   - 추출된 feature 담을 벡터 생성, avgpool layer의 output을 복사할 함수 연결
    # The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(3072) # 특정 벡터를 저장할 공간 정의 - 여기서는 512

    # 4. Define a fuction that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()

pic_one_vector = get_vector("./output/output.tif")
pic_two_vector = get_vector("./output/output2.tif")

# Using PyTorch Cosine Similarity(pytorch cosine 유사성)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
cos_sim = cos(pic_one_vector.unsqueeze(0),
              pic_two_vector.unsqueeze(0))

print('\nCosine similarity: {0}\n'.format(cos_sim))