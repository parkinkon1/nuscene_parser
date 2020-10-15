# 데이터셋에서 특정 1개의 샘플을 가져오는 함수
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset

import os
from PIL import Image
import pickle


class NusLoaderQ10(Dataset):  # 커스텀 데이터 셋 만들기
    def __init__(self, root):
        self.root = root
        self.maps = list(sorted(os.listdir(os.path.join(root, "Maps"))))
        self.ego_poses = list(sorted(os.listdir(os.path.join(root, "Ego_poses"))))

        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root, "Maps", self.maps[idx])



        map_path = os.path.join(self.root, "Maps", self.maps[idx])
        ego_pose_path = os.path.join(self.root, "Ego_poses", self.ego_poses[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        tree = ET.parse(boxes_path)
        root = tree.getroot()
        num_objs = len(root.findall('object'))
        for member in root.findall('object'):
            value = (
                root.find('folder').text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            boxes.append(value[1:])

            if value[0] == 'Daisy':
                label = 0
            elif value[0] == 'Dandelion':
                label = 1
            elif value[0] == 'Sunflower':
                label = 2
            elif value[0] == 'Rose':
                label = 3
            elif value[0] == 'Tulip':
                label = 4
            labels.append(label)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target





        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y





dataset = NusceneParserQ10()  # 객체 생성

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,  # 불러올 때 마다 데이터 섞기(순서외우기 못하게 하려고)
)


# ---------------------------------------------
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
        ))  # 이터레이션은 3! (가중치와 편향의 업데이트 횟수 per 1에포크)


