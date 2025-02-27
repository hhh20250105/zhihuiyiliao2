import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from torchvision import transforms

# 加载 NIfTI 格式图像
import cv2

# 加载图像
def load_image(file_path):
    # 使用OpenCV加载JPEG图像
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
    return img


# 图像预处理：低分辨率和高分辨率
def preprocess_image(image, low_res_size=(32, 32), high_res_size=(64, 64)):
    high_res = cv2.resize(image, high_res_size, interpolation=cv2.INTER_CUBIC)
    low_res = cv2.resize(high_res, low_res_size, interpolation=cv2.INTER_CUBIC)
    return low_res, high_res

# 自定义数据集类
# 自定义数据集类
class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.jpeg')]  # 只读取jpeg文件
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.data_dir, file_name)
        image = load_image(img_path)  # 使用加载jpeg图像的函数

        low_res, high_res = preprocess_image(image)

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res


# EDSR模型定义
class EDSR(nn.Module):
    def __init__(self, num_channels=1, num_feats=64, num_blocks=16, upscale_factor=2):
        super(EDSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=1)

        self.residual_blocks = nn.ModuleList([
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1) for _ in range(num_blocks)
        ])

        self.conv2 = nn.Conv2d(num_feats, num_channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_feats * upscale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        self.adjust_residual_channels = nn.Conv2d(num_feats, num_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        res = x

        for block in self.residual_blocks:
            x = block(x)

        x = self.conv2(x)

        res = self.adjust_residual_channels(res)

        x += res
        x = self.upsample(x)
        return x

# 训练模型
def train_model(model, train_loader, num_epochs=50, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for low_res, high_res in train_loader:
            low_res, high_res = low_res.float(), high_res.float()  # 使用float类型

            optimizer.zero_grad()
            output = model(low_res)

            loss = criterion(output, high_res)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    return train_losses

# 测试模型
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res, high_res = low_res.float(), high_res.float()  # 使用float类型
            output = model(low_res)

            output_img = output.cpu().squeeze(0).numpy()

            if len(output_img.shape) == 3:
                output_img = output_img[:, :, output_img.shape[2] // 2]

            high_res_img = high_res.cpu().squeeze(0).numpy()
            if len(high_res_img.shape) == 3:
                high_res_img = high_res_img[:, :, high_res_img.shape[2] // 2]

            plt.subplot(1, 2, 1)
            plt.imshow(output_img, cmap='gray')
            plt.title('Super-Resolution')

            plt.subplot(1, 2, 2)
            plt.imshow(high_res_img, cmap='gray')
            plt.title('Ground Truth')

            plt.show()

# 数据增强
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练和测试数据集
train_dataset = MedicalImageDataset(data_dir=r'F:/python/exam/ChestXRay2017/chest_xray/train', transform=transform)
test_dataset = MedicalImageDataset(data_dir=r'F:/python/exam/ChestXRay2017/chest_xray/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化EDSR模型
model = EDSR(num_channels=1, num_feats=64, num_blocks=16, upscale_factor=2).float()  # 使用float类型

# 训练模型
train_losses = train_model(model, train_loader, num_epochs=30)

# 绘制训练损失曲线
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# 测试模型
test_model(model, test_loader)
