import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 1. Сфиральный сверточный блок для сети
class SfiralBlock(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, bias=False
    )
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)

    # Топологическое разделение по ширине (W) на триединство
    b, c, h, w = out.shape
    if w < 4:
      return out

    mid = w // 2
    s_zone = 2

    left = out[:, :, :, :mid]
    s_node = out[:, :, :, mid : mid + s_zone]
    right = out[:, :, :, mid + s_zone :]

    # Физика витков: зеркальная антисимметрия
    p_left = left * 1.0
    p_right = right * -1.0
    p_s = s_node * 1.0

    min_w = min(p_left.shape[3], p_right.shape[3])
    combined = torch.cat(
        [
            p_left[:, :, :, :min_w],
            p_s,
            p_right[:, :, :, : min(p_right.shape[3], min_w)],
        ],
        dim=3,
    )
    return combined


# 2. Полноценная Сфиральная Сверточная Сеть
class SfiralCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = SfiralBlock(1, 16)
    self.layer2 = SfiralBlock(16, 32)
    self.fc = None  # бейзлайн инициализируем динамически

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)

    # Динамически определяем размерность для любого размера тензора
    if self.fc is None:
      flat_size = x.shape[1] * x.shape[2] * x.shape[3]
      self.fc = nn.Linear(flat_size, 10).to(x.device)

    x = x.view(x.size(0), -1)  # растягиваем в вектор
    x = self.fc(x)
    return x


# 3. Обучение на реальных данных (MNIST)
def train_model():
  print('=' * 50)
  print(' ОБУЧЕНИЕ СФИРАЛЬНОЙ СВЕРТОЧНОЙ СЕТИ НА MNIST')
  print('=' * 50)

  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
  )
  train_dataset = datasets.MNIST(
      root='./data', train=True, download=True, transform=transform
  )
  train_subset = torch.utils.data.Subset(train_dataset, range(2000))
  train_loader = torch.utils.data.DataLoader(
      train_subset, batch_size=64, shuffle=True
  )

  model = SfiralCNN()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  model.train()
  print('[+] Старт обучения по эпохам...')

  for epoch in range(2):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(
        f'Эпоха {epoch + 1} завершена. Ошибка (Loss):'
        f' {running_loss / len(train_loader):.4f}'
    )

  print(
      '[УСПЕХ] Сфиральная сверточная сеть успешно обучена на реальных данных!'
  )
  print('=' * 50)


if __name__ == '__main__':
  train_model()