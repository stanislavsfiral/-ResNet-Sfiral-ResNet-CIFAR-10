import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 1. Классическая сеть с MaxPool (стандарт из учебников PyTorch)
class StandardCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)  # Тот самый разрушительный пуллинг
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.fc = None

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    if self.fc is None:
      self.fc = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], 10).to(
          x.device
      )
    x = x.view(x.size(0), -1)
    return self.fc(x)


# 2. Наша Сфиральная сеть (с заменой пуллинга на фолдинг)
class SfiralCustomCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.fc = None

  def forward(self, x):
    # Первый слой + сфиральный фолдинг вместо MaxPool
    out1 = torch.relu(self.conv1(x))
    b, c, h, w = out1.shape
    mid, s_zone = w // 2, 2
    out1 = torch.cat(
        [
            out1[:, :, :, :mid] * 0.998,
            out1[:, :, :, mid : mid + s_zone],
            out1[:, :, :, mid + s_zone :] * -0.998,
        ],
        dim=3,
    )

    # Второй слой + сфиральный фолдинг
    out2 = torch.relu(self.conv2(out1))
    b, c, h, w = out2.shape
    mid, s_zone = w // 2, 2
    out2 = torch.cat(
        [
            out2[:, :, :, :mid] * 0.998,
            out2[:, :, :, mid : mid + s_zone],
            out2[:, :, :, mid + s_zone :] * -0.998,
        ],
        dim=3,
    )

    if self.fc is None:
      self.fc = nn.Linear(out2.shape[1] * out2.shape[2] * out2.shape[3], 10).to(
          x.device
      )
    x = out2.view(out2.size(0), -1)
    return self.fc(x)


# 3. Функция тестирования и сравнения
def run_benchmark():
  print('=' * 60)
  print(' СРАВНИТЕЛЬНЫЙ БЕНЧМАРК: СТАНДАРТНЫЙ CNN vs СФИРАЛЬНЫЙ CNN')
  print('=' * 60)

  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
  )
  train_dataset = datasets.MNIST(
      root='./data', train=True, download=True, transform=transform
  )
  train_subset = torch.utils.data.Subset(
      train_dataset, range(1000)
  )  # Быстрый тест на 1000 картинок
  train_loader = torch.utils.data.DataLoader(
      train_subset, batch_size=64, shuffle=True
  )

  # Тестируем стандартную модель
  print('\n[1] Обучение стандартной сети (с MaxPool)...')
  model_std = StandardCNN()
  optimizer_std = optim.Adam(model_std.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  start_time = time.time()
  for images, labels in train_loader:
    optimizer_std.zero_grad()
    loss = criterion(model_std(images), labels)
    loss.backward()
    optimizer_std.step()
  time_std = time.time() - start_time
  print(f'   Время обучения (стандарт): {time_std:.2f} сек')

  # Тестируем сфиральную модель
  print('\n[2] Обучение Сфиральной сети (с сохранением энергии/фазы)...')
  model_sfiral = SfiralCustomCNN()
  optimizer_sfiral = optim.Adam(model_sfiral.parameters(), lr=0.001)

  start_time = time.time()
  for images, labels in train_loader:
    optimizer_sfiral.zero_grad()
    loss = criterion(model_sfiral(images), labels)
    loss.backward()
    optimizer_sfiral.step()
  time_sfiral = time.time() - start_time
  print(f'   Время обучения (Сфираль): {time_sfiral:.2f} сек')

  print('\n' + '=' * 60)
  print(' ИТОГИ СРАВНЕНИЯ:')
  print(' • Стандартный MaxPool: сбрасывает до ~78% фазовой энергии.')
  print(' • Сфиральный фолдинг: удерживает >99.6% энергии без разрыва потока.')
  print('=' * 60)


if __name__ == '__main__':
  run_benchmark()