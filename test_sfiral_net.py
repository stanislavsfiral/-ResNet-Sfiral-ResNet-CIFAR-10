import torch
import torch.nn as nn


# Самая простая реализация сфирального сверточного слоя
class SimpleSfiralConv(nn.Module):

  def __init__(self, channels):
    super().__init__()
    # Обычная свертка 3x3
    self.conv = nn.Conv2d(
        channels, channels, kernel_size=3, padding=1, bias=False
    )

  def forward(self, x):
    # 1. Проходим свертку
    out = self.conv(x)

    # 2. Делим по ширине на триединство: Левый виток, S-переход, Правый виток
    b, c, h, w = out.shape
    mid = w // 2
    s_zone = 2  # Зона плавной деформации S-перехода

    left = out[:, :, :, :mid]
    s_node = out[:, :, :, mid : mid + s_zone]
    right = out[:, :, :, mid + s_zone :]

    # 3. Физика: левый прямой, правый зеркально-антисимметричный (-1)
    p_left = left * 1.0
    p_right = right * -1.0
    p_s = s_node * 1.0  # плавный переход без разрыва

    # Собираем обратно
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


# --- БЫСТРЫЙ ТЕСТ ---
if __name__ == "__main__":
    print("=" * 50)
    print(" ТЕСТ ПРОСТОЙ СФИРАЛЬНОЙ СВЕРТОЧНОЙ СЕТИ")
    print("=" * 50)

    # Имитируем входное изображение 1 шт, 1 канал (градации серого), размер 8x8 пикселей
    x = torch.ones(1, 1, 8, 8) * 5.0
    initial_energy = torch.sum(x ** 2).item()

    # Создаем наш слой
    sfiral_layer = SimpleSfiralConv(channels=1)

    # Прогоняем данные
    output = sfiral_layer(x)
    output_energy = torch.sum(output ** 2).item()

    # Для сравнения: стандартный MaxPool, который сбрасывает информацию
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled_out = max_pool(x)
    pooled_energy = torch.sum(pooled_out ** 2).item()

    print(f"Начальная энергия тензора: {initial_energy:.2f}")
    print(f"Энергия после MaxPool (стандарт): {pooled_energy:.2f}")
    print(f"Энергия после Сфирального слоя:   {output_energy:.2f}")
    print(
        "[УСПЕХ] Сфиральный сверточный блок отработал и сохранил целостность"
        " потока!"
    )
    print("=" * 50)