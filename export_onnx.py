import torch
import torch.nn as nn
import torchvision.models as models

# 1. Создаем экземпляр модели (или ваш класс сфирального ResNet)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10) # 10 классов CIFAR-10

# 2. Загружаем ваши обученные веса из папки checkpoints
checkpoint_path = "checkpoints/resnet_best.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Веса успешно загружены из checkpoints/resnet_best.pt!")
except Exception as e:
    print(f"⚠️ Предупреждение при загрузке весов: {e}. Экспортируем структуру.")

model.eval()

# 3. Создаем тестовый тензор под размерность CIFAR-10 (батч 1, 3 канала, 32x32 пикселя)
dummy_input = torch.randn(1, 3, 32, 32)

# 4. Экспортируем модель в ОДИН единый файл model.onnx (без внешних .data файлов)
output_filename = "model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    output_filename,
    export_params=True,        # Сохранять обученные веса внутрь файла
    opset_version=12,          # Стабильная версия опесета для браузера
    do_constant_folding=True,  # Оптимизация констант
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
)

print(f"🎉 Успех! Модель успешно экспортирована в единый файл {output_filename} (без внешних зависимостей).")