import torch
import torch.nn as nn

# Так как в app_visual.py используется стандартная архитектура ResNet, 
# импортируем её из torchvision или из вашего файла описания сети.
# Если вы используете модифицированный ResNet, импортируйте ваш класс:
import torchvision.models as models

# 1. Создаем экземпляр вашей модели (такой же, как при обучении)
# Например, ResNet-20 или стандартный ResNet под CIFAR-10
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10) # 10 классов CIFAR-10

# 2. Загружаем ваши обученные веса из папки checkpoints
checkpoint_path = "checkpoints/resnet_best.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Если веса сохранены через state_dict:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Веса успешно загружены!")
except Exception as e:
    print(f"Не удалось загрузить веса: {e}. Экспортируем структуру с инициализированными весами.")

model.eval()

# 3. Создаем тестовый тензор под размерность CIFAR-10 (батч 1, 3 канала, 32x32 пикселя)
dummy_input = torch.randn(1, 3, 32, 32)

# 4. Экспортируем в ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "sfiral_resnet.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Успех! Модель успешно экспортирована в sfiral_resnet.onnx")