import torch
import torch.nn as nn
import torchvision.models as models
import os

# 1. Создаем модель и загружаем веса
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

checkpoint_path = "checkpoints/resnet_best.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Веса успешно загружены!")
except Exception as e:
    print(f"⚠️ Ошибка загрузки весов: {e}")

model.eval()
dummy_input = torch.randn(1, 3, 32, 32)

# 2. Экспортируем (даже если PyTorch создаст .data файл)
output_filename = "model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    output_filename,
    export_params=True,
    opset_version=12,
    input_names=['input'],
    output_names=['output']
)

# 3. А теперь с помощью пакета onnx принудительно впекаем веса внутрь одного файла!
try:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    
    print("🔄 Объединяем внешние данные в единый файл model.onnx...")
    onnx_model = onnx.load(output_filename, load_external_data=True)
    # Сохраняем модель без разделения на внешние файлы
    onnx.save(onnx_model, output_filename)
    
    # Удаляем образовавшийся хвост .data, если он остался
    data_file = output_filename + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        
    print("🎉 Успех! Создан абсолютно чистый единый файл model.onnx со всеми весами внутри.")
except ImportError:
    print("💡 Установите пакет onnx для авто-объединения: pip install onnx")
except Exception as e:
    print(f"ℹ️ Примечание к слиянию: {e}")