import os
import glob
import torch
from models.resnet import resnet_cifar
from config import cfg

def export():
    print("🔄 Ищем последний чекпоинт в папке checkpoints...")
    ckpt_files = glob.glob("checkpoints/resnet_*.pt")
    
    if not ckpt_files:
        raise FileNotFoundError("⚠️ В папке checkpoints не найдены файлы весов (resnet_*.pt).")
    
    ckpt_path = sorted(ckpt_files)[-1]
    print(f"📥 Загружаем веса из: {ckpt_path}")

    # Создаем модель в точности как при обучении
    model = resnet_cifar(cfg.n_blocks_per_stage, cfg.stage_channels, cfg.num_classes)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Извлекаем state_dict с учетом ключа "model_state" из utils.py
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Убираем префикс 'module.', если модель обучалась в DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # Создаем тестовый тензор под размерность CIFAR-10 (1, 3, 32, 32)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    onnx_path = "model.onnx"
    print(f"🔄 Экспортируем в {onnx_path}...")
    
    # Экспорт с гарантированной упаковкой весов внутрь единого файла
    export_kwargs = {
        "export_params": True,
        "opset_version": 12,
        "do_constant_folding": True,
        "input_names": ['input'],
        "output_names": ['output'],
        "dynamic_axes": {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    }
    
    try:
        # Пробуем передать параметр подавления внешних данных, если поддерживается
        torch.onnx.export(model, dummy_input, onnx_path, external_data=False, **export_kwargs)
    except TypeError:
        # Фолбек для стандартного вызова
        torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)

    print(f"🎉 Успех! Актуальные веса из {os.path.basename(ckpt_path)} успешно зашиты в единый {onnx_path}!")

if __name__ == "__main__":
    export()