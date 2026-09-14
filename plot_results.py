import matplotlib.pyplot as plt

# Данные из наших логов обучения Сфирального ResNet (15 эпох)
epochs = list(range(1, 16))

train_loss = [
    1.6398,
    1.2248,
    1.0384,
    0.9311,
    0.8611,
    0.8021,
    0.7603,
    0.7298,
    0.6105,
    0.5754,
    0.5598,
    0.5494,
    0.5348,
    0.5303,
    0.5251,
]
val_loss = [
    1.4029,
    1.1627,
    1.0239,
    0.9743,
    0.9341,
    0.8441,
    0.8342,
    0.8894,
    0.5865,
    0.5829,
    0.5756,
    0.5713,
    0.5582,
    0.5546,
    0.5564,
]

train_acc = [
    38.47,
    55.46,
    62.74,
    66.82,
    69.56,
    71.76,
    73.23,
    74.21,
    78.53,
    79.89,
    80.35,
    80.84,
    81.25,
    81.26,
    81.56,
]
val_acc = [
    48.69,
    58.79,
    63.65,
    66.48,
    67.39,
    70.91,
    71.15,
    70.63,
    79.95,
    80.13,
    80.42,
    80.49,
    80.87,
    80.98,
    80.87,
]

# Исправлено: используем figsize вместо style
plt.figure(figsize=(12, 5))

# 1. График функции потерь (Loss)
plt.subplot(1, 2, 1)
plt.plot(
    epochs,
    train_loss,
    marker="o",
    linestyle="-",
    color="#1f77b4",
    label="Train Loss",
)
plt.plot(
    epochs,
    val_loss,
    marker="s",
    linestyle="--",
    color="#ff7f0e",
    label="Val Loss",
)
plt.title("Sfiral-ResNet: Loss Dynamics", fontsize=12, fontweight="bold")
plt.xlabel("Epochs", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# 2. График точности (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(
    epochs,
    train_acc,
    marker="o",
    linestyle="-",
    color="#2ca02c",
    label="Train Accuracy (%)",
)
plt.plot(
    epochs,
    val_acc,
    marker="s",
    linestyle="--",
    color="#d62728",
    label="Val Accuracy (%)",
)

# Аннотация пикового значения (80.98% на 14 эпохе)
plt.annotate(
    "Peak: 80.98%\n(Epoch 14)",
    xy=(14, 80.98),
    xytext=(10, 73),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
    fontsize=9,
    fontweight="bold",
)

plt.title("Sfiral-ResNet: Accuracy Dynamics", fontsize=12, fontweight="bold")
plt.xlabel("Epochs", fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()

# Сохранение готовой картинки в высоком разрешении для публикации
output_filename = "sfiral_resnet_training_curves.png"
plt.savefig(output_filename, dpi=300)
print(f"График успешно сохранен в файл: {output_filename}")

plt.show()