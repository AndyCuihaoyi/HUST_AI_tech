import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# 设置随机种子确保可复现性
torch.manual_seed(1)

# 超参数
EPOCH = 3  # 训练轮数
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True  # 自动下载数据集
MODEL_PATH = 'cnn2.pkl'  # 模型保存路径

# 下载/加载MNIST数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=torchvision.transforms.ToTensor()
)

# 批训练加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 修复测试数据加载（兼容新版本PyTorch）
test_x = test_data.data[:2000].unsqueeze(1).type(torch.FloatTensor) / 255.0
test_y = test_data.targets[:2000]


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# 初始化模型（无CNN结构打印）
cnn = CNN()

# 优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# ===================== 新增：记录训练过程的准确率和损失 =====================
train_metrics = {
    'epochs': [],  # 记录轮数
    'steps': [],  # 记录步数
    'losses': [],  # 记录损失值
    'accuracies': []  # 记录准确率（百分比）
}

# 强制训练模型
print("\n🚀 开始训练模型...")
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每50步打印训练状态并记录指标
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].numpy()
            accuracy = float((pred_y == test_y.numpy()).sum()) / float(test_y.size(0)) * 100  # 百分比

            # 记录训练指标
            train_metrics['epochs'].append(epoch)
            train_metrics['steps'].append(step)
            train_metrics['losses'].append(loss.item())
            train_metrics['accuracies'].append(accuracy)

            # 打印训练状态（百分比准确率）
            print(f'Epoch: {epoch}/{EPOCH} | Step: {step} | Loss: {loss.item():.4f} | Test Acc: {accuracy:.2f}%')

# 保存模型
torch.save(cnn.state_dict(), MODEL_PATH)
print(f"\n✅ 模型已保存到：{MODEL_PATH}")

# 测试前32个样本
inputs = test_x[:32]
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].numpy()
true_y = test_y[:32].numpy()

# 打印预测结果
print("\n📊 预测结果（前10个）：")
print("预测数字:", pred_y[:10])
print("真实数字:", true_y[:10])

# ===================== 1. 绘制准确率随训练过程变化曲线 =====================
plt.figure(figsize=(10, 5))

# 子图1：损失值变化
plt.subplot(1, 2, 1)
plt.plot(train_metrics['steps'], train_metrics['losses'], 'b-', linewidth=1.5, label='Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2：准确率变化（按epoch标注）
plt.subplot(1, 2, 2)
# 绘制准确率曲线
plt.plot(train_metrics['steps'], train_metrics['accuracies'], 'r-', linewidth=1.5, label='Test Accuracy')
# 添加epoch分隔线
epoch_steps = []
epoch_accs = []
for e in range(EPOCH):
    # 找到每个epoch最后一步的索引
    epoch_indices = [i for i, ep in enumerate(train_metrics['epochs']) if ep == e]
    if epoch_indices:
        last_idx = epoch_indices[-1]
        epoch_steps.append(train_metrics['steps'][last_idx])
        epoch_accs.append(train_metrics['accuracies'][last_idx])
        # 绘制epoch分隔线
        plt.axvline(x=train_metrics['steps'][last_idx], color='gray', linestyle='--', alpha=0.5)
        # 标注epoch
        plt.text(train_metrics['steps'][last_idx], np.max(train_metrics['accuracies']),
                 f'Epoch {e + 1}', rotation=90, va='top', ha='right', fontsize=8)

plt.xlabel('Training Steps')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Curve (vs Training Steps)')
plt.ylim(0, 100)  # 准确率范围0-100%
plt.grid(True, alpha=0.3)
plt.legend()

plt.suptitle('Training Metrics (Loss & Accuracy)', fontsize=12)
plt.tight_layout()
plt.savefig('training_accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ===================== 2. 绘制带标注的32个样本预测结果 =====================
plt.figure(figsize=(16, 8))  # 调整画布大小以容纳标注
n_rows = 4  # 32个样本分为4行8列
n_cols = 8

# 逐个绘制图片并添加标注
for i in range(32):
    plt.subplot(n_rows, n_cols, i + 1)
    # 获取单张图片并调整维度
    img = inputs[i].squeeze().numpy()
    plt.imshow(img, cmap='gray')

    # 设置图注：区分预测正确/错误（不同颜色）
    true_label = true_y[i]
    pred_label = pred_y[i]
    if true_label == pred_label:
        # 预测正确：绿色标注
        title_text = f'True: {true_label}\nPred: {pred_label}'
        plt.title(title_text, color='green', fontsize=8)
    else:
        # 预测错误：红色标注
        title_text = f'True: {true_label}\nPred: {pred_label}'
        plt.title(title_text, color='red', fontsize=8)

    # 隐藏坐标轴
    plt.xticks([])
    plt.yticks([])

# 整体标题（百分比准确率）
correct = (pred_y == true_y).sum()
accuracy = correct / 32 * 100
plt.suptitle(f'MNIST prediction result (accuracy: {correct}/32 = {accuracy:.1f}%)', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出顶部标题空间
plt.savefig('mnist_predictions_with_labels.png', dpi=150, bbox_inches='tight')
plt.show()

# 保留OpenCV显示（可选）
img_grid = torchvision.utils.make_grid(inputs, nrow=8, padding=2)
img_grid = img_grid.numpy().transpose(1, 2, 0)
img_grid = np.clip(img_grid, 0, 1)
cv2_img = (img_grid * 255).astype(np.uint8)
cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

cv2.namedWindow('MNIST Predictions (Grid View)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MNIST Predictions (Grid View)', 800, 400)
cv2.imshow('MNIST Predictions (Grid View)', cv2_img)
print("\n按任意键关闭OpenCV窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===================== 3. 错误样本单独可视化（带标注） =====================
wrong_idx = np.where(pred_y != true_y)[0]
if len(wrong_idx) > 0:
    error_rate = len(wrong_idx) / len(pred_y) * 100
    print(f"\n num of false result：{len(wrong_idx)}，error rate：{error_rate:.2f}%")

    # 绘制错误样本（最多16个）
    plt.figure(figsize=(12, 6))
    n_wrong = min(len(wrong_idx), 16)
    n_wrong_rows = n_wrong // 4 if n_wrong % 4 == 0 else n_wrong // 4 + 1

    for i, idx in enumerate(wrong_idx[:16]):
        plt.subplot(n_wrong_rows, 4, i + 1)
        img = inputs[idx].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        # 红色标注错误样本
        plt.title(f'True: {true_y[idx]}\nPred: {pred_y[idx]}', color='red', fontsize=10)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f'错误预测样本（错误率：{error_rate:.2f}%）', fontsize=12, color='red')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('mnist_wrong_predictions.png', dpi=150)
    plt.show()
else:
    print("\n 全部预测正确！准确率：100.00%")