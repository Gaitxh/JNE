import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
zscore_scaler = StandardScaler()
from sklearn.preprocessing import MinMaxScaler
maxmin_scaler = MinMaxScaler()
class residual_linear_module(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layer_number,
                 dropout_rate=0, bias=True):
        super(residual_linear_module, self).__init__()

        self.lin0 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.Dropout(dropout_rate)
        )

        # 使用 nn.ModuleList 来存储多个残差块
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_number)
        ])

        self.lin1 = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=bias)
        )

    def forward(self, x):
        # 第一层线性变换
        x = self.lin0(x)

        for res_block in self.mlp:
            residual = x  # 更新残差为当前层输出
            x = res_block(x) + residual  # 残差连接

        # 最后一层线性变换
        x = self.lin1(x)

        return x


class residual_nonlinear_module(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layer_number,
                 dropout_rate=0, activation='gelu',
                 bias=True):
        super(residual_nonlinear_module, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.lin0 = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        # 使用 nn.ModuleList 来存储多个残差块
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                self.activation,
                nn.Dropout(dropout_rate)
            ) for _ in range(layer_number)
        ])

        self.lin1 = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=bias)
        )

    def forward(self, x):
        # 第一层线性变换
        x = self.lin0(x)

        for res_block in self.mlp:
            residual = x  # 更新残差为当前层输出
            x = res_block(x) + residual  # 残差连接

        # 最后一层线性变换
        x = self.lin1(x)

        return x


class encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 dropout_rate=0, bias=True,
                 encoder_type='res_nonlinear'):
        super(encoder, self).__init__()

        if encoder_type == 'res_linear':
            self.encoder = residual_linear_module(input_size=input_size, output_size=output_size,
                                                  hidden_size=hidden_size,
                                                  dropout_rate=dropout_rate, layer_number=2, bias=bias)
        elif encoder_type == 'res_nonlinear_relu':
            self.encoder = residual_nonlinear_module(input_size=input_size, output_size=output_size,
                                                     hidden_size=hidden_size,
                                                     dropout_rate=dropout_rate, activation='relu', layer_number=2,
                                                     bias=bias)
    def forward(self, x):
        return self.encoder(x)


def r2_score(Real, Pred):
    # print(Real.shape)
    # print(Pred.shape)
    SSres = np.mean((Real - Pred) ** 2, 0)
    # print(SSres.shape)
    SStot = np.var(Real, 0)
    # print(SStot.shape)
    return np.nan_to_num(1 - SSres / SStot)


def encoder_train(X_train, y_train, X_val, y_val, X_test, y_test,
                  encoder_type, save_model_dir,
                  hidden_size=1024, bias=True,
                  max_epochs=512, patience=8):
    brain_map_model = encoder(input_size=X_train.shape[1], output_size=y_train.shape[1],
                              hidden_size=hidden_size, dropout_rate=0, bias=bias,
                              encoder_type=encoder_type)

    loss_func = nn.MSELoss()

    # 设置优化器
    optimizer = optim.AdamW([
        {'params': brain_map_model.parameters()}
    ])

    # 提前停止的相关变量
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    best_model_state = None

    # 初始化记录列表
    train_losses = []
    val_losses = []
    val_r2 = []
    val_r2_mean = []
    stop_epoch = 0

    # 开始训练
    for epoch in range(max_epochs):
        begin_time = time.time()
        ### ++++++++++++++++++++++++ 训练集 ++++++++++++++++++++++++
        brain_map_model.train()
        optimizer.zero_grad()
        an_brain_train = brain_map_model(X_train)
        loss_train = loss_func(an_brain_train, y_train)
        loss_train.backward()
        train_losses.append(loss_train.item())
        optimizer.step()

        ### ++++++++++++++++++++++++ 验证集 ++++++++++++++++++++++++
        brain_map_model.eval()
        with torch.no_grad():
            an_brain_val = brain_map_model(X_val)
            loss_val = loss_func(an_brain_val, y_val)
            R2_ = r2_score(y_val.cpu().numpy(), np.nan_to_num(an_brain_val.cpu().numpy(), nan=0))
            R2_ = np.nan_to_num(R2_, neginf=0)
            val_r2.append(R2_.max())
            val_r2_mean.append(R2_.mean())
            val_losses.append(loss_val.item())

        ### 检查验证损失是否改善
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            no_improvement_epochs = 0
            # 保存最佳模型状态
            best_model_state = brain_map_model.state_dict()
            stop_epoch = epoch

        else:
            no_improvement_epochs += 1

        end_time = time.time()
        # print(f"{encoder_type}"
        #       f" Epoch{epoch}"
        #       f" TrainLoss{loss_train.item():.2f}"
        #       f" ValLoss{loss_val.item():.2f}"
        #       f" ValR{val_r2[epoch]:.2f}"
        #       f" mean{val_r2_mean[epoch]:.2f}"
        #       f" {end_time - begin_time:.2f}s"
        #       f" {no_improvement_epochs+1}/{patience}")

        # 如果验证损失在 patience 个 epoch 内没有提升，则停止训练
        if no_improvement_epochs >= patience:
            break

    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_r2_max": val_r2,
        "val_r2_mean": val_r2_mean,
        "stop_epoch": stop_epoch
    }

    ## 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, save_model_dir)

    return results


def fig_polt(results, dir_path):
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes = axes.flatten()
    # 第一张子图: Losses
    axes[0].plot(results["train_losses"], label='Train Loss')
    axes[0].plot(results["val_losses"], label='Validation Loss')
    axes[0].axvline(results["stop_epoch"], color='red', linestyle='--')  # 添加竖线
    axes[0].legend()

    axes[1].plot(results["val_r2_max"], label='val r2 max')
    axes[1].axvline(results["stop_epoch"], color='red', linestyle='--')  # 添加竖线
    axes[1].legend()

    axes[2].plot(results["val_r2_mean"], label='val r2 mean')
    axes[2].axvline(results["stop_epoch"], color='red', linestyle='--')  # 添加竖线
    axes[2].legend()

    # 调整布局并显示图形
    plt.tight_layout()
    # 保存图像
    plt.savefig(dir_path)
    # plt.show()
    plt.close(fig)


def Jacobian_mat_cal(X_train, x_train_output):
    Jacobian_mat_train = torch.zeros(X_train.shape[0], X_train.shape[1], x_train_output.shape[1])
    for brain_i in tqdm(range(x_train_output.shape[1])):
        grad_output = torch.zeros_like(x_train_output)
        grad_output[:, brain_i] = 1  ##计算所有batch所有AN下对输入的偏导
        Jacobian_mat_train[:, :, brain_i] = torch.autograd.grad(outputs=x_train_output, inputs=X_train,
                                                                grad_outputs=grad_output,
                                                                retain_graph=True, create_graph=False)[0]
    return Jacobian_mat_train

def calculate_noise_ceiling(data1, data2, data3, n=3):
    """
    来自论文：A massive 7T fMRI dataset to bridge cognitive and computational neuroscience
    :param data1:
    :param data2:
    :param data3:
    :param n:
    :return:
    """
    """
    基于三次重复fMRI数据计算每个体素的噪声上限（Noise Ceiling）

    参数：
    data1, data2, data3 : numpy.ndarray
        三维重复的fMRI数据，形状为（n_voxels, n_trials）
        要求：三个数组形状相同，且相同位置的trial对应相同刺激
    n : int, 默认=3
        试次平均次数，决定噪声上限的最终表达形式：
        n=1表示单次试次，n=3表示三次试次平均后的噪声上限

    返回：
    noise_ceiling : numpy.ndarray
        每个体素的噪声上限（百分比），形状为（n_voxels,）
    """

    # 合并数据为三维数组：体素 × 试次 × 重复
    data = np.stack([data1, data2, data3], axis=2)  # 形状：(V, T, 3)

    # 步骤1：计算每个体素-试次的方差（无偏估计）
    variances = np.var(data, axis=2, ddof=1)        # 形状：(V, T)

    # 步骤2：平均所有试次的方差，得到噪声方差σ²_noise
    avg_variance = np.mean(variances, axis=1)      # 形状：(V,)

    # 步骤3：计算噪声标准差σ_noise和信号标准差σ_signal
    sigma_noise = np.sqrt(avg_variance)
    sigma_signal = np.sqrt(np.maximum(1 - avg_variance, 0))  # 总方差=1（已z-score）

    # 步骤4：计算信噪比（ncsnr）
    ncsnr = sigma_signal / sigma_noise

    # 步骤5：计算噪声上限（百分比）
    noise_ceiling = (n * ncsnr**2) / (1 + n * ncsnr**2) * 100

    # 处理除零情况（当sigma_noise=0时，上限为100%）
    noise_ceiling = np.nan_to_num(noise_ceiling, nan=100.0, posinf=100.0, neginf=0.0)

    return noise_ceiling

def plot_noise_ceiling(A, B):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # 计算点的密度
    xy = np.vstack([A, B])
    density = gaussian_kde(xy)(xy)

    # 创建图形对象
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # 绘制散点密度图
    sc = ax.scatter(A, B, c=density, s=10, cmap='magma', norm=plt.Normalize(vmin=1, vmax=25))

    # 绘制参考线
    ax.plot([0, 1], [0, 1], 'r-', linewidth=2, label='Noise ceiling')      # 噪声上限
    ax.plot([0, 1], [0, 0.8], 'y--', linewidth=2, label='80% noise ceiling') # 85% 噪声上限

    # 设置标签
    ax.set_xlabel('Noise ceiling', fontsize=16)
    ax.set_ylabel('Model performance (R²)', fontsize=16)

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    # 移除网格线
    ax.grid(False)

    # 颜色条设置
    cbar = fig.colorbar(sc, label='Density', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # 刻度设置
    ax.tick_params(axis='both', which='major', labelsize=14,
                  width=2, length=6, color='black')

    # 图例设置
    ax.legend(fontsize=16, loc='upper left', frameon=False)

    plt.tight_layout()
    plt.show()

def plot_scatter_(A, B):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # 计算点的密度
    xy = np.vstack([A, B])
    density = gaussian_kde(xy)(xy)

    # 创建图形对象
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # 绘制散点密度图
    sc = ax.scatter(A, B, c=density, s=2, cmap='magma', norm=plt.Normalize(vmin=1, vmax=50))

    # 绘制参考线
    ax.plot([0, 1], [0, 1], 'r-', linewidth=2)      # 噪声上限

    # 设置标签
    ax.set_xlabel('Noise ceiling', fontsize=16)
    ax.set_ylabel('Model performance (R²)', fontsize=16)

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    # 移除网格线
    ax.grid(False)

    # 颜色条设置
    cbar = fig.colorbar(sc, label='Density', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # 刻度设置
    ax.tick_params(axis='both', which='major', labelsize=14,
                  width=2, length=6, color='black')

    # 图例设置
    ax.legend(fontsize=16, loc='upper left', frameon=False)

    plt.tight_layout()

    plt.show()
