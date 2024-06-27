import torch  # 导入PyTorch库，用于深度学习任务
from denoising_diffusion_pytorch import Unet, GaussianDiffusion  # 从denoising_diffusion_pytorch库导入Unet和GaussianDiffusion类

# 初始化一个Unet模型，用于去噪扩散模型中的去噪网络
model = Unet(
    dim = 64,  # 模型的基础维度
    dim_mults = (1, 2, 4, 8),  # 维度倍增因子，用于模型中的不同层
    flash_attn = True  # 是否使用FlashAttention技术
)

# 初始化GaussianDiffusion去噪扩散模型
diffusion = GaussianDiffusion(
    model,  # 使用上面定义的Unet模型
    image_size = 128,  # 生成图像的尺寸
    timesteps = 1000  # 扩散过程中的时间步数
)

# 生成训练用的随机图像数据，形状为(batch_size, channels, height, width)
training_images = torch.rand(8, 3, 128, 128)  # 图像数据被标准化到0到1之间

# 使用去噪扩散模型计算训练图像的损失
loss = diffusion(training_images)

# 执行反向传播，计算损失相对于模型参数的梯度
loss.backward()

# 经过大量的训练之后...

# 使用训练好的去噪扩散模型采样生成新的图像
sampled_images = diffusion.sample(batch_size = 4)

# 输出生成图像的形状，这里应该是(batch_size, channels, height, width)
sampled_images.shape  # (4, 3, 128, 128)