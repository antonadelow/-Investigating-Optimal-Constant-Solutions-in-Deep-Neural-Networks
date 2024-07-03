import torch
import torch.nn as nn
import torchvision.transforms as transforms

from einops import rearrange
from einops.layers.torch import Rearrange

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=0.25):
      super().__init__()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.se = nn.Sequential(
          nn.Conv2d(out_dim, int(in_dim * expansion), kernel_size=1),
          nn.GELU(),
          nn.Conv2d(int(in_dim * expansion), out_dim, kernel_size=1),
          nn.Sigmoid()
      )
    def forward(self,x):
      x = self.se(self.avg_pool(x)).sigmoid()*x
      return x

class MBConv(nn.Module):
  def __init__(self, in_dim, out_dim, image_size, expansion=4, stride=1, kernel_size=3, **kwargs):
    super().__init__()
    self.stride = stride
    self.in_dim, self.out_dim = in_dim, out_dim
    if stride > 1:
      self.pool = nn.MaxPool2d(kernel_size, stride, 1)
      self.proj = nn.Conv2d(in_dim, out_dim, 1, 1)

    self.conv = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim * expansion, 1, stride, bias=False),
            nn.BatchNorm2d(in_dim * expansion),
            nn.GELU(),
            nn.Conv2d(in_dim * expansion, in_dim * expansion, 3, 1, 1,
                      groups=in_dim * expansion, bias=False),
            nn.BatchNorm2d(in_dim * expansion),
            nn.GELU(),
            SqueezeExcitation(in_dim, in_dim * expansion),
            nn.Conv2d(in_dim * expansion, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim))
    
    self.conv2 = nn.Conv2d(in_dim, out_dim, 1)
    
  def forward(self, x):
    if self.stride > 1:
        return self.proj(self.pool(x)) + self.conv(x)
    else:
        if(self.in_dim != self.out_dim):
            return self.conv2(x) + self.conv(x)
        else: 
            return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, image_size, heads=8, head_dim=32, dropout=0.3):
        super().__init__()
        self.im_h, self.im_w = image_size

        self.heads, self.head_dim = heads, head_dim

        self.relative_bias_table = nn.Parameter(
            torch.randn((2 * self.im_h - 1) * (2 * self.im_w - 1), heads))

        relative_positions = torch.flatten(torch.stack(torch.meshgrid((torch.arange(self.im_h), torch.arange(self.im_w)))), 1)
        relative_positions = relative_positions[:, :, None] - relative_positions[:, None, :]

        relative_positions[0].add_(self.im_h - 1).mul_(2 * self.im_w - 1)
        relative_positions[1].add_(self.im_w - 1)

        relative_positions = rearrange(relative_positions, 'c h w -> h w c')
        relative_indices = relative_positions.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_indices", relative_indices)

        self.linear = nn.Linear(heads * head_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.Q = nn.Linear(in_dim, head_dim*heads, bias=False)
        self.K = nn.Linear(in_dim, head_dim*heads, bias=False)
        self.V = nn.Linear(in_dim, head_dim*heads, bias=False)

    def forward(self, x):
        Q = rearrange(self.Q(x),'b n (h d) -> b h n d', h=self.heads)
        K = rearrange(self.K(x),'b n (h d) -> b h n d', h=self.heads)
        V = rearrange(self.V(x),'b n (h d) -> b h n d', h=self.heads)

        relative_bias = rearrange(
            self.relative_bias_table.gather(0, self.relative_indices.repeat(1, self.heads)),
            '(h w) c -> 1 c h w', h=self.im_h*self.im_w, w=self.im_h*self.im_w)

        attention = torch.matmul(self.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.head_dim ** -0.5 + relative_bias),V)
        x = self.dropout(self.linear(rearrange(attention, 'b h n d -> b n (h d)')))
        return x


class TFM(nn.Module):
    def __init__(self, in_dim, out_dim, image_size, heads=8, head_dim=32, stride=1, dropout=0.3, expansion = 4):
        super().__init__()
        im_h, im_w = image_size
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(3, stride, 1)
            self.proj = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)


        self.attention = nn.Sequential(
            Rearrange('b c im_h im_w -> b (im_h im_w) c'),
            nn.LayerNorm(in_dim),
            Attention(in_dim, out_dim, image_size, heads=8, head_dim=32),
            Rearrange('b (im_h im_w) c -> b c im_h im_w', im_h=im_h, im_w=im_w)
        )

        self.ffn = nn.Sequential(
            Rearrange('b c im_h im_w -> b (im_h im_w) c'),
            nn.LayerNorm(out_dim),
             nn.Sequential(
                nn.Linear(out_dim, in_dim*expansion),
                nn.GELU(),
                nn.Linear(in_dim*expansion, out_dim),
                nn.Dropout(dropout)),
            Rearrange('b (im_h im_w) c -> b c im_h im_w', im_h=im_h, im_w=im_w)
        )

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
            x = self.proj(x) + self.attention(x)
        else:
            x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class CoAtNet(nn.Sequential):
    def __init__(self, image_size, num_classes, L, D, num_channels, strides = [2, 2, 2, 2, 2], dropout=0.3):
        im_h, im_w = image_size
        self.dropout = dropout
        im_h, im_w = im_h // strides[0], im_w // strides[0]
        s0 = self.init_stage(self.conv_3x3, num_channels, D[0], L[0], (im_h, im_w), strides[0])
        layers = [s0]

        blocks=[MBConv, MBConv, TFM, TFM]

        for i in range(4):
          im_h, im_w = im_h // strides[i+1], im_w // strides[i+1]
          layers.append(self.init_stage(blocks[i], D[i], D[i+1], L[i+1], (im_h, im_w), strides[i+1]))

        layers.extend([nn.AdaptiveAvgPool2d(1),Rearrange('b c h w -> b (c h w)'), nn.Dropout(dropout), nn.Linear(D[-1], num_classes)])
        super().__init__(*layers)

    def conv_3x3(self, in_channels, out_channels, size, stride=1, **kwargs):
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU())

    def init_stage(self, block, in_dim, out_dim, l, size, stride):
      blocks = nn.ModuleList([])
      blocks.append(block(in_dim, out_dim, size, stride=stride, dropout=self.dropout))
      for i in range(l-1):
          blocks.append(block(out_dim, out_dim, size, dropout=self.dropout))
      return nn.Sequential(*blocks)