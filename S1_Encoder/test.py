from S1_Encoder.encoder import attention_score
import torch
import torch.nn as nn
import torch.nn.functional as F

class proj():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = torch.randn(in_dim, out_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, in_dim]
        # output: [batch_size, seq_len, out_dim]
        return torch.matmul(x, self.weight)


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = proj(embed_dim, embed_dim)
        self.k_proj = proj(embed_dim, embed_dim)
        self.v_proj = proj(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attention_score=torch.matmul(Q,K.transpose(-1,-2))/(self.embed_dim**0.5)


        # MASK 理解这里的mask，为什么是-Inf?
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attention_score, dim=-1)
        output = torch.matmul(attn_weights,V)

        return output

if __name__ == "__main__":
    x = torch.randn(1, 3, 4)
    model = SimpleSelfAttention(4)
    output = model(x)
    print(output)