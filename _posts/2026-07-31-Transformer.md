---
title: "Transformer (Theory)"
date: 2026-07-31 18:30:00 +0900
math: true
categories: [DeepLearning]
tags: [DeepLearning]
---

## Introduction

Attention is all you need는 딥러닝의 근간을 바꾼 논문이라고 해도 과언이 아니다. 여기서 등장한 Self-Attention은 NLP아키텍처 중에서 현시점으로 최고의 성능을 자랑하는 아키텍처이다.

위 논문에서 다루는 모델인 Transformer를 만들것이다. Self-Attention부터 시작해서 feedforwrd네트워크 등을 살펴보고 전체 아키텍처를 구성할 것이다. 그리고 top-k와 temperature같이 단어를 어떻게 생성할 것인지에 다루고 데이터 전처리에 간단히 다룰것이다.

## Reference

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

밑바닥 부터 만들면서 배우는 LLM

## Attention

### Self Attention

![Transformer1](assets/img/transformer1.png)

위 그림은 지난 블로그에서 봤던 어텐션 레이어를 크게 2개로 나눠서 구현을 해보았다. Self-attention은 hs,h가 각각 encoder decoder에서 온 벡터가 아닌 둘다 같은 벡터를 쓴다는 점이다. 물론 Transformer모델은 GPT나 BERT와 달리 Encoder와 Decoder 둘 다 가지고 있는 아키텍처이기 때문에 hs와 h를 각각 encoder와 decoder에서 뽑아오는 레이어는 존재한다. 하지만 Transformer는 더 나아가 encoder decoder에서 독립적으로 self_attention을 한다는 특징을 보인다. Self-Attention을 이용시 장점으론 다음과 같다.

- 시간 복잡도가 줄어듦

기존 Recurrent 레이어나 Convolutional 레이어에선 $O(n \cdot d^2)$ $O(k \cdot n \cdot d^2)$ 인 차원이 square형태로 나타나지만 Self_attention에선 시간복잡도가 O(n^2 \cdot d)이다. 시퀀스 길이인 n보단 단어 임베딩 차원인 d가 더 숫자가 높다는 측면에서 Self-attention의 시간복잡도는 효율적이다. 더 자세한 내용은 [논문](https://arxiv.org/pdf/1706.03762)의 4절을 참고하자.

- **멀리 떨어져 있는 단어여도 의존성을 유지할 수 있음.**

이 점이 Self-Attention를 쓰는 이유의 가장 핵심이다. path length가 긴. 즉, 멀리 떨어져 있는 토큰 이여도 서로에 대한 dependency를 수치화 시킬 수 있는 것이다. 기존 RNN의 seq2seq모델에서 가장 큰 문제점은 **긴 문장의 데이터를 단일한 벡터로 표현**하는 것이었는데 Attention레이어가 encoder의 LSTM에서 나온 벡터들을 이어붙여서 decoder의 LSTM값과 내적시킴으로써 이 점을 보완했던 것을 기억할 것이다. 이 Self-Attention은 이걸 보완해서 토큰과 토큰으로 직접 스칼라를 계산할 수 있고 RNN의 큰 문제점이었던 **Sequence를 한꺼번에 넣는 것**을 못하는 것을 완벽히 보완했다. "the cute dog is wagging its tail"을 단어별로 tokening 시켰다고 가정을 하자. self-Attention은 query,key,value를 입력으로 받는데, 위 그림과 비교하면 query는 위 그림의 h와 같다. 즉, **query와 key를 내적시키면 위 그림의 Attention weight와 같고 그 가중치인 a를 value와 내적 시키는 것은 위 그림의 Weight sum과 비교할 수 있다.** 이때 a부분(가중치)을 self-attention에서 표현하면 다음 그림과 같다.

![Transformer3](assets/img/transformer3.png)

### Scaled Dot-Product Attention

이제 Transformer에서 쓰는 Attention의 구조에 대해 살펴보자. query,key,value를 각각 Q,K,V라고 하자. 이때, 디코더에서 인코더의 값을 가져올 때 기존에 썼던 어텐션을 생각해보면 K와 V에서 가져옴을 알 수 있다. 입력의 시퀀스 길이가 M이라 하고 decorder의 시퀀스 길이를 N이라 하고 임베딩 차원을 $d_k$라 하자. 이때, Q, K, V의 차원은 각각 $N \times d_k$  $M \times d_k$  $M \times d_k$ 이다. 그리고 가중치(a)를 나타내면 다음과 같이 표현할 수 있다.  

$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

$\sqrt{d_k}$ 로 나누는 이유는 softmax로 확률을 표현할 때 분산값을 줄이기 위함이다. 최종적으로 (Mask(Opt.)를 제외하고) attention값을 나타내면 다음과 같이 나타낼 수 있다. 여기서 attention의 차원은 $N \times d_k$ 이다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi Head Attention

![Transformer2](assets/img/transformer2.png)

모델을 훈련하면서 파라미터를 갱신하거나 그래야 하는데 위에 본 Scaled Dot-Product에선 파라미터가 없다. 그래서 Q,K,V 들어오는곳과 최종 결과에서 Linear fc를 쓴다. 여기서 각각 가중치를 $W^Q$ $W^K$ $W^V$ $W^O$라 하자.

여기서 representation을 하나만 뽑는것 보단 여러개를 뽑는게 직관적으로 이득이다. 그래서 h개수 만큼 Scaled-dot product를 둔다. 입력할 때의 차원과 출력의 차원은 같다. 멀티헤드 어텐션에서 하나의 scaled_dot product의 차원을 $d_k$라 하고 입력했을때의 차원은 $d_{model}$이라고 했을때, $d_k= d_{model}/h$ 이다. 논문에선 $h=8$ 이고 $d_k=64$ 이다.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Causal attention

Self-Attention에서 Q와 K를 내적하고 그 가중치를 그대로 내보내면 문제가 생긴다. 가중치를 토대로 V가 내적을 해서 값을 내보내는데. 이때, V는 Q에 대한 토큰의 정보와 내적하면서 값을 내보낸다. Q의 기준에선 그 뒤에 어떤 정보가 올 지 몰라야 된다.

![Transformer5](assets/img/transformer5.png)

예를 들어, Q 입장에선 "Your jorney starts" 까지 봤다면 "with one step" 에 대한 정보는 누락되어야 한다. 그래서 위 그림과 같이 해당 부분을 아예 없었던 것처럼 처리하면 된다. 

![Transformer2-1](assets/img/transformer2.png)

Mask레이어에서 해당 부분을 $-\infty$ 값으로 만들어 버리고 나머지 부분에 대해 softmax를 진행한다.

### 코드

우선 n_heads로 나누고 linear를 하는것 보단 한꺼번에 linear시키고 n_heads 만큼 잘라내는 것이 좋을 거 같아서 Linear을 입력 출력 차원을 같게 해 놓은것을 볼 수 있다.

q,k,v를 (배치 크기,시퀀스 길이,차원) 이렇게 구성되어 있었는데 이것을 (배치크기, 시퀀스 길이, heads, head 차원)으로 만들었다.

이리저리 transpose하는것 보단 einsum이라는 좋은게 있어서 그것을 이용해서 코드를 짰다. 다음 코드를 참고하면 된다.

```python
class SelfAttention(nn.Module):
  def __init__(self, d_model, heads, dropout=0.0,qkv_bias=False):
    super(SelfAttention, self).__init__()
    assert (d_model % heads == 0), "d_model must be divisible by heads"

    self.d_model = d_model
    self.heads = heads
    self.head_dim = d_model // heads

    self.query= nn.Linear(d_model,d_model,bias=qkv_bias)
    self.key= nn.Linear(d_model,d_model,bias=qkv_bias)
    self.value= nn.Linear(d_model,d_model,bias=qkv_bias)
    self.out_fc=nn.Linear(d_model,d_model)
    self.dropout=nn.Dropout(dropout)


  def forward(self, q, k, v, mask=None):
    batch_size=q.shape[0]
    q_len,k_len,v_len=q.shape[1],k.shape[1],v.shape[1]

    q = self.query(q)
    k = self.key(k)
    v = self.value(v)

    q=q.view(batch_size,q_len,self.heads,self.head_dim)
    k=k.view(batch_size,k_len,self.heads,self.head_dim)
    v=v.view(batch_size,v_len,self.heads,self.head_dim)

    energy=torch.einsum("nqhd,nkhd->nhqk",[q,k])

    if mask is not None:
      energy=energy.masked_fill(mask==0,-torch.inf)

    attention=torch.softmax(energy/(self.head_dim**(1/2)),dim=-1)
    attention=self.dropout(attention)

    out=torch.einsum("nhql,nlhd->nqhd",[attention,v])
    out=out.reshape(batch_size,q_len,self.d_model)

    return self.out_fc(out)
```

외부에서 mask를 받아서 causal attention을 만들것인데 이건 더 필요한 개념이 있어서 나중에 설명하겠다. 

## Architecture

Attention부분을 마쳤으므로 나머지 레이어에 대해서 살펴보자. 크게 FeedForward 부분과 Add&Norm, Embedding 부분에서 Positional Encoding을 볼 수 있다. 전체적인 구조는 아래와 같다.

![Transformer4](assets/img/transformer4.png)

### Add & Norm

Add는 shortcut으로 Resnet에서 본 것과 같다. 위 그림과 같이 전 레이어의 입력값과 출력값을 더하는 것이다.

Norm이 중요한데 여기서는 Layer-Norm을 뜻한다. 임베딩 값들에 대하여 하나하나를 정규화한다. 단어별로 정규화 시켜야 포화 상태를 줄일 수 있다.

### Feedforward

우리가 아는 단순한 NLP나열이다. 이 레이어의 역할을 표현력을 높이는 레이어이며 hidden_layer가 하나인데 차원이 입력값보다 4배 크다. 활성화 함수를 ReLU로 했을 때, 아래와 같다.

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

최근에는 ReLU대신 GELU활성화 함수를 쓴다. 코드로 나타내면 다음과 같다.

```pythonclass
FeedForward(nn.Module):
  def __init__(self,d_model):
    super(FeedForward,self).__init__()
    self.layers=nn.Sequential(
        nn.Linear(d_model,d_model*4),
        nn.ReLU(),
        nn.Linear(d_model*4,d_model)
    )

  def forward(self,x):
    return self.layers(x)
```

### Positional Encoding

시퀀스에서 다른 위치에서 같은 단어가 등장할 때, 단어들의 위치가 어떻게 되는지 정보를 넣어주는 것이 좋다. 최근에는 단어 위치마다 차원이 같은 임베딩 벡터를 만들어 단순히 더하는 정도로 구현이 편하지만 이 논문에서는 Sinusoidal version을 쓴다. i를 단어 하나에서 임베딩 위치로 하고 *pos*를 시퀀스에서 단어의 위치로 표현하면 아래와 같다.

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

$10000^{2i/d_{\text{model}}}$ 를 $\text{exp}(2i \times \log 10000 / d_{\text{model}})$으로 변형시켜서 각각의 pos로 결합하는 식으로 코드를 짰다. 

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", self._get_pe(d_model, max_len))

    def _get_pe(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False

        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        x = x + self.pe[:x.shape[1]]
        return x
```

### Mask

Causal Attention처럼 Q기준에서 계산을 하면 안 되는 부분이 또 존재한다. 시퀀스를 넣는 길이를 일정하게 만들기 위해서 <PAD>를 넣어 패딩을 한다. <PAD>는 그냥 길이만 맞출려고 넣은 것이므로 쿼리에서는 이것을 문자로 처리하면 안 된다. Causal attention처럼 위에 나왔던 것 처럼 단순히 Pad에 대한 부분을 $-\infty$로 만들면 된다. 이는 논문에 없고 내가 추가한 부분이다. 코드는 아래와 같다.

```python
class Transformer(nn.Module):
  def __init__(self,cfg):
    super(Transformer,self).__init__()
    #self.encoder=Encoder(cfg)
    #self.decoder=Decoder(cfg)
    self.device=cfg["device"]
    self.pad_idx = cfg["pad_idx"]

  def make_src_mask(self,src):
    #query쪽은 신경 ㄴㄴ Key 쪽에서만 padding 처리
    batch_size,seq_len=src.shape
    mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(self.device)

  def make_trg_mask(self,trg):
    batch_size, seq_len = trg.shape
    pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return (pad_mask & causal_mask).to(self.device)

  def forward(self,src,trg):
    # ...
```

src와 trg값은 각각 입력 출력 값이다.

### Overall

파라미터를 객체에 계속 뭐넣어라 뭐 넣어라 하면 좀 걸리적 거린다. 밑바닥부터 만들면서 배우는 LLM의 저자 *세바스찬 라시카*는 아예 변수를 이렇게 만들어 버렸다.

```python
TRANSFORMER_BASE_CONFIG={
    "vocab_size":VOCAB_SIZE,
    "max_len":1024,
    "d_model":512,
    "n_heads":8,
    "n_layers":6,
    "drop_attn":0.1,
    "drop_res":0.2,
    "drop_emb":0.2,
    "pad_idx":PAD_ID,
    "qkv_bias": False,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

나도 이 아이디어를 빌렸다. Transformer객체에 cfg를 받는것을 볼 수 있는데, 이걸 받는것이다.

![Transformer4-1](assets/img/transformer4.png)

아무튼 구조가 위에 것과 같이 나타나는데 Encoder와 Decoder블럭이 각각 N개(6개)씩 이어져 있고 그리고 그걸 다 받는 Encoder_block Decoder_block을 만들어야 된다.

