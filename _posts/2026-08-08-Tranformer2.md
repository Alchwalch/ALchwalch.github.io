---
title: "Transformer (Implementation)"
date: 2026-07-31 18:30:00 +0900
math: true
categories: [DeepLearning]
tags: [DeepLearning]
---

## 1차

### Dataset

WMT14 en-de로 진행했다.

```python
from datasets import load_dataset

print("loading...")
dataset = load_dataset("regisss/wmt14-en-de-pre-processed")
print(dataset)
```

4548885 개의 train 시퀀스가 있지만 내가 군대에 있는것과 같은 이유로 train dataset을 16만개로 줄임
나머지는 val 2169 train 2999 개는 그대로 동일

토크나이저는 gpt2 tokenized을 이용 (밑바닥 부터 만들면서 배우는 LLM 차용)

```python
import tiktoken
tokenizer=tiktoken.get_encoding("gpt2")

BOS_ID = tokenizer.n_vocab
EOS_ID = tokenizer.n_vocab+1
PAD_ID = tokenizer.n_vocab+2
```

Dataset 구성

dataset의 아이템 마다 (en,de) pair로 구성시켜서 Dataset을 만들자. 코드는 아래와 같이 짰음.

```pytorch
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.max_length = max_length
        self.pairs = []
        for item in dataset:
            item = item["translation"]
            src = tokenizer.encode(item["en"])[: max_length - 2]
            tgt = tokenizer.encode(item["de"])[: max_length - 2]
            self.pairs.append((
                torch.tensor([BOS_ID] + src + [EOS_ID]),
                torch.tensor([BOS_ID] + tgt + [EOS_ID]),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
```

dataset    - train,val,test같은 dataset 불러오는 파라미터.
tokenizer  - 토크나이저 받는 파라미터
max_length - 최대 context 길이

원래 Dataset에서 Padding처리 할려 했었는데 collate_fn쓰면 간편하게 할 수 있다 함. 그래서 LLM서비스 도움으로 collate_fn작성하고 각각의 dataset을 받아와서 Dataloader을 만들어 버리는 함수(create_dataloader)를 만듦.

```pytorch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
  src_batch, tgt_batch = zip(*batch)

  src_batch = pad_sequence(
      src_batch,
      batch_first=True,
      padding_value=PAD_ID
  )

  tgt_batch = pad_sequence(
      tgt_batch,
      batch_first=True,
      padding_value=PAD_ID
  )

  return src_batch, tgt_batch

def create_dataloader(dataset, tokenizer, batch_size, max_length,shuffle=True,drop_last=True,collate_fn=None,num_workers=2):
  dataset=TranslationDataset(dataset,tokenizer,max_length)
  dataloader=DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      collate_fn=collate_fn,
      num_workers=num_workers
  )

  return dataloader
```

train,val,test 각각 만듦.

```python
train_loader=create_dataloader(
    train_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
)

val_loader=create_dataloader(
    val_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
)

test_loader=create_dataloader(
    test_dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=2
```

### Setting

```python
TRANSFORMER_BASE_CONFIG={
    "vocab_size":VOCAB_SIZE,
    "max_len":1024,
    "d_model":512,
    "n_heads":8,
    "n_layers":6,
    "drop_attn":0.0,
    "drop_res":0.1,
    "drop_emb":0.1,
    "pad_idx":PAD_ID,
    "qkv_bias": False,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

논문 구현을 우선으로함. drop_attn은 Self_attention에서 attention에서 dropout을 시키는 rate임.

optimizer는 Adam을 썼고 나머지는 논문에 나와있는 스케줄러 차용. (자세한건 코드)

```python
optimizer=optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: noam_lr_lambda(step, TRANSFORMER_BASE_CONFIG["d_model"], warmup_steps)
)
```

calc_loss_batch는 이터레이션 한 번 돌때마다 배치단위로 cross_entropy를 추적하는 함수임.

$$P(w_1, \dots , w_{m})=\Pi_{t=1}^m P(w_t \mid w_1, \dots, w_{t-1})$$

$$\sum_{t=1}^m \log P(w_t \mid w_1, \dots , w_{t-1})$$

이므로 정답에 대해 cross_entropy로 나타낼 수 있음. 밑바닥 부터 만들면서 배우는 LLM참고함.

```python
def calc_loss_batch(input_batch,target_batch,model,device):
  input_batch=input_batch.to(device)
  target_batch=target_batch.to(device)

  tgt_input=target_batch[:,:-1]
  tgt_output=target_batch[:,1:]
  out=model(input_batch,tgt_input)

  loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),tgt_output.reshape(-1),ignore_index=PAD_ID)

  return loss
```

이거는 loader 전체에 대해서 나타내는 loss 함수. val과 test에서 썼음. 이것도 밑바닥 부터 만들면서 배우는 LLM 참고함.

```python
def calc_loss_loader(data_loader,model,device,num_batches=None):
  total_loss=0
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches=len(data_loader)
  else:
    num_batches=min(num_batches,len(data_loader))

  for i,(src,trg) in enumerate(data_loader):
    if i >= num_batches:
      break

    loss=calc_loss_batch(src,trg,model,device)
    total_loss+=loss.item()

  return total_loss/num_batches

def evaluate_model(model, data_loader, device,eval_iter):
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device, enabled=use_amp):
        loss=calc_loss_loader(data_loader,model,device,eval_iter)
    model.train()
    return loss
```

### 결과

validation, train에 대한 평가를 매 epoch에서 기록함. 결과가 다음과 같아짐.

