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

## Self Attention

![Transformer1](assets/img/transformer1.png)

위 그림은 지난 블로그에서 봤던 어텐션 레이어를 크게 2개로 나눠서 구현을 해보았다. Self-attention은 hs,h가 각각 encoder decoder에서 온 벡터가 아닌 둘다 같은 벡터를 쓴다는 점이다. 물론 Transformer모델은 GPT나 BERT와 달리 Encoder와 Decoder 둘 다 가지고 있는 아키텍처이기 때문에 hs와 h를 각각 encoder와 decoder에서 뽑아오는 레이어는 존재한다. 하지만 Transformer는 더 나아가 encoder decoder에서 독립적으로 self_attention을 한다는 특징을 보인다. Self-Attention을 이용시 장점으론 다음과 같다.

- 시간 복잡도가 줄어듦

기존 Recurrent 레이어나 Convolutional 레이어에선 $O(n \dot d^2)$ $O(k \dot n \dot d^2)$ 인 차원이 square형태로 나타나지만 Self_attention에선 시간복잡도가 O(n^2 \dot d)이다. 시퀀스 길이인 n보단 단어 임베딩 차원인 d가 더 숫자가 높다는 측면에서 Self-attention의 시간복잡도는 효율적이다. 더 자세한 내용은 [논문](https://arxiv.org/pdf/1706.03762)의 4절을 참고하자.

- **멀리 떨어져 있는 단어여도 의존성을 유지할 수 있음.**

이 점이 Self-Attention를 쓰는 이유의 가장 핵심이다. path length가 긴. 즉, 멀리 떨어져 있는 토큰 이여도 서로에 대한 dependency를 수치화 시킬 수 있는 것이다. 기존 RNN의 seq2seq모델에서 가장 큰 문제점은 **긴 문장의 데이터를 단일한 벡터로 표현**하는 것이었는데 Attention레이어가 encoder의 LSTM에서 나온 벡터들을 이어붙여서 decoder의 LSTM값과 내적시킴으로써 이 점을 보완했던 것을 기억할 것이다. 이 Self-Attention은 이걸 보완해서 토큰과 토큰으로 직접 스칼라를 계산할 수 있고 RNN의 큰 문제점이었던 **Sequence를 한꺼번에 넣는 것**을 못하는 것을 완벽히 보완했다. "the cute dog is wagging its tail"을 단어별로 tokening 시켰다고 가정을 하자. self-Attention은 query,key,value를 입력으로 받는데, 위 그림과 비교하면 query는 위 그림의 h와 같다. 즉, **query와 key를 내적시키면 위 그림의 Attention weight와 같고 그 가중치인 a를 value와 내적 시키는 것은 위 그림의 Weight sum과 비교할 수 있다.** 이때 a부분(가중치)을 self-attention에서 표현하면 다음 그림과 같다.

![Transformer3](assets/img/transformer3.png)

이제 Transformer에서 쓰는 Attention의 구조에 대해 살펴보자. query,key,value를 각각 Q,K,V라고 하자. 이때, 디코더에서 인코더의 값을 가져올 때 기존에 썼던 어텐션을 생각해보면 Q에서 가져옴을 알 수 있다. 입력의 시퀀스 길이가 N이라 하고 decorder의 시퀀스 길이를 M이라 하고 임베딩 차원을 $d_k$라 하자. 이때, Q, K, V의 차원은 각각 $N \dot d_k$  

![Transformer2](assets/img/transformer2.png)

![Transformer4](assets/img/transformer4.png)

