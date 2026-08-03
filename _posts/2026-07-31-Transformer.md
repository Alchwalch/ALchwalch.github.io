---
title: "Transformer"
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

