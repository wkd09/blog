---
title: "BERT 논문 정리: 양방향 Transformer Encoder로 언어 이해하기"
date: 2026-05-29 01:20:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - NLP
  - BERT
  - Paper
source: "arXiv:1810.04805"
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

이 글은 논문 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)를 바탕으로 정리한 글이다.

논문의 핵심은 간단하다. 언어를 왼쪽에서 오른쪽으로만 보지 말고, Transformer Encoder를 사용해 양방향 context를 함께 보도록 사전학습하면 자연어 이해 task 성능이 크게 좋아진다는 것이다. BERT는 이를 위해 `Masked Language Model`과 `Next Sentence Prediction`을 사용했다.

![BERT 원문 Figure: input embedding 구성]({{ '/assets/images/blog/bert-paper-figure.png' | relative_url }})

*출처: 논문 Figure 2. BERT 입력은 token embedding, segment embedding, position embedding을 합쳐 만든다.*

## 1. 이 논문이 나온 배경

BERT 이전에도 ELMo, GPT 같은 사전학습 언어 모델은 강력했다. 하지만 당시 접근에는 한계가 있었다.

- GPT 계열은 left-to-right language model이라 각 token이 이전 token만 볼 수 있다.
- ELMo는 양방향 정보를 사용하지만, Transformer 기반의 깊은 bidirectional representation과는 구조가 다르다.
- 문장 분류, 질의응답, 자연어 추론처럼 양쪽 문맥이 중요한 task에서는 단방향 제약이 불리할 수 있다.

예를 들어 다음 문장을 보자.

```text
The animal didn't cross the street because it was too tired.
```

`it`이 무엇을 가리키는지 이해하려면 앞뒤 문맥을 함께 보는 것이 중요하다. BERT는 이런 자연어 이해 문제를 위해 처음부터 양방향 encoder representation을 학습하도록 설계되었다.

## 2. BERT란?

BERT는 `Bidirectional Encoder Representations from Transformers`의 약자다.

구조적으로는 Transformer Encoder를 사용한다. GPT처럼 다음 token을 생성하는 decoder-only 모델이 아니라, 입력 문장 전체를 보고 각 token의 contextual representation을 만든다.

```text
입력 문장 전체
   |
   v
Transformer Encoder layers
   |
   v
각 token의 양방향 contextual representation
```

BERT의 특징은 다음과 같다.

- encoder-only Transformer 구조를 사용한다.
- 모든 token이 self-attention을 통해 양쪽 context를 볼 수 있다.
- pretraining 후 downstream task에 fine-tuning한다.
- 생성보다 자연어 이해, 분류, 추출형 QA 같은 task에 강하다.

## 3. 입력 표현

BERT 입력은 세 가지 embedding의 합으로 구성된다.

![bert embedding]({{ '/assets/images/blog/bertembedding.png' | relative_url }})

### Token Embedding

BERT는 WordPiece tokenization을 사용한다. 단어를 그대로 쓰지 않고 subword 단위로 나누어, 드문 단어도 여러 조각의 조합으로 표현할 수 있게 한다.

예를 들어 vocabulary에 없는 단어라도 자주 쓰이는 subword 조각으로 나누면 모델이 처리할 수 있다.

### Segment Embedding

BERT는 두 문장을 함께 입력으로 받을 수 있다.

```text
[CLS] sentence A [SEP] sentence B [SEP]
```

이때 각 token이 첫 번째 문장에 속하는지, 두 번째 문장에 속하는지를 segment embedding으로 구분한다. 이는 문장 쌍 관계를 다루는 자연어 추론, QA, sentence pair classification에서 중요하다.

### Position Embedding

Transformer는 recurrent 구조가 없기 때문에 token 순서를 별도로 알려줘야 한다. BERT는 학습 가능한 position embedding을 사용한다.

## 4. Pre-training 방법

BERT의 pretraining은 두 task로 구성된다.

![bert pretraining concept]({{ '/assets/images/blog/bertscore.png' | relative_url }})

### Masked Language Model

일반적인 left-to-right language model은 다음 token을 예측한다. 하지만 BERT는 양방향 context를 보게 만들고 싶기 때문에 단순한 next-token prediction을 사용할 수 없다.

그래서 입력 token 일부를 가리고, 주변 문맥을 이용해 원래 token을 맞히게 한다.

```text
Original: My dog is cute.
Masked:   My [MASK] is cute.
Target:   dog
```

논문은 선택된 token 중 일부만 실제 `[MASK]`로 바꾸고, 일부는 random token으로 바꾸거나 그대로 둔다. 이유는 fine-tuning 때는 `[MASK]` token이 등장하지 않기 때문에, pretraining과 fine-tuning 사이의 입력 불일치를 줄이기 위해서다.

### Next Sentence Prediction

NSP는 두 문장 A, B가 실제로 이어지는 문장인지 맞히는 task다.

```text
Sentence A: The man went to the store.
Sentence B: He bought a gallon of milk.
Label: IsNext
```

반대로 B를 corpus에서 무작위로 가져오면 `NotNext` label을 붙인다. 논문은 이 task가 QA나 natural language inference처럼 문장 간 관계를 이해해야 하는 task에 도움이 된다고 보았다.

## 5. Fine-tuning 방식

BERT는 pretraining 후 각 downstream task에 맞게 fine-tuning한다.

![bert fine-tuning]({{ '/assets/images/blog/bertfinetune.png' | relative_url }})

fine-tuning의 장점은 구조가 단순하다는 점이다. 대부분의 task에서 task-specific layer를 조금 붙이고, BERT 전체 파라미터를 labeled data로 업데이트한다.

예시는 다음과 같다.

| Task | 사용 방식 |
|---|---|
| 문장 분류 | `[CLS]` representation으로 classification |
| 문장 쌍 분류 | sentence A, B를 함께 입력 |
| 토큰 분류 | 각 token representation으로 NER 등 수행 |
| 추출형 QA | answer span의 start/end 위치 예측 |

## 6. 실험 결과 요약

논문은 GLUE, SQuAD, SWAG 등 여러 benchmark에서 BERT를 평가했다.

### GLUE

![bert experiment result 1]({{ '/assets/images/blog/bertscore01.png' | relative_url }})

*GLUE는 여러 자연어 이해 task를 묶은 benchmark다. BERT는 기존 방법보다 강한 성능을 보였다.*

BERT는 sentence classification, paraphrase, natural language inference 등 다양한 task에서 좋은 결과를 보였다. 특히 BERT-Large는 많은 task에서 BERT-Base보다 더 높은 성능을 냈다.

### SQuAD

![bert experiment result 2]({{ '/assets/images/blog/bertscore02.png' | relative_url }})

*SQuAD는 주어진 지문에서 질문의 답이 되는 span을 찾는 추출형 QA benchmark다.*

BERT는 질문과 지문을 함께 encoding하고, 답의 시작과 끝 위치를 예측한다. 양방향 context를 보는 encoder 구조는 이런 추출형 QA에 잘 맞는다.

### SWAG와 모델 크기

![bert experiment result 3]({{ '/assets/images/blog/bertscore03.png' | relative_url }})

*SWAG는 주어진 상황 다음에 올 자연스러운 문장을 고르는 task다.*

![bert experiment result 4]({{ '/assets/images/blog/bertscore.png' | relative_url }})

*논문은 모델 크기가 커질수록 성능이 좋아지는 경향도 함께 보여준다.*

## 7. 한계점

BERT의 한계도 분명하다.

### 생성형 모델은 아니다

BERT는 bidirectional encoder라 자연어 이해에는 강하지만, 왼쪽에서 오른쪽으로 긴 텍스트를 생성하는 용도에는 GPT 계열보다 자연스럽지 않다.

### Fine-tuning이 필요하다

BERT는 downstream task마다 fine-tuning하는 패러다임이다. GPT-3 이후 중요해진 zero-shot, few-shot prompting과는 사용 방식이 다르다.

### Pretraining objective의 한계

MLM은 강력하지만, 실제 fine-tuning 입력에는 `[MASK]`가 등장하지 않는다. 논문은 이를 완화하는 masking 전략을 썼지만, objective mismatch가 완전히 사라지는 것은 아니다.

### NSP의 필요성 논쟁

이후 연구에서는 NSP가 항상 필요한지에 대해 여러 논의가 있었다. 즉 BERT의 모든 설계 요소가 그대로 유지되어야 하는 것은 아니다.

## 8. 이 논문의 핵심 의의

BERT의 핵심 의의는 Transformer Encoder 기반 양방향 사전학습을 자연어 이해 task의 표준 방식으로 만든 데 있다.

GPT 계열이 language generation과 prompting 흐름으로 이어졌다면, BERT는 encoder representation, text classification, retrieval, reranking, extraction 기반 NLP 시스템의 중요한 기반이 되었다.

특히 다음 흐름을 만들었다.

- pretraining 후 fine-tuning하는 NLP 패러다임의 대중화
- MLM 기반 bidirectional representation 학습
- `[CLS]` 기반 sentence representation 활용
- downstream task를 하나의 pretrained encoder 위에 얹는 방식

## 내가 이해한 핵심

BERT는 "문장을 왼쪽에서 오른쪽으로만 읽는 언어 모델"의 한계를 넘기 위해 양방향 Transformer Encoder를 사전학습한 모델이다.

핵심은 다음과 같다.

```text
입력 일부를 가리고,
양쪽 문맥으로 원래 token을 맞히게 하며,
그 representation을 downstream task에 fine-tuning한다.
```

BERT는 생성형 LLM과는 다른 방향이지만, 자연어 이해 모델의 기준선을 크게 끌어올린 논문이다.

## 한 줄 요약

BERT는 Masked Language Model로 양방향 Transformer Encoder를 사전학습해, 다양한 자연어 이해 task에서 강한 성능을 보인 encoder 기반 language representation 모델이다.

## 참고 자료

- Jacob Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805), NAACL 2019.
