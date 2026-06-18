"""BERT(encoder-only) 사전학습 패키지.

번역용 Transformer의 인코더 부품(EncoderLayer, PositionalEncoding, make_pad_mask)과
데이터 부품(tokenize, Vocab)을 재사용해 MLM(Masked Language Model)을 구현한다.
MLM 전용으로 시작하되, segment embedding 슬롯 / [CLS] pooler / 헤드 분리 구조를
미리 두어 풀 BERT(NSP·문장쌍)와 분류 fine-tuning으로 무리 없이 확장할 수 있다.
"""
