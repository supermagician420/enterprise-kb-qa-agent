#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简企业知识库问答助手（教学版）

数据文件：同目录下的 knowledge.txt，段落之间用空行分隔。
依赖：仅 Python 3.8+ 标准库，不需要 pip install。

运行前设置环境变量（以 DeepSeek 的 OpenAI 兼容接口为例）：
    PowerShell:
        $env:LLM_API_KEY="你的 key"
        $env:LLM_BASE_URL="https://api.deepseek.com/v1"
        $env:LLM_MODEL="deepseek-chat"
        python .\kb_assistant.py

说明：这个脚本故意保留了 3 个缺陷，用于演示“能跑”和“可靠”之间的差距：
    1. 检索不到相关内容时仍然把前几段塞给模型，且不要求模型“不知道就说不知道”；
    2. 不校验空输入/超长输入；
    3. API 请求没有任何异常处理。
"""

import json
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge.txt")

API_BASE = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
TOP_K = 2
TIMEOUT = 60


def load_chunks(path):
    """按空行切分知识库，每个非空段落作为一个知识块。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def question_tokens(text):
    """非常朴素的“关键词”：英文/数字按单词，中文按相邻两字切片。"""
    text = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for i in range(len(block) - 1):
            tokens.add(block[i:i + 2])
    return tokens


def retrieve(question, chunks, top_k=TOP_K):
    """按关键词出现次数打分，取分数最高的 top_k 段。"""
    tokens = question_tokens(question)
    scored = []
    for index, chunk in enumerate(chunks):
        score = sum(chunk.lower().count(token) for token in tokens)
        scored.append((score, index, chunk))

    # 分数相同时按原顺序，保证结果可复现。
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    # 【缺陷 1】没有判断 score == 0，也没有相关性阈值。
    # 即使一个问题跟知识库毫无关系，也会把前几段当作“资料”交给模型。
    return [chunk for score, index, chunk in scored[:top_k]]


def read_question():
    """支持命令行参数，也支持用管道喂入一行超长文本。"""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return input("请输入你的问题：")


def build_prompt(question, chunks):
    """把检索片段直接拼进提示词。"""
    context = "\n\n".join(chunks)
    # 【缺陷 1，续】这里没有“资料中没有答案时必须拒答”的约束，
    # 模型很容易利用常识补全出一份看似合理、实际不存在的公司制度。
    return (
        "你是企业知识库问答助手。请根据下面的知识库资料回答员工问题。\n\n"
        "【知识库资料】\n"
        f"{context}\n\n"
        "【员工问题】\n"
        f"{question}\n"
    )


def call_llm(prompt):
    """调用 OpenAI 兼容的 /chat/completions 接口。"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是企业知识库问答助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )

    # 【缺陷 3】没有 try/except：断网、401、超时、上下文超限都会直接抛异常。
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = response.read().decode("utf-8")

    data = json.loads(body)
    return data["choices"][0]["message"]["content"]


def main():
    chunks = load_chunks(KB_PATH)

    # 【缺陷 2】不检查空输入，也不限制问题长度；空字符串会照常进入检索和 API 调用，
    # 超长文本会原样拼进提示词，直到服务端返回 400 或本地内存/耗时失控。
    question = read_question()

    hits = retrieve(question, chunks)
    print("== 检索到的知识片段 ==")
    for i, chunk in enumerate(hits, 1):
        print(f"[{i}] {chunk.splitlines()[0]}")
    print()

    prompt = build_prompt(question, hits)
    answer = call_llm(prompt)
    print("== 回答 ==")
    print(answer)


if __name__ == "__main__":
    main()
