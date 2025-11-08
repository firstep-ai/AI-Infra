#!/usr/bin/env python3
"""
测试脚本：测试 OpenRouter API 的 max_completion_tokens 行为

此脚本测试 OpenRouter API（使用 OpenAI o1 模型）的 max_completion_tokens 参数：
- max_completion_tokens 只限制最终内容
- 推理过程（reasoning_content）不计入 max_completion_tokens
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# 从 .env 文件加载环境变量
load_dotenv()

# 配置
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenRouter 使用相同的环境变量

# 测试参数
TEST_PROMPT = "用简单的话解释量子纠缠的概念。"
MAX_COMPLETION_TOKENS = 100  # 只限制最终内容


def test_openai_api():
    """使用 OpenRouter API 测试（通过 OpenAI o1 模型，预期行为）"""
    print("=" * 80)
    print("测试 OpenRouter API（OpenAI o1 模型）")
    print("=" * 80)
    
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openai-api-key-here":
        print("⚠️  OpenRouter API key 未配置，跳过 OpenRouter 测试")
        print("   请在 .env 文件中设置 OPENAI_API_KEY")
        return None
    
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
    )
    
    try:
        # 使用支持推理的模型（如 o1-preview, o1-mini）
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",  # 或 "openai/o1-preview"
            messages=[
                {"role": "user", "content": TEST_PROMPT}
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        
        # 提取信息
        choice = response.choices[0]
        message = choice.message
        
        # 统计 token 数量
        reasoning_tokens = 0
        content_tokens = 0
        print(response)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_tokens = len(message.reasoning_content.split())  # 粗略估计
        
        if message.content:
            content_tokens = len(message.content.split())  # 粗略估计
        
        total_tokens = response.usage.completion_tokens if response.usage else 0
        
        print(f"\n✅ OpenRouter API 响应:")
        print(f"   模型: {response.model}")
        print(f"   最大完成 Token 数: {MAX_COMPLETION_TOKENS}")
        print(f"   推理 Token 数（估计）: {reasoning_tokens}")
        print(f"   内容 Token 数（估计）: {content_tokens}")
        print(f"   总完成 Token 数: {total_tokens}")
        print(f"\n   推理内容长度: {len(message.reasoning_content) if hasattr(message, 'reasoning_content') else 0} 字符")
        print(f"   最终内容长度: {len(message.content) if message.content else 0} 字符")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"\n   推理内容（前 200 字符）:")
            print(f"   {message.reasoning_content[:200]}...")
        
        print(f"\n   最终内容:")
        print(f"   {message.content}")
        
        print(f"\n📊 预期行为:")
        print(f"   - 推理 token 不应计入 max_completion_tokens")
        print(f"   - 只有最终内容应被限制在 {MAX_COMPLETION_TOKENS} tokens")
        print(f"   - 总 token 数可以超过 {MAX_COMPLETION_TOKENS}")
        
        return {
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": content_tokens,
            "total_tokens": total_tokens,
            "response": response
        }
        
    except Exception as e:
        print(f"❌ 测试 OpenRouter API 时出错: {e}")
        return None




def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("OpenRouter API max_completion_tokens 测试")
    print("=" * 80)
    print(f"\n测试配置:")
    print(f"  提示词: {TEST_PROMPT}")
    print(f"  最大完成 Tokens: {MAX_COMPLETION_TOKENS}")
    
    # 测试 OpenRouter API
    openai_result = test_openai_api()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    if openai_result:
        print("\n📊 结果分析:")
        print(f"   - 推理 Token 数: {openai_result['reasoning_tokens']}")
        print(f"   - 内容 Token 数: {openai_result['content_tokens']}")
        print(f"   - 总 Token 数: {openai_result['total_tokens']}")
        print(f"   - max_completion_tokens 设置: {MAX_COMPLETION_TOKENS}")
        
        if openai_result['total_tokens'] > MAX_COMPLETION_TOKENS:
            print(f"\n✅ 符合预期: 总 Token 数 ({openai_result['total_tokens']}) > max_completion_tokens ({MAX_COMPLETION_TOKENS})")
            print(f"   说明推理过程不计入 max_completion_tokens 限制")
        else:
            print(f"\n⚠️  总 Token 数 ({openai_result['total_tokens']}) ≤ max_completion_tokens ({MAX_COMPLETION_TOKENS})")
            print(f"   可能推理内容较少或模型未使用推理模式")


if __name__ == "__main__":
    main()

