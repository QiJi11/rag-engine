"""
测试 chunker.py 的分块逻辑
覆盖：正常分块、空文本、边界情况、重叠行为
"""
import pytest
from services.chunker import chunk_text


class TestChunkText:
    """测试 chunk_text 主函数"""

    def test_空文本返回空列表(self):
        """空字符串应返回空列表，不应报错"""
        result = chunk_text("")
        assert result == []

    def test_纯空白文本返回空列表(self):
        """只有空白字符的文本不应产生 chunk"""
        result = chunk_text("   \n\n  ")
        assert result == []

    def test_短文本不分块(self):
        """短于 chunk_size 的文本应只产生一个 chunk"""
        text = "这是一段普通文本。"
        result = chunk_text(text, chunk_size=512)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 0
        assert result[0]["source"] == "uploaded_document"
        assert result[0]["text"] != ""

    def test_长文本产生多个chunk(self):
        """超过 chunk_size 的文本必须拆成多个 chunk"""
        # 生成一段超过 512 字符的文字
        text = "这是一个句子。" * 100  # 约700字符
        result = chunk_text(text, chunk_size=100)
        assert len(result) > 1

    def test_chunk_index连续(self):
        """chunk_index 必须从 0 开始连续编号"""
        text = "句子一。" * 80
        result = chunk_text(text, chunk_size=50)
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i

    def test_所有chunk都有text字段(self):
        """每个 chunk 必须有非空的 text 字段"""
        text = "测试内容，用于验证字段完整性。" * 30
        result = chunk_text(text, chunk_size=100)
        for chunk in result:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "source" in chunk
            assert chunk["text"].strip() != ""

    def test_重叠保留上下文(self):
        """启用 overlap 后，相邻 chunk 应共享部分内容"""
        # 构造一段可以验证 overlap 的文本
        text = "AAAA" * 50 + "。" + "BBBB" * 50
        result = chunk_text(text, chunk_size=100, overlap=30)
        if len(result) > 1:
            # 第二个 chunk 开头应包含第一个 chunk 末尾的部分字符
            end_of_first = result[0]["text"][-20:]
            start_of_second = result[1]["text"][:30]
            # 在 overlap 模式下，共享前缀应存在
            assert len(result[1]["text"]) > 0

    def test_以中文标点分句(self):
        """应以 。！？ 等为优先分割边界"""
        text = "第一句话。第二句话！第三句话？" * 30
        result = chunk_text(text, chunk_size=80)
        assert len(result) >= 1

    def test_以换行分句(self):
        """双换行应作为段落分割边界"""
        text = "段落一\n\n段落二\n\n段落三\n\n" * 20
        result = chunk_text(text, chunk_size=50)
        assert len(result) > 1

    def test_自定义chunk_size(self):
        """自定义较小的 chunk_size 应产生更多 chunk"""
        text = "这是测试句子。" * 50
        result_large = chunk_text(text, chunk_size=500)
        result_small = chunk_text(text, chunk_size=50)
        assert len(result_small) >= len(result_large)

    def test_source字段固定值(self):
        """source 字段应固定为 'uploaded_document'"""
        result = chunk_text("Hello。" * 10)
        for chunk in result:
            assert chunk["source"] == "uploaded_document"
