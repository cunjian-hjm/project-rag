# eval/_vertexai_shim.py
"""
兼容性补丁（必须在 import ragas 之前加载）
------------------------------------------------
背景：ragas 0.4.3 的 ragas/llms/base.py 第 12 行仍写着
    from langchain_community.chat_models.vertexai import ChatVertexAI
但 langchain-community 0.4.x 已把 VertexAI 拆包到 langchain-google-vertexai，
导致任何 `import ragas` 直接抛 ModuleNotFoundError（上游已知 bug，issue #2745）。

处理：ChatVertexAI 在 ragas 内部仅用于 isinstance 类型登记，本项目走
DashScope/OpenAI 兼容接口，永远不会用到 Vertex。这里在 sys.modules 中
预注册一个假的模块 + 占位类，让 ragas 的 import 链顺利穿过。

注意：本文件必须在 eval_main.py 中"先于 ragas 相关导入"执行。
"""

import sys
import types


class _StubChatVertexAI:
    """占位类：仅为满足 ragas 的 `from ... import ChatVertexAI`，永不实例化"""
    pass


def install() -> None:
    """把桩模块注册进 sys.modules（幂等，可重复调用）"""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)
    stub.ChatVertexAI = _StubChatVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = stub
