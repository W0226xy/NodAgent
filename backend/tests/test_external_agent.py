import json
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage

from backend.app.agents import external_agent
from backend.app.mcp_servers import (
    external_info_server,
)


class FakeMCPAdapter:
    def __init__(
        self,
        client,
    ):
        self.client = client

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    async def list_tools(
        self,
        cache_mode,
    ):
        return [
            f"{self.client}-tool"
        ]


class FakeAgent:
    async def ainvoke(
        self,
        payload,
        config=None,
    ):
        del payload, config
        return {
            "messages": [
                AIMessage(
                    content="模拟的外部工具回答"
                )
            ]
        }


class FailingAsyncClient:
    def __init__(
        self,
        **kwargs,
    ):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    async def get(
        self,
        *args,
        **kwargs,
    ):
        raise httpx.ConnectError(
            "simulated network failure"
        )


class ExternalAgentTests(
    unittest.IsolatedAsyncioTestCase
):
    def test_select_mcp_servers(self):
        self.assertEqual(
            external_agent.select_mcp_servers(
                "上海今天天气怎么样？"
            ),
            "external_info",
        )
        self.assertEqual(
            external_agent.select_mcp_servers(
                "查看 GitHub 仓库的 README"
            ),
            "github",
        )
        self.assertEqual(
            external_agent.select_mcp_servers(
                "上海天气和 GitHub 仓库状态"
            ),
            "both",
        )

    async def test_weather_does_not_create_github_client(self):
        github_factory = Mock(
            side_effect=RuntimeError(
                "GitHub MCP is unavailable"
            )
        )
        web_search_factory = Mock()

        with (
            patch.object(
                external_agent,
                "create_external_info_client",
                return_value="external-info",
            ),
            patch.object(
                external_agent,
                "create_github_mcp_client",
                github_factory,
            ),
            patch.object(
                external_agent,
                "create_web_search_client",
                web_search_factory,
            ),
            patch.object(
                external_agent,
                "MCPAdapter",
                side_effect=FakeMCPAdapter,
            ),
            patch.object(
                external_agent,
                "get_chat_model",
                return_value=object(),
            ),
            patch.object(
                external_agent,
                "create_agent",
                return_value=FakeAgent(),
            ),
        ):
            answer = await (
                external_agent.run_external_agent(
                    "上海今天天气怎么样？"
                )
            )

        self.assertEqual(
            answer,
            "模拟的外部工具回答",
        )
        github_factory.assert_not_called()
        web_search_factory.assert_not_called()

    async def test_github_task_does_not_create_local_client(self):
        local_factory = Mock()

        with (
            patch.object(
                external_agent,
                "create_external_info_client",
                local_factory,
            ),
            patch.object(
                external_agent,
                "create_github_mcp_client",
                return_value="github",
            ),
            patch.object(
                external_agent,
                "MCPAdapter",
                side_effect=FakeMCPAdapter,
            ),
            patch.object(
                external_agent,
                "get_chat_model",
                return_value=object(),
            ),
            patch.object(
                external_agent,
                "create_agent",
                return_value=FakeAgent(),
            ),
        ):
            answer = await (
                external_agent.run_external_agent(
                    "查看 GitHub 仓库的 README"
                )
            )

        self.assertEqual(
            answer,
            "模拟的外部工具回答",
        )
        local_factory.assert_not_called()

    async def test_github_failure_does_not_block_available_web_search(self):
        github_factory = Mock(
            side_effect=RuntimeError(
                "GitHub MCP is unavailable"
            )
        )

        with (
            patch.object(
                external_agent,
                "TAVILY_MCP_URL",
                "https://mcp.example.test/mcp/",
            ),
            patch.object(
                external_agent,
                "TAVILY_API_KEY",
                "test-key",
            ),
            patch.object(
                external_agent,
                "create_github_mcp_client",
                github_factory,
            ),
            patch.object(
                external_agent,
                "create_web_search_client",
                return_value="web-search",
            ),
            patch.object(
                external_agent,
                "MCPAdapter",
                side_effect=FakeMCPAdapter,
            ),
            patch.object(
                external_agent,
                "get_chat_model",
                return_value=object(),
            ),
            patch.object(
                external_agent,
                "create_agent",
                return_value=FakeAgent(),
            ),
        ):
            answer = await external_agent.run_external_agent(
                "搜索 GitHub 上的 RAG 仓库",
                need_web_search=True,
            )

        self.assertEqual(
            answer,
            "模拟的外部工具回答",
        )
        github_factory.assert_called_once()

    async def test_local_mcp_failure_returns_safe_answer(self):
        with (
            patch.object(
                external_agent,
                "create_external_info_client",
                return_value="external-info",
            ),
            patch.object(
                external_agent,
                "MCPAdapter",
                side_effect=RuntimeError(
                    "Local MCP is unavailable"
                ),
            ),
        ):
            answer = await (
                external_agent.run_external_agent(
                    "上海今天天气怎么样？"
                )
            )

        self.assertEqual(
            answer,
            "暂时无法获取实时天气或时间信息，请稍后重试。",
        )

    async def test_weather_network_failure_is_safe(self):
        with patch.object(
            external_info_server.httpx,
            "AsyncClient",
            FailingAsyncClient,
        ):
            response = await (
                external_info_server.get_weather(
                    "上海"
                )
            )

        payload = json.loads(
            response
        )

        self.assertEqual(
            payload["error"]["code"],
            "weather_service_unavailable",
        )
        self.assertEqual(
            payload["error"]["message"],
            "天气服务暂时不可访问，请稍后重试。",
        )


if __name__ == "__main__":
    unittest.main()
