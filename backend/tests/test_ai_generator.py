"""
Tests for AIGenerator tool calling functionality

These tests verify that the AIGenerator correctly:
1. Extracts tool calls from response text
2. Handles tool execution
3. Cleans responses properly
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from unittest.mock import Mock, patch, MagicMock


class TestToolCallExtraction:
    """Tests for _extract_tool_call method"""

    @pytest.fixture
    def ai_generator(self):
        """Create an AIGenerator for testing (doesn't connect to Ollama)"""
        return AIGenerator(
            base_url="http://localhost:11434",
            model="test-model"
        )

    def test_extract_tool_call_with_valid_json(self, ai_generator):
        """Test extraction of valid tool call"""
        text = '<tool_call>{"name": "search_course_content", "arguments": {"query": "Python basics"}}</tool_call>'

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "search_course_content"
        assert result["arguments"]["query"] == "Python basics"

    def test_extract_tool_call_with_multiple_arguments(self, ai_generator):
        """Test extraction of tool call with multiple arguments"""
        text = '<tool_call>{"name": "search_course_content", "arguments": {"query": "API", "course_name": "MCP"}}</tool_call>'

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "search_course_content"
        assert result["arguments"]["query"] == "API"
        assert result["arguments"]["course_name"] == "MCP"

    def test_extract_tool_call_with_outline_tool(self, ai_generator):
        """Test extraction of get_course_outline tool call"""
        text = '<tool_call>{"name": "get_course_outline", "arguments": {"course_title": "MCP"}}</tool_call>'

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "get_course_outline"
        assert result["arguments"]["course_title"] == "MCP"

    def test_extract_tool_call_with_text_before(self, ai_generator):
        """Test extraction when there's text before the tool call"""
        text = 'Let me search for that information.\n<tool_call>{"name": "search_course_content", "arguments": {"query": "test"}}</tool_call>'

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "search_course_content"

    def test_extract_tool_call_with_text_after(self, ai_generator):
        """Test extraction when there's text after the tool call"""
        text = '<tool_call>{"name": "search_course_content", "arguments": {"query": "test"}}</tool_call>\nSearching now...'

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "search_course_content"

    def test_extract_tool_call_with_no_tool_call(self, ai_generator):
        """Test that None is returned when no tool call is present"""
        text = "This is just a regular response without any tool call."

        result = ai_generator._extract_tool_call(text)

        assert result is None

    def test_extract_tool_call_with_invalid_json(self, ai_generator):
        """Test handling of invalid JSON in tool call"""
        text = '<tool_call>{"name": "search_course_content", "arguments": {invalid}}</tool_call>'

        result = ai_generator._extract_tool_call(text)

        assert result is None

    def test_extract_tool_call_with_multiline_json(self, ai_generator):
        """Test extraction of multiline JSON tool call"""
        text = '''<tool_call>{
            "name": "search_course_content",
            "arguments": {
                "query": "Python basics"
            }
        }</tool_call>'''

        result = ai_generator._extract_tool_call(text)

        assert result is not None
        assert result["name"] == "search_course_content"


class TestResponseCleaning:
    """Tests for _clean_response method"""

    @pytest.fixture
    def ai_generator(self):
        return AIGenerator(
            base_url="http://localhost:11434",
            model="test-model"
        )

    def test_clean_response_removes_tool_call(self, ai_generator):
        """Test that tool calls are removed from response"""
        text = 'Before <tool_call>{"name": "test"}</tool_call> After'

        result = ai_generator._clean_response(text)

        assert "<tool_call>" not in result
        assert "</tool_call>" not in result
        assert "Before" in result
        assert "After" in result

    def test_clean_response_strips_whitespace(self, ai_generator):
        """Test that response is stripped of leading/trailing whitespace"""
        text = '  \n  Response content  \n  '

        result = ai_generator._clean_response(text)

        assert result == "Response content"

    def test_clean_response_handles_no_tool_call(self, ai_generator):
        """Test that clean response works when there's no tool call"""
        text = "Just a regular response"

        result = ai_generator._clean_response(text)

        assert result == "Just a regular response"


class TestToolManagerIntegration:
    """Tests for ToolManager.execute_tool integration"""

    def test_execute_tool_calls_correct_tool(self, tool_manager):
        """Test that execute_tool calls the correct tool"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="Python"
        )

        assert result is not None
        assert isinstance(result, str)

    def test_execute_tool_with_unknown_tool(self, tool_manager):
        """Test handling of unknown tool name"""
        result = tool_manager.execute_tool(
            "unknown_tool",
            query="test"
        )

        assert "not found" in result.lower()

    def test_execute_tool_passes_arguments(self, tool_manager):
        """Test that arguments are passed correctly to the tool"""
        # This test verifies the tool receives the correct arguments
        result = tool_manager.execute_tool(
            "search_course_content",
            query="functions",
            course_name="Python"
        )

        assert result is not None


class TestAIGeneratorToolExecution:
    """Tests for the full tool execution flow in AIGenerator"""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client"""
        mock_client = MagicMock()
        return mock_client

    def test_handle_tool_execution_calls_tool_manager(self):
        """Test that _handle_tool_execution properly calls the tool manager"""
        ai_generator = AIGenerator(
            base_url="http://localhost:11434",
            model="test-model"
        )

        # Create a mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Python is great"

        # Create mock Ollama client that returns a response
        ai_generator.client = Mock()
        ai_generator.client.chat.return_value = {
            "message": {"content": "Python is a programming language used for many things."}
        }

        tool_call = {
            "name": "search_course_content",
            "arguments": {"query": "Python"}
        }
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What is Python?"}
        ]

        result = ai_generator._handle_tool_execution(
            tool_call=tool_call,
            initial_response="<tool_call>...</tool_call>",
            messages=messages,
            tool_manager=mock_tool_manager
        )

        # Verify tool was called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python"
        )

        # Verify result is returned
        assert result is not None
        assert isinstance(result, str)

    def test_generate_response_without_tool_call(self):
        """Test generate_response when model doesn't use a tool"""
        ai_generator = AIGenerator(
            base_url="http://localhost:11434",
            model="test-model"
        )

        # Mock client to return a direct response (no tool call)
        ai_generator.client = Mock()
        ai_generator.client.chat.return_value = {
            "message": {"content": "Python is a programming language."}
        }

        result = ai_generator.generate_response(
            query="What is Python?",
            tool_manager=None
        )

        assert result == "Python is a programming language."

    def test_generate_response_extracts_and_executes_tool(self):
        """Test the full flow: extract tool call and execute it"""
        ai_generator = AIGenerator(
            base_url="http://localhost:11434",
            model="test-model"
        )

        # Mock client: first call returns tool call, second call returns final answer
        ai_generator.client = Mock()
        ai_generator.client.chat.side_effect = [
            {"message": {"content": '<tool_call>{"name": "search_course_content", "arguments": {"query": "Python"}}</tool_call>'}},
            {"message": {"content": "Python is a programming language used for web development."}}
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python content from course..."

        result = ai_generator.generate_response(
            query="What is Python?",
            tool_manager=mock_tool_manager
        )

        # Tool should have been called
        mock_tool_manager.execute_tool.assert_called_once()

        # Should return the final response
        assert "Python" in result
        assert "programming language" in result
