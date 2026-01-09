import ollama
import re
import json
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""

    # System prompt with tool calling instructions
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a search tool for course information.

## Available Tool
You have access to ONE tool:
- **search_course_content**: Search course materials for specific information

## When to Use the Tool
- Use the search tool ONLY for questions about specific course content or detailed educational materials
- For general knowledge questions, answer directly without searching
- Maximum ONE search per query

## How to Use the Tool
When you need to search, output EXACTLY this format (no extra text before it):
<tool_call>{"name": "search_course_content", "arguments": {"query": "your search query here"}}</tool_call>

Optional arguments:
- "course_name": filter by course (partial matches work, e.g., "MCP", "Introduction")
- "lesson_number": filter by lesson number (integer)

Example with filters:
<tool_call>{"name": "search_course_content", "arguments": {"query": "API endpoints", "course_name": "MCP"}}</tool_call>

## Response Guidelines
- Be brief and concise - get to the point quickly
- Provide direct answers only - no meta-commentary about searching
- Do not mention "based on the search results"
- Include relevant examples when they aid understanding
"""

    def __init__(self, base_url: str, model: str, api_key: str = None):
        self.base_url = base_url
        self.model = model
        # Support for Ollama Cloud with API key
        if api_key:
            self.client = ollama.Client(
                host=base_url,
                headers={"Authorization": f"Bearer {api_key}"}
            )
        else:
            self.client = ollama.Client(host=base_url)

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (used for prompt building)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content with conversation history if present
        system_content = self.SYSTEM_PROMPT
        if conversation_history:
            system_content = f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"

        # Build messages for Ollama
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Get initial response from Ollama
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0}
        )

        response_text = response["message"]["content"]

        # Check if the model wants to use a tool
        tool_call = self._extract_tool_call(response_text)

        if tool_call and tool_manager:
            return self._handle_tool_execution(tool_call, response_text, messages, tool_manager)

        # Return direct response (clean any partial tool artifacts)
        return self._clean_response(response_text)

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from response text if present."""
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        return None

    def _handle_tool_execution(self, tool_call: Dict[str, Any],
                               initial_response: str,
                               messages: List[Dict],
                               tool_manager) -> str:
        """
        Handle execution of tool call and get follow-up response.

        Args:
            tool_call: Parsed tool call with name and arguments
            initial_response: The original response containing the tool call
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Execute the tool
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})

        tool_result = tool_manager.execute_tool(tool_name, **tool_args)

        # Build follow-up messages with tool result
        messages_with_result = messages.copy()
        messages_with_result.append({
            "role": "assistant",
            "content": initial_response
        })
        messages_with_result.append({
            "role": "user",
            "content": f"Tool result:\n{tool_result}\n\nNow provide your final answer based on these search results. Be concise and do not mention that you searched."
        })

        # Get final response
        final_response = self.client.chat(
            model=self.model,
            messages=messages_with_result,
            options={"temperature": 0}
        )

        return self._clean_response(final_response["message"]["content"])

    def _clean_response(self, text: str) -> str:
        """Remove any tool call artifacts from the response."""
        # Remove tool_call tags and their content
        cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        return cleaned.strip()
