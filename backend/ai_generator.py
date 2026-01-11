import ollama
import re
import json
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""

    # System prompt with tool calling instructions
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

## Available Tools
You have access to TWO tools:

1. **search_course_content**: Search course materials for specific information or content
   - Use for: questions about specific topics, concepts, or detailed content within courses
   - Arguments: query (required), course_name (optional), lesson_number (optional)

2. **get_course_outline**: Get the complete structure/syllabus of a course
   - Use for: questions about course structure, lesson lists, what a course covers, syllabus
   - Arguments: course_title (required)

## When to Use Each Tool
- Use **get_course_outline** for: "What lessons are in X course?", "Show me the outline of X", "What does X course cover?", "List the topics in X"
- Use **search_course_content** for: "What is X?", "How do I do X?", "Explain X from course Y"
- For general knowledge questions, answer directly without searching
- Maximum ONE tool call per query

## How to Use Tools
When you need to use a tool, output EXACTLY this format (no extra text before it):
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

Examples:
<tool_call>{"name": "get_course_outline", "arguments": {"course_title": "MCP"}}</tool_call>
<tool_call>{"name": "search_course_content", "arguments": {"query": "API endpoints", "course_name": "MCP"}}</tool_call>

## Response Guidelines
- Be brief and concise - get to the point quickly
- Provide direct answers only - no meta-commentary about searching
- Do not mention "based on the search results"
- When presenting a course outline, include the course title, course link, and all lessons with their numbers and titles
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
