"""
Tests for RAG system content query handling

These tests verify that the RAG system correctly:
1. Processes content-related queries
2. Uses the tool manager properly
3. Returns responses with sources
4. Handles configuration properly
"""
import pytest
import sys
import os
import tempfile
import shutil
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import VectorStore
from search_tools import ToolManager, CourseSearchTool
from models import Course, Lesson, CourseChunk


@dataclass
class TestConfig:
    """Test configuration with proper MAX_RESULTS"""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "test-model"
    OLLAMA_API_KEY: str = ""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Proper value, not 0
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = ""


@dataclass
class BuggyConfig:
    """Configuration that reproduces the MAX_RESULTS=0 bug"""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "test-model"
    OLLAMA_API_KEY: str = ""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 0  # THE BUG: This causes all searches to fail
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = ""


class TestRAGSystemConfiguration:
    """Tests related to RAG system configuration"""

    def test_config_max_results_value(self):
        """
        CRITICAL: Test that config.MAX_RESULTS is not 0

        This test documents the bug where MAX_RESULTS=0 causes searches to fail.
        """
        from config import config

        print(f"Current config.MAX_RESULTS = {config.MAX_RESULTS}")

        # This assertion will FAIL if the bug exists
        # MAX_RESULTS should be > 0 for search to work
        if config.MAX_RESULTS == 0:
            pytest.fail(
                f"BUG DETECTED: config.MAX_RESULTS is {config.MAX_RESULTS}. "
                "This causes all content searches to return empty results. "
                "Fix by setting MAX_RESULTS to a positive integer (e.g., 5)."
            )

    def test_vector_store_initialized_with_config_max_results(self):
        """Test that VectorStore receives MAX_RESULTS from config"""
        temp_dir = tempfile.mkdtemp()
        try:
            config = TestConfig()
            config.CHROMA_PATH = temp_dir

            # Create VectorStore directly to test initialization
            store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )

            assert store.max_results == 5, f"Expected max_results=5, got {store.max_results}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_course(self):
        """Create a sample course"""
        return Course(
            title="Test Course",
            course_link="http://example.com",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Intro", lesson_link="http://example.com/0"),
                Lesson(lesson_number=1, title="Basics", lesson_link="http://example.com/1"),
            ]
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample chunks"""
        return [
            CourseChunk(
                content="Python is a programming language.",
                course_title=sample_course.title,
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables store data values.",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=1
            ),
        ]

    def test_rag_system_with_working_config(self, temp_dir, sample_course, sample_chunks):
        """Test RAG system with proper MAX_RESULTS configuration"""
        config = TestConfig()
        config.CHROMA_PATH = temp_dir

        # Mock AIGenerator to avoid actual Ollama calls
        with patch('rag_system.AIGenerator') as MockAIGenerator:
            mock_ai = Mock()
            # Simulate the AI calling the search tool and returning a response
            mock_ai.generate_response.return_value = "Python is a programming language."
            MockAIGenerator.return_value = mock_ai

            rag = RAGSystem(config)

            # Add test data
            rag.vector_store.add_course_metadata(sample_course)
            rag.vector_store.add_course_content(sample_chunks)

            # Verify vector store has proper max_results
            assert rag.vector_store.max_results == 5

            # Test direct search (bypassing AI)
            results = rag.vector_store.search(query="Python")
            assert not results.is_empty(), "Search should return results with MAX_RESULTS=5"

    def test_rag_system_with_buggy_config(self, temp_dir, sample_course, sample_chunks):
        """
        Test RAG system with MAX_RESULTS=0 to demonstrate the bug
        """
        config = BuggyConfig()
        config.CHROMA_PATH = temp_dir

        # Mock AIGenerator
        with patch('rag_system.AIGenerator') as MockAIGenerator:
            mock_ai = Mock()
            MockAIGenerator.return_value = mock_ai

            rag = RAGSystem(config)

            # Add test data
            rag.vector_store.add_course_metadata(sample_course)
            rag.vector_store.add_course_content(sample_chunks)

            # Verify vector store has buggy max_results
            assert rag.vector_store.max_results == 0, "This test requires max_results=0"

            # Test direct search - this will fail/return empty due to the bug
            results = rag.vector_store.search(query="Python")

            print(f"Search with max_results=0: documents={results.documents}, error={results.error}")

            # Document the buggy behavior
            # ChromaDB with n_results=0 either errors or returns empty
            assert results.is_empty() or results.error is not None, \
                "With MAX_RESULTS=0, search should fail or return empty"


class TestToolManagerInRAGSystem:
    """Tests for ToolManager integration in RAG system"""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_tool_manager_has_both_tools_registered(self, temp_dir):
        """Test that both search and outline tools are registered"""
        config = TestConfig()
        config.CHROMA_PATH = temp_dir

        with patch('rag_system.AIGenerator'):
            rag = RAGSystem(config)

            tool_defs = rag.tool_manager.get_tool_definitions()
            tool_names = [t["name"] for t in tool_defs]

            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names

    def test_tool_manager_executes_search_tool(self, temp_dir):
        """Test that tool manager can execute search tool"""
        config = TestConfig()
        config.CHROMA_PATH = temp_dir

        with patch('rag_system.AIGenerator'):
            rag = RAGSystem(config)

            # Add some test data (must include all required fields to avoid None in metadata)
            course = Course(
                title="Test",
                course_link="http://example.com",
                instructor="Test Instructor",
                lessons=[Lesson(lesson_number=0, title="Test Lesson", lesson_link="http://example.com/0")]
            )
            chunk = CourseChunk(
                content="Test content about Python programming.",
                course_title="Test",
                lesson_number=0,
                chunk_index=0
            )
            rag.vector_store.add_course_metadata(course)
            rag.vector_store.add_course_content([chunk])

            # Execute tool through manager
            result = rag.tool_manager.execute_tool(
                "search_course_content",
                query="Python"
            )

            assert result is not None
            assert isinstance(result, str)
            # With proper config, should find content
            assert "No relevant content" not in result or "Test" in result


class TestEndToEndContentQuery:
    """End-to-end tests for content queries"""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_content_query_flow_with_mocked_ai(self, temp_dir):
        """Test the full query flow with mocked AI"""
        config = TestConfig()
        config.CHROMA_PATH = temp_dir

        with patch('rag_system.AIGenerator') as MockAIGenerator:
            # Setup mock AI that simulates tool calling
            mock_ai = Mock()

            def mock_generate(query, conversation_history=None, tools=None, tool_manager=None):
                # Simulate AI deciding to search
                if tool_manager:
                    # Execute the search
                    search_result = tool_manager.execute_tool(
                        "search_course_content",
                        query="Python"
                    )
                    return f"Based on the course materials: {search_result[:100]}"
                return "I don't have information about that."

            mock_ai.generate_response.side_effect = mock_generate
            mock_ai.model = config.OLLAMA_MODEL
            MockAIGenerator.return_value = mock_ai

            rag = RAGSystem(config)

            # Add test data (must include all required fields to avoid None in metadata)
            course = Course(
                title="Python Basics",
                course_link="http://example.com",
                instructor="Test Instructor",
                lessons=[Lesson(lesson_number=0, title="Intro", lesson_link="http://example.com/0")]
            )
            chunk = CourseChunk(
                content="Python is an interpreted programming language.",
                course_title="Python Basics",
                lesson_number=0,
                chunk_index=0
            )
            rag.vector_store.add_course_metadata(course)
            rag.vector_store.add_course_content([chunk])

            # Execute query
            response, sources = rag.query("What is Python?")

            print(f"Response: {response}")
            print(f"Sources: {sources}")

            assert response is not None
            assert isinstance(response, str)
            # Should have found content
            assert "Python" in response


class TestCurrentProductionConfig:
    """Tests using the actual production config to identify issues"""

    def test_production_config_max_results(self):
        """Check if production config has the bug"""
        from config import config

        if config.MAX_RESULTS == 0:
            print("\n" + "="*60)
            print("BUG FOUND IN PRODUCTION CONFIG!")
            print("="*60)
            print(f"config.MAX_RESULTS = {config.MAX_RESULTS}")
            print("\nThis causes all content searches to fail.")
            print("\nFIX: In backend/config.py, change:")
            print("  MAX_RESULTS: int = 0")
            print("TO:")
            print("  MAX_RESULTS: int = 5")
            print("="*60 + "\n")

            pytest.fail("Production config has MAX_RESULTS=0 bug")
        else:
            print(f"\nProduction config.MAX_RESULTS = {config.MAX_RESULTS} (OK)")
