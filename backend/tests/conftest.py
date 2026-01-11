"""
Shared fixtures for RAG system tests
"""
import pytest
import sys
import os
import tempfile
import shutil

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from models import Course, Lesson, CourseChunk


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_chroma_path):
    """Create a VectorStore with test configuration"""
    return VectorStore(
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5  # Use a reasonable default, not 0
    )


@pytest.fixture
def vector_store_with_zero_results(temp_chroma_path):
    """Create a VectorStore with max_results=0 to test the bug"""
    return VectorStore(
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=0  # This simulates the current production config
    )


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=0, title="Introduction", lesson_link="http://example.com/lesson0"),
        Lesson(lesson_number=1, title="Getting Started", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced Topics", lesson_link="http://example.com/lesson2"),
    ]
    return Course(
        title="Test Course: Python Basics",
        course_link="http://example.com/course",
        instructor="Test Instructor",
        lessons=lessons
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a programming language. It is used for web development, data science, and automation.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python store data. You can use integers, strings, and floats.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Functions are reusable blocks of code. Define them with the def keyword.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        ),
    ]


@pytest.fixture
def populated_vector_store(vector_store, sample_course, sample_chunks):
    """Create a VectorStore populated with test data"""
    vector_store.add_course_metadata(sample_course)
    vector_store.add_course_content(sample_chunks)
    return vector_store


@pytest.fixture
def populated_vector_store_zero_results(vector_store_with_zero_results, sample_course, sample_chunks):
    """Create a VectorStore with max_results=0 and test data"""
    vector_store_with_zero_results.add_course_metadata(sample_course)
    vector_store_with_zero_results.add_course_content(sample_chunks)
    return vector_store_with_zero_results


@pytest.fixture
def course_search_tool(populated_vector_store):
    """Create a CourseSearchTool with populated store"""
    return CourseSearchTool(populated_vector_store)


@pytest.fixture
def course_search_tool_zero_results(populated_vector_store_zero_results):
    """Create a CourseSearchTool with max_results=0"""
    return CourseSearchTool(populated_vector_store_zero_results)


@pytest.fixture
def tool_manager(course_search_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager
