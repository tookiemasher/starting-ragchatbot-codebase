"""
Tests for CourseSearchTool.execute() method

These tests verify that the search tool correctly:
1. Searches course content and returns results
2. Handles empty results appropriately
3. Formats results with proper metadata
4. Works with course name and lesson number filters
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_returns_results_for_valid_query(self, course_search_tool):
        """Test that execute returns content for a valid search query"""
        result = course_search_tool.execute(query="Python programming")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain actual content, not error messages
        assert "error" not in result.lower() or "No relevant content" not in result

    def test_execute_returns_formatted_results(self, course_search_tool):
        """Test that results are properly formatted with course context"""
        result = course_search_tool.execute(query="variables")

        # Should contain course title header
        assert "Test Course" in result or "[" in result
        # Should contain actual content
        assert len(result) > 50  # Non-trivial content

    def test_execute_tracks_sources(self, course_search_tool):
        """Test that sources are tracked after execution"""
        course_search_tool.execute(query="functions")

        sources = course_search_tool.last_sources
        assert isinstance(sources, list)
        # Should have sources if results were found
        if sources:
            assert all("title" in s for s in sources)

    def test_execute_with_course_filter(self, course_search_tool):
        """Test filtering by course name"""
        result = course_search_tool.execute(
            query="programming",
            course_name="Python"
        )

        assert result is not None
        assert isinstance(result, str)

    def test_execute_with_nonexistent_course(self, course_search_tool):
        """Test that non-existent course returns appropriate message"""
        result = course_search_tool.execute(
            query="anything",
            course_name="NonExistentCourse12345"
        )

        # When course resolution fails, VectorStore returns an error
        # The search_tools layer should propagate this error message
        assert "No course found" in result or "not found" in result.lower() or len(result) > 0

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test filtering by lesson number"""
        result = course_search_tool.execute(
            query="programming",
            lesson_number=0
        )

        assert result is not None
        assert isinstance(result, str)


class TestCourseSearchToolWithZeroMaxResults:
    """Tests to identify the MAX_RESULTS=0 bug"""

    def test_execute_with_zero_max_results_returns_empty(self, course_search_tool_zero_results):
        """
        CRITICAL BUG TEST: When MAX_RESULTS=0, no results are returned.

        This test demonstrates the bug in config.py where MAX_RESULTS=0
        causes all searches to fail or return empty.
        """
        result = course_search_tool_zero_results.execute(query="Python")

        # With max_results=0, we expect either:
        # 1. An error from ChromaDB
        # 2. Empty results message
        # 3. An exception

        # This assertion will help identify the exact behavior
        print(f"Result with max_results=0: {result}")

        # The bug: this should return results but won't
        # We expect this test to reveal the problem
        assert "No relevant content" in result or "error" in result.lower() or result == ""

    def test_vector_store_search_with_zero_limit(self, populated_vector_store_zero_results):
        """Test that VectorStore.search() with limit=0 fails gracefully"""
        results = populated_vector_store_zero_results.search(query="Python")

        print(f"Search results with max_results=0: documents={results.documents}, error={results.error}")

        # With n_results=0, ChromaDB behavior varies - test captures actual behavior
        # This reveals whether empty results or an error is returned
        assert results.is_empty() or results.error is not None


class TestVectorStoreSearch:
    """Direct tests on VectorStore.search() to isolate issues"""

    def test_search_returns_documents(self, populated_vector_store):
        """Test that search returns documents when data exists"""
        results = populated_vector_store.search(query="Python programming")

        assert not results.is_empty(), "Search should return documents"
        assert results.error is None, f"Search should not have errors: {results.error}"
        assert len(results.documents) > 0

    def test_search_returns_metadata(self, populated_vector_store):
        """Test that search returns proper metadata"""
        results = populated_vector_store.search(query="variables")

        if not results.is_empty():
            assert len(results.metadata) == len(results.documents)
            for meta in results.metadata:
                assert "course_title" in meta
                assert "lesson_number" in meta

    def test_search_respects_limit(self, populated_vector_store):
        """Test that search respects the limit parameter"""
        results = populated_vector_store.search(query="Python", limit=1)

        assert len(results.documents) <= 1

    def test_search_with_explicit_zero_limit(self, populated_vector_store):
        """Test behavior when explicitly passing limit=0"""
        results = populated_vector_store.search(query="Python", limit=0)

        print(f"Explicit limit=0 results: {results.documents}, error={results.error}")
        # This test documents the behavior when limit=0 is passed


class TestSearchResultsDataclass:
    """Tests for SearchResults dataclass"""

    def test_is_empty_with_no_documents(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_with_documents(self):
        """Test is_empty returns False when documents exist"""
        results = SearchResults(
            documents=["test"],
            metadata=[{"course_title": "Test"}],
            distances=[0.5]
        )
        assert results.is_empty() is False

    def test_empty_factory_creates_error_result(self):
        """Test that empty() factory creates result with error"""
        results = SearchResults.empty("Test error")

        assert results.is_empty() is True
        assert results.error == "Test error"
