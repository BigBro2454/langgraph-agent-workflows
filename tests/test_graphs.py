import unittest
import os

# Set dummy key so imports and schema setups don't fail
os.environ.setdefault("GEMINI_API_KEY", "fake_key_for_testing")

import main
import hierarchical_blog_writer as hbw


class TestLangGraphWorkflows(unittest.TestCase):
    """Test suite for LangGraph agent workflow graphs and schemas."""

    def test_simple_graph_compilation_and_topology(self):
        """Verify the foundational chatbot StateGraph compiles and has correct nodes."""
        self.assertIsNotNone(main.graph)
        nodes = main.graph.nodes
        self.assertIn("chatbot", nodes)
        self.assertIn("__start__", nodes)

    def test_research_subgraph_topology(self):
        """Verify Research Team subgraph has supervisor and worker nodes."""
        self.assertIsNotNone(hbw.research_graph)
        nodes = hbw.research_graph.nodes
        self.assertIn("supervisor", nodes)
        self.assertIn("search_agent", nodes)
        self.assertIn("web_scraper_agent", nodes)

    def test_writing_subgraph_topology(self):
        """Verify Writing Team subgraph has supervisor and worker nodes."""
        self.assertIsNotNone(hbw.writing_graph)
        nodes = hbw.writing_graph.nodes
        self.assertIn("supervisor", nodes)
        self.assertIn("doc_writer_agent", nodes)
        self.assertIn("note_taker_agent", nodes)
        self.assertIn("chart_generator_agent", nodes)

    def test_super_graph_hierarchy(self):
        """Verify Top-Level Supervisor orchestrates subgraphs as first-class nodes."""
        self.assertIsNotNone(hbw.super_graph)
        nodes = hbw.super_graph.nodes
        self.assertIn("supervisor", nodes)
        self.assertIn("research_team", nodes)
        self.assertIn("writing_team", nodes)

    def test_pydantic_routing_schemas(self):
        """Verify Pydantic structured output models validate correctly."""
        # Research routing schema
        route1 = hbw.ResearchRoute(next="search_agent")
        self.assertEqual(route1.next, "search_agent")
        route2 = hbw.ResearchRoute(next="FINISH")
        self.assertEqual(route2.next, "FINISH")

        # Writing routing schema
        route3 = hbw.WritingRoute(next="doc_writer_agent")
        self.assertEqual(route3.next, "doc_writer_agent")
        route4 = hbw.WritingRoute(next="FINISH")
        self.assertEqual(route4.next, "FINISH")

        # Super routing schema
        route5 = hbw.SuperRoute(next="research_team")
        self.assertEqual(route5.next, "research_team")
        route6 = hbw.SuperRoute(next="FINISH")
        self.assertEqual(route6.next, "FINISH")


if __name__ == "__main__":
    unittest.main()
