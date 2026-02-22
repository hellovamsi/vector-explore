from __future__ import annotations

from vector_explore.app import CompareCommand, QueryCommand, WizardCommand
from vector_explore.cli import _build_parser, _command_from_args


def test_cli_defaults_to_wizard_command():
    parser = _build_parser()
    args = parser.parse_args([])
    cmd = _command_from_args(args)
    assert isinstance(cmd, WizardCommand)
    assert cmd.embed_batch_size == 64
    assert cmd.index_batch_size == 256


def test_cli_query_maps_to_query_command():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "query",
            "--novel",
            "mobydick",
            "--method",
            "semantic",
            "--embed-backend",
            "st",
            "--embed-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--store",
            "lancedb",
            "--question",
            "What are the major themes?",
            "--top-k",
            "7",
            "--no-llm",
            "--no-vectors",
        ]
    )
    cmd = _command_from_args(args)
    assert isinstance(cmd, QueryCommand)
    assert cmd.novel == "mobydick"
    assert cmd.top_k == 7
    assert cmd.no_llm is True
    assert cmd.no_vectors is True


def test_cli_compare_splits_methods():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--novel",
            "pride",
            "--methods",
            "fixed, semantic, sentences",
            "--embed-backend",
            "st",
            "--embed-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--store",
            "lancedb",
            "--question",
            "Q",
        ]
    )
    cmd = _command_from_args(args)
    assert isinstance(cmd, CompareCommand)
    assert cmd.methods == ["fixed", "semantic", "sentences"]

