from mare.objects import extract_document_objects
from mare.types import ObjectType


def test_extract_document_objects_finds_procedures_and_sections() -> None:
    text = """
    1. Loosen the screws.
    2. Lift the board.
    Figure 1 shows the connector alignment.
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=1)
    object_types = [obj.object_type for obj in objects]

    assert ObjectType.PROCEDURE in object_types
    assert ObjectType.FIGURE in object_types
    assert ObjectType.SECTION in object_types


def test_extract_document_objects_only_creates_tables_for_real_table_mentions() -> None:
    text = "Use the adjustable torque driver carefully."
    objects = extract_document_objects(text, doc_id="doc-1", page=1)

    assert all(obj.object_type != ObjectType.TABLE for obj in objects)


def test_extract_document_objects_supports_inline_numbered_manual_steps() -> None:
    text = "Wired LAN 1 Connect a LAN adapter to the wired LAN port of the computer. 2 Connect the LAN cable to the LAN adapter."
    objects = extract_document_objects(text, doc_id="doc-1", page=59)
    procedures = [obj for obj in objects if obj.object_type == ObjectType.PROCEDURE]

    assert len(procedures) >= 2
    assert procedures[0].content.startswith("1 Connect")
    assert procedures[1].content.startswith("2 Connect")


def test_extract_document_objects_creates_grouped_procedure_for_heading_pages() -> None:
    text = (
        "Wake on LAN (WOL) feature "
        "1 Open settings. "
        "2 Right-click Ethernet. "
        "3 Select Configure. "
        "4 Restart the computer."
    )
    objects = extract_document_objects(text, doc_id="doc-1", page=61)
    grouped = [obj for obj in objects if obj.metadata.get("grouped") == "true"]

    assert grouped
    assert "Wake on LAN" in grouped[0].content


def test_extract_document_objects_extracts_table_blocks_with_metadata() -> None:
    text = """
    Table 2. Model comparison
    Model    Recall    Latency
    MARE     0.81      120ms
    BM25     0.69      90ms
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=7)
    tables = [obj for obj in objects if obj.object_type == ObjectType.TABLE]

    assert len(tables) == 1
    assert "Model comparison" in tables[0].content
    assert tables[0].metadata["label"] == "Table 2"
    assert int(tables[0].metadata["columns_estimate"]) >= 3
    assert tables[0].metadata["line_start"] == "1"
    assert tables[0].metadata["line_end"] == "4"
    assert tables[0].metadata["line_total"] == "4"


def test_extract_document_objects_extracts_figure_caption_blocks_with_metadata() -> None:
    text = """
    Figure 3. Retrieval pipeline overview
    Dashed arrows show reranking between stages.
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=4)
    figures = [obj for obj in objects if obj.object_type == ObjectType.FIGURE]

    assert len(figures) == 1
    assert "Dashed arrows" in figures[0].content
    assert figures[0].metadata["label"] == "Figure 3"
    assert figures[0].metadata["line_start"] == "1"
    assert figures[0].metadata["line_end"] == "2"


def test_extract_document_objects_extracts_line_aware_sections() -> None:
    text = """
    Introduction
    We propose a retrieval system for PDF evidence.
    It returns page, snippet, and highlight proof.
    Results
    The built-in path performs strongly on manuals.
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=2)
    sections = [obj for obj in objects if obj.object_type == ObjectType.SECTION]

    assert sections
    assert sections[0].metadata["line_start"] == "1"
    assert sections[0].metadata["line_end"] == "3"


def test_extract_document_objects_extracts_markdown_heading_sections() -> None:
    text = """
    # Setup
    Connect the adapter to the laptop.
    Restart the computer.

    ## Verification
    Confirm the charging light is on.
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=1)
    sections = [obj for obj in objects if obj.object_type == ObjectType.SECTION]

    assert sections
    assert sections[0].metadata["heading"] == "Setup"
    assert sections[0].metadata["line_start"] == "1"
    assert sections[0].metadata["line_end"] == "3"


def test_extract_document_objects_extracts_markdown_list_steps_with_heading_and_lines() -> None:
    text = """
    # Setup
    1. Connect the adapter.
    2. Restart the computer.
    - Confirm the charging light is on.
    """
    objects = extract_document_objects(text, doc_id="doc-1", page=1)
    procedures = [obj for obj in objects if obj.object_type == ObjectType.PROCEDURE]

    assert len(procedures) >= 3
    assert procedures[0].metadata["heading"] == "Setup"
    assert procedures[0].metadata["line_start"] == "2"
    assert procedures[1].metadata["line_start"] == "3"
