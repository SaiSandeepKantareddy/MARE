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
