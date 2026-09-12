"""Unit tests for P1 data loader and consolidator."""

from src.data_loader import consolidate_annotator_records, load_deduplicated_dataset
from src.schemas import Report


def test_consolidate_annotator_records():
    # Simulate two annotators assessing the same tweet
    records = [
        {
            "item_id": "MSG_100",
            "source_item_id": "MSG_100",
            "event_id": "TRECIS-CTIT-H-035",
            "event_name": "athensEarthquake2020",
            "event_type": "earthquake",
            "text": "Tremors felt across the capital.",
            "task2_categories": ["Location"],
            "all_categories": ["Location", "News"],
            "priority": "Low",
            "is_variant": False,
            "transform_type": None
        },
        {
            "item_id": "MSG_100",
            "source_item_id": "MSG_100",
            "event_id": "TRECIS-CTIT-H-035",
            "event_name": "athensEarthquake2020",
            "event_type": "earthquake",
            "text": "Tremors felt across the capital.",
            "task2_categories": ["EmergingThreats"],
            "all_categories": ["EmergingThreats", "Location"],
            "priority": "High",
            "is_variant": False,
            "transform_type": None
        }
    ]

    report, meta = consolidate_annotator_records(records)

    assert report.id == "MSG_100"
    assert report.text == "Tremors felt across the capital."
    assert meta.item_id == "MSG_100"
    assert meta.event_id == "TRECIS-CTIT-H-035"
    assert meta.priority == "High"  # Maximum urgency selected
    assert meta.priority_score == 4.0
    assert set(meta.task2_categories) == {"Location", "EmergingThreats"}
    assert meta.primary_category == "EmergingThreats"  # Higher actionable priority than Location
    assert meta.num_annotator_assessments == 2
