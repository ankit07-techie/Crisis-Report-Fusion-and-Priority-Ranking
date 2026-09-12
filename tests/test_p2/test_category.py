"""Unit tests for official P2 Category Prediction (src/category.py)."""

import pytest
from src.category import predict_category, CATEGORIES

def test_predict_category_official_taxonomy():
    assert predict_category("Trapped under rubble, rooftop rescue needed immediately!") == "SearchAndRescue"
    assert predict_category("Casualties with severe bleeding and fractures after vehicle crash, need doctor.") == "GoodsServices"
    assert predict_category("Toxic chemical gas odor leaking from industrial storage tank.") == "EmergingThreats"
    assert predict_category("Route 9 main suspension overpass at GPS 34.05,-118.25.") == "Location"
    assert predict_category("Mandatory evacuation order issued for coastal zones by buses.") == "MovePeople"
    assert predict_category("I can see thick black smoke rising from my window on Oak Street.") == "FirstPartyObservation"
    assert predict_category("Is the bridge on Route 9 still open to emergency traffic or completely blocked?") == "InformationWanted"
    assert predict_category("Red Cross mobile medical van operating free hotspot Wi-Fi.") == "ServiceAvailable"
    assert predict_category("Volunteers needed for sandbagging along river levee.") == "Volunteer"
    assert predict_category("Video showing massive flood surge: http://example.com/video") == "MultimediaShare"
    assert predict_category("Flash flood warning upgraded to immediate dam failure alert.") == "NewSubEvent"

def test_predict_category_output_schema():
    cat = predict_category("Flood waters rising rapidly on Main Street.")
    assert isinstance(cat, str)
    assert cat in CATEGORIES

def test_predict_category_edge_cases():
    assert predict_category("") == "FirstPartyObservation"
    assert predict_category("   ") == "FirstPartyObservation"
    assert predict_category("Hello world") in CATEGORIES
