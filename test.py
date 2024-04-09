import os
import csv
import cv2
import pytest
from detector import drawRectangle, spots

# Test data directory
TEST_DATA_DIR = 'tests/data'

@pytest.fixture
def test_image():
    """Load a test image for the tests"""
    image_path = os.path.join(TEST_DATA_DIR, 'test_image.jpg')
    print(f"Loading test image from: {image_path}")
    return cv2.imread(image_path)

@pytest.fixture
def test_rois():
    """Load test ROIs from a CSV file"""
    rois_path = os.path.join(TEST_DATA_DIR, 'test_rois.csv')
    print(f"Loading test ROIs from: {rois_path}")
    with open(rois_path, 'r', newline='') as inf:
        csvr = csv.reader(inf)
        rois = list(csvr)
    return [[int(float(j)) for j in i] for i in rois]

def test_drawRectangle_available(test_image, test_rois):
    """Test that the drawRectangle function correctly identifies an available spot"""
    print("Running test_drawRectangle_available")
    spots.loc = 0
    a, b, c, d = test_rois[0]
    drawRectangle(test_image, a, b, c, d)
    assert spots.loc == 1

def test_drawRectangle_occupied(test_image, test_rois):
    """Test that the drawRectangle function correctly identifies an occupied spot"""
    print("Running test_drawRectangle_occupied")
    spots.loc = 0
    a, b, c, d = test_rois[1]
    drawRectangle(test_image, a, b, c, d)
    assert spots.loc == 0

def test_spots_counter(test_image, test_rois):
    """Test that the spots counter is correctly updated"""
    print("Running test_spots_counter")
    spots.loc = 0
    for i in range(len(test_rois)):
        a, b, c, d = test_rois[i]
        drawRectangle(test_image, a, b, c, d)
    assert spots.loc == 1