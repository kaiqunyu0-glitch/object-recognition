"""Multi-object detection on phone photos.

One photo can contain many objects. This package turns labeled boxes into a
YOLO dataset, fine-tunes a pretrained detector, and scores only the held-out
test photos.

The single-label classifier still lives in ``src/engine.py``. Detection is
the path the README teaches first.
"""
