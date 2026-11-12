import numpy as np
import pytest
from football_tracking.tracking import TrackedObject
from football_tracking.tracking.team_classifier import TeamClassifier

def make_tracked_object(track_id: int, class_name: str = "player") -> TrackedObject:
    return TrackedObject(track_id=track_id, bbox=(10, 10, 60, 110), confidence=0.9, class_id=2, class_name=class_name)

def make_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((120, 80, 3), dtype=np.uint8)
    frame[25:65, 20:50] = color
    return frame

def test_classifier_initially_returns_no_assignments():
    classifier = TeamClassifier()
    frame = make_frame((0, 0, 255))
    assignments = classifier.classify(frame, [make_tracked_object(1)])
    assert assignments == {}

def test_classifier_groups_players_into_two_teams():
    classifier = TeamClassifier(minimum_samples=4, samples_per_player=2)
    red_frame = make_frame((0, 0, 255))
    blue_frame = make_frame((255, 0, 0))
    players = [make_tracked_object(1), make_tracked_object(2), make_tracked_object(3), make_tracked_object(4)]
    classifier.classify(red_frame, [players[0], players[1]])
    classifier.classify(blue_frame, [players[2], players[3]])
    classifier.classify(red_frame, [players[0], players[1]])
    assignments = classifier.classify(blue_frame, [players[2], players[3]])
    assert len(assignments) == 4
    assert assignments[1] == assignments[2]
    assert assignments[3] == assignments[4]
    assert assignments[1] != assignments[3]

def test_classifier_ignores_goalkeepers_and_referees():
    classifier = TeamClassifier()
    frame = make_frame((0, 0, 255))
    assignments = classifier.classify(frame, [make_tracked_object(1, "goalkeeper"), make_tracked_object(2, "referee")])
    assert assignments == {}

def test_classifier_keeps_existing_assignments():
    classifier = TeamClassifier(minimum_samples=2, samples_per_player=2)
    red_frame = make_frame((0, 0, 255))
    blue_frame = make_frame((255, 0, 0))
    players = [make_tracked_object(1), make_tracked_object(2)]
    classifier.classify(red_frame, players)
    assignments = classifier.classify(red_frame, players)
    original_assignment = assignments.copy()
    classifier.classify(blue_frame, players)
    assert classifier.classify(blue_frame, players) == original_assignment

def test_classifier_reset_clears_state():
    classifier = TeamClassifier()
    classifier.classify(make_frame((0, 0, 255)), [make_tracked_object(1)])
    classifier.reset()
    assert classifier.classify(make_frame((0, 0, 255)), [make_tracked_object(1)]) == {}

def test_classifier_rejects_invalid_frame():
    classifier = TeamClassifier()

    with pytest.raises(ValueError):
        classifier.classify(np.zeros((120, 80), dtype=np.uint8), [])

def test_classifier_rejects_invalid_number_of_teams():
    with pytest.raises(ValueError):
        TeamClassifier(number_of_teams=3)
