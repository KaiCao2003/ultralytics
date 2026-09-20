# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import cv2
import numpy as np
import pytest

from headplate_access import AccessGate
from headplate_workflow import (
    build_dataset,
    configure_project,
    heading_degrees,
    infer_round,
    prepare_round1,
    resolve_under,
    select_review_frames,
    train_round,
)


def make_video(path: Path, frames: int = 12) -> None:
    """Write a small deterministic video fixture."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = 255 * (index % 2) * np.ones((48, 64, 3), dtype="uint8")
        writer.write(image)
    writer.release()


def annotated_tasks(import_path: Path) -> list[dict]:
    """Add complete front/back annotations to generated import tasks."""
    tasks = json.loads(import_path.read_text())
    for task in tasks:
        task["annotations"] = [
            {
                "result": [
                    {
                        "type": "keypointlabels",
                        "value": {"x": 40, "y": 30, "keypointlabels": ["front"]},
                    },
                    {
                        "type": "keypointlabels",
                        "value": {"x": 60, "y": 70, "keypointlabels": ["back"]},
                    },
                ]
            }
        ]
    return tasks


def test_prepare_round1_and_build_dataset(tmp_path, monkeypatch):
    root = tmp_path / "senzailab"
    project = root / "cohort-a" / "mouse-01"
    project.mkdir(parents=True)
    video = project / "session.avi"
    make_video(video)
    monkeypatch.setenv("YOLO_WORKFLOW_ROOT", str(root))

    configure_project(project, [video], {"round1_frames": 4, "val_fraction": 0.25})
    label_dir = prepare_round1(project)
    assert len(list((label_dir / "frames").glob("*.jpg"))) == 4
    tasks = json.loads((label_dir / "label_studio_import.json").read_text())
    assert len(tasks) == 4
    assert all(task["data"]["image"].startswith("/data/local-files/?d=cohort-a/mouse-01/") for task in tasks)

    annotations = project / "round1-export.json"
    annotations.write_text(json.dumps(annotated_tasks(label_dir / "label_studio_import.json")))
    dataset, summary = build_dataset(project, 1, annotations)
    assert summary == {"valid": 4, "incomplete": 0, "unmatched": 0}
    assert len(list((dataset / "images" / "train").glob("*.jpg"))) == 3
    assert len(list((dataset / "images" / "val").glob("*.jpg"))) == 1
    assert all(len(path.read_text().split()) == 9 for path in dataset.glob("labels/*/*.txt"))


def test_resolve_under_rejects_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_under(root, "../outside", create=True)


@pytest.mark.parametrize(
    ("front", "back", "expected"),
    [((0, 0), (0, 1), 0), ((0, 0), (1, 0), 90), ((0, 1), (0, 0), 180), ((1, 0), (0, 0), 270)],
)
def test_heading_convention(front, back, expected):
    assert heading_degrees(front, back) == expected


def test_review_selection_uses_all_three_reasons():
    rows = [
        {
            "frame": str(index),
            "hd_deg": str((index * 17) % 360),
            "det_conf": "nan" if index < 2 else str(index / 20),
        }
        for index in range(20)
    ]
    selected = select_review_frames(rows, 10, 42)
    assert len(selected) == 10
    assert {reason for _, reason in selected} == {"low_confidence", "heading_jump", "random"}


def test_infer_round_tracks_identity_and_holds_missing_pose(tmp_path, monkeypatch):
    root = tmp_path / "senzailab"
    project = root / "mouse-01"
    project.mkdir(parents=True)
    video = project / "session.avi"
    make_video(video, frames=4)
    monkeypatch.setenv("YOLO_WORKFLOW_ROOT", str(root))
    configure_project(project, [video], {"device": "cpu", "imgsz": 64})
    model_path = project / "model.pt"
    model_path.touch()

    class ArrayValue:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def __getitem__(self, index):
            return ArrayValue(self.values[index])

        def int(self):
            return ArrayValue(self.values.astype(int))

        def numpy(self):
            return self.values

    class Collection(SimpleNamespace):
        def __len__(self):
            return len(self.conf.values if hasattr(self, "conf") else self.xy.values)

    def result(ids=None, confidences=None, points=None):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        if ids is None:
            return SimpleNamespace(boxes=None, keypoints=None, orig_img=image)
        boxes = Collection(
            conf=ArrayValue(confidences),
            id=ArrayValue(ids),
            xyxy=ArrayValue([[4, 4, 30, 30] for _ in ids]),
        )
        keypoints = Collection(xy=ArrayValue(points))
        return SimpleNamespace(boxes=boxes, keypoints=keypoints, orig_img=image)

    tracked_results = [
        result(),
        result([3, 7], [0.8, 0.9], [[[18, 10], [8, 10]], [[20, 10], [10, 10]]]),
        result([7, 8], [0.55, 0.99], [[[22, 10], [12, 10]], [[50, 30], [40, 30]]]),
        result(),
    ]

    class FakeYOLO:
        def __init__(self, _model):
            pass

        def track(self, **_settings):
            return tracked_results

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    artifact = infer_round(project, 1, model_path)[0]

    with (project / artifact["csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["center_x"] == "" and math.isnan(float(rows[0]["det_conf"]))
    assert rows[1]["track_id"] == "7" and float(rows[1]["center_x"]) == 15
    assert rows[2]["track_id"] == "7" and float(rows[2]["center_x"]) == 17
    assert rows[3]["center_x"] == rows[2]["center_x"] and math.isnan(float(rows[3]["det_conf"]))

    with (project / artifact["position_csv"]).open(newline="") as handle:
        position_rows = list(csv.DictReader(handle))
    assert position_rows[3]["center_x"] == position_rows[2]["center_x"]
    hd = json.loads((project / artifact["json"]).read_text())["hp4"]
    assert hd["frames"] == [0, 1, 2, 3]
    assert hd["hd"] == [None, 270.0, 270.0, 270.0]
    assert artifact["detected_frames"] == 2 and artifact["held_frames"] == 1


def test_train_round_uses_trainer_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "senzailab"
    project = root / "mouse-01"
    project.mkdir(parents=True)
    video = project / "session.avi"
    video.touch()
    monkeypatch.setenv("YOLO_WORKFLOW_ROOT", str(root))
    configure_project(project, [video], {})
    dataset = project / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").touch()
    trainer_best = project / "trainer-selected-best.pt"

    class FakeYOLO:
        def __init__(self, _model):
            self.trainer = SimpleNamespace(best=trainer_best)

        def add_callback(self, _name, _callback):
            pass

        def train(self, **_settings):
            trainer_best.write_bytes(b"trained")

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    stable = train_round(project, 1, dataset)
    assert stable.read_bytes() == b"trained"


def test_access_gate_sessions(tmp_path):
    gate = AccessGate("Answer", "1", tmp_path / "sessions.sqlite3")
    assert gate.validate_answer(" answer ")
    assert not gate.validate_answer("wrong")
    token, csrf = gate.issue_session()
    assert gate.validate_session(token)
    assert gate.validate_csrf(token, csrf)
    assert not gate.validate_csrf(token, "wrong")
    gate.revoke(token)
    assert not gate.validate_session(token)


def test_web_nested_folder_creation_and_project_initialization(tmp_path, monkeypatch):
    root = tmp_path / "senzailab"
    project = root / "cohort-a" / "mouse-01"
    project.mkdir(parents=True)
    (project / "session.avi").touch()
    (root / ".hidden").mkdir()
    (root / "headplate-yolo").mkdir()
    monkeypatch.setenv("YOLO_WORKFLOW_ROOT", str(root))
    monkeypatch.setenv("YOLO_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("YOLO_BASE_PATH", "")
    monkeypatch.setenv("MOUSELINE_LOGIN_ANSWER", "answer")
    monkeypatch.setenv("MOUSELINE_AUTH_GENERATION", "test")

    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    import headplate_web

    headplate_web = importlib.reload(headplate_web)
    with TestClient(headplate_web.create_app()) as client:
        assert client.get("/", follow_redirects=False).status_code == 303
        assert client.get("/login").status_code == 200
        response = client.post(
            "/login",
            content=urlencode({"answer": "answer", "next": "/"}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        folders = client.get("/api/folders")
        assert folders.status_code == 200
        assert folders.json()["current"] == "."
        assert folders.json()["parent"] is None
        assert folders.json()["folders"] == [{"name": "cohort-a", "path": "cohort-a"}]
        nested = client.get("/api/folders", params={"folder": "cohort-a"})
        assert nested.status_code == 200
        assert nested.json()["parent"] == "."
        assert nested.json()["folders"] == [{"name": "mouse-01", "path": "cohort-a/mouse-01"}]
        csrf = client.cookies.get("headplate_yolo_csrf")
        security_headers = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        created = client.post(
            "/api/folders",
            json={"parent": "cohort-a", "name": "mouse-02"},
            headers=security_headers,
        )
        assert created.status_code == 201
        assert created.json()["current"] == "cohort-a/mouse-02"
        assert (root / "cohort-a" / "mouse-02").is_dir()
        deeper = client.post(
            "/api/folders",
            json={"parent": "cohort-a/mouse-02", "name": "session-a"},
            headers=security_headers,
        )
        assert deeper.status_code == 201
        assert deeper.json()["current"] == "cohort-a/mouse-02/session-a"
        duplicate = client.post(
            "/api/folders",
            json={"parent": "cohort-a", "name": "mouse-02"},
            headers=security_headers,
        )
        assert duplicate.status_code == 409
        for invalid_name in (" ", "..", "../outside", "child/folder", ".hidden", "headplate-yolo"):
            rejected = client.post(
                "/api/folders",
                json={"parent": "cohort-a", "name": invalid_name},
                headers=security_headers,
            )
            assert rejected.status_code == 400
        assert client.get("/api/folders", params={"folder": str(root)}).status_code == 400
        assert client.get("/api/folders", params={"folder": "../outside"}).status_code == 400
        assert client.get("/api/project", params={"project": "."}).status_code == 400
        assert client.get("/api/project", params={"project": "cohort-a/mouse-01/session.avi"}).status_code == 400
        invalid_parent = client.post(
            "/api/folders",
            json={"parent": "cohort-a/mouse-01/session.avi", "name": "child"},
            headers=security_headers,
        )
        assert invalid_parent.status_code == 400
        initialized = client.post(
            "/api/initialize",
            json={"project": "cohort-a/mouse-01", "videos": ["session.avi"], "config": {}},
            headers=security_headers,
        )
        assert initialized.status_code == 200
        assert initialized.json()["state"]["stage"] == "CONFIGURED"
