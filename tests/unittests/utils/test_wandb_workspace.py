# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace API contract and preservation tests; never contact a live W&B server."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from northrl.utils import wandb_workspace as workspace


def _root(spec, view_type):
    return spec["section"] if view_type == "project-view" else spec


def _panel(name):
    return {"__id__": name, "viewType": "Run History Line Plot", "config": {"metrics": [name]}}


def _existing_spec(view_type):
    root = {
        "workspaceSettings": {"linePlot": {"smoothingWeight": 0.3}},
        "panelBankConfig": {
            "state": 1,
            "settings": {"unrecognized": "preserve me"},
            "sections": [
                {
                    "__id__": "weights",
                    "name": "09_timing_weights",
                    "type": "flow",
                    "panels": [],
                    "isOpen": False,
                },
                {
                    "__id__": "algorithm",
                    "name": "01_algorithm",
                    "type": "flow",
                    "isOpen": True,
                    "pinned": True,
                    "sorted": 0,
                    "panels": [
                        _panel(f"01_algorithm/{n:02d}_metric") for n in (42, 28, 10, 8, 16, 5)
                    ],
                },
                {"name": "Hidden Panels", "panels": [_panel("training/train_step")], "sorted": 0},
                {
                    "name": "Custom grid",
                    "type": "grid",
                    "sorted": 1,
                    "panels": [_panel("z"), _panel("a")],
                    "gridConfig": {"layout": [{"i": "z", "x": 0, "y": 0}]},
                },
            ],
        },
        "runSets": [{"filters": {"search": "special"}, "selections": ["my-run"]}],
    }
    spec = {"section": root} if view_type == "project-view" else root
    spec["futureServerField"] = {"keep": [1, 2, 3]}
    return spec


@pytest.mark.parametrize("view_type", ["project-view", "run-view"])
def test_ordering_preserves_charts_settings_filters_and_hidden_bank(view_type):
    original = _existing_spec(view_type)
    snapshot = copy.deepcopy(original)
    result = workspace.ordered_workspace_spec(original, view_type, "scope")
    root = _root(result, view_type)
    old = _root(original, view_type)
    sections = root["panelBankConfig"]["sections"]
    assert [s["name"] for s in sections[:25]] == list(workspace._SECTION_NAMES)
    assert sections[-1] == old["panelBankConfig"]["sections"][2]
    assert root["runSets"] == old["runSets"]
    assert result["futureServerField"] == original["futureServerField"]
    assert root["workspaceSettings"]["linePlot"] == {"smoothingWeight": 0.3}
    assert root["workspaceSettings"]["sortAlphabetically"] is True
    assert root["workspaceSettings"]["autoOrganizePrefix"] == 1
    assert root["panelBankConfig"]["settings"]["unrecognized"] == "preserve me"
    algorithm = sections[0]
    assert algorithm["__id__"] == "algorithm"
    assert algorithm["isOpen"] is True
    assert algorithm["pinned"] is False
    assert algorithm["sorted"] == 2  # Future auto-panels must also insert alphabetically.
    assert algorithm["panels"] == [
        _panel(f"01_algorithm/{n:02d}_metric") for n in (5, 8, 10, 16, 28, 42)
    ]
    grid = next(s for s in sections if s["name"] == "Custom grid")
    assert grid == {**old["panelBankConfig"]["sections"][3], "pinned": False}
    assert original == snapshot
    assert workspace.ordered_workspace_spec(result, view_type, "scope") == result


@pytest.mark.parametrize("view_type", ["project-view", "run-view"])
def test_new_views_reserve_late_metrics_and_have_stable_ids(view_type):
    first = workspace.ordered_workspace_spec({}, view_type, "team/project/user/view")
    second = workspace.ordered_workspace_spec({}, view_type, "team/project/user/view")
    assert first == second
    root = _root(first, view_type)
    sections = root["panelBankConfig"]["sections"]
    # Existing workspace sections keep their IDs when the package is renamed.
    assert sections[0]["__id__"] == "c36b0f172a37586db071cdc7c235a9d5"
    assert len({s["__id__"] for s in sections}) == len(sections)
    assert all(s["isPanelsAuto"] and s["sorted"] == 2 for s in sections[:-1])
    assert root["workspaceSettings"]["showEmptySections"] is False
    assert root["workspaceSettings"]["shouldAutoGeneratePanels"] is True
    assert root["panelBankConfig"]["state"] == 1
    assert sections[-1]["name"] == "Hidden Panels"
    assert sections[-1]["isPanelsAuto"] is False
    if view_type == "project-view":
        assert root["runSets"][0]["selections"] == {"root": 1, "bounds": [], "tree": []}


@pytest.mark.parametrize("settings_key", ["workspaceSettings", "settings"])
def test_manual_chart_generation_setting_is_preserved(settings_key):
    result = workspace.ordered_workspace_spec(
        {settings_key: {"shouldAutoGeneratePanels": False}}, "run-view", "scope"
    )
    assert result["workspaceSettings"]["shouldAutoGeneratePanels"] is False


def test_panels_sort_by_displayed_title_then_media_then_metric():
    panels = [
        {"config": {"chartTitle": "c", "metrics": ["00"]}},
        {"config": {"mediaKeys": ["B"], "metrics": ["00"]}},
        {"config": {"metrics": ["a"]}},
        {"config": {"chartTitle": "", "metrics": ["z"]}},
    ]
    spec = {"panelBankConfig": {"sections": [{"name": "Charts", "panels": panels}]}}
    result = workspace.ordered_workspace_spec(spec, "run-view", "scope")
    charts = next(s for s in result["panelBankConfig"]["sections"] if s["name"] == "Charts")
    assert charts["panels"] == [panels[3], panels[2], panels[1], panels[0]]


class _Server:
    """Minimal allViews/upsertView contract with independent project and run views."""

    def __init__(self, existing=True):
        self.views = {}
        self.posts = []
        self.writes = []
        self.before_read = lambda variables: None
        self.error_type = None
        if existing:
            for view_type in ("project-view", "run-view"):
                self.views[view_type] = {
                    "id": view_type + "-id",
                    "name": "nw-nwuseralice-w",
                    "type": view_type,
                    "displayName": "My custom title",
                    "spec": json.dumps(_existing_spec(view_type)),
                }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, *, json, timeout, allow_redirects):
        import json as json_module

        self.posts.append(copy.deepcopy(json))
        assert url == "https://wandb.test/graphql"
        assert 0 < timeout <= 5
        assert allow_redirects is False
        variables = json["variables"]
        if "NorthRLWorkspaceUser" in json["query"]:
            data = {"viewer": {"username": "alice"}}
        elif "NorthRLOrderWorkspace" in json["query"]:
            payload = variables["input"]
            self.writes.append(copy.deepcopy(payload))
            if "id" in payload:
                view = next(v for v in self.views.values() if v["id"] == payload["id"])
                view["spec"] = payload["spec"]
            else:
                view = {**payload, "id": payload["type"] + "-id"}
                self.views[payload["type"]] = view
            assert isinstance(json_module.loads(view["spec"]), dict)
            data = {"upsertView": {"view": {"id": view["id"], "name": view["name"]}}}
        else:
            assert variables["entity"] == "team"  # Entity is not the credential's username.
            assert variables["project"] == "project"
            assert variables["userName"] == "alice"
            assert variables["viewName"] == "nw-nwuseralice-w"
            if variables["viewType"] == self.error_type:
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"errors": [{}]})
            self.before_read(variables)
            view = self.views.get(variables["viewType"])
            data = {"project": {"allViews": {"edges": [{"node": view}] if view else []}}}
        snapshot = copy.deepcopy(data)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": snapshot})


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setenv("WANDB_KEY", "secret-test-key")
    return SimpleNamespace(
        settings=SimpleNamespace(mode="online", base_url="https://wandb.test/"),
        entity="team",
        project="project",
        disabled=False,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_startup_creates_or_updates_both_views_and_repeat_is_read_only(
    monkeypatch, online, existing
):
    server = _Server(existing)
    monkeypatch.setattr(workspace.requests, "Session", lambda: server)
    workspace.configure_wandb_workspaces(online)
    assert server.auth == ("api", "secret-test-key")
    assert len(server.writes) == 2
    for view_type, view in server.views.items():
        spec = json.loads(view["spec"])
        assert workspace.ordered_workspace_spec(spec, view_type, "any") == spec
        assert view["name"] == "nw-nwuseralice-w"
        if existing:
            assert view["displayName"] == "My custom title"
    if existing:
        assert all(set(w) == {"id", "spec"} for w in server.writes)
    workspace.configure_wandb_workspaces(online)
    assert len(server.writes) == 2


def test_concurrent_edit_is_merged_on_retry(monkeypatch, online):
    server = _Server()
    reads = 0

    def edit(variables):
        nonlocal reads
        if variables["viewType"] != "project-view":
            return
        reads += 1
        if reads == 2:
            spec = json.loads(server.views["project-view"]["spec"])
            spec["section"]["runSets"] = [{"new": "browser filter"}]
            server.views["project-view"]["spec"] = json.dumps(spec)

    server.before_read = edit
    monkeypatch.setattr(workspace.requests, "Session", lambda: server)
    workspace.configure_wandb_workspaces(online)
    assert len(server.writes) == 2
    assert json.loads(server.views["project-view"]["spec"])["section"]["runSets"] == [
        {"new": "browser filter"}
    ]


@pytest.mark.parametrize(
    "bad_spec",
    [
        "invalid JSON",
        "null",
        '{"section": null}',
        '{"section":{"panelBankConfig":{"sections":{}}}}',
    ],
)
def test_malformed_view_is_not_replaced_and_other_view_still_updates(
    monkeypatch, online, bad_spec, caplog
):
    server = _Server()
    server.views["project-view"]["spec"] = bad_spec
    monkeypatch.setattr(workspace.requests, "Session", lambda: server)
    workspace.configure_wandb_workspaces(online)
    assert server.views["project-view"]["spec"] == bad_spec
    assert [w["id"] for w in server.writes] == ["run-view-id"]
    assert "training continues" in caplog.text


def test_unsupported_view_api_does_not_block_other_view(monkeypatch, online, caplog):
    server = _Server()
    server.error_type = "project-view"
    monkeypatch.setattr(workspace.requests, "Session", lambda: server)
    workspace.configure_wandb_workspaces(online)
    assert [w["id"] for w in server.writes] == ["run-view-id"]
    assert "project-view ordering unavailable" in caplog.text


@pytest.mark.parametrize("error", [requests.Timeout, requests.HTTPError, ValueError])
def test_transport_failures_do_not_raise_or_expose_credentials(monkeypatch, online, caplog, error):
    server = _Server()
    server.post = Mock(side_effect=error("secret-test-key"))
    monkeypatch.setattr(workspace.requests, "Session", lambda: server)
    workspace.configure_wandb_workspaces(online)
    assert "training continues" in caplog.text
    assert "secret-test-key" not in caplog.text
    assert server.writes == []


@pytest.mark.parametrize("mode", ["offline", "dryrun", "disabled"])
def test_offline_and_disabled_skip_http(monkeypatch, online, mode):
    online.settings.mode = mode
    session = Mock()
    monkeypatch.setattr(workspace.requests, "Session", session)
    workspace.configure_wandb_workspaces(online)
    session.assert_not_called()


def test_request_budget_expires_without_another_request(monkeypatch):
    session = Mock()
    monkeypatch.setattr(workspace.time, "monotonic", lambda: 100)
    client = workspace._WorkspaceClient(session, "https://wandb.test")
    monkeypatch.setattr(workspace.time, "monotonic", lambda: 121)
    with pytest.raises(TimeoutError):
        client.execute("query", {})
    session.post.assert_not_called()
