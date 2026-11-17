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

"""Best-effort ordering of the API user's personal W&B project and run views.

W&B's view API is internal and varies between server versions. Keep the raw
specification so editing ordering does not discard filters, charts or settings.
The project and run pages store separate views, even for the same workspace name.
"""

import copy
import json
import logging
import os
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import requests

from northrl.utils.metric_catalog import WANDB_SECTION_ORDER

logger = logging.getLogger(__name__)
_HIDDEN = "Hidden Panels"
_SECTION_NAMES = tuple(f"{i:02d}_{name}" for i, name in enumerate(WANDB_SECTION_ORDER, 1))
_VIEWS_QUERY = """
query NorthRLWorkspace($entity: String!, $project: String!, $viewType: String!,
                     $viewName: String!, $userName: String!) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: $viewType, viewName: $viewName, userName: $userName) {
      edges { node { id name displayName type spec } }
    }
  }
}
"""
_SAVE_MUTATION = """
mutation NorthRLOrderWorkspace($input: UpsertViewInput!) {
  upsertView(input: $input) { view { id name } }
}
"""


def _stable_id(scope: str, name: str) -> str:
    # Reuse existing workspace sections instead of creating new IDs after the rename.
    return uuid5(NAMESPACE_URL, f"nexrl/wandb/{scope}/{name}").hex


def _new_section(scope: str, name: str) -> dict[str, Any]:
    return {
        "__id__": _stable_id(scope, name),
        "name": name,
        "isOpen": name == "01_algorithm",
        "type": "flow",
        "sorted": 2,
        "pinned": False,
        "isPanelsAuto": name != _HIDDEN,
        "panels": [],
    }


def _new_spec(view_type: str, scope: str) -> dict[str, Any]:
    if view_type == "run-view":
        return {
            "customExpanded": False,
            "libraryExpanded": True,
            "panels": {
                "views": {"0": {"name": "Panels", "defaults": [], "config": [], "panelRefs": []}},
                "tabs": ["0"],
            },
            "settings": {},
        }
    return {
        "vizExpanded": False,
        "libraryExpanded": True,
        "section": {
            "name": "",
            "version": 1,
            "openRunSet": 0,
            "openViz": True,
            "runSets": [
                {
                    "id": _stable_id(scope, "run-set"),
                    "name": "Run set",
                    "enabled": True,
                    "runNameTruncationType": "Middle",
                    "runFeed": {
                        "version": 2,
                        "columnVisible": {"run:name": False},
                        "columnPinned": {},
                        "columnWidths": {},
                        "columnOrder": [],
                        "pageSize": 20,
                        "onlyShowSelected": False,
                    },
                    "search": {"query": ""},
                    "searchHistory": [],
                    "filters": {"op": "OR", "filters": [{"op": "AND", "filters": []}]},
                    "grouping": [],
                    "sort": {
                        "keys": [
                            {"key": {"section": "run", "name": "createdAt"}, "ascending": False}
                        ]
                    },
                    # W&B's default select-all state includes future runs too.
                    # root=0 with an empty tree would produce a blank workspace.
                    "selections": {"root": 1, "bounds": [], "tree": []},
                    "expandedRowAddresses": [],
                }
            ],
            "panelBankSectionConfig": {
                **_new_section(scope, "Report Panels"),
                "type": "grid",
                "sorted": 0,
            },
            "customRunColors": {},
        },
    }


def _panel_key(panel: dict[str, Any]) -> str:
    config = panel.get("config", {})
    # Match W&B's title/media/metric/key precedence for persisted panels.
    if "chartTitle" in config:
        return str(config["chartTitle"] or "").casefold()
    return str(
        next(iter(config.get("mediaKeys") or []), "")
        or next(iter(config.get("metrics") or []), "")
        or panel.get("key", "")
    ).casefold()


def ordered_workspace_spec(spec: dict[str, Any], view_type: str, scope: str) -> dict[str, Any]:
    """Return a copy with reserved metric sections and alphabetical flow panels.

    Explicit grid geometry, chart definitions, filters and hidden panels survive.
    Visible sections are unpinned so their stored order is also their display order.
    """
    if view_type not in ("project-view", "run-view") or not isinstance(spec, dict):
        raise ValueError("Unsupported workspace specification")
    result = copy.deepcopy(spec) if spec else _new_spec(view_type, scope)
    root = result.setdefault("section", {}) if view_type == "project-view" else result
    settings = root.setdefault("workspaceSettings", {})
    settings.update(sortAlphabetically=True, autoOrganizePrefix=1, showEmptySections=False)
    # Respect an existing manual workspace's choice not to generate new charts.
    settings.setdefault(
        "shouldAutoGeneratePanels", root.get("settings", {}).get("shouldAutoGeneratePanels", True)
    )
    bank = root.setdefault("panelBankConfig", {})
    bank["state"] = 1  # Initialized: don't let first render replace the reserved order.
    bank.setdefault("settings", {}).update(sortAlphabetically=True, showEmptySections=False)
    sections = bank.setdefault("sections", [])
    if not isinstance(sections, list) or any(not isinstance(s, dict) for s in sections):
        raise ValueError("Unsupported workspace sections")
    names = {section["name"] for section in sections}
    for name in (*_SECTION_NAMES, "Charts", "System", _HIDDEN):
        if name not in names:
            sections.append(_new_section(scope, name))
    for section in sections:
        if section["name"] == _HIDDEN:
            continue
        section["pinned"] = False
        if section.get("type", "flow") == "flow":
            section["sorted"] = 2  # Alphabetical; also governs future auto-insertions.
            section.setdefault("panels", []).sort(key=_panel_key)
    # W&B treats the final section as the hidden bank, regardless of its name.
    sections.sort(key=lambda s: (s["name"] == _HIDDEN, s["name"].casefold()))
    return result


class _WorkspaceClient:
    def __init__(self, session: requests.Session, base_url: str):
        self.session = session
        self.url = base_url.rstrip("/") + "/graphql"
        self.deadline = time.monotonic() + 20

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Workspace setup time budget exhausted")
        response = self.session.post(
            self.url,
            json={"query": query, "variables": variables},
            timeout=min(5, remaining),
            allow_redirects=False,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("errors") or not isinstance(result.get("data"), dict):
            raise ValueError("Workspace GraphQL operation failed")
        return result["data"]

    def read_view(self, variables: dict[str, Any]) -> dict[str, Any] | None:
        data = self.execute(_VIEWS_QUERY, variables)
        nodes = [edge["node"] for edge in data["project"]["allViews"]["edges"]]
        if len(nodes) > 1 or any(
            n["name"] != variables["viewName"] or n["type"] != variables["viewType"] for n in nodes
        ):
            raise ValueError("Ambiguous personal workspace")
        return nodes[0] if nodes else None

    def order_view(self, variables: dict[str, Any]) -> None:
        scope = "/".join(variables[k] for k in ("entity", "project", "viewName", "viewType"))
        # Re-read before saving and retry one concurrent edit. The API has no CAS;
        # a browser autosave after our last read can still overwrite this update.
        for _ in range(2):
            view = self.read_view(variables)
            spec = json.loads(view["spec"]) if view else {}
            ordered = ordered_workspace_spec(spec, variables["viewType"], scope)
            if view and ordered == spec:
                return
            if self.read_view(variables) != view:
                continue
            payload = {"spec": json.dumps(ordered)}
            if view:
                payload["id"] = view["id"]
            else:
                payload.update(
                    entityName=variables["entity"],
                    projectName=variables["project"],
                    type=variables["viewType"],
                    name=variables["viewName"],
                    displayName=f"{variables['userName']}'s "
                    + ("run workspace" if variables["viewType"] == "run-view" else "workspace"),
                )
            saved = self.execute(_SAVE_MUTATION, {"input": payload})["upsertView"]["view"]
            if not saved or not saved.get("id"):
                raise ValueError("Workspace save was not acknowledged")
            verified = self.read_view(variables)
            if verified:
                saved_spec = json.loads(verified["spec"])
                if ordered_workspace_spec(saved_spec, variables["viewType"], scope) == saved_spec:
                    logger.info("W&B %s ordering configured for %s", variables["viewType"], scope)
                    return
        raise RuntimeError("Workspace changed during ordering; retry at next startup")


def configure_wandb_workspaces(run: Any) -> None:
    """Configure only this run's project and credential owner's personal views.

    Layout failures must never stop training. No run data or history is changed.
    """
    try:
        settings = run.settings
        if getattr(run, "disabled", False) or settings.mode in ("offline", "dryrun", "disabled"):
            return
        with requests.Session() as session:
            # Tracking already authenticates W&B using these credentials.
            session.auth = ("api", os.environ["WANDB_KEY"])
            client = _WorkspaceClient(session, settings.base_url)
            username = client.execute("query NorthRLWorkspaceUser { viewer { username } }", {})[
                "viewer"
            ]["username"]
            if not isinstance(username, str) or not username:
                raise ValueError("Workspace user is unavailable")
            for view_type in ("project-view", "run-view"):
                try:
                    client.order_view(
                        {
                            "entity": run.entity,
                            "project": run.project,
                            "viewName": f"nw-nwuser{username}-w",
                            "userName": username,
                            "viewType": view_type,
                        }
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "W&B %s ordering unavailable (%s); training continues. "
                        "Check workspace API support and write access.",
                        view_type,
                        type(exc).__name__,
                    )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Do not print transport error bodies/URLs that might contain credentials.
        logger.warning(
            "W&B workspace ordering unavailable (%s); training continues", type(exc).__name__
        )
