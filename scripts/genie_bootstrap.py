#!/usr/bin/env python3
"""Create or update the Flood Underwriter Genie Space from the code spec.

DABs do not yet have a `genie_spaces` resource type, so we keep the space
configuration in `resources/genie/flood_underwriter.json` (reviewable in git,
no UI clicks) and use this script to push it into a workspace. Re-running is
idempotent: if a space with the same `display_name` already exists in the
workspace it is updated in place, otherwise a new one is created.

Usage:
    python scripts/genie_bootstrap.py \
        --profile        <databricks-cli-profile> \
        --catalog        flood_demo \
        --schema         montreal \
        --warehouse-id   <sql-warehouse-id>

Prints the resulting `space_id` to stdout so it can be fed straight back into
the DABs deploy as `--var=genie_space_id=...`:

    GENIE_SPACE_ID=$(python scripts/genie_bootstrap.py --profile $P ...)
    databricks bundle deploy -t dev --var=genie_space_id=$GENIE_SPACE_ID

This script intentionally uses only the Databricks SDK (no MCP tools) so it
runs in CI and in a plain dev environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


def _load_spec(spec_path: Path, catalog: str, schema: str) -> dict[str, Any]:
    """Read the JSON spec and substitute `{catalog}` / `{schema}` placeholders.

    Leaves `{aoi}`, `{scenario}`, `{n}` untouched - those are filled at chat
    time by Genie from the user prompt.
    """
    spec = json.loads(spec_path.read_text())

    def sub(s: str) -> str:
        return s.replace("{catalog}", catalog).replace("{schema}", schema)

    tables = [sub(t) for t in spec["table_identifiers_template"]]

    return {
        "display_name":      spec["display_name"],
        "description":       spec["description"],
        "table_identifiers": tables,
        "instructions":      sub(spec["instructions"]),
        "general_instructions": sub(spec["general_instructions"]),
        "sample_questions":  spec["sample_questions"],
        "certified_questions": [
            {"title": cq["title"],
             "question": cq["question"],
             "sql_template": sub(cq["sql_template"])}
            for cq in spec["certified_questions"]
        ],
    }


def _find_existing(client: Any, display_name: str) -> str | None:
    """Walk Genie spaces in the workspace and return the id matching name, or
    None. The SDK's `list_spaces` returns a typed response wrapper whose
    `.spaces` attribute holds the page, plus an optional `.next_page_token`."""
    token: str | None = None
    while True:
        resp = client.genie.list_spaces(page_token=token) if token \
            else client.genie.list_spaces()
        for s in (getattr(resp, "spaces", None) or []):
            title = getattr(s, "title", None) or getattr(s, "display_name", None)
            if title == display_name:
                return getattr(s, "space_id", None) or getattr(s, "id", None)
        token = getattr(resp, "next_page_token", None)
        if not token:
            return None


def _serialized_space(payload: dict[str, Any]) -> str:
    """Build the `serialized_space` JSON the create/update endpoint expects.

    Schema (proto `databricks.datarooms.export.GenieSpaceExport`, version 2):

        {
          "version": 2,
          "config": {
            "sample_questions": [{"id": "<32-hex uuid>", "question": ["..."]}]
          },
          "data_sources": {
            "tables": [{"identifier": "catalog.schema.table"}]  # MUST be sorted
          },
          "instructions": {
            "text_instructions": [
              {"id": "<32-hex uuid>", "content": ["..."]}
            ],
            "example_question_sqls": [
              {"id": "<32-hex uuid>", "question": ["..."], "sql": ["..."]}
            ]
          }
        }

    Notes on the schema after probing the proto:
    * `tables` must be sorted alphabetically by identifier.
    * Free-text fields are arrays of strings throughout (the proto allows
      multiple phrasings / SQL variants per entry); we ship the single
      canonical form.
    * Every `id` must be a lowercase 32-hex UUID (no hyphens) and is stable
      across runs - we derive it from the entry's text so re-runs update in
      place instead of duplicating.
    * The persona prompt and the terse-answer style rules go in
      `instructions.text_instructions` as two separate entries so the SA can
      edit tone in the UI without touching the data dictionary, and the
      sample questions go in `config.sample_questions` (note: `config`, not
      `instructions` - they are chips on the empty chat, not model context).
    """
    sorted_tables = sorted(payload["table_identifiers"])

    def _id(seed: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex

    example_question_sqls = []
    for cq in payload["certified_questions"]:
        title_seed = cq.get("title") or cq["question"]
        example_question_sqls.append({
            "id": _id(title_seed),
            "question": [cq["question"]],
            "sql": [cq["sql_template"]],
        })
    # Proto requires example_question_sqls sorted by id (lex order on hex).
    example_question_sqls.sort(key=lambda e: e["id"])

    # The proto caps text_instructions at a single entry, so the data
    # dictionary (`instructions`) and the answer-style rules
    # (`general_instructions`) are concatenated into one block. They stay
    # separate in the JSON spec because they are edited independently.
    text_instructions = [{
        "id": _id(f"{payload['display_name']}::instructions"),
        "content": [payload["instructions"], "\n\n", payload["general_instructions"]],
    }]

    sample_questions = sorted(
        ({"id": _id(q), "question": [q]} for q in payload["sample_questions"]),
        key=lambda e: e["id"],
    )

    spec = {
        "version": 2,
        "config": {
            "sample_questions": sample_questions,
        },
        "data_sources": {
            "tables": [{"identifier": t} for t in sorted_tables],
        },
        "instructions": {
            "text_instructions": text_instructions,
            "example_question_sqls": example_question_sqls,
        },
    }
    return json.dumps(spec)


def _create_or_update(
    client: Any,
    warehouse_id: str,
    space_id: str | None,
    payload: dict[str, Any],
) -> str:
    """Use the typed SDK methods `genie.create_space` / `genie.update_space`.

    The modern API consolidates the whole space definition (instructions,
    tables, sample + certified questions) into a single `serialized_space`
    YAML string instead of separate sub-resources. We build that string
    here from the JSON spec.
    """
    serialized = _serialized_space(payload)
    if space_id is None:
        resp = client.genie.create_space(
            warehouse_id=warehouse_id,
            serialized_space=serialized,
            title=payload["display_name"],
            description=payload["description"],
        )
        space_id = getattr(resp, "space_id", None) or getattr(resp, "id", None)
        print(f"[genie] created new space {space_id!r}", file=sys.stderr)
    else:
        client.genie.update_space(
            space_id=space_id,
            serialized_space=serialized,
            title=payload["display_name"],
            description=payload["description"],
        )
        print(f"[genie] updated existing space {space_id!r}", file=sys.stderr)

    if not space_id:
        raise RuntimeError("Genie API did not return a space id")
    print(f"[genie] applied {len(payload['table_identifiers'])} tables, "
          f"{len(payload['certified_questions'])} example queries, "
          f"{len(payload['sample_questions'])} sample questions and the "
          "instruction block via serialized_space", file=sys.stderr)
    return space_id


def _grant_sp_can_query(client: Any, space_id: str, sp_application_id: str) -> None:
    """Grant the app's service principal CAN_QUERY on the Genie space.

    DABs has no `genie_space` resource type so this can't live in
    resources/app.yml. The app SP needs CAN_QUERY (to ask questions) plus
    USAGE/SELECT on the underlying gold tables (granted elsewhere)."""
    # Genie spaces use the standard workspace permission levels CAN_VIEW,
    # CAN_RUN (chat), CAN_EDIT, CAN_MANAGE - NOT the CAN_QUERY level used by
    # SQL warehouses. The app SP only needs CAN_RUN to ask questions.
    try:
        client.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/genie/{space_id}",
            body={
                "access_control_list": [
                    {
                        "service_principal_name": sp_application_id,
                        "permission_level": "CAN_RUN",
                    }
                ]
            },
        )
        print(f"[genie] granted CAN_RUN to SP {sp_application_id}",
              file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[genie] WARN: could not grant CAN_RUN to {sp_application_id} "
              f"({e}). Grant in the UI: open the space -> Share -> add the "
              "app SP with CAN_RUN.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="Databricks CLI profile to use for auth")
    ap.add_argument("--host", help="Workspace host (overrides profile)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--warehouse-id", required=True,
                    help="SQL warehouse id the space will query")
    ap.add_argument("--grant-sp",
                    help="Application id of the app service principal; if "
                         "provided, grants it CAN_QUERY on the new space")
    ap.add_argument("--spec",
                    default=str(Path(__file__).parent.parent
                                / "resources" / "genie" / "flood_underwriter.json"))
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient
    client = WorkspaceClient(profile=args.profile, host=args.host) \
        if (args.profile or args.host) else WorkspaceClient()

    payload = _load_spec(Path(args.spec), args.catalog, args.schema)
    space_id = _find_existing(client, payload["display_name"])
    space_id = _create_or_update(client, args.warehouse_id, space_id, payload)

    if args.grant_sp:
        _grant_sp_can_query(client, space_id, args.grant_sp)

    # The deploy pipeline captures this stdout to plumb the id into the bundle.
    print(space_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
