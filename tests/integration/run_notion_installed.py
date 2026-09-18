"""Exercise an installed wheel without pytest's source-checkout path injection."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-root", type=Path, required=True)
    args = parser.parse_args()
    installed = args.installed_root.resolve()
    with tempfile.TemporaryDirectory(prefix="memorizz-notion-installed-") as directory:
        root = Path(directory)
        # Set before importing the SDK: approval/audit defaults must not write
        # to the developer's real Memorizz home during artifact verification.
        os.environ["MEMORIZZ_HOME"] = str(root / "runtime")
        os.environ["MEMORIZZ_UI_AUDIT_LOG"] = str(root / "audit.jsonl")
        _exercise(installed, root)


def _exercise(installed, root):
    # The repo supplies test fixtures only; the installed package must win.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(installed))

    from typer.testing import CliRunner

    import memorizz
    from memorizz.cli.app import app
    from tests.integration.test_cli_configuration_e2e import (
        test_actual_terminal_hides_credentials_and_next_process_redacts,
        test_fresh_process_setting_save_load_and_precedence,
    )
    from tests.integration.test_notion_provider_e2e import (
        test_real_agent_stream_memory_persistence_trace_and_human_edit,
        test_wire_level_http_provision_store_search_update_delete,
    )

    origin = Path(memorizz.__file__).resolve()
    if not origin.is_relative_to(installed):
        raise AssertionError("The installed wheel was not imported")
    assert memorizz.NotionProvider is not None
    assert (
        "Notion + separate semantic provider"
        in (origin.parent / "ui" / "templates" / "connect.html").read_text()
    )
    help_result = CliRunner().invoke(app, ["notion", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "repair" in help_result.output
    assert "connect" in help_result.output
    for arguments in (["config", "--help"], ["memory", "configure", "--help"]):
        result = CliRunner().invoke(app, arguments)
        assert result.exit_code == 0, result.output
    test_fresh_process_setting_save_load_and_precedence(root / "config")
    if os.name != "nt":
        test_actual_terminal_hides_credentials_and_next_process_redacts(
            root / "terminal"
        )
    test_real_agent_stream_memory_persistence_trace_and_human_edit(root / "agent")
    test_wire_level_http_provision_store_search_update_delete(root / "http")
    print(
        json.dumps(
            {
                "ok": True,
                "package_origin": str(origin),
                "end_to_end_tests": 4 if os.name != "nt" else 3,
                "cli_help": True,
                "ui_template": True,
            }
        )
    )


if __name__ == "__main__":
    main()
