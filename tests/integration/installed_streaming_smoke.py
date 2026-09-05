"""Run directly with PYTHONPATH pointing at an isolated installed wheel."""

import os
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    with TemporaryDirectory(prefix="memorizz-installed-stream-") as root:
        os.environ["MEMORIZZ_STREAM_FIXTURE_DIR"] = root
        os.environ["MEMORIZZ_HOME"] = root
        import memorizz
        from memorizz import CancellationToken, EventStream, StreamEvent
        from memorizz.mcp.streaming import MCPEventDispatcher

        expected = Path(os.environ["MEMORIZZ_EXPECTED_PACKAGE_ROOT"]).resolve()
        assert Path(memorizz.__file__).resolve().is_relative_to(expected)
        fixture = runpy.run_path(str(Path(__file__).with_name("streaming_fixture.py")))
        agent = fixture["agent"]()
        events = []
        with agent.run_stream_events("hello") as stream:
            assert isinstance(stream, EventStream)
            for event in stream:
                assert isinstance(event, StreamEvent)
                events.append(event)
                if event["type"] == "answer.delta" and event["delta"] == "Hello ":
                    assert not Path(root, "provider_complete").exists()
                    Path(root, "release").touch()
        assert events[-1]["status"] == "completed"
        assert (
            "".join(e["delta"] for e in events if e["type"] == "answer.delta")
            == "Hello 世界 \n"
        )
        assert Path(root, "provider_closed").exists()
        assert CancellationToken and MCPEventDispatcher
        print("Installed-wheel barrier proof passed:", memorizz.__file__)


if __name__ == "__main__":
    main()
