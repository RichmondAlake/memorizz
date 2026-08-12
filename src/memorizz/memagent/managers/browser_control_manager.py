# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Manager that exposes bounded browser-control providers as one governed tool."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from ...browser_control import (
    BrowserControlProvider,
    BrowserControlResult,
    create_browser_control_provider,
)
from ...tooling import governed_tool


class BrowserControlManager:
    def __init__(self, provider: Optional[BrowserControlProvider] = None) -> None:
        self.provider = provider

    @classmethod
    def from_config(
        cls, config: Union[str, Dict[str, Any], BrowserControlProvider]
    ) -> "BrowserControlManager":
        if isinstance(config, BrowserControlProvider):
            issue = config.validate_configuration()
            if issue:
                raise ValueError(issue)
            return cls(config)
        if isinstance(config, str):
            provider_name, provider_config = config, {}
        elif isinstance(config, dict):
            provider_config = dict(config)
            provider_name = str(provider_config.pop("provider", "browseruse"))
        else:
            raise ValueError(
                "browser_control must be a provider name, config dict, or "
                "BrowserControlProvider instance"
            )
        provider = create_browser_control_provider(provider_name, provider_config)
        if provider is None:
            raise ValueError(
                f"Unknown browser-control provider '{provider_name}'. "
                "Supported providers: browseruse."
            )
        return cls(provider)

    def set_provider(
        self, provider: Optional[BrowserControlProvider]
    ) -> Optional[BrowserControlProvider]:
        previous = self.provider
        if previous is not None and previous is not provider:
            previous.close()
        self.provider = provider
        return previous

    def is_enabled(self) -> bool:
        return self.provider is not None

    def get_provider_name(self) -> Optional[str]:
        return self.provider.get_provider_name() if self.provider else None

    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        return self.provider.get_config() if self.provider else None

    def run_task(
        self,
        task: str,
        *,
        max_steps: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> BrowserControlResult:
        if self.provider is None:
            raise ValueError("Browser-control provider is not configured")
        return self.provider.run_task(task, max_steps=max_steps, timeout=timeout)

    def get_tools(self) -> List[Callable[..., str]]:
        manager = self

        @governed_tool(
            deterministic=False,
            side_effects=True,
            requires_approval=True,
            approval_reason=(
                "Browser automation can navigate, click, type, submit forms, "
                "and change external systems"
            ),
            domains=("browser", "external"),
        )
        def browser_control(task: str, max_steps: int = 25) -> str:
            """Complete an approved task in the configured web browser.

            Browser control may click, type, submit forms, or otherwise change
            external systems. Describe the complete intended outcome in one
            task. MemoRizz pauses this call for host approval before execution.

            Args:
                task: Complete natural-language browser task.
                max_steps: Maximum browser-agent steps, capped by host policy.
            """
            return manager.run_task(task, max_steps=max_steps).to_json()

        return [browser_control]

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()
