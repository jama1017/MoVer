"""One source-scoped browser page for repeated animation rebuild and capture."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from mover.converter.mover_converter import (
    _get_bound_port,
    _normalize_capture_duration,
    _wait_for_server_start,
    setup_fastapi_app,
)
from mover.converter.raster_capture import capture_png_frames_at_times


LOGGER = logging.getLogger(__name__)
_SERVER_CLOSE_TIMEOUT_S = 10.0


async def _await_cleanup_task(task: asyncio.Task) -> None:
    """Finish cleanup before propagating cancellation of its caller."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
    if cancellation is not None:
        if not task.cancelled() and task.exception() is not None:
            LOGGER.error(
                "RenderSession cleanup failed during cancellation",
                exc_info=task.exception(),
            )
        raise cancellation
    task.result()


class RenderSession:
    """A serial, async-only render session for one trusted HTML source."""

    def __init__(
        self,
        html_file: str | Path,
        *,
        output_dir: str | Path | None = None,
        capture_duration: float | None = None,
        browser_launch_options: Mapping[str, Any] | None = None,
        context_options: Mapping[str, Any] | None = None,
        navigation_timeout_ms: float = 60_000,
    ) -> None:
        self.html_file = Path(html_file).resolve()
        self.output_dir = (
            str(Path(output_dir).resolve()) if output_dir is not None else None
        )
        self.capture_duration = _normalize_capture_duration(capture_duration)
        self.navigation_timeout_ms = navigation_timeout_ms
        self._browser_launch_options = {
            "headless": True,
            **dict(browser_launch_options or {}),
        }
        executable = os.environ.get("MOVER_BROWSER_TEST_EXECUTABLE")
        if executable and "executable_path" not in self._browser_launch_options:
            self._browser_launch_options["executable_path"] = executable

        self._context_options = {
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
            **dict(context_options or {}),
        }
        if self._context_options.get("device_scale_factor") != 1:
            raise ValueError("RenderSession requires device_scale_factor == 1")

        self._state = "new"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_operation: str | None = None
        self._close_task: asyncio.Task | None = None
        self._capture_requires_rebuild = False
        self._timeline_requires_rebuild_between_captures = False

        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._port: int | None = None
        self.timeline_selection: dict[str, Any] | None = None
        self.timeline_info: dict[str, Any] | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def page(self) -> Page:
        return self._require_page()

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("RenderSession is not started")
        return self._port

    async def __aenter__(self) -> "RenderSession":
        return await self.start()

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.close()

    async def start(self) -> "RenderSession":
        if self._state == "started":
            self._require_loop()
            return self
        if self._state != "new":
            raise RuntimeError(
                f"RenderSession cannot start from state {self._state!r}"
            )
        if not self.html_file.is_file():
            raise FileNotFoundError(self.html_file)

        self._state = "starting"
        self._loop = asyncio.get_running_loop()
        try:
            app = setup_fastapi_app(
                str(self.html_file),
                str(self.html_file.parent),
                self.html_file.stem,
                output_dir=self.output_dir,
            )
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=0,
                log_level="error",
            )
            self._server = uvicorn.Server(config)
            self._server_task = asyncio.create_task(self._server.serve())
            await _wait_for_server_start(self._server, self._server_task)
            self._port = _get_bound_port(self._server)

            await self._acquire_resource(
                "_playwright",
                async_playwright().start(),
            )
            assert self._playwright is not None
            await self._acquire_resource(
                "_browser",
                self._playwright.chromium.launch(
                    **self._browser_launch_options
                ),
            )
            assert self._browser is not None
            await self._acquire_resource(
                "_context",
                self._browser.new_context(**self._context_options),
            )
            assert self._context is not None
            await self._acquire_resource("_page", self._context.new_page())
            await self._initialize_page()
        except BaseException:
            self._state = "closing"
            cleanup_task = asyncio.create_task(self._close_resources())
            try:
                await _await_cleanup_task(cleanup_task)
            except BaseException:
                LOGGER.exception("RenderSession startup rollback failed")
            self._state = "closed"
            raise

        self._state = "started"
        return self

    async def _acquire_resource(self, attribute: str, awaitable) -> None:
        """Retain late-created resources before propagating cancellation."""
        acquisition = asyncio.create_task(awaitable)
        cancellation: asyncio.CancelledError | None = None
        while not acquisition.done():
            try:
                await asyncio.shield(acquisition)
            except asyncio.CancelledError as error:
                if acquisition.cancelled():
                    raise
                cancellation = error
        try:
            resource = acquisition.result()
        except BaseException:
            if cancellation is not None:
                raise cancellation
            raise
        setattr(self, attribute, resource)
        if cancellation is not None:
            raise cancellation

    async def _initialize_page(self) -> None:
        assert self._page is not None
        await self._page.goto(
            f"http://127.0.0.1:{self.port}/",
            wait_until="networkidle",
            timeout=self.navigation_timeout_ms,
        )
        await self._page.evaluate("document.fonts.ready")
        self.timeline_selection = await self._page.evaluate(
            "duration => initializeTimelineControl(duration)",
            self.capture_duration,
        )
        self.timeline_info = await self._page.evaluate(
            "prepareTimelineForCapture()"
        )
        await self._refresh_capture_reuse_policy()

    async def _refresh_capture_reuse_policy(self) -> None:
        assert self._page is not None
        self._timeline_requires_rebuild_between_captures = (
            await self._page.evaluate(
                "timelineRequiresRebuildBetweenCaptures()"
            )
        )

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        page = self._begin_operation("evaluate")
        try:
            return await page.evaluate(expression, argument)
        finally:
            self._active_operation = None

    async def get_animation_info(self, fps: int = 60) -> dict[str, Any]:
        page = self._begin_operation("get_animation_info")
        try:
            return await page.evaluate("fps => getAnimationInfo(fps)", fps)
        finally:
            self._active_operation = None

    async def rebuild(
        self,
        complete_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(complete_params, Mapping):
            raise ValueError("complete_params must be a mapping")
        page = self._begin_operation("rebuild")
        try:
            result = await page.evaluate(
                """([params, captureDuration]) =>
                    rebuildAnimationForCapture(params, captureDuration)""",
                [dict(complete_params), self.capture_duration],
            )
            self.timeline_selection = result["selection"]
            self.timeline_info = result["info"]
            self._timeline_requires_rebuild_between_captures = bool(
                result["requiresRebuildBetweenCaptures"]
            )
            self._capture_requires_rebuild = False
            return result
        except BaseException:
            await self._close_after_failure()
            raise
        finally:
            self._active_operation = None

    async def capture(
        self,
        seek_times: Sequence[float],
        *,
        width: int,
        height: int,
        hide_grid: bool = False,
        omit_background: bool = False,
    ) -> list[np.ndarray]:
        if self._capture_requires_rebuild:
            self._require_page()
            raise RuntimeError(
                "RenderSession capture already consumed the current timeline "
                "state; call rebuild() before capturing again, or start a new "
                "session for sources without a rebuild hook"
            )
        page = self._begin_operation("capture")
        try:
            frames = await capture_png_frames_at_times(
                page,
                seek_times,
                width=width,
                height=height,
                hide_grid=hide_grid,
                omit_background=omit_background,
            )
            self._capture_requires_rebuild = (
                self._timeline_requires_rebuild_between_captures
            )
            return frames
        except BaseException:
            await self._close_after_failure()
            raise
        finally:
            self._active_operation = None

    def _begin_operation(self, name: str) -> Page:
        page = self._require_page()
        if self._active_operation is not None:
            raise RuntimeError(
                "RenderSession capacity is one; "
                f"{self._active_operation!r} is already active"
            )
        self._active_operation = name
        return page

    def _require_page(self) -> Page:
        self._require_loop()
        if self._state != "started" or self._page is None:
            raise RuntimeError("RenderSession is not started")
        return self._page

    def _require_loop(self) -> None:
        if (
            self._loop is not None
            and self._loop is not asyncio.get_running_loop()
        ):
            raise RuntimeError(
                "RenderSession must be used from the event loop that started it"
            )

    async def _close_after_failure(self) -> None:
        try:
            await self.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("RenderSession cleanup after operation failure failed")

    async def close(self) -> None:
        self._require_loop()
        if self._state == "closed":
            return
        if self._state == "new":
            self._state = "closed"
            return
        if self._close_task is None:
            self._state = "closing"
            self._close_task = asyncio.create_task(self._close_resources())
        try:
            await _await_cleanup_task(self._close_task)
        finally:
            if self._close_task.done():
                self._state = "closed"

    async def _close_resources(self) -> None:
        errors: list[BaseException] = []

        async def close_resource(name: str, resource) -> None:
            if resource is None:
                return
            try:
                await resource.close()
            except Exception as error:
                LOGGER.warning("Could not close RenderSession %s: %s", name, error)
                errors.append(error)

        page, self._page = self._page, None
        await close_resource("page", page)
        context, self._context = self._context, None
        await close_resource("context", context)
        browser, self._browser = self._browser, None
        await close_resource("browser", browser)

        playwright, self._playwright = self._playwright, None
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception as error:
                LOGGER.warning("Could not stop Playwright: %s", error)
                errors.append(error)

        server, self._server = self._server, None
        server_task, self._server_task = self._server_task, None
        self._port = None
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            try:
                await asyncio.wait_for(
                    server_task,
                    timeout=_SERVER_CLOSE_TIMEOUT_S,
                )
            except TimeoutError as error:
                if server is not None:
                    server.force_exit = True
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
                errors.append(error)
            except Exception as error:
                errors.append(error)

        if errors:
            raise RuntimeError(
                f"RenderSession cleanup failed: {errors[0]}"
            ) from errors[0]
