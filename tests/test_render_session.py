import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from playwright.async_api import Error as PlaywrightError

from mover.converter.raster_capture import capture_png_frames_at_times
from mover.converter.render_session import RenderSession


REBUILD_FIXTURE = """<!doctype html>
<html>
<head>
    <meta charset="utf-8"/>
    <style>html, body { margin: 0; padding: 0; }</style>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <svg width="200" height="200" viewBox="0 0 200 200">
        <rect width="200" height="200" fill="#f4f4f4"/>
        <circle id="dot" cx="40" cy="100" r="18" fill="#d62728"/>
    </svg>
    <script>
        const dot = document.querySelector("#dot");
        const requiredKeys = ["color", "distance", "duration"];

        function validateCompleteParams(params) {
            const keys = Object.keys(params).sort();
            if (
                keys.length !== requiredKeys.length
                || !requiredKeys.every((key, index) => key === keys[index])
            ) {
                throw new Error("complete parameter set is required");
            }
            if (
                !Number.isFinite(params.distance)
                || !Number.isFinite(params.duration)
                || params.duration <= 0
                || typeof params.color !== "string"
            ) {
                throw new Error("invalid rebuild parameters");
            }
        }

        function resetScene(params) {
            dot.removeAttribute("style");
            dot.removeAttribute("transform");
            dot.setAttribute("cx", "40");
            dot.setAttribute("cy", "100");
            dot.setAttribute("fill", params.color);
        }

        function buildTimeline(params) {
            resetScene(params);
            const timeline = gsap.timeline({paused: true});
            timeline.to(dot, {
                x: params.distance,
                duration: params.duration,
                ease: "none",
            });
            return timeline;
        }

        let tl = buildTimeline({
            color: "#d62728",
            distance: 80,
            duration: 1,
        });

        function rebuildAnimation(completeParams) {
            validateCompleteParams(completeParams);
            if (typeof tl_to_use !== "undefined" && tl_to_use) {
                tl_to_use
                    .getChildren(true, true, true)
                    .forEach(animation => animation.kill());
                tl_to_use.kill();
            }
            gsap.killTweensOf(dot);
            return buildTimeline(completeParams);
        }
    </script>
    <script src="./convert.js"></script>
</body>
</html>
"""

CLEAR_PROPS_REBUILD_FIXTURE = """<!doctype html>
<html>
<head>
    <meta charset="utf-8"/>
    <style>html, body { margin: 0; padding: 0; }</style>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <svg width="200" height="200" viewBox="0 0 200 200">
        <rect width="200" height="200" fill="#f4f4f4"/>
        <g id="actor" transform="matrix(1.5,0,0,1.5,20,-50)">
            <circle cx="40" cy="100" r="18" fill="#6f4cff"/>
            <path d="M40 118 L75 145" fill="none" stroke="#111" stroke-width="8"/>
        </g>
    </svg>
    <script>
        const actor = document.querySelector("#actor");
        const initialTransform = "matrix(1.5,0,0,1.5,20,-50)";

        function resetScene(params) {
            actor.removeAttribute("style");
            actor.removeAttribute("data-svg-origin");
            actor.setAttribute("transform", initialTransform);
            actor.setAttribute("opacity", "1");
            actor.querySelector("circle").setAttribute("fill", params.color);
        }

        function buildTimeline(params) {
            resetScene(params);
            const timeline = gsap.timeline({paused: true});
            timeline.set(actor, {clearProps: "all"}, 0);
            timeline.set(actor, {
                svgOrigin: "40 100",
                rotation: 0,
                y: 0,
            }, 0);
            timeline.to(actor, {
                y: -params.distance,
                rotation: 20,
                duration: params.duration / 2,
                ease: "power2.out",
            }, 0.1);
            timeline.to(actor, {
                y: 0,
                rotation: 0,
                duration: params.duration / 2,
                ease: "power2.inOut",
            });
            return timeline;
        }

        let tl = buildTimeline({
            color: "#6f4cff",
            distance: 30,
            duration: 1,
        });

        function rebuildAnimation(completeParams) {
            if (
                !completeParams
                || typeof completeParams.color !== "string"
                || !Number.isFinite(completeParams.distance)
                || !Number.isFinite(completeParams.duration)
            ) {
                throw new Error("complete clearProps parameters are required");
            }
            if (typeof tl_to_use !== "undefined" && tl_to_use) {
                tl_to_use
                    .getChildren(true, true, true)
                    .forEach(animation => animation.kill());
                tl_to_use.kill();
            }
            gsap.killTweensOf(actor);
            tl = buildTimeline(completeParams);
            return tl;
        }
    </script>
    <script src="./convert.js"></script>
</body>
</html>
"""


class RenderSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.html_path = Path(self.temp_dir.name) / "rebuild.html"
        self.html_path.write_text(REBUILD_FIXTURE, encoding="utf-8")
        self.clear_props_path = (
            Path(self.temp_dir.name) / "clear_props_rebuild.html"
        )
        self.clear_props_path.write_text(
            CLEAR_PROPS_REBUILD_FIXTURE,
            encoding="utf-8",
        )
        self.sessions: list[RenderSession] = []

    async def asyncTearDown(self) -> None:
        for session in reversed(self.sessions):
            try:
                await session.close()
            except Exception:
                pass
        self.temp_dir.cleanup()

    async def _start_session(
        self,
        html_path: Path | None = None,
    ) -> RenderSession:
        session = RenderSession(html_path or self.html_path)
        self.sessions.append(session)
        try:
            return await session.start()
        except PlaywrightError as error:
            if os.environ.get("MOVER_REQUIRE_BROWSER") == "1":
                raise
            self.skipTest(f"Chromium is unavailable: {error}")

    async def _assert_port_closed(self, port: int) -> None:
        with self.assertRaises(OSError):
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()

    async def test_start_capture_and_close_are_idempotent(self) -> None:
        session = await self._start_session()
        self.assertIs(await session.start(), session)
        port = session.port

        frames = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )
        self.assertEqual([frame.shape for frame in frames], [(128, 128, 4)] * 3)
        self.assertGreater(np.abs(frames[0] - frames[-1]).mean(), 0.001)

        await session.close()
        await session.close()
        self.assertEqual(session.state, "closed")
        with self.assertRaisesRegex(RuntimeError, "not started"):
            _ = session.page
        await self._assert_port_closed(port)

    async def test_context_manager_closes_after_body_failure(self) -> None:
        session = RenderSession(self.html_path)
        self.sessions.append(session)
        port = None
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            async with session:
                port = session.port
                raise RuntimeError("body failure")
        self.assertEqual(session.state, "closed")
        assert port is not None
        await self._assert_port_closed(port)

    async def test_startup_failure_rolls_back_every_resource(self) -> None:
        session = RenderSession(self.html_path)
        self.sessions.append(session)
        ports: list[int] = []

        async def fail_page_initialization() -> None:
            ports.append(session.port)
            raise RuntimeError("injected startup failure")

        with patch.object(
            session,
            "_initialize_page",
            side_effect=fail_page_initialization,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected startup failure"):
                await session.start()

        self.assertEqual(session.state, "closed")
        self.assertIsNone(session._page)
        self.assertIsNone(session._context)
        self.assertIsNone(session._browser)
        self.assertIsNone(session._playwright)
        self.assertIsNone(session._server_task)
        await self._assert_port_closed(ports[0])

    async def test_startup_cancellation_rolls_back_every_resource(self) -> None:
        session = RenderSession(self.html_path)
        self.sessions.append(session)
        entered = asyncio.Event()
        ports: list[int] = []

        async def block_page_initialization() -> None:
            ports.append(session.port)
            entered.set()
            await asyncio.Event().wait()

        with patch.object(
            session,
            "_initialize_page",
            side_effect=block_page_initialization,
        ):
            start_task = asyncio.create_task(session.start())
            await entered.wait()
            start_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await start_task

        self.assertEqual(session.state, "closed")
        self.assertIsNone(session._page)
        self.assertIsNone(session._context)
        self.assertIsNone(session._browser)
        self.assertIsNone(session._playwright)
        self.assertIsNone(session._server_task)
        await self._assert_port_closed(ports[0])

    async def test_cancelled_acquisition_retains_late_resource_for_cleanup(
        self,
    ) -> None:
        session = RenderSession(self.html_path)
        self.sessions.append(session)
        session._state = "starting"
        session._loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = asyncio.Event()

        class LatePage:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        resource = LatePage()

        async def create_late_page():
            entered.set()
            await release.wait()
            return resource

        acquisition = asyncio.create_task(
            session._acquire_resource("_page", create_late_page())
        )
        await entered.wait()
        acquisition.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await acquisition
        self.assertIs(session._page, resource)

        await session._close_resources()
        session._state = "closed"
        self.assertTrue(resource.closed)
        self.assertIsNone(session._page)

    async def test_repeated_safe_capture_reuses_loaded_page(self) -> None:
        session = await self._start_session()
        page = session.page
        context = page.context
        await session.evaluate("window._sessionMarker = 42")
        navigation_count = await session.evaluate(
            "performance.getEntriesByType('navigation').length"
        )

        first = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )
        second = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )

        self.assertIs(session.page, page)
        self.assertIs(session.page.context, context)
        self.assertEqual(await session.evaluate("window._sessionMarker"), 42)
        self.assertEqual(
            await session.evaluate(
                "performance.getEntriesByType('navigation').length"
            ),
            navigation_count,
        )
        for first_frame, second_frame in zip(first, second):
            np.testing.assert_array_equal(first_frame, second_frame)

    async def test_clear_props_second_capture_requires_rebuild(self) -> None:
        session = await self._start_session(self.clear_props_path)
        times = [0.0, 0.3, 0.6, 1.1]

        await session.capture(
            times,
            width=128,
            height=128,
            hide_grid=True,
        )
        with self.assertRaisesRegex(RuntimeError, "call rebuild"):
            await session.capture(
                times,
                width=128,
                height=128,
                hide_grid=True,
            )
        self.assertEqual(session.state, "started")

    async def test_clear_props_fresh_batch_matches_fresh_sequential(
        self,
    ) -> None:
        sequential_session = await self._start_session(self.clear_props_path)
        batch_session = await self._start_session(self.clear_props_path)
        times = [0.0, 0.3, 0.6, 1.1]

        sequential = await capture_png_frames_at_times(
            sequential_session.page,
            times,
            width=128,
            height=128,
            strategy="sequential",
            hide_grid=True,
        )
        batched = await batch_session.capture(
            times,
            width=128,
            height=128,
            hide_grid=True,
        )

        for sequential_frame, batch_frame in zip(sequential, batched):
            self.assertLess(
                float(np.abs(sequential_frame - batch_frame).mean()),
                0.001,
            )

    async def test_clear_props_rebuild_makes_capture_repeatable(self) -> None:
        session = await self._start_session(self.clear_props_path)
        params = {
            "color": "#6f4cff",
            "distance": 30,
            "duration": 1.0,
        }
        times = [0.0, 0.3, 0.6, 1.1]

        await session.rebuild(params)
        first = await session.capture(
            times,
            width=128,
            height=128,
            hide_grid=True,
        )
        await session.rebuild(params)
        second = await session.capture(
            times,
            width=128,
            height=128,
            hide_grid=True,
        )

        for first_frame, second_frame in zip(first, second):
            np.testing.assert_array_equal(first_frame, second_frame)

    async def test_rebuild_is_repeatable_and_restores_a_after_b(self) -> None:
        session = await self._start_session()
        page = session.page
        marker = await session.evaluate(
            "() => { window._rebuildMarker = {}; return true; }"
        )
        self.assertTrue(marker)
        params_a = {
            "color": "#d62728",
            "distance": 80,
            "duration": 1.0,
        }
        params_b = {
            "color": "#1f77b4",
            "distance": 120,
            "duration": 1.5,
        }

        result_a1 = await session.rebuild(params_a)
        frames_a1 = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )
        result_a2 = await session.rebuild(params_a)
        frames_a2 = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )
        result_b = await session.rebuild(params_b)
        frames_b = await session.capture(
            [0.0, 0.75, 1.5],
            width=128,
            height=128,
        )
        result_a3 = await session.rebuild(params_a)
        frames_a3 = await session.capture(
            [0.0, 0.5, 1.0],
            width=128,
            height=128,
        )

        self.assertIs(session.page, page)
        self.assertEqual(
            await session.evaluate(
                "() => typeof window._rebuildMarker === 'object'"
            ),
            True,
        )
        self.assertEqual(result_a1["info"]["animDuration"], 1.0)
        self.assertEqual(result_a2["info"]["animDuration"], 1.0)
        self.assertEqual(result_b["info"]["animDuration"], 1.5)
        self.assertEqual(result_a3["info"]["animDuration"], 1.0)
        for a1, a2, a3 in zip(frames_a1, frames_a2, frames_a3):
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(a1, a3)
        self.assertGreater(np.abs(frames_a1[-1] - frames_b[-1]).mean(), 0.001)

    async def test_rebuild_failure_invalidates_session(self) -> None:
        session = await self._start_session()
        port = session.port
        with self.assertRaisesRegex(PlaywrightError, "complete parameter set"):
            await session.rebuild({"distance": 80})
        self.assertEqual(session.state, "closed")
        await self._assert_port_closed(port)

    async def test_async_rebuild_hook_is_rejected(self) -> None:
        session = await self._start_session()
        port = session.port
        await session.evaluate(
            """() => {
                window.rebuildAnimation = async () => window.tl_to_use;
            }"""
        )
        with self.assertRaisesRegex(PlaywrightError, "must be synchronous"):
            await session.rebuild(
                {
                    "color": "#d62728",
                    "distance": 80,
                    "duration": 1,
                }
            )
        self.assertEqual(session.state, "closed")
        await self._assert_port_closed(port)

    async def test_capture_seek_failure_invalidates_session(self) -> None:
        session = await self._start_session()
        port = session.port
        await session.evaluate(
            """() => {
                window.seekToTime = () => {
                    throw new Error("injected session seek failure");
                };
            }"""
        )
        with self.assertRaisesRegex(
            PlaywrightError,
            "injected session seek failure",
        ):
            await session.capture([0.0, 1.0], width=128, height=128)
        self.assertEqual(session.state, "closed")
        await self._assert_port_closed(port)

    async def test_capture_cancellation_closes_session(self) -> None:
        session = await self._start_session()
        port = session.port
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_capture(*_args, **_kwargs):
            entered.set()
            await release.wait()

        with patch(
            "mover.converter.render_session.capture_png_frames_at_times",
            side_effect=blocked_capture,
        ):
            capture_task = asyncio.create_task(
                session.capture([0.0], width=128, height=128)
            )
            await entered.wait()
            capture_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await capture_task

        self.assertEqual(session.state, "closed")
        await self._assert_port_closed(port)

    async def test_capacity_one_rejects_overlapping_operations(self) -> None:
        session = await self._start_session()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_capture(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return []

        with patch(
            "mover.converter.render_session.capture_png_frames_at_times",
            side_effect=blocked_capture,
        ):
            capture_task = asyncio.create_task(
                session.capture([0.0], width=128, height=128)
            )
            await entered.wait()
            with self.assertRaisesRegex(RuntimeError, "capacity is one"):
                await session.evaluate("1 + 1")
            release.set()
            self.assertEqual(await capture_task, [])
        self.assertEqual(session.state, "started")

    async def test_close_finishes_cleanup_before_propagating_cancellation(
        self,
    ) -> None:
        session = await self._start_session()
        port = session.port
        entered = asyncio.Event()
        release = asyncio.Event()
        original_close = session.page.close

        async def delayed_page_close() -> None:
            entered.set()
            await release.wait()
            await original_close()

        with patch.object(
            session.page,
            "close",
            side_effect=delayed_page_close,
        ):
            close_task = asyncio.create_task(session.close())
            await entered.wait()
            close_task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

        self.assertEqual(session.state, "closed")
        self.assertIsNone(session._page)
        self.assertIsNone(session._context)
        self.assertIsNone(session._browser)
        self.assertIsNone(session._playwright)
        self.assertIsNone(session._server_task)
        await self._assert_port_closed(port)


if __name__ == "__main__":
    unittest.main(verbosity=2)
