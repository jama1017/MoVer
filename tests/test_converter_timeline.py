import json
import os
import tempfile
import unittest
from pathlib import Path

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from mover.converter.mover_converter import convert_animation


ASSETS = (
    Path(__file__).parent.parent
    / "src"
    / "mover"
    / "converter"
    / "assets"
)
GSAP_JS = ASSETS / "gsap.min.js"
VIS_JS = ASSETS / "vis.js"
CONVERT_JS = ASSETS / "convert.js"
VIS_SOURCE = VIS_JS.read_text(encoding="utf-8")
CONVERT_SOURCE = CONVERT_JS.read_text(encoding="utf-8")

BASE_DOCUMENT = """<!doctype html>
<html>
<body>
    <p id="prompt">Timeline test</p>
    <svg width="120" height="80" viewBox="0 0 120 80">
        <defs>
            <linearGradient id="paint">
                <stop id="paint-stop" offset="0%" stop-color="red"/>
                <stop offset="100%" stop-color="blue"/>
            </linearGradient>
        </defs>
        <rect id="selected" x="5" y="5" width="10" height="10" fill="black"/>
        <rect id="legacy" x="5" y="25" width="10" height="10" fill="green"/>
        <rect id="ambient" x="5" y="45" width="10" height="10" fill="blue"/>
        <circle id="callback-shape" cx="80" cy="20" r="8" fill="url(#paint)"/>
    </svg>
</body>
</html>
"""

EXPORTED_ROOT_FIXTURE = """<!doctype html>
<html>
<head>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <p id="prompt">Automatic exported-root integration test</p>
    <svg width="80" height="40" viewBox="0 0 80 40">
        <rect id="selected" x="5" y="5" width="10" height="10" fill="black"/>
        <rect id="legacy" x="5" y="25" width="10" height="10" fill="green"/>
    </svg>
    <script>
        const renamedPrimaryAnimation = gsap.timeline();
        renamedPrimaryAnimation.to(
            "#selected",
            {x: 20, duration: 1, ease: "none"}
        );

        const independentAnimation = gsap.timeline();
        independentAnimation.to(
            "#legacy",
            {x: 40, duration: 1, ease: "none"}
        );
    </script>
    <script src="./vis.js"></script>
    <script src="./convert.js"></script>
</body>
</html>
"""

INFINITE_ROOT_FIXTURE = """<!doctype html>
<html>
<head>
    <script src="./gsap.min.js"></script>
</head>
<body>
    <p id="prompt">Infinite exported-root integration test</p>
    <svg width="80" height="40" viewBox="0 0 80 40">
        <rect id="looping" x="5" y="5" width="10" height="10" fill="black"/>
    </svg>
    <script>
        const loopingAnimation = gsap.timeline();
        loopingAnimation.to(
            "#looping",
            {
                x: 20,
                duration: 1,
                repeat: -1,
                repeatDelay: 0.1,
                yoyo: true,
                ease: "none",
            }
        );
    </script>
    <script src="./vis.js"></script>
    <script src="./convert.js"></script>
</body>
</html>
"""


class TimelineControlBrowserTest(unittest.IsolatedAsyncioTestCase):
    """Focused automatic timeline cases adapted from ACL animation patterns."""

    async def asyncSetUp(self) -> None:
        self.playwright = await async_playwright().start()
        launch_options = {"headless": True}
        executable_path = os.environ.get("MOVER_BROWSER_TEST_EXECUTABLE")
        if executable_path:
            launch_options["executable_path"] = executable_path

        try:
            self.browser = await self.playwright.chromium.launch(**launch_options)
        except PlaywrightError as error:
            await self.playwright.stop()
            if os.environ.get("MOVER_REQUIRE_BROWSER") == "1":
                raise
            self.skipTest(f"Chromium is unavailable: {error}")

    async def asyncTearDown(self) -> None:
        if hasattr(self, "browser"):
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def _load_page(self, setup_script: str):
        page = await self.browser.new_page()
        await page.set_content(BASE_DOCUMENT)
        await page.add_script_tag(path=str(GSAP_JS))
        await page.add_script_tag(
            content="\n".join((setup_script, VIS_SOURCE, CONVERT_SOURCE))
        )
        return page

    async def test_prepaused_tl_and_active_sibling_are_both_controlled(self) -> None:
        page = await self._load_page(
            """
            window.legacyState = {value: 0};
            window.siblingState = {value: 0};
            const tl = gsap.timeline({paused: true});
            tl.to(legacyState, {value: 100, duration: 1, ease: "none"});
            tl.eventCallback("onComplete", () => tl.pause());
            const siblingTimeline = gsap.timeline();
            siblingTimeline.to(
                siblingState,
                {value: 200, duration: 1, ease: "none"}
            );
            """
        )

        result = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                seekToTime(0.5);
                const firstValues = [
                    legacyState.value,
                    siblingState.value,
                ];
                seekToTime(1);
                seekToTime(0);
                seekToTime(0.5);
                return {
                    selection,
                    firstValues,
                    secondValues: [
                        legacyState.value,
                        siblingState.value,
                    ],
                    usesWrapper: (
                        tl_to_use !== tl
                        && tl_to_use !== siblingTimeline
                        && tl_to_use !== gsap.globalTimeline
                    ),
                    legacyPaused: tl.paused(),
                    rootPaused: gsap.globalTimeline.paused(),
                };
            }"""
        )

        self.assertEqual(result["selection"]["source"], "exported-root")
        self.assertTrue(result["usesWrapper"])
        self.assertFalse(result["legacyPaused"])
        self.assertTrue(result["rootPaused"])
        for values in (
            result["firstValues"],
            result["secondValues"],
        ):
            self.assertAlmostEqual(values[0], 50, places=4)
            self.assertAlmostEqual(values[1], 100, places=4)

    async def test_preseeked_paused_root_keeps_effective_delay(self) -> None:
        page = await self._load_page(
            """
            window.preseekedState = {value: 0};
            window.preseekedRoot = gsap.timeline({
                paused: true,
                delay: 0.2,
            });
            preseekedRoot.to(
                preseekedState,
                {value: 100, duration: 1, ease: "none"}
            );
            preseekedRoot.totalTime(0.5, false);
            """
        )

        result = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                const controlledRoot = tl_to_use.getChildren(
                    false, true, true
                )[0];
                const initialized = {
                    childStart: controlledRoot.startTime(),
                    childTime: controlledRoot.totalTime(),
                    duration: tl_to_use.totalDuration(),
                    value: preseekedState.value,
                };
                seekToTime(0.1);
                const beforeDelay = preseekedState.value;
                seekToTime(0.7);
                const middle = preseekedState.value;
                seekToTime(1.2);
                return {
                    beforeDelay,
                    initialized,
                    middle,
                    paused: preseekedRoot.paused(),
                    selection,
                    end: preseekedState.value,
                };
            }"""
        )

        self.assertEqual(result["selection"]["rootCount"], 1)
        self.assertEqual(result["selection"]["resumedChildren"], 1)
        self.assertAlmostEqual(result["initialized"]["childStart"], 0.2, places=6)
        self.assertAlmostEqual(result["initialized"]["childTime"], 0, places=6)
        self.assertAlmostEqual(result["initialized"]["duration"], 1.2, places=6)
        self.assertAlmostEqual(result["initialized"]["value"], 0, places=4)
        self.assertAlmostEqual(result["beforeDelay"], 0, places=4)
        self.assertAlmostEqual(result["middle"], 50, places=4)
        self.assertAlmostEqual(result["end"], 100, places=4)
        self.assertFalse(result["paused"])

    async def test_visual_playback_advances_and_reinitializes_cleanly(self) -> None:
        page = await self._load_page(
            """
            window.playbackState = {value: 0};
            const playbackAnimation = gsap.timeline();
            playbackAnimation.to(
                playbackState,
                {value: 100, duration: 1, ease: "none"}
            );
            """
        )

        await page.evaluate(
            """() => {
                initializeTimelineControl();
                window.firstControlledRoot = tl_to_use;
                play();
            }"""
        )
        await page.wait_for_timeout(75)
        playing = await page.evaluate(
            """() => ({
                globalPaused: gsap.globalTimeline.paused(),
                parentIsNull: tl_to_use.parent === null,
                time: tl_to_use.totalTime(),
                unexpectedRoots: getUnexpectedRootAnimations().length,
                value: playbackState.value,
            })"""
        )
        reinitialized = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                const sameRoot = tl_to_use === firstControlledRoot;
                seekToTime(0.5);
                const result = {
                    sameRoot,
                    selection,
                    unexpectedRoots: getUnexpectedRootAnimations().length,
                    value: playbackState.value,
                };
                stopTimelineVisualizationPlayback();
                return result;
            }"""
        )

        self.assertTrue(playing["globalPaused"])
        self.assertTrue(playing["parentIsNull"])
        self.assertGreater(playing["time"], 0)
        self.assertGreater(playing["value"], 0)
        self.assertEqual(playing["unexpectedRoots"], 0)
        self.assertTrue(reinitialized["sameRoot"])
        self.assertEqual(reinitialized["selection"]["rootCount"], 1)
        self.assertEqual(reinitialized["unexpectedRoots"], 0)
        self.assertAlmostEqual(reinitialized["value"], 50, places=4)

    async def test_visual_playback_callback_can_cancel_next_frame(self) -> None:
        page = await self._load_page(
            """
            window.visualCallbackState = {value: 0};
            window.visualCallbackCount = 0;
            const visualCallbackAnimation = gsap.timeline();
            visualCallbackAnimation.to(
                visualCallbackState,
                {value: 100, duration: 1, ease: "none"},
                0
            );
            visualCallbackAnimation.call(() => {
                visualCallbackCount++;
                pause();
            }, null, 0.01);
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const pending = new Map();
                let nextFrameId = 1;
                window.requestAnimationFrame = callback => {
                    const id = nextFrameId++;
                    pending.set(id, callback);
                    return id;
                };
                window.cancelAnimationFrame = id => pending.delete(id);
                play();
                const entry = pending.entries().next().value;
                pending.delete(entry[0]);
                entry[1](performance.now() + 100);
                return {
                    callbackCount: visualCallbackCount,
                    pendingFrames: pending.size,
                    value: visualCallbackState.value,
                };
            }"""
        )

        self.assertEqual(result["callbackCount"], 1)
        self.assertEqual(result["pendingFrames"], 0)
        self.assertGreater(result["value"], 0)
        self.assertLess(result["value"], 100)

    async def test_timeline_validator_matches_total_time_contract(self) -> None:
        page = await self._load_page(
            """
            window.validatorAnimation = gsap.timeline();
            validatorAnimation.to({}, {duration: 1});
            """
        )

        result = await page.evaluate(
            """() => {
                const oldShape = {
                    seek() {},
                    pause() {},
                    totalDuration() { return 1; },
                    getChildren() { return []; },
                    getTweensOf() { return []; },
                    totalProgress() {},
                };
                const currentShape = {
                    duration() { return 1; },
                    pause() { return this; },
                    totalTime() { return this; },
                    totalDuration() { return 1; },
                    getChildren() { return []; },
                    getTweensOf() { return []; },
                };
                return {
                    actual: isControllableGsapTimeline(validatorAnimation),
                    currentShape: isControllableGsapTimeline(currentShape),
                    missingDuration: isControllableGsapTimeline({
                        ...currentShape,
                        duration: null,
                    }),
                    missingTotalTime: isControllableGsapTimeline({
                        ...currentShape,
                        totalTime: null,
                    }),
                    oldShape: isControllableGsapTimeline(oldShape),
                };
            }"""
        )

        self.assertTrue(result["actual"])
        self.assertTrue(result["currentShape"])
        self.assertFalse(result["missingDuration"])
        self.assertFalse(result["missingTotalTime"])
        self.assertFalse(result["oldShape"])

    async def test_renamed_timeline_needs_no_registration(self) -> None:
        page = await self._load_page(
            """
            window.renamedState = {value: 0};
            const animationWithAnyName = gsap.timeline();
            animationWithAnyName.to(
                renamedState,
                {value: 100, duration: 1, ease: "none"}
            );
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                seekToTime(tl_to_use.totalDuration());
                return {
                    value: renamedState.value,
                    usedRetainedWrapper: (
                        tl_to_use !== animationWithAnyName
                        && tl_to_use !== gsap.globalTimeline
                    ),
                };
            }"""
        )

        self.assertTrue(result["usedRetainedWrapper"])
        self.assertAlmostEqual(result["value"], 100, places=4)

    async def test_independent_renamed_timelines_are_collected_together(self) -> None:
        page = await self._load_page(
            """
            window.firstState = {value: 0};
            window.secondState = {value: 0};

            const firstAnimation = gsap.timeline();
            firstAnimation.to(
                firstState,
                {value: 100, duration: 1, ease: "none"}
            );

            const anotherAnimation = gsap.timeline();
            anotherAnimation.to(
                secondState,
                {value: 200, duration: 1, ease: "none"}
            );
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                seekToTime(0.75);
                return {
                    first: firstState.value,
                    second: secondState.value,
                    childCount: tl_to_use.getChildren(
                        true, true, true
                    ).length,
                    directRootCount: tl_to_use.getChildren(
                        false, true, true
                    ).length,
                    hierarchyPreserved: (
                        firstAnimation.parent === tl_to_use
                        && anotherAnimation.parent === tl_to_use
                        && firstAnimation.getChildren(
                            false, true, true
                        )[0].parent === firstAnimation
                        && anotherAnimation.getChildren(
                            false, true, true
                        )[0].parent === anotherAnimation
                    ),
                };
            }"""
        )

        self.assertGreaterEqual(result["childCount"], 4)
        self.assertEqual(result["directRootCount"], 2)
        self.assertTrue(result["hierarchyPreserved"])
        self.assertAlmostEqual(result["first"], 75, places=4)
        self.assertAlmostEqual(result["second"], 150, places=4)

    async def test_root_origin_is_zeroed_without_losing_declared_delay(self) -> None:
        page = await self._load_page(
            """
            window.immediateState = {value: 0};
            window.delayedState = {value: 0};
            const offsetRootTween = gsap.to(
                immediateState,
                {value: 100, duration: 1, ease: "none"}
            );
            const delayedRootTween = gsap.to(
                delayedState,
                {value: 100, duration: 1, delay: 0.5, ease: "none"}
            );
            [offsetRootTween, delayedRootTween].forEach(animation => {
                animation.startTime(animation.startTime() + 0.03);
            });
            window.expectedRootEpoch = Math.min(
                offsetRootTween.startTime() - offsetRootTween.delay(),
                delayedRootTween.startTime() - delayedRootTween.delay(),
            );
            """
        )

        result = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                const info = getAnimationInfo(20);
                seekToTime(0.25);
                const early = [immediateState.value, delayedState.value];
                seekToTime(0.75);
                return {
                    info,
                    early,
                    later: [immediateState.value, delayedState.value],
                    selection,
                    autoRemoveChildren: tl_to_use.autoRemoveChildren,
                    smoothChildTiming: tl_to_use.smoothChildTiming,
                    expectedRootEpoch,
                };
            }"""
        )

        self.assertAlmostEqual(result["info"]["animDuration"], 1.5, places=4)
        self.assertAlmostEqual(result["early"][0], 25, places=4)
        self.assertAlmostEqual(result["early"][1], 0, places=4)
        self.assertAlmostEqual(result["later"][0], 75, places=4)
        self.assertAlmostEqual(result["later"][1], 25, places=4)
        self.assertEqual(result["selection"]["strategy"], "recorded-root")
        self.assertEqual(result["selection"]["rootCount"], 2)
        self.assertAlmostEqual(
            result["selection"]["commonEpoch"],
            result["expectedRootEpoch"],
            places=6,
        )
        self.assertFalse(result["autoRemoveChildren"])
        self.assertTrue(result["smoothChildTiming"])

    async def test_callback_free_prepare_does_not_traverse_end_state(self) -> None:
        page = await self._load_page(
            """
            window.prepareState = {value: 0};
            window.prepareSawEnd = false;
            const ordinaryAnimation = gsap.timeline();
            ordinaryAnimation.to(
                prepareState,
                {
                    value: 100,
                    duration: 1,
                    ease: "none",
                    onUpdate: () => {
                        if (prepareState.value >= 99) {
                            prepareSawEnd = true;
                        }
                    },
                }
            );
            """
        )

        result = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                const before = {
                    time: tl_to_use.totalTime(),
                    value: prepareState.value,
                };
                const info = prepareTimelineForCapture();
                return {
                    before,
                    info,
                    preparedTime: tl_to_use.totalTime(),
                    preparedValue: prepareState.value,
                    prepareSawEnd,
                    selection,
                };
            }"""
        )

        self.assertEqual(result["selection"]["strategy"], "recorded-root")
        self.assertEqual(result["info"]["animDuration"], 1)
        self.assertEqual(result["before"]["time"], 0)
        self.assertAlmostEqual(result["before"]["value"], 0, places=4)
        self.assertEqual(result["preparedTime"], 0)
        self.assertAlmostEqual(result["preparedValue"], 0, places=4)
        self.assertFalse(result["prepareSawEnd"])

    async def test_immediate_root_set_remains_static_across_replays(self) -> None:
        page = await self._load_page(
            """
            window.setupState = {value: 0};
            gsap.set(setupState, {value: 10});
            const contentAnimation = gsap.timeline();
            contentAnimation.to(
                setupState,
                {value: 20, duration: 1, ease: "none"}
            );
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const directChildren = tl_to_use.getChildren(
                    false, true, true
                );
                seekToTime(0);
                const firstStart = setupState.value;
                seekToTime(1);
                const firstEnd = setupState.value;
                seekToTime(0);
                const secondStart = setupState.value;
                seekToTime(1);
                return {
                    firstStart,
                    firstEnd,
                    secondStart,
                    secondEnd: setupState.value,
                    directZeroDurationTweens: directChildren.filter(child => (
                        typeof child.targets === "function"
                        && child.duration() === 0
                    )).length,
                };
            }"""
        )

        self.assertEqual(result["directZeroDurationTweens"], 0)
        self.assertAlmostEqual(result["firstStart"], 10, places=4)
        self.assertAlmostEqual(result["firstEnd"], 20, places=4)
        self.assertAlmostEqual(result["secondStart"], 10, places=4)
        self.assertAlmostEqual(result["secondEnd"], 20, places=4)

    async def test_materialized_root_setup_tween_is_retained(self) -> None:
        page = await self._load_page(
            """
            window.retainedSetupState = {value: 0};
            window.retainedMotionState = {value: 0};
            gsap.globalTimeline.autoRemoveChildren = false;
            gsap.set(retainedSetupState, {value: 7});
            gsap.to(
                retainedMotionState,
                {value: 100, duration: 1, ease: "none"}
            );
            """
        )

        result = await page.evaluate(
            """() => {
                const selection = initializeTimelineControl();
                const directChildren = tl_to_use.getChildren(
                    false, true, true
                );
                seekToTime(0);
                const start = {
                    setup: retainedSetupState.value,
                    motion: retainedMotionState.value,
                };
                seekToTime(1);
                return {
                    directZeroDurationTweens: directChildren.filter(child => (
                        typeof child.targets === "function"
                        && child.duration() === 0
                    )).length,
                    end: {
                        setup: retainedSetupState.value,
                        motion: retainedMotionState.value,
                    },
                    selection,
                    start,
                };
            }"""
        )

        self.assertEqual(result["selection"]["rootCount"], 2)
        self.assertEqual(result["directZeroDurationTweens"], 1)
        self.assertEqual(result["start"], {"setup": 7, "motion": 0})
        self.assertEqual(result["end"], {"setup": 7, "motion": 100})

    async def test_gsdevtools_is_disabled_without_killing_content(self) -> None:
        page = await self._load_page(
            """
            window.devToolsKillCount = 0;
            window.devToolsState = {value: 0};
            const tl = gsap.timeline();
            tl.to(
                devToolsState,
                {value: 100, duration: 1, ease: "none"}
            );
            const devToolsInstance = {
                kill() {
                    devToolsKillCount++;
                    window.devToolsInstanceKilled = true;
                },
            };
            window.GSDevTools = {
                getByAnimation(animation) {
                    if (
                        animation === tl
                        && !window.devToolsInstanceKilled
                    ) {
                        return devToolsInstance;
                    }
                    return null;
                },
            };
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                seekToTime(0.5);
                return {
                    killCount: devToolsKillCount,
                    duration: getAnimationInfo(20).animDuration,
                    value: devToolsState.value,
                };
            }"""
        )

        self.assertEqual(result["killCount"], 1)
        self.assertEqual(result["duration"], 1)
        self.assertAlmostEqual(result["value"], 50, places=4)

    async def test_export_root_retains_multiple_loose_animations(self) -> None:
        page = await self._load_page(
            """
            window.firstState = {value: 0};
            window.secondState = {value: 0};
            gsap.to(firstState, {value: 100, duration: 1, ease: "none"});
            gsap.to(secondState, {value: 200, duration: 1, ease: "none"});
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const childCount = tl_to_use.getChildren(
                    true, true, true
                ).length;
                seekToTime(0.5);
                const middle = [firstState.value, secondState.value];
                seekToTime(1);
                const end = [firstState.value, secondState.value];
                seekToTime(0);
                const start = [firstState.value, secondState.value];
                seekToTime(1);
                const repeatedEnd = [firstState.value, secondState.value];
                return {
                    childCount,
                    isRawRoot: tl_to_use === gsap.globalTimeline,
                    middle,
                    end,
                    start,
                    repeatedEnd,
                };
            }"""
        )

        self.assertFalse(result["isRawRoot"])
        self.assertGreaterEqual(result["childCount"], 2)
        for actual, expected in (
            (result["middle"], [50, 100]),
            (result["end"], [100, 200]),
            (result["start"], [0, 0]),
            (result["repeatedEnd"], [100, 200]),
        ):
            for value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(value, expected_value, places=4)

    async def test_tl_and_infinite_sibling_share_explicit_window(self) -> None:
        page = await self._load_page(
            """
            window.selectedState = {value: 0};
            window.ambientState = {value: 0};

            const tl = gsap.timeline({paused: true});
            tl.to(
                selectedState,
                {value: 100, duration: 1, ease: "none"}
            );
            window.ambientTween = gsap.to(
                ambientState,
                {
                    value: 100,
                    duration: 1,
                    repeat: -1,
                    yoyo: true,
                    ease: "none",
                }
            );
            """
        )

        await page.evaluate("initializeTimelineControl()")
        with self.assertRaisesRegex(PlaywrightError, "capture_duration"):
            await page.evaluate("getAnimationInfo(20)")

        info = await page.evaluate(
            """() => {
                setMoverCaptureDuration(2);
                const prepared = prepareTimelineForCapture();
                seekToTime(0.5);
                return prepared;
            }"""
        )
        before = await page.evaluate(
            "() => ({selected: selectedState.value, ambient: ambientState.value})"
        )
        await page.wait_for_timeout(75)
        after = await page.evaluate(
            "() => ({selected: selectedState.value, ambient: ambientState.value})"
        )

        self.assertEqual(info["animDuration"], 2)
        self.assertEqual(info["steps"], 120)
        self.assertAlmostEqual(before["selected"], 50, places=4)
        self.assertAlmostEqual(before["ambient"], 50, places=4)
        self.assertAlmostEqual(after["selected"], 50, places=4)
        self.assertAlmostEqual(
            before["ambient"],
            after["ambient"],
            places=4,
        )

    async def test_zero_duration_timeline_produces_one_frame(self) -> None:
        page = await self._load_page(
            """
            window.zeroState = {value: 0};
            const tl = gsap.timeline({paused: true});
            tl.set(zeroState, {value: 7});
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const info = getAnimationInfo(60);
                const frameCount = seekAndAppendToDomUsingTimes([0]);
                const renderedValue = zeroState.value;
                const reset = resetSeekAndAppend();
                return {info, frameCount, renderedValue, reset};
            }"""
        )

        self.assertEqual(result["info"]["animDuration"], 0)
        self.assertEqual(result["info"]["steps"], 0)
        self.assertEqual(result["frameCount"], 1)
        self.assertEqual(result["renderedValue"], 7)
        self.assertTrue(result["reset"])

    async def test_multiple_infinite_roots_require_explicit_duration(self) -> None:
        page = await self._load_page(
            """
            window.firstLoopState = {value: 0};
            window.secondLoopState = {value: 0};
            gsap.to(
                firstLoopState,
                {
                    value: 100,
                    duration: 1,
                    repeat: -1,
                    yoyo: true,
                    ease: "none",
                }
            );
            gsap.to(
                secondLoopState,
                {
                    value: 100,
                    duration: 0.5,
                    repeat: -1,
                    repeatDelay: 0.1,
                    ease: "none",
                }
            );
            """
        )

        await page.evaluate("initializeTimelineControl()")
        with self.assertRaisesRegex(PlaywrightError, "capture_duration"):
            await page.evaluate("getAnimationInfo(20)")

        result = await page.evaluate(
            """() => {
                setMoverCaptureDuration(2.5);
                prepareTimelineForCapture();
                const info = getAnimationInfo(20);
                seekToTime(0.25);
                const firstSample = [
                    firstLoopState.value,
                    secondLoopState.value,
                ];
                seekToTime(0);
                seekToTime(0.25);
                return {
                    info,
                    firstSample,
                    secondSample: [
                        firstLoopState.value,
                        secondLoopState.value,
                    ],
                };
            }"""
        )

        self.assertEqual(result["info"]["animDuration"], 2.5)
        self.assertEqual(result["info"]["steps"], 50)
        for first, second in zip(
            result["firstSample"],
            result["secondSample"],
        ):
            self.assertAlmostEqual(first, second, places=4)

    async def test_callback_mutation_restores_identically(self) -> None:
        # Adapted from ACL's gradient-glow proxy/onUpdate pattern.
        page = await self._load_page(
            """
            window.colorState = {offset: 0};
            const gradientAnimation = gsap.timeline();
            gradientAnimation.to(
                colorState,
                {
                    offset: 100,
                    duration: 1,
                    ease: "none",
                    onUpdate: () => {
                        document.querySelector("#paint-stop").setAttribute(
                            "offset",
                            `${colorState.offset}%`
                        );
                    },
                }
            );
            """
        )

        states = await page.evaluate(
            """() => {
                initializeTimelineControl();
                seekToTime(0.25);
                const results = [];
                for (let i = 0; i < 2; i++) {
                    beginServerDrivenCapture();
                    seekToTime(0.75);
                    restoreServerDrivenCapture();
                    results.push({
                        time: tl_to_use.totalTime(),
                        offset: parseFloat(
                            document.querySelector("#paint-stop")
                                .getAttribute("offset")
                        ),
                    });
                }
                return results;
            }"""
        )

        self.assertEqual(len(states), 2)
        for state in states:
            self.assertAlmostEqual(state["time"], 0.25, places=6)
            self.assertAlmostEqual(state["offset"], 25, places=4)

    async def test_callback_created_child_can_join_controlled_timeline(self) -> None:
        # Adapted from ACL's recursive twinkle pattern, with an explicit
        # idempotent page contract instead of a loose global child.
        page = await self._load_page(
            """
            window.sparkleState = {value: 0};
            window.sparkleTrigger = {value: 0};
            window.sparkleChild = null;
            window.sparkleBuildCount = 0;
            const sparkleAnimation = gsap.timeline();
            sparkleAnimation.to(
                sparkleTrigger,
                {value: 1, duration: 0.1, ease: "none"}
            );
            sparkleAnimation.call(() => {
                if (sparkleChild === null) {
                    sparkleBuildCount++;
                    sparkleChild = gsap.timeline();
                    sparkleChild.to(
                        sparkleState,
                        {value: 100, duration: 1, ease: "none"}
                    );
                    sparkleAnimation.add(sparkleChild, 0.1);
                }
            }, null, 0.1);
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const prepared = prepareTimelineForCapture();
                seekToTime(0.6);
                const first = sparkleState.value;
                seekToTime(0);
                seekToTime(0.6);
                return {
                    first,
                    second: sparkleState.value,
                    buildCount: sparkleBuildCount,
                    childIsAttached: sparkleChild.parent === sparkleAnimation,
                    duration: prepared.animDuration,
                };
            }"""
        )

        self.assertAlmostEqual(result["first"], 50, places=4)
        self.assertAlmostEqual(result["second"], 50, places=4)
        self.assertEqual(result["buildCount"], 1)
        self.assertTrue(result["childIsAttached"])
        self.assertAlmostEqual(result["duration"], 1.1, places=4)

    async def test_loose_post_snapshot_animation_fails_clearly(self) -> None:
        page = await self._load_page(
            """
            window.triggerState = {value: 0};
            window.lateState = {value: 0};
            const sourceAnimation = gsap.timeline();
            sourceAnimation.to(
                triggerState,
                {value: 1, duration: 0.1, ease: "none"}
            );
            sourceAnimation.call(() => {
                gsap.to(
                    lateState,
                    {value: 100, duration: 1, ease: "none"}
                );
            }, null, 0.1);
            """
        )

        await page.evaluate("initializeTimelineControl()")
        with self.assertRaisesRegex(
            PlaywrightError,
            "Unsupported post-snapshot GSAP animation",
        ):
            await page.evaluate("prepareTimelineForCapture()")

    async def test_delayed_call_is_excluded_without_false_failure(self) -> None:
        page = await self._load_page(
            """
            window.finiteState = {value: 0};
            window.delayedCallFired = false;
            gsap.to(
                finiteState,
                {value: 100, duration: 1, ease: "none"}
            );
            gsap.delayedCall(0.1, () => {
                delayedCallFired = true;
            });
            """
        )

        result = await page.evaluate(
            """() => {
                initializeTimelineControl();
                const info = prepareTimelineForCapture();
                seekToTime(0.5);
                return {
                    info,
                    value: finiteState.value,
                    delayedCallFired,
                    unexpectedRoots: getUnexpectedRootAnimations().length,
                };
            }"""
        )

        self.assertEqual(result["info"]["animDuration"], 1)
        self.assertAlmostEqual(result["value"], 50, places=4)
        self.assertFalse(result["delayedCallFired"])
        self.assertEqual(result["unexpectedRoots"], 0)

    async def test_raf_created_loose_root_is_detected(self) -> None:
        page = await self._load_page(
            """
            window.triggerState = {value: 0};
            window.rafState = {value: 0};
            const sourceAnimation = gsap.timeline();
            sourceAnimation.to(
                triggerState,
                {value: 1, duration: 0.1, ease: "none"}
            );
            sourceAnimation.call(() => {
                requestAnimationFrame(() => {
                    gsap.to(
                        rafState,
                        {value: 100, duration: 1, ease: "none"}
                    );
                });
            }, null, 0.1);
            """
        )

        await page.evaluate(
            """() => {
                initializeTimelineControl();
                seekToTime(0.1);
            }"""
        )
        await page.evaluate(
            "() => new Promise(resolve => "
            "requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )
        with self.assertRaisesRegex(
            PlaywrightError,
            "Unsupported post-snapshot GSAP animation",
        ):
            await page.evaluate("assertNoLateRootAnimations()")

    async def test_unstable_attached_children_fail_warmup(self) -> None:
        page = await self._load_page(
            """
            window.triggerState = {value: 0};
            window.unstableState = {value: 0};
            const sourceAnimation = gsap.timeline();
            sourceAnimation.to(
                triggerState,
                {value: 1, duration: 0.1, ease: "none"}
            );
            sourceAnimation.call(() => {
                const child = gsap.timeline();
                child.to(
                    unstableState,
                    {value: 100, duration: 0.2, ease: "none"}
                );
                sourceAnimation.add(child, 0.1);
            }, null, 0.1);
            """
        )

        await page.evaluate("initializeTimelineControl()")
        with self.assertRaisesRegex(
            PlaywrightError,
            "did not stabilize during warm-up",
        ):
            await page.evaluate("prepareTimelineForCapture()")


class TimelineControlIntegrationTest(unittest.TestCase):
    def test_exported_root_drives_json_png_and_svg_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_path = root / "automatic_root.html"
            output_dir = root / "output"
            html_path.write_text(EXPORTED_ROOT_FIXTURE, encoding="utf-8")

            convert_animation(
                str(html_path),
                port=0,
                create_video=True,
                output_format="png",
                video_fps=2,
                output_dir=str(output_dir),
            )

            data = json.loads(
                (output_dir / "automatic_root_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(data["selected"]["tweens"])
            self.assertTrue(data["legacy"]["tweens"])
            self.assertEqual(len(data["selected"]["transformedPts"]), 3)
            self.assertEqual(len(data["legacy"]["transformedPts"]), 3)
            self.assertNotEqual(
                data["selected"]["transformedPts"][0],
                data["selected"]["transformedPts"][-1],
            )
            self.assertNotEqual(
                data["legacy"]["transformedPts"][0],
                data["legacy"]["transformedPts"][-1],
            )

            frames_dir = output_dir / "automatic_root_animation_2_png"
            self.assertEqual(len(list(frames_dir.glob("frame_*.png"))), 3)

            convert_animation(
                str(html_path),
                port=0,
                create_video=True,
                output_format="svg",
                video_fps=2,
                output_dir=str(output_dir),
            )
            svg_frames_dir = output_dir / "automatic_root_animation_2_svg"
            svg_frames = sorted(svg_frames_dir.glob("frame_*.svg"))
            self.assertEqual(len(svg_frames), 3)
            self.assertNotEqual(
                svg_frames[0].read_text(encoding="utf-8"),
                svg_frames[-1].read_text(encoding="utf-8"),
            )

    def test_infinite_root_requires_capture_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_path = root / "infinite_root.html"
            output_dir = root / "output"
            html_path.write_text(INFINITE_ROOT_FIXTURE, encoding="utf-8")

            with self.assertRaisesRegex(PlaywrightError, "capture_duration"):
                convert_animation(
                    str(html_path),
                    port=0,
                    output_dir=str(output_dir),
                    video_fps=2,
                )

            self.assertFalse(
                (output_dir / "infinite_root_data.json").exists()
            )

    def test_infinite_root_uses_explicit_duration_for_json_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            html_path = root / "infinite_root.html"
            output_dir = root / "output"
            html_path.write_text(INFINITE_ROOT_FIXTURE, encoding="utf-8")

            convert_animation(
                str(html_path),
                port=0,
                create_video=True,
                output_format="png",
                video_fps=2,
                output_dir=str(output_dir),
                capture_duration=1.5,
            )

            data = json.loads(
                (output_dir / "infinite_root_data.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(data["info"]["duration"], 1.5)
            self.assertEqual(data["info"]["steps"], 3)
            self.assertEqual(len(data["looping"]["transformedPts"]), 4)
            frames_dir = output_dir / "infinite_root_animation_2_png"
            self.assertEqual(len(list(frames_dir.glob("frame_*.png"))), 4)


if __name__ == "__main__":
    unittest.main()
