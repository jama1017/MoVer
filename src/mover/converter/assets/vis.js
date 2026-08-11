let tl_to_use = null

// Use duration() instead of totalDuration() to handle infinite repeats (repeat(-1))
let totalSteps = 0;
let moverVisualizationFrame = null;
let moverVisualizationPlaybackToken = 0;

// create a new p element and append it after the element with id="prompt"
let frameCount = document.createElement("p");
let frameNum = 0;
frameCount.textContent = `frame: 0 / ${totalSteps}`;

// Preview utility only: play/pause and frame count. Capture must not depend on this
// file being loaded — freezing the authored root, disabling GSDevTools, and collecting
// the recorded root all live in convert.js, which pages may embed without vis.js.
// Exception: this file loads before convert.js and the root advances across the gap.
if (typeof gsap !== "undefined" && gsap.globalTimeline) {
    gsap.globalTimeline.pause()
}
// tl_to_use.eventCallback("onUpdate", showFrame);

document.getElementById("prompt")?.after(frameCount);

function refreshTimelineVisualization() {
    if (!tl_to_use) return false
    totalSteps = Math.ceil(tl_to_use.duration() * 60)
    showFrame()
    return true
}

function getTimelineForVisualization() {
    if (!tl_to_use && typeof initializeTimelineControl === "function") {
        initializeTimelineControl()
    }
    if (!tl_to_use) {
        throw new Error("MoVer timeline control is not initialized")
    }
    refreshTimelineVisualization()
    return tl_to_use
}

function showFrame() {
    if (!tl_to_use) return;
    frameNum = Math.floor(tl_to_use.time() * 60);
    frameCount.textContent = `frame: ${frameNum} / ${totalSteps}`;
}

function stopTimelineVisualizationPlayback() {
    moverVisualizationPlaybackToken++
    if (moverVisualizationFrame !== null) {
        cancelAnimationFrame(moverVisualizationFrame)
        moverVisualizationFrame = null
    }
    if (tl_to_use) tl_to_use.pause()
    return true
}

function play() {
    const timeline = getTimelineForVisualization()
    stopTimelineVisualizationPlayback()
    timeline.totalTime(0, false).pause()
    const startedAt = performance.now()
    const playbackToken = ++moverVisualizationPlaybackToken
    const render = now => {
        if (
            tl_to_use !== timeline
            || moverVisualizationPlaybackToken !== playbackToken
        ) {
            moverVisualizationFrame = null
            return
        }
        const elapsed = Math.max(0, (now - startedAt) / 1000)
        const duration = timeline.totalDuration()
        timeline.totalTime(Math.min(elapsed, duration), false)
        showFrame()
        const updatedDuration = timeline.totalDuration()
        if (
            tl_to_use !== timeline
            || moverVisualizationPlaybackToken !== playbackToken
        ) {
            moverVisualizationFrame = null
        } else if (elapsed < updatedDuration) {
            moverVisualizationFrame = requestAnimationFrame(render)
        } else {
            moverVisualizationFrame = null
        }
    }
    moverVisualizationFrame = requestAnimationFrame(render)
}

async function pause() {
    stopTimelineVisualizationPlayback()
}