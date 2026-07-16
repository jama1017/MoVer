let tl_to_use = null

// Freeze and snapshot authored roots before the later converter script loads.
// Reparenting happens once in initializeTimelineControl(), where the complete
// recorded-root policy is available.
if (
    typeof GSDevTools !== "undefined"
    && typeof GSDevTools.getByAnimation === "function"
) {
    const moverDevToolsAnimations = new Set(
        gsap.globalTimeline.getChildren(true, true, true)
    )
    if (typeof tl !== "undefined" && tl) {
        moverDevToolsAnimations.add(tl)
    }
    moverDevToolsAnimations.forEach(animation => {
        const instance = GSDevTools.getByAnimation(animation)
        if (instance && typeof instance.kill === "function") {
            instance.kill()
        }
    })
}
const moverInitialRootSnapshot = gsap.globalTimeline
    .getChildren(false, true, true)
    .map(animation => ({
        animation,
        delay: (
            typeof animation.delay === "function"
        ) ? animation.delay() : 0,
        startTime: animation.startTime(),
    }))
gsap.globalTimeline.pause()
// tl_to_use.eventCallback("onUpdate", showFrame);

// Use duration() instead of totalDuration() to handle infinite repeats (repeat(-1))
let totalSteps = 0;
let moverVisualizationFrame = null;
let moverVisualizationPlaybackToken = 0;

// create a new p element and append it after the element with id="prompt"
let frameCount = document.createElement("p");
let frameNum = 0;
frameCount.textContent = `frame: 0 / ${totalSteps}`;
document.getElementById("prompt").after(frameCount);

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