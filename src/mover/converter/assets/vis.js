let tl_to_use = null

// Retain every root animation that exists when MoVer initializes. A legacy
// `tl` is treated exactly like every sibling timeline or standalone tween.
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
gsap.globalTimeline.pause()
tl_to_use = gsap.exportRoot({
    paused: true,
    smoothChildTiming: false,
}, false)
const moverRootChildren = tl_to_use
    .getChildren(false, true, true)
    .filter(child => {
        const targets = (
            typeof child.targets === "function"
        ) ? child.targets() : []
        const isImmediateSetupTween = (
            typeof child.duration === "function"
            && child.duration() === 0
            && typeof child.delay === "function"
            && child.delay() === 0
            && targets.length > 0
            && targets.every(target => typeof target !== "function")
        )
        if (isImmediateSetupTween) {
            tl_to_use.remove(child)
            return false
        }
        return true
    })
tl_to_use.getChildren(true, true, true).forEach(child => {
    const delay = typeof child.delay === "function" ? child.delay() : 0
    if (moverRootChildren.includes(child)) {
        child.startTime(delay)
    }
    if (typeof child.paused === "function" && child.paused()) {
        child.paused(false)
    }
})
tl_to_use.seek(0, false).pause();
// tl_to_use.eventCallback("onUpdate", showFrame);

// Use duration() instead of totalDuration() to handle infinite repeats (repeat(-1))
const totalSteps = Math.ceil(tl_to_use.duration() * 60);

// create a new p element and append it after the element with id="prompt"
let frameCount = document.createElement("p");
let frameNum = 0;
frameCount.textContent = `frame: 0 / ${totalSteps}`;
document.getElementById("prompt").after(frameCount);

function showFrame() {
    frameNum = Math.floor(tl_to_use.time() * 60);
    frameCount.textContent = `frame: ${frameNum} / ${totalSteps}`;
}

function play() {
    tl_to_use.seek(0)
    tl_to_use.play()
}

async function pause() {
    tl_to_use.pause()
}