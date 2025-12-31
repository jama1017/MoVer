let tl_to_use = null

if (typeof tl !== 'undefined') {
    tl_to_use = tl
} else {
    tl_to_use = gsap.globalTimeline
}

tl_to_use.seek(0)
tl_to_use.pause()
tl_to_use.eventCallback("onUpdate", showFrame);

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