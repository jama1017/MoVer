let tl_to_use = null

if (typeof tl !== 'undefined') {
    tl_to_use = tl
} else {
    tl_to_use = gsap.globalTimeline
}

tl_to_use.seek(0)
tl_to_use.pause()
tl_to_use.eventCallback("onUpdate", showFrame);

// create a new p element and append it after the element with id="prompt"
let frameCount = document.createElement("p");
let frameNum = 0;
frameCount.textContent = `frame: 0 / ${tl_to_use.totalDuration() * 60}`;
document.getElementById("prompt").after(frameCount);

function showFrame() {
    frameNum = Math.floor(tl_to_use.time() * 60);
    frameCount.textContent = `frame: ${frameNum} / ${tl_to_use.totalDuration() * 60}`;
}

function play() {
    tl_to_use.seek(0)
    tl_to_use.play()
}

async function pause() {
    tl_to_use.pause()
}