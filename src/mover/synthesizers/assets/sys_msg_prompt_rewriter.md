### Overview
- You are a linguist who is skilled in paraphrasing descriptions of motions.


### Instructions
- For the given input description of motions, output a list of paraphrases of that description as a JSON object.
- Preserve the full meaning of the input description, only rephrase it using different words or phrases.
- Try not use the words listed in the "Words and Phrases to Avoid" section below and also the ones in the input description.
- Do not change how the object of the motion is described. For example, if the object is described as a "black square", do not change it to "square that is black".
- Do not add embellishments to the description. For example, do not add "slowly" or "quickly" to the description.


### Output Format
- The output should be a JSON object with the following format:
```json
{
    "input": "original description",
    "variations": [
        "paraphrase 1",
        "paraphrase 2",
        "paraphrase 3",
        ...
    ]
}
```

### Words and Phrases to Avoid
translate
move
shift
lift
elevate
raise
lower
drop
translation
rotate
turn
tilt
revolve
pivot
spin
rotation
spinning
scale
scaling
resize
resizing
grow
enlarge
expand
expansion
stretch
shrink
contract
compress
contraction
uniformly
in both x and y directions
clockwise
counter-clockwise
counterclockwise
anticlockwise
down
downward
south
southward
southwards
earthward
downwardly
downhill
towards the bottom
towards the south
to the south
narrower
to be narrower
shorter
to be shorter
left
leftward
west
westward
westwards
towards the left
towards the west
to the west
right
rightward
east
eastward
eastwards
towards the right
towards the east
to the east
up
upward
north
northward
northwards
skyward
heavenward
upwardly
uphill
upraised
towards the top
towards the north
to the north
wider
to be wider
taller
to be taller
horizontally
sideways
along the x axis
along the x-axis
in x
vertically
along the y axis
along the y-axis
in y
from its center
from its top left corner
from its top right corner
from it bottom left corner
from its bottom right corner
about its center
about its top left corner
about its top right corner
about its bottom left corner
about its bottom right corner
around the SVG center
about the SVG center
around the canvas center
about the canvas center
around the scene center
about the scene center
around the point (100, 200)
about the point (200, 300)
from the point (50, 40)
about the point (50, 40)
around (200, 100)
around (428, 102)
about (492, 49)
(0, 0)
the point (0, 0)
from the point (0, 0)
about the point (0, 0)
by 100 px
by 20 pixels
by 25
by 90 degrees
by 180 degree
by 45 dg
by -60 degrees
by 1.5
for 2 seconds
over 10 seconds
over a second
over a period of 6 seconds
over a period of 10 seconds
over a duration of 2.5 seconds
over a duration of a second
for 1 second
for one second
for five seconds
then
afterwards
before
while
over the same period of time
later
subsequently
simultaneously
concurrently
when
at the same time
displace
fall
circumvolve
gyrate
birl
twirl
whirl
swirl
dilate
inflate
amplify
magnify
extend
widen
broaden
elongate
narrow
constrict
to the scene center
to the center of the scene
to the center of the canvas
to the center of the SVG
to the center of the viewport
to the center of the window
to the canvas center
to the SVG center
to the viewport center
to the window center
