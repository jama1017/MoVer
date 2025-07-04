/**
 * This function adds a tween to the timeline to translates an SVG element.
 * @param {object} timeline - The timeline object to add the translation tween to.
 * @param {object} element - The SVG element to be translated.
 * @param {number} duration - The duration of the tween in seconds
 * @param {number} toX - The amount of pixels to translate the element along the x-axis from its current position. This value is a relative offset from the element's current x value.
 * @param {number} toY - The amount of pixels to translate the element along the y-axis from its current position. This value is a relative offset from the element's current y value.
 * @param {number} startTime - The absolute time in the global timeline at which the tween should start.
 * @param {string} [easing='none'] - The easing function to use for the tween. The default value is linear easing. The easing functions are: "power2", "power4", "expo", and "sine". Each function should be appended with ".in", ".out", or ".inOut" to specify how the rate of change should change over time. ".in" means a slow start and speeds up later. ".out" means a fast start and slows down at the end. ".inOut" means both a slow start and a slow ending. For example, "power2.in" specifies a quadratic easing in. Another easing function is "slow(0.1, 0.4)", which slows down in the middle and speeds up at both the beginning and the end. The first number (0-1) is the porportion of the tween that is slowed down, and the second number (0-1) is the easing strength.
 * @returns {void} - This function does not return anything.
 * @example
 * // Translates the square element 25 pixels to the right and 25 pixels down over 1 second.
 * translate(tl, square, 1, 25, 25, 0);
 * 
 * // Translates the square element 25 pixels to the right and 25 pixels down over 1 second, starting at 2 seconds into the timeline. The easing function used is power1.out.
 * translate(tl, square, 1, 25, 25, 2, 'power1.out');
 * 
 * // Translates the square to move in a square path counterclockwise. Each side of the square path is 100px wide and takes 1 second.
 * translate(tl, square, 1, -100, 0, 0);
 * translate(tl, square, 1, 0, 100, 1);
 * translate(tl, square, 1, 100, 0, 2);
 * translate(tl, square, 1, 0, -100, 3);
 */
function translate(timeline, element, duration, toX, toY, startTime, easing = 'none') {
    timeline.fromTo(
        element,
        {
            x: `+=${0}`,
            y: `+=${0}`,
        },
        {
            x: `+=${toX}`,
            y: `+=${toY}`,
            // transformOrigin: "50% 50%",
            duration: duration,
            ease: easing,
            immediateRender: false,
            data: { "type": "translate" }
        },
        startTime,
    );
}


/**
 * This function adds a tween to the timeline to scale an SVG element.
 * @param {object} timeline - The timeline object to add the translation tween to.
 * @param {object} element - The SVG element to be scaled.
 * @param {number} duration - The duration of the tween in seconds
 * @param {number} scaleX - The scale factor to apply to the element along the x-axis. This value is absolute and not relative to the element's current scaleX factor.
 * @param {number} scaleY - The scale factor to apply to the element along the y-axis. This value is absolute and not relative to the element's current scaleY factor.
 * @param {number} startTime - Same as the startTime parameter for the translate function.
 * @param {string} [elementTransformOriginX='50%'] - The x-axis transform origin from which the transformation is applied. The origin is in the element's coordinate space and is relative to the top left corner of the element. The default value is 50%, which means 50% of the element's width from the left edge of the element (horizontal center of the element).
 * @param {string} [elementTransformOriginY='50%'] - The y-axis transform origin from which the transformation is applied. The origin is in the element's coordinate space and is relative to the top left corner of the element. The default value is 50%, which means 50% of the element's height from the top edge of the element (vertical center of the element).
 * @param {string} [absoluteTransformOriginX=null] - Similar to elementTransformOriginX, but the origin is in the absolute coordinate space of the SVG document and should be specified as a pixel value. Specify only elementTransformOriginX and elementTransformOriginY or absoluteTransformOriginX and absoluteTransformOriginY, but not both. When both are specified, elementTransformOriginX and elementTransformOriginY take precedence.
 * @param {string} [absoluteTransformOriginY=null] - Similar to elementTransformOriginY, but the origin is in the absolute coordinate space of the SVG document and should be specified as a pixel value. Specify only elementTransformOriginX and elementTransformOriginY or absoluteTransformOriginX and absoluteTransformOriginY, but not both. When both are specified, elementTransformOriginX and elementTransformOriginY take precedence.
 * @param {string} [easing='none'] - Same as the easing parameter for the translate function.
 * @returns {void} - This function does not return anything.
 * @example
 * // Scale the square element to double its size over 1 second from its center.
 * scale(tl, square, 1, 2, 2, 0);
 * 
 * // Scale the square element to double along x-axis and triple along y-axis over 2 second from its center, starting at 2 seconds into the timeline. The easing function used is power1.out.
 * scale(tl, square, 1, 2, 3, 2, '50%', '50%', null, null, 'power1.out');
 * 
 * // Scale the square element to be 4 times as large from its bottom right corner over 1 second.
 * scale(tl, square, 1, 4, 4, 0, '100%', '100%');
 * 
 * // Scale the square element to be 5 times as large along the x-axis and 2 times as large along the y-axis from (100px, 100px) in the SVG document over 2 second.
 * scale(tl, square, 2, 5, 2, 0, null, null, 100, 100);
 * 
 * // Scale the square element to 0 in both x-axis and y-axis over 1 second from its center. Then, scale it back up to 1 in both x-axis and y-axis over 1 second from the point (300, 200).
 * scale(tl, square, 1, 0, 0, 0);
 * scale(tl, square, 1, 1, 1, 1, null, null, 300, 200);
 */
function scale(timeline, element, duration, scaleX, scaleY, startTime, elementTransformOriginX = '50%', elementTransformOriginY = '50%', absoluteTransformOriginX = null, absoluteTransformOriginY = null, easing = 'none') {

    let transformOrigin = `${elementTransformOriginX} ${elementTransformOriginY}`;
    let svgOrigin = `${absoluteTransformOriginX} ${absoluteTransformOriginY}`;

    if (elementTransformOriginX === null || elementTransformOriginY === null) {
        transformOrigin = undefined;
    }

    if (absoluteTransformOriginX === null || absoluteTransformOriginY === null) {
        svgOrigin = undefined;
    }

    timeline.to(
        element,
        {
            scaleX: scaleX,
            scaleY: scaleY,
            duration: duration,
            ease: easing,
            transformOrigin: transformOrigin,
            svgOrigin: svgOrigin,
            data: { "type": "scale" }
        },
        startTime
    );
}


/**
 * This function adds a tween to the timeline to rotate an SVG element.
 * @param {object} timeline - The timeline object to add the translation tween to.
 * @param {object} element - The SVG element to be rotated.
 * @param {number} duration - The duration of the tween in seconds
 * @param {number} angle - The rotation angle in degree. This value is absolute and not relative to the element's current rotation angle.
 * @param {number} startTime - Same as the startTime parameter for the translate function.
 * @param {string} [elementTransformOriginX='50%'] - Same as the elementTransformOriginX parameter for the scale function.
 * @param {string} [elementTransformOriginY='50%'] - Same as the elementTransformOriginY parameter for the scale function.
 * @param {string} [absoluteTransformOriginX=null] - Same as the absoluteTransformOriginX parameter for the scale function.
 * @param {string} [absoluteTransformOriginY=null] - Same as the absoluteTransformOriginY parameter for the scale function.
 * @param {string} [easing='none'] - Same as the easing parameter for the translate function.
 * @returns {void} - This function does not return anything.
 * @example
 * // Rotate the square element by 45 degrees (clockwise) from its center over 1 second.
 * rotate(tl, square, 1, 45, 0);
 * 
 * // Rotate the square element by -90 degrees (counterclockwise) from its bottom right corner over 1 second, starting at 1.5 seconds into the timeline. The easing function used is power1.out.
 * rotate(tl, square, 1, -90, 1.5, '100%', '100%', null, null, 'power1.out');
 * 
 * // Rotate the square element by 135 degrees (clockwise) from (300, 300) in the SVG document over 2 second.
 * rotate(tl, square, 2, 135, 0, null, null, 300, 300);
 */
function rotate(timeline, element, duration, angle, startTime, elementTransformOriginX = '50%', elementTransformOriginY = '50%', absoluteTransformOriginX = null, absoluteTransformOriginY = null, easing = 'none') {

    let transformOrigin = `${elementTransformOriginX} ${elementTransformOriginY}`;
    let svgOrigin = `${absoluteTransformOriginX} ${absoluteTransformOriginY}`;

    if (elementTransformOriginX === null || elementTransformOriginY === null) {
        transformOrigin = undefined;
    }

    if (absoluteTransformOriginX === null || absoluteTransformOriginY === null) {
        svgOrigin = undefined;
    }

    timeline.to(
        element,
        {
            rotation: angle,
            duration: duration,
            ease: easing,
            transformOrigin: transformOrigin,
            svgOrigin: svgOrigin,
            data: { "type": "rotate" }
        },
        startTime
    );
}

/**
 * This function returns the offset translation needed to align elementA to elementB.
 * Use this function if you want the same edges of the two elements to be aligned.
 * @param {object} elementA - The SVG element to be translated
 * @param {object} elementB - The target SVG element to be aligned to
 * @param {string="left", "horizontal-center", "right", "top", "vertical-center", "bottom"} alignTo - specifies how element A should be aligned to element B. "left" aligns the left edges of the two elements, "horizontal-center" aligns the horizontal center of the two elements, and "right" aligns the right edges of the two elements. "top" aligns the top edges of the two elements, "vertical-center" aligns the vertical centers of the two elements, and "bottom" aligns the bottom edges of the two elements.
 * @returns {object} - The offset translation {x: x-value, y: y-value} needed to align elementA to elementB
 * @example
 * // Align the square element to the left of the circle element.
 * const translation = computeTranslationForAlignTo(square, circle, 'left');
 * translate(tl, square, 1, translation.x, translation.y, 0);
 * 
 * // Move the ball so that the right edge of the ball aligns with the right edge of the board.
 * // Notice that here you should not use computeTranslationForAdjacentTo because same edges are being aligned.
 * const translation = computeTranslationForAlignTo(ball, board, 'right');
 * translate(tl, ball, 1, translation.x, translation.y, 0);
 */
function computeTranslationForAlignTo(elementA, elementB, alignTo) {
    let bbA = getAABB(elementA);
    let bbB = getAABB(elementB);

    switch (alignTo) {
        case 'left':
            return { x: bbB.left - bbA.left, y: 0 }

        case 'horizontal-center':
            return { x: (bbB.x + bbB.width / 2) - (bbA.x + bbA.width / 2), y: 0 }

        case 'right':
            return { x: bbB.right - bbA.right, y: 0 }

        case 'top':
            return { x: 0, y: bbB.top - bbA.top }

        case 'vertical-center':
            return { x: 0, y: (bbB.y + bbB.height / 2) - (bbA.y + bbA.height / 2) }

        case 'bottom':
            return { x: 0, y: bbB.bottom - bbA.bottom }
    }
}

/**
 * This function returns the offset translation needed to place elementA adjacent (touching but not overlapping) to elementB.
 * Use this function if you want the opposite edges of the two elements to be aligned.
 * @param {object} elementA - The SVG element to be translated
 * @param {object} elementB - The target SVG element to be adjacent to
 * @param {string="left", "right", "top", "bottom"} adjacentTo - specifies how element A should be adjacent to element B. "left" places element A to the immediate left of element B, "right" places element A to the immediate right element B, "top" places element A right above element B, and "bottom" places element right below element B.
 * @returns {object} - The offset translation {x: x-value, y: y-value} needed to place elementA adjacent to elementB
 * @example
 * // Animate the ball to move to the top of the board so that the bottom of the ball touches the top of the board.
 * const translation = computeTranslationForAdjacentTo(ball, board, 'top');
 * translate(tl, ball, 1, translation.x, translation.y, 0);
 * 
 * // Move the ball so that the left edge of the ball aligns with the right edge of the board.
 * // Notice that here you should not use computeTranslationForAlignTo because different edges are being aligned.
 * const translation = computeTranslationForAdjacentTo(ball, board, 'right');
 * translate(tl, ball, 1, translation.x, translation.y, 0);
 */
function computeTranslationForAdjacentTo(elementA, elementB, adjacentTo) {
    let bbA = getAABB(elementA);
    let bbB = getAABB(elementB);

    switch (adjacentTo) {
        case 'left':
            return { x: bbB.left - bbA.right, y: 0 }
        case 'right':
            return { x: bbB.right - bbA.left, y: 0 }
        case 'top':
            return { x: 0, y: bbB.top - bbA.bottom }
        case 'bottom':
            return { x: 0, y: bbB.bottom - bbA.top }
    }
}

/**
 * Creates a timeline object that stores and sequences all the animation tweens.
 * @returns {object} The created timeline.
 */
function createTimeline() {
    return gsap.timeline();
}

/**
 * Retrieves the value of a specified property for a given SVG element. Do not use this function to get the size and position of an element. Use getSize and getCenterPosition instead.
 * @param {string} elementID - The ID of the SVG element.
 * @param {string} property - The name of the property to retrieve.
 * @returns {*} The value of the specified property.
 */
function getProperty(elementID, property) {
    return gsap.getProperty(elementID, property);
}

/**
 * This function returns the x and y coordinates of the center of an SVG element in pixels.
 * @param {object} element - The SVG element.
 * @returns {object} - The x and y coordinates of the center of the element in pixels.
 * @example
 * // Get the y coordinate of the center of the square element.
 * const squareY = getCenterPosition(square).y;
 */
function getCenterPosition(element) {
    const rect = element.getBoundingClientRect();
    const svgRef = element.ownerSVGElement;
    return {
        x: rect.x + rect.width / 2 - svgRef.getBoundingClientRect().x,
        y: rect.y + rect.height / 2 - svgRef.getBoundingClientRect().y,
    };
}

/**
 * This function returns the width and height of an SVG element in pixels.
 * @param {object} element - The SVG element.
 * @returns {object} - The width and height the element in pixels.
 * @example
 * // Get the width of the square element.
 * const squareWidth = getSize(square).width;
 * 
 * // Get the radius of the circle element (don't use getProperty() to get the radius).
 * const circleRadius = getSize(circle).width / 2; // Can also use height instead of width.
 */
function getSize(element) {
    const rect = element.getBoundingClientRect();
    return {
        width: rect.width,
        height: rect.height,
    };
}

/**
 * This function returns the axis-aligned bounding box of an SVG element
 * @param {object} element - The SVG element
 * @returns {object} - The bounding box of the element
*/
function getAABB(element) {
    let rect = element.getBoundingClientRect();
    rect.y = rect.y + window.scrollY;
    rect.top = rect.top + window.scrollY;
    return rect;
}