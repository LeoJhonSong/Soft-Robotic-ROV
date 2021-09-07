function setup() {
    createCanvas(windowWidth, windowHeight);
    gui = createGui();
    size = 0.15
    b_auv = createButton("AUV Reset", 0.05 * windowWidth, 0.1 * windowHeight, 150, 50);
    c_auv = createCrossfaderV("AUV Vz", 0.05 * windowWidth, windowHeight - (size + 0.05) * windowWidth - 0.25 * windowHeight, 32, 0.2 * windowHeight);
    j_auv = createJoystick("AUV Vxy", 0.05 * windowWidth, windowHeight - (size + 0.05) * windowWidth, size * windowWidth, size * windowWidth);
    j_auv.setStyle("rounding", size * windowWidth);
    b_arm = createButton("Arm Reset", (0.95 - size) * windowWidth, 0.1 * windowHeight, 150, 50);
    c_arm = createSliderV("Arm elongation", windowWidth - 50, windowHeight - (size + 0.05) * windowWidth - 0.25 * windowHeight, 32, 0.2 * windowHeight);
    c_arm.val = 1;
    s2_arm = createSlider2d("Arm xy", (0.95 - size) * windowWidth, windowHeight - (size + 0.05) * windowWidth, size * windowWidth, size * windowWidth);

    x_arm = 0;
    y_arm = 0;
}

function draw() {
    drawGui();

    if (b_auv.isPressed) {
        // post('/resets/auv', {});
        c_auv.val = 0;
        c_auv.isChanged = true;
        j_auv.valX = 0;
        j_auv.valY = 0;
    }
    if (c_auv.isChanged || j_auv.isChanged) {
        print(j_auv.label + " = {" + j_auv.valX + ", " + j_auv.valY + "}" + c_arm.label + ': ' + c_arm.val);
        post('/auv/joystick', {
            'vx': j_auv.valY,
            'vy': j_auv.valX,
            'vz': c_auv.val
        });
    }

    if (b_arm.isPressed) {
        c_arm.val = 1;
        s2_arm.valX = 0;
        s2_arm.valY = 0;
        post('/arm/reset', {});
    }
    if (s2_arm.isChanged) {
        // Print a message when Slider 1 is changed
        // that displays its value.
        post('/arm/joystick', {
            'x': 150 * s2_arm.valX,
            'y': 150 * s2_arm.valY
        });
        print(s2_arm.label + " = {" + s2_arm.valX + ", " + s2_arm.valY + "}");
    }

}

// Add these lines below sketch to prevent scrolling and zooming on mobile
function touchMoved() {
    return false;
}