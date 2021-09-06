function setup() {
    createCanvas(windowWidth, windowHeight);
    gui = createGui();
    size = 0.15
    b_rov = createButton("ROV Reset", 0.05 * windowWidth, 0.1 * windowHeight, 150, 50);
    j_rov = createJoystick("ROV", 0.05 * windowWidth, windowHeight - (size + 0.05) * windowWidth, size * windowWidth, size * windowWidth);
    j_rov.setStyle("rounding", size * windowWidth);
    c_rov = createCrossfaderV("ROV Vz", 0.05 * windowWidth, windowHeight - (size + 0.05) * windowWidth - 0.25 * windowHeight, 32, 0.2 * windowHeight);
    b_arm = createButton("Arm Reset", (0.95 - size) * windowWidth, 0.1 * windowHeight, 150, 50);
    s_arm = createSlider2d("Arm", (0.95 - size) * windowWidth, windowHeight - (size + 0.05) * windowWidth, size * windowWidth, size * windowWidth);
    c_arm = createSliderV("Arm elongation", windowWidth - 50, windowHeight - (size + 0.05) * windowWidth - 0.25 * windowHeight, 32, 0.2 * windowHeight);
    c_arm.val = 1;

    x_arm = 0;
    y_arm = 0;
}

function draw() {
    drawGui();

    if (b_rov.isPressed) {
        // post('/resets/rov', {});
        c_rov.val = 0;
        j_rov.valX = 0;
        j_rov.valY = 0;
    }
    if (j_rov.isChanged) {
        print(j_rov.label + " = {" + j_rov.valX + ", " + j_rov.valY + "}");
        post('/joysticks/rov', {
            'vx': j_rov.valX,
            'vy': j_rov.valY
        });
    }

    if (b_arm.isPressed) {
        c_arm.val = 1;
        s_arm.valX = 0;
        s_arm.valY = 0;
        post('/resets/arm', {});
    }
    if (s_arm.isChanged) {
        // Print a message when Slider 1 is changed
        // that displays its value.
        post('/joysticks/arm', {
            'x': 150 * s_arm.valX,
            'y': 150 * s_arm.valY
        });
        print(s_arm.label + " = {" + s_arm.valX + ", " + s_arm.valY + "}");
    }

}

// Add these lines below sketch to prevent scrolling and zooming on mobile
function touchMoved() {
    return false;
}