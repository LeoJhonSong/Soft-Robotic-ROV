function setup() {
    createCanvas(windowWidth, windowHeight);
    gui = createGui();
    size = 0.15
    release_arm = createButton("Arm Release", (0.05 + size / 2) * windowWidth - 75, 0.1 * windowHeight - 55, 150, 50);
    reset_auv = createButton("AUV Reset", (0.05 + size / 2) * windowWidth - 75, 0.1 * windowHeight, 150, 50);
    steer_auv = createJoystick("AUV steer", (0.05 - 0.2 * size) * windowWidth, windowHeight - (size + 0.05) * windowWidth - 0.4 * windowHeight, 1.4 * size * windowWidth, 32);
    xy_arm = createSlider2d("Arm xy", 0.05 * windowWidth, 0.85 * windowHeight - 2 * size * windowWidth, size * windowWidth, size * windowWidth);
    vxy_auv = createJoystick("AUV Vxy", 0.05 * windowWidth, 0.85 * windowHeight - 0.9 * size * windowWidth, size * windowWidth, size * windowWidth);
    open_hand = createToggle("Open", 0.95 * windowWidth - 150 + 2.5, 0.18 * windowHeight, 70, 50);
    close_hand = createToggle("Close", 0.95 * windowWidth - 150 + 77.5, 0.18 * windowHeight, 70, 50);
    reset_arm = createButton("Arm Reset", 0.95 * windowWidth - 150, 0.26 * windowHeight, 150, 50);
    collect_arm = createButton("Collect", 0.95 * windowWidth - 150, 0.34 * windowHeight, 150, 50);
    fold_arm = createToggle("Fold", 0.95 * windowWidth - 150, 0.42 * windowHeight, 150, 50);
    vz_auv = createCrossfaderV("AUV Vz", 0.95 * windowWidth - 150 + 21.5, 0.65 * windowHeight, 32, 0.2 * windowHeight);
    elg_arm = createSliderV("Arm elongation", 0.95 * windowWidth - 150 + 96.5, 0.65 * windowHeight, 32, 0.2 * windowHeight);
    vxy_auv.setStyle("rounding", size * windowWidth);
    vz_auv.setStyle("strokeCenter", color("#ff0000"));
    elg_arm.setStyle("fillTrack", color("#0550ae"));
    elg_arm.val = 1;
}

function draw() {
    drawGui();
    hand_status = 'idle';

    if (reset_auv.isPressed) {
        vz_auv.val = 0;
        vz_auv.isChanged = true;
        steer_auv.valX = 0;
        steer_auv.valY = 0;
        steer_auv.isChanged = true;
        vxy_auv.valX = 0;
        vxy_auv.valY = 0;
        vxy_auv.isChanged = true;
    }
    if (steer_auv.isChanged || vz_auv.isChanged || vxy_auv.isChanged) {
        print("AUV movement: " + vxy_auv.valX + ", " + vxy_auv.valY + ", " + steer_auv.valX + ", " + vz_auv.val);
        post('/auv/joystick', {
            'vx': vxy_auv.valY,
            'vy': -vxy_auv.valX,
            'steer': steer_auv.valX,
            'vz': vz_auv.val
        });
    }

    if (release_arm.isPressed) {
        elg_arm.val = 1;
        xy_arm.valX = 0;
        xy_arm.valY = 0;
        open_hand.val = false;
        close_hand.val = false;
        fold_arm.val = false;
        post('/arm/release', {});
    }
    if (open_hand.isPressed) {
        if (close_hand.val) {
            close_hand.val = false;
        }
    }
    if (close_hand.isPressed) {
        if (open_hand.val) {
            open_hand.val = false;
        }
    }
    if (reset_arm.isPressed) {
        elg_arm.val = 1;
        xy_arm.valX = 0;
        xy_arm.valY = 0;
        post('/arm/reset', {});
    }
    if (collect_arm.isPressed) {
        elg_arm.val = 1;
        xy_arm.valX = 0;
        xy_arm.valY = 0;
        fold_arm.val = false;
        post('/arm/collect', {});
    }
    if (fold_arm.isPressed){
        elg_arm.val = 1;
        elg_arm.isChanged = true;
        post('/arm/fold', {'state': fold_arm.val});
    }
    if (open_hand.isPressed || close_hand.isPressed  || elg_arm.isChanged || xy_arm.isChanged) {
        if (open_hand.val) {
            hand_status = 'open';
        }
        else if (close_hand.val) {
            hand_status = 'close';
        }
        print('[Arm] x: ' + -xy_arm.valX + ', y: ' + xy_arm.valY)
        post('/arm/joystick', {
            'x': -xy_arm.valX,
            'y': xy_arm.valY,
            'elg': 1 - elg_arm.val,
            'hand': hand_status
        });
        // print(xy_arm.label + " = {" + xy_arm.valX + ", " + xy_arm.valY + "}");
    }

}

// Add these lines below sketch to prevent scrolling and zooming on mobile
function touchMoved() {
    return false;
}