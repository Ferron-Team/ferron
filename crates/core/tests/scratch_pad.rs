use orrin_core::scene::input::{
    Actions, Gamepads, InputState, config, keys::PadAxis, keys::PadButton,
};
use std::path::Path;

#[test]
fn scratch() {
    let mut gamepads = match Gamepads::new() {
        Ok(g) => g,
        Err(e) => {
            println!("no gamepads: {e}");
            return;
        }
    };
    let specs = config::parse(
        r#"
format_version = 1
[actions]
Jump = ["Space", "Gamepad.South"]
Fire = ["Gamepad.RightTrigger"]
[axes.LookX]
source = "Gamepad.RightStick.X"
deadzone = 0.15
[axes.MoveY]
source = "Gamepad.LeftStick.Y"
"#,
        Path::new("input.toml"),
    )
    .expect("parses");

    let mut actions = Actions::new();
    actions.apply(specs);
    let jump = actions.id("Jump");
    let fire = actions.id("Fire");
    let lookx = actions.id("LookX");
    let movey = actions.id("MoveY");

    let mut input = InputState::new();
    println!("press buttons / move sticks for 12 seconds...");
    let start = std::time::Instant::now();
    let mut last = String::new();
    while start.elapsed().as_secs() < 12 {
        gamepads.pump(&mut input, true);
        actions.resolve(&input);
        if actions.pressed(jump, 0) {
            println!("  Jump pressed");
        }
        if actions.released(jump, 0) {
            println!("  Jump released");
        }
        if actions.pressed(fire, 0) {
            println!("  Fire pressed (trigger past threshold)");
        }
        let now = format!(
            "connected={} lookx={:+.2} movey={:+.2} jump={} fire={}",
            input.pad_connected(0),
            actions.axis(lookx, 0),
            actions.axis(movey, 0),
            actions.held(jump, 0),
            actions.held(fire, 0),
        );
        if now != last {
            println!("  {now}");
            last = now;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    println!(
        "raw resting: LStickX={:.4} LStickY={:.4} RStickX={:.4} LTrigger={:.4}",
        input.pad_axis(0, PadAxis::LeftStickX),
        input.pad_axis(0, PadAxis::LeftStickY),
        input.pad_axis(0, PadAxis::RightStickX),
        input.pad_axis(0, PadAxis::LeftTrigger)
    );

    // Focus loss must zero the pad rather than leave a stick pushed.
    gamepads.pump(&mut input, false);
    actions.resolve(&input);
    println!(
        "unfocused: lookx={:.3} south={} still connected={}",
        actions.axis(lookx, 0),
        input.pad_button_down(0, PadButton::South),
        input.pad_connected(0)
    );

    // Player 1 has no pad, so every slot-1 query is quiet.
    println!(
        "player1: jump={} lookx={:.3}",
        actions.held(jump, 1),
        actions.axis(lookx, 1)
    );
}
