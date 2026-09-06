//! The input vocabulary: one table naming every source a binding can name.
//!
//! The engine's key codes are its own numbering rather than winit's, because
//! winit's discriminants are not a stable ABI and these numbers cross two seams
//! that outlive any dependency — binding files users commit, and the C#
//! `KeyCode` enum. This table is where the three meet: the name a config
//! spells, the code stored and sent across the scripting boundary, and the
//! winit key it arrives as. Adding a key is a row here plus the matching C#
//! variant, and nothing else needs to know.
//!
//! The names are the engine's own. A binding file is a document, and a
//! dependency's rename must not invalidate one — so where the C# enum already
//! spelled a key, that spelling wins and there is exactly one vocabulary either
//! side of the boundary. Aliases cover the spellings people reach for anyway.
//!
//! Keyboard names are bare (`A`, `Space`); every other device is prefixed
//! (`Mouse.Left`, `Gamepad.South`). That is what lets `Gamepad.A` exist
//! alongside the `A` key without either needing a longer name.

use winit::keyboard::KeyCode;

/// One keyboard key: what a config calls it, what the engine stores, and which
/// winit key produces it.
pub struct Key {
    pub name: &'static str,
    pub code: u32,
    pub aliases: &'static [&'static str],
    pub winit: KeyCode,
}

const fn key(name: &'static str, code: u32, winit: KeyCode) -> Key {
    Key {
        name,
        code,
        aliases: &[],
        winit,
    }
}

const fn aliased(
    name: &'static str,
    code: u32,
    winit: KeyCode,
    aliases: &'static [&'static str],
) -> Key {
    Key {
        name,
        code,
        aliases,
        winit,
    }
}

/// Every keyboard key the engine knows. Codes are frozen: they are what C#
/// sends across the scripting boundary and what older binding files contain.
pub const KEYS: &[Key] = &[
    key("A", 1, KeyCode::KeyA),
    key("B", 2, KeyCode::KeyB),
    key("C", 3, KeyCode::KeyC),
    key("D", 4, KeyCode::KeyD),
    key("E", 5, KeyCode::KeyE),
    key("F", 6, KeyCode::KeyF),
    key("G", 7, KeyCode::KeyG),
    key("H", 8, KeyCode::KeyH),
    key("I", 9, KeyCode::KeyI),
    key("J", 10, KeyCode::KeyJ),
    key("K", 11, KeyCode::KeyK),
    key("L", 12, KeyCode::KeyL),
    key("M", 13, KeyCode::KeyM),
    key("N", 14, KeyCode::KeyN),
    key("O", 15, KeyCode::KeyO),
    key("P", 16, KeyCode::KeyP),
    key("Q", 17, KeyCode::KeyQ),
    key("R", 18, KeyCode::KeyR),
    key("S", 19, KeyCode::KeyS),
    key("T", 20, KeyCode::KeyT),
    key("U", 21, KeyCode::KeyU),
    key("V", 22, KeyCode::KeyV),
    key("W", 23, KeyCode::KeyW),
    key("X", 24, KeyCode::KeyX),
    key("Y", 25, KeyCode::KeyY),
    key("Z", 26, KeyCode::KeyZ),
    aliased("Alpha0", 30, KeyCode::Digit0, &["0"]),
    aliased("Alpha1", 31, KeyCode::Digit1, &["1"]),
    aliased("Alpha2", 32, KeyCode::Digit2, &["2"]),
    aliased("Alpha3", 33, KeyCode::Digit3, &["3"]),
    aliased("Alpha4", 34, KeyCode::Digit4, &["4"]),
    aliased("Alpha5", 35, KeyCode::Digit5, &["5"]),
    aliased("Alpha6", 36, KeyCode::Digit6, &["6"]),
    aliased("Alpha7", 37, KeyCode::Digit7, &["7"]),
    aliased("Alpha8", 38, KeyCode::Digit8, &["8"]),
    aliased("Alpha9", 39, KeyCode::Digit9, &["9"]),
    key("LeftArrow", 40, KeyCode::ArrowLeft),
    key("RightArrow", 41, KeyCode::ArrowRight),
    key("UpArrow", 42, KeyCode::ArrowUp),
    key("DownArrow", 43, KeyCode::ArrowDown),
    key("Space", 44, KeyCode::Space),
    aliased("Return", 45, KeyCode::Enter, &["Enter"]),
    aliased("Escape", 46, KeyCode::Escape, &["Esc"]),
    key("Tab", 47, KeyCode::Tab),
    key("Backspace", 48, KeyCode::Backspace),
    aliased("LeftShift", 49, KeyCode::ShiftLeft, &["Shift"]),
    key("RightShift", 50, KeyCode::ShiftRight),
    aliased(
        "LeftControl",
        51,
        KeyCode::ControlLeft,
        &["Control", "Ctrl"],
    ),
    key("RightControl", 52, KeyCode::ControlRight),
    aliased("LeftAlt", 53, KeyCode::AltLeft, &["Alt"]),
    key("RightAlt", 54, KeyCode::AltRight),
];

/// Mouse buttons, by the number [`super::InputState`] and the C# `MouseButton`
/// enum both address them with. An index, not a mask bit — the mask is private
/// to `state`, which shifts by these.
pub const MOUSE_BUTTONS: &[(&str, u8)] = &[("Left", 0), ("Right", 1), ("Middle", 2)];

/// Mouse movement, in window pixels accumulated over the frame.
pub const MOUSE_AXES: &[(&str, MouseAxis)] = &[("X", MouseAxis::X), ("Y", MouseAxis::Y)];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseAxis {
    X,
    Y,
}

/// Face and shoulder buttons, named by position rather than by legend: `South`
/// is the same physical button whether its cap reads A or ✕. The vendor letters
/// are aliases so a config may still spell what is printed on the pad.
pub const PAD_BUTTONS: &[(&str, PadButton, &[&str])] = &[
    ("South", PadButton::South, &["A", "Cross"]),
    ("East", PadButton::East, &["B", "Circle"]),
    ("West", PadButton::West, &["X", "Square"]),
    ("North", PadButton::North, &["Y", "Triangle"]),
    ("LeftBumper", PadButton::LeftBumper, &["LB", "L1"]),
    ("RightBumper", PadButton::RightBumper, &["RB", "R1"]),
    ("LeftStick", PadButton::LeftStick, &["L3"]),
    ("RightStick", PadButton::RightStick, &["R3"]),
    ("DPadUp", PadButton::DPadUp, &[]),
    ("DPadDown", PadButton::DPadDown, &[]),
    ("DPadLeft", PadButton::DPadLeft, &[]),
    ("DPadRight", PadButton::DPadRight, &[]),
    ("Start", PadButton::Start, &[]),
    ("Select", PadButton::Select, &["Back"]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadButton {
    South,
    East,
    West,
    North,
    LeftBumper,
    RightBumper,
    LeftStick,
    RightStick,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    Start,
    Select,
}

impl PadButton {
    /// Bit position in [`super::InputState`]'s per-pad button mask.
    pub fn index(self) -> u32 {
        self as u32
    }
}

/// Analog pad sources. Triggers are axes rather than buttons; an action that
/// binds one compares it against a threshold.
pub const PAD_AXES: &[(&str, PadAxis)] = &[
    ("LeftStick.X", PadAxis::LeftStickX),
    ("LeftStick.Y", PadAxis::LeftStickY),
    ("RightStick.X", PadAxis::RightStickX),
    ("RightStick.Y", PadAxis::RightStickY),
    ("LeftTrigger", PadAxis::LeftTrigger),
    ("RightTrigger", PadAxis::RightTrigger),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

impl PadAxis {
    /// How many analog sources a pad has, and so how wide its axis store is.
    pub const COUNT: usize = 6;

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Look a keyboard name up, canonical spelling or alias.
///
/// Matching ignores case: this table is read from a file someone typed, and
/// refusing `space` teaches nothing that accepting it does not. Errors quote the
/// canonical spelling, so the file still converges on one form.
pub fn key_by_name(name: &str) -> Option<u32> {
    KEYS.iter()
        .find(|key| {
            key.name.eq_ignore_ascii_case(name)
                || key
                    .aliases
                    .iter()
                    .any(|alias| alias.eq_ignore_ascii_case(name))
        })
        .map(|key| key.code)
}

/// The engine code for a winit key, or `None` for one the engine has no name
/// for. Linear over a few dozen rows, on an event that arrives at human speed.
pub fn key_from_winit(code: KeyCode) -> Option<u32> {
    KEYS.iter()
        .find(|key| key.winit == code)
        .map(|key| key.code)
}
