//! The gamepad backend, and the only module that knows gilrs exists.
//!
//! Pads are polled, not evented: winit never delivers one, so gilrs is pumped
//! once a frame and the pad's *level* state is sampled into
//! [`InputState`](super::InputState) afterwards. Sampling rather than
//! translating each button event is deliberate — actions derive their own edges
//! from level state, so an event stream would be converted back into levels
//! anyway, and a pad that connects mid-frame reports its resting state
//! correctly with no replay.
//!
//! **Players are slots, not devices.** gilrs hands out a fresh `GamepadId` on
//! every connect, so a pad that is unplugged and plugged back in is a different
//! id and would otherwise be a different player. A slot is claimed on connect,
//! released on disconnect, and the lowest free one wins — which is what makes
//! "player 2" mean the same person across a flat battery.
//!
//! **Stick Y is flipped.** gilrs reports a stick pushed forward as positive;
//! winit reports the mouse moved down as positive. A `LookY` axis bound to
//! either should turn the same way, so the pad is converted to the screen's
//! convention here rather than leaving every game to discover the difference.

use gilrs::{Axis, Button, GamepadId, Gilrs};

use super::keys::{PadAxis, PadButton};
use super::{InputState, MAX_PLAYERS};

/// Our buttons, and the gilrs button each samples. gilrs calls the shoulder
/// buttons `LeftTrigger` and the analog triggers `LeftTrigger2`; the triggers
/// are read as axes below, so only the shoulders appear here.
const BUTTONS: &[(PadButton, Button)] = &[
    (PadButton::South, Button::South),
    (PadButton::East, Button::East),
    (PadButton::West, Button::West),
    (PadButton::North, Button::North),
    (PadButton::LeftBumper, Button::LeftTrigger),
    (PadButton::RightBumper, Button::RightTrigger),
    (PadButton::LeftStick, Button::LeftThumb),
    (PadButton::RightStick, Button::RightThumb),
    (PadButton::DPadUp, Button::DPadUp),
    (PadButton::DPadDown, Button::DPadDown),
    (PadButton::DPadLeft, Button::DPadLeft),
    (PadButton::DPadRight, Button::DPadRight),
    (PadButton::Start, Button::Start),
    (PadButton::Select, Button::Select),
];

const STICKS: &[(PadAxis, Axis, f32)] = &[
    (PadAxis::LeftStickX, Axis::LeftStickX, 1.0),
    (PadAxis::LeftStickY, Axis::LeftStickY, -1.0),
    (PadAxis::RightStickX, Axis::RightStickX, 1.0),
    (PadAxis::RightStickY, Axis::RightStickY, -1.0),
];

const TRIGGERS: &[(PadAxis, Button)] = &[
    (PadAxis::LeftTrigger, Button::LeftTrigger2),
    (PadAxis::RightTrigger, Button::RightTrigger2),
];

/// Owns the gilrs context and the slot assignment.
///
/// Lives on the app rather than in the world, like the build watcher: it is a
/// backend holding an OS handle, and the world's copy of what it found is
/// [`InputState`](super::InputState).
pub struct Gamepads {
    gilrs: Gilrs,
    slots: [Option<GamepadId>; MAX_PLAYERS],
    /// Whether the pads found at startup have been announced to the world.
    /// gilrs reports a `Connected` event only for a pad that arrives while it
    /// is running, so one plugged in before launch is claimed here instead and
    /// would otherwise never reach [`InputState`](super::InputState).
    announced: bool,
}

impl Gamepads {
    /// Open the gamepad backend, or report why not.
    ///
    /// A machine with no gamepad subsystem is not an error worth stopping for —
    /// the session runs on keyboard and mouse, and every pad binding reads as
    /// unpressed, which is the same answer an unplugged pad gives.
    pub fn new() -> Result<Self, String> {
        let gilrs = Gilrs::new().map_err(|error| format!("gamepads unavailable: {error}"))?;
        let mut slots = [None; MAX_PLAYERS];
        for (slot, (id, _)) in gilrs.gamepads().take(MAX_PLAYERS).enumerate() {
            slots[slot] = Some(id);
        }
        Ok(Self {
            gilrs,
            slots,
            announced: false,
        })
    }

    /// Pump gilrs and sample every assigned pad into `input`.
    ///
    /// `focused` gates sampling, not pumping: connects and disconnects still
    /// have to be tracked while the window is in the background, but a stick
    /// held there must not drive a game nobody is looking at. Losing focus
    /// zeroes the pads for the same reason the keyboard clears its held set.
    pub fn pump(&mut self, input: &mut InputState, focused: bool) {
        if !self.announced {
            for (slot, assigned) in self.slots.iter().enumerate() {
                if assigned.is_some() {
                    input.set_pad_connected(slot, true);
                }
            }
            self.announced = true;
        }

        while let Some(event) = self.gilrs.next_event() {
            match event.event {
                gilrs::EventType::Connected => self.attach(event.id, input),
                gilrs::EventType::Disconnected => self.detach(event.id, input),
                _ => {}
            }
        }

        for (slot, assigned) in self.slots.iter().enumerate() {
            let Some(id) = assigned else { continue };
            if !focused {
                input.clear_pad(slot);
                continue;
            }
            let pad = self.gilrs.gamepad(*id);
            for (ours, theirs) in BUTTONS {
                input.set_pad_button(slot, *ours, pad.is_pressed(*theirs));
            }
            for (ours, theirs, sign) in STICKS {
                input.set_pad_axis(slot, *ours, pad.value(*theirs) * sign);
            }
            for (ours, theirs) in TRIGGERS {
                let value = pad.button_data(*theirs).map_or(0.0, |data| data.value());
                input.set_pad_axis(slot, *ours, value);
            }
        }
    }

    /// The player slot a pad holds, for the editor to show.
    pub fn slot_of(&self, player: usize) -> Option<GamepadId> {
        self.slots.get(player).copied().flatten()
    }

    fn attach(&mut self, id: GamepadId, input: &mut InputState) {
        if self.slots.contains(&Some(id)) {
            return;
        }
        if let Some(slot) = self.slots.iter().position(Option::is_none) {
            self.slots[slot] = Some(id);
            input.set_pad_connected(slot, true);
        }
    }

    fn detach(&mut self, id: GamepadId, input: &mut InputState) {
        if let Some(slot) = self.slots.iter().position(|held| *held == Some(id)) {
            self.slots[slot] = None;
            input.set_pad_connected(slot, false);
        }
    }
}
