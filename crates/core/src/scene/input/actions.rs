//! Named actions and axes, resolved from bindings once per frame.
//!
//! Game code names an action, never a key. The indirection exists so that
//! remapping is a config change rather than an API break, which is only true if
//! nothing above this module ever learns what a binding is.
//!
//! **Resolution runs once per frame**, after the event pump and before scripts
//! tick, and it runs whether or not events arrived and whether or not scripts
//! are ticking. A resolve gated on either leaves the previous frame's state
//! standing, and every edge derived from it fires a frame late or twice.
//!
//! **Edges are the action's own, not any binding's.** An action is held when
//! any of its bindings is, and pressed on the frame its aggregate goes from
//! unheld to held. Aggregating each binding's own edges instead would fire a
//! second press when a second bound key joins one already down.
//!
//! **Ids are interned per name and stable for the process.** They are handed
//! out on first lookup, not by a config, so reloading bindings rewrites what an
//! id points at and never what it is — a script may cache one across a reload,
//! and a name nobody bound still has an id that reads as unpressed. That is
//! what makes reload cheap enough to do on every save.
//!
//! **Axes compose, they do not re-bind.** An axis names two actions, so a key
//! appears in the file exactly once and rebinding `Left` moves `Horizontal`
//! with it. An axis may also read one analog source directly; it may never name
//! another axis, which is what keeps resolution two passes and acyclic.

use std::collections::HashMap;

use super::binding::{Binding, TRIGGER_THRESHOLD};
use super::keys::MouseAxis;
use super::{InputState, MAX_PLAYERS};

/// A name's handle. Stable for the life of the process; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

/// The largest deadzone a config may ask for. Past this the rescale divides by
/// almost nothing and a stick becomes a switch.
const MAX_DEADZONE: f32 = 0.9;

/// What a name resolves to. A name may be interned before anything defines it,
/// which is the shape a typo takes: an id that answers no to everything.
#[derive(Clone, Debug, PartialEq)]
pub enum Definition {
    Unbound,
    Action(Vec<Binding>),
    Axis(Axis),
}

/// A definition as a config file states it, before names become ids. An axis
/// names its components; only [`Actions`] may turn a name into an id, because
/// only it knows which ids exist.
#[derive(Clone, Debug, PartialEq)]
pub enum Spec {
    Action(Vec<Binding>),
    Composed {
        positive: String,
        negative: String,
    },
    Analog {
        source: Binding,
        deadzone: f32,
        scale: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Axis {
    /// One action each way, read as −1, 0 or +1.
    Composed {
        positive: ActionId,
        negative: ActionId,
    },
    /// One analog source, deadzoned if it is bounded, then scaled.
    Analog {
        source: Binding,
        deadzone: f32,
        scale: f32,
    },
}

/// The resolved input state game code reads.
#[derive(Default)]
pub struct Actions {
    names: Vec<String>,
    ids: HashMap<String, ActionId>,
    defs: Vec<Definition>,
    held: Vec<[bool; MAX_PLAYERS]>,
    held_last: Vec<[bool; MAX_PLAYERS]>,
    values: Vec<[f32; MAX_PLAYERS]>,
    /// Set when bindings change, cleared by the next resolve, which seeds the
    /// previous frame from the current one so the swap itself fires no edges.
    /// A rebind while the old key is held would otherwise read as a release,
    /// and the new one as a press nobody made.
    reseed: bool,
    warnings: Vec<String>,
}

impl Actions {
    pub fn new() -> Self {
        Self::default()
    }

    /// The id for a name, interning it if this is the first time it is asked
    /// for. Names are matched exactly: they are identifiers the game author
    /// chose on both sides of the file, and a case-folded match would hide the
    /// mismatch rather than report it.
    pub fn id(&mut self, name: &str) -> ActionId {
        let known = self.ids.contains_key(name);
        let id = self.intern(name);
        if !known && self.defs[id.0 as usize] == Definition::Unbound {
            self.warnings.push(format!(
                "input action `{name}` is not bound; nothing in the input config defines it"
            ));
        }
        id
    }

    fn intern(&mut self, name: &str) -> ActionId {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = ActionId(self.names.len() as u32);
        self.names.push(name.to_owned());
        self.ids.insert(name.to_owned(), id);
        self.defs.push(Definition::Unbound);
        self.held.push([false; MAX_PLAYERS]);
        self.held_last.push([false; MAX_PLAYERS]);
        self.values.push([0.0; MAX_PLAYERS]);
        id
    }

    /// The id for a name only if it already has one.
    pub fn lookup(&self, name: &str) -> Option<ActionId> {
        self.ids.get(name).copied()
    }

    pub fn name(&self, id: ActionId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    /// Replace every definition. Names absent from `defs` keep their ids and
    /// fall back to unbound, so a binding deleted from the file stops firing
    /// without stranding a script that still asks for it.
    pub fn apply(&mut self, specs: Vec<(String, Spec)>) {
        for def in &mut self.defs {
            *def = Definition::Unbound;
        }
        for (name, spec) in specs {
            let id = self.intern(&name);
            let def = match spec {
                Spec::Action(bindings) => Definition::Action(bindings),
                Spec::Composed { positive, negative } => Definition::Axis(Axis::Composed {
                    positive: self.intern(&positive),
                    negative: self.intern(&negative),
                }),
                Spec::Analog {
                    source,
                    deadzone,
                    scale,
                } => Definition::Axis(Axis::Analog {
                    source,
                    deadzone,
                    scale,
                }),
            };
            self.defs[id.0 as usize] = def;
        }
        self.reseed = true;
    }

    /// Seed the previous frame from the current one, suppressing edges on the
    /// next resolve. Bindings do this for themselves; the other caller is the
    /// edit-to-play transition, where the same stale comparison would fire a
    /// press for every key already held when play began.
    pub fn reseed(&mut self) {
        self.reseed = true;
    }

    /// Drain the warnings interning has accumulated, for the caller to log.
    /// Kept out of this module so resolution has no opinion about console.
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Recompute every action and axis. Actions first: an axis reads its
    /// components' aggregates, so the two passes cannot merge.
    pub fn resolve(&mut self, input: &InputState) {
        std::mem::swap(&mut self.held, &mut self.held_last);

        for (index, def) in self.defs.iter().enumerate() {
            let mut held = [false; MAX_PLAYERS];
            if let Definition::Action(bindings) = def {
                for (player, slot) in held.iter_mut().enumerate() {
                    *slot = bindings
                        .iter()
                        .any(|binding| digital(input, *binding, player));
                }
            }
            self.held[index] = held;
        }

        for (index, def) in self.defs.iter().enumerate() {
            let mut values = [0.0; MAX_PLAYERS];
            if let Definition::Axis(axis) = def {
                for (player, slot) in values.iter_mut().enumerate() {
                    *slot = match axis {
                        Axis::Composed { positive, negative } => {
                            let plus = self.held[positive.0 as usize][player];
                            let minus = self.held[negative.0 as usize][player];
                            f32::from(plus) - f32::from(minus)
                        }
                        Axis::Analog {
                            source,
                            deadzone,
                            scale,
                        } => {
                            let raw = analog(input, *source, player);
                            let shaped = if source.is_normalised() {
                                apply_deadzone(raw, *deadzone)
                            } else {
                                raw
                            };
                            shaped * scale
                        }
                    };
                }
            }
            self.values[index] = values;
        }

        if self.reseed {
            self.held_last.copy_from_slice(&self.held);
            self.reseed = false;
        }
    }

    pub fn held(&self, id: ActionId, player: usize) -> bool {
        self.slot(&self.held, id, player)
    }

    pub fn pressed(&self, id: ActionId, player: usize) -> bool {
        self.slot(&self.held, id, player) && !self.slot(&self.held_last, id, player)
    }

    pub fn released(&self, id: ActionId, player: usize) -> bool {
        !self.slot(&self.held, id, player) && self.slot(&self.held_last, id, player)
    }

    /// An axis's value. An id naming an action rather than an axis reads zero,
    /// as does a player past [`MAX_PLAYERS`]: both are the caller asking for
    /// something that exists but holds nothing, which is the same answer an
    /// unbound name gives.
    pub fn axis(&self, id: ActionId, player: usize) -> f32 {
        self.values
            .get(id.0 as usize)
            .and_then(|players| players.get(player))
            .copied()
            .unwrap_or(0.0)
    }

    fn slot(&self, table: &[[bool; MAX_PLAYERS]], id: ActionId, player: usize) -> bool {
        table
            .get(id.0 as usize)
            .and_then(|players| players.get(player))
            .copied()
            .unwrap_or(false)
    }
}

/// Whether a source reads as pressed for this player.
///
/// Keyboard and mouse answer for player 0 only. They are one device between
/// however many players are in the room, and letting them answer for every slot
/// would have player two jumping on player one's spacebar.
fn digital(input: &InputState, binding: Binding, player: usize) -> bool {
    match binding {
        Binding::Key(code) => player == 0 && input.key_down(code),
        Binding::MouseButton(button) => player == 0 && input.mouse_button_down(button as u32),
        Binding::MouseAxis(_) => {
            player == 0 && analog(input, binding, player).abs() >= TRIGGER_THRESHOLD
        }
        Binding::PadButton(button) => input.pad_button_down(player, button),
        Binding::PadAxis(axis) => input.pad_axis(player, axis).abs() >= TRIGGER_THRESHOLD,
    }
}

/// A source's analog value. Digital sources read as 0 or 1, so a key may stand
/// in for half an axis.
fn analog(input: &InputState, binding: Binding, player: usize) -> f32 {
    match binding {
        Binding::MouseAxis(MouseAxis::X) if player == 0 => input.mouse_delta().0,
        Binding::MouseAxis(MouseAxis::Y) if player == 0 => input.mouse_delta().1,
        Binding::MouseAxis(_) => 0.0,
        Binding::PadAxis(axis) => input.pad_axis(player, axis),
        _ => f32::from(digital(input, binding, player)),
    }
}

/// Scaled radial deadzone, on a source that rests near zero and saturates at
/// one. The rescale is the half that gets left out: without it a stick's full
/// deflection reads `1 - deadzone` and no game ever reaches full speed.
///
/// Applied per component, so a stick pushed to a corner still travels further
/// than one pushed straight up. Deadzoning the vector needs to know that two
/// axes are one stick, which this schema does not yet say.
fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
    let deadzone = deadzone.clamp(0.0, MAX_DEADZONE);
    let magnitude = value.abs();
    if magnitude <= deadzone {
        return 0.0;
    }
    (((magnitude - deadzone) / (1.0 - deadzone)).min(1.0)).copysign(value)
}

impl Binding {
    /// Whether this source rests at zero and saturates at one, and so can be
    /// deadzoned and rescaled. Mouse movement is neither: it is a delta in
    /// pixels with no upper bound, and clamping it to one would cap how fast a
    /// player may turn.
    fn is_normalised(self) -> bool {
        matches!(self, Self::PadAxis(_))
    }
}
