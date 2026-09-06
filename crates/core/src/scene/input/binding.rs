//! What a binding name in a config file resolves to.
//!
//! A binding is one physical source, never a combination: the list an action
//! carries is an alternatives list, so `["Space", "Gamepad.South"]` means either
//! button jumps. Chords are deliberately absent — a binding stays a single
//! string, which is what leaves room to spell one later without changing the
//! shape of the file.
//!
//! Sources divide into digital and analog, and either can serve either role.
//! An action binding to an analog source compares it against a threshold; an
//! axis binding to a digital source reads it as zero or one. That is what makes
//! a trigger usable as a jump button and `W` usable as half of an axis without
//! two vocabularies.

use super::keys::{self, MouseAxis, PadAxis, PadButton};

/// How far a *bounded* analog source — a stick, a trigger — must travel to read
/// as a pressed button. Past the resting slop of a worn trigger, short of the
/// point where a player would say they had pulled it.
pub const TRIGGER_THRESHOLD: f32 = 0.5;

/// How far the mouse must travel in a frame to read as a pressed button.
///
/// Its own constant rather than [`TRIGGER_THRESHOLD`]'s, because the two are not
/// the same quantity: a trigger is a fraction of full travel, mouse motion is a
/// pixel delta with no upper bound. Half a pixel is the smallest honest answer
/// to "did the mouse move this frame", and a fraction-of-travel number would
/// mean nothing here.
pub const MOTION_THRESHOLD: f32 = 0.5;

/// One physical source a binding can name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Binding {
    Key(u32),
    MouseButton(u8),
    MouseAxis(MouseAxis),
    PadButton(PadButton),
    PadAxis(PadAxis),
}

impl Binding {
    /// Whether this source rests at zero and saturates at one, and so can be
    /// deadzoned and rescaled.
    ///
    /// Mouse movement is neither: it is a delta in pixels with no upper bound,
    /// and clamping it to one would cap how fast a player may turn. A digital
    /// source is not one either — it is already exactly zero or one, so there is
    /// no slop to cut out. Read by resolution to decide whether to deadzone, and
    /// by [`config`](super::config) to refuse a deadzone that would do nothing.
    pub fn is_normalised(self) -> bool {
        matches!(self, Self::PadAxis(_))
    }
}

/// A binding name that named nothing, carrying the nearest thing it might have
/// meant.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownBinding {
    pub name: String,
    pub suggestion: Option<String>,
}

impl std::fmt::Display for UnknownBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown input source `{}`", self.name)?;
        match &self.suggestion {
            Some(near) => write!(f, "; did you mean `{near}`?"),
            None => write!(
                f,
                "; keyboard keys are unprefixed (`Space`), other devices are \
                 not (`Mouse.Left`, `Gamepad.South`)"
            ),
        }
    }
}

/// Resolve a binding name. Keyboard names are bare; everything else carries its
/// device, and the remainder after the first `.` names the source within it.
pub fn parse(name: &str) -> Result<Binding, UnknownBinding> {
    let parsed = match name.split_once('.') {
        Some((device, rest)) if device.eq_ignore_ascii_case("Mouse") => parse_mouse(rest),
        Some((device, rest)) if device.eq_ignore_ascii_case("Gamepad") => parse_pad(rest),
        Some(_) => None,
        None => keys::key_by_name(name).map(Binding::Key),
    };
    parsed.ok_or_else(|| UnknownBinding {
        name: name.to_owned(),
        suggestion: nearest(name),
    })
}

fn parse_mouse(rest: &str) -> Option<Binding> {
    let button = keys::MOUSE_BUTTONS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(rest))
        .map(|(_, bit)| Binding::MouseButton(*bit));
    button.or_else(|| {
        keys::MOUSE_AXES
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(rest))
            .map(|(_, axis)| Binding::MouseAxis(*axis))
    })
}

fn parse_pad(rest: &str) -> Option<Binding> {
    let button = keys::PAD_BUTTONS
        .iter()
        .find(|(name, _, aliases)| {
            name.eq_ignore_ascii_case(rest)
                || aliases.iter().any(|alias| alias.eq_ignore_ascii_case(rest))
        })
        .map(|(_, button, _)| Binding::PadButton(*button));
    button.or_else(|| {
        keys::PAD_AXES
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(rest))
            .map(|(_, axis)| Binding::PadAxis(*axis))
    })
}

/// Every name a binding may spell, canonical form only. Suggestions point at
/// the spelling the file should settle on, so aliases are not candidates.
fn candidates() -> impl Iterator<Item = String> {
    let keys = keys::KEYS.iter().map(|key| key.name.to_owned());
    let mouse = keys::MOUSE_BUTTONS
        .iter()
        .map(|(name, _)| format!("Mouse.{name}"))
        .chain(
            keys::MOUSE_AXES
                .iter()
                .map(|(name, _)| format!("Mouse.{name}")),
        );
    let pad = keys::PAD_BUTTONS
        .iter()
        .map(|(name, _, _)| format!("Gamepad.{name}"))
        .chain(
            keys::PAD_AXES
                .iter()
                .map(|(name, _)| format!("Gamepad.{name}")),
        );
    keys.chain(mouse).chain(pad)
}

/// The closest candidate within a third of the typed name's length, which keeps
/// `Retrun` suggesting `Return` without `Q` suggesting the whole alphabet.
fn nearest(name: &str) -> Option<String> {
    let budget = (name.chars().count() / 3).max(1);
    candidates()
        .map(|candidate| (distance(name, &candidate), candidate))
        .filter(|(distance, _)| *distance <= budget)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

/// Case-insensitive Levenshtein distance, two rows at a time.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().flat_map(char::to_lowercase).collect();
    let b: Vec<char> = b.chars().flat_map(char::to_lowercase).collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != cb);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}
