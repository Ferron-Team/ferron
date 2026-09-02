//! Reading `input.toml`: the file that says which key does what.
//!
//! Its own file rather than a section of `orrin.toml`, because validating a
//! binding name needs the key table and the key table names winit types. A
//! manifest crate that reaches for a windowing library to check that `Space`
//! is spelled correctly is the wrong shape, and moving the check downstream
//! instead would report a typo without the line it is on.
//!
//! Version header first, for the same reason the scene format carries one: this
//! is a document people commit, and a file the engine cannot read must be
//! refused by name rather than parsed optimistically.
//!
//! Names are matched case-insensitively and errors quote the canonical
//! spelling. The file is hand-written, so refusing `space` teaches nothing that
//! accepting it does not — but the message still points at one spelling, so a
//! project converges on it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::actions::Spec;
use super::binding::{self, Binding, UnknownBinding};

/// Bumped when the grammar changes in a way an older file does not satisfy.
pub const FORMAT_VERSION: u32 = 1;

/// The default file, relative to the project root.
pub const FILE_NAME: &str = "input.toml";

/// Resting slop on a stick that has never been touched. Sticks need one and
/// triggers do not care, so it applies to any bounded analog source that does
/// not name its own.
const DEFAULT_DEADZONE: f32 = 0.15;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format_version: u32,
    #[serde(default)]
    actions: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    axes: BTreeMap<String, RawAxis>,
}

/// One axis, either shape. Serde cannot tell them apart without trying both,
/// and an untagged enum reports the failure as "matched no variant" with no
/// line and no hint — so every field is optional here and the combination is
/// checked by hand, where the error can say what was found and what was wanted.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAxis {
    positive: Option<String>,
    negative: Option<String>,
    source: Option<String>,
    deadzone: Option<f32>,
    scale: Option<f32>,
}

#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        message: String,
    },
    UnsupportedVersion {
        path: PathBuf,
        found: u32,
    },
    UnknownSource {
        name: String,
        source: UnknownBinding,
    },
    AxisShape {
        axis: String,
        message: &'static str,
    },
    UnknownComponent {
        axis: String,
        component: String,
    },
    ComponentIsAxis {
        axis: String,
        component: String,
    },
    NameCollision {
        name: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "could not read {}: {source}", path.display()),
            Self::Parse { path, message } => {
                write!(f, "could not parse {}: {message}", path.display())
            }
            Self::UnsupportedVersion { path, found } => write!(
                f,
                "{} is format version {found}, and this engine reads version \
                 {FORMAT_VERSION}",
                path.display()
            ),
            Self::UnknownSource { name, source } => write!(f, "`{name}`: {source}"),
            Self::AxisShape { axis, message } => write!(f, "axis `{axis}`: {message}"),
            Self::UnknownComponent { axis, component } => write!(
                f,
                "axis `{axis}` names `{component}`, which no action defines"
            ),
            Self::ComponentIsAxis { axis, component } => write!(
                f,
                "axis `{axis}` names `{component}`, which is itself an axis; an \
                 axis may only compose actions"
            ),
            Self::NameCollision { name } => {
                write!(f, "`{name}` is defined as both an action and an axis")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Read and validate the file at `path`.
pub fn load(path: &Path) -> Result<Vec<(String, Spec)>, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_owned(),
        source,
    })?;
    parse(&text, path)
}

/// Validate a file's text. Split from [`load`] so the grammar is checkable
/// without a filesystem, and so the editor can validate a buffer before saving.
pub fn parse(text: &str, path: &Path) -> Result<Vec<(String, Spec)>, ConfigError> {
    let file: File = toml::from_str(text).map_err(|error| ConfigError::Parse {
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    if file.format_version != FORMAT_VERSION {
        return Err(ConfigError::UnsupportedVersion {
            path: path.to_owned(),
            found: file.format_version,
        });
    }

    let mut specs = Vec::with_capacity(file.actions.len() + file.axes.len());
    for (name, sources) in &file.actions {
        if file.axes.contains_key(name) {
            return Err(ConfigError::NameCollision { name: name.clone() });
        }
        let bindings = sources
            .iter()
            .map(|source| {
                binding::parse(source).map_err(|error| ConfigError::UnknownSource {
                    name: name.clone(),
                    source: error,
                })
            })
            .collect::<Result<Vec<Binding>, _>>()?;
        specs.push((name.clone(), Spec::Action(bindings)));
    }

    for (name, axis) in &file.axes {
        specs.push((name.clone(), axis_spec(name, axis, &file)?));
    }
    Ok(specs)
}

fn axis_spec(name: &str, axis: &RawAxis, file: &File) -> Result<Spec, ConfigError> {
    match (&axis.positive, &axis.negative, &axis.source) {
        (Some(positive), Some(negative), None) => {
            for component in [positive, negative] {
                if file.axes.contains_key(component) {
                    return Err(ConfigError::ComponentIsAxis {
                        axis: name.to_owned(),
                        component: component.clone(),
                    });
                }
                if !file.actions.contains_key(component) {
                    return Err(ConfigError::UnknownComponent {
                        axis: name.to_owned(),
                        component: component.clone(),
                    });
                }
            }
            if axis.deadzone.is_some() || axis.scale.is_some() {
                return Err(ConfigError::AxisShape {
                    axis: name.to_owned(),
                    message: "`deadzone` and `scale` shape an analog source; a \
                              composed axis is already −1, 0 or 1",
                });
            }
            Ok(Spec::Composed {
                positive: positive.clone(),
                negative: negative.clone(),
            })
        }
        (None, None, Some(source)) => {
            let source = binding::parse(source).map_err(|error| ConfigError::UnknownSource {
                name: name.to_owned(),
                source: error,
            })?;
            Ok(Spec::Analog {
                source,
                deadzone: axis.deadzone.unwrap_or(DEFAULT_DEADZONE),
                scale: axis.scale.unwrap_or(1.0),
            })
        }
        (None, None, None) => Err(ConfigError::AxisShape {
            axis: name.to_owned(),
            message: "needs either `positive` and `negative`, or `source`",
        }),
        _ => Err(ConfigError::AxisShape {
            axis: name.to_owned(),
            message: "mixes a composed axis with an analog one; give either \
                      `positive` and `negative`, or `source`, not both",
        }),
    }
}
