//! A human-readable label attached to an entity.

/// An optional display name for an entity. Purely for tooling — it doesn't
/// affect simulation or rendering, so entities without one are unaffected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name(pub String);

impl Name {
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Name {
    #[inline]
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}
