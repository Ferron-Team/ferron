//! A queryable gameplay label attached to an entity.

/// A string label scripts look up via `World.FindByTag` / `FindAllByTag`.
/// Distinct from [`Name`](super::Name): a `Name` is an editor-facing display
/// label (unique-ish, human-readable), a `Tag` is gameplay-facing identity —
/// many entities may share one tag, and lookups match on exact equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tag(pub String);

impl Tag {
    #[inline]
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Tag {
    #[inline]
    fn from(tag: &str) -> Self {
        Self(tag.to_owned())
    }
}
