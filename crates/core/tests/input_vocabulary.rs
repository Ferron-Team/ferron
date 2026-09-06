//! The key table and the C# `KeyCode` enum are one vocabulary in two files.
//!
//! Nothing else can catch them drifting. The numbers cross the scripting
//! boundary, so a variant added to one side alone is not a compile error on
//! either — it is a key that silently does nothing, found by a player rather
//! than by CI. This is the same check `Orrin.MathTests` makes for the blittable
//! structs, pointed the other way across the seam.

use std::collections::BTreeMap;
use std::path::PathBuf;

use orrin_core::scene::input::keys::KEYS;

fn input_cs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripting/Orrin/Input.cs")
}

/// The `KeyCode` variants as C# sees them, implicit increments resolved.
fn csharp_variants(source: &str) -> BTreeMap<String, u32> {
    let body = source
        .split_once("enum KeyCode : uint")
        .expect("Input.cs declares `enum KeyCode : uint`")
        .1;
    let body = body
        .split_once('{')
        .expect("the enum has a body")
        .1
        .split_once('}')
        .expect("the enum body is closed")
        .0;

    let mut variants = BTreeMap::new();
    let mut next = 0;
    for line in body.lines() {
        let line = line.split("//").next().unwrap_or_default();
        for entry in line.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let (name, value) = match entry.split_once('=') {
                Some((name, value)) => (
                    name.trim(),
                    value.trim().parse().expect("an explicit value is a number"),
                ),
                None => (entry, next),
            };
            variants.insert(name.to_owned(), value);
            next = value + 1;
        }
    }
    variants
}

#[test]
fn the_key_table_and_the_csharp_enum_agree() {
    let source = std::fs::read_to_string(input_cs()).expect("Input.cs is readable");
    let csharp = csharp_variants(&source);
    let rust: BTreeMap<String, u32> = KEYS
        .iter()
        .map(|key| (key.name.to_owned(), key.code))
        .collect();

    let missing: Vec<_> = rust
        .keys()
        .filter(|name| !csharp.contains_key(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "keys.rs has rows with no C# variant: {missing:?} — add them to \
         `KeyCode` in scripting/Orrin/Input.cs"
    );

    let extra: Vec<_> = csharp
        .keys()
        .filter(|name| !rust.contains_key(*name))
        .collect();
    assert!(
        extra.is_empty(),
        "the C# `KeyCode` enum has variants with no row in keys.rs: {extra:?}"
    );

    let disagreed: Vec<_> = rust
        .iter()
        .filter(|(name, code)| csharp.get(*name) != Some(code))
        .map(|(name, code)| format!("{name}: keys.rs says {code}, C# says {:?}", csharp[name]))
        .collect();
    assert!(
        disagreed.is_empty(),
        "the two sides number keys differently: {disagreed:#?}"
    );
}
