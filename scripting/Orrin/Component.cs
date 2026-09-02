namespace Orrin;

/// Gives a Behaviour a stable identity in the component registry, so its fields
/// are saved with the scene, shown in the inspector, and carried across a hot
/// reload.
///
/// The id is written into every scene file that references the component, and
/// is the *only* thing that ties a saved entity back to this class. It is
/// therefore deliberately not derived from the type's name: rename
/// `Spinner` to `Rotator` and every scene keeps loading, because the id did not
/// move. Change the id and they all stop — which is a rename of the component
/// type as far as the project is concerned, and should be as loud as one.
///
/// Convention is `game.thing`, lower case, dotted. Anything unique will do; the
/// engine reserves the `orrin.` prefix for its own components.
///
/// A Behaviour without this attribute still runs — it just has no registry
/// entry, so its fields are not saved or inspectable. Hot reload still carries
/// them (see [BehaviourState]).
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
public sealed class ComponentAttribute(string id) : Attribute
{
    public string Id { get; } = id;

    /// What the inspector calls it. Defaults to the type name, and unlike [Id]
    /// may change freely.
    public string? Name { get; init; }
}
