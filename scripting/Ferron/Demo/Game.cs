using System.Numerics;

namespace Ferron.Demo;

// Demo entry behaviour (attached by the engine at startup; see FERRON_ENTRY):
// spawns a player cube, moves it with the arrow keys, and drops a cube with a
// cycling material each time Space is pressed.
public class Game : Behaviour
{
    public float MoveSpeed = 6f;

    private static readonly string[] Palette = ["gold", "copper", "neon", "glossy"];

    private Entity _player;
    // Source of truth for the player's transform: the spawn command applies
    // after this tick, so reading it back through GetTransform in the same
    // frame would see nothing (and zero out the scale).
    private Transform _playerTransform;
    private int _spawned;

    public override void OnStart()
    {
        _playerTransform = Transform.Identity;
        _playerTransform.Position = new Vector3(0f, 2.5f, 0f);
        _player = World.SpawnRenderable("cube", "gold", _playerTransform);
        Native.Log($"Game started; player = {_player} (arrow keys move, Space spawns)");
    }

    public override void OnUpdate(float deltaTime)
    {
        var direction = Vector3.Zero;
        if (Input.GetKey(KeyCode.LeftArrow)) direction.X -= 1f;
        if (Input.GetKey(KeyCode.RightArrow)) direction.X += 1f;
        if (Input.GetKey(KeyCode.UpArrow)) direction.Z -= 1f;
        if (Input.GetKey(KeyCode.DownArrow)) direction.Z += 1f;

        if (direction != Vector3.Zero)
        {
            _playerTransform.Position += Vector3.Normalize(direction) * MoveSpeed * deltaTime;
            Native.SetTransform(_player, _playerTransform);
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            var drop = Transform.Identity;
            drop.Position = _playerTransform.Position + new Vector3(0f, 1.75f, 0f);
            drop.Scale = new Vector3(0.5f);
            var material = Palette[_spawned++ % Palette.Length];
            var entity = World.SpawnRenderable("cube", material, drop);
            Native.Log($"spawned {entity} ({material})");
        }
    }
}
