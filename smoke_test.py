import os
import vizdoom as vzd

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.set_window_visible(False)
game.set_labels_buffer_enabled(True)
game.init()
print("ViZDoom jalan. Episode dimulai.")
game.new_episode()
n_buttons = len(game.get_available_buttons())
for i in range(10):
    state = game.get_state()
    if state is None:
        break
    labels = state.labels if state.labels is not None else []
    info = [(getattr(lb, "object_name", "?"), getattr(lb, "object_category", "?")) for lb in labels]
    print(i, "shape:", state.screen_buffer.shape, "labels:", len(labels), info)
    game.make_action([0] * n_buttons)
game.close()
print("Smoke test selesai.")
