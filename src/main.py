import json
import ttkbootstrap as tb
from ttk_bootstrap_gui import ImageEditorApp

def main():
    # Load configuration from JSON file safely
    with open('src/config.json') as config_file:
        config = json.load(config_file)

    root = tb.Window(themename=config["themename"])
    app = ImageEditorApp(root, config)
    root.mainloop()

if __name__ == "__main__":
    main()
