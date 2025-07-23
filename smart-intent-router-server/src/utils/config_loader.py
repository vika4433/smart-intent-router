import yaml
import threading
import time
from pathlib import Path

# Fix: Use the correct config path relative to project root
CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "smart_intent_router_config.yaml"

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.last_mtime = None
        self.config = None
        self.lock = threading.Lock()
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # TODO: Validate config format here!
        with self.lock:
            self.config = config
            self.last_mtime = self.config_path.stat().st_mtime

    def get_config(self):
        with self.lock:
            return self.config

    def auto_reload(self, interval=3):
        while True:
            time.sleep(interval)
            try:
                mtime = self.config_path.stat().st_mtime
                if mtime != self.last_mtime:
                    self.load_config()
                    print("Config reloaded!")
            except Exception as e:
                print("Failed to reload config:", e)

# Usage:
# loader = ConfigLoader(CONFIG_PATH)
# threading.Thread(target=loader.auto_reload, daemon=True).start()
# # In your model selector: config = loader.get_config()
