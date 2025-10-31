import json
from pathlib import Path

# Load JSON data once at package import and export for global use
_base_dir = Path(__file__).parent
_properties_path = _base_dir / "data" / "properties.json"
_requirements_path = _base_dir / "data" / "requirements.json"

with open(_properties_path, "r", encoding="utf-8") as _f:
    properties = json.load(_f)

with open(_requirements_path, "r", encoding="utf-8") as _f:
    requirements = json.load(_f)

if properties and requirements:
    print("✅ JSON data loaded successfully!")
else:
    print("❌ Failed to load JSON data!")

__all__ = ["properties", "requirements"]

