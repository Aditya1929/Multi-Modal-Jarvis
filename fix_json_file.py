import sys
import json
import shutil

def fix_json_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        line = line.strip()
        if not line or line in ("[", "]", ","):
            continue
        if line.endswith(","):
            line = line[:-1]
        try:
            obj = json.loads(line)
            objects.append(obj)
        except Exception:
            print(f"Skipping invalid line: {line}")

    # Backup original
    shutil.copy(filename, filename + ".bak")

    with open(filename, "w") as f:
        json.dump(objects, f, indent=2)

    print(f"Fixed {filename} (backup at {filename}.bak)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_json_file.py <path/to/file.json>")
        sys.exit(1)
    fix_json_file(sys.argv[1])