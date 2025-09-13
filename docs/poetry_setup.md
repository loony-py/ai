Here’s the full setup guide (works like `npm init` + `npm install` in Node.js):

---

## 🔹 1. Install Python

Check if Python is installed:

```bash
python3 --version
```

If not, install it (Ubuntu/Debian example):

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

---

## 🔹 2. Create Project Folder

```bash
mkdir my_project && cd my_project
```

---

## 🔹 3. Setup Virtual Environment (like `node_modules`)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows (PowerShell)
```

Now your terminal should show `(venv)` prefix.

---

## 🔹 4. Install Poetry (dependency manager, like `npm/yarn`)

You already have it at `/home/sankar/.local/bin/poetry`, so just add it to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

(Optional: add that line to `~/.bashrc` or `~/.zshrc` so it works permanently.)

Check:

```bash
poetry --version
```

---

## 🔹 5. Initialize Poetry (like `npm init`)

Inside project folder:

```bash
poetry init
```

This creates **`pyproject.toml`** (like `package.json`).
Add dependencies:

```bash
poetry add flask requests
```

---

## 🔹 6. Project Structure

```
my_project/
 ├─ pyproject.toml     # like package.json
 ├─ poetry.lock        # lockfile
 ├─ venv/              # virtual env
 ├─ my_project/        # source code
 │   ├─ __init__.py
 │   └─ main.py
 └─ tests/             # test files
```

Example `main.py`:

```python
def main():
    print("Hello, Python project setup complete!")

if __name__ == "__main__":
    main()
```

---

## 🔹 7. Run Your App

```bash
poetry run python my_project/main.py
```

Or define a script in `pyproject.toml`:

```toml
[tool.poetry.scripts]
start = "my_project.main:main"
```

Then run:

```bash
poetry run start
```

---

✅ Now you have a full Python setup, managed like Node.js.

Do you want me to generate a **ready-to-use boilerplate project (files + pyproject.toml)** so you can just copy-paste it and run `poetry install`?
