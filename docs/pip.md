Good question 👍
Where `pip install` puts Python packages depends on **your environment** (system Python, virtualenv, conda, user install, etc).

Here’s the breakdown:

---

## 🔎 Common Install Locations

### 1. **System-wide install (with sudo)**

```bash
sudo pip install somepackage
```

Packages go into:

```
/usr/local/lib/pythonX.Y/dist-packages/    # Ubuntu/Debian
/usr/lib/pythonX.Y/site-packages/          # Fedora/CentOS
```

(`X.Y` = your Python version, e.g. `3.10`)

---

### 2. **User install (without sudo)**

```bash
pip install --user somepackage
```

Packages go into:

```
~/.local/lib/pythonX.Y/site-packages/
```

---

### 3. **Virtual environment (recommended)**

If you’re inside a venv:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install somepackage
```

Packages go into:

```
./myenv/lib/pythonX.Y/site-packages/
```

---

### 4. **Conda environment**

If using Conda:

```
~/anaconda3/envs/myenv/lib/pythonX.Y/site-packages/
```

---

## 🛠 How to check where _your_ pip installs

Run:

```bash
python -m site
```

or

```bash
pip show somepackage
```

Example output:

```
Location: /usr/local/lib/python3.10/dist-packages
```

---

👉 Do you want me to show you how to **force pip to install globally with `sudo`**, or would you prefer the **safer approach using a virtual environment**?
