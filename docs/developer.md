Here’s a **refined and secure developer setup section** that includes **PowerShell Execution Policy instructions** for Windows users, with clear separation between macOS/Linux and Windows, and with your VS Code workflow. You can copy-paste directly to your `README.md` or `DEVELOPER.md`.

---

## 🚀 Project Setup & Installation

### 1. **Clone the Project and Open in VS Code**

```bash
git clone https://github.com/vika4433/smart-intent-router.git
cd smart-intent-router
code .
```

*(This opens the project in Visual Studio Code)*

---

### 2. **Open Terminal in VS Code**

From the menu, select:
**View → Terminal**

---

### 3. **Create a Virtual Environment**

#### 🧪 On **macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 🪟 On **Windows**:

* **a. Command Prompt**:

  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **b. PowerShell**:

  If you get a **security error** about script execution, run:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

  Then:

  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

---

### 4. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

---

### 5. **Run the Project**

*(Replace the following with your actual project run command, if different)*

```bash
python src/mcp_server/server.py
```

---

> **Tip:**
> For MongoDB/LM Studio model installation, see `setup_instactions.md`.


