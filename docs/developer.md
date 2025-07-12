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
python -m pip install -upgrade pip
pip install -r requirements.txt
```

---


> **Tip:**
> For MongoDB/LM Studio model installation, see `setup_instactions.md`.


Absolutely! Here’s **clear documentation for your `launch.json` setup** with a practical explanation for developers in your `README.md` (or `DEVELOPER.md`).
This assumes you want new team members to quickly understand how to launch both the backend **MCP server** and the **Streamlit web client** in VS Code.

---

## 🏃‍♂️ Running the Project: VS Code `launch.json` Explained

### **How to Run Both Backend and Web UI Together**

1. **Open your project in VS Code** (see previous setup instructions).

2. **Make sure your virtual environment is activated** and dependencies are installed.

3. **Find the “Run and Debug” icon** on the left sidebar in VS Code
   (looks like a play button with a bug).

4. Click the dropdown menu at the top of the Run & Debug pane.
   You will see these launch configurations from `.vscode/launch.json`:

   * **Run MCP Server**:
     Starts the backend Smart Intent Router server (`server.py`).

   * **Run Web Client**:
     Launches the web UI with Streamlit (`app.py`) on port 8501.

   * **Run Server and Web Client** *(Compound)*:
     Starts **both** at once — the backend and the web client.

5. **How to launch both**:

   * Select **Run Server and Web Client** in the dropdown.
   * Click the green ▶️ "Start Debugging" button (or press `F5`).
   * The MCP backend server will start in one terminal tab, and your default browser will open the web client at [http://localhost:8501](http://localhost:8501).

---

### **launch.json Example Explained**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run MCP Server",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/smart-intent-router-server/src/mcp_server/server.py",
      "console": "integratedTerminal"
    },
    {
      "name": "Run Web Client",
      "type": "python",
      "request": "launch",
      "module": "streamlit",
      "args": [
        "run",
        "${workspaceFolder}/web-client/src/app.py",
        "--server.port",
        "8501"
      ],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    }
  ],
  "compounds": [
    {
      "name": "Run Server and Web Client",
      "configurations": [
        "Run Web Client",
        "Run MCP Server"
      ],
      "stopAll": true
    }
  ]
}
```

* **Compounds** let you launch multiple processes together.
* You can comment/uncomment configurations for test clients as needed.

---

### **Summary Steps for Developers**

1. **Clone the repo and set up the environment** (see above).
2. **Open VS Code and the Run/Debug tab**.
3. **Select** `Run Server and Web Client`.
4. **Click "Start Debugging" or press F5**.
5. **Your browser will open the web client**; the backend runs in the terminal.

