## 🛠️ Install Visual Studio Code

### **1. Download and Install VS Code**

#### 🧪 **macOS / Windows / Linux**

* Go to the official Visual Studio Code website:
  👉 [https://code.visualstudio.com/](https://code.visualstudio.com/)

* Click **Download for your OS** (macOS, Windows, or Linux).

* Run the downloaded installer and follow the setup wizard.

#### **(Optional) Add 'code' Command to Your PATH**

* On first launch, VS Code will often prompt to add the `code` command to your system PATH.
* If not prompted, you can do this manually:

  * **macOS/Linux:**
    Open VS Code, press <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, type “Shell Command: Install ‘code’ command in PATH”, and press Enter.
  * **Windows:**
    Usually added automatically. If not, see the official docs:
    [Add 'code' to PATH](https://code.visualstudio.com/docs/setup/windows#_command-line-interface-cli)

---

**Once installed, you can open a project folder from your terminal:**

```bash
cd <your-project-directory>
code .
```

Or simply use the VS Code GUI to “Open Folder…”




## ✅ MongoDB + Compass Installation Guide

### 🧪 macOS

#### 1. **Install MongoDB via Homebrew**

If you don't have Homebrew installed, install it first from [https://brew.sh/](https://brew.sh/).

```bash
brew tap mongodb/brew
brew install mongodb-community@7.0
```

#### 2. **Start the MongoDB service**

```bash
brew services start mongodb-community@7.0
```

### 🧪 Windows

#### ✅ **Recommended: Manual Installation via MSI (Best for All Users)**

##### 1. **Download MongoDB Installer**

* Go to: 👉 [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
* Choose:

  * **Version:** 7.0 (or latest stable)
  * **Platform:** Windows
  * **Package:** MSI
* Download the `.msi` installer.

##### 2. **Install MongoDB**

* Run the downloaded `.msi` file.
* **Enable “Install MongoDB as a Service”** during setup (recommended!).
* Let the installer complete.

##### 3. **Check MongoDB Service**

* Open **PowerShell as Administrator**.
* Run:

  ```powershell
  Get-Service | Where-Object { $_.Name -like "*mongo*" }
  ```
* You should see a service named `MongoDB` or similar.

  * **Status should be `Running`** after installation if service setup was chosen.

##### 4. **Start MongoDB Service (if not already running)**

```powershell
Start-Service -Name 'MongoDB'
```

Or from Command Prompt:

```cmd
net start MongoDB
```

##### 5. **Verify MongoDB is Running**

```powershell
mongo --eval 'db.runCommand({ connectionStatus: 1 })'
```

* Look for `"ok" : 1` in the output.

##### 6. **Install MongoDB Compass (GUI Tool)**

* Go to: 👉 [https://www.mongodb.com/try/download/compass](https://www.mongodb.com/try/download/compass)
* Download the `.exe` file for **Windows**.
* Run the installer.

##### 7. **Connect with Compass**

* Open MongoDB Compass.
* Use the default connection string:

  ```
  mongodb://localhost:27017
  ```
* Click **Connect** and you should see your local MongoDB server.

---

### ⚡️ **Quick Reference: All-in-One PowerShell Commands**

```powershell
# Check service installed and status
Get-Service | Where-Object { $_.Name -like "*mongo*" }

# Start service if needed
Start-Service -Name 'MongoDB'

# Test the connection
mongo --eval 'db.runCommand({ connectionStatus: 1 })'
```

---

#### ℹ️ **Troubleshooting**

* If you don’t see the service, re-run the MSI and make sure you select “Install as a Service.”
* For errors related to file paths (e.g., log file or data directory), check your `mongod.cfg` and make sure the paths exist and are accessible.

---

### 🎨 **GUI Access with MongoDB Compass**

* Download from the official [Compass page](https://www.mongodb.com/try/download/compass).
* Use `mongodb://localhost:27017` to connect to your local MongoDB.




## 🚀 LM Studio Installation Guide

---

### 🧪 macOS

#### 1. **Download LM Studio**

* Go to the official website:
  👉 [https://lmstudio.ai/download](https://lmstudio.ai/download)

* Choose **macOS**.

* Download the `.dmg` file (Apple Silicon or Intel, according to your Mac).

#### 2. **Install LM Studio**

* Double-click the downloaded `.dmg` file.
* Drag the **LM Studio** app icon to your **Applications** folder.

#### 3. **Open LM Studio**

* Find **LM Studio** in your Applications folder or Launchpad.
* Open the app (the first time, you may need to right-click and choose “Open” to bypass Gatekeeper).

---

### 🧪 Windows

#### 1. **Download LM Studio**

* Go to:
  👉 [https://lmstudio.ai/download](https://lmstudio.ai/download)

* Choose **Windows**.

* Download the `.exe` installer.

#### 2. **Install LM Studio**

* Double-click the downloaded `.exe` file.
* Follow the setup wizard steps.
* Complete installation.

#### 3. **Open LM Studio**

* Find **LM Studio** in your Start menu or Desktop.
* Launch the application.

---

### ⚡️ **Tips After Installation**

* **First Run:** LM Studio will ask you to select a folder for downloaded models.
* **Models:** You can browse and download models directly within the app (“Models” tab).
* **HTTP API:**

  * Enable the **OpenAI-compatible API server** via the Settings page to allow your MCP server to interact with LM Studio.
  * Note the API port (default is `1234`).

---

### 📝 **Connect to LM Studio API**

To connect your application (e.g., MCP server), use the following endpoint in your config:

```
http://localhost:1234/v1/chat/completions
```

You can verify it is running by visiting [http://localhost:1234/docs](http://localhost:1234/docs) in your browser.

Here’s an updated and **concise guide** for downloading exactly these models in LM Studio, **specifying the correct quantization/variant for each**:

---

## 📥 Downloading Models in LM Studio

### 1. **Open LM Studio**

* Start the LM Studio app on your machine.

---

### 2. **Go to the “Models” Tab**

* Click the **Models** icon on the sidebar.

---

### 3. **Search and Download Models**

#### **A) Qwen2.5-Coder-7B-Instruct (4-bit MLX, for Apple Silicon)**

1. **In the search bar**, enter:

   ```
   Qwen2.5-Coder-7B-Instruct
   ```
2. **Find the MLX 4-bit version**:

   * Look for a variant named **“Qwen2.5-Coder-7B-Instruct-MLX-4bit”**
     or similar (may appear as "Qwen2.5-Coder-7B-Instruct-MLX-4bit-mlx").
   * *For Mac M1/M2/M3, always prefer the “MLX-4bit” version for best performance.*
3. **Click Download** on that entry and wait until it’s marked as “Ready”.

---

#### **B) Llama-2-7b-chat (Q4\_K\_S, GGUF Format)**

1. **In the search bar**, enter:

   ```
   llama-2-7b-chat
   ```
2. **Find the Q4\_K\_S quantized file**:

   * Look for a file like **“llama-2-7b-chat.Q4\_K\_S.gguf”**.
   * *The Q4\_K\_S variant is a 4-bit quantized file, well supported for local inference and fast loading.*
3. **Click Download** next to this variant.

---

### 4. **Loading and Using the Model**

* After download, click **Load** on the model you want to activate.
* To allow apps to connect (e.g., via OpenAI API), go to **Settings → API** and toggle on the API server.



