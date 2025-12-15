set NODE_OPTIONS=--no-deprecation
ccr code

Here is a rewritten, streamlined, and polished version of the tutorial. I have organized it for better readability, clearer OS-specific instructions, and ease of copy-pasting.

***

# 🚀 Guide: Run Claude Code for Free with Google Gemini
**Dec 2025 Edition**

This guide allows you to use **Claude Code** (Anthropic's premium terminal-based coding agent) without a subscription. By using the community tool **Claude Code Router**, we bridge the polished Claude interface with Google's **Gemini models**, which offer a generous free tier.

**The Result:** You get full agentic coding capabilities (file editing, command execution) powered by Gemini's fast 2.5 Flash models—**for free.**

---

### ✅ Prerequisites
1.  **Node.js v18+**: [Download here](https://nodejs.org) if not installed.
2.  **Google Account**: To generate a free Gemini API key.
3.  **OS**: Windows, macOS, or Linux.

---

### Step 1: Get Your Free Gemini API Key
1.  Navigate to **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2.  Sign in with your Google account.
3.  Click **Create API key**.
4.  **Copy the key** (starts with `AIzaSy...`) and save it for Step 5.

---

### Step 2: Install Tools
Open your terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
npm install -g @anthropic-ai/claude-code @musistudio/claude-code-router
```
*Note: Ignore any deprecation warnings during installation.*

---

### Step 3: Configure the Router
You need to create a configuration file that tells Claude Code to talk to Gemini instead of Anthropic.

**1. Create the directories:**
*   **Windows:**
    ```cmd
    mkdir "%USERPROFILE%\.claude-code-router"
    mkdir "%USERPROFILE%\.claude"
    ```
*   **macOS/Linux:**
    ```bash
    mkdir -p ~/.claude-code-router ~/.claude
    ```

**2. Create the Config File:**
Create a file named `config.json` inside the `.claude-code-router` folder created above.
*   **Windows:** `notepad "%USERPROFILE%\.claude-code-router\config.json"`
*   **macOS/Linux:** `nano ~/.claude-code-router/config.json`

**3. Paste the following JSON:**
*This configuration uses the Gemini 2.5 Flash models (current as of Dec 2025).*

```json
{
  "LOG": true,
  "LOG_LEVEL": "info",
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "gemini",
      "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
      "api_key": "$GOOGLE_API_KEY",
      "models": [
        "gemini-2.5-flash",
        "gemini-2.0-flash"
      ],
      "transformer": {
        "use": ["gemini"]
      }
    }
  ],
  "Router": {
    "default": "gemini,gemini-2.5-flash",
    "background": "gemini,gemini-2.5-flash",
    "think": "gemini,gemini-2.5-flash",
    "longContext": "gemini,gemini-2.5-flash",
    "longContextThreshold": 60000
  }
}
```
**Save and close the file.**

---

### Step 4: Set the API Key Variable
For the router to access your key securely, set it as an Environment Variable.

**Option A: Permanent (Recommended)**
*   **Windows:** 
    1. Search for "Environment Variables" in the Start menu.
    2. Click "Edit the system environment variables" -> "Environment Variables".
    3. Under **User variables**, click **New**.
    4. **Name:** `GOOGLE_API_KEY` | **Value:** `Paste-Your-Key-Here`.
*   **macOS/Linux:**
    Run this in your terminal to append it to your profile:
    ```bash
    echo 'export GOOGLE_API_KEY="your_key_here"' >> ~/.zshrc
    source ~/.zshrc
    ```

**Option B: Temporary (One-time session)**
*   **Windows:** `set GOOGLE_API_KEY=your_key_here`
*   **Mac/Linux:** `export GOOGLE_API_KEY=your_key_here`

---

### Step 5: Launch Claude Code
You do not need to manually start a server. The router tool handles everything.

1.  Open a new terminal window (to ensure the API key loads).
2.  Navigate to your coding project folder:
    ```bash
    cd path/to/your/project
    ```
3.  **Run the magic command:**
    ```bash
    ccr code
    ```

**Success!** The Claude Code interface will launch.
*   *Note: If you see "⚠️ API key is not set. HOST is forced to 127.0.0.1", you can safely ignore it. This refers to remote server access, not your Gemini key.*

---

### 💡 Usage & Troubleshooting

**What can I do?**
Once the interface loads, treat it like a senior developer sitting next to you:
*   *"Refactor main.py to use async/await."*
*   *"Write a unit test for this component."*
*   *"Explain what this bug in the logs means."*

**Common Issues:**
*   **Auth Errors/No Response:** Ensure your `GOOGLE_API_KEY` is set correctly. Run `echo %GOOGLE_API_KEY%` (Win) or `echo $GOOGLE_API_KEY` (Mac) to verify.
*   **Rate Limits:** The Gemini free tier allows ~20-50 requests/day (varies by region/load). If you hit a limit, wait a few hours or switch to a paid Pay-as-you-go plan.
*   **Check Status:** Run `ccr status` to see if the router is healthy.
*   **Stop the Tool:** Run `ccr stop`.

*Disclaimer: This setup utilizes community-maintained tools (`claude-code-router`) and is not officially supported by Anthropic or Google.*
