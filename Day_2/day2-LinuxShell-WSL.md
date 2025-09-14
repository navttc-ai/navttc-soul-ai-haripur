# Linux Shell Fundamentals & WSL Setup Tutorial

## Introduction

This tutorial provides developers and engineers with essential Linux command line skills and Windows Subsystem for Linux (WSL) setup. Whether you're working with cloud computing, AI/ML development, or server management, mastering the command line interface is crucial for modern development workflows.

### Why Learn the Command Line?

- **Cloud Computing & Servers**: Most production environments run on Linux (AWS, Google Cloud, Azure)
- **Development Efficiency**: Command line tools are often faster and more powerful than GUI alternatives
- **Automation**: Script repetitive tasks and build automated workflows
- **Direct System Control**: Access system features not available through graphical interfaces

---

## Part 1: Setting Up WSL (Windows Users)

### What is WSL?

Windows Subsystem for Linux (WSL) allows you to run a native Linux environment directly on Windows without requiring a virtual machine. It's ideal for developers who need Linux tools while maintaining their Windows workflow.

### System Requirements

Before installation, ensure your system meets these requirements:

- **Windows 10**: Version 2004 or higher
- **Windows 11**: Any version
- Administrator access to your computer

If your Windows version is older, perform a Windows Update first.

### Installation Steps

#### Method 1: Simple Installation (Recommended)

1. **Open PowerShell or Command Prompt as Administrator**
   - Right-click Start button → "Windows PowerShell (Admin)" or "Command Prompt (Admin)"

2. **Run the installation command:**
   ```bash
   wsl --install
   ```

3. **Restart your computer** when prompted

4. **Complete first-time setup:**
   - Create a UNIX username (lowercase, no spaces)
   - Set a secure password
   - Confirm password

#### Method 2: Custom Distribution

If you prefer a specific Linux distribution:

1. **List available distributions:**
   ```bash
   wsl.exe --list --online
   ```

2. **Install specific distribution:**
   ```bash
   wsl.exe --install <DistroName>
   ```
   
   Example for Debian:
   ```bash
   wsl.exe --install Debian
   ```

### Verification

After setup, test your installation:
```bash
ls -l
```

You should see your home directory contents listed in long format.

---

## Part 2: Linux Basics & Navigation

### Understanding the Interface

| Interface Type | Description | Use Case | Example |
|----------------|-------------|----------|---------|
| **GUI** (Graphical User Interface) | Visual interface with windows, icons, menus | User-friendly for beginners | Windows Desktop, macOS |
| **CLI** (Command Line Interface) | Text-based interface | Powerful and efficient for experts | Linux Terminal (Bash) |

### Getting Started with the Terminal

1. **Open your terminal application**
2. **Understand the prompt structure:**
   ```
   user@computer:~$
   ```
   - `user`: Your username
   - `computer`: Machine name
   - `~`: Current directory (home directory)
   - `$`: Standard user prompt (`#` for root user)

3. **Basic command structure:**
   ```bash
   command [options] [arguments]
   ```

### Core Navigation Commands

#### Essential Directory Commands

| Command | Description | Example |
|---------|-------------|---------|
| `pwd` | Print Working Directory - shows current location | `pwd` |
| `ls` | List files and directories | `ls` |
| `cd <directory>` | Change Directory - navigate to different folder | `cd Documents` |

#### Navigation Practice Commands

```bash
# Basic listing
ls

# Long format with details
ls -l

# Show all files (including hidden)
ls -a

# Combine options
ls -la

# Navigation shortcuts
cd ..        # Go up one directory
cd ~         # Go to home directory
cd /         # Go to root directory
cd -         # Go to previous directory
```

### Getting Help

The most important command for learning:

```bash
# Show manual for any command
man <command_name>

# Examples
man ls
man cd
man chmod
```

**Navigation in manual pages:**
- Use arrow keys to scroll
- Press `q` to quit
- Press `/` to search
- Press `n` for next search result

---

## Part 3: File and Directory Management

### Viewing File Contents

| Command | Description | Best Use Case |
|---------|-------------|---------------|
| `cat <file>` | Display entire file content | Small files |
| `less <file>` | View content page by page | Large files |
| `more <file>` | Similar to less (older version) | Basic paging |
| `head <file>` | Show first 10 lines | Quick file preview |
| `tail <file>` | Show last 10 lines | Log file monitoring |

#### Advanced Viewing Options

```bash
# Show specific number of lines
head -n 20 filename.txt    # First 20 lines
tail -n 5 filename.txt     # Last 5 lines

# Follow file changes (useful for logs)
tail -f logfile.log
```

### Creating and Editing Files

#### File Creation

```bash
# Create empty file
touch filename.txt

# Create multiple files
touch file1.txt file2.txt file3.txt

# Create file with specific timestamp
touch -t 202312251200 christmas_file.txt
```

#### Text Editing with Nano

```bash
# Open or create file in nano editor
nano filename.txt
```

**Nano keyboard shortcuts:**
- `Ctrl + O`: Save file (Write Out)
- `Ctrl + X`: Exit editor
- `Ctrl + K`: Cut line
- `Ctrl + U`: Paste line
- `Ctrl + W`: Search text

### Directory and File Operations

#### Directory Management

```bash
# Create single directory
mkdir project_folder

# Create nested directories
mkdir -p projects/ai_course/day2

# Create multiple directories
mkdir dir1 dir2 dir3
```

#### Copying and Moving

```bash
# Copy file
cp source.txt destination.txt

# Copy directory recursively
cp -r source_directory/ destination_directory/

# Move/rename file
mv oldname.txt newname.txt

# Move file to different directory
mv file.txt /path/to/destination/

# Move multiple files
mv *.txt documents/
```

#### Removing Files and Directories

```bash
# Remove file
rm filename.txt

# Remove multiple files
rm file1.txt file2.txt

# Remove empty directory
rmdir empty_directory

# Remove directory and all contents (DANGEROUS!)
rm -r directory_name
```

> ⚠️ **Critical Warning**: The `rm -r` command permanently deletes directories and ALL contents. There is no recycle bin in Linux. Always double-check your command before executing.

---

## Part 4: System Management & Permissions

### Understanding File Permissions

File permissions in Linux follow this format: `rwxrwxrwx`

| Position | User Type | Permission Types |
|----------|-----------|------------------|
| 1-3 | Owner (user) | r(read) w(write) x(execute) |
| 4-6 | Group | r(read) w(write) x(execute) |
| 7-9 | Others | r(read) w(write) x(execute) |

#### Viewing Permissions

```bash
# List files with permissions
ls -l

# Example output explanation:
# -rw-r--r-- 1 user group 1024 Dec 25 12:00 file.txt
# - : file type (- = file, d = directory)
# rw-r--r-- : permissions (owner: rw-, group: r--, others: r--)
```

#### Modifying Permissions

```bash
# Make file executable
chmod +x script.sh

# Remove write permission for group and others
chmod go-w sensitive_file.txt

# Set specific permissions (numerical method)
chmod 755 script.sh  # rwxr-xr-x
chmod 644 document.txt  # rw-r--r--
```

**Common Permission Numbers:**
- `755`: rwxr-xr-x (executable files)
- `644`: rw-r--r-- (regular files)
- `600`: rw------- (private files)
- `777`: rwxrwxrwx (all permissions - rarely recommended)

### Process Management

#### Viewing Processes

```bash
# List your processes
ps

# List all processes with details
ps aux

# Real-time process monitor
top

# More user-friendly process viewer (if available)
htop
```

#### Managing Processes

```bash
# Kill process by PID (Process ID)
kill 1234

# Force kill process
kill -9 1234

# Kill process by name
killall process_name

# Run command in background
command_name &

# Bring background job to foreground
fg
```

#### Administrative Commands

```bash
# Run command as administrator
sudo command_name

# Switch to root user (use with caution)
sudo su

# Change file ownership
sudo chown user:group filename

# System control
shutdown          # Shutdown system
shutdown -r now   # Restart immediately
shutdown -h +60   # Shutdown in 60 minutes
```

---

## Part 5: Package Management

### APT Package Manager (Debian/Ubuntu)

APT (Advanced Package Tool) manages software installation and updates.

#### Essential APT Commands

```bash
# Update package list
sudo apt update

# Upgrade all installed packages
sudo apt upgrade

# Update and upgrade in one command
sudo apt update && sudo apt upgrade

# Install new software
sudo apt install package_name

# Install multiple packages
sudo apt install git curl wget vim

# Remove package
sudo apt remove package_name

# Remove package and configuration files
sudo apt purge package_name

# Remove unused packages
sudo apt autoremove

# Search for packages
apt search search_term
```

#### Common Development Packages

```bash
# Essential development tools
sudo apt install build-essential git curl wget

# Python development
sudo apt install python3 python3-pip

# Node.js development
sudo apt install nodejs npm

# Text editors
sudo apt install vim nano

# System utilities
sudo apt install htop tree unzip
```

---

## Part 6: Advanced Features

### Input/Output Redirection

Control where command output goes and how input is processed.

#### Output Redirection

```bash
# Redirect output to file (overwrite)
ls -l > file_list.txt

# Append output to file
ls -l >> file_list.txt

# Redirect error output
command 2> error_log.txt

# Redirect both output and errors
command > output.txt 2> error.txt

# Combine output and errors
command > combined.txt 2>&1
```

#### Pipes and Filtering

```bash
# Pipe output to another command
ls -l | grep ".txt"

# Chain multiple commands
cat file.txt | grep "pattern" | sort | uniq

# Count lines, words, characters
wc filename.txt

# Search for patterns in files
grep "search_term" filename.txt

# Sort file contents
sort filename.txt

# Remove duplicate lines
uniq filename.txt
```

### Creating Command Aliases

```bash
# Create temporary alias
alias ll='ls -l'
alias la='ls -la'
alias ..='cd ..'

# Make aliases permanent (add to ~/.bashrc)
echo "alias ll='ls -l'" >> ~/.bashrc
source ~/.bashrc
```

---

## Part 7: Practical Exercises

### Exercise 1: Basic Navigation and File Management

```bash
# 1. Check current directory
pwd

# 2. List all files including hidden
ls -la

# 3. Create project structure
mkdir -p ~/projects/linux_tutorial/practice

# 4. Navigate to practice directory
cd ~/projects/linux_tutorial/practice

# 5. Create test files
touch readme.txt notes.md script.sh

# 6. Add execute permission to script
chmod +x script.sh

# 7. List files with permissions
ls -l
```

### Exercise 2: Text File Manipulation

```bash
# 1. Create a file with content
echo "This is line 1" > sample.txt
echo "This is line 2" >> sample.txt
echo "This is line 3" >> sample.txt

# 2. View file contents
cat sample.txt

# 3. Copy and modify
cp sample.txt backup.txt
echo "This is line 4" >> sample.txt

# 4. Compare files
diff sample.txt backup.txt

# 5. Search for content
grep "line 2" sample.txt
```

### Exercise 3: System Information

```bash
# 1. Check system information
uname -a

# 2. Check disk usage
df -h

# 3. Check memory usage
free -h

# 4. Check running processes
ps aux | head -10

# 5. Check network information
ip addr show
```

---

## Homework Assignment

Complete this practical assignment to reinforce your learning:

### Task Requirements

1. **Create directory structure:**
   ```bash
   ~/navttc_ai/homework/day2/
   ```

2. **Create and edit a documentation file:**
   ```bash
   cd ~/navttc_ai/homework/day2/
   nano commands.txt
   ```

3. **Document three commands:**
   In the `commands.txt` file, write:
   - Command name and syntax
   - What the command does
   - An example of how to use it

4. **Practice navigation:**
   - Navigate between directories using `cd`
   - Use `pwd` to verify your location
   - Use `ls` to explore directory contents

### Solution Template

Create this content in your `commands.txt` file:

```
Linux Commands Documentation
============================

1. Command: ls -la
   Purpose: Lists all files and directories in long format, including hidden files
   Example: ls -la /home/user/documents

2. Command: chmod +x filename
   Purpose: Adds execute permission to a file, making it runnable
   Example: chmod +x myscript.sh

3. Command: grep "pattern" filename
   Purpose: Searches for a specific pattern within a file
   Example: grep "error" logfile.txt
```

---

## Command Reference Quick Guide

### Navigation
| Command | Description |
|---------|-------------|
| `pwd` | Show current directory |
| `ls` | List directory contents |
| `ls -la` | List all files with details |
| `cd <dir>` | Change directory |
| `cd ..` | Go up one level |
| `cd ~` | Go to home directory |

### File Operations
| Command | Description |
|---------|-------------|
| `touch <file>` | Create empty file |
| `cat <file>` | Display file contents |
| `nano <file>` | Edit file |
| `cp <src> <dest>` | Copy file |
| `mv <src> <dest>` | Move/rename file |
| `rm <file>` | Delete file |

### Directory Operations
| Command | Description |
|---------|-------------|
| `mkdir <dir>` | Create directory |
| `rmdir <dir>` | Remove empty directory |
| `rm -r <dir>` | Remove directory and contents |

### System Management
| Command | Description |
|---------|-------------|
| `man <command>` | Show manual |
| `chmod <perms> <file>` | Change permissions |
| `sudo <command>` | Run as administrator |
| `ps` | List processes |
| `kill <PID>` | Stop process |

### Package Management (Debian/Ubuntu)
| Command | Description |
|---------|-------------|
| `sudo apt update` | Update package list |
| `sudo apt upgrade` | Upgrade packages |
| `sudo apt install <package>` | Install package |

---

## Next Steps

After mastering these Linux fundamentals, you'll be ready to:

1. **Advanced Shell Scripting**: Automate tasks with bash scripts
2. **Server Management**: Deploy and manage applications on Linux servers
3. **Cloud Computing**: Work effectively with AWS, Google Cloud, or Azure
4. **Development Workflows**: Use Linux-based development tools and environments
5. **Container Technologies**: Work with Docker and Kubernetes

The command line skills you've learned here form the foundation for advanced system administration, cloud computing, and development workflows. Practice regularly to build muscle memory and confidence with these essential tools.
