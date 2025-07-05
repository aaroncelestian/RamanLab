# Auto-Update Feature Documentation

## Overview

RamanLab now includes an **automatic update feature** that allows users to update the application with a single click, eliminating the need for manual terminal commands.

## How It Works

When you go to **Help → Check for Updates** and an update is available, you'll see a new dialog with four options:

1. **🔄 Auto Update** (Green button, primary action)
2. **📥 Download from GitHub** 
3. **📋 Copy 'git pull'**
4. **Later**

## Using Auto-Update

### Step 1: Check for Updates
- Go to **Help → Check for Updates** in the menu
- Wait for the system to check GitHub for new versions

### Step 2: Auto-Update Process
- Click the **🔄 Auto Update** button
- The system will:
  1. Check if you're in a git repository
  2. Check for uncommitted local changes
  3. Warn you about potential conflicts
  4. Perform the git pull operation
  5. Show progress feedback
  6. Display success message

### Step 3: Restart
- After successful update, restart RamanLab to use the new version

## Requirements

- **Git Repository**: Must be running from a git-cloned copy of RamanLab
- **Internet Connection**: Required to download updates from GitHub
- **Git Installation**: Git must be installed and accessible from command line

## Error Handling

The auto-update feature handles common issues gracefully:

### Not a Git Repository
- **Issue**: Directory is not a git repository
- **Solution**: Shows message suggesting to use "Download from GitHub" instead

### Uncommitted Changes
- **Issue**: You have modified files that haven't been committed
- **Solution**: Warns user and asks for confirmation before proceeding

### Merge Conflicts
- **Issue**: Local changes conflict with remote updates
- **Solution**: Provides detailed instructions for manual resolution

### Authentication Errors
- **Issue**: Git requires authentication credentials
- **Solution**: Suggests using manual update method or checking GitHub credentials

### Network Timeouts
- **Issue**: Slow internet connection or network problems
- **Solution**: Graceful fallback with helpful error messages

## Safety Features

- **Progress Feedback**: Shows what's happening during the update
- **Cancellation**: User can cancel the operation at any time
- **Backup Compatibility**: Maintains all existing manual update methods
- **Change Detection**: Warns before overwriting local modifications
- **Timeout Protection**: Prevents hanging on slow connections

## Fallback Methods

If auto-update doesn't work, you can still use:

1. **Manual Git**: Open terminal and run `git pull`
2. **GitHub Download**: Download ZIP from GitHub releases
3. **Copy Command**: Copy `git pull` to clipboard for terminal use

## Benefits

- **User-Friendly**: No terminal knowledge required
- **Safe**: Comprehensive error checking and warnings
- **Fast**: Direct git pull operation
- **Reliable**: Handles edge cases gracefully
- **Non-Destructive**: Preserves existing functionality

## Technical Details

The auto-update feature uses:
- `subprocess` for git operations
- `PySide6.QtWidgets.QProgressDialog` for progress feedback
- Timeout protection (10s for status, 30s for pull)
- Cross-platform compatibility (Windows, macOS, Linux)

## Troubleshooting

### "Not a Git Repository" Error
If you get this error, you likely downloaded RamanLab as a ZIP file instead of cloning it with git.

**Solution**: Clone the repository properly:
```bash
git clone https://github.com/aaroncelestian/RamanLab.git
cd RamanLab
python launch_ramanlab.py
```

### Authentication Required
If git asks for credentials, you may need to:
1. Use GitHub Personal Access Token
2. Set up SSH keys
3. Use the manual update method instead

### Update Fails
If the auto-update fails:
1. Use the "📋 Copy 'git pull'" button
2. Open terminal in RamanLab directory
3. Paste and run the command
4. Restart RamanLab

## Version History

- **v1.0.5**: Added auto-update functionality
- **v1.0.4**: Manual update methods only 