# RamanLab Update Checker

## Overview

The RamanLab Update Checker is a built-in feature that allows users to check for and download the latest version of RamanLab directly from the GitHub repository. This feature ensures users always have access to the newest features, bug fixes, and improvements.

## Features

### 🔍 **Automatic Update Detection**
- Checks GitHub releases for newer versions
- Compares semantic version numbers (e.g., 2.6.3 vs 2.6.1)
- Fallback to commit checking if no releases are available
- Smart version comparison using the `packaging` library

### 🚀 **Multiple Update Methods**
1. **Download Latest Version**: Opens GitHub release page in browser
2. **Auto-Update (Git)**: Automatically pulls latest changes if repository is git-cloned
3. **Copy Git Command**: Copies `git pull` command to clipboard for manual execution

### 📋 **Comprehensive Update Information**
- Current vs. latest version comparison
- Release date information
- Full release notes display
- Scrollable release notes for detailed information

### 🛡️ **Safe and Reliable**
- Non-blocking asynchronous update checks
- Graceful error handling for network issues
- Progress indicators for git operations
- Confirmation dialogs for all operations

## How to Use

### From the Application Menu
1. Open RamanLab
2. Go to **Help** → **Check for Updates**
3. The update checker will automatically check for new versions
4. If updates are available, an update dialog will appear

### Update Dialog Options

When an update is available, you'll see a dialog with several options:

#### 📥 **Download Latest Version**
- Opens the GitHub release page in your default web browser
- Best for users who downloaded RamanLab as a ZIP file
- Allows manual download and installation

#### 🔄 **Auto-Update (Git)**
- Automatically updates RamanLab using `git pull`
- Only works if RamanLab was cloned from GitHub
- Shows progress dialog during update
- Requires restart after successful update

#### 📋 **Copy Git Command**
- Copies `git pull` command to clipboard
- Useful for manual terminal/command prompt execution
- Provides instructions for manual update

## Installation Requirements

The update checker requires these additional dependencies:

```bash
pip install requests>=2.25.0 packaging>=20.0 pyperclip>=1.8.0
```

These are automatically included in the RamanLab requirements.txt file.

## Technical Details

### GitHub API Integration
- Uses GitHub REST API v3
- Endpoint: `https://api.github.com/repos/aaroncelestian/RamanLab`
- Checks `/releases/latest` for version information
- Falls back to `/commits` if no releases available

### Version Comparison
- Uses semantic versioning (SemVer) comparison
- Handles version tags with or without 'v' prefix
- Robust parsing using the `packaging` library

### Git Integration
- Detects if current directory is a git repository
- Validates git installation and availability
- Handles merge conflicts and update failures gracefully

## Error Handling

The update checker handles various error scenarios:

### Network Issues
- **No Internet Connection**: Shows user-friendly error message
- **GitHub API Unavailable**: Graceful fallback with manual update instructions
- **Request Timeout**: 10-second timeout with retry suggestions

### Git Issues
- **Not a Git Repository**: Suggests using download option
- **Git Not Installed**: Provides installation instructions
- **Merge Conflicts**: Offers manual resolution guidance

### Permission Issues
- **File Access Errors**: Suggests running with appropriate permissions
- **Write Protection**: Provides alternative update methods

## Configuration

### GitHub Repository Settings
The update checker is configured for the official RamanLab repository:
- **Repository**: `aaroncelestian/RamanLab`
- **API Base**: `https://api.github.com/repos/aaroncelestian/RamanLab`
- **Web URL**: `https://github.com/aaroncelestian/RamanLab`

### Customization
To use with a different repository, modify the `UpdateChecker` class:

```python
checker = UpdateChecker(current_version="2.6.3")
checker.github_api_url = "https://api.github.com/repos/your-username/your-repo"
checker.github_repo_url = "https://github.com/your-username/your-repo"
```

## Security Considerations

### Safe Operations
- **Read-Only API Access**: Only reads public repository information
- **No Automatic Downloads**: All downloads require user confirmation
- **Local Git Operations**: Git commands only affect local repository

### Privacy
- **No User Data Collection**: Update checker doesn't send user information
- **Anonymous API Calls**: GitHub API calls don't require authentication
- **Local Version Checking**: Version comparison happens locally

## Troubleshooting

### Common Issues

#### "Update Checker Unavailable"
**Cause**: Missing dependencies (requests, packaging, pyperclip)
**Solution**: Install dependencies with `pip install requests packaging pyperclip`

#### "Git Not Available"
**Cause**: Git not installed or not in PATH
**Solution**: Install Git or use the download option

#### "Could not check for updates"
**Cause**: Network connectivity issues
**Solution**: Check internet connection and try again

#### "Git pull failed"
**Cause**: Local changes conflict with remote updates
**Solution**: Resolve conflicts manually or use fresh download

### Manual Update Process

If the automatic update checker fails, you can update manually:

1. **For Git Users**:
   ```bash
   cd /path/to/RamanLab
   git pull origin main
   ```

2. **For ZIP Download Users**:
   - Visit https://github.com/aaroncelestian/RamanLab
   - Click "Code" → "Download ZIP"
   - Extract and replace old files

## Integration with RamanLab

### Menu Integration
The update checker is integrated into the Help menu:
- **Location**: Help → Check for Updates
- **Availability**: Only shown if update checker dependencies are available
- **Fallback**: Manual update instructions if dependencies missing

### Version Information
- **Current Version**: Read from `version.py` file
- **Version Display**: Shown in About dialog and update checker
- **Version Format**: Semantic versioning (e.g., 2.6.3)

## Future Enhancements

### Planned Features
- **Automatic Update Notifications**: Check for updates on startup
- **Update Scheduling**: Configurable automatic update checks
- **Changelog Integration**: Enhanced release notes display
- **Rollback Functionality**: Ability to revert to previous versions

### Advanced Options
- **Beta Channel**: Option to receive pre-release updates
- **Update Preferences**: User-configurable update settings
- **Offline Mode**: Cached update information for offline use

## Support

For issues with the update checker:

1. **Check Dependencies**: Ensure all required packages are installed
2. **Verify Network**: Confirm internet connectivity
3. **Manual Update**: Use manual update methods as fallback
4. **Report Issues**: Submit bug reports on GitHub

## Version History

- **v2.6.1**: Initial release of update checker
- **v2.6.2**: Enhanced peak fitting UX and individual peak R² calculations
- **Future**: Planned enhancements and improvements

---

*The RamanLab Update Checker ensures you always have access to the latest features and improvements. Keep your spectral analysis tools up to date!* 