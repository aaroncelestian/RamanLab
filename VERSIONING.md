# RamanLab Versioning Guide

## Versioning System

RamanLab uses **Semantic Versioning (SemVer)** with the format: `MAJOR.MINOR.PATCH`

### Version Number Rules

- **MAJOR** version (X.0.0) - Incompatible API changes, breaking changes
- **MINOR** version (0.X.0) - New functionality, backward compatible
- **PATCH** version (0.0.X) - Bug fixes, backward compatible

### Current Version: 1.0.0 "Debut"

This is the inaugural stable release of RamanLab Qt6.

## Version History

### 1.0.0 - "Debut" (January 26, 2025)
- Initial stable release of RamanLab Qt6
- Complete migration from tkinter to Qt6 framework
- Cross-platform compatibility
- Modern dependency management

## Future Versioning Strategy

### Minor Releases (1.X.0)
Examples of features that warrant a minor version bump:
- New analysis algorithms
- Additional file format support
- New GUI components or tools
- Enhanced visualization features
- Database schema improvements (backward compatible)

### Patch Releases (1.0.X)
Examples of changes for patch releases:
- Bug fixes
- Performance improvements
- Documentation updates
- Minor UI improvements
- Security patches

### Major Releases (X.0.0)
Examples of changes that require a major version bump:
- Breaking API changes
- Incompatible database schema changes
- Major architectural changes
- Removal of deprecated features
- Framework changes (e.g., Qt6 to Qt7)

## Release Process

### 1. Update Version Information

Update the following files:
- `version.py` - Main version definition
- `VERSION.txt` - Human-readable release notes
- `README.md` - Update version references if needed

### 2. Version.py Updates

```python
__version__ = "1.1.0"  # Update version
__version_info__ = (1, 1, 0)  # Update tuple
__release_date__ = "YYYY-MM-DD"  # Update date
__release_name__ = "Release Name"  # Update name

# Add to __changes__ dictionary
"1.1.0": {
    "date": "YYYY-MM-DD",
    "name": "Release Name",
    "description": "Brief description",
    "major_features": [...],
    "bug_fixes": [...],
    "breaking_changes": [...]  # If any
}
```

### 3. Testing

Before releasing:
- Run `python check_dependencies.py` to verify setup
- Test on target platforms
- Verify all features work correctly
- Check documentation is up to date

### 4. Git Tagging

```bash
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0
```

## Version Utilities

The `version.py` module provides utility functions:

```python
from version import get_version, get_version_info, get_full_version

# Get version string
version = get_version()  # "1.0.0"

# Get version tuple
version_info = get_version_info()  # (1, 0, 0)

# Get full version information
full_info = get_full_version()  # Complete version dict
```

## Pre-release Versions

For development and testing:
- **Alpha**: `1.1.0a1`, `1.1.0a2` (early development)
- **Beta**: `1.1.0b1`, `1.1.0b2` (feature complete, testing)
- **Release Candidate**: `1.1.0rc1` (final testing)

Example:
```python
__version__ = "1.1.0a1"  # Alpha version
__release_status__ = "alpha"  # Update status
```

## Compatibility Information

Always update compatibility info in `version.py`:
- `__python_requires__` - Minimum Python version
- `__qt_version__` - Required Qt version
- `__platforms__` - Supported platforms

## Distribution

When creating releases:
1. Update all version files
2. Test thoroughly
3. Create git tag
4. Generate distribution packages
5. Update documentation
6. Announce release

---

*This versioning guide ensures consistent and predictable releases for RamanLab users and developers.* 