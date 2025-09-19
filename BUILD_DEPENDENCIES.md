# Build Dependencies Guide

This document outlines the build dependencies required for the LLM inference pipeline, specifically for compiling `llama-cpp-python`.

## Required Build Dependencies

### Ubuntu/Debian Systems
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    gcc-12 \
    g++-12 \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv
```

### RHEL/CentOS Systems
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake \
    gcc11 \
    gcc11-c++ \
    git \
    wget \
    curl \
    python3-devel \
    python3-pip
```

## Why GCC 12?

The `llama-cpp-python` package requires GCC 12 specifically because:
- **C++17 Support**: Better C++17 standard library support
- **CUDA Compatibility**: Improved compatibility with CUDA toolkits
- **Performance Optimizations**: Better optimization for ML workloads
- **Memory Management**: Improved memory handling for large models

## Environment Variables

After installing the dependencies, set the compiler environment variables:

```bash
export CC=gcc-12
export CXX=g++-12
```

### Persistent Setup
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export CC=gcc-12' >> ~/.bashrc
echo 'export CXX=g++-12' >> ~/.bashrc
source ~/.bashrc
```

## Verification

### Check Compiler Versions
```bash
gcc-12 --version
g++-12 --version
cmake --version
```

### Test llama-cpp-python Installation
```bash
python -c "
import llama_cpp
print('✅ llama-cpp-python installed successfully')
print(f'Version: {llama_cpp.__version__}')
"
```

## Troubleshooting

### Common Issues

**1. Compiler Not Found**
```bash
# Error: Could not find compiler set in environment variable CC: gcc-12
# Solution: Install GCC 12
sudo apt install -y gcc-12 g++-12
```

**2. CMake Configuration Failed**
```bash
# Error: CMAKE_C_COMPILER not set
# Solution: Set environment variables
export CC=gcc-12
export CXX=g++-12
```

**3. Build Dependencies Missing**
```bash
# Error: No such file or directory: 'gcc'
# Solution: Install build-essential
sudo apt install -y build-essential
```

**4. Python Development Headers Missing**
```bash
# Error: Python.h not found
# Solution: Install python3-dev
sudo apt install -y python3-dev
```

### Docker Build Issues

If building with Docker fails:

```bash
# Rebuild with clean cache
docker build --no-cache -t llm-inference .

# Check build logs
docker build --progress=plain -t llm-inference .

# Verify compiler in container
docker run --rm llm-inference gcc-12 --version
```

## Performance Notes

### Compilation Time
- **First build**: 5-10 minutes (depending on system)
- **Subsequent builds**: 1-2 minutes (with cache)
- **Clean builds**: 5-10 minutes

### Memory Requirements
- **Build process**: 2-4GB RAM
- **Compilation**: 1-2GB RAM peak
- **Disk space**: 500MB-1GB for build artifacts

## Alternative Installation Methods

### Pre-compiled Wheels
If compilation fails, try pre-compiled wheels:
```bash
pip install llama-cpp-python --only-binary=all
```

### Conda Installation
```bash
conda install -c conda-forge llama-cpp-python
```

### System Package Manager
Some distributions provide pre-compiled packages:
```bash
# Ubuntu (if available)
sudo apt install python3-llama-cpp-python

# Arch Linux
sudo pacman -S python-llama-cpp-python
```

## Best Practices

1. **Always use virtual environments** to avoid conflicts
2. **Set compiler environment variables** before installation
3. **Install build dependencies first** before Python packages
4. **Use specific GCC versions** (GCC 12 recommended)
5. **Clean build cache** if encountering issues

## References

- [llama-cpp-python Installation Guide](https://github.com/abetlen/llama-cpp-python#installation)
- [GCC 12 Release Notes](https://gcc.gnu.org/gcc-12/changes.html)
- [CMake Documentation](https://cmake.org/documentation/)
