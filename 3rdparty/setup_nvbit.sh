#!/bin/bash

# Script to download and setup NVBit 1.7.5 release binaries

NVBIT_VERSION="1.7.5"
NVBIT_URL="https://github.com/NVlabs/NVBit/releases/download/v${NVBIT_VERSION}/nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2"
NVBIT_ARCHIVE="nvbit-Linux-x86_64-${NVBIT_VERSION}.tar.bz2"
NVBIT_DIR="nvbit_release"

cd "$(dirname "$0")"

echo "Setting up NVBit ${NVBIT_VERSION} release binaries..."

# Check if NVBit release already exists
if [ -d "${NVBIT_DIR}" ]; then
    echo "NVBit ${NVBIT_VERSION} release already exists at 3rdparty/${NVBIT_DIR}"
    exit 0
fi

# Download NVBit release
echo "Downloading NVBit ${NVBIT_VERSION}..."
if [ ! -f "${NVBIT_ARCHIVE}" ]; then
    wget "${NVBIT_URL}" || {
        echo "Failed to download NVBit ${NVBIT_VERSION}"
        exit 1
    }
fi

# Extract NVBit
echo "Extracting NVBit ${NVBIT_VERSION}..."
tar -xjf "${NVBIT_ARCHIVE}" || {
    echo "Failed to extract NVBit archive"
    exit 1
}

# Rename to nvbit_release
if [ -d "nvbit_release_x86_64" ]; then
    mv nvbit_release_x86_64 "${NVBIT_DIR}"
fi

# Clean up archive
rm -f "${NVBIT_ARCHIVE}"

echo "NVBit ${NVBIT_VERSION} setup complete at 3rdparty/${NVBIT_DIR}"