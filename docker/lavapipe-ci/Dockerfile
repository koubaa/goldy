# CI image for Goldy lavapipe (Mesa software Vulkan) jobs on GitHub Actions.
# Bakes in Mesa 25.0+ / Vulkan 1.4 so CI jobs skip repeated apt-get upgrade/install.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y mesa-vulkan-drivers libgl1-mesa-dri \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++-13 \
        git \
        libgl1-mesa-dri \
        libvulkan-dev \
        libvulkan1 \
        libxcb-render0-dev \
        libxcb-shape0-dev \
        libxcb-xfixes0-dev \
        libxcb1-dev \
        libxkbcommon-dev \
        mesa-vulkan-drivers \
        ninja-build \
        pkg-config \
        vulkan-tools \
        vulkan-validationlayers \
    && rm -rf /var/lib/apt/lists/*

ENV VK_LAYER_PATH=""
ENV CXX=g++-13

# Resolve lavapipe ICD at build time (Ubuntu may ship lvp_icd.json or lvp_icd.x86_64.json).
RUN LVP_ICD="$(find /usr/share/vulkan/icd.d -name 'lvp_icd*.json' | head -1)" \
    && test -n "$LVP_ICD" \
    && ln -sf "$LVP_ICD" /etc/goldy-lavapipe-icd.json \
    && echo "VK_ICD_FILENAMES=/etc/goldy-lavapipe-icd.json" >> /etc/environment
ENV VK_ICD_FILENAMES=/etc/goldy-lavapipe-icd.json

# Fail the image build if lavapipe does not expose Vulkan 1.4.
RUN DISPLAY= vulkaninfo --summary 2>&1 | tee /tmp/vulkaninfo-summary.txt \
    && grep -E 'apiVersion[[:space:]]+= 1\.4\.' /tmp/vulkaninfo-summary.txt \
    && grep -Ei 'deviceName.*(llvmpipe|lavapipe)' /tmp/vulkaninfo-summary.txt

WORKDIR /workspace
