FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Base packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    zip \
    jq \
    sudo \
    software-properties-common \
    build-essential \
    clang \
    lld \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    scons \
    openjdk-17-jdk \
    xz-utils \
    locales \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    binutils-arm-linux-gnueabihf \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# Create runner user
# -----------------------------------------------------------------------------
RUN useradd -m -s /bin/bash runner && \
    usermod -aG sudo runner && \
    echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/runner

# -----------------------------------------------------------------------------
# Install GitHub Actions runner
# -----------------------------------------------------------------------------
ARG RUNNER_VERSION=2.329.0

RUN curl -L \
    https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    -o actions-runner.tar.gz && \
    tar xzf actions-runner.tar.gz && \
    rm actions-runner.tar.gz

# Runner dependencies
RUN ./bin/installdependencies.sh

# -----------------------------------------------------------------------------
# Emscripten
# -----------------------------------------------------------------------------
RUN git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk && \
    cd /opt/emsdk && \
    ./emsdk install latest && \
    ./emsdk activate latest

ENV EMSDK=/opt/emsdk
ENV PATH="${EMSDK}:${EMSDK}/upstream/emscripten:${PATH}"

# -----------------------------------------------------------------------------
# Android SDK (WITHOUT NDK)
# setup-ndk action will install the requested NDK version.
# -----------------------------------------------------------------------------
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk

RUN mkdir -p ${ANDROID_HOME}/cmdline-tools && \
    cd /tmp && \
    wget https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip && \
    unzip commandlinetools-linux-13114758_latest.zip && \
    mv cmdline-tools ${ANDROID_HOME}/cmdline-tools/latest && \
    yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses && \
    ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager \
        "platform-tools" \
        "platforms;android-35" \
        "build-tools;35.0.0"

ENV PATH="${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}"

RUN chown -R runner:runner /home/runner /opt/android-sdk /opt/emsdk

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && chown runner:runner /entrypoint.sh

USER runner



ENTRYPOINT ["/entrypoint.sh"]
