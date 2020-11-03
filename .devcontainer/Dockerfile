FROM nvidia/opencl:devel-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive \ 
    apt-get update && apt-get install -y -q \
    build-essential \
    gcc-8 g++-8 gpgconf \
    ocl-icd-opencl-dev opencl-clhpp-headers \
    git cmake gdb wget unzip python

ENV ANDROID_NDK_HOME /opt/android-ndk
ENV ANDROID_NDK_VERSION r21c

# ------------------------------------------------------
# --- Android NDK (https://github.com/bitrise-io/android-ndk/blob/master/Dockerfile)

# download
RUN mkdir /opt/android-ndk-tmp && \
    cd /opt/android-ndk-tmp && \
    wget -q https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip && \
    # uncompress
    unzip -q android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip && \
    # move to its final location
    mv ./android-ndk-${ANDROID_NDK_VERSION} ${ANDROID_NDK_HOME} && \
    # remove temp dir
    cd ${ANDROID_NDK_HOME} && \
    rm -rf /opt/android-ndk-tmp

# add to PATH
ENV PATH ${PATH}:${ANDROID_NDK_HOME}

RUN mkdir /mnn
WORKDIR /mnn