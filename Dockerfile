# Use the official CUDA image as a base
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    xz-utils \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    xvfb \
    libglu1 \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 3.6.9
RUN wget https://download.blender.org/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    tar -xf /tmp/blender.tar.xz -C /opt && \
    rm /tmp/blender.tar.xz

ENV PATH="/opt/blender-3.6.9-linux-x64:${PATH}"

# Set up a working directory
WORKDIR /workspace

# Copy the setup script
COPY setup.py .

# Run the setup script to install PyYAML
RUN /opt/blender-3.6.9-linux-x64/3.6/python/bin/python3.10 setup.py

# Copy the rest of your code
COPY . .

# Ensure the script has execute permissions
RUN chmod +x /workspace/DataGenerator.py

# # Set Blender to use GPU and OptiX
# RUN echo "import bpy; bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'; bpy.context.preferences.addons['cycles'].preferences.get_devices();" > set_gpu.py

# Run the script
CMD ["blender", "/workspace/Empty.blend", "--background", "--python", "/workspace/DataGenerator.py"]
