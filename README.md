# BlendCon
## Official implementation of [Make it till you fake it: Construction-centric computational framework for simultaneous image synthetization and multimodal labeling](https://doi.org/10.1016/j.autcon.2024.105696)

BlendCon is a fully automated framework designed for synthesizing and labeling construction imagery data. It facilitates the simulation of a construction site by orchestrating 3D mobile objects against a 3D background and produces multimodal labels for target entities. This tool is especially valuable in training object detection models in construction environments.

## Configuration
Adjust the `config.yaml` file with the following sample content to set the hyperparameters for your data generation:

```yaml
# Hyperparameters for iterations of data generation
Iterations_Avatar_Location_Randomization: 5
Iterations_Lighting_Randomization: 5
Number_of_Image_Sequences: 100

# Hyperparameters for scene setup
Camera_Radius: 3
Number_of_Workers: 6
 
# Hyperparameters for rendering type
Drone_View: False

# Hyperparameters for rendering settings
max_bounces: 4                      # Sets the maximum number of light bounces for ray tracing, affecting the realism and computational load.
samples: 1024                       # Sets the number of samples per pixel for rendering, affecting image quality and render time.
tile_size: 256                     # Defines the size of render tiles, impacting render efficiency. 
adaptive_threshold: 0.001           # Sets the threshold for adaptive sampling, improving efficiency by reducing samples in areas with less noise.
resolution_x: 1024                  # Sets the horizontal resolution of the render to 1920 pixels.
resolution_y: 1024                  # Sets the vertical resolution of the render to 1920 pixels.
Framerate: 100                     # Sets the frames per second for the animation, and the number of frames to skip for each step in animation playback.
```

## Installation

### Requirements
- Blender 3.6.9

### Setup Instructions
#### Without Docker
1. Update and upgrade your system packages:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. Download and extract Blender:
   ```bash
   wget "https://download.blender.org/release/Blender3.3/blender-3.3.0-linux-x64.tar.xz"
   sudo tar -xvf blender-3.3.0-linux-x64.tar.xz
   rm blender-3.3.0-linux-x64.tar.xz
   mv blender-3.3.0-linux-x64 blender330
   ```

3. Add Blender to the system paths

4. Copy Avatars and Scenes into "Avatars" and "Scenes" folder

5. Install PyYmal to the Blender's python using this code snippet:
   ```bash
   blender --background --python setup.py
   ```

6. To run the data generation code, execute the following command in the terminal (ensure Blender is installed and accessible from the terminal):
   ```bash
   blender Empty.blend --background --python DataGenerator.py
   ```

#### With Docker
##### Prerequisites
Ensure you have the necessary GPU drivers and CUDA toolkit installed on your host system.

1. **Install NVIDIA Drivers and CUDA Toolkit**:
   - **Ubuntu**:
     ```sh
     sudo apt update
     sudo apt install -y nvidia-driver-470
     sudo apt install -y nvidia-cuda-toolkit
     ```
   - **Windows**:
     Download and install the latest drivers from [NVIDIA's official website](https://www.nvidia.com/Download/index.aspx).
     
2. **Install NVIDIA Container Toolkit**:
   - **Ubuntu**:
     ```sh
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```
   - **Windows**:
     Ensure Docker Desktop is configured to use the WSL 2 backend and that WSL 2 is configured correctly to use the GPU. Follow the [WSL 2 GPU setup instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

3. **Verify GPU Support in Docker**:
   Run the following command to verify that Docker can access your GPU:
   ```sh
   docker run -it --rm --gpus all ubuntu nvidia-smi
   ```

##### Build and Run the Docker Container
3. **Dockerfile**
   We created a `Dockerfile` in the root of the repo. Use that Dockerfile to build a Docker image.

4. **Build the Docker Image**:
    ```sh
    docker build -t blendcon .
    ```

5. **Run the Docker Container with GPU Support**:
    ```sh
    docker run --gpus all -it --rm -v /absolute/path/to/your/output:/workspace/Dataset blendcon
    ```
    Replace `/absolute/path/to/your/output` with the absolute path to your desired output directory on your host machine.

## Output
After running the code, the following outputs will be generated in the "Dataset" folder:
- Image sequences: A series of `.jpg` images.
- Label file: `JointTracker.pickle` containing the labels for each image.
- Depth Maps: A folder containing depth maps for each image.
- Semantic Segmentation Masks: A folder containing segmentation masks of workers.

## License
MIT License.

## Contributing
Contributions to BlendCon are welcome. Please follow the standard procedures for contributing to open-source projects.
