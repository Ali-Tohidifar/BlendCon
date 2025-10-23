@echo off
setlocal enabledelayedexpansion

echo Checking for existing blendcon containers...

:: Remove all running containers and images if any exist
for /f "tokens=*" %%i in ('docker ps -a -q --filter "ancestor=blendcon"') do (
    echo Removing container %%i...
    docker rm -f %%i
)

for /f "tokens=*" %%i in ('docker images -q blendcon') do (
    echo Removing image %%i...
    docker rmi -f %%i
)

echo Removing dangling images...
docker image prune -f

echo Building blendcon Docker image...
docker build -t blendcon .

:: Run four Docker containers in parallel with different volume mount points
for /L %%i in (1,1,4) do (
    set "container_name=blendcon_%%i"
    echo Running Docker container !container_name!...
    docker run --gpus all -d --rm ^
        -v "C:\Users\Windows\OneDrive - University of Toronto\UofT_PhD\Research Projects\21-09-21_Synthetic_Data\3_Blender_Data_Generator\GitRepo\BlendCon\Dataset\D%%i:/workspace/Dataset" ^
        -v "C:\Users\Windows\OneDrive - University of Toronto\UofT_PhD\Research Projects\21-09-21_Synthetic_Data\3_Blender_Data_Generator\GitRepo\BlendCon\logs_%%i:/workspace/logs" ^
        -v "D:\active_learning_dataset\3DAssets_GCPBucket\Avatars:/workspace/Avatars" ^
        -v "D:\active_learning_dataset\3DAssets_GCPBucket\Scenes:/workspace/Scenes" ^
        -v "C:\Users\Windows\OneDrive - University of Toronto\UofT_PhD\Research Projects\21-09-21_Synthetic_Data\3_Blender_Data_Generator\GitRepo\BlendCon\config.yaml:/workspace/config.yaml" ^
        --name !container_name! blendcon
)

:: Wait for all containers to finish
:wait_loop
set "all_done=true"
set "status_line="

for /L %%i in (1,1,4) do (
    set "container_name=blendcon_%%i"
    docker ps -q --filter "name=!container_name!" | findstr . >nul
    if not errorlevel 1 (
        set "status_line=!status_line!!container_name! is still running. "
        set "all_done=false"
    ) else (
        set "status_line=!status_line!!container_name! has finished. "
    )
)

:: Print status on the same line using carriage return
<nul set /p "=Checking status of containers... !status_line!`r"

if "%all_done%"=="false" (
    timeout /t 10 >nul
    goto wait_loop
)

echo All Docker containers have completed their work.

endlocal
