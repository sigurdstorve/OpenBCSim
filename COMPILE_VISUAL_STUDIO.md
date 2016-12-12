# Notes on how to build

## On Windows with Visual Studio 2013 Community Edition
The recommended way is to do it stepwise as sketched out below. This was tested with VS2013, but the same steps should apply to other versions of Visual Studio as well.

### Step 1: Build basic library, CPU only
- Start ```cmake-gui```, create a build directory and point CMake to the "src" folder
- Press "Configure" and select "Visual Studio 12 2013 Win64". Press "Finish".
- Ensure that the Boost library was found successfully.
- Press "Generate" and exit the CMake GUI.
- Open the generated .sln file with Visual Studio.
- Select "Release" configuration.
- Build the solution

### Step 2: Enable "Utils" library
- In the CMake GUI, turn on the option "BCSIM_BUILD_UTILS"
- Press "Configure"
- Ensure that the HDF5 library was found successfully.
- Press "Generate"
- Exit CMake GUI and rebuild in Visual Studio.

### Step 3: Enable the Qt5 GUI
- In the CMake GUI, turn on the option "BCSIM_BUILD_QT5_GUI".
- Press "Configure"
- Ensure that the Qt5 library was found correctly.
- Press "Generate"
- Rebuild solution.
- Set the "BCSimGUI" project as startup project (right-click and select "Set as StartUp project"). 
- Copy required DLLs into the build folder, eg. qt5gui/Release for a "Release" build. Otherwise it will not be possible to launch the program.
    - Qt5 DLLs: **Qt5Core.dll**, **Qt5Gui.dll**, **Qt5OpenGL.dll**, and **Qt5Widgets.dll**
    - HDF5 DLLs: **hdf5.dll** and **hdf5_cpp.dll**
- Launch (Ctrl+F5)
- Should now see a log window and the GUI window.
- Recommended to now maximize the main window size.
- In the "Simulate" menu, press "Simulate".
- You should now see a simulated B-mode scan of the default auto-generate left-ventricle phantom.
  The default resolution is quite poor, this can be improved by increasing the number of lines (```#lines```).

### Step 4: Enable the GPU algorithms (optional, requires an NVIDIA GPU)
TBD

### Step 5: Enable the Python interface (for scripting)
TBD

### Steo 6: Enable the rest
TBD
