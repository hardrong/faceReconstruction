# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/FaceReconstruction-master/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/FaceReconstruction-master/build1

# Include any dependencies generated for this target.
include CMakeFiles/fit-model.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fit-model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fit-model.dir/flags.make

CMakeFiles/fit-model.dir/fit-model.cpp.o: CMakeFiles/fit-model.dir/flags.make
CMakeFiles/fit-model.dir/fit-model.cpp.o: /root/FaceReconstruction-master/examples/fit-model.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /root/FaceReconstruction-master/build1/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fit-model.dir/fit-model.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fit-model.dir/fit-model.cpp.o -c /root/FaceReconstruction-master/examples/fit-model.cpp

CMakeFiles/fit-model.dir/fit-model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fit-model.dir/fit-model.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /root/FaceReconstruction-master/examples/fit-model.cpp > CMakeFiles/fit-model.dir/fit-model.cpp.i

CMakeFiles/fit-model.dir/fit-model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fit-model.dir/fit-model.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /root/FaceReconstruction-master/examples/fit-model.cpp -o CMakeFiles/fit-model.dir/fit-model.cpp.s

CMakeFiles/fit-model.dir/fit-model.cpp.o.requires:
.PHONY : CMakeFiles/fit-model.dir/fit-model.cpp.o.requires

CMakeFiles/fit-model.dir/fit-model.cpp.o.provides: CMakeFiles/fit-model.dir/fit-model.cpp.o.requires
	$(MAKE) -f CMakeFiles/fit-model.dir/build.make CMakeFiles/fit-model.dir/fit-model.cpp.o.provides.build
.PHONY : CMakeFiles/fit-model.dir/fit-model.cpp.o.provides

CMakeFiles/fit-model.dir/fit-model.cpp.o.provides.build: CMakeFiles/fit-model.dir/fit-model.cpp.o

# Object files for target fit-model
fit__model_OBJECTS = \
"CMakeFiles/fit-model.dir/fit-model.cpp.o"

# External object files for target fit-model
fit__model_EXTERNAL_OBJECTS =

fit-model: CMakeFiles/fit-model.dir/fit-model.cpp.o
fit-model: CMakeFiles/fit-model.dir/build.make
fit-model: /usr/lib64/libopencv_calib3d.so
fit-model: /usr/lib64/libopencv_contrib.so
fit-model: /usr/lib64/libopencv_core.so
fit-model: /usr/lib64/libopencv_features2d.so
fit-model: /usr/lib64/libopencv_flann.so
fit-model: /usr/lib64/libopencv_highgui.so
fit-model: /usr/lib64/libopencv_imgproc.so
fit-model: /usr/lib64/libopencv_legacy.so
fit-model: /usr/lib64/libopencv_ml.so
fit-model: /usr/lib64/libopencv_objdetect.so
fit-model: /usr/lib64/libopencv_photo.so
fit-model: /usr/lib64/libopencv_stitching.so
fit-model: /usr/lib64/libopencv_superres.so
fit-model: /usr/lib64/libopencv_ts.so
fit-model: /usr/lib64/libopencv_video.so
fit-model: /usr/lib64/libopencv_videostab.so
fit-model: /usr/lib64/libboost_system-mt.so
fit-model: /usr/lib64/libboost_filesystem-mt.so
fit-model: /usr/lib64/libboost_program_options-mt.so
fit-model: CMakeFiles/fit-model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable fit-model"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fit-model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fit-model.dir/build: fit-model
.PHONY : CMakeFiles/fit-model.dir/build

CMakeFiles/fit-model.dir/requires: CMakeFiles/fit-model.dir/fit-model.cpp.o.requires
.PHONY : CMakeFiles/fit-model.dir/requires

CMakeFiles/fit-model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fit-model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fit-model.dir/clean

CMakeFiles/fit-model.dir/depend:
	cd /root/FaceReconstruction-master/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/FaceReconstruction-master/examples /root/FaceReconstruction-master/examples /root/FaceReconstruction-master/build1 /root/FaceReconstruction-master/build1 /root/FaceReconstruction-master/build1/CMakeFiles/fit-model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fit-model.dir/depend

