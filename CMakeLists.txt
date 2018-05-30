cmake_minimum_required(VERSION 2.8.3)
project(projekt_wma)

# Find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("-- OpenCV ${OpenCV_VERSION} was found")
    include_directories(${OpenCV_INCLUDE_DIRS})
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV was not found")
endif(OpenCV_FOUND)

# Definitions
add_definitions(-DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")

# Declare a cpp executable (projekt)
add_executable(projekt src/projekt.cpp)
# Specify libraries to link a library or executable target against (projekt)
target_link_libraries(projekt ${OpenCV_LIBS})

# Declare a cpp executable (projekt1)
add_executable(projekt1 src/projekt1.cpp)
# Specify libraries to link a library or executable target against (projekt1)
target_link_libraries(projekt1 ${OpenCV_LIBS})

# Declare a cpp executable (main)
#add_executable(main src/main.cpp)
# Specify libraries to link a library or executable target against (main)
#target_link_libraries(main ${OpenCV_LIBS})
