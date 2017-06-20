cmake_minimum_required(VERSION 2.8)
project( Platos )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(SOURCES
 	${CMAKE_CURRENT_SOURCE_DIR}/licenseRecognizer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/hotShit.cpp
)
add_executable( licenseRecog ${SOURCES})
target_link_libraries( licenseRecog ${OpenCV_LIBS} )
