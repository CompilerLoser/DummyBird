cmake_minimum_required(VERSION 3.0)

project(dummybird)

set(PYTHON_INCLUDE_DIRS "/home/v-weifanghu/anaconda3/include/python3.9")
set(PYTHON_LIBRARYS "/home/v-weifanghu/anaconda3/lib/libpython3.9.so")
set(PYTHON_LINK_LIBRARYS "/home/v-weifanghu/anaconda3/lib/python3.9/config-3.9-x86_64-linux-gnu")

option(TIME OFF)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

set(CMAKE_BUILD_TYPE Debug)

if (TIME)
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DTIME)
endif (TIME)


add_subdirectory(utils)

aux_source_directory(src SRC)
add_library(${PROJECT_NAME} STATIC ${SRC})

link_directories(${PYTHON_LINK_LIBRARYS})

add_executable(
    ${PROJECT_NAME}_test 
    test/test.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_test 
    ${PROJECT_NAME} 
    ${PYTHON_LIBRARYS}
    utils
)  


