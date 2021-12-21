cmake_minimum_required(VERSION 3.0)

project(dummybird)

option(TIME OFF)

include_directories(
    ${CMAKE_SOURCE_DIR}
)


set(CMAKE_BUILD_TYPE Debug)

if (TIME)
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DTIME)
endif (TIME)



add_subdirectory(utils)

aux_source_directory(src SRC)

add_library(${PROJECT_NAME} STATIC ${SRC})

add_executable(${PROJECT_NAME}_test test/test.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} utils)  


