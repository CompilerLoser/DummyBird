project(utils)

include_directories(
    .
)

add_library(${PROJECT_NAME} STATIC 
    helper.cpp
)