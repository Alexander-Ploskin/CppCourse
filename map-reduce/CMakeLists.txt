cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(map-reduce VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP)

add_executable(map-reduce main.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}
    # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
    $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
