cmake_minimum_required(VERSION 3.5)

include_directories(${AADC_SOURCE_DIR}/include)
include_directories(${AADC_SOURCE_DIR}/3rdparty)
include_directories(${AADC_SOURCE_DIR}/3rdparty/adept)

find_package (Threads)

add_executable(xva_server XVAJobRequest.cpp XVAServer.cpp)

target_link_libraries(xva_server aadc) 

target_link_libraries (xva_server ${CMAKE_THREAD_LIBS_INIT})
