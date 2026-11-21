cmake_minimum_required(VERSION 3.18)
project(littlebit VERSION 0.1.0 LANGUAGES C)

find_package(Threads REQUIRED)
add_library(littlebit src/kernels.c src/layer.c src/threads.c)
target_include_directories(littlebit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(littlebit PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_compile_options(littlebit PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(littlebit PRIVATE Threads::Threads)

include(GNUInstallDirs)
install(TARGETS littlebit)
install(FILES include/littlebit.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CTest)
if(BUILD_TESTING)
    add_executable(test_kernels tests/test_kernels.c)
    set_target_properties(test_kernels PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
    target_link_libraries(test_kernels PRIVATE littlebit m)
    add_test(NAME kernels COMMAND test_kernels)
endif()
