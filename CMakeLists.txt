cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 添加 Google Test 子模块
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# 设置为静态库
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# 获取 GoogleTest
FetchContent_MakeAvailable(googletest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 执行程序
add_executable(crun_cpp crun.h crun.cpp)

enable_testing()

# 测试程序
add_executable(crun_test crun.cpp crun_unittest.cc)
target_compile_definitions(crun_test PRIVATE RUN_TESTS)
target_link_libraries(
  crun_test
  gtest_main
)

include(GoogleTest)
add_test(NAME example_test COMMAND crun_test)