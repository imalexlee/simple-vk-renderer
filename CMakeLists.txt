cmake_minimum_required(VERSION 3.18)

project(simple-vk-renderer)
find_package(glfw3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_link_options("-lglfw;-lvulkan;-lXxf86vm;-lXrandr;-lXi")

add_executable(${PROJECT_NAME} src/main.cpp src/vk_engine.cpp)

if(APPLE)
  enable_language(OBJC)
endif()

# add_subdirectory(thirdparty/glfw)
add_subdirectory(thirdparty/fmt)

# add_subdirectory(thirdparty/SDL)

# I set my systems VULKAN_SDK_PATH to ~/VulkanSDK/version#/macOS
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC src
  # PUBLIC thirdparty/glfw/include PUBLIC thirdparty/SDL/include
  PUBLIC thirdparty/glm
  PUBLIC thirdparty/fmt/include
  PUBLIC $ENV{VULKAN_SDK_PATH}/include)

target_link_directories(
  ${PROJECT_NAME}
  PUBLIC
  # thirdparty/glfw/src
  ${PROJECT_NAME}
  PUBLIC
  thirdparty/fmt/src
  # PUBLIC thirdparty/SDL/src
  PUBLIC
  $ENV{VULKAN_SDK_PATH}/lib)

target_link_libraries(
  ${PROJECT_NAME}
  #
  PRIVATE glfw
  PRIVATE fmt
  # PRIVATE sdl2
  PRIVATE vulkan)
