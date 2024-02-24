cmake_minimum_required(VERSION 3.18)

project(simple-vk-renderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME} src/main.cpp)

add_subdirectory(thirdparty/glfw)

# I set my systems VULKAN_SDK_PATH to ~/VulkanSDK/version#/macOS
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC thirdparty/glfw/include
  PUBLIC thirdparty/glm
  PUBLIC $ENV{VULKAN_SDK_PATH}/include)

target_link_directories(${PROJECT_NAME} PUBLIC thirdparty/glfw/src PUBLIC
                        $ENV{VULKAN_SDK_PATH}/lib)
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE glfw
  PRIVATE vulkan)
