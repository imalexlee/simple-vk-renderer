cmake_minimum_required(VERSION 3.18)

project(simple-vk-renderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME} src/main.cpp src/vk_engine.cpp src/vk_images.cpp
                               src/vk_initializers.cpp)

if(APPLE)
  enable_language(OBJC)
endif()

add_subdirectory(thirdparty/glfw)
add_subdirectory(thirdparty/fmt)
add_subdirectory(thirdparty/vk-bootstrap)
add_subdirectory(thirdparty/VulkanMemoryAllocator)
# add_subdirectory(thirdparty/SDL EXCLUDE_FROM_ALL)

# I set my systems VULKAN_SDK_PATH to ~/VulkanSDK/version#/macOS
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC src
  PUBLIC thirdparty/glfw/include
  # PUBLIC thirdparty/SDL/include
  PUBLIC thirdparty/glm
  PUBLIC thirdparty/fmt/include
  PUBLIC thirdparty/vk-bootstrap/src
  PUBLIC thirdparty/VulkanMemoryAllocator/include
  PUBLIC $ENV{VULKAN_SDK_PATH}../MoltenVK/include)
# PUBLIC $ENV{VULKAN_SDK_PATH}/include)

target_link_directories(
  ${PROJECT_NAME}
  PUBLIC
  thirdparty/glfw/src
  ${PROJECT_NAME}
  PUBLIC
  thirdparty/fmt/src
  PUBLIC
  thirdparty/vk-bootstrap/src
  PUBLIC
  thirdparty/VulkanMemoryAllocator/src
  PUBLIC
  $ENV{VULKAN_SDK_PATH}/lib)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE glfw
  PRIVATE fmt
  PRIVATE vk-bootstrap
  PRIVATE VulkanMemoryAllocator
  # PRIVATE SDL2
  PRIVATE vulkan)
