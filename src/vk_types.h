
#pragma once

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

// #include <vk_mem_alloc.h>

#include "fmt/base.h"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "vk_mem_alloc.h"
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = x;                                                          \
    if (err) {                                                                 \
      fmt::println("Detected Vulkan error: {}", string_VkResult(err));         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

struct DeletionQueue {
  std::deque<std::function<void()>> deletors;

  void push_function(std::function<void()>&& function) {
    deletors.push_back(function);
  }

  void flush() {
    // reverse iterate the deletion queue to execute all the functions
    for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
      (*it)(); // call functors
    }

    deletors.clear();
  }
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool is_complete() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

struct FrameData {
  VkCommandPool command_pool;
  VkCommandBuffer main_command_buffer;
  VkSemaphore _swapchain_semaphore;
  VkSemaphore _render_semaphore;
  VkFence _render_fence;
  DeletionQueue deletion_queue;
};

struct AllocatedImage {
  VkImage image;
  VkImageView image_view;
  VmaAllocation allocation;
  VkExtent3D image_extent;
  VkFormat image_format;
};

struct ComputePushConstants {
  glm::vec4 data1;
  glm::vec4 data2;
  glm::vec4 data3;
  glm::vec4 data4;
};

struct ComputeEffect {
  const char* name;
  VkPipeline pipeline;
  VkPipelineLayout pipeline_layout;
  ComputePushConstants data;
};

struct AllocatedBuffer {
  VkBuffer buffer;
  VmaAllocationInfo info;
  VmaAllocation allocation;
};

struct Vertex {
  glm::vec3 position;
  float uv_x;
  glm::vec3 normal;
  float uv_y;
  glm::vec4 color;
};

// resources for a mesh
struct GPUMeshBuffers {
  AllocatedBuffer index_buf;
  AllocatedBuffer vertex_buf;
  VkDeviceAddress vertex_buf_address;
};

// push constants for mesh drawing
struct GPUDrawPushConstants {
  glm::mat4 world_mat;
  VkDeviceAddress vertex_buf_address;
};
