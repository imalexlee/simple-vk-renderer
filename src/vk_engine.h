#pragma once

#include "vk_descriptors.h"
#include "vk_mem_alloc.h"
#include "vulkan/vulkan_core.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vk_types.h>

constexpr static uint32_t FRAME_OVERLAP = 3;

class VulkanEngine {
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;

  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device;

  VkQueue _graphics_queue;
  DeletionQueue _main_deletion_queue;

  uint32_t _graphics_queue_family;
  VkQueue _present_queue;
  uint32_t _present_queue_family;

  VkPhysicalDeviceFeatures _device_features;
  VkSurfaceKHR _surface;

  VkSwapchainKHR _swap_chain;
  VkFormat _swap_chain_format;
  VkExtent2D _swap_chain_extent;
  std::vector<VkImage> _swap_chain_images;
  std::vector<VkImageView> _swap_chain_image_views;

  std::array<FrameData, FRAME_OVERLAP> _frames{};

  VmaAllocator _allocator;
  AllocatedImage _draw_image;
  VkExtent2D _draw_extent;

  DescriptorAllocator _global_descriptor_allocator;
  VkDescriptorSet _draw_image_descriptors;
  VkDescriptorSetLayout _draw_image_descriptor_layout;

  VkPipeline _gradient_pipeline;
  VkPipelineLayout _gradient_pipeline_layout;

  // immediate submit structures
  VkFence _imm_fence;
  VkCommandBuffer _imm_cmd_buffer;
  VkCommandPool _imm_cmd_pool;
  VkDescriptorPool _imm_descriptor_pool = VK_NULL_HANDLE;

  std::vector<ComputeEffect> _background_effects;
  int32_t _current_background_effect{0};

  VkPipelineLayout _triangle_pipeline_layout;
  VkPipeline _triangle_pipeline;
  VkPipelineLayout _mesh_pipeline_layout;
  VkPipeline _mesh_pipeline;

  GPUMeshBuffers rectangle;

  FrameData& get_current_frame() {
    return _frames[_frame_number % FRAME_OVERLAP];
  };

  // initializers
  void setup_debug_messenger();
  void populate_debug_info(VkDebugUtilsMessengerCreateInfoEXT& create_info);
  void create_instance();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_allocator();
  void create_swapchain();
  void create_image_views();
  void init_commands();
  void init_sync_structures();
  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();
  void init_imgui();
  void init_default_data();

  // pipeline functions
  void init_triangle_pipeline();
  void init_mesh_pipeline();

  // draws
  void draw_background(VkCommandBuffer cmd);
  void draw_geometry(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view);

  // utils
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageType,
                 const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                 void* pUserData);

  bool is_device_suitable(VkPhysicalDevice physical_device);
  QueueFamilyIndices find_queue_families(VkPhysicalDevice physical_device);
  SwapChainSupportDetails
  query_swap_chain_support(VkPhysicalDevice physical_device);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities);
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  AllocatedBuffer create_buffer(size_t alloc_size, VkBufferUsageFlags buf_usage,
                                VmaMemoryUsage mem_usage);
  void destroy_buffer(const AllocatedBuffer& buffer);
  GPUMeshBuffers upload_mesh(std::span<uint32_t> indices,
                             std::span<Vertex> vertices);

public:
  int _frame_number{0};
  VkExtent2D _window_extent{1700, 900};

  struct GLFWwindow* _window{nullptr};

  static VulkanEngine& Get();

  void init();
  void cleanup();
  void draw();
  void run();
};
