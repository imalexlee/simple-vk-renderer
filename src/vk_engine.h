#pragma once

#include "glm/ext/vector_float4.hpp"
#include "vk_descriptors.h"
#include "vk_mem_alloc.h"
#include "vulkan/vulkan_core.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vk_loader.h>
#include <vk_types.h>

struct MeshNode : public Node {
  std::shared_ptr<MeshAsset> mesh;
  virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct RenderObject {
  uint32_t index_count;
  uint32_t first_index;
  VkBuffer index_buffer;

  MaterialInstance* material;
  glm::mat4 transform;
  VkDeviceAddress vertex_buf_addr;
};

struct DrawContext {
  std::vector<RenderObject> opaque_surfaces;
};

struct GLTFMettallicRoughness {
  MaterialPipeline opaque_pipeline;
  MaterialPipeline transparent_pipeline;

  VkDescriptorSetLayout material_desc_layout;

  struct MaterialConstants {
    glm::vec4 color_factors;
    glm::vec4 metal_rough_factors;
    // padding
    glm::vec4 extra;
  };

  struct MaterialResources {
    AllocatedImage color_image;
    VkSampler color_sampler;
    AllocatedImage metal_rough_image;
    VkSampler metal_rough_sampler;
    VkBuffer data_buffer;
    uint32_t data_buffer_offset;
  };

  DescriptorWriter desc_writer;

  void build_pipelines(VulkanEngine* engine);
  void clear_resources(VkDevice device);

  MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources&,
                                  DescriptorAllocatorGrowable& descriptor_allocator);
};

constexpr static uint32_t FRAME_OVERLAP = 3;

struct VulkanEngine {
  VulkanEngine(){};
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;

  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device;

  struct GLFWwindow* _window{nullptr};
  bool _resize_requested{false};

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
  GPUSceneData _scene_data;
  VkDescriptorSetLayout _gpu_scene_descriptor_layout;

  VmaAllocator _allocator;
  AllocatedImage _draw_image;
  AllocatedImage _depth_image;
  VkExtent2D _draw_extent;
  float _render_scale = 1.f;

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
  std::vector<std::shared_ptr<MeshAsset>> _test_meshes;

  AllocatedImage _white_image;
  AllocatedImage _black_image;
  AllocatedImage _grey_image;
  AllocatedImage _error_checkerboard_image;

  VkSampler _default_sampler_linear;
  VkSampler _default_sampler_nearest;

  VkDescriptorSetLayout _single_image_desc_layout;

  MaterialInstance default_data;
  GLTFMettallicRoughness metal_rough_material;

  FrameData& get_current_frame() { return _frames[_frame_number % FRAME_OVERLAP]; };

  // initializers
  void setup_debug_messenger();
  void populate_debug_info(VkDebugUtilsMessengerCreateInfoEXT& create_info);
  void create_instance();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_allocator();
  void init_swapchain();
  void create_swapchain(uint32_t width, uint32_t height);
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
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                       VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                       void* pUserData);

  bool is_device_suitable(VkPhysicalDevice physical_device);
  QueueFamilyIndices find_queue_families(VkPhysicalDevice physical_device);
  SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice physical_device);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities);
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  AllocatedBuffer create_buffer(size_t alloc_size, VkBufferUsageFlags buf_usage, VmaMemoryUsage mem_usage);
  AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
  AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                              bool mipmapped = false);
  void destroy_buffer(const AllocatedBuffer& buffer);
  void destroy_image(const AllocatedImage& img);

  void destroy_swapchain();
  void destroy_sync_structures();
  void resize_swapchain();

public:
  GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
  int _frame_number{0};
  VkExtent2D _window_extent{1700, 900};

  static VulkanEngine& Get();

  void init();
  void cleanup();
  void draw();
  void run();
};
