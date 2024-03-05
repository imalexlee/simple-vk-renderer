#pragma once

#include "vk_mem_alloc.h"
#include "vulkan/vulkan_core.h"
#include <cstdint>
#include <optional>
#include <vk_types.h>

struct DeletionQueue {
  std::deque<std::function<void()>> deletors;

  void push_function(std::function<void()> &&function) {
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

constexpr static uint32_t FRAME_OVERLAP = 2;

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

  FrameData &get_current_frame() {
    return _frames[_frame_number % FRAME_OVERLAP];
  };

  void setup_debug_messenger();
  void populate_debug_info(VkDebugUtilsMessengerCreateInfoEXT &create_info);
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageType,
                 const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                 void *pUserData);

  bool is_device_suitable(VkPhysicalDevice physical_device);
  QueueFamilyIndices find_queue_families(VkPhysicalDevice physical_device);
  SwapChainSupportDetails
  query_swap_chain_support(VkPhysicalDevice physical_device);

  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities);

  void create_instance();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_swapchain();
  void create_image_views();
  void init_commands();
  void init_sync_structures();

  void guide_init_vulkan();
  void guide_init_swapchain();
  void guide_init_commands();
  void guide_init_sync_structures();
  void guide_draw();

public:
  int _frame_number{0};
  VkExtent2D _window_extent{1700, 900};

  struct GLFWwindow *_window{nullptr};
  // struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  void init();
  void cleanup();
  void draw();
  void run();
};
