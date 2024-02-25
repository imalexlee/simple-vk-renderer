#pragma once

#include <cstdint>
#include <optional>
#include <vk_types.h>

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  bool is_complete() { return graphics_family.has_value(); }
};

class VulkanEngine {
private:
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device;
  VkQueue _graphics_queue;
  VkPhysicalDeviceFeatures _device_features;
  VkSurfaceKHR _surface;

  void setup_debug_messenger();
  void populate_debug_info(VkDebugUtilsMessengerCreateInfoEXT &create_info);
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                 VkDebugUtilsMessageTypeFlagsEXT messageType,
                 const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                 void *pUserData);

  bool is_device_suitable(VkPhysicalDevice physical_device);
  QueueFamilyIndices find_queue_families(VkPhysicalDevice physical_device);

  void create_instance();
  void pick_physical_device();
  void create_logical_device();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();

public:
  int _frame_number{0};
  VkExtent2D _window_extent{1700, 900};

  struct GLFWwindow *_window{nullptr};

  static VulkanEngine &Get();

  void init();
  void cleanup();
  void draw();
  void run();
};
