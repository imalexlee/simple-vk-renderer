#include "vk_engine.h"
#include "vulkan/vulkan_core.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vk_initializers.h>
#include <vk_types.h>

const std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
constexpr bool use_validation_layers = false;
#else
constexpr bool use_validation_layers = true;
#endif

VulkanEngine *loaded_engine = nullptr;

VulkanEngine &VulkanEngine::Get() { return *loaded_engine; }

bool check_validation_support() {
  uint32_t layer_count{};
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> avail_layers(layer_count);

  vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());

  for (const char *layer : validation_layers) {
    if (std::find_if(avail_layers.begin(), avail_layers.end(),
                     [&](VkLayerProperties &avail_layer) {
                       return strcmp(layer, avail_layer.layerName);
                     }) == avail_layers.end()) {
      return false;
    }
  }
  return true;
}

std::vector<const char *> get_required_extensions() {

  uint32_t ext_count{0};
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&ext_count);

  // portability subset for mac

  std::vector<const char *> extensions{};
  for (uint32_t i{0}; i < ext_count; ++i) {
    extensions.emplace_back(glfw_extensions[i]);
  }
  extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

  if (use_validation_layers) {
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

VkBool32 VulkanEngine::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
};

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}
void VulkanEngine::setup_debug_messenger() {
  if (!use_validation_layers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  populate_debug_info(create_info);
  VK_CHECK(CreateDebugUtilsMessengerEXT(_instance, &create_info, nullptr,
                                        &_debug_messenger));
}
void VulkanEngine::populate_debug_info(
    VkDebugUtilsMessengerCreateInfoEXT &create_info) {
  create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = VulkanEngine::debug_callback;
}

void VulkanEngine::init() {
  // only one engine initialization is allowed with the application.
  assert(loaded_engine == nullptr);
  loaded_engine = this;

  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  _window = glfwCreateWindow(_window_extent.width, _window_extent.height,
                             "Vulkan window", nullptr, nullptr);

  create_instance();
  setup_debug_messenger();
  pick_physical_device();
  create_logical_device();

  init_swapchain();

  init_commands();

  init_sync_structures();
}
void VulkanEngine::create_instance() {

  // enable validation if in debug mode
  if (use_validation_layers && !check_validation_support()) {
    throw std::runtime_error("Could not enable validation layers");
  }

  // fill in app info
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "simple app";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "simple-vk-renderer";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_3;
  app_info.pNext = nullptr;

  auto extensions = get_required_extensions();

  // fill createInstance struct
  VkInstanceCreateInfo instance_create_info{};
  instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_create_info.pApplicationInfo = &app_info;
  instance_create_info.flags |=
      VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  instance_create_info.enabledExtensionCount =
      static_cast<uint32_t>(extensions.size());
  instance_create_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  if (use_validation_layers) {
    instance_create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    instance_create_info.ppEnabledLayerNames = validation_layers.data();
    populate_debug_info(debug_create_info);
    instance_create_info.pNext = static_cast<const void *>(&debug_create_info);

  } else {
    instance_create_info.enabledLayerCount = 0;
    instance_create_info.ppEnabledLayerNames = nullptr;
  }

  // call create instance and give the instance to our member variable
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &_instance));
};

bool VulkanEngine::is_device_suitable(VkPhysicalDevice physical_device) {
  QueueFamilyIndices queue_families = find_queue_families(physical_device);

  VkPhysicalDeviceSynchronization2Features sync_features{};

  // check for some vulkan 1.3 feautres
  sync_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;

  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  dynamic_rendering_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  dynamic_rendering_features.pNext = &sync_features;

  VkPhysicalDeviceFeatures2 physical_features{};
  physical_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  physical_features.pNext = &dynamic_rendering_features;

  vkGetPhysicalDeviceFeatures2(physical_device, &physical_features);
  if (sync_features.synchronization2 != VK_TRUE ||
      dynamic_rendering_features.dynamicRendering != VK_TRUE) {
    return false;
  }

  _device_features = physical_features.features;

  return queue_families.is_complete();
}

void VulkanEngine::pick_physical_device() {

  uint32_t physical_device_count{};
  VK_CHECK(
      vkEnumeratePhysicalDevices(_instance, &physical_device_count, nullptr));

  if (physical_device_count == 0) {
    throw std::runtime_error("Failed to find a physical compatible GPU");
  }

  std::vector<VkPhysicalDevice> physical_devices(physical_device_count);

  VK_CHECK(vkEnumeratePhysicalDevices(_instance, &physical_device_count,
                                      physical_devices.data()));
  for (const auto &physical_device : physical_devices) {
    if (is_device_suitable(physical_device)) {
      _physical_device = physical_device;
      std::cout << "found suitable device\n";
    }
  }
  if (_physical_device == VK_NULL_HANDLE) {
    throw std::runtime_error("Could not find a suitable physical GPU");
  }
};

QueueFamilyIndices
VulkanEngine::find_queue_families(VkPhysicalDevice physical_device) {
  QueueFamilyIndices indices;
  uint32_t family_count{};
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count,
                                           queue_families.data());
  // find family that supports VK_QUEUE_GRAPHICS_BIT
  for (uint32_t i{0}; i < queue_families.size(); ++i) {
    if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }
    if (indices.is_complete()) {
      break;
    }
  }

  return indices;
};

void VulkanEngine::create_logical_device() {

  QueueFamilyIndices family_indices = find_queue_families(_physical_device);

  constexpr float graphics_queue_priority{1.0f};

  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueCount = 1;
  queue_create_info.queueFamilyIndex = family_indices.graphics_family.value();
  queue_create_info.pQueuePriorities = &graphics_queue_priority;

  VkPhysicalDeviceVulkan13Features features_1_3{};
  features_1_3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_1_3.synchronization2 = VK_TRUE;
  features_1_3.dynamicRendering = VK_TRUE;

  // for bindless rendering
  VkPhysicalDeviceVulkan12Features features_1_2{};
  features_1_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_1_2.bufferDeviceAddress = VK_TRUE;
  features_1_2.descriptorIndexing = VK_TRUE;
  features_1_2.pNext = &features_1_3;

  VkDeviceCreateInfo device_create_info{};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pQueueCreateInfos = &queue_create_info;

  if (use_validation_layers) {
    device_create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    device_create_info.ppEnabledLayerNames = validation_layers.data();
  }
  device_create_info.enabledLayerCount = 0;
  device_create_info.enabledExtensionCount = 1;
  std::vector<const char *> device_extensions{"VK_KHR_portability_subset"};

  device_create_info.ppEnabledExtensionNames = device_extensions.data();
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pEnabledFeatures = &_device_features;
  device_create_info.pNext = &features_1_2;

  VK_CHECK(
      vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));

  vkGetDeviceQueue(_device, family_indices.graphics_family.value(), 0,
                   &_graphics_queue);
};

void VulkanEngine::init_swapchain(){};

void VulkanEngine::init_commands(){};

void VulkanEngine::init_sync_structures(){};

void VulkanEngine::cleanup() {
  vkDestroyDevice(_device, nullptr);
  if (use_validation_layers) {
    DestroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
  }
  vkDestroyInstance(_instance, nullptr);

  glfwDestroyWindow(_window);
  glfwTerminate();
  loaded_engine = nullptr;
}

void VulkanEngine::run() {
  while (!glfwWindowShouldClose(_window)) {
    glfwPollEvents();
    draw();
  }
}

void VulkanEngine::draw() {}
