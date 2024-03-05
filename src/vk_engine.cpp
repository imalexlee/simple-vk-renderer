#include "vk_engine.h"
#define GLFW_INCLUDE_VULKAN
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include <GLFW/glfw3.h>

#include "vk_images.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <strings.h>
#include <vector>
#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"

// #define ALLOW_MAILBOX_MODE

const std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
constexpr bool use_validation_layers = false;
#else
constexpr bool use_validation_layers = true;
#endif

constexpr std::array<const char *, 2> device_extensions{
    "VK_KHR_portability_subset",
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

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
  create_surface();
  pick_physical_device();
  create_logical_device();
  create_swapchain();
  create_image_views();
  init_commands();
  init_sync_structures();

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = _physical_device;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocatorInfo, &_allocator);

  _main_deletion_queue.push_function(
      [&]() { vmaDestroyAllocator(_allocator); });

  // guide_init_vulkan();
  // guide_init_swapchain();
  // guide_init_commands();
  // guide_init_sync_structures();
}

// END OF GUIDE FUNCTIONS

void VulkanEngine::create_surface() {
  VK_CHECK(glfwCreateWindowSurface(_instance, _window, nullptr, &_surface));
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

/// check for required queue families, instance extensions, and device
/// extensions
bool VulkanEngine::is_device_suitable(VkPhysicalDevice physical_device) {
  QueueFamilyIndices queue_families = find_queue_families(physical_device);

  SwapChainSupportDetails swap_chain_details =
      query_swap_chain_support(physical_device);

  // I'll be using synchronization 2 and dynamic rendering instead of render
  // passes
  VkPhysicalDeviceSynchronization2Features sync_features{};
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

  if (swap_chain_details.present_modes.empty() ||
      swap_chain_details.formats.empty()) {
    return false;
  }

  _device_features = physical_features.features;

  if (queue_families.is_complete()) {
    _graphics_queue_family = queue_families.graphics_family.value();
    _present_queue_family = queue_families.present_family.value();
    return true;
  }
  return false;
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
    // already have chosen a surface before querying device suitability

    if (is_device_suitable(physical_device)) {
      _physical_device = physical_device;
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
    VkBool32 present_support = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, _surface,
                                         &present_support);
    if (present_support == VK_TRUE) {
      indices.present_family = i;
    }
    if (indices.is_complete()) {
      break;
    }
  }

  return indices;
};

SwapChainSupportDetails
VulkanEngine::query_swap_chain_support(VkPhysicalDevice physical_device) {
  SwapChainSupportDetails swap_chain_details{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      physical_device, _surface, &swap_chain_details.capabilities));

  uint32_t surface_format_count{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
      physical_device, _surface, &surface_format_count, nullptr));

  if (surface_format_count > 0) {
    swap_chain_details.formats.resize(surface_format_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, _surface, &surface_format_count,
        swap_chain_details.formats.data()));
  }

  uint32_t present_modes_count{};
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device, _surface, &present_modes_count, nullptr));

  if (present_modes_count > 0) {
    swap_chain_details.present_modes.resize(present_modes_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, _surface, &present_modes_count,
        swap_chain_details.present_modes.data()));
  }
  return swap_chain_details;
};

VkSurfaceFormatKHR choose_surface_format(
    const std::vector<VkSurfaceFormatKHR> &available_formats) {
  for (const auto &format : available_formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR choose_present_mode(
    const std::vector<VkPresentModeKHR> &available_present_modes) {
#ifdef ALLOW_MAILBOX_MODE
  for (const auto &present_mode : available_present_modes) {
    if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return present_mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
#else
  return VK_PRESENT_MODE_FIFO_KHR;
#endif
}

VkExtent2D
VulkanEngine::choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    VkExtent2D actual_extent = {static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height)};
    actual_extent.width =
        std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actual_extent.height =
        std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);
    return actual_extent;
  }
}

void VulkanEngine::create_logical_device() {

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos{};
  std::set<uint32_t> unique_queue_family_indices{_graphics_queue_family,
                                                 _present_queue_family};
  constexpr float queue_priority{1.0f};
  for (const uint32_t &index : unique_queue_family_indices) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueCount = 1;
    queue_create_info.queueFamilyIndex = index;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

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
  device_create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  device_create_info.pQueueCreateInfos = queue_create_infos.data();

  if (use_validation_layers) {
    device_create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    device_create_info.ppEnabledLayerNames = validation_layers.data();
  } else {
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
  }
  device_create_info.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  device_create_info.ppEnabledExtensionNames = device_extensions.data();
  device_create_info.pEnabledFeatures = &_device_features;
  device_create_info.pNext = &features_1_2;

  VK_CHECK(
      vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));

  vkGetDeviceQueue(_device, _graphics_queue_family, 0, &_graphics_queue);
  vkGetDeviceQueue(_device, _present_queue_family, 0, &_present_queue);
};

void VulkanEngine::create_swapchain() {
  SwapChainSupportDetails swap_chain_details =
      query_swap_chain_support(_physical_device);

  VkExtent2D extent = choose_swap_extent(swap_chain_details.capabilities);
  VkPresentModeKHR present_mode =
      choose_present_mode(swap_chain_details.present_modes);
  VkSurfaceFormatKHR surface_format =
      choose_surface_format(swap_chain_details.formats);

  uint32_t image_count = swap_chain_details.capabilities.minImageCount + 1;
  if (swap_chain_details.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_details.capabilities.maxImageCount) {
    image_count = swap_chain_details.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR swap_chain_create_info{};
  swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swap_chain_create_info.surface = _surface;
  swap_chain_create_info.minImageCount = image_count;
  swap_chain_create_info.imageExtent = extent;
  swap_chain_create_info.imageFormat = surface_format.format;
  swap_chain_create_info.imageColorSpace = surface_format.colorSpace;
  swap_chain_create_info.imageArrayLayers = 1;
  swap_chain_create_info.imageUsage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices queue_family_indices =
      find_queue_families(_physical_device);
  uint32_t indices[]{queue_family_indices.graphics_family.value(),
                     queue_family_indices.present_family.value()};

  if (queue_family_indices.present_family.value() !=
      queue_family_indices.graphics_family.value()) {
    swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swap_chain_create_info.queueFamilyIndexCount = 2;
    swap_chain_create_info.pQueueFamilyIndices = indices;
  } else {
    swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap_chain_create_info.queueFamilyIndexCount = 0;
    swap_chain_create_info.pQueueFamilyIndices = nullptr;
  }
  swap_chain_create_info.preTransform =
      swap_chain_details.capabilities.currentTransform;
  swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swap_chain_create_info.presentMode = present_mode;
  swap_chain_create_info.clipped = VK_TRUE;
  swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

  VK_CHECK(vkCreateSwapchainKHR(_device, &swap_chain_create_info, nullptr,
                                &_swap_chain));

  VK_CHECK(
      vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count, nullptr));
  _swap_chain_images.resize(image_count);

  VK_CHECK(vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count,
                                   _swap_chain_images.data()));
  _swap_chain_format = surface_format.format;
  _swap_chain_extent = extent;
};

void VulkanEngine::create_image_views() {
  _swap_chain_image_views.resize(_swap_chain_images.size());
  for (size_t i{0}; i < _swap_chain_images.size(); ++i) {
    VkImageViewCreateInfo image_view_create_info{};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = _swap_chain_images[i];
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = _swap_chain_format;
    image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(_device, &image_view_create_info, nullptr,
                               &_swap_chain_image_views[i]));
  }
};

void VulkanEngine::init_commands() {
  // create graphics queue command pool
  VkCommandPoolCreateInfo command_pool_create_info{};
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.queueFamilyIndex = _graphics_queue_family;
  command_pool_create_info.pNext = nullptr;
  command_pool_create_info.flags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  // allocating a command pool and buffer in pairs to allow double buffering in
  // rendering
  for (uint32_t i{0}; i < FRAME_OVERLAP; ++i) {
    VK_CHECK(vkCreateCommandPool(_device, &command_pool_create_info, nullptr,
                                 &_frames[i].command_pool));

    VkCommandBufferAllocateInfo buffer_alloc_info{};
    buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    buffer_alloc_info.pNext = nullptr;
    buffer_alloc_info.commandPool = _frames[i].command_pool;
    buffer_alloc_info.commandBufferCount = 1;
    buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VK_CHECK(vkAllocateCommandBuffers(_device, &buffer_alloc_info,
                                      &_frames[i].main_command_buffer));
  }
};

void VulkanEngine::init_sync_structures() {

  // start fence in signaled state to fit with first wait
  VkFenceCreateInfo fence_create_info{};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  fence_create_info.pNext = nullptr;

  VkSemaphoreCreateInfo semaphore_create_info{};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = nullptr;

  for (uint32_t i{0}; i < FRAME_OVERLAP; ++i) {
    VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr,
                           &_frames[i]._render_fence));
    VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr,
                               &_frames[i]._swapchain_semaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr,
                               &_frames[i]._render_semaphore));
  }
};

void VulkanEngine::cleanup() {

  _main_deletion_queue.flush();

  vkDestroySwapchainKHR(_device, _swap_chain, nullptr);
  for (FrameData &frame_data : _frames) {
    vkDestroyCommandPool(_device, frame_data.command_pool, nullptr);

    vkDestroyFence(_device, frame_data._render_fence, nullptr);
    vkDestroySemaphore(_device, frame_data._render_semaphore, nullptr);
    vkDestroySemaphore(_device, frame_data._swapchain_semaphore, nullptr);
  }
  for (VkImageView &image_view : _swap_chain_image_views) {
    vkDestroyImageView(_device, image_view, nullptr);
  }
  vkDestroyDevice(_device, nullptr);
  if (use_validation_layers) {
    DestroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
  }
  vkDestroySurfaceKHR(_instance, _surface, nullptr);
  vkDestroyInstance(_instance, nullptr);

  glfwDestroyWindow(_window);
  glfwTerminate();
  loaded_engine = nullptr;
}

void VulkanEngine::run() {
  while (!glfwWindowShouldClose(_window)) {
    glfwPollEvents();
    draw();
    //   guide_draw();
  }
  vkDeviceWaitIdle(_device);
}

void VulkanEngine::draw() {

  std::cout << _frame_number << "\n";
  std::cout << "BRUH 0000000\n";
  // timeout of 1 second
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._render_fence, true,
                           100000000000000));
  get_current_frame().deletion_queue.flush();
  std::cout << "BRUH 1111111\n";
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._render_fence));

  uint32_t image_index{};
  VK_CHECK(vkAcquireNextImageKHR(_device, _swap_chain, 100000000000000,
                                 get_current_frame()._swapchain_semaphore,
                                 nullptr, &image_index));

  VkCommandBuffer cmd = get_current_frame().main_command_buffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo command_begin_info{};
  command_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_begin_info.pNext = nullptr;
  command_begin_info.pInheritanceInfo = nullptr;
  command_begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(cmd, &command_begin_info));

  // transition img into a writeable mode
  vkutil::transition_image(cmd, _swap_chain_images[image_index],
                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  VkClearColorValue clear_value{};
  float flash = abs(sin(_frame_number / 120.f));
  clear_value = {{0.0f, 0.0f, flash, 1.0f}};

  VkImageSubresourceRange clear_range =
      vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

  vkCmdClearColorImage(cmd, _swap_chain_images[image_index],
                       VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &clear_range);

  // transition img into a presentable mode
  vkutil::transition_image(cmd, _swap_chain_images[image_index],
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  VK_CHECK(vkEndCommandBuffer(cmd));

  // we've got a complete command buffer, now hook up with sync structures
  VkSemaphoreSubmitInfo wait_semaphore_info{};
  wait_semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  wait_semaphore_info.semaphore = get_current_frame()._swapchain_semaphore;
  wait_semaphore_info.pNext = nullptr;
  wait_semaphore_info.value = 1;
  wait_semaphore_info.deviceIndex = 0;
  wait_semaphore_info.stageMask =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;

  VkSemaphoreSubmitInfo signal_semaphore_info{};
  signal_semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  signal_semaphore_info.semaphore = get_current_frame()._render_semaphore;
  signal_semaphore_info.pNext = nullptr;
  signal_semaphore_info.value = 1;
  signal_semaphore_info.deviceIndex = 0;
  signal_semaphore_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

  VkCommandBufferSubmitInfo cmd_submit_info{};
  cmd_submit_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  cmd_submit_info.pNext = nullptr;
  cmd_submit_info.commandBuffer = cmd;
  cmd_submit_info.deviceMask = 0;

  VkSubmitInfo2 submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_submit_info;
  submit_info.waitSemaphoreInfoCount = 1;
  submit_info.pWaitSemaphoreInfos = &wait_semaphore_info;
  submit_info.signalSemaphoreInfoCount = 1;
  submit_info.pSignalSemaphoreInfos = &signal_semaphore_info;

  VK_CHECK(vkQueueSubmit2(_graphics_queue, 1, &submit_info,
                          get_current_frame()._render_fence));

  std::cout << "BRUH 2222222\n";
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &_swap_chain;
  present_info.pImageIndices = &image_index;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &get_current_frame()._render_semaphore;
  present_info.pNext = nullptr;

  VK_CHECK(vkQueuePresentKHR(_graphics_queue, &present_info));

  std::cout << "BRUH 333333\n";
  ++_frame_number;
}

// void VulkanEngine::guide_init_vulkan() {
//
//   vkb::InstanceBuilder builder{};
//
//   auto extensions = get_required_extensions();
//
//   // make the vulkan instance, with basic debug features
//   auto inst_ret = builder.set_app_name("Example Vulkan Application")
//                       .request_validation_layers(use_validation_layers)
//                       .use_default_debug_messenger()
//                       .require_api_version(1, 3, 0)
//                       .enable_extensions(extensions)
//                       .build();
//
//   vkb::Instance vkb_inst = inst_ret.value();
//
//   // grab the instance
//   _instance = vkb_inst.instance;
//   _debug_messenger = vkb_inst.debug_messenger;
//
//   VK_CHECK(glfwCreateWindowSurface(_instance, _window, nullptr, &_surface));
//   // vulkan 1.3 features
//   VkPhysicalDeviceVulkan13Features features{};
//   features.dynamicRendering = true;
//   features.synchronization2 = true;
//
//   // vulkan 1.2 features
//   VkPhysicalDeviceVulkan12Features features12{};
//   features12.bufferDeviceAddress = true;
//   features12.descriptorIndexing = true;
//
//   // use vkbootstrap to select a gpu.
//   // We want a gpu that can write to the SDL surface and supports vulkan 1.3
//   // with the correct features
//   vkb::PhysicalDeviceSelector selector{vkb_inst, _surface};
//   vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 2)
//                                            .set_required_features_13(features)
//                                            .set_required_features_12(features12)
//                                            .select()
//                                            .value();
//
//   // create the final vulkan device
//   vkb::DeviceBuilder deviceBuilder{physicalDevice};
//
//   vkb::Device vkbDevice = deviceBuilder.build().value();
//
//   // Get the VkDevice handle used in the rest of a vulkan application
//   _device = vkbDevice.device;
//   _physical_device = physicalDevice.physical_device;
//
//   _graphics_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
//   _graphics_queue_family =
//       vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
// }
//
// void VulkanEngine::guide_init_swapchain() {
//   // nothing yet
//   vkb::SwapchainBuilder swapchainBuilder{_physical_device, _device,
//   _surface};
//
//   _swap_chain_format = VK_FORMAT_B8G8R8A8_UNORM;
//
//   vkb::Swapchain vkbSwapchain =
//       swapchainBuilder
//           //.use_default_format_selection()
//           .set_desired_format(VkSurfaceFormatKHR{
//               .format = _swap_chain_format,
//               .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
//           // use vsync present mode
//           .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
//           .set_desired_extent(_window_extent.width, _window_extent.height)
//           .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
//           .build()
//           .value();
//
//   _swap_chain_extent = vkbSwapchain.extent;
//   // store swapchain and its related images
//   _swap_chain = vkbSwapchain.swapchain;
//   _swap_chain_images = vkbSwapchain.get_images().value();
//   _swap_chain_image_views = vkbSwapchain.get_image_views().value();
// }
// void VulkanEngine::guide_init_commands() {
//   // nothing yet
//   // create a command pool for commands submitted to the graphics queue.
//   // we also want the pool to allow for resetting of individual command
//   buffers VkCommandPoolCreateInfo commandPoolInfo =
//   vkinit::command_pool_create_info(
//       _graphics_queue_family,
//       VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
//
//   for (int i = 0; i < FRAME_OVERLAP; i++) {
//
//     VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
//                                  &_frames[i].command_pool));
//
//     // allocate the default command buffer that we will use for rendering
//     VkCommandBufferAllocateInfo cmdAllocInfo =
//         vkinit::command_buffer_allocate_info(_frames[i].command_pool, 1);
//
//     VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
//                                       &_frames[i].main_command_buffer));
//   }
// }
// void VulkanEngine::guide_init_sync_structures() {
//   // nothing yet
//   VkFenceCreateInfo fenceCreateInfo =
//       vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
//   VkSemaphoreCreateInfo semaphoreCreateInfo =
//   vkinit::semaphore_create_info();
//
//   for (int i = 0; i < FRAME_OVERLAP; i++) {
//     VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr,
//                            &_frames[i]._render_fence));
//     VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
//                                &_frames[i]._swapchain_semaphore));
//     VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr,
//                                &_frames[i]._render_semaphore));
//   }
// }

// void VulkanEngine::guide_draw() {
//   VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._render_fence,
//   true,
//                            1000000000000));
//   VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._render_fence));
//   uint32_t swapchainImageIndex;
//   VK_CHECK(vkAcquireNextImageKHR(_device, _swap_chain, 1000000000000,
//                                  get_current_frame()._swapchain_semaphore,
//                                  nullptr, &swapchainImageIndex));
//
//   // naming it cmd for shorter writing
//   VkCommandBuffer cmd = get_current_frame().main_command_buffer;
//
//   // now that we are sure that the commands finished executing, we can safely
//   // reset the command buffer to begin recording again.
//   VK_CHECK(vkResetCommandBuffer(cmd, 0));
//
//   // begin the command buffer recording. We will use this command buffer
//   exactly
//   // once, so we want to let vulkan know that
//   VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
//       VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
//
//   // start the command buffer recording
//   VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
//
//   // make the swapchain image into writeable mode before rendering
//   vkutil::transition_image(cmd, _swap_chain_images[swapchainImageIndex],
//                            VK_IMAGE_LAYOUT_UNDEFINED,
//                            VK_IMAGE_LAYOUT_GENERAL);
//
//   // make a clear-color from frame number. This will flash with a 120 frame
//   // period.
//   VkClearColorValue clearValue;
//   float flash = abs(sin(_frame_number / 120.f));
//   clearValue = {{0.0f, 0.0f, flash, 1.0f}};
//
//   VkImageSubresourceRange clearRange =
//       vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
//
//   // clear image
//   vkCmdClearColorImage(cmd, _swap_chain_images[swapchainImageIndex],
//                        VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);
//
//   // make the swapchain image into presentable mode
//   vkutil::transition_image(cmd, _swap_chain_images[swapchainImageIndex],
//                            VK_IMAGE_LAYOUT_GENERAL,
//                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
//
//   // finalize the command buffer (we can no longer add commands, but it can
//   now
//   // be executed)
//   VK_CHECK(vkEndCommandBuffer(cmd));
//
//   // prepare the submission to the queue.
//   // we want to wait on the _presentSemaphore, as that semaphore is signaled
//   // when the swapchain is ready we will signal the _renderSemaphore, to
//   signal
//   // that rendering has finished
//
//   VkCommandBufferSubmitInfo cmdinfo =
//   vkinit::command_buffer_submit_info(cmd);
//
//   VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
//       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
//       get_current_frame()._swapchain_semaphore);
//   VkSemaphoreSubmitInfo signalInfo =
//       vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
//                                     get_current_frame()._render_semaphore);
//
//   VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo,
//   &waitInfo);
//
//   // submit command buffer to the queue and execute it.
//   //  _renderFence will now block until the graphic commands finish execution
//   VK_CHECK(vkQueueSubmit2(_graphics_queue, 1, &submit,
//                           get_current_frame()._render_fence));
//
//   VkPresentInfoKHR presentInfo = {};
//   presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
//   presentInfo.pNext = nullptr;
//   presentInfo.pSwapchains = &_swap_chain;
//   presentInfo.swapchainCount = 1;
//
//   presentInfo.pWaitSemaphores = &get_current_frame()._render_semaphore;
//   presentInfo.waitSemaphoreCount = 1;
//
//   presentInfo.pImageIndices = &swapchainImageIndex;
//
//   VK_CHECK(vkQueuePresentKHR(_graphics_queue, &presentInfo));
//
//   // increase the number of frames drawn
//   _frame_number++;
// }
