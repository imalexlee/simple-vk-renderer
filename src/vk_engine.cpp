#define GLFW_INCLUDE_VULKAN
#define VMA_IMPLEMENTATION
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "vk_engine.h"
#include "fmt/base.h"
#include "glm/gtx/transform.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "vk_descriptors.h"
#include "vk_images.h"
#include "vk_loader.h"
#include "vk_mem_alloc.h"
#include "vk_pipelines.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <strings.h>
#include <vector>
#include <vk_initializers.h>
#include <vk_types.h>
#include <vulkan/vulkan_core.h>

#define ALLOW_MAILBOX_MODE

const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
constexpr bool use_validation_layers = false;
#else
constexpr bool use_validation_layers = true;
#endif

constexpr std::array<const char*, 1> device_extensions{
    //    "VK_KHR_portability_subset",
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

VulkanEngine* loaded_engine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loaded_engine; }

bool check_validation_support() {
  uint32_t layer_count{};
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> avail_layers(layer_count);

  vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());

  for (const char* layer : validation_layers) {
    if (std::find_if(avail_layers.begin(), avail_layers.end(),
                     [&](VkLayerProperties& avail_layer) {
                       return strcmp(layer, avail_layer.layerName);
                     }) == avail_layers.end()) {
      return false;
    }
  }
  return true;
}

std::vector<const char*> get_required_extensions() {

  uint32_t ext_count{0};
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&ext_count);

  // portability subset for mac

  std::vector<const char*> extensions{};
  for (uint32_t i{0}; i < ext_count; ++i) {
    extensions.emplace_back(glfw_extensions[i]);
  }
  // extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

  if (use_validation_layers) {
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

VkBool32 VulkanEngine::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
};

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
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
                                   const VkAllocationCallbacks* pAllocator) {
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
    VkDebugUtilsMessengerCreateInfoEXT& create_info) {
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

  use_validation_layers ? fmt::println("in debug") : fmt::println("in release");
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  _window = glfwCreateWindow(_window_extent.width, _window_extent.height,
                             "Vulkan window", nullptr, nullptr);
  create_instance();
  setup_debug_messenger();
  create_surface();
  pick_physical_device();
  create_logical_device();
  create_allocator();
  init_swapchain();
  create_image_views();
  init_commands();
  init_sync_structures();
  init_descriptors();
  init_pipelines();
  init_imgui();

  init_default_data();
}

void VulkanEngine::init_default_data() {

  _test_meshes = load_gltf_meshes(this, "../../assets/basicmesh.glb").value();
}

void VulkanEngine::create_allocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = _physical_device;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

  VK_CHECK(vmaCreateAllocator(&allocatorInfo, &_allocator));

  _main_deletion_queue.push_function(
      [&]() { vmaDestroyAllocator(_allocator); });
};

AllocatedBuffer VulkanEngine::create_buffer(size_t alloc_size,
                                            VkBufferUsageFlags buf_usage,
                                            VmaMemoryUsage mem_usage) {

  VkBufferCreateInfo buffer_ci{};
  buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_ci.usage = buf_usage;
  buffer_ci.size = alloc_size;

  VmaAllocationCreateInfo vma_ai{};
  vma_ai.usage = mem_usage;
  vma_ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
  AllocatedBuffer new_buffer{};

  VK_CHECK(vmaCreateBuffer(_allocator, &buffer_ci, &vma_ai, &new_buffer.buffer,
                           &new_buffer.allocation, &new_buffer.info));

  return new_buffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer) {
  vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices,
                                         std::span<Vertex> vertices) {
  const size_t vertex_buf_size = vertices.size() * sizeof(Vertex);
  const size_t index_buf_size = indices.size() * sizeof(uint32_t);

  GPUMeshBuffers new_surface{};
  new_surface.vertex_buf = create_buffer(
      vertex_buf_size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  new_surface.index_buf = create_buffer(index_buf_size,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_GPU_ONLY);

  VkBufferDeviceAddressInfo device_address_info{};
  device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  device_address_info.buffer = new_surface.vertex_buf.buffer;

  new_surface.vertex_buf_address =
      vkGetBufferDeviceAddress(_device, &device_address_info);

  AllocatedBuffer staging = create_buffer(vertex_buf_size + index_buf_size,
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VMA_MEMORY_USAGE_CPU_COPY);

  void* data = staging.allocation->GetMappedData();

  memcpy(data, vertices.data(), vertex_buf_size);
  memcpy((char*)data + vertex_buf_size, indices.data(), index_buf_size);

  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy vertex_copy{};
    vertex_copy.dstOffset = 0;
    vertex_copy.srcOffset = 0;
    vertex_copy.size = vertex_buf_size;

    vkCmdCopyBuffer(cmd, staging.buffer, new_surface.vertex_buf.buffer, 1,
                    &vertex_copy);

    VkBufferCopy index_copy{};
    index_copy.dstOffset = 0;
    index_copy.srcOffset = vertex_buf_size;
    index_copy.size = index_buf_size;

    vkCmdCopyBuffer(cmd, staging.buffer, new_surface.index_buf.buffer, 1,
                    &index_copy);
  });

  destroy_buffer(staging);
  _main_deletion_queue.push_function([=, this]() {
    destroy_buffer(new_surface.index_buf);
    destroy_buffer(new_surface.vertex_buf);
  });

  return new_surface;
}

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
  //  instance_create_info.flags |=
  //      VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  instance_create_info.enabledExtensionCount =
      static_cast<uint32_t>(extensions.size());
  instance_create_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  if (use_validation_layers) {
    instance_create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    instance_create_info.ppEnabledLayerNames = validation_layers.data();
    populate_debug_info(debug_create_info);
    instance_create_info.pNext = static_cast<const void*>(&debug_create_info);

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

  for (const auto& physical_device : physical_devices) {
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
    const std::vector<VkSurfaceFormatKHR>& available_formats) {
  for (const auto& format : available_formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR choose_present_mode(
    const std::vector<VkPresentModeKHR>& available_present_modes) {
#ifdef ALLOW_MAILBOX_MODE
  for (const auto& present_mode : available_present_modes) {
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
VulkanEngine::choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities) {
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
  for (const uint32_t& index : unique_queue_family_indices) {
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

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {

  SwapChainSupportDetails swap_chain_details =
      query_swap_chain_support(_physical_device);
  VkPresentModeKHR present_mode =
      choose_present_mode(swap_chain_details.present_modes);
  VkSurfaceFormatKHR surface_format =
      choose_surface_format(swap_chain_details.formats);

  uint32_t image_count = swap_chain_details.capabilities.minImageCount + 1;

  if (swap_chain_details.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_details.capabilities.maxImageCount) {
    image_count = swap_chain_details.capabilities.maxImageCount;
  }
  VkExtent2D extent = {width, height};
  VkSwapchainCreateInfoKHR swap_chain_create_info{};
  swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swap_chain_create_info.surface = _surface;
  swap_chain_create_info.minImageCount = image_count;
  swap_chain_create_info.imageExtent = extent;
  swap_chain_create_info.imageArrayLayers = 1;
  swap_chain_create_info.imageFormat = surface_format.format;
  swap_chain_create_info.imageColorSpace = surface_format.colorSpace;

  swap_chain_create_info.imageUsage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

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

  VK_CHECK(vkCreateSwapchainKHR(_device, &swap_chain_create_info, nullptr,
                                &_swap_chain));

  VK_CHECK(
      vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count, nullptr));
  _swap_chain_images.clear();
  _swap_chain_images.resize(image_count);

  VK_CHECK(vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count,
                                   _swap_chain_images.data()));
  _swap_chain_format = surface_format.format;
  _swap_chain_extent = extent;

  create_image_views();
}

void VulkanEngine::init_swapchain() {
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

  VkExtent3D draw_image_extent{
      .width = _window_extent.width,
      .height = _window_extent.height,
      .depth = 1,
  };

  _draw_image.image_format = VK_FORMAT_R16G16B16A16_SFLOAT;
  _draw_image.image_extent = draw_image_extent;

  VkImageUsageFlags draw_image_usages{};
  draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
  draw_image_usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VkImageCreateInfo image_create_info = vkinit::image_create_info(
      _draw_image.image_format, draw_image_usages, draw_image_extent);

  VmaAllocationCreateInfo image_alloc_info{};

  image_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  image_alloc_info.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vmaCreateImage(_allocator, &image_create_info, &image_alloc_info,
                          &_draw_image.image, &_draw_image.allocation,
                          nullptr));

  VkImageViewCreateInfo image_view_crete_info = vkinit::imageview_create_info(
      _draw_image.image_format, _draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);

  VK_CHECK(vkCreateImageView(_device, &image_view_crete_info, nullptr,
                             &_draw_image.image_view));

  _depth_image.image_format = VK_FORMAT_D32_SFLOAT;
  _depth_image.image_extent = draw_image_extent;

  VkImageUsageFlags depth_image_usages{};
  depth_image_usages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  VkImageCreateInfo depth_image_ci = vkinit::image_create_info(
      _depth_image.image_format, depth_image_usages, _depth_image.image_extent);

  vmaCreateImage(_allocator, &depth_image_ci, &image_alloc_info,
                 &_depth_image.image, &_depth_image.allocation, nullptr);

  VkImageViewCreateInfo depth_image_view_ci = vkinit::imageview_create_info(
      _depth_image.image_format, _depth_image.image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(_device, &depth_image_view_ci, nullptr,
                             &_depth_image.image_view));

  _main_deletion_queue.push_function([=, this]() {
    vkDestroyImageView(_device, _draw_image.image_view, nullptr);
    vmaDestroyImage(_allocator, _draw_image.image, _draw_image.allocation);
    vkDestroyImageView(_device, _depth_image.image_view, nullptr);
    vmaDestroyImage(_allocator, _depth_image.image, _depth_image.allocation);
  });
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
  VkCommandPoolCreateInfo command_pool_create_info =
      vkinit::command_pool_create_info(
          _graphics_queue_family,
          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // allocating a command pool and buffer in pairs to allow double buffering in
  // rendering
  for (uint32_t i{0}; i < FRAME_OVERLAP; ++i) {
    VK_CHECK(vkCreateCommandPool(_device, &command_pool_create_info, nullptr,
                                 &_frames[i].command_pool));

    VkCommandBufferAllocateInfo buffer_alloc_info =
        vkinit::command_buffer_allocate_info(_frames[i].command_pool);

    VK_CHECK(vkAllocateCommandBuffers(_device, &buffer_alloc_info,
                                      &_frames[i].main_command_buffer));
  }

  // _imm commands
  VK_CHECK(vkCreateCommandPool(_device, &command_pool_create_info, nullptr,
                               &_imm_cmd_pool));
  VkCommandBufferAllocateInfo imm_buffer_alloc_info =
      vkinit::command_buffer_allocate_info(_imm_cmd_pool);

  VK_CHECK(vkAllocateCommandBuffers(_device, &imm_buffer_alloc_info,
                                    &_imm_cmd_buffer));

  _main_deletion_queue.push_function(
      [=, this]() { vkDestroyCommandPool(_device, _imm_cmd_pool, nullptr); });
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

  // _imm sync structures
  VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &_imm_fence));
};

void VulkanEngine::destroy_sync_structures() {
  for (FrameData& frame_data : _frames) {

    vkDestroyFence(_device, frame_data._render_fence, nullptr);
    vkDestroySemaphore(_device, frame_data._render_semaphore, nullptr);
    vkDestroySemaphore(_device, frame_data._swapchain_semaphore, nullptr);
  }

  vkDestroyFence(_device, _imm_fence, nullptr);
}

void VulkanEngine::init_descriptors() {
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes{
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};

  _global_descriptor_allocator.init_pool(_device, 10, sizes);

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _draw_image_descriptor_layout =
        builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }

  // allocate a set from the pool just created
  _draw_image_descriptors = _global_descriptor_allocator.allocate(
      _device, _draw_image_descriptor_layout);

  VkDescriptorImageInfo image_info{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_info.imageView = _draw_image.image_view;

  VkWriteDescriptorSet draw_image_write{};
  draw_image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  draw_image_write.pNext = nullptr;
  draw_image_write.dstSet = _draw_image_descriptors;
  draw_image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  draw_image_write.descriptorCount = 1;
  draw_image_write.pImageInfo = &image_info;

  vkUpdateDescriptorSets(_device, 1, &draw_image_write, 0, nullptr);

  _main_deletion_queue.push_function(
      [&]() { _global_descriptor_allocator.destroy_pool(_device); });
}

void VulkanEngine::init_pipelines() {

  init_background_pipelines();
  init_mesh_pipeline();
}

void VulkanEngine::init_background_pipelines() {
  VkShaderModule gradient_shader{};
  if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv",
                                  _device, &gradient_shader)) {
    fmt::println("Error building compute shader");
  }

  VkShaderModule sky_shader{};
  if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device,
                                  &sky_shader)) {
    fmt::println("Error building compute shader");
  }

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.pNext = nullptr;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = gradient_shader;
  stage_info.pName = "main";

  VkPipelineLayoutCreateInfo compute_layout{};
  compute_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  compute_layout.pNext = nullptr;
  compute_layout.pSetLayouts = &_draw_image_descriptor_layout;
  compute_layout.setLayoutCount = 1;

  VkPushConstantRange push_constant_range{};
  push_constant_range.size = sizeof(ComputePushConstants);
  push_constant_range.offset = 0;
  push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  compute_layout.pPushConstantRanges = &push_constant_range;
  compute_layout.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &compute_layout, nullptr,
                                  &_gradient_pipeline_layout));

  VkComputePipelineCreateInfo compute_pipeline_create_info{};
  compute_pipeline_create_info.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  compute_pipeline_create_info.pNext = nullptr;
  compute_pipeline_create_info.layout = _gradient_pipeline_layout;
  compute_pipeline_create_info.stage = stage_info;

  ComputeEffect gradient{};
  gradient.name = "gradient";
  gradient.pipeline_layout = _gradient_pipeline_layout;
  gradient.data = {};
  gradient.data.data1 = glm::vec4(1, 0, 0, 1);
  gradient.data.data2 = glm::vec4(0, 0, 1, 1);

  ComputeEffect sky{};
  sky.name = "gradient";
  sky.pipeline_layout = _gradient_pipeline_layout;
  sky.data = {};
  sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

  VK_CHECK(vkCreateComputePipelines(_device, nullptr, 1,
                                    &compute_pipeline_create_info, nullptr,
                                    &gradient.pipeline));

  // change the module of the shader stage. only thing that needs to change
  compute_pipeline_create_info.stage.module = sky_shader;

  VK_CHECK(vkCreateComputePipelines(_device, nullptr, 1,
                                    &compute_pipeline_create_info, nullptr,
                                    &sky.pipeline));

  _background_effects.push_back(gradient);
  _background_effects.push_back(sky);

  vkDestroyShaderModule(_device, gradient_shader, nullptr);
  vkDestroyShaderModule(_device, sky_shader, nullptr);

  _main_deletion_queue.push_function([=, this]() {
    vkDestroyPipelineLayout(_device, _gradient_pipeline_layout, nullptr);
    vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    vkDestroyPipeline(_device, sky.pipeline, nullptr);
  });
}

void VulkanEngine::init_mesh_pipeline() {
  VkShaderModule triangle_vert_shader{};
  if (!vkutil::load_shader_module(
          "../../shaders/colored_triangle_mesh.vert.spv", _device,
          &triangle_vert_shader)) {
    fmt::print("Triangle vertex shader could not be loaded");
  }
  VkShaderModule triangle_frag_shader{};
  if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv",
                                  _device, &triangle_frag_shader)) {
    fmt::print("Triangle fragment could not be loaded");
  }

  VkPushConstantRange buffer_range{};
  buffer_range.size = sizeof(GPUDrawPushConstants);
  buffer_range.offset = 0;
  buffer_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo pipeline_layout_ci =
      vkinit::pipeline_layout_create_info();
  pipeline_layout_ci.pPushConstantRanges = &buffer_range;
  pipeline_layout_ci.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_ci, nullptr,
                                  &_mesh_pipeline_layout));

  PipelineBuilder pipeline_builder;
  pipeline_builder._pipeline_layout = _mesh_pipeline_layout;
  pipeline_builder.set_multisampling();
  pipeline_builder.set_shaders(triangle_vert_shader, triangle_frag_shader);
  pipeline_builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pipeline_builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  pipeline_builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  pipeline_builder.enable_blending_additive();
  pipeline_builder.set_color_attachment_formats(_draw_image.image_format);
  pipeline_builder.set_depth_format(_depth_image.image_format);
  pipeline_builder.set_depth_test(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

  _mesh_pipeline = pipeline_builder.build_pipeline(_device);

  vkDestroyShaderModule(_device, triangle_vert_shader, nullptr);
  vkDestroyShaderModule(_device, triangle_frag_shader, nullptr);

  _main_deletion_queue.push_function([&]() {
    vkDestroyPipelineLayout(_device, _mesh_pipeline_layout, nullptr);
    vkDestroyPipeline(_device, _mesh_pipeline, nullptr);
  });
}

void VulkanEngine::init_imgui() {
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr,
                                  &_imm_descriptor_pool));

  ImGui::CreateContext();

  ImGui_ImplGlfw_InitForVulkan(_window, true);

  VkPipelineRenderingCreateInfoKHR pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  pipeline_info.pColorAttachmentFormats = &_swap_chain_format;
  pipeline_info.colorAttachmentCount = 1;

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = _instance;
  init_info.PhysicalDevice = _physical_device;
  init_info.Device = _device;
  init_info.Queue = _graphics_queue;
  init_info.DescriptorPool = _imm_descriptor_pool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = true;
  init_info.PipelineRenderingCreateInfo = pipeline_info;

  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info);

  immediate_submit(
      [&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(); });

  ImGui_ImplVulkan_DestroyFontsTexture();

  _main_deletion_queue.push_function([=, this]() {
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
  });
}

void VulkanEngine::immediate_submit(
    std::function<void(VkCommandBuffer cmd)>&& function) {

  VK_CHECK(vkResetFences(_device, 1, &_imm_fence));
  VK_CHECK(vkResetCommandBuffer(_imm_cmd_buffer, 0));

  // VkCommandBuffer cmd = _imm_cmd_buffer;
  VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(_imm_cmd_buffer, &cmd_begin_info));

  function(_imm_cmd_buffer);

  VK_CHECK(vkEndCommandBuffer(_imm_cmd_buffer));

  VkCommandBufferSubmitInfo cmd_info =
      vkinit::command_buffer_submit_info(_imm_cmd_buffer);
  VkSubmitInfo2 submit_info = vkinit::submit_info(&cmd_info, nullptr, nullptr);
  VK_CHECK(vkQueueSubmit2(_graphics_queue, 1, &submit_info, _imm_fence));
  VK_CHECK(vkWaitForFences(_device, 1, &_imm_fence, VK_TRUE, 9999999999));
};

void VulkanEngine::cleanup() {
  _main_deletion_queue.flush();
  vkDestroyDescriptorSetLayout(_device, _draw_image_descriptor_layout, nullptr);
  vkDestroyDescriptorPool(_device, _imm_descriptor_pool, nullptr);

  destroy_swapchain();
  for (FrameData& frame_data : _frames) {
    vkDestroyCommandPool(_device, frame_data.command_pool, nullptr);
    frame_data.deletion_queue.flush();
  }
  destroy_sync_structures();
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

    if (_resize_requested) {
      resize_swapchain();
    }

    ImGui_ImplVulkan_NewFrame();

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::Begin("background effect :)")) {

      ComputeEffect& effect = _background_effects[_current_background_effect];

      ImGui::Text("Selected Effect: (%s)", effect.name);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / io.Framerate, io.Framerate);
      ImGui::SliderInt("Effect Index", &_current_background_effect, 0,
                       _background_effects.size() - 1);
      ImGui::InputFloat4("data1", (float*)&effect.data.data1);
      ImGui::InputFloat4("data2", (float*)&effect.data.data2);
      ImGui::InputFloat4("data3", (float*)&effect.data.data3);
      ImGui::InputFloat4("data4", (float*)&effect.data.data4);
    }

    ImGui::End();
    ImGui::Render();
    draw();

    glfwPollEvents();
  }
  vkDeviceWaitIdle(_device);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd,
                              VkImageView target_image_view) {

  VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(
      target_image_view, nullptr, VK_IMAGE_LAYOUT_GENERAL);
  VkRenderingInfo render_info =
      vkinit::rendering_info(_swap_chain_extent, &color_attachment, nullptr);

  vkCmdBeginRendering(cmd, &render_info);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
}

void VulkanEngine::draw() {

  _draw_extent.height =
      std::min(_swap_chain_extent.height, _draw_image.image_extent.height) *
      _render_scale;

  _draw_extent.width =
      std::min(_swap_chain_extent.width, _draw_image.image_extent.width) *
      _render_scale;

  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._render_fence, true,
                           10000000000));
  get_current_frame().deletion_queue.flush();

  uint32_t image_index{};
  VkResult result = vkAcquireNextImageKHR(
      _device, _swap_chain, 10000000000,
      get_current_frame()._swapchain_semaphore, nullptr, &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    _resize_requested = true;
    return;
  }

  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._render_fence));

  VkCommandBuffer cmd = get_current_frame().main_command_buffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo command_begin_info{};
  command_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_begin_info.pNext = nullptr;
  command_begin_info.pInheritanceInfo = nullptr;
  command_begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  _draw_extent.width = _draw_image.image_extent.width;
  _draw_extent.height = _draw_image.image_extent.height;

  VK_CHECK(vkBeginCommandBuffer(cmd, &command_begin_info));

  // transition drawing image to a writeable mode
  vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_GENERAL);

  draw_background(cmd);

  vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  vkutil::transition_image(cmd, _depth_image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  draw_geometry(cmd);

  // transition swapchain image into a transfer destination
  vkutil::transition_image(cmd, _draw_image.image,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  // transition drawing image to a transer source
  vkutil::transition_image(cmd, _swap_chain_images[image_index],
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  // copy the _draw_image that was drawn to the swapchain image
  vkutil::copy_image(cmd, _draw_image.image, _swap_chain_images[image_index],
                     _draw_extent, _swap_chain_extent);

  // transition swapchain image to a presentable mode after it was copied into
  vkutil::transition_image(cmd, _swap_chain_images[image_index],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  draw_imgui(cmd, _swap_chain_image_views[image_index]);

  vkutil::transition_image(cmd, _swap_chain_images[image_index],
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &_swap_chain;
  present_info.pImageIndices = &image_index;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &get_current_frame()._render_semaphore;
  present_info.pNext = nullptr;

  result = vkQueuePresentKHR(_graphics_queue, &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    _resize_requested = true;
    return;
  }

  ++_frame_number;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {

  const ComputeEffect& effect = _background_effects[_current_background_effect];

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          effect.pipeline_layout, 0, 1,
                          &_draw_image_descriptors, 0, nullptr);
  ComputePushConstants pc;
  pc.data1 = glm::vec4(1, 0, 0, 1);
  pc.data2 = glm::vec4(0, 0, 1, 1);

  vkCmdPushConstants(cmd, _background_effects[1].pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(ComputePushConstants), &effect.data);
  vkCmdDispatch(cmd, std::ceil(_draw_extent.width / 16.0),
                std::ceil(_draw_extent.height / 16.0), 1);
};

void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {
  VkRenderingAttachmentInfo color_attachment_info = vkinit::attachment_info(
      _draw_image.image_view, nullptr, VK_IMAGE_LAYOUT_GENERAL);

  VkRenderingAttachmentInfo depth_attachment_info =
      vkinit::depth_attachment_info(_depth_image.image_view,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  VkRenderingInfo rendering_info = vkinit::rendering_info(
      _draw_extent, &color_attachment_info, &depth_attachment_info);

  vkCmdBeginRendering(cmd, &rendering_info);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _mesh_pipeline);

  VkViewport viewport = {};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = _draw_extent.width;
  viewport.height = _draw_extent.height;
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor = {};
  scissor.offset.x = 0;
  scissor.offset.y = 0;
  scissor.extent.width = viewport.width;
  scissor.extent.height = viewport.height;

  vkCmdSetScissor(cmd, 0, 1, &scissor);

  GPUDrawPushConstants push_constants{};
  push_constants.world_mat = glm::mat4{1.f};

  push_constants.vertex_buf_address =
      _test_meshes[2]->meshBuffers.vertex_buf_address;

  // flip monkay
  glm::mat4 view = glm::translate(glm::vec3{0, 0, -5});

  glm::mat4 projection = glm::perspective(
      glm::radians(70.f),
      (float)_draw_extent.width / (float)_draw_extent.height, 10000.f, 0.1f);

  projection[1][1] *= -1;
  push_constants.world_mat *= projection * view;

  vkCmdPushConstants(cmd, _mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(GPUDrawPushConstants), &push_constants);
  vkCmdBindIndexBuffer(cmd, _test_meshes[2]->meshBuffers.index_buf.buffer, 0,
                       VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(cmd, _test_meshes[2]->surfaces[0].count, 1,
                   _test_meshes[2]->surfaces[0].startIndex, 0, 0);
  vkCmdEndRendering(cmd);
}
void VulkanEngine::destroy_swapchain() {
  vkDestroySwapchainKHR(_device, _swap_chain, nullptr);
  for (auto& image_view : _swap_chain_image_views) {
    vkDestroyImageView(_device, image_view, nullptr);
  }
};
void VulkanEngine::resize_swapchain() {
  vkDeviceWaitIdle(_device);
  destroy_swapchain();
  destroy_sync_structures();
  int w, h;
  glfwGetWindowSize(_window, &w, &h);

  _window_extent.width = w;
  _window_extent.height = h;

  create_swapchain(_window_extent.width, _window_extent.height);
  init_sync_structures();

  _resize_requested = false;
}
