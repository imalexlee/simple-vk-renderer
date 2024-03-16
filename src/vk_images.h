#pragma once
#include <vulkan/vulkan.h>

namespace vkutil {
  void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
  void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dest, VkExtent2D src_extent, VkExtent2D dst_extent);
} // namespace vkutil
