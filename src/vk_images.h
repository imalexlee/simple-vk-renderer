#pragma once
#include <vulkan/vulkan.h>

namespace vkutil {
  void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
  void copy_image(VkCommandBuffer cmd, VkImage src, VkImage dest, VkExtent2D src_extent, VkExtent2D dst_extent);
  void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D image_size);
} // namespace vkutil
