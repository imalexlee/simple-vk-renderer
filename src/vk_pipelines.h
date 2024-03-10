#pragma once

#include "vulkan/vulkan_core.h"
#include <vector>
#include <vk_initializers.h>

namespace vkutil {
  bool load_shader_module(const char* filePath, VkDevice device,
                          VkShaderModule* outShaderModule);
} // namespace vkutil

class PipelineBuilder {
  std::vector<VkPipelineShaderStageCreateInfo> _shader_stages;
  VkPipelineInputAssemblyStateCreateInfo _input_assembly;
  VkPipelineRasterizationStateCreateInfo _rasterizer;
  VkPipelineColorBlendAttachmentState _color_blend_attachment;
  VkPipelineMultisampleStateCreateInfo _multisampling;
  VkPipelineDepthStencilStateCreateInfo _depth_stencil;
  VkPipelineRenderingCreateInfo _render_info;
  VkFormat _color_attachment_format;

  void clear();

public:
  PipelineBuilder() { clear(); }

  VkPipelineLayout _pipeline_layout;
  void set_shaders(VkShaderModule vert_shader, VkShaderModule frag_shader);
  void set_input_topology(VkPrimitiveTopology topology);
  void set_polygon_mode(VkPolygonMode poly_mode);
  void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
  void set_multisampling();
  void set_blending();
  void set_color_attachment_formats(VkFormat format);
  void set_depth_format(VkFormat format);
  void disable_depth_test();
  void set_depth_test(bool write_enabled, VkCompareOp compare_op);
  VkPipeline build_pipeline(VkDevice device);
};
