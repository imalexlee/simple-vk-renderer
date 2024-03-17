#include "fmt/base.h"
#include "vulkan/vulkan_core.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vk_initializers.h>
#include <vk_pipelines.h>

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule) {
  // open the file. With cursor at the end
  std::ifstream file(filePath, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    return false;
  }

  // find what the size of the file is by looking up the location of the cursor
  // because the cursor is at the end, it gives the size directly in bytes
  size_t fileSize = (size_t)file.tellg();

  // spirv expects the buffer to be on uint32, so make sure to reserve a int
  // vector big enough for the entire file
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

  // put file cursor at beginning
  file.seekg(0);

  // load the entire file into the buffer
  file.read((char*)buffer.data(), fileSize);

  // now that the file is loaded into the buffer, we can close it
  file.close();

  // create a new shader module, using the buffer we loaded
  VkShaderModuleCreateInfo createInfo = {};

  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pNext = nullptr;

  // codeSize has to be in bytes, so multply the ints in the buffer by size of
  // int to know the real size of the buffer
  createInfo.codeSize = buffer.size() * sizeof(uint32_t);
  createInfo.pCode = buffer.data();

  // check that the creation goes well.
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    return false;
  }
  *outShaderModule = shaderModule;
  return true;
}

void PipelineBuilder::clear() {
  _input_assembly = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  _rasterizer = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  _color_blend_attachment = {};
  _multisampling = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  _pipeline_layout = {};
  _depth_stencil = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  _render_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

  _shader_stages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device) {

  VkPipelineViewportStateCreateInfo viewport_state{};

  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineColorBlendStateCreateInfo color_blend_state{};
  color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.pNext = nullptr;
  color_blend_state.logicOpEnable = VK_FALSE;
  color_blend_state.logicOp = VK_LOGIC_OP_COPY;
  color_blend_state.attachmentCount = 1;
  color_blend_state.pAttachments = &_color_blend_attachment;

  VkPipelineVertexInputStateCreateInfo _vertex_input_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

  constexpr std::array<VkDynamicState, 2> dynamic_states = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};

  VkPipelineDynamicStateCreateInfo dynamic_state_ci{};
  dynamic_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state_ci.pDynamicStates = dynamic_states.data();
  dynamic_state_ci.dynamicStateCount = dynamic_states.size();

  pipeline_info.pNext = &_render_info;
  pipeline_info.stageCount = _shader_stages.size();
  pipeline_info.pStages = _shader_stages.data();
  pipeline_info.pVertexInputState = &_vertex_input_info;
  pipeline_info.layout = _pipeline_layout;
  pipeline_info.pMultisampleState = &_multisampling;
  pipeline_info.pRasterizationState = &_rasterizer;
  pipeline_info.pColorBlendState = &color_blend_state;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pInputAssemblyState = &_input_assembly;
  pipeline_info.pDepthStencilState = &_depth_stencil;

  pipeline_info.pDynamicState = &dynamic_state_ci;

  VkPipeline pipeline{};
  if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
    fmt::println("Failed to create graphics pipeline");
    return VK_NULL_HANDLE;
  }
  return pipeline;
}

void PipelineBuilder::set_shaders(VkShaderModule vert_shader, VkShaderModule frag_shader) {
  _shader_stages.clear();

  _shader_stages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vert_shader));
  _shader_stages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, frag_shader));
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology) {
  _input_assembly.topology = topology;
  _input_assembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode poly_mode) {
  _rasterizer.polygonMode = poly_mode;
  _rasterizer.lineWidth = 1.f;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face) {
  _rasterizer.cullMode = cull_mode;
  _rasterizer.frontFace = front_face;
}

void PipelineBuilder::set_multisampling(VkSampleCountFlagBits samples) {

  _multisampling.rasterizationSamples = samples;
  _multisampling.sampleShadingEnable = VK_FALSE;
  _multisampling.minSampleShading = 1.0f;
  _multisampling.pSampleMask = nullptr;
  _multisampling.alphaToCoverageEnable = VK_FALSE;
  _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::disable_blending() {
  _color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  _color_blend_attachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::enable_blending_additive() {
  _color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  _color_blend_attachment.blendEnable = VK_TRUE;
  _color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  _color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
  _color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  _color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  _color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  _color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
}
void PipelineBuilder::enable_blending_alphablend() {
  _color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  _color_blend_attachment.blendEnable = VK_TRUE;
  _color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  _color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
  _color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  _color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  _color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  _color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
}
void PipelineBuilder::set_color_attachment_formats(VkFormat format) {
  _color_attachment_format = format;
  _render_info.pColorAttachmentFormats = &_color_attachment_format;
  _render_info.colorAttachmentCount = 1;
}

void PipelineBuilder::set_depth_format(VkFormat format) { _render_info.depthAttachmentFormat = format; }
void PipelineBuilder::disable_depth_test() {
  _depth_stencil.depthTestEnable = VK_FALSE;
  _depth_stencil.depthWriteEnable = VK_FALSE;
  _depth_stencil.depthCompareOp = VK_COMPARE_OP_NEVER;
  _depth_stencil.depthBoundsTestEnable = VK_FALSE;
  _depth_stencil.stencilTestEnable = VK_FALSE;
  _depth_stencil.front = {};
  _depth_stencil.back = {};
  _depth_stencil.minDepthBounds = 0.f;
  _depth_stencil.maxDepthBounds = 1.f;
}
void PipelineBuilder::set_depth_test(bool write_enabled, VkCompareOp compare_op) {
  _depth_stencil.depthTestEnable = VK_TRUE;
  _depth_stencil.depthWriteEnable = write_enabled;
  _depth_stencil.depthCompareOp = compare_op;
  _depth_stencil.depthBoundsTestEnable = VK_FALSE;
  _depth_stencil.stencilTestEnable = VK_FALSE;
  _depth_stencil.front = {};
  _depth_stencil.back = {};
  _depth_stencil.minDepthBounds = 0.f;
  _depth_stencil.maxDepthBounds = 1.f;
}
