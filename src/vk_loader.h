#pragma once

#include "vk_descriptors.h"
#include <filesystem>
#include <vk_types.h>

struct GLTFMaterial {
  MaterialInstance data;
};

struct GeoSurface {
  uint32_t startIndex;
  uint32_t count;
  std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
  std::string name;
  std::vector<GeoSurface> surfaces;
  GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;

struct LoadedGLTF : public IRenderable {
public:
  std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
  std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
  std::unordered_map<std::string, AllocatedImage> images;
  std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

  // nodes that don't have a parent
  std::vector<std::shared_ptr<Node>> top_nodes;

  std::vector<VkSampler> samplers;

  DescriptorAllocatorGrowable descriptor_pool;

  AllocatedBuffer material_data_buffer;

  VulkanEngine* creator;

  ~LoadedGLTF() { clear_all(); }

  void Draw(const glm::mat4& top_matrix, DrawContext& ctx) override;

private:
  void clear_all();
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf_meshes(VulkanEngine* engine, std::filesystem::path filePath);
