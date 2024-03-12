#pragma once

#include <filesystem>
#include <unordered_map>
#include <vk_types.h>

struct GeoSurface {
  uint32_t startIndex;
  uint32_t count;
};

struct MeshAsset {
  std::string name;

  std::vector<GeoSurface> surfaces;
  GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>>
load_gltf_meshes(VulkanEngine* engine, std::filesystem::path filePath);
