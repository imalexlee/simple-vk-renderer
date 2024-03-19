#define GLM_ENABLE_EXPERIMENTAL 1

#include <iostream>
#include <memory>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_types.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

constexpr bool MIPMAP_ENABLED = true;

static VkFilter extract_filter(fastgltf::Filter filter) {
  switch (filter) {
  // nearest samplers
  case fastgltf::Filter::Nearest:
  case fastgltf::Filter::NearestMipMapNearest:
  case fastgltf::Filter::NearestMipMapLinear:
    return VK_FILTER_NEAREST;

  // linear samplers
  case fastgltf::Filter::Linear:
  case fastgltf::Filter::LinearMipMapNearest:
  case fastgltf::Filter::LinearMipMapLinear:
  default:
    return VK_FILTER_LINEAR;
  }
}

static VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter) {
  switch (filter) {
  case fastgltf::Filter::NearestMipMapNearest:
  case fastgltf::Filter::LinearMipMapNearest:
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;

  case fastgltf::Filter::NearestMipMapLinear:
  case fastgltf::Filter::LinearMipMapLinear:
  default:
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

void LoadedGLTF::Draw(const glm::mat4& top_matrix, DrawContext& ctx) {
  for (auto& n : top_nodes) {
    n->Draw(top_matrix, ctx);
  }
}

void LoadedGLTF::clear_all() {
  VkDevice dv = creator->_device;
  descriptor_pool.destroy_pools(dv);
  creator->destroy_buffer(material_data_buffer);

  for (auto& [k, v] : images) {

    if (v.image == creator->_error_checkerboard_image.image) {
      // dont destroy the default images
      continue;
    }
    creator->destroy_image(v);
  }

  for (auto& sampler : samplers) {
    vkDestroySampler(dv, sampler, nullptr);
  }
}

std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image) {
  AllocatedImage newImage{};

  int width, height, nrChannels;
  std::visit(fastgltf::visitor{
                 [](auto& arg) {},
                 [&](fastgltf::sources::URI& filePath) {
                   assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                   assert(filePath.uri.isLocalPath());   // We're only capable of loading
                                                         // local files.

                   const std::string path(filePath.uri.path().begin(),
                                          filePath.uri.path().end()); // Thanks C++.
                   unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                   if (data) {
                     VkExtent3D imagesize;
                     imagesize.width = width;
                     imagesize.height = height;
                     imagesize.depth = 1;

                     newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                     VK_IMAGE_USAGE_SAMPLED_BIT, MIPMAP_ENABLED);

                     stbi_image_free(data);
                   }
                 },
                 [&](fastgltf::sources::Array& vector) {
                   unsigned char* data = stbi_load_from_memory(
                       vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                   if (data) {
                     VkExtent3D imagesize;
                     imagesize.width = width;
                     imagesize.height = height;
                     imagesize.depth = 1;

                     newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                     VK_IMAGE_USAGE_SAMPLED_BIT, MIPMAP_ENABLED);

                     stbi_image_free(data);
                   }
                 },
                 [&](fastgltf::sources::BufferView& view) {
                   auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                   auto& buffer = asset.buffers[bufferView.bufferIndex];

                   std::visit(fastgltf::visitor{// We only care about VectorWithMime here, because we
                                                // specify LoadExternalBuffers, meaning all buffers
                                                // are already loaded into a vector.
                                                [](auto& arg) {},
                                                [&](fastgltf::sources::Array& vector) {
                                                  unsigned char* data =
                                                      stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset,
                                                                            static_cast<int>(bufferView.byteLength),
                                                                            &width, &height, &nrChannels, 4);
                                                  if (data) {
                                                    VkExtent3D imagesize;
                                                    imagesize.width = width;
                                                    imagesize.height = height;
                                                    imagesize.depth = 1;

                                                    newImage = engine->create_image(
                                                        data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                        VK_IMAGE_USAGE_SAMPLED_BIT, MIPMAP_ENABLED);

                                                    stbi_image_free(data);
                                                  }
                                                }},
                              buffer.data);
                 },
             },
             image.data);

  // if any of the attempts to load the data failed, we havent written the image
  // so handle is null
  if (newImage.image == VK_NULL_HANDLE) {
    return {};
  } else {
    return newImage;
  }
}

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf_meshes(VulkanEngine* engine, std::filesystem::path filePath) {
  std::cout << "Loading GLTF: " << filePath << std::endl;

  std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
  scene->creator = engine;
  LoadedGLTF& file = *scene.get();

  static constexpr auto supported_extensions = fastgltf::Extensions::KHR_mesh_quantization |
                                               fastgltf::Extensions::KHR_texture_transform |
                                               fastgltf::Extensions::KHR_materials_variants;

  fastgltf::Parser parser(supported_extensions);

  constexpr auto gltf_options = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
                                fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers |
                                fastgltf::Options::LoadExternalImages | fastgltf::Options::GenerateMeshIndices;
  fastgltf::GltfDataBuffer data;
  data.loadFromFile(filePath);

  fastgltf::Asset gltf;

  std::filesystem::path path = filePath;

  auto load = parser.loadGltf(&data, path.parent_path(), gltf_options);
  if (load) {
    gltf = std::move(load.get());
  } else {
    std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
    return {};
  }
  std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                                                                   {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
                                                                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
  file.descriptor_pool.init(engine->_device, gltf.materials.size(), sizes);

  for (auto& gltf_sampler : gltf.samplers) {
    VkSamplerCreateInfo sampler_ci{};
    sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_ci.pNext = nullptr;
    sampler_ci.maxLod = VK_LOD_CLAMP_NONE;
    sampler_ci.minLod = 0;
    sampler_ci.magFilter = extract_filter(gltf_sampler.magFilter.value_or(fastgltf::Filter::Nearest));
    sampler_ci.minFilter = extract_filter(gltf_sampler.minFilter.value_or(fastgltf::Filter::Nearest));
    sampler_ci.mipmapMode = extract_mipmap_mode(gltf_sampler.minFilter.value_or(fastgltf::Filter::Nearest));

    VkSampler new_sampler;
    vkCreateSampler(engine->_device, &sampler_ci, nullptr, &new_sampler);

    file.samplers.push_back(new_sampler);
  }

  std::vector<std::shared_ptr<MeshAsset>> meshes;
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<AllocatedImage> images;
  std::vector<std::shared_ptr<GLTFMaterial>> materials;

  for (fastgltf::Image& image : gltf.images) {
    std::optional<AllocatedImage> img = load_image(engine, gltf, image);

    if (img.has_value()) {
      images.push_back(*img);
      file.images[image.name.c_str()] = *img;
    } else {
      // we failed to load, so lets give the slot a default white texture to not
      // completely break loading
      images.push_back(engine->_error_checkerboard_image);
      std::cout << "gltf failed to load texture " << image.name << std::endl;
    }
  }

  file.material_data_buffer =
      engine->create_buffer(sizeof(GLTFMettallicRoughness::MaterialConstants) * gltf.materials.size(),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  uint32_t data_index{0};
  GLTFMettallicRoughness::MaterialConstants* scene_material_constants =
      (GLTFMettallicRoughness::MaterialConstants*)file.material_data_buffer.info.pMappedData;

  // we have an asset now with materials. loop over the materials and load their properties into materials vector
  for (fastgltf::Material& mat : gltf.materials) {
    std::shared_ptr<GLTFMaterial> new_mat = std::make_shared<GLTFMaterial>();
    materials.push_back(new_mat);
    file.materials[mat.name.c_str()] = new_mat;

    GLTFMettallicRoughness::MaterialConstants constants;
    constants.color_factors.x = mat.pbrData.baseColorFactor[0];
    constants.color_factors.y = mat.pbrData.baseColorFactor[1];
    constants.color_factors.z = mat.pbrData.baseColorFactor[2];
    constants.color_factors.w = mat.pbrData.baseColorFactor[3];
    constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
    constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;

    scene_material_constants[data_index] = constants;

    MaterialPass pass_type;
    if (mat.alphaMode == fastgltf::AlphaMode::Blend) {
      pass_type = MaterialPass::Transparent;
    } else {
      pass_type = MaterialPass::MainColor;
    }

    GLTFMettallicRoughness::MaterialResources material_resources;
    material_resources.color_image = engine->_white_image;
    material_resources.color_sampler = engine->_default_sampler_linear;
    material_resources.metal_rough_image = engine->_white_image;
    material_resources.metal_rough_sampler = engine->_default_sampler_linear;
    material_resources.data_buffer = file.material_data_buffer.buffer;
    material_resources.data_buffer_offset = data_index * sizeof(GLTFMettallicRoughness::MaterialConstants);

    // grab gltf textures
    if (mat.pbrData.baseColorTexture.has_value()) {
      size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
      size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

      material_resources.color_image = images[img];
      material_resources.color_sampler = file.samplers[sampler];
    }

    new_mat->data = engine->metal_rough_material.write_material(engine->_device, pass_type, material_resources,
                                                                file.descriptor_pool);

    data_index++;
  }

  std::vector<uint32_t> indices;
  std::vector<Vertex> vertices;

  for (fastgltf::Mesh& mesh : gltf.meshes) {
    std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
    meshes.push_back(newmesh);
    file.meshes[mesh.name.c_str()] = newmesh;
    newmesh->name = mesh.name;

    // clear the mesh arrays each mesh, we dont want to merge them by error
    indices.clear();
    vertices.clear();

    for (auto&& p : mesh.primitives) {
      GeoSurface newSurface;
      newSurface.startIndex = (uint32_t)indices.size();
      newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

      size_t initial_vtx = vertices.size();

      // load indexes
      {
        fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
        indices.reserve(indices.size() + indexaccessor.count);

        fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                                                 [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
      }

      // load vertex positions
      {
        fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
        vertices.resize(vertices.size() + posAccessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
          Vertex newvtx;
          newvtx.position = v;
          newvtx.normal = {1, 0, 0};
          newvtx.color = glm::vec4{1.f};
          newvtx.uv_x = 0;
          newvtx.uv_y = 0;
          vertices[initial_vtx + index] = newvtx;
        });
      }

      // load vertex normals
      auto normals = p.findAttribute("NORMAL");
      if (normals != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            gltf, gltf.accessors[(*normals).second],
            [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
      }

      // load UVs
      auto uv = p.findAttribute("TEXCOORD_0");
      if (uv != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                                                      [&](glm::vec2 v, size_t index) {
                                                        vertices[initial_vtx + index].uv_x = v.x;
                                                        vertices[initial_vtx + index].uv_y = v.y;
                                                      });
      }

      // load vertex colors
      auto colors = p.findAttribute("COLOR_0");
      if (colors != p.attributes.end()) {

        fastgltf::iterateAccessorWithIndex<glm::vec4>(
            gltf, gltf.accessors[(*colors).second],
            [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
      }

      if (p.materialIndex.has_value()) {
        newSurface.material = materials[p.materialIndex.value()];
      } else {
        newSurface.material = materials[0];
      }

      glm::vec3 min_pos = vertices[initial_vtx].position;
      glm::vec3 max_pos = vertices[initial_vtx].position;
      for (auto& vert : vertices) {
        min_pos = glm::min(min_pos, vert.position);
        max_pos = glm::max(max_pos, vert.position);
      }
      newSurface.bounds.origin = (max_pos + min_pos) / 2.f;
      // box size
      newSurface.bounds.extents = (max_pos - min_pos) / 2.f;
      newSurface.bounds.sphere_radius = glm::length(newSurface.bounds.extents);
      newmesh->surfaces.push_back(newSurface);
    }

    newmesh->meshBuffers = engine->upload_mesh(indices, vertices);
  }

  // load all nodes and their meshes
  for (fastgltf::Node& node : gltf.nodes) {
    std::shared_ptr<Node> newNode;

    // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode class
    if (node.meshIndex.has_value()) {
      newNode = std::make_shared<MeshNode>();
      static_cast<MeshNode*>(newNode.get())->mesh = meshes[*node.meshIndex];
    } else {
      newNode = std::make_shared<Node>();
    }

    nodes.push_back(newNode);
    file.nodes[node.name.c_str()];

    std::visit(fastgltf::visitor{[&](fastgltf::Node::TransformMatrix matrix) {
                                   newNode->local_transform = glm::make_mat4x4(matrix.data());
                                 },
                                 [&](fastgltf::TRS transform) {
                                   glm::vec3 tl = glm::make_vec3(transform.translation.data());
                                   glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1],
                                                 transform.rotation[2]);
                                   glm::vec3 sc = glm::make_vec3(transform.scale.data());

                                   glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                   glm::mat4 rm = glm::toMat4(rot);
                                   glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                   newNode->local_transform = tm * rm * sm;
                                 }},
               node.transform);

    // newNode->local_transform[1][1] *= -1;
    // newNode->local_transform = glm::rotate(newNode->local_transform, glm::radians(-90.f), glm::vec3{1, 0, 0});
  }

  // run loop again to setup transform hierarchy
  for (int i = 0; i < gltf.nodes.size(); i++) {
    fastgltf::Node& node = gltf.nodes[i];
    std::shared_ptr<Node>& sceneNode = nodes[i];

    for (auto& c : node.children) {
      sceneNode->children.push_back(nodes[c]);
      nodes[c]->parent = sceneNode;
    }
  }

  // find the top nodes, with no parents
  for (auto& node : nodes) {
    if (node->parent.lock() == nullptr) {
      file.top_nodes.push_back(node);
      node->refresh_transform(glm::mat4{1.f});
    }
  }
  return scene;
}
