// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IAssetLoader.h"
#include "irr/asset/CGLTFPipelineMetadata.h"

namespace irr
{
	namespace asset
	{
		//! glTF Loader capable of loading .gltf files
		/*
			glTF bridges the gap between 3D content creation tools and modern 3D applications 
			by providing an efficient, extensible, interoperable format for the transmission and loading of 3D content.
		*/

		class CGLTFLoader final : public asset::IAssetLoader
		{
			protected:
				virtual ~CGLTFLoader() {}

			public:
				CGLTFLoader(asset::IAssetManager* _m_assetMgr);

				bool isALoadableFileFormat(io::IReadFile* _file) const override;

				const char** getAssociatedFileExtensions() const override
				{
					static const char* extensions[]{ "gltf", nullptr };
					return extensions;
				}

				uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

				asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

				_IRR_STATIC_INLINE std::string getPipelineCacheKey(const E_PRIMITIVE_TOPOLOGY& primitiveType, const SVertexInputParams& vertexInputParams)
				{
					return "irr/builtin/pipelines/loaders/glTF/" + std::to_string(primitiveType) + vertexInputParams.to_string();
				}

			private:

				struct CGLTFHeader
				{
					uint32_t version;
					std::optional<uint32_t> minVersion;
					std::optional<std::string> generator;
					std::optional<std::string> copyright;
				};

				/*
					A node in the node hierarchy. A node can have either a matrix or any combination of 
					translation/rotation/scale (TRS) properties.
				*/

				struct SGLTFNode
				{
					SGLTFNode() {}
					virtual ~SGLTFNode() {}

					std::optional<std::string> name;				//!< The user-defined name of this object.
					std::optional<std::vector<uint32_t>> children;	//!< The indices of this node's children.
					std::optional<uint32_t> mesh;					//!< The index of the mesh in this node.
					std::optional<uint32_t> camera;					//!< The index of the camera referenced by this node.
					std::optional<uint32_t> skin;					//!< The index of the skin referenced by this node.
					std::optional<uint32_t> weights;				//!< The weights of the instantiated Morph Target. Number of elements must match number of Morph Targets of used mesh.

					struct SGLTFNTransformationTRS
					{
						core::vector3df_SIMD translation;			//!< The node's translation along the x, y, and z axes.
						core::vector3df_SIMD scale;					//!< The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
						core::vector4df_SIMD rotation;				//!< The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
					};

					union SGLTFTransformation
					{
						SGLTFTransformation() {}
						~SGLTFTransformation() {}

						SGLTFTransformation(SGLTFTransformation& copy)
						{
							std::memmove(this, &copy, sizeof(SGLTFTransformation));
						}

						SGLTFTransformation(const SGLTFTransformation& copy)
						{
							std::memmove(this, &copy, sizeof(SGLTFTransformation));
						}

						SGLTFTransformation& operator=(SGLTFTransformation& copy)
						{
							std::memmove(this, &copy, sizeof(SGLTFTransformation));
							return *this;
						}

						SGLTFTransformation& operator=(const SGLTFTransformation& copy)
						{
							std::memmove(this, &copy, sizeof(SGLTFTransformation));
							return *this;
						}

						SGLTFNTransformationTRS trs;	
						core::matrix4SIMD matrix;		//!< A floating-point 4x4 transformation matrix stored in column-major order.
					} transformation;

					/*
						A set of primitives to be rendered. A node can contain one mesh.
						A node's transform places the mesh in the scene.
					*/

					struct SGLTFMesh
					{
						struct SPrimitive
						{
							std::optional<uint32_t> indices;									//!< The index of the accessor that contains the indices.
							std::optional<uint32_t> material;									//!< The index of the material to apply to this primitive when rendering.
							std::optional<uint32_t> mode;										//!< The type of primitives to render.
							std::optional<std::unordered_map<std::string, uint32_t>> targets;	//!< An array of Morph Targets, each Morph Target is a dictionary mapping attributes (only POSITION, NORMAL, and TANGENT supported) to their deviations in the Morph Target.

							/*
								An accessor provides a typed view into a bufferView or a subset of a bufferView similar to
								how WebGL's vertexAttribPointer() defines an attribute in a buffer.
							*/

							struct SGLTFAccessor
							{
								std::optional<uint32_t> bufferView;
								std::optional<uint32_t> byteOffset;
								std::optional<uint32_t> componentType;
								std::optional<bool> normalized;
								std::optional<uint32_t> count;
								std::optional<std::string> type;
								std::optional<std::vector<double>> max; // todo - common number types
								std::optional<std::vector<double>> min; // todo - common number types
								// TODO - sparse;
								std::optional<std::string> name;

								enum SCompomentType
								{
									SCT_BYTE = 5120,
									SCT_UNSIGNED_BYTE = 5121,
									SCT_SHORT = 5122,
									SCT_UNSIGNED_SHORT = 5123,
									SCT_UNSIGNED_INT = 5125,
									SCT_FLOAT = 5126
								};

								struct SType
								{
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view SCALAR = "SCALAR";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view VEC2 = "VEC2";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view VEC3 = "VEC3";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view VEC4 = "VEC4";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view MAT2 = "MAT2";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view MAT3 = "MAT3";
									_IRR_STATIC_INLINE_CONSTEXPR std::string_view MAT4 = "MAT4";
								};

								bool validate()
								{
									if (!componentType.has_value())
										return false;

									if (!count.has_value())
										return false;
									else
										if (count.has_value() < 1)
											return false;

									if (!type.has_value())
										return false;

									return true;
								}
							};

							/*
								Valid attribute semantic property names include POSITION, NORMAL, TANGENT, TEXCOORD_0, TEXCOORD_1, COLOR_0, JOINTS_0, and WEIGHTS_0.
								Application-specific semantics must start with an underscore, e.g., _TEMPERATURE. TEXCOORD, COLOR, JOINTS, and WEIGHTS attribute
								semantic property names must be of the form [semantic]_[set_index], e.g., TEXCOORD_0, TEXCOORD_1, COLOR_0.
							*/

							std::unordered_map<std::string, SGLTFAccessor> accessors; //! An accessor is queried with an atribute name (like POSITION)

							enum SGLTFPrimitiveTopology
							{
								SGLTFPT_POINTS,
								SGLTFPT_LINES,
								SGLTFPT_LINE_LOOP,
								SGLTFPT_LINE_STRIP,
								SGLTFPT_TRIANGLES,
								SGLTFPT_TRIANGLE_STRIP,
								SGLTFPT_TRIANGLE_FAN
							};

							bool validate()
							{
								for(auto& [attribute, accessor] : accessors)
									if (!accessor.validate())
										return false;

								return true;
							}
						};

						std::vector<SPrimitive> primitives;					//!< An array of primitives, each defining geometry to be rendered with a material.
						std::optional<std::vector<uint32_t>> weights;		//!< Array of weights to be applied to the Morph Targets.
						std::optional<std::string> name;					//!< The user-defined name of this object.

						bool validate()
						{
							for (auto& primitive : primitives)
								if (!primitive.validate())
									return false;

							return true;
						}
					};

					SGLTFMesh glTFMesh;

					bool validate(bool validateEntireTree = false)
					{
						if (mesh.has_value() && mesh.value() < 0)
							return false;

						if (camera.has_value() && camera.value() < 0)
							return false;

						if (skin.has_value() && skin.value() < 0)
							return false;

						if (validateEntireTree)
							return glTFMesh.validate();
						else
							return true;
					}
				};

				struct SGLTF
				{
					std::vector<SGLTFNode> nodes;

					struct SGLTFBuffer
					{
						std::optional<std::string> uri;
						std::optional<uint32_t> byteLength;
						std::optional<std::string> name;

						bool validate()
						{
							if (!uri.has_value())
								return false;

							if (!byteLength.has_value())
								return false;
							else
								if (byteLength.value() < 1)
									return false;

							return true;
						}
					};

					struct SGLTFBufferView
					{
						std::optional<uint32_t> buffer;
						std::optional<uint32_t> byteOffset;
						std::optional<uint32_t> byteLength;
						std::optional<uint32_t> byteStride;
						std::optional<uint32_t> target;
						std::optional<std::string> name;

						enum SGLTFTarget
						{
							SGLTFT_ARRAY_BUFFER = 34962,
							SGLTFT_ELEMENT_ARRAY_BUFFER = 34963
						};

						bool validate()
						{
							if (!buffer.has_value())
								return false;

							if (!byteLength.has_value())
								return false;

							return true;
						}
					};

					struct SGLTFImage
					{
						std::optional<std::string> uri;
						std::optional<std::string> mimeType;
						std::optional<uint32_t> bufferView;
						std::optional<std::string> name;

						struct SMIMEType
						{
							_IRR_STATIC_INLINE_CONSTEXPR std::string_view JPEG = "image/jpeg";
							_IRR_STATIC_INLINE_CONSTEXPR std::string_view PNG = "image/png";
						};

						bool validate()
						{
							if (!uri.has_value() || (mimeType.has_value() && !bufferView.has_value()))
								return false;

							return true;
						}
					};

					struct SGLTFSampler
					{
						std::optional<uint32_t> magFilter;
						std::optional<uint32_t> minFilter;
						std::optional<uint32_t> wrapS;
						std::optional<uint32_t> wrapT;
						std::optional<std::string> name;

						enum STextureParameter
						{
							STP_NEAREST = 9728,
							STP_LINEAR = 9729,
							STP_NEAREST_MIPMAP_NEAREST = 9984,
							STP_LINEAR_MIPMAP_NEAREST = 9985,
							STP_NEAREST_MIPMAP_LINEAR = 9986,
							STP_LINEAR_MIPMAP_LINEAR = 9987,
							STP_CLAMP_TO_EDGE = 33071,
							STP_MIRRORED_REPEAT = 33648,
							STP_REPEAT = 33648
						};
					};

					struct SGLTFMaterial
					{
						std::optional<uint32_t> wrapT;

						struct SPBRMetalicRoughness
						{
							// TODO
						};

						struct SNormalTexture
						{
							// TODO
						};

						struct SOcclusionTexture
						{
							// TODO
						};

						struct SEmissiveTexture
						{

						};

						struct SAlphaMode
						{
							_IRR_STATIC_INLINE_CONSTEXPR std::string_view OPAQUE_MODE = "OPAQUE";
							_IRR_STATIC_INLINE_CONSTEXPR std::string_view MASK_MODE = "MASK";
							_IRR_STATIC_INLINE_CONSTEXPR std::string_view BLEND_MODE = "BLEND";
						};

						std::optional<std::string> name;
						std::optional<SPBRMetalicRoughness> pbrMetallicRoughness;
						std::optional<SNormalTexture> normalTexture;
						std::optional<SOcclusionTexture> occlusionTexture;
						std::optional<SEmissiveTexture> emissiveTexture;
						std::optional<std::array<double, 3>> emissiveFactor;
						std::optional<std::string> alphaMode;
						std::optional<double> alphaCutoff;
						std::optional<bool> doubleSided;
					};

					/*
						Various resources referenced by accessors, etc and may be the same,
						not unique, so holding it bellow is necessary
					*/
					
					std::vector<SGLTFBufferView> bufferViews; 
					std::vector<SGLTFBuffer> buffers;
					std::vector<SGLTFImage> images;
					std::vector<SGLTFSampler> samplers;
					std::vector<SGLTFMaterial> materials;
				};

				void loadAndGetGLTF(SGLTF& glTF, io::IReadFile* _file);
				core::smart_refctd_ptr<ICPUPipelineLayout> makePipelineLayoutFromGLTF(const bool isDS3Available);

				asset::IAssetManager* const assetManager;
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
