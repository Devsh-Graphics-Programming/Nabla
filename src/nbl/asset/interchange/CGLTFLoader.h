// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in Nabla.h

#ifndef __NBL_ASSET_C_MESH_LOADER_GLTF__
#define __NBL_ASSET_C_MESH_LOADER_GLTF__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_GLTF_LOADER_

#include "nbl/core/declarations.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IRenderpassIndependentPipelineLoader.h"
#include "nbl/asset/metadata/CGLTFMetadata.h"

namespace nbl::asset
{

//! glTF Loader capable of loading .gltf files
/*
	glTF bridges the gap between 3D content creation tools and modern 3D applications 
	by providing an efficient, extensible, interoperable format for the transmission and loading of 3D content.
*/	
class CGLTFLoader final : public IRenderpassIndependentPipelineLoader
{
	public:
		CGLTFLoader(asset::IAssetManager* _m_assetMgr);

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "gltf", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

		// TODO: THIS IS WRONG
		static inline std::string getImageViewCacheKey(const std::string& uri)
		{
			return "nbl/builtin/image_views/loaders/glTF/" + uri;
		}

	protected:
		virtual ~CGLTFLoader() {}

		struct SContext
		{
			SContext(const SAssetLoadParams& _params, system::IFile* _mainFile, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel) : loadContext(_params, _mainFile), loaderOverride(_override), hierarchyLevel(_hierarchyLevel) {}
			~SContext() {}

			SAssetLoadContext loadContext;
			asset::IAssetLoader::IAssetLoaderOverride* loaderOverride;
			uint32_t hierarchyLevel;
		};

	private:
		virtual void initialize() override;
		

		using VertexShaderUVCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/uv.vert");
		using VertexShaderColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/color.vert");
		using VertexShaderNoUVColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/no_uv_color.vert");

		using VertexShaderSkinnedUVCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/uv.vert");
		using VertexShaderSkinnedColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/color.vert");
		using VertexShaderSkinnedNoUVColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/no_uv_color.vert");

		using FragmentShaderUVCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/uv.frag");
		using FragmentShaderColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/color.frag");
		using FragmentShaderNoUVColorCacheKey = NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/shader/loader/gltf/no_uv_color.frag");

		
		static inline constexpr const char* DescriptorSetLayoutCacheKey = "nbl/builtin/descriptor_set_layout/loaders/glTF/3";
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> getDescriptorSetLayout(const SContext& context) const
		{
			return context.loaderOverride->findDefaultAsset<ICPUDescriptorSetLayout>(DescriptorSetLayoutCacheKey,context.loadContext,0u).first; // cached builtins have level 0
		}

		static inline std::string getPipelineLayoutCacheKey(const bool skinned)
		{
			if (skinned)
				return "nbl/builtin/pipeline_layout/loaders/glTF/skinned";
			else
				return "nbl/builtin/pipeline_layout/loaders/glTF/static";
		}
		core::smart_refctd_ptr<ICPUPipelineLayout> getPipelineLayout(const SContext& context, const bool skinned) const
		{
			const std::string layoutCacheKey = getPipelineLayoutCacheKey(skinned);
			auto layout = context.loaderOverride->findDefaultAsset<ICPUPipelineLayout>(layoutCacheKey,context.loadContext,0u).first; // cached builtins have level 0
			if (layout)
				return layout;

			//! camera UBO DS
			auto cpuDs1Layout = context.loaderOverride->findDefaultAsset<ICPUDescriptorSetLayout>("nbl/builtin/descriptor_set_layout/basic_view_parameters",context.loadContext,0u).first;

			asset::SPushConstantRange pushConstantRange = {asset::IShader::ESS_FRAGMENT,0u,sizeof(CGLTFPipelineMetadata::SGLTFMaterialParameters)};
			layout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(&pushConstantRange,&pushConstantRange+1u,nullptr,std::move(cpuDs1Layout),nullptr,getDescriptorSetLayout(context));

			SAssetBundle bundle(nullptr,{layout}); // insert as immutable ?
			context.loaderOverride->insertAssetIntoCache(bundle,layoutCacheKey,context.loadContext,0u); // cached builtins have level 0

			return layout;
		}

		static inline std::string getPipelineCacheKey(const E_PRIMITIVE_TOPOLOGY& primitiveType, const SVertexInputParams& vertexInputParams, const bool skinned)
		{
			if (skinned)
				return "nbl/builtin/pipeline/loaders/glTF/skinned/" + std::to_string(primitiveType) + vertexInputParams.to_string();
			else
				return "nbl/builtin/pipeline/loaders/glTF/static/" + std::to_string(primitiveType) + vertexInputParams.to_string();
		}
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> getPipeline(
			const SContext& context, const E_PRIMITIVE_TOPOLOGY& primitiveType, const SVertexInputParams& vertexInputParams, const bool skinned, const bool hasUV, const bool hasColor
		) const
		{
			const std::string pipelineCacheKey = getPipelineCacheKey(primitiveType,vertexInputParams,skinned);
			auto pipeline = context.loaderOverride->findDefaultAsset<ICPURenderpassIndependentPipeline>(pipelineCacheKey,context.loadContext,0u).first; // cached builtins have level 0
			if (pipeline)
				return pipeline;

			//auto cpuPipelineLayout = makePipelineLayoutFromGLTF(context, pushConstants, materialDependencyData, skinningEnabled);

			SBlendParams blendParams = {};
			SPrimitiveAssemblyParams primitiveAssemblyParams = {};
			primitiveAssemblyParams.primitiveType = primitiveType;
			SRasterizationParams rastarizationParams = {};
			core::smart_refctd_ptr<ICPUSpecializedShader> shaders[2];
			if (skinned)
			{
				if (hasUV) // if both UV and Color defined - we use the UV
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderUVCacheKey::value,context.loadContext,0u).first;
				else if (hasColor)
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderColorCacheKey::value,context.loadContext,0u).first;
				else
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderNoUVColorCacheKey::value,context.loadContext,0u).first;
			}
			else
			{
				if (hasUV) // if both UV and Color defined - we use the UV
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderSkinnedUVCacheKey::value,context.loadContext,0u).first;
				else if (hasColor)
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderSkinnedColorCacheKey::value,context.loadContext,0u).first;
				else
					shaders[0] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(VertexShaderSkinnedNoUVColorCacheKey::value,context.loadContext,0u).first;
			}
			if (hasUV) // if both UV and Color defined - we use the UV
				shaders[1] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(FragmentShaderUVCacheKey::value,context.loadContext,0u).first;
			else if (hasColor)
				shaders[1] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(FragmentShaderColorCacheKey::value,context.loadContext,0u).first;
			else
				shaders[1] = context.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(FragmentShaderNoUVColorCacheKey::value,context.loadContext,0u).first;
			pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(getPipelineLayout(context,skinned),&shaders->get(),&shaders->get()+2u,vertexInputParams,blendParams,primitiveAssemblyParams,rastarizationParams);

			auto meta = core::make_smart_refctd_ptr<CGLTFMetadata>(1u);
			meta->placeMeta(0u,pipeline.get(),{core::smart_refctd_ptr(m_basicViewParamsSemantics)});

			SAssetBundle bundle(std::move(meta),{pipeline}); // insert as immutable ?
			context.loaderOverride->insertAssetIntoCache(bundle,pipelineCacheKey,context.loadContext,0u); // cached builtins have level 0

			return pipeline;
		}

		struct CGLTFHeader
		{
			uint32_t version;
			std::optional<uint32_t> minVersion;
			std::optional<std::string> generator;
			std::optional<std::string> copyright;
		};

		struct SGLTF
		{
			/*
				A set of primitives to be rendered.
			*/

			struct SGLTFMesh
			{
				struct SPrimitive
				{
					std::optional<uint32_t> indices;									//!< The index of the accessor that contains the indices.
					std::optional<uint32_t> material;									//!< The index of the material to apply to this primitive when rendering.
					std::optional<uint32_t> mode;										//!< The type of primitives to render.
					std::optional<std::unordered_map<std::string, uint32_t>> targets;	//!< An array of Morph Targets, each Morph Target is a dictionary mapping attributes (only POSITION, NORMAL, and TANGENT supported) to their deviations in the Morph Target.

					struct Attributes
					{
						_NBL_STATIC_INLINE_CONSTEXPR uint16_t MAX_JOINTS_ATTRIBUTES = 16u;
						_NBL_STATIC_INLINE_CONSTEXPR uint16_t MAX_WEIGHTS_ATTRIBUTES = 16u;

						std::optional<size_t> position;										//!< The index of the accessor that contains the position.
						std::optional<size_t> normal;										//!< The index of the accessor that contains the normal.
						std::optional<size_t> tangent;										//!< The index of the accessor that contains the tangent.
						std::optional<size_t> texcoord;										//!< The index of the accessor that contains the UV.
						std::optional<size_t> color;										//!< The index of the accessor that contains the color.
						std::array<std::optional<size_t>, MAX_JOINTS_ATTRIBUTES> joints;	//!< The indices of the accessors containing the joints
						std::array<std::optional<size_t>, MAX_WEIGHTS_ATTRIBUTES> weights;	//!< The indices of the accessors containing the the weights.
					};

					Attributes attributes;

					/*struct AccessorHash
					{
						template <class T1, class T2>
						std::size_t operator () (const std::pair<T1, T2>& p) const
						{
							auto h1 = std::hash<T1>{}(p.first);
							auto h2 = std::hash<T2>{}(p.second);

							return h1 ^ h2;
						}
					};*/

					/*
						Valid attribute semantic property names include POSITION, NORMAL, TANGENT, TEXCOORD_0, TEXCOORD_1, COLOR_0, JOINTS_0, and WEIGHTS_0.
						Application-specific semantics must start with an underscore, e.g., _TEMPERATURE. TEXCOORD, COLOR, JOINTS, and WEIGHTS attribute
						semantic property names must be of the form [semantic]_[set_index], e.g., TEXCOORD_0, TEXCOORD_1, COLOR_0.
					*/

					//std::unordered_map<std::pair<SGLTFAttribute, uint8_t>, SGLTFAccessor, AccessorHash> accessors; //! An accessor is queried with an atribute name (like POSITION)

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
						/*for (auto& [attribute, accessor] : accessors)
							if (!accessor.validate())
								return false;*/

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

			struct SGLTFScene
			{
				std::vector<uint32_t> nodes;
			};

			/*
				A node in the node hierarchy. A node can contain one mesh.
				A node's transform places the mesh in the scene. A node can have either 
				a matrix or any combination of translation/rotation/scale (TRS) properties.
			*/

			struct SGLTFNode
			{
				SGLTFNode() {}
				virtual ~SGLTFNode() {}

				std::optional<std::string> name;				//!< The user-defined name of this object.
				std::vector<uint32_t> children;					//!< The indices of this node's children.
				std::optional<uint32_t> mesh;					//!< The index of the mesh in this node.
				std::optional<uint32_t> camera;					//!< The index of the camera referenced by this node.
				std::optional<uint32_t> skin;					//!< The index of the skin referenced by this node.
				std::optional<uint32_t> weights;				//!< The weights of the instantiated Morph Target. Number of elements must match number of Morph Targets of used mesh.

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

					core::matrix3x4SIMD matrix;
				} transformation;

				bool validate(bool validateEntireTree = false)
				{
					if (mesh.has_value() && mesh.value() < 0)
						return false;

					if (camera.has_value() && camera.value() < 0)
						return false;

					if (skin.has_value() && skin.value() < 0)
						return false;

					return true;
				}
			};

			/*
				An accessor provides a typed view into a bufferView or a subset of a bufferView similar to
				how WebGL's vertexAttribPointer() defines an attribute in a buffer.
			*/

			struct SGLTFAccessor
			{
				enum SGLTFType
				{
					SGLTFT_SCALAR,
					SGLTFT_VEC2,
					SGLTFT_VEC3,
					SGLTFT_VEC4,
					SGLTFT_MAT2,
					SGLTFT_MAT3,
					SGLTFT_MAT4
				};
				enum SCompomentType
				{
					SCT_BYTE = 5120,
					SCT_UNSIGNED_BYTE = 5121,
					SCT_SHORT = 5122,
					SCT_UNSIGNED_SHORT = 5123,
					SCT_UNSIGNED_INT = 5125,
					SCT_FLOAT = 5126
				};

				std::optional<uint32_t> bufferView;
				std::optional<size_t> byteOffset;
				std::optional<SCompomentType> componentType;
				std::optional<bool> normalized;
				std::optional<uint32_t> count;
				std::optional<SGLTFType> type;
				std::optional<std::vector<double>> max; // todo - common number types
				std::optional<std::vector<double>> min; // todo - common number types
				// TODO - sparse;
				std::optional<std::string> name;

				struct SType
				{
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view SCALAR = "SCALAR";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view VEC2 = "VEC2";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view VEC3 = "VEC3";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view VEC4 = "VEC4";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view MAT2 = "MAT2";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view MAT3 = "MAT3";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view MAT4 = "MAT4";
				};

				bool validate()
				{
					if (!componentType.has_value())
						return false;

					if (!count.has_value())
						return false;
					else if (count.value() < 1)
							return false;

					if (!type.has_value())
						return false;

					return true;
				}

				static inline E_FORMAT getFormat(SCompomentType componentType, SGLTFType type)
				{
					switch (componentType)
					{
						case SGLTF::SGLTFAccessor::SCT_BYTE:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R8_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R8G8_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R8G8B8_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R8G8B8A8_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R8G8B8A8_SINT;
						} break;

						case SGLTF::SGLTFAccessor::SCT_FLOAT:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R32_SFLOAT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R32G32_SFLOAT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R32G32B32_SFLOAT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R32G32B32A32_SFLOAT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R32G32B32A32_SFLOAT;
						} break;

						case SGLTF::SGLTFAccessor::SCT_SHORT:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R16_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R16G16_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R16G16B16_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R16G16B16A16_SINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R16G16B16A16_SINT;
						} break;

						case SGLTF::SGLTFAccessor::SCT_UNSIGNED_BYTE:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R8_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R8G8_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R8G8B8_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R8G8B8A8_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R8G8B8A8_UINT;
						} break;

						case SGLTF::SGLTFAccessor::SCT_UNSIGNED_INT:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R32_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R32G32_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R32G32B32_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R32G32B32A32_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R32G32B32A32_UINT;
						} break;

						case SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT:
						{
							if (type == SGLTF::SGLTFAccessor::SGLTFT_SCALAR)
								return EF_R16_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC2)
								return EF_R16G16_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC3)
								return EF_R16G16B16_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								return EF_R16G16B16A16_UINT;
							else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT2)
								return EF_R16G16B16A16_UINT;
						} break;
					}
					return EF_UNKNOWN;
				}
			};

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
				std::optional<size_t> byteOffset;
				std::optional<size_t> byteLength;
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
				std::optional<size_t> bufferView;
				std::optional<std::string> name;

				struct SMIMEType
				{
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view JPEG = "image/jpeg";
					_NBL_STATIC_INLINE_CONSTEXPR std::string_view PNG = "image/png";
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
					STP_REPEAT = 10497
				};

				uint32_t magFilter = STP_NEAREST;
				uint32_t minFilter = STP_NEAREST;
				uint32_t wrapS = STP_REPEAT;
				uint32_t wrapT = STP_REPEAT;
				std::optional<std::string> name;
			};

			struct SGLTFTexture
			{
				/*
					The index of the sampler used by this texture. When undefined, 
					a sampler with repeat wrapping and auto filtering should be used.
				*/

				std::optional<uint32_t> sampler;

				/*
					The index of the image used by this texture. When undefined, it is expected that an extension or 
					other mechanism will supply an alternate texture source, otherwise behavior is undefined.
				*/

				std::optional<uint32_t> source;
				std::optional<std::string> name;
			};

			struct SGLTFMaterial
			{
				enum E_GLTF_TEXTURES
				{
					EGT_BASE_COLOR_TEXTURE,
					EGT_METALLIC_ROUGHNESS_TEXTURE,
					EGT_NORMAL_TEXTURE,
					EGT_OCCLUSION_TEXTURE,
					EGT_EMISSIVE_TEXTURE,
					EGT_COUNT,
				};
						
				/*
					Basic reference to a texture.
				*/

				struct STextureInfo
				{
					std::optional<uint32_t> index; //!< The index of the texture.

					/*
						This integer value is used to construct a string in the format TEXCOORD_<set index> which is a reference to a key in mesh.primitives.attributes (e.g. A value of 0 corresponds to TEXCOORD_0).
						Mesh must have corresponding texture coordinate attributes for the material to be applicable to it.
					*/

					std::optional<uint32_t> texCoord;    
				};

				/*
					A set of parameter values that are used to define the metallic-roughness material model from Physically-Based Rendering (PBR) methodology. 
					When not specified, all the default values of pbrMetallicRoughness apply.
				*/

				struct SPBRMetalicRoughness
				{
					using SBaseColorTexture = STextureInfo;
					using SMetallicRoughnessTexture = STextureInfo;

					struct SMetallic
					{
						_NBL_STATIC_INLINE_CONSTEXPR double METAL = 1.0;
						_NBL_STATIC_INLINE_CONSTEXPR double DIELECTRIC = 0.0;
					};

					struct SRoughness
					{
						_NBL_STATIC_INLINE_CONSTEXPR double ROUGH = 1.0;
						_NBL_STATIC_INLINE_CONSTEXPR double SMOOTH = 0.0;
					};

					/*
						The RGBA components of the base color of the material. The fourth component (A) is the alpha coverage of the material. 
						The alphaMode property specifies how alpha is interpreted. These values are linear. If a baseColorTexture is specified, this value is multiplied with the texel values.
					*/

					std::optional<std::array<double, 4>> baseColorFactor;

					/*
						The base color texture. The first three components (RGB) are encoded with the sRGB transfer function. They specify the base color of the material. If the fourth component (A) is present, it represents the linear alpha coverage of the material.
						Otherwise, an alpha of 1.0 is assumed. The alphaMode property specifies how alpha is interpreted. The stored texels must not be premultiplied.
					*/

					std::optional<SBaseColorTexture> baseColorTexture;

					/*
						The metalness of the material. A value of 1.0 means the material is a metal. A value of 0.0 means the material is a dielectric. Values in between are for blending between metals and dielectrics such as dirty metallic surfaces.
						This value is linear. If a metallicRoughnessTexture is specified, this value is multiplied with the metallic texel values.
					*/

					std::optional<double> metallicFactor;

					/*
						The roughness of the material. A value of 1.0 means the material is completely rough. A value of 0.0 means the material is completely smooth. 
						This value is linear. If a metallicRoughnessTexture is specified, this value is multiplied with the roughness texel values.
					*/

					std::optional<double> roughnessFactor;

					/*
						The metallic-roughness texture. The metalness values are sampled from the B channel. The roughness values are sampled from the G channel. 
						These values are linear. If other channels are present (R or A), they are ignored for metallic-roughness calculations.
					*/

					std::optional<SMetallicRoughnessTexture> metallicRoughnessTexture;
				};

				struct SNormalTexture : public STextureInfo
				{
					std::optional<double> scale; //! The scalar multiplier applied to each normal vector of the normal texture.
				};

				struct SOcclusionTexture : public STextureInfo
				{
					std::optional<double> strength; //! A scalar multiplier controlling the amount of occlusion applied.
				};

				struct SEmissiveTexture : public STextureInfo
				{
							
				};

				enum E_ALPHA_MODE : uint32_t
				{
					EAM_OPAQUE = core::createBitmask({ 0 }),
					EAM_MASK = core::createBitmask({ 1 }),
					EAM_BLEND = core::createBitmask({ 2 })
				};

				std::optional<std::string> name;
				std::optional<SPBRMetalicRoughness> pbrMetallicRoughness;
				std::optional<SNormalTexture> normalTexture;
				std::optional<SOcclusionTexture> occlusionTexture;
				std::optional<SEmissiveTexture> emissiveTexture;
				std::optional<std::array<double, 3>> emissiveFactor;
				std::optional<E_ALPHA_MODE> alphaMode;
				std::optional<double> alphaCutoff;
				std::optional<bool> doubleSided;
			};
					
			struct SGLTFAnimation
			{
				struct SGLTFChannel
				{
					enum SGLTFPath
					{
						SGLTFP_TRANSLATION,
						SGLTFP_ROTATION,
						SGLTFP_SCALE,
						SGLTFP_WEIGHTS
					};

					struct SGLTFTarget
					{
						std::optional<size_t> node;					//!< The index of the node to target.
						SGLTFPath path;								//!< The name of the node's TRS property to modify, or the "weights" of the Morph Targets it instantiates. For the "translation" property, the values that are provided by the sampler are the translation along the x, y, and z axes. For the "rotation" property, the values are a quaternion in the order (x, y, z, w), where w is the scalar. For the "scale" property, the values are the scaling factors along the x, y, and z axes.
						std::optional<std::string> extensions;	
						std::optional<std::string> extras;
					};

					SGLTFTarget target;
					size_t sampler;									//!< Summarizes the actual animation data 
				};

				struct SGLTFSampler
				{
					enum SGLTFInterpolation
					{
						/*
							The animated values are linearly interpolated between keyframes.
							When targeting a rotation, spherical linear interpolation (slerp) should be used 
							to interpolate quaternions. The number output of elements must equal the number of input elements.
						*/

						SGLTFI_LINEAR,

						/*
							The animated values remain constant to the output of the first keyframe, 
							until the next keyframe. The number of output elements must equal the number of input elements.
						*/
								
						SGLTFI_STEP,

						/*
							The animation's interpolation is computed using a cubic spline with specified tangents.
							The number of output elements must equal three times the number of input elements. 
							For each input element, the output stores three elements, an in-tangent, a spline vertex,
							and an out-tangent. There must be at least two keyframes when using this interpolation.
						*/

						SGLTFI_CUBICSPLINE
					};

					size_t input;												//!< The index of an accessor containing keyframe input values, e.g., time.	
					SGLTFInterpolation interpolation = SGLTFI_LINEAR;			//!< Interpolation algorithm.
					size_t output;												//!< The index of an accessor, containing keyframe output values.
					std::optional<std::string> extensions;
				};

				std::optional<std::string> name;
				std::vector<SGLTFChannel> channels;
				std::vector<SGLTFSampler> samplers;
				std::optional<std::string> extensions;
				std::optional<std::string> extras;
			};

			struct SGLTFSkin
			{
				_NBL_STATIC_INLINE_CONSTEXPR uint16_t MAX_JOINTS_REFERENCES = 256; // TODO: can we up this to 1024 or 16k ?

				std::optional<std::string> name;
				std::optional<size_t> inverseBindMatrices;						//! The index of the accessor containing the floating-point 4x4 inverse-bind matrices. The default is that each matrix is a 4x4 identity matrix, which implies that inverse-bind matrices were pre-applied.
				std::optional<size_t> skeleton;									//! The index of the node used as a skeleton root.
				std::vector<uint32_t> joints;									//! Indices of skeleton nodes, used as joints in this skin.
				std::optional<std::string> extensions;
				std::optional<std::string> extras;
			};

			/*
				Various resources established by using their 
				indices to look up the objects in arrays 
			*/
					
			std::optional<uint32_t> defaultScene;
			std::vector<SGLTFScene> scenes;
			std::vector<SGLTFMesh> meshes;
			std::vector<SGLTFNode> nodes;
			std::vector<SGLTFAccessor> accessors;
			std::vector<SGLTFBufferView> bufferViews; 
			std::vector<SGLTFBuffer> buffers;
			std::vector<SGLTFImage> images;
			std::vector<SGLTFSampler> samplers;
			std::vector<SGLTFTexture> textures;
			std::vector<SGLTFMaterial> materials;
			std::vector<SGLTFSkin> skins;
			std::vector<SGLTFAnimation> animations;
		};

		bool loadAndGetGLTF(SGLTF& glTF, SContext& context);

		asset::IAssetManager* const assetManager;
};

}

#endif // _NBL_COMPILE_WITH_GLTF_LOADER_
#endif // __NBL_ASSET_C_MESH_LOADER_GLTF__
