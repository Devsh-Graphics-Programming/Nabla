// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in Nabla.h

#include "CGLTFLoader.h"

#ifdef _NBL_COMPILE_WITH_GLTF_LOADER_

#include "nbl/asset/utils/CDerivativeMapCreator.h"
#include "nbl/asset/metadata/CGLTFMetadata.h"
#include "simdjson/singleheader/simdjson.h"
#include <algorithm>

#define VERT_SHADER_UV_CACHE_KEY "nbl/builtin/shader/loader/gltf/vertex_uv.vert"
#define VERT_SHADER_COLOR_CACHE_KEY "nbl/builtin/shader/loader/gltf/vertex_color.vert"
#define VERT_SHADER_NO_UV_COLOR_CACHE_KEY "nbl/builtin/shader/loader/gltf/vertex_no_uv_color.vert"

#define FRAG_SHADER_UV_CACHE_KEY "nbl/builtin/shader/loader/gltf/fragment_uv.frag"
#define FRAG_SHADER_COLOR_CACHE_KEY "nbl/builtin/shader/loader/gltf/fragment_color.frag"
#define FRAG_SHADER_NO_UV_COLOR_CACHE_KEY "nbl/builtin/shader/loader/gltf/fragment_no_uv_color.frag"

namespace nbl
{
	namespace asset
	{
		enum WEIGHT_ENCODING
		{
			WE_UNORM8,
			WE_UNORM16,
			WE_SFLOAT,
			WE_COUNT
		};

		template<typename AssetType, IAsset::E_TYPE assetType>
		static core::smart_refctd_ptr<AssetType> getDefaultAsset(const char* _key, IAssetManager* _assetMgr)
		{
			size_t storageSz = 1ull;
			asset::SAssetBundle bundle;
			const IAsset::E_TYPE types[]{ assetType, static_cast<IAsset::E_TYPE>(0u) };

			_assetMgr->findAssets(storageSz, &bundle, _key, types);
			if (bundle.getContents().empty())
				return nullptr;
			auto assets = bundle.getContents();
			//assert(assets.first != assets.second);

			return core::smart_refctd_ptr_static_cast<AssetType>(assets.begin()[0]);
		}

		namespace SAttributes
		{
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t POSITION_ATTRIBUTE_LAYOUT_ID = 0;
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t UV_ATTRIBUTE_LAYOUT_ID = 1;
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t COLOR_ATTRIBUTE_LAYOUT_ID = 2;	
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t NORMAL_ATTRIBUTE_LAYOUT_ID = 3;
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t JOINTS_ATTRIBUTE_LAYOUT_ID = 4;	
			_NBL_STATIC_INLINE_CONSTEXPR uint8_t WEIGHTS_ATTRIBUTE_LAYOUT_ID = 5;
		}

		/*
			Each glTF asset must have an asset property. 
			In fact, it's the only required top-level property
			for JSON to be a valid glTF.
		*/

		CGLTFLoader::CGLTFLoader(asset::IAssetManager* _m_assetMgr) 
			: IRenderpassIndependentPipelineLoader(_m_assetMgr), assetManager(_m_assetMgr)
		{
			auto registerShader = [&](auto constexprStringType, ICPUSpecializedShader::E_SHADER_STAGE stage) -> void
			{
				auto shaderData = assetManager->getSystem()->loadBuiltinData<decltype(constexprStringType)>();
				auto unspecializedShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shaderData), asset::ICPUShader::buffer_contains_glsl);

				ICPUSpecializedShader::SInfo specInfo({}, nullptr, "main", stage, stage != ICPUSpecializedShader::ESS_VERTEX ? "?IrrlichtBAW glTFLoader FragmentShader?" : "?IrrlichtBAW glTFLoader VertexShader?");
				auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedShader), std::move(specInfo));

				auto insertShaderIntoCache = [&](const char* path)
				{
					asset::SAssetBundle bundle(nullptr, { cpuShader });
					assetManager->changeAssetKey(bundle, path);
					assetManager->insertAssetIntoCache(bundle);
				};

				insertShaderIntoCache(decltype(constexprStringType)::value);
			};

			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_UV_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);
			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);
			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_NO_UV_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);

			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_UV_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
			registerShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_NO_UV_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
		}

		void CGLTFLoader::initialize()
		{
			IRenderpassIndependentPipelineLoader::initialize();
		}
		
		bool CGLTFLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
		{
			/*
				TODO: https://github.com/Devsh-Graphics-Programming/Nabla/pull/196#issuecomment-906426010
			*/

			#define NBL_COMPILE_WITH_SYSTEM_BUG // remove this after above fixed

			#ifdef NBL_COMPILE_WITH_SYSTEM_BUG
			if (_file->getFileName().string() == "missing_checkerboard_texture.png")
				return false;
			#endif // NBL_COMPILE_WITH_SYSTEM_BUG

			simdjson::dom::parser parser;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				system::future<size_t> future;
				_file->read(future, jsonBuffer->getPointer(), 0u, jsonBuffer->getSize());
				future.get();
			}
			simdjson::dom::object tweets = parser.parse(reinterpret_cast<uint8_t*>(jsonBuffer->getPointer()), jsonBuffer->getSize());
			simdjson::dom::element element;

			if (tweets.at_key("asset").get(element) == simdjson::error_code::SUCCESS)
				if (element.at_key("version").get(element) == simdjson::error_code::SUCCESS)
					return true;

			return false;
		}

		asset::SAssetBundle CGLTFLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			auto overrideAssetLoadParams = _params;

			/*
				TODO: https://github.com/Devsh-Graphics-Programming/Nabla/pull/196#issuecomment-906469117
				it doesn't work
			*/

			const std::string relativeDirectory = _file->getFileName().parent_path().string() + "/";
			//overrideAssetLoadParams.relativeDir = relativeDirectory.c_str();
			SContext context(overrideAssetLoadParams, _file, _override, _hierarchyLevel);

			SGLTF glTF;
			if(!loadAndGetGLTF(glTF, context))
				return {};

			auto getURIAbsolutePath = [&](std::string uri) -> std::string
			{
				return relativeDirectory + uri;
			};

			// TODO: having validated and loaded glTF data we can use it to create pipelines and data

			core::vector<core::smart_refctd_ptr<ICPUBuffer>> cpuBuffers;
			for (auto& glTFBuffer : glTF.buffers)
			{
				auto buffer_bundle = assetManager->getAsset(getURIAbsolutePath(glTFBuffer.uri.value()), context.loadContext.params);
				if (buffer_bundle.getContents().empty())
					return {};

				auto cpuBuffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer_bundle.getContents().begin()[0]);
				cpuBuffers.emplace_back() = core::smart_refctd_ptr<ICPUBuffer>(cpuBuffer);
			}

			core::vector<core::smart_refctd_ptr<ICPUImageView>> cpuImageViews;
			{
				for (auto& glTFImage : glTF.images)
				{
					auto& cpuImageView = cpuImageViews.emplace_back();

					if (glTFImage.uri.has_value())
					{
						auto image_bundle = assetManager->getAsset(getURIAbsolutePath(glTFImage.uri.value()), context.loadContext.params);
						if (image_bundle.getContents().empty())
							return {};

						auto cpuAsset = image_bundle.getContents().begin()[0];

						switch (cpuAsset->getAssetType())
						{
							case IAsset::ET_IMAGE:
							{
								ICPUImageView::SCreationParams viewParams;
								viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
								viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(cpuAsset);
								viewParams.format = viewParams.image->getCreationParameters().format;
								viewParams.viewType = IImageView<ICPUImage>::ET_2D;
								viewParams.subresourceRange.baseArrayLayer = 0u;
								viewParams.subresourceRange.layerCount = 1u;
								viewParams.subresourceRange.baseMipLevel = 0u;
								viewParams.subresourceRange.levelCount = 1u;

								cpuImageView = ICPUImageView::create(std::move(viewParams));
							} break;

							case IAsset::ET_IMAGE_VIEW:
							{
								cpuImageView = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(cpuAsset);
							} break;

							default:
							{
								context.loadContext.params.logger.log("GLTF: EXPECTED IMAGE ASSET TYPE!", system::ILogger::ELL_WARNING);
								return {};
							}
						}
					}
					else
					{
						if (!glTFImage.mimeType.has_value() || !glTFImage.bufferView.has_value())
							return {};
						
						return {}; // TODO FUTURE: load image where it's data is embeded in memory
					}
				}
			}

			core::vector<SSamplerCacheKey> cpuSamplers;
			{
				for (auto& glTFSampler : glTF.samplers)
				{
					typedef std::remove_reference<decltype(glTFSampler)>::type SGLTFSampler;

					ICPUSampler::SParams samplerParams;

					switch (glTFSampler.magFilter)
					{
						case SGLTFSampler::STP_NEAREST:
						{
							samplerParams.MaxFilter = ISampler::ETF_NEAREST;
						} break;

						case SGLTFSampler::STP_LINEAR:
						{
							samplerParams.MaxFilter = ISampler::ETF_LINEAR;
						} break;
					}

					switch (glTFSampler.minFilter)
					{
						case SGLTFSampler::STP_NEAREST:
						{
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
						} break;

						case SGLTFSampler::STP_LINEAR:
						{
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
						} break;

						case SGLTFSampler::STP_NEAREST_MIPMAP_NEAREST:
						{
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
							samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
						} break;

						case SGLTFSampler::STP_LINEAR_MIPMAP_NEAREST:
						{
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
							samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
						} break;

						case SGLTFSampler::STP_NEAREST_MIPMAP_LINEAR:
						{
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
							samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
						} break;

						case SGLTFSampler::STP_LINEAR_MIPMAP_LINEAR:
						{
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
							samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
						} break;
					}
					
					switch (glTFSampler.wrapS)
					{
						case SGLTFSampler::STP_CLAMP_TO_EDGE:
						{
							samplerParams.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
						} break;

						case SGLTFSampler::STP_MIRRORED_REPEAT:
						{
							samplerParams.TextureWrapU = ISampler::ETC_MIRROR;
						} break;

						case SGLTFSampler::STP_REPEAT:
						{
							samplerParams.TextureWrapU = ISampler::ETC_REPEAT;
						} break;
					}

					switch (glTFSampler.wrapT)
					{
						case SGLTFSampler::STP_CLAMP_TO_EDGE:
						{
							samplerParams.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
						} break;

						case SGLTFSampler::STP_MIRRORED_REPEAT:
						{
							samplerParams.TextureWrapV = ISampler::ETC_MIRROR;
						} break;

						case SGLTFSampler::STP_REPEAT:
						{
							samplerParams.TextureWrapV = ISampler::ETC_REPEAT;
						} break;
					}

					const std::string cacheKey = getSamplerCacheKey(samplerParams);
					cpuSamplers.push_back(cacheKey);

					const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_SAMPLER, (asset::IAsset::E_TYPE)0u };
					auto sampler_bundle = _override->findCachedAsset(cacheKey, types, context.loadContext, _hierarchyLevel /*TODO + what here?*/);
					if (sampler_bundle.getContents().empty())
					{
						SAssetBundle samplerBundle = SAssetBundle(nullptr, {core::make_smart_refctd_ptr<ICPUSampler>(std::move(samplerParams))});
						_override->insertAssetIntoCache(samplerBundle, cacheKey, context.loadContext, _hierarchyLevel /*TODO + what here?*/);
					}
				}
			}

			STextures cpuTextures;
			{
				for (auto& glTFTexture : glTF.textures)
				{
					auto& [cpuImageView, samplerCacheKey] = cpuTextures.emplace_back();

					if (glTFTexture.sampler.has_value())
						samplerCacheKey = cpuSamplers[glTFTexture.sampler.value()];
					else
						samplerCacheKey = "nbl/builtin/sampler/default";

					if (glTFTexture.source.has_value())
						cpuImageView = core::smart_refctd_ptr<ICPUImageView>(cpuImageViews[glTFTexture.source.value()]);
					else
					{
						auto default_imageview_bundle = assetManager->getAsset("nbl/builtin/image_view/dummy2d", context.loadContext.params);
						const bool status = !default_imageview_bundle.getContents().empty();
						if (status)
						{
							auto cpuDummyImageView = core::smart_refctd_ptr_static_cast<ICPUImageView>(default_imageview_bundle.getContents().begin()[0]);
							cpuImageView = core::smart_refctd_ptr<ICPUImageView>(cpuDummyImageView);
						}
						else
						{
							context.loadContext.params.logger.log("GLTF: COULD NOT LOAD BUILTIN DUMMY IMAGE VIEW!", system::ILogger::ELL_WARNING);
							return {};
						}
					}
				}
			}

			std::vector<std::vector<core::smart_refctd_ptr<asset::CGLTFPipelineMetadata>>> globalMetadataContainer; // TODO: to optimize in future
			core::vector<core::smart_refctd_ptr<ICPUMesh>> cpuMeshes;
			{
				for (const auto& glTFMesh : glTF.meshes)
				{
					auto& globalPipelineMeta = globalMetadataContainer.emplace_back();
					auto& cpuMesh = cpuMeshes.emplace_back() = core::make_smart_refctd_ptr<ICPUMesh>();

					for (const auto& glTFprimitive : glTFMesh.primitives)
					{
						typedef std::remove_reference<decltype(glTFprimitive)>::type SGLTFPrimitive;

						auto cpuMeshBuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();

						cpuMeshBuffer->setPositionAttributeIx(SAttributes::POSITION_ATTRIBUTE_LAYOUT_ID);
						cpuMeshBuffer->setNormalAttributeIx(SAttributes::NORMAL_ATTRIBUTE_LAYOUT_ID);
						cpuMeshBuffer->setJointIDAttributeIx(SAttributes::JOINTS_ATTRIBUTE_LAYOUT_ID);
						cpuMeshBuffer->setJointWeightAttributeIx(SAttributes::WEIGHTS_ATTRIBUTE_LAYOUT_ID);

						auto getMode = [&](uint32_t modeValue) -> E_PRIMITIVE_TOPOLOGY
						{
							switch (modeValue)
							{
								case SGLTFPrimitive::SGLTFPT_POINTS:
									return EPT_POINT_LIST;
								case SGLTFPrimitive::SGLTFPT_LINES:
									return EPT_LINE_LIST;
								case SGLTFPrimitive::SGLTFPT_LINE_LOOP:
									return EPT_LINE_LIST_WITH_ADJACENCY; // check it
								case SGLTFPrimitive::SGLTFPT_LINE_STRIP:
									return EPT_LINE_STRIP;
								case SGLTFPrimitive::SGLTFPT_TRIANGLES:
									return EPT_TRIANGLE_LIST;
								case SGLTFPrimitive::SGLTFPT_TRIANGLE_STRIP:
									return EPT_TRIANGLE_STRIP;
								case SGLTFPrimitive::SGLTFPT_TRIANGLE_FAN:
									return EPT_TRIANGLE_STRIP_WITH_ADJACENCY; // check it
							}
						};

						auto [hasUV, hasColor] = std::make_pair<bool, bool>(false, false);

						using BufferViewReferencingBufferID = uint32_t;
						std::unordered_map<BufferViewReferencingBufferID, core::smart_refctd_ptr<ICPUBuffer>> idReferenceBindingBuffers;

						SVertexInputParams vertexInputParams;
						SBlendParams blendParams;
						SPrimitiveAssemblyParams primitiveAssemblyParams;
						SRasterizationParams rastarizationParmas;

						auto handleAccessor = [&](SGLTF::SGLTFAccessor& glTFAccessor, const std::optional<uint32_t> queryAttributeId = {}) -> bool
						{
							auto getFormat = [&](uint32_t componentType, SGLTF::SGLTFAccessor::SGLTFType type)
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
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
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT3)
											return EF_UNKNOWN; // ?
										else if (type == SGLTF::SGLTFAccessor::SGLTFT_MAT4)
											return EF_UNKNOWN; // ?
									} break;
								}
							};

							const E_FORMAT format = getFormat(glTFAccessor.componentType.value(), glTFAccessor.type.value());
							if (format == EF_UNKNOWN)
							{
								context.loadContext.params.logger.log("GLTF: COULD NOT SPECIFY NABLA FORMAT!", system::ILogger::ELL_WARNING);
								return false;
							}

							auto& glTFbufferView = glTF.bufferViews[glTFAccessor.bufferView.value()];
							const uint32_t& bufferBindingId = glTFAccessor.bufferView.value();
							const uint32_t& bufferDataId = glTFbufferView.buffer.value();
							const auto& globalOffsetInBufferBindingResource = glTFbufferView.byteOffset.has_value() ? glTFbufferView.byteOffset.value() : 0u;
							const auto& relativeOffsetInBufferViewAttribute = glTFAccessor.byteOffset.has_value() ? glTFAccessor.byteOffset.value() : 0u;

							typedef std::remove_reference<decltype(glTFbufferView)>::type SGLTFBufferView;

							auto setBufferBinding = [&](uint32_t target) -> void
							{
								asset::SBufferBinding<ICPUBuffer> bufferBinding;
								bufferBinding.offset = globalOffsetInBufferBindingResource;

								idReferenceBindingBuffers[bufferDataId] = cpuBuffers[bufferDataId];
								bufferBinding.buffer = idReferenceBindingBuffers[bufferDataId];

								auto isDataInterleaved = [&]()
								{
									return glTFbufferView.byteStride.has_value();
								};

								switch (target)
								{
									case SGLTFBufferView::SGLTFT_ARRAY_BUFFER:
									{
										cpuMeshBuffer->setVertexBufferBinding(std::move(bufferBinding), bufferBindingId);

										vertexInputParams.enabledBindingFlags |= core::createBitmask({ bufferBindingId });
										vertexInputParams.bindings[bufferBindingId].inputRate = EVIR_PER_VERTEX;
										vertexInputParams.bindings[bufferBindingId].stride = isDataInterleaved() ? glTFbufferView.byteStride.value() : getTexelOrBlockBytesize(format); // TODO: change it when handling matrices as well

										const auto attributeId = queryAttributeId.value();
										vertexInputParams.enabledAttribFlags |= core::createBitmask({ attributeId });
										vertexInputParams.attributes[attributeId].binding = bufferBindingId;
										vertexInputParams.attributes[attributeId].format = format;
										vertexInputParams.attributes[attributeId].relativeOffset = relativeOffsetInBufferViewAttribute;
									} break;

									case SGLTFBufferView::SGLTFT_ELEMENT_ARRAY_BUFFER:
									{
										// TODO: make sure glTF data has validated index type

										bufferBinding.offset += relativeOffsetInBufferViewAttribute;
										cpuMeshBuffer->setIndexBufferBinding(std::move(bufferBinding));
									} break;
								}
							};

							setBufferBinding(queryAttributeId.has_value() ? SGLTF::SGLTFBufferView::SGLTFT_ARRAY_BUFFER : SGLTF::SGLTFBufferView::SGLTFT_ELEMENT_ARRAY_BUFFER);
							return true;
						};

						const E_PRIMITIVE_TOPOLOGY primitiveTopology = getMode(glTFprimitive.mode.value());
						primitiveAssemblyParams.primitiveType = primitiveTopology;

						if (glTFprimitive.indices.has_value())
						{
							const size_t accessorID = glTFprimitive.indices.value();

							auto& glTFIndexAccessor = glTF.accessors[accessorID];
							if (!handleAccessor(glTFIndexAccessor))
								return {};

							switch (glTFIndexAccessor.componentType.value())
							{
								case SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT:
								{
									cpuMeshBuffer->setIndexType(EIT_16BIT);
								} break;

								case SGLTF::SGLTFAccessor::SCT_UNSIGNED_INT:
								{
									cpuMeshBuffer->setIndexType(EIT_32BIT);
								} break;
							}

							cpuMeshBuffer->setIndexCount(glTFIndexAccessor.count.value());
						}

						if (glTFprimitive.attributes.position.has_value())
						{
							const size_t accessorID = glTFprimitive.attributes.position.value();

							auto& glTFPositionAccessor = glTF.accessors[accessorID];
							if (!handleAccessor(glTFPositionAccessor, cpuMeshBuffer->getPositionAttributeIx()))
								return {};

							if (!glTFprimitive.indices.has_value())
								cpuMeshBuffer->setIndexCount(glTFPositionAccessor.count.value());
						}
						else
						{
							context.loadContext.params.logger.log("GLTF: COULD NOT DETECT POSITION ATTRIBUTE!", system::ILogger::ELL_WARNING);
							return false;
						}
							
						if (glTFprimitive.attributes.normal.has_value())
						{
							const size_t accessorID = glTFprimitive.attributes.normal.value();

							auto& glTFNormalAccessor = glTF.accessors[accessorID];
							if (!handleAccessor(glTFNormalAccessor, cpuMeshBuffer->getNormalAttributeIx()))
								return {};
						}

						if (glTFprimitive.attributes.texcoord.has_value())
						{
							const size_t accessorID = glTFprimitive.attributes.texcoord.value();

							hasUV = true;
							auto& glTFTexcoordXAccessor = glTF.accessors[accessorID];
							if (!handleAccessor(glTFTexcoordXAccessor, SAttributes::UV_ATTRIBUTE_LAYOUT_ID))
								return {};
						}

						if (glTFprimitive.attributes.color.has_value())
						{
							const size_t accessorID = glTFprimitive.attributes.color.value();

							hasColor = true;
							auto& glTFColorXAccessor = glTF.accessors[accessorID];
							if (!handleAccessor(glTFColorXAccessor, SAttributes::COLOR_ATTRIBUTE_LAYOUT_ID))
								return {};
						}

						struct OverrideReference
						{
							E_FORMAT format;
							SGLTF::SGLTFAccessor* accessor;
							asset::SBufferRange<asset::ICPUBuffer> bufferRange;
							void* data; //! begin data with offset according to buffer range
						};
						
						std::vector<OverrideReference> overrideJointsReference;
						std::vector<OverrideReference> overrideWeightsReference;
						
						for (uint8_t i = 0; i < glTFprimitive.attributes.joints.size(); ++i)
						{
							if (glTFprimitive.attributes.joints[i].has_value())
							{
								const size_t accessorID = glTFprimitive.attributes.joints[i].value();

								auto& glTFJointsXAccessor = glTF.accessors[accessorID];

								if (glTFJointsXAccessor.type.value() != SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								{
									context.loadContext.params.logger.log("GLTF: JOINTS ACCESSOR MUST HAVE VEC4 TYPE!", system::ILogger::ELL_WARNING);
									return {};
								}

								const asset::E_FORMAT jointsFormat = [&]()
								{
									if (glTFJointsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_BYTE)
										return EF_R8G8B8A8_UINT;
									else if (glTFJointsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT)
										return EF_R16G16B16A16_UINT;
									else
										EF_UNKNOWN;
								}();

								if (jointsFormat == EF_UNKNOWN)
								{
									context.loadContext.params.logger.log("GLTF: DETECTED JOINTS BUFFER WITH INVALID COMPONENT TYPE!", system::ILogger::ELL_WARNING);
									return {};
								}

								if (!glTFJointsXAccessor.bufferView.has_value())
								{
									context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!", system::ILogger::ELL_WARNING);
									return {};
								}

								const auto& bufferViewID = glTFJointsXAccessor.bufferView.value();
								const auto& glTFBufferView = glTF.bufferViews[bufferViewID];

								if (!glTFBufferView.buffer.has_value())
								{
									context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!", system::ILogger::ELL_WARNING);
									return {};
								}

								const auto& bufferID = glTFBufferView.buffer.value();
								auto cpuBuffer = cpuBuffers[bufferID];

								const size_t globalOffset = [&]()
								{
									const size_t bufferViewOffset = glTFBufferView.byteOffset.has_value() ? glTFBufferView.byteOffset.value() : 0u;
									const size_t relativeAccessorOffset = glTFJointsXAccessor.byteOffset.has_value() ? glTFJointsXAccessor.byteOffset.value() : 0u;

									return bufferViewOffset + relativeAccessorOffset;
								}();

								auto& overrideRef = overrideJointsReference.emplace_back();
								overrideRef.accessor = &glTFJointsXAccessor;
								overrideRef.format = jointsFormat;

								overrideRef.bufferRange.buffer = core::smart_refctd_ptr(cpuBuffer);
								overrideRef.bufferRange.offset = globalOffset;
								overrideRef.bufferRange.size = overrideRef.accessor->count.value() * asset::getTexelOrBlockBytesize(overrideRef.format);

								auto* bufferData = reinterpret_cast<uint8_t*>(overrideRef.bufferRange.buffer->getPointer());
								overrideRef.data = bufferData + overrideRef.bufferRange.offset;
							}
						}

						for (uint8_t i = 0; i < glTFprimitive.attributes.weights.size(); ++i)
						{
							if (glTFprimitive.attributes.weights[i].has_value())
							{
								const size_t accessorID = glTFprimitive.attributes.weights[i].value();

								auto& glTFWeightsXAccessor = glTF.accessors[accessorID];

								if (glTFWeightsXAccessor.type.value() != SGLTF::SGLTFAccessor::SGLTFT_VEC4)
								{
									context.loadContext.params.logger.log("GLTF: WEIGHTS ACCESSOR MUST HAVE VEC4 TYPE!", system::ILogger::ELL_WARNING);
									return {};
								}
								
								const asset::E_FORMAT weightsFormat = [&]()
								{
									if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_FLOAT)
										return EF_R32G32B32A32_SFLOAT;
									else if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_BYTE)
										return EF_R8G8B8A8_UINT;
									else if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT)
										return EF_R16G16B16A16_UINT;
									else
										EF_UNKNOWN;
								}();

								if (weightsFormat == EF_UNKNOWN)
								{
									context.loadContext.params.logger.log("GLTF: DETECTED WEIGHTS BUFFER WITH INVALID COMPONENT TYPE!", system::ILogger::ELL_WARNING);
									return {};
								}

								if (!glTFWeightsXAccessor.bufferView.has_value())
								{
									context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!", system::ILogger::ELL_WARNING);
									return {};
								}

								const auto& bufferViewID = glTFWeightsXAccessor.bufferView.value();
								const auto& glTFBufferView = glTF.bufferViews[bufferViewID];

								if (!glTFBufferView.buffer.has_value())
								{
									context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!", system::ILogger::ELL_WARNING);
									return {};
								}

								const auto& bufferID = glTFBufferView.buffer.value();
								auto cpuBuffer = cpuBuffers[bufferID];

								const size_t globalOffset = [&]()
								{
									const size_t bufferViewOffset = glTFBufferView.byteOffset.has_value() ? glTFBufferView.byteOffset.value() : 0u;
									const size_t relativeAccessorOffset = glTFWeightsXAccessor.byteOffset.has_value() ? glTFWeightsXAccessor.byteOffset.value() : 0u;

									return bufferViewOffset + relativeAccessorOffset;
								}();

								auto& overrideRef = overrideWeightsReference.emplace_back();
								overrideRef.accessor = &glTFWeightsXAccessor;
								overrideRef.format = weightsFormat;

								overrideRef.bufferRange.buffer = core::smart_refctd_ptr(cpuBuffer);
								overrideRef.bufferRange.offset = globalOffset;
								overrideRef.bufferRange.size = overrideRef.accessor->count.value() * asset::getTexelOrBlockBytesize(overrideRef.format);

								auto* bufferData = reinterpret_cast<uint8_t*>(overrideRef.bufferRange.buffer->getPointer());
								overrideRef.data = bufferData + overrideRef.bufferRange.offset;
							}
						}

						uint32_t maxJointsPerVertex = 0xdeadbeef;

						if (overrideJointsReference.size() && overrideWeightsReference.size())
						{
							if (overrideJointsReference.size() != overrideWeightsReference.size())
							{
								context.loadContext.params.logger.log("GLTF: JOINTS ATTRIBUTES VERTEX BUFFERS AMOUNT MUST BE EQUAL TO WEIGHTS ATTRIBUTES VERTEX BUFFERS AMOUNT!", system::ILogger::ELL_WARNING);
								return {};
							}
						
							if (overrideJointsReference.size() > 1u || overrideWeightsReference.size() > 1u)
							{
								if (!std::equal(std::begin(overrideJointsReference) + 1, std::end(overrideJointsReference), std::begin(overrideJointsReference), [](const OverrideReference& lhs, const OverrideReference& rhs) { return lhs.format == rhs.format && lhs.accessor->count.value() == rhs.accessor->count.value(); }))
								{
									context.loadContext.params.logger.log("GLTF: JOINTS ATTRIBUTES VERTEX BUFFERS MUST NOT HAVE VARIOUS DATA TYPE OR LENGTH!", system::ILogger::ELL_WARNING);
									return {};
								}

								if (!std::equal(std::begin(overrideWeightsReference) + 1, std::end(overrideWeightsReference), std::begin(overrideWeightsReference), [](const OverrideReference& lhs, const OverrideReference& rhs) { return lhs.format == rhs.format && lhs.accessor->count.value() == rhs.accessor->count.value(); }))
								{
									context.loadContext.params.logger.log("GLTF: WEIGHTS ATTRIBUTES VERTEX BUFFERS MUST NOT HAVE VARIOUS DATA TYPE OR LENGTH!", system::ILogger::ELL_WARNING);
									return {};
								}

								/*
									TODO: it is not enough, I should have checked if joints attribute buffers are the same
									because if they are different then sorting weights is wrong.
								*/
							}

							struct OverrideSkinningBuffers
							{
								struct Override
								{
									core::smart_refctd_ptr<asset::ICPUBuffer> cpuBuffer;
									E_FORMAT format;
								};

								Override jointsAttributes;
								Override weightsAttributes;
							} overrideSkinningBuffers;
							{
								const uint16_t overrideReferencesCount = overrideJointsReference.size(); //! doesn't matter if overrideJointsReference or overrideWeightsReference
								const size_t vCommonOverrideAttributesCount = overrideJointsReference[0].accessor->count.value(); //! doesn't matter if overrideJointsReference or overrideWeightsReference

								const E_FORMAT vJointsFormat = overrideJointsReference[0].format;
								const size_t vJointsTexelByteSize = asset::getTexelOrBlockBytesize(vJointsFormat);

								const E_FORMAT vWeightsFormat = overrideWeightsReference[0].format;
								const size_t vWeightsTexelByteSize = asset::getTexelOrBlockBytesize(vWeightsFormat);

								core::smart_refctd_ptr<asset::ICPUBuffer> vOverrideJointsBuffer = nullptr;
								core::smart_refctd_ptr<asset::ICPUBuffer> vOverrideWeightsBuffer = nullptr;

								auto createOverrideBuffers = [&]<typename JointComponentT, typename WeightCompomentT>() -> void
								{
									constexpr bool isValidJointComponentT = std::is_same<JointComponentT, uint8_t>::value || std::is_same<JointComponentT, uint16_t>::value;
									constexpr bool isValidWeighComponentT = std::is_same<WeightCompomentT, uint8_t>::value || std::is_same<WeightCompomentT, uint16_t>::value || std::is_same<WeightCompomentT, float>::value;
									static_assert(isValidJointComponentT && isValidWeighComponentT);

									vOverrideJointsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * vJointsTexelByteSize);
									vOverrideWeightsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * vWeightsTexelByteSize);

									for (size_t vAttributeIx = 0; vAttributeIx < vCommonOverrideAttributesCount; ++vAttributeIx)
									{
										const size_t commonVJointsOffset = vAttributeIx * vJointsTexelByteSize;
										const size_t commonVWeightsOffset = vAttributeIx * vWeightsTexelByteSize;

										struct VertexInfluenceData
										{
											struct ComponentData
											{
												JointComponentT joint;
												WeightCompomentT weight;
											};

											std::array<ComponentData, 4u> perVertexComponentsData;
										};

										std::vector<VertexInfluenceData> vertexInfluenceDataContainer;

										for (uint16_t i = 0; i < overrideReferencesCount; ++i)
										{
											VertexInfluenceData& vertexInfluenceData = vertexInfluenceDataContainer.emplace_back();

											auto* vJointsComponentDataRaw = reinterpret_cast<uint8_t*>(overrideJointsReference[i].data) + commonVJointsOffset;
											auto* vWeightsComponentDataRaw = reinterpret_cast<uint8_t*>(overrideWeightsReference[i].data) + commonVWeightsOffset;

											for (uint16_t i = 0; i < vertexInfluenceData.perVertexComponentsData.size(); ++i) //! iterate over single components
											{
												VertexInfluenceData::ComponentData& skinComponent = vertexInfluenceData.perVertexComponentsData[i];

												JointComponentT* vJoint = reinterpret_cast<JointComponentT*>(vJointsComponentDataRaw) + i;
												WeightCompomentT* vWeight = reinterpret_cast<WeightCompomentT*>(vWeightsComponentDataRaw) + i;

												skinComponent.joint = *vJoint;
												skinComponent.weight = *vWeight;
											}
										}

										std::vector<VertexInfluenceData::ComponentData> skinComponentUnlimitedStream;
										{
											for(const auto& vertexInfluenceData : vertexInfluenceDataContainer)
												for (const auto& skinComponent : vertexInfluenceData.perVertexComponentsData)
												{
													auto& data = skinComponentUnlimitedStream.emplace_back();

													data.joint = skinComponent.joint;
													data.weight = skinComponent.weight;
												}
										}

										//! sort, cache and keep only biggest influencers
										std::sort(std::begin(skinComponentUnlimitedStream), std::end(skinComponentUnlimitedStream), [&](const VertexInfluenceData::ComponentData& lhs, const VertexInfluenceData::ComponentData& rhs) { return lhs.weight < rhs.weight; });
										{
											auto iteratorEnd = skinComponentUnlimitedStream.begin() + (vertexInfluenceDataContainer.size() - 1u) * 4u;
											if (skinComponentUnlimitedStream.begin() != iteratorEnd)
												skinComponentUnlimitedStream.erase(skinComponentUnlimitedStream.begin(), iteratorEnd);

											std::sort(std::begin(skinComponentUnlimitedStream), std::end(skinComponentUnlimitedStream), [&](const VertexInfluenceData::ComponentData& lhs, const VertexInfluenceData::ComponentData& rhs) { return lhs.joint < rhs.joint; });
										}

										auto* vOverrideJointsData = reinterpret_cast<uint8_t*>(vOverrideJointsBuffer->getPointer()) + commonVJointsOffset;
										auto* vOverrideWeightsData = reinterpret_cast<uint8_t*>(vOverrideWeightsBuffer->getPointer()) + commonVWeightsOffset;

										uint32_t validWeights = {};
										for (uint16_t i = 0; i < 4u; ++i)
										{
											const auto& skinComponent = skinComponentUnlimitedStream[i];

											JointComponentT* vOverrideJoint = reinterpret_cast<JointComponentT*>(vOverrideJointsData) + i;
											WeightCompomentT* vOverrideWeight = reinterpret_cast<WeightCompomentT*>(vOverrideWeightsData) + i;

											*vOverrideJoint = skinComponent.joint;
											*vOverrideWeight = skinComponent.weight;

											if (*vOverrideWeight != 0)
												++validWeights;
										}

										maxJointsPerVertex = std::max(maxJointsPerVertex == 0xdeadbeef ? 0u : maxJointsPerVertex, validWeights);
									}

									using REPACK_JOINTS_FORMAT = E_FORMAT;
									using REPACK_WEIGHTS_FORMAT = E_FORMAT;

									auto getRepackFormats = [&]() -> std::pair<REPACK_JOINTS_FORMAT, REPACK_WEIGHTS_FORMAT>
									{
										E_FORMAT repackJointsFormat = EF_UNKNOWN;
										E_FORMAT repackWeightsFormat = EF_UNKNOWN;

										switch (maxJointsPerVertex)
										{
											case 1u:
											{
												if constexpr (std::is_same<JointComponentT, uint8_t>::value)
													repackJointsFormat = EF_R8_UINT;
												else if (std::is_same<JointComponentT, uint16_t>::value)
													repackJointsFormat = EF_R16_UINT;

												if constexpr (std::is_same<WeightCompomentT, uint8_t>::value)
													repackWeightsFormat = EF_R8_UINT;
												else if (std::is_same<WeightCompomentT, uint16_t>::value)
													repackWeightsFormat = EF_R16_UINT;
												else if (std::is_same<WeightCompomentT, float>::value)
													repackWeightsFormat = EF_R32_SFLOAT;
											} break;

											case 2u:
											{
												if constexpr (std::is_same<JointComponentT, uint8_t>::value)
													repackJointsFormat = EF_R8G8_UINT;
												else if (std::is_same<JointComponentT, uint16_t>::value)
													repackJointsFormat = EF_R16G16_UINT;

												if constexpr (std::is_same<WeightCompomentT, uint8_t>::value)
													repackWeightsFormat = EF_R8G8_UINT;
												else if (std::is_same<WeightCompomentT, uint16_t>::value)
													repackWeightsFormat = EF_R16G16_UINT;
												else if (std::is_same<WeightCompomentT, float>::value)
													repackWeightsFormat = EF_R32G32_SFLOAT;
											} break;

											default:
											{
												if constexpr (std::is_same<JointComponentT, uint8_t>::value)
													repackJointsFormat = EF_R8G8B8A8_UINT;
												else if (std::is_same<JointComponentT, uint16_t>::value)
													repackJointsFormat = EF_R16G16B16A16_UINT;

												if constexpr (std::is_same<WeightCompomentT, uint8_t>::value)
													repackWeightsFormat = EF_R8G8B8A8_UINT;
												else if (std::is_same<WeightCompomentT, uint16_t>::value)
													repackWeightsFormat = EF_R16G16B16A16_UINT;
												else if (std::is_same<WeightCompomentT, float>::value)
													repackWeightsFormat = EF_R32G32B32A32_SFLOAT;
											} break; //! vertex formats need to be PoT
										}

										return std::make_pair(repackJointsFormat, repackWeightsFormat);
									};

									const auto [repackJointsFormat, repackWeightsFormat] = getRepackFormats();
									{
										const size_t repackJointsTexelByteSize = asset::getTexelOrBlockBytesize(repackJointsFormat);
										const size_t repackWeightsTexelByteSize = asset::getTexelOrBlockBytesize(repackWeightsFormat);

										auto vOverrideRepackedJointsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * repackJointsTexelByteSize);
										auto vOverrideRepackedWeightsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * repackWeightsTexelByteSize);

										memset(vOverrideRepackedJointsBuffer->getPointer(), 0, vOverrideRepackedJointsBuffer->getSize());
										memset(vOverrideRepackedWeightsBuffer->getPointer(), 0, vOverrideRepackedWeightsBuffer->getSize());
										{ //! pack buffers and quantize weights buffer

											_NBL_STATIC_INLINE_CONSTEXPR uint16_t MAX_INFLUENCE_WEIGHTS_PER_VERTEX = 4;

											struct QuantRequest
											{
												QuantRequest()
												{
													std::get<WEIGHT_ENCODING>(encodeData[0]) = WE_UNORM8;
													std::get<E_FORMAT>(encodeData[0]) = EF_R8G8B8A8_UNORM;

													std::get<WEIGHT_ENCODING>(encodeData[1]) = WE_UNORM16;
													std::get<E_FORMAT>(encodeData[1]) = EF_R16G16B16A16_UNORM;

													std::get<WEIGHT_ENCODING>(encodeData[2]) = WE_SFLOAT;
													std::get<E_FORMAT>(encodeData[2]) = EF_R32G32B32A32_SFLOAT;	
												}

												using QUANT_BUFFER = uint8_t[32]; //! for entire weights glTF vec4 entry
												using ERROR_TYPE = float; // for each weight component
												using ERROR_BUFFER = ERROR_TYPE[MAX_INFLUENCE_WEIGHTS_PER_VERTEX]; //! abs(decode(encode(weight)) - weight)
												std::array<std::tuple<WEIGHT_ENCODING, E_FORMAT, QUANT_BUFFER, ERROR_BUFFER>, WE_COUNT> encodeData;

												struct BestWeightsFit
												{
													WEIGHT_ENCODING quantizeEncoding = WE_UNORM8;
													ERROR_TYPE smallestError = FLT_MAX;
												} bestWeightsFit;
											} quantRequest;
										
											for (size_t vAttributeIx = 0; vAttributeIx < vCommonOverrideAttributesCount; ++vAttributeIx)
											{
												auto* unpackedJointsData = reinterpret_cast<JointComponentT*>(reinterpret_cast<uint8_t*>(vOverrideJointsBuffer->getPointer()) + vAttributeIx * vJointsTexelByteSize);
												auto* unpackedWeightsData = reinterpret_cast<WeightCompomentT*>(reinterpret_cast<uint8_t*>(vOverrideWeightsBuffer->getPointer()) + vAttributeIx * vWeightsTexelByteSize);

												auto* packedJointsData = reinterpret_cast<JointComponentT*>(reinterpret_cast<uint8_t*>(vOverrideRepackedJointsBuffer->getPointer()) + vAttributeIx * repackJointsTexelByteSize);
												auto* packedWeightsData = reinterpret_cast<WeightCompomentT*>(reinterpret_cast<uint8_t*>(vOverrideRepackedWeightsBuffer->getPointer()) + vAttributeIx * repackWeightsTexelByteSize);

												auto quantize = [&](const core::vectorSIMDf& input, void* data, const E_FORMAT requestQuantizeFormat)
												{
													return ICPUMeshBuffer::setAttribute(input, data, requestQuantizeFormat);
												};

												auto decodeQuant = [&](void* data, const E_FORMAT requestQuantizeFormat)
												{
													core::vectorSIMDf out;
													ICPUMeshBuffer::getAttribute(out, data, requestQuantizeFormat);
													return out;
												};

												core::vectorSIMDf packedWeightsStream; //! always go with full vectorSIMDf stream, weights being not used are leaved with default vector's compoment value and are not considered

												for (uint16_t i = 0, vxSkinComponentOffset = 0; i < 4u; ++i) //! packing
												{
													if (unpackedWeightsData[i])
													{
														packedJointsData[vxSkinComponentOffset] = unpackedJointsData[i];
														packedWeightsStream.pointer[i] = packedWeightsData[vxSkinComponentOffset] = unpackedWeightsData[i];

														++vxSkinComponentOffset;
														assert(vxSkinComponentOffset <= maxJointsPerVertex);
													}
												}

												for(uint16_t i = 0; i < quantRequest.encodeData.size(); ++i) //! quantization test
												{
													auto& encode = quantRequest.encodeData[i];
													auto* quantBuffer = std::get<QuantRequest::QUANT_BUFFER>(encode);
													auto* errorBuffer = std::get<QuantRequest::ERROR_BUFFER>(encode);
													const WEIGHT_ENCODING requestWeightEncoding = std::get<WEIGHT_ENCODING>(encode);
													const E_FORMAT requestQuantFormat = std::get<E_FORMAT>(encode);

													quantize(packedWeightsStream, quantBuffer, requestQuantFormat);
													core::vectorSIMDf quantsDecoded = decodeQuant(quantBuffer, requestQuantFormat);

													for (uint16_t i = 0; i < MAX_INFLUENCE_WEIGHTS_PER_VERTEX; ++i)
													{
														const auto& weightInput = packedWeightsStream.pointer[i];
														if (weightInput)
														{
															const QuantRequest::ERROR_TYPE& errorComponent = errorBuffer[i] = core::abs(quantsDecoded.pointer[i] - weightInput);

															if (errorComponent)
															{
																if (errorComponent < quantRequest.bestWeightsFit.smallestError)
																{
																	//! update request quantization format
																	quantRequest.bestWeightsFit.smallestError = errorComponent;
																	quantRequest.bestWeightsFit.quantizeEncoding = requestWeightEncoding;
																}
															}
														}
													}
												}
											}

											auto getWeightsQuantizeFormat = [&]() -> E_FORMAT
											{
												switch (maxJointsPerVertex)
												{
													case 1u:
													{
														switch (quantRequest.bestWeightsFit.quantizeEncoding)
														{
															case WE_UNORM8:
															{
																return EF_R8_UNORM;
															} break;

															case WE_UNORM16:
															{
																return EF_R16_UNORM;
															} break;

															case WE_SFLOAT:
															{
																return EF_R32_SFLOAT;
															} break;
														}

													} break;

													case 2u:
													{
														switch (quantRequest.bestWeightsFit.quantizeEncoding)
														{
															case WE_UNORM8:
															{
																return EF_R8G8_UNORM;
															} break;

															case WE_UNORM16:
															{
																return EF_R16G16_UNORM;
															} break;

															case WE_SFLOAT:
															{
																return EF_R32G32_SFLOAT;
															} break;
														}
													} break;

													default:
													{
														switch (quantRequest.bestWeightsFit.quantizeEncoding)
														{
															case WE_UNORM8:
															{
																return EF_R8G8B8A8_UNORM;
															} break;

															case WE_UNORM16:
															{
																return EF_R16G16B16A16_UNORM;
															} break;

															case WE_SFLOAT:
															{
																return EF_R32G32B32A32_SFLOAT;
															} break;
														}
													} break;
												}

												return EF_UNKNOWN;
											};

											vOverrideJointsBuffer = std::move(vOverrideRepackedJointsBuffer);
											overrideSkinningBuffers.jointsAttributes.cpuBuffer = std::move(vOverrideJointsBuffer);
											overrideSkinningBuffers.jointsAttributes.format = repackJointsFormat;

											const E_FORMAT weightsQuantizeFormat = getWeightsQuantizeFormat();
											const size_t weightComponentsByteStride = asset::getTexelOrBlockBytesize(weightsQuantizeFormat);
											assert(weightsQuantizeFormat != EF_UNKNOWN);
											{
												vOverrideWeightsBuffer = std::move(core::smart_refctd_ptr<asset::ICPUBuffer>()); //! free memory
												auto vOverrideQuantizedWeightsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(weightComponentsByteStride * vCommonOverrideAttributesCount);
												{
													for (size_t vAttributeIx = 0; vAttributeIx < vCommonOverrideAttributesCount; ++vAttributeIx)
													{
														const size_t quantizedVWeightsOffset = vAttributeIx * weightComponentsByteStride;
														void* quantizedWeightsData = reinterpret_cast<uint8_t*>(vOverrideQuantizedWeightsBuffer->getPointer()) + quantizedVWeightsOffset;

														core::vectorSIMDf packedWeightsStream; //! always go with full vectorSIMDf stream, weights being not used are leaved with default vector's compoment value and are not considered
														auto* packedWeightsData = reinterpret_cast<WeightCompomentT*>(reinterpret_cast<uint8_t*>(vOverrideRepackedWeightsBuffer->getPointer()) + vAttributeIx * repackWeightsTexelByteSize);

														for (uint16_t i = 0; i < maxJointsPerVertex; ++i)
															packedWeightsStream.pointer[i] = packedWeightsData[i];
														
														ICPUMeshBuffer::setAttribute(packedWeightsStream, quantizedWeightsData, weightsQuantizeFormat); //! quantize
													}
												}
												
												overrideSkinningBuffers.weightsAttributes.cpuBuffer = std::move(vOverrideQuantizedWeightsBuffer);
												overrideSkinningBuffers.weightsAttributes.format = weightsQuantizeFormat;
											}
										}
			
									}
								};

								switch (vJointsFormat)
								{
									case EF_R8G8B8A8_UINT:
									{
										using JointCompomentT = uint8_t;

										switch (vWeightsFormat)
										{
											case EF_R32G32B32A32_SFLOAT:
											{
												using WeightCompomentT = float;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;

											case EF_R8G8B8A8_UINT:
											{
												using WeightCompomentT = uint8_t;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;

											case EF_R16G16B16A16_UINT:
											{
												using WeightCompomentT = uint16_t;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;
										}
									} break;

									case EF_R16G16B16A16_UINT:
									{
										using JointCompomentT = uint16_t;

										switch (vWeightsFormat)
										{
											case EF_R32G32B32A32_SFLOAT:
											{
												using WeightCompomentT = float;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;

											case EF_R8G8B8A8_UINT:
											{
												using WeightCompomentT = uint8_t;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;

											case EF_R16G16B16A16_UINT:
											{
												using WeightCompomentT = uint16_t;
												createOverrideBuffers.template operator() < JointCompomentT, WeightCompomentT > ();
											} break;
										}
									} break;

									default:
									{
										assert(false); //! at this line probably impossible
									} break;
								}

								auto setOverrideBufferBinding = [&](OverrideSkinningBuffers::Override& overrideData, uint16_t attributeID) -> bool
								{
									asset::SBufferBinding<ICPUBuffer> bufferBinding;
									bufferBinding.buffer = core::smart_refctd_ptr(overrideData.cpuBuffer);
									bufferBinding.offset = 0u;

									auto findFreeBinding = [&]() -> uint32_t
									{
										for (uint16_t i = 0; i < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
											if (!(vertexInputParams.enabledBindingFlags & core::createBitmask({ i })))
												return i;
									
										return 0xdeadbeef;
									};

									const uint32_t bufferBindingId = findFreeBinding();
									if (bufferBindingId == 0xdeadbeef)
										return false;

									cpuMeshBuffer->setVertexBufferBinding(std::move(bufferBinding), bufferBindingId);

									vertexInputParams.enabledBindingFlags |= core::createBitmask({ bufferBindingId });
									vertexInputParams.bindings[bufferBindingId].inputRate = EVIR_PER_VERTEX;
									vertexInputParams.bindings[bufferBindingId].stride = asset::getTexelOrBlockBytesize(overrideData.format);

									vertexInputParams.enabledAttribFlags |= core::createBitmask({ attributeID });
									vertexInputParams.attributes[attributeID].binding = bufferBindingId;
									vertexInputParams.attributes[attributeID].format = overrideData.format;
									vertexInputParams.attributes[attributeID].relativeOffset = 0;

									return true;
								};

								if (!setOverrideBufferBinding(overrideSkinningBuffers.jointsAttributes, cpuMeshBuffer->getJointIDAttributeIx()))
								{
									context.loadContext.params.logger.log("GLTF: COULD NOT SET OVERRIDE JOINTS BUFFER!", system::ILogger::ELL_WARNING);
									return {};
								}

								if(!setOverrideBufferBinding(overrideSkinningBuffers.weightsAttributes, cpuMeshBuffer->getJointWeightAttributeIx()))
								{
									context.loadContext.params.logger.log("GLTF: COULD NOT SET OVERRIDE WEIGHTS BUFFER!", system::ILogger::ELL_WARNING);
									return {};
								}
							}
						}

						auto getShaders = [&](bool hasUV, bool hasColor) -> std::pair<core::smart_refctd_ptr<ICPUSpecializedShader>, core::smart_refctd_ptr<ICPUSpecializedShader>>
						{
							auto loadShader = [&](const std::string_view& cacheKey) -> core::smart_refctd_ptr<ICPUSpecializedShader>
							{
								size_t storageSz = 1ull;
								asset::SAssetBundle bundle;
								const IAsset::E_TYPE types[]{ IAsset::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

								assetManager->findAssets(storageSz, &bundle, cacheKey.data(), types);
								if (bundle.getContents().empty())
									return nullptr;
								auto assets = bundle.getContents();

								return core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assets.begin()[0]);
							};

							if (hasUV) // if both UV and Color defined - we use the UV
								return std::make_pair(loadShader(VERT_SHADER_UV_CACHE_KEY), loadShader(FRAG_SHADER_UV_CACHE_KEY));
							else if (hasColor)
								return std::make_pair(loadShader(VERT_SHADER_COLOR_CACHE_KEY), loadShader(FRAG_SHADER_COLOR_CACHE_KEY));
							else
								return std::make_pair(loadShader(VERT_SHADER_NO_UV_COLOR_CACHE_KEY), loadShader(FRAG_SHADER_NO_UV_COLOR_CACHE_KEY));
						};

						core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> cpuPipeline;
						const std::string& pipelineCacheKey = getPipelineCacheKey(primitiveTopology, vertexInputParams);
						{
							const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
							auto pipeline_bundle = _override->findCachedAsset(pipelineCacheKey, types, context.loadContext, _hierarchyLevel + ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW);
							if (!pipeline_bundle.getContents().empty())
								cpuPipeline = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(pipeline_bundle.getContents().begin()[0]);
							else
							{
								CGLTFPipelineMetadata::SGLTFMaterialParameters pushConstants;
								CGLTFPipelineMetadata::SGLTFSkinParameters skinParameters;
								skinParameters.perVertexJointsAmount = maxJointsPerVertex;

								SMaterialDependencyData materialDependencyData;
								const bool ds3lAvailableFlag = glTFprimitive.material.has_value();

								materialDependencyData.cpuMeshBuffer = cpuMeshBuffer.get();
								materialDependencyData.glTFMaterial = ds3lAvailableFlag ? &glTF.materials[glTFprimitive.material.value()] : nullptr;
								materialDependencyData.cpuTextures = &cpuTextures;

								auto cpuPipelineLayout = makePipelineLayoutFromGLTF(context, pushConstants, materialDependencyData);
								auto [cpuVertexShader, cpuFragmentShader] = getShaders(hasUV, hasColor);

								if (!cpuVertexShader || !cpuFragmentShader)
								{
									context.loadContext.params.logger.log("GLTF: COULD NOT LOAD SHADERS!", system::ILogger::ELL_WARNING);
									return false;
								}

								cpuPipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(cpuPipelineLayout), nullptr, nullptr, vertexInputParams, blendParams, primitiveAssemblyParams, rastarizationParmas);
								cpuPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, cpuVertexShader.get());
								cpuPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, cpuFragmentShader.get());

								core::smart_refctd_ptr<CGLTFPipelineMetadata> glTFPipelineMetadata;
								{
									if (ds3lAvailableFlag)
									{
										if (materialDependencyData.glTFMaterial->pbrMetallicRoughness.has_value())
										{
											auto& glTFMetallicRoughness = materialDependencyData.glTFMaterial->pbrMetallicRoughness.value();

											if (glTFMetallicRoughness.baseColorFactor.has_value())
												for (uint8_t i = 0; i < glTFMetallicRoughness.baseColorFactor.value().size(); ++i)
													pushConstants.metallicRoughness.baseColorFactor[i] = glTFMetallicRoughness.baseColorFactor.value()[i];

											if (glTFMetallicRoughness.metallicFactor.has_value())
												pushConstants.metallicRoughness.metallicFactor = glTFMetallicRoughness.metallicFactor.value();

											if (glTFMetallicRoughness.roughnessFactor.has_value())
												pushConstants.metallicRoughness.roughnessFactor = glTFMetallicRoughness.roughnessFactor.value();
										}

										if (materialDependencyData.glTFMaterial->alphaCutoff.has_value())
											pushConstants.alphaCutoff = materialDependencyData.glTFMaterial->alphaCutoff.value();

										CGLTFLoader::SGLTF::SGLTFMaterial::E_ALPHA_MODE alphaModeStream = decltype(alphaModeStream)::EAM_OPAQUE;

										if (materialDependencyData.glTFMaterial->alphaMode.has_value())
											alphaModeStream = materialDependencyData.glTFMaterial->alphaMode.value();

										switch (alphaModeStream)
										{
										case decltype(alphaModeStream)::EAM_OPAQUE:
										{
											pushConstants.alphaMode = CGLTFPipelineMetadata::EAM_OPAQUE;
										} break;

										case decltype(alphaModeStream)::EAM_MASK:
										{
											pushConstants.alphaMode = CGLTFPipelineMetadata::EAM_MASK;
										} break;

										case decltype(alphaModeStream)::EAM_BLEND:
										{
											pushConstants.alphaMode = CGLTFPipelineMetadata::EAM_BLEND;
										} break;
										}

										if (materialDependencyData.glTFMaterial->emissiveFactor.has_value())
											for (uint8_t i = 0; i < materialDependencyData.glTFMaterial->emissiveFactor.value().size(); ++i)
												pushConstants.emissiveFactor[i] = materialDependencyData.glTFMaterial->emissiveFactor.value()[i];

										glTFPipelineMetadata = core::make_smart_refctd_ptr<CGLTFPipelineMetadata>(pushConstants, skinParameters, core::smart_refctd_ptr(m_basicViewParamsSemantics));
									}
								}

								globalPipelineMeta.push_back(core::smart_refctd_ptr(glTFPipelineMetadata));
								SAssetBundle pipelineBundle = SAssetBundle(core::smart_refctd_ptr(glTFPipelineMetadata), { cpuPipeline });

								_override->insertAssetIntoCache(pipelineBundle, pipelineCacheKey, context.loadContext, _hierarchyLevel + ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW);

								cpuMeshBuffer->setPipeline(std::move(cpuPipeline));
								cpuMesh->getMeshBufferVector().push_back(std::move(cpuMeshBuffer));
							}
						}
					}
				}
			}

			struct SkeletonData
			{
				struct HierarchyBuffer
				{
					std::string glTFNodeName;
					uint32_t glTFGlobalNodeID;
					core::matrix3x4SIMD defaultNodeTransform; // _defaultTransforms

					uint32_t localJointID; // in range [0, jointsSize - 1]
					uint32_t localParentJointID; // for _parentJointIDsBinding 
				};

				std::vector<HierarchyBuffer> hierarchyBuffer;

				struct ToPass
				{
					SBufferBinding<ICPUBuffer> parentJointIDs;
					SBufferBinding<ICPUBuffer> defaultTransforms;
					std::vector<const char*> jointNames;
				} toPass;
			};

			std::vector<SkeletonData> skeletons;

			for (size_t i = 0; i < glTF.skins.size(); ++i)
			{
				auto& skeleton = skeletons.emplace_back();

				const auto& glTFSkin = glTF.skins[i];

				for (size_t z = 0; z < glTFSkin.joints.size(); ++z)
				{
					const auto& nodeID = glTFSkin.joints[z];

					auto& nodeHierarchyData = skeleton.hierarchyBuffer.emplace_back();
					auto& glTFNode = glTF.nodes[nodeID];

					nodeHierarchyData.glTFNodeName = glTFNode.name.has_value() ? glTFNode.name.value() : "NBL_IDENTITY";
					nodeHierarchyData.glTFGlobalNodeID = nodeID;
					nodeHierarchyData.defaultNodeTransform = glTFNode.transformation.matrix;
					nodeHierarchyData.localJointID = z;
				}
				
				if (glTFSkin.skeleton.has_value()) //! we can explicitly point skin root node
				{
					const size_t& rootNodeID = glTFSkin.skeleton.value();
					auto rootIterator = std::find_if(std::begin(skeleton.hierarchyBuffer), std::end(skeleton.hierarchyBuffer), [rootNodeID](const auto& nodeHierarchyData) {return nodeHierarchyData.glTFGlobalNodeID == rootNodeID; });
					assert(rootIterator != std::end(skeleton.hierarchyBuffer));

					auto& root = *rootIterator;

					typedef decltype(skeleton.hierarchyBuffer) HIERARCHY_BUFFER;
					typedef decltype(glTF.nodes) GLTF_NODES;

					auto setParents = [](SkeletonData::HierarchyBuffer& nodeHierarchyData, uint32_t localParentID, HIERARCHY_BUFFER& hierarchyBuffer, GLTF_NODES& glTFNodes) -> uint32_t
					{
						auto setParents_impl = [](SkeletonData::HierarchyBuffer& nodeHierarchyData, uint32_t localParentID, HIERARCHY_BUFFER& hierarchyBuffer, GLTF_NODES& glTFNodes, auto& impl) -> uint32_t
						{
							auto& glTFNode = glTFNodes[nodeHierarchyData.glTFGlobalNodeID];

							if (glTFNode.children.size())
							{
								for (const auto& childID : glTFNode.children)
								{
									auto& glTFChildNode = glTFNodes[childID];

									auto childIterator = std::find_if(std::begin(hierarchyBuffer), std::end(hierarchyBuffer), [childID](const auto& nodeHierarchyData) {return nodeHierarchyData.glTFGlobalNodeID == childID; });
									assert(childIterator != std::end(hierarchyBuffer));

									SkeletonData::HierarchyBuffer& childNodeHierarchyData = *childIterator;
									childNodeHierarchyData.localParentJointID = impl(childNodeHierarchyData, nodeHierarchyData.localJointID, hierarchyBuffer, glTFNodes, impl);
								}
							}
							else
								return localParentID;

							return localParentID;
						};

						return nodeHierarchyData.localParentJointID = setParents_impl(nodeHierarchyData, localParentID, hierarchyBuffer, glTFNodes, setParents_impl);
					};

					setParents(root, 0xdeadbeef, skeleton.hierarchyBuffer, glTF.nodes);
				}
				else
				{
					// this needs a case-handle too!
				}

				skeleton.toPass.parentJointIDs.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * skeleton.hierarchyBuffer.size());
				skeleton.toPass.parentJointIDs.offset = 0u;

				skeleton.toPass.defaultTransforms.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(core::matrix3x4SIMD) * skeleton.hierarchyBuffer.size());
				skeleton.toPass.defaultTransforms.offset = 0u;

				for (size_t x = 0; x < skeleton.hierarchyBuffer.size(); ++x)
				{
					const auto& nodeHierarchyData = skeleton.hierarchyBuffer[x];
					skeleton.toPass.jointNames.push_back(nodeHierarchyData.glTFNodeName.c_str());

					auto* parentJointID = reinterpret_cast<uint32_t*>(skeleton.toPass.parentJointIDs.buffer->getPointer()) + x;
					*parentJointID = nodeHierarchyData.localParentJointID;

					auto* defaultTransform = reinterpret_cast<core::matrix3x4SIMD*>(skeleton.toPass.defaultTransforms.buffer->getPointer()) + x;
					*defaultTransform = nodeHierarchyData.defaultNodeTransform;
				}
			}

			for (uint32_t index = 0; index < glTF.nodes.size(); ++index)
			{
				auto& glTFnode = glTF.nodes[index];

				const uint32_t meshID = glTFnode.mesh.has_value() ? glTFnode.mesh.value() : 0xdeadbeef;
				const uint32_t skinID = glTFnode.skin.has_value() ? glTFnode.skin.value() : 0xdeadbeef;

				const bool skinDefined = meshID != 0xdeadbeef && skinID != 0xdeadbeef;
			
				if (skinDefined)
				{
					const auto& skeletonData = skeletons[skinID];

					const auto& glTFSkin = glTF.skins[skinID];

					const auto& accessorInverseBindMatricesID = glTFSkin.inverseBindMatrices.has_value() ? glTFSkin.inverseBindMatrices.value() : 0xdeadbeef;
					const auto& glTFjointNodeIDs = glTFSkin.joints; 

					auto cpuMesh = cpuMeshes[meshID];
					auto cpuMeshBuffer = cpuMesh->getMeshBufferVector()[0];
					const auto cpuPipeline = cpuMeshBuffer->getPipeline();
					const auto pipelineCacheKey = getPipelineCacheKey(cpuPipeline->getPrimitiveAssemblyParams().primitiveType, cpuPipeline->getVertexInputParams());


					const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
					auto pipeline_bundle = _override->findCachedAsset(pipelineCacheKey, types, context.loadContext, context.hierarchyLevel + ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW);
					assert(!pipeline_bundle.getContents().empty());

					const auto* pipelineMetadata = static_cast<const asset::CGLTFPipelineMetadata*>(pipeline_bundle.getMetadata());

					const uint32_t jointsPerVertex = pipelineMetadata->m_skinParams.perVertexJointsAmount;
					using bnd_t = asset::SBufferBinding<asset::ICPUBuffer>;
					core::smart_refctd_ptr<ICPUSkeleton> skeleton = core::make_smart_refctd_ptr<ICPUSkeleton>(bnd_t(skeletonData.toPass.parentJointIDs), bnd_t(skeletonData.toPass.defaultTransforms), skeletonData.toPass.jointNames.begin(), skeletonData.toPass.jointNames.end());

					SBufferBinding<ICPUBuffer> inverseBindPoseBufferBinding;
					inverseBindPoseBufferBinding.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glTFjointNodeIDs.size() * sizeof(core::matrix3x4SIMD));
					inverseBindPoseBufferBinding.offset = 0u;
					{
						if (accessorInverseBindMatricesID == 0xdeadbeef)
						{
							const core::matrix3x4SIMD identity;

							auto* data = reinterpret_cast<core::matrix3x4SIMD*>(inverseBindPoseBufferBinding.buffer->getPointer());
							auto* end = data + inverseBindPoseBufferBinding.buffer->getSize() / sizeof(core::matrix3x4SIMD);

							std::fill(data, end, identity);
						}
						else
						{
							const auto& glTFAccessor = glTF.accessors[accessorInverseBindMatricesID];

							if (!glTFAccessor.bufferView.has_value())
							{
								context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!", system::ILogger::ELL_WARNING);
								return false;
							}

							const auto& bufferViewID = glTFAccessor.bufferView.value();
							const auto& glTFBufferView = glTF.bufferViews[bufferViewID];

							if (!glTFBufferView.buffer.has_value())
							{
								context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!", system::ILogger::ELL_WARNING);
								return false;
							}

							const auto& bufferID = glTFBufferView.buffer.value();
							auto cpuBuffer = cpuBuffers[bufferID];

							const size_t globalIBPOffset = [&]()
							{
								const size_t bufferViewOffset = glTFBufferView.byteOffset.has_value() ? glTFBufferView.byteOffset.value() : 0u;
								const size_t relativeAccessorOffset = glTFAccessor.byteOffset.has_value() ? glTFAccessor.byteOffset.value() : 0u;

								return bufferViewOffset + relativeAccessorOffset;
							}();

							auto* inData = reinterpret_cast<core::matrix4SIMD*>(reinterpret_cast<uint8_t*>(cpuBuffer->getPointer()) + globalIBPOffset); //! glTF stores 4x4 IBP matrices
							auto* outData = reinterpret_cast<core::matrix3x4SIMD*>(inverseBindPoseBufferBinding.buffer->getPointer());

							for (uint32_t i = 0; i < glTFjointNodeIDs.size(); ++i)
								*(outData + i) = (inData + i)->extractSub3x4();
						}
					}

					SBufferBinding<ICPUBuffer> jointAABBBufferBinding;
					{
						std::vector<core::aabbox3df> jointBoundingBoxes(glTFjointNodeIDs.size());
						{
							struct JointVertexPair
							{
								core::vectorSIMDf vPosition;
								core::vector4du32_SIMD vJoint;
							} currentJointVertexPair;

							const uint16_t jointIDAttributeIx = cpuMeshBuffer->getJointIDAttributeIx();
							const auto* inverseBindPoseMatrices = reinterpret_cast<core::matrix3x4SIMD*>(reinterpret_cast<uint8_t*>(inverseBindPoseBufferBinding.buffer->getPointer()) + inverseBindPoseBufferBinding.offset);

							for (size_t i = 0; i < cpuMeshBuffer->getIndexCount(); ++i)
							{
								currentJointVertexPair.vPosition = cpuMeshBuffer->getPosition(i);
								assert(cpuMeshBuffer->getAttribute(currentJointVertexPair.vJoint.pointer, jointIDAttributeIx, i));

								auto updateBoundingBoxBuffers = [&](const uint32_t vtxJointID)
								{
									const auto& inverseBindPoseMatrix = inverseBindPoseMatrices[vtxJointID];
									inverseBindPoseMatrix.transformVect(currentJointVertexPair.vPosition);
									jointBoundingBoxes[vtxJointID].addInternalPoint(currentJointVertexPair.vPosition.getAsVector3df());
								};

								for (uint32_t i = 0; i < jointsPerVertex; ++i)
									updateBoundingBoxBuffers(currentJointVertexPair.vJoint.pointer[i]);
							}
						}

						jointAABBBufferBinding.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(jointBoundingBoxes.size() * sizeof(core::aabbox3df));
						memcpy(jointAABBBufferBinding.buffer->getPointer(), jointBoundingBoxes.data(), jointAABBBufferBinding.buffer->getSize());
						jointAABBBufferBinding.offset = 0u;
					}

					cpuMeshBuffer->setSkin(std::move(inverseBindPoseBufferBinding), std::move(jointAABBBufferBinding), std::move(skeleton), jointsPerVertex);
				}
			}

			/*
			* 
			*   TODO: IT MAY BE AN ISSUE VERY SOON!
			* 
				TODO: it needs hashes and better system for meta since gltf bundle may return more than one mesh
				and each mesh may have more than one meshbuffer, so more meta as well
			*/

			auto getGlobalPipelineCount = [&]() // TODO change it
			{
				size_t count = {};
				for (auto& meshMeta : globalMetadataContainer)
					for (auto& pipelineMeta : meshMeta)
						++count;
				return count;
			};

			core::smart_refctd_ptr<CGLTFMetadata> glTFPipelineMetadata = core::make_smart_refctd_ptr<CGLTFMetadata>(getGlobalPipelineCount());

			for (size_t i = 0; i < globalMetadataContainer.size(); ++i) // TODO change it 
				for (size_t z = 0; z < globalMetadataContainer[i].size(); ++z)
					glTFPipelineMetadata->placeMeta(z, cpuMeshes[i]->getMeshBufferVector()[z]->getPipeline(), *globalMetadataContainer[i][z]); 

			return SAssetBundle(core::smart_refctd_ptr(glTFPipelineMetadata), cpuMeshes);
		}

		bool CGLTFLoader::loadAndGetGLTF(SGLTF& glTF, SContext& context)
		{
			simdjson::dom::parser parser;
			auto* _file = context.loadContext.mainFile;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				system::future<size_t> future;
				_file->read(future, jsonBuffer->getPointer(), 0u, jsonBuffer->getSize());
				future.get();
			}

			simdjson::dom::object tweets = parser.parse(reinterpret_cast<uint8_t*>(jsonBuffer->getPointer()), jsonBuffer->getSize());
			simdjson::dom::element element;

			//std::filesystem::path filePath(_file->getFileName().c_str());
			//const std::string rootAssetDirectory = std::filesystem::absolute(filePath.remove_filename()).u8string();

			const auto& extensionsUsed = tweets.at_key("extensionsUsed");
			const auto& extensionsRequired = tweets.at_key("extensionsRequired");
			const auto& accessors = tweets.at_key("accessors");
			const auto& animations = tweets.at_key("animations");
			const auto& asset = tweets.at_key("asset");
			const auto& buffers = tweets.at_key("buffers");
			const auto& bufferViews = tweets.at_key("bufferViews");
			const auto& cameras = tweets.at_key("cameras");
			const auto& images = tweets.at_key("images");
			const auto& materials = tweets.at_key("materials");
			const auto& meshes = tweets.at_key("meshes");
			const auto& nodes = tweets.at_key("nodes");
			const auto& samplers = tweets.at_key("samplers");
			const auto& scene = tweets.at_key("scene");
			const auto& scenes = tweets.at_key("scenes");
			const auto& skins = tweets.at_key("skins");
			const auto& textures = tweets.at_key("textures");
			const auto& extensions = tweets.at_key("extensions");
			const auto& extras = tweets.at_key("extras");

			if (buffers.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& jsonBuffers = buffers.get_array();
				for (const auto& jsonBuffer : jsonBuffers)
				{
					auto& glTFBuffer = glTF.buffers.emplace_back();

					const auto& uri = jsonBuffer.at_key("uri");
					const auto& name = jsonBuffer.at_key("name");
					const auto& extensions = jsonBuffer.at_key("extensions");
					const auto& extras = jsonBuffer.at_key("extras");

					if (uri.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBuffer.uri = uri.get_string().value().data();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBuffer.name = name.get_string().value();
				}
			}

			if (bufferViews.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& jsonBufferViews = bufferViews.get_array();
				for (const auto& jsonBufferView : jsonBufferViews)
				{
					auto& glTFBufferView = glTF.bufferViews.emplace_back();

					const auto& buffer = jsonBufferView.at_key("buffer");
					const auto& byteOffset = jsonBufferView.at_key("byteOffset");
					const auto& byteLength = jsonBufferView.at_key("byteLength");
					const auto& byteStride = jsonBufferView.at_key("byteStride");
					const auto& target = jsonBufferView.at_key("target");
					const auto& name = jsonBufferView.at_key("name");
					const auto& extensions = jsonBufferView.at_key("extensions");
					const auto& extras = jsonBufferView.at_key("extras");

					if (buffer.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.buffer = buffer.get_uint64().value();

					if (byteOffset.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteOffset = byteOffset.get_uint64().value();

					if (byteLength.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteLength = byteLength.get_uint64().value();

					if (byteStride.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteStride = byteStride.get_uint64().value();

					if (target.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.target = target.get_uint64().value();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.name = name.get_string().value();
				}
			}

			if (images.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& imagesData = images.get_array();
				for (const auto& image : imagesData)
				{
					auto& glTFImage = glTF.images.emplace_back();

					const auto& uri = image.at_key("uri");
					const auto& mimeType = image.at_key("mimeType");
					const auto& bufferViewId = image.at_key("bufferView");
					const auto& name = image.at_key("name");
					const auto& extensions = image.at_key("extensions");
					const auto& extras = image.at_key("extras");

					if (uri.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFImage.uri = uri.get_string().value();

					if (mimeType.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFImage.mimeType = uri.get_string().value();

					if (bufferViewId.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFImage.bufferView = uri.get_uint64().value();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFImage.name = name.get_string().value();
				}
			}

			if (samplers.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& samplersData = samplers.get_array();
				for (const auto& sampler : samplersData)
				{
					auto& glTFSampler = glTF.samplers.emplace_back();

					const auto& magFilter = sampler.at_key("magFilter");
					const auto& minFilter = sampler.at_key("minFilter");
					const auto& wrapS = sampler.at_key("wrapS");
					const auto& wrapT = sampler.at_key("wrapT");
					const auto& name = sampler.at_key("name");
					const auto& extensions = sampler.at_key("extensions");
					const auto& extras = sampler.at_key("extras");

					if (magFilter.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSampler.magFilter = magFilter.get_uint64().value();

					if (minFilter.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSampler.minFilter = minFilter.get_uint64().value();

					if (wrapS.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSampler.wrapS = wrapS.get_uint64().value();

					if (wrapT.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSampler.wrapT = wrapT.get_uint64().value();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSampler.name = name.get_string().value();
				}
			}

			if (textures.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& texturesData = textures.get_array();
				for (const auto& texture : texturesData)
				{
					auto& glTFTexture = glTF.textures.emplace_back();

					const auto& sampler = texture.at_key("sampler");
					const auto& source = texture.at_key("source");
					const auto& name = texture.at_key("name");

					if (sampler.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFTexture.sampler = sampler.get_uint64().value();

					if (source.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFTexture.source = source.get_uint64().value();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFTexture.name = name.get_string().value();
				}
			}

			if (materials.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& materialsData = materials.get_array();
				for (const auto& material : materialsData)
				{
					auto& glTFMaterial = glTF.materials.emplace_back();

					const auto& name = material.at_key("name");
					const auto& pbrMetallicRoughness = material.at_key("pbrMetallicRoughness");
					const auto& normalTexture = material.at_key("normalTexture");
					const auto& occlusionTexture = material.at_key("occlusionTexture");
					const auto& emissiveTexture = material.at_key("emissiveTexture");
					const auto& emissiveFactor = material.at_key("emissiveFactor");
					const auto& alphaMode = material.at_key("alphaMode");
					const auto& alphaCutoff = material.at_key("alphaCutoff");
					const auto& doubleSided = material.at_key("doubleSided");

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFMaterial.name = name.get_string().value();

					if (pbrMetallicRoughness.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFMetalicRoughness = glTFMaterial.pbrMetallicRoughness.emplace();
						const auto& pbrMRData = pbrMetallicRoughness.get_object();

						const auto& baseColorFactor = pbrMRData.at_key("baseColorFactor");
						const auto& baseColorTexture = pbrMRData.at_key("baseColorTexture");
						const auto& metallicFactor = pbrMRData.at_key("metallicFactor");
						const auto& roughnessFactor = pbrMRData.at_key("roughnessFactor");
						const auto& metallicRoughnessTexture = pbrMRData.at_key("metallicRoughnessTexture");

						if (baseColorFactor.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& glTFBaseColorFactor = glTFMetalicRoughness.baseColorFactor.emplace();
							const auto& bcfData = baseColorFactor.get_array();

							for (uint32_t i = 0; i < bcfData.size(); ++i)
								glTFBaseColorFactor[i] = bcfData.at(i).get_double();
						}

						if (baseColorTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& glTFBaseColorTexture = glTFMetalicRoughness.baseColorTexture.emplace();
							const auto& bctData = baseColorTexture.get_object();

							const auto& index = bctData.at_key("index");
							const auto& texCoord = bctData.at_key("texCoord");

							if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFBaseColorTexture.index = index.get_uint64().value();

							if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFBaseColorTexture.texCoord = texCoord.get_uint64().value();
						}

						if (metallicFactor.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFMetalicRoughness.metallicFactor = metallicFactor.get_double().value();

						if (roughnessFactor.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFMetalicRoughness.roughnessFactor = roughnessFactor.get_double().value();

						if (metallicRoughnessTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& glTFMetallicRoughnessTexture = glTFMetalicRoughness.metallicRoughnessTexture.emplace();
							const auto& mrtData = metallicRoughnessTexture.get_object();

							const auto& index = mrtData.at_key("index");
							const auto& texCoord = mrtData.at_key("texCoord");

							if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFMetallicRoughnessTexture.index = index.get_uint64().value();

							if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFMetallicRoughnessTexture.texCoord = texCoord.get_uint64().value();
						}
					}

					if (normalTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFNormalTexture = glTFMaterial.normalTexture.emplace();
						const const auto& normalTextureData = normalTexture.get_object();

						const auto& index = normalTextureData.at_key("index");
						const auto& texCoord = normalTextureData.at_key("texCoord");
						const auto& scale = normalTextureData.at_key("scale");

						if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFNormalTexture.index = index.get_uint64().value();

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFNormalTexture.texCoord = texCoord.get_uint64().value();

						if (scale.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFNormalTexture.scale = texCoord.get_double().value();
					}

					if (occlusionTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFOcclusionTexture = glTFMaterial.occlusionTexture.emplace();
						const auto& occlusionTextureData = occlusionTexture.get_object();

						const auto& index = occlusionTextureData.at_key("index");
						const auto& texCoord = occlusionTextureData.at_key("texCoord");
						const auto& strength = occlusionTextureData.at_key("strength");

						if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFOcclusionTexture.index = index.get_uint64().value();

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFOcclusionTexture.texCoord = texCoord.get_uint64().value();

						if (strength.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFOcclusionTexture.strength = texCoord.get_double().value();
					}

					if (emissiveTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFEmissiveTexture = glTFMaterial.emissiveTexture.emplace();
						const auto& emissiveTextureData = emissiveTexture.get_object();

						const auto& index = emissiveTextureData.at_key("index");
						const auto& texCoord = emissiveTextureData.at_key("texCoord");

						if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFEmissiveTexture.index = index.get_uint64().value();

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFEmissiveTexture.texCoord = texCoord.get_uint64().value();
					}

					if (emissiveFactor.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFEmissiveFactor = glTFMaterial.emissiveFactor.emplace();
						const auto& efData = emissiveFactor.get_array();

						for (uint32_t i = 0; i < efData.size(); ++i)
							glTFEmissiveFactor[i] = efData.at(i).value();
					}
						
					if (alphaMode.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto streamAlphaMode = alphaMode.get_string().value().data();

						if (streamAlphaMode == "EAM_OPAQUE")
							glTFMaterial.alphaMode = CGLTFLoader::SGLTF::SGLTFMaterial::E_ALPHA_MODE::EAM_OPAQUE;
						else if (streamAlphaMode == "EAM_MASK")
							glTFMaterial.alphaMode = CGLTFLoader::SGLTF::SGLTFMaterial::E_ALPHA_MODE::EAM_MASK;
						else if (streamAlphaMode == "EAM_OPAQUE")
							glTFMaterial.alphaMode = CGLTFLoader::SGLTF::SGLTFMaterial::E_ALPHA_MODE::EAM_BLEND;
					}
						

					if (alphaCutoff.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFMaterial.alphaCutoff = alphaCutoff.get_double().value();

					if (doubleSided.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFMaterial.doubleSided = doubleSided.get_bool().value();
				}
			}

			if (animations.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& animationsData = animations.get_array();
				for (const auto& animation : animationsData)
				{
					auto& gltfAnimation = glTF.animations.emplace_back();

					const auto& channels = animation.at_key("channels");
					const auto& samplers = animation.at_key("samplers");
					const auto& name = animation.at_key("name");

					if(channels.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						const auto& channelsData = channels.get_array();
						for (const auto& channel : channelsData)
						{
							auto& gltfChannel = gltfAnimation.channels.emplace_back();

							const auto& sampler = channel.at_key("sampler");
							const auto& target = channel.at_key("target");

							if (sampler.error() != simdjson::error_code::NO_SUCH_FIELD)
								gltfChannel.sampler = sampler.get_uint64().value();	

							if (target.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const auto& targetData = target.get_object();

								const auto& node = targetData.at_key("node");
								const auto& path = targetData.at_key("path");

								if (node.error() != simdjson::error_code::NO_SUCH_FIELD)
									gltfChannel.target.node = node.get_uint64().value();

								if (path.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									const std::string gltfPath = path.get_string().value().data();

									if (gltfPath == "translation")
										gltfChannel.target.path = SGLTF::SGLTFAnimation::SGLTFChannel::SGLTFP_TRANSLATION;
									else if (gltfPath == "rotation")
										gltfChannel.target.path = SGLTF::SGLTFAnimation::SGLTFChannel::SGLTFP_ROTATION;
									else if (gltfPath == "scale")
										gltfChannel.target.path = SGLTF::SGLTFAnimation::SGLTFChannel::SGLTFP_SCALE;
									else if (gltfPath == "weights")
										gltfChannel.target.path = SGLTF::SGLTFAnimation::SGLTFChannel::SGLTFP_WEIGHTS;
									else
									{
										context.loadContext.params.logger.log("GLTF: UNSUPPORTED TARGET PATH!", system::ILogger::ELL_WARNING);
										return false;
									}
								}
							}
						}
					}

					if(samplers.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						const auto& samplersData = samplers.get_array();
						for (const auto& sampler : samplersData)
						{
							auto& gltfSampler = gltfAnimation.samplers.emplace_back();

							const auto& gltfInput = sampler.at_key("input");
							const auto& gltfOutput = sampler.at_key("output");
							const auto& gltfInterpolation = sampler.at_key("interpolation");

							if (gltfInput.error() != simdjson::error_code::NO_SUCH_FIELD)
								gltfSampler.input = gltfInput.get_uint64().value();

							if (gltfOutput.error() != simdjson::error_code::NO_SUCH_FIELD)
								gltfSampler.output = gltfOutput.get_uint64().value();

							if (gltfInterpolation.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const std::string interpolation = gltfInterpolation.get_string().value().data();

								if (interpolation == "LINEAR")
									gltfSampler.interpolation = SGLTF::SGLTFAnimation::SGLTFSampler::SGLTFI_LINEAR;
								else if (interpolation == "STEP")
									gltfSampler.interpolation = SGLTF::SGLTFAnimation::SGLTFSampler::SGLTFI_STEP;
								else if (interpolation == "CUBICSPLINE")
									gltfSampler.interpolation = SGLTF::SGLTFAnimation::SGLTFSampler::SGLTFI_CUBICSPLINE;
								else
								{
									context.loadContext.params.logger.log("GLTF: UNSUPPORTED INTERPOLATION!", system::ILogger::ELL_WARNING);
									return false;
								}
							}

						}
					}

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						gltfAnimation.name = name.get_string().value();
				}
			}

			if (skins.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& skinsData = skins.get_array();
				for (const auto& skin : skinsData)
				{
					auto& glTFSkin = glTF.skins.emplace_back();

					const auto& inverseBindMatrices = skin.at_key("inverseBindMatrices");
					const auto& skeleton = skin.at_key("skeleton");
					const auto& joints = skin.at_key("joints");
					const auto& name = skin.at_key("name");

					if (inverseBindMatrices.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSkin.inverseBindMatrices = inverseBindMatrices.get_uint64().value();

					if (skeleton.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSkin.skeleton = skeleton.get_uint64().value();

					if (joints.error() != simdjson::error_code::NO_SUCH_FIELD)
						for (const auto& joint : joints.get_array())
							glTFSkin.joints.push_back(joint.get_uint64().value());

					if (glTFSkin.joints.size() > SGLTF::SGLTFSkin::MAX_JOINTS_REFERENCES)
					{
						context.loadContext.params.logger.log("GLTF: DETECTED TOO MANY JOINTS REFERENCES!", system::ILogger::ELL_WARNING);
						return false;
					}

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFSkin.name = name.get_string().value();
				}
			}

			if (accessors.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& aData = accessors.get_array();
				for (const auto& accessor : aData)
				{
					auto& glTFAccessor = glTF.accessors.emplace_back();

					const auto& bufferView = accessor.at_key("bufferView");
					const auto& byteOffset = accessor.at_key("byteOffset");
					const auto& componentType = accessor.at_key("componentType");
					const auto& normalized = accessor.at_key("normalized");
					const auto& count = accessor.at_key("count");
					const auto& type = accessor.at_key("type");
					const auto& max = accessor.at_key("max");
					const auto& min = accessor.at_key("min");
					const auto& sparse = accessor.at_key("sparse");
					const auto& name = accessor.at_key("name");
					const auto& extensions = accessor.at_key("extensions");
					const auto& extras = accessor.at_key("extras");

					if (bufferView.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.bufferView = bufferView.get_uint64().value();

					if (byteOffset.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.byteOffset = byteOffset.get_uint64().value();

					if (componentType.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.componentType = componentType.get_uint64().value();

					if (normalized.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.normalized = normalized.get_bool().value();

					if (count.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.count = count.get_uint64().value();

					if (type.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						std::string typeStream = type.get_string().value().data();

						if (typeStream == "SCALAR")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_SCALAR;
						else if (typeStream == "VEC2")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_VEC2;
						else if (typeStream == "VEC3")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_VEC3;
						else if (typeStream == "VEC4")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_VEC4;
						else if (typeStream == "MAT2")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_MAT2;
						else if (typeStream == "MAT3")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_MAT3;
						else if (typeStream == "MAT4")
							glTFAccessor.type = SGLTF::SGLTFAccessor::SGLTFT_MAT4;
						else
						{
							context.loadContext.params.logger.log("GLTF: DETECTED UNSUPPORTED TYPE!", system::ILogger::ELL_WARNING);
							return false;
						}
					}

					if (max.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						glTFAccessor.max.emplace();
						const auto& maxArray = max.get_array();
						for (uint32_t i = 0; i < maxArray.size(); ++i)
							glTFAccessor.max.value().push_back(maxArray.at(i).get_double().value());
					}

					if (min.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						glTFAccessor.min.emplace();
						const auto& minArray = min.get_array();
						for (uint32_t i = 0; i < minArray.size(); ++i)
							glTFAccessor.min.value().push_back(minArray.at(i).get_double().value());
					}

					//if (sparse.error() != simdjson::error_code::NO_SUCH_FIELD)
					//	glTFAccessor.sparse = ; //! TODO: in future

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.name = count.get_string().value();

					/*if (!glTFAccessor.validate())
						return false;*/ // TODO!
				}
			}

			if (meshes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& mData = meshes.get_array();
				for (const auto& mesh : mData)
				{
					auto& glTFMesh = glTF.meshes.emplace_back();

					const auto& primitives = mesh.at_key("primitives");
					const auto& weights = mesh.at_key("weights");
					const auto& name = mesh.at_key("name");
					const auto& extensions = mesh.at_key("extensions");
					const auto& extras = mesh.at_key("extras");

					if (primitives.error() == simdjson::error_code::NO_SUCH_FIELD)
					{
						context.loadContext.params.logger.log("GLTF: COULD NOT DETECT ANY PRIMITIVE!", system::ILogger::ELL_WARNING);
						return false;
					}

					const auto& pData = primitives.get_array();
					for (const auto& primitive : pData)
					{
						auto& glTFPrimitive = glTFMesh.primitives.emplace_back();
						
						const auto& attributes = primitive.at_key("attributes");
						const auto& indices = primitive.at_key("indices");
						const auto& material = primitive.at_key("material");
						const auto& mode = primitive.at_key("mode");
						const auto& targets = primitive.at_key("targets");
						const auto& extensions = primitive.at_key("extensions");
						const auto& extras = primitive.at_key("extras");

						if (indices.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFPrimitive.indices = indices.get_uint64().value();

						if (material.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFPrimitive.material = material.get_uint64().value();

						if (mode.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFPrimitive.mode = mode.get_uint64().value();
						else
							glTFPrimitive.mode = 4;

						if (targets.error() != simdjson::error_code::NO_SUCH_FIELD)
							for (const auto& [targetKey, targetID] : targets.get_object())
								glTFPrimitive.targets.emplace()[targetKey.data()] = targetID.get_uint64().value();

						if (attributes.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							for (const auto& [attributeKey, accessorID] : attributes.get_object())
							{
								const auto& requestedAccessor = accessorID.get_uint64().value();

								std::pair<std::string, uint8_t> attributeMap;
								{
									const std::string key = attributeKey.data();
									auto foundIndexAttribute = key.find_last_of("_");

									if (foundIndexAttribute != std::string::npos)
										attributeMap = std::make_pair(key.substr(0, foundIndexAttribute), std::stoi(key.substr(foundIndexAttribute + 1)));
									else
										attributeMap = std::make_pair(key, 0);
								}

								if (attributeMap.first == "POSITION")
									glTFPrimitive.attributes.position = requestedAccessor;
								else if (attributeMap.first == "NORMAL")
									glTFPrimitive.attributes.normal = requestedAccessor;
								else if (attributeMap.first == "TANGENT")
									glTFPrimitive.attributes.tangent = requestedAccessor;
								else if (attributeMap.first == "TEXCOORD")
								{
									if (attributeMap.second >= 1u)
									{
										context.loadContext.params.logger.log("GLTF: LOADER DOESN'T SUPPORT MULTIPLE UV ATTRIBUTES!", system::ILogger::ELL_WARNING);
										return false;
									}

									glTFPrimitive.attributes.texcoord = requestedAccessor;
								}
								else if (attributeMap.first == "COLOR")
								{
									if (attributeMap.second >= 1u)
									{
										context.loadContext.params.logger.log("GLTF: LOADER DOESN'T SUPPORT MULTIPLE COLOR ATTRIBUTES!", system::ILogger::ELL_WARNING);
										return false;
									}

									glTFPrimitive.attributes.color = requestedAccessor;
								}
								else if (attributeMap.first == "JOINTS")
								{
									if (attributeMap.second >= glTFPrimitive.attributes.MAX_JOINTS_ATTRIBUTES)
									{
										context.loadContext.params.logger.log("GLTF: EXCEEDED 'MAX_JOINTS_ATTRIBUTES' FOR JOINTS ATTRIBUTES!", system::ILogger::ELL_WARNING);
										return false;
									}

									glTFPrimitive.attributes.joints[attributeMap.second] = requestedAccessor;
								}
								else if (attributeMap.first == "WEIGHTS")
								{
									if (attributeMap.second >= glTFPrimitive.attributes.MAX_WEIGHTS_ATTRIBUTES)
									{
										context.loadContext.params.logger.log("GLTF: EXCEEDED 'MAX_WEIGHTS_ATTRIBUTES' FOR JOINTS ATTRIBUTES!", system::ILogger::ELL_WARNING);
										return false;
									}

									glTFPrimitive.attributes.weights[attributeMap.second] = requestedAccessor;
								}
							}
						}
					}

					//! Array of weights to be applied to the Morph Targets.
					// weights - TODO in future

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFMesh.name = name.get_string().value();
				}
			}

			if (nodes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& nData = nodes.get_array();
				for (size_t iteratorID = 0; iteratorID < nData.size(); ++iteratorID)
				{
					// TODO: fill the node and get down through the tree (mesh, primitives, attributes, buffer views, materials, etc) till the end.

					auto handleTheGLTFTree = [&]()
					{
						auto& glTFnode = glTF.nodes.emplace_back();
						const auto& jsonNode = nData.at(iteratorID);

						const auto& camera = jsonNode.at_key("camera");
						const auto& children = jsonNode.at_key("children");
						const auto& skin = jsonNode.at_key("skin");
						const auto& matrix = jsonNode.at_key("matrix");
						const auto& mesh = jsonNode.at_key("mesh");
						const auto& rotation = jsonNode.at_key("rotation");
						const auto& scale = jsonNode.at_key("scale");
						const auto& translation = jsonNode.at_key("translation");
						const auto& weights = jsonNode.at_key("weights");
						const auto& name = jsonNode.at_key("name");
						const auto& extensions = jsonNode.at_key("extensions");
						const auto& extras = jsonNode.at_key("extras");

						if (camera.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.camera = camera.get_uint64().value();

						if (children.error() != simdjson::error_code::NO_SUCH_FIELD)
							for (const auto& child : children)
								glTFnode.children.push_back(child.get_uint64().value());

						if (skin.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.skin = skin.get_uint64().value();

						if (matrix.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							const auto& matrixArray = matrix.get_array();
							core::matrix4SIMD tmpMatrix;

							for (uint32_t i = 0; i < matrixArray.size(); ++i)
								*(tmpMatrix.pointer() + i) = matrixArray.at(i).get_double().value();

							// TODO tmpMatrix (coulmn major) to row major (currentNode.matrix)

							glTFnode.transformation.matrix = tmpMatrix.extractSub3x4();
						}
						else
						{
							struct SGLTFNTransformationTRS
							{
								core::vector3df_SIMD translation;			//!< The node's translation along the x, y, and z axes.
								core::vector3df_SIMD scale;					//!< The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
								core::vector4df_SIMD rotation;				//!< The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
							} trs;

							if (translation.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const auto& translationArray = translation.get_array();

								size_t index = {};
								for (const auto& val : translationArray)
									trs.translation[index++] = val.get_double().value();
							}

							if (rotation.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const auto& rotationArray = rotation.get_array();

								size_t index = {};
								for (const auto& val : rotationArray)
									trs.rotation[index++] = val.get_double().value();
							}

							if (scale.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const auto& scaleArray = scale.get_array();

								size_t index = {};
								for (const auto& val : scaleArray)
									trs.scale[index++] = val.get_double().value();
							}

							core::quaternion quaterion = core::quaternion(trs.rotation.x, trs.rotation.y, trs.rotation.z, trs.rotation.w);
							glTFnode.transformation.matrix.setScaleRotationAndTranslation(trs.scale, quaterion, trs.translation);
						}

						if (mesh.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.mesh = mesh.get_uint64().value();

						if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.name = name.get_string().value();

						// TODO camera, skinning, etc HERE

						return glTFnode.validate();
					};

					if (!handleTheGLTFTree()) //! TODO more validations in future for glTF objects
					{
						context.loadContext.params.logger.log("GLTF: NODE VALIDATION FAILED!", system::ILogger::ELL_WARNING);
						return false;
					}
				}
			}

			return true;
		}

		core::smart_refctd_ptr<ICPUPipelineLayout> CGLTFLoader::makePipelineLayoutFromGLTF(SContext& context, CGLTFPipelineMetadata::SGLTFMaterialParameters& pushConstants, SMaterialDependencyData& materialData)
		{
			/*
				Assumes all supported textures are always present
				since vulkan doesnt support bindings with no/null descriptor,
				absent textures are filled with dummy 2D texture (while creating descriptor set)
			*/

			std::array<core::smart_refctd_ptr<ICPUImageView>, SGLTF::SGLTFMaterial::EGT_COUNT> IMAGE_VIEWS;
			{
				auto default_imageview_bundle = assetManager->getAsset("nbl/builtin/image_view/dummy2d", context.loadContext.params);
				const bool status = !default_imageview_bundle.getContents().empty();
				assert(status);

				auto cpuDummyImageView = core::smart_refctd_ptr_static_cast<ICPUImageView>(default_imageview_bundle.getContents().begin()[0]);
				for(auto& imageView : IMAGE_VIEWS)
					imageView = core::smart_refctd_ptr(cpuDummyImageView);
			}

			std::array<core::smart_refctd_ptr<ICPUSampler>, SGLTF::SGLTFMaterial::EGT_COUNT> SAMPLERS;
			{
				for (auto& sampler : SAMPLERS)
					sampler = getDefaultAsset<ICPUSampler, IAsset::ET_SAMPLER>("nbl/builtin/sampler/default", assetManager);
			}

			auto getCpuDs3Layout = [&]() -> core::smart_refctd_ptr<ICPUDescriptorSetLayout>
			{
				//! Samplers
				_NBL_STATIC_INLINE_CONSTEXPR size_t samplerBindingsAmount = SGLTF::SGLTFMaterial::EGT_COUNT;
				auto cpuDS3Bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSetLayout::SBinding>>(samplerBindingsAmount);
				{
					ICPUDescriptorSetLayout::SBinding cpuSamplerBinding;
					cpuSamplerBinding.count = 1u;
					cpuSamplerBinding.stageFlags = ICPUSpecializedShader::ESS_FRAGMENT;
					cpuSamplerBinding.type = EDT_COMBINED_IMAGE_SAMPLER;
					std::fill(cpuDS3Bindings->begin(), cpuDS3Bindings->end(), cpuSamplerBinding);
				}

				if (materialData.glTFMaterial)
				{
					auto& material = materialData;

					auto fillAssets = [&](uint32_t globalTextureIndex, SGLTF::SGLTFMaterial::E_GLTF_TEXTURES localTextureIndex)
					{
						auto& [cpuImageView, cpuSamplerCacheKey] = (*material.cpuTextures)[globalTextureIndex];

						IMAGE_VIEWS[localTextureIndex] = cpuImageView;

						const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_SAMPLER, (asset::IAsset::E_TYPE)0u };
						auto sampler_bundle = context.loaderOverride->findCachedAsset(cpuSamplerCacheKey, types, context.loadContext, context.hierarchyLevel /*TODO + what here?*/);

						if (!sampler_bundle.getContents().empty())
							SAMPLERS[localTextureIndex] = core::smart_refctd_ptr_static_cast<ICPUSampler>(sampler_bundle.getContents().begin()[0]);
					};

					if (material.glTFMaterial->pbrMetallicRoughness.has_value())
					{
						auto& pbrMetallicRoughness = material.glTFMaterial->pbrMetallicRoughness.value();
						{
							if (pbrMetallicRoughness.baseColorTexture.has_value())
							{
								auto& baseColorTexture = pbrMetallicRoughness.baseColorTexture.value();

								if (baseColorTexture.index.has_value())
								{
									fillAssets(baseColorTexture.index.value(), SGLTF::SGLTFMaterial::EGT_BASE_COLOR_TEXTURE);
									pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_BASE_COLOR_TEXTURE;
								}
									
								/*
									if (baseColorTexture.texCoord.has_value())
									{
										;// TODO: the default is 0, but in no-0 value is present it is a relation between UV attribute with unique ID which is texCoord ID, so UV_<texCoord>
									}
								*/
							}

							if (pbrMetallicRoughness.metallicRoughnessTexture.has_value())
							{
								auto& metallicRoughnessTexture = pbrMetallicRoughness.metallicRoughnessTexture.value();

								if (metallicRoughnessTexture.index.has_value())
								{
									fillAssets(metallicRoughnessTexture.index.value(), SGLTF::SGLTFMaterial::EGT_METALLIC_ROUGHNESS_TEXTURE);
									pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_METALLIC_ROUGHNESS_TEXTURE;
								}
								
								/*
									if (metallicRoughnessTexture.texCoord.has_value())
									{
										;// TODO: the default is 0, but in no-0 value is present it is a relation between UV attribute with unique ID which is texCoord ID, so UV_<texCoord>
									}
								*/
							}
						}
					}

					if (material.glTFMaterial->normalTexture.has_value())
					{
						auto& normalTexture = material.glTFMaterial->normalTexture.value();

						if (normalTexture.index.has_value())
						{
							fillAssets(normalTexture.index.value(), SGLTF::SGLTFMaterial::EGT_NORMAL_TEXTURE);
							pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_NORMAL_TEXTURE;

							auto cpuNormalTexture = IMAGE_VIEWS[SGLTF::SGLTFMaterial::EGT_NORMAL_TEXTURE];
							IMAGE_VIEWS[SGLTF::SGLTFMaterial::EGT_NORMAL_TEXTURE] = CDerivativeMapCreator::createDerivativeMapViewFromNormalMap(cpuNormalTexture->getCreationParameters().image.get());
							
							// fetch from cpuDerivativeNormalTexture scale using meta
							// const auto& absLayerScaleValues = state.getAbsoluteLayerScaleValue(0);

							/*
								if (normalTexture.texCoord.has_value())
								{
									;// TODO: the default is 0, but in no-0 value is present it is a relation between UV attribute with unique ID which is texCoord ID, so UV_<texCoord>
								}
							*/
						}
					}

					if (material.glTFMaterial->occlusionTexture.has_value())
					{
						auto& occlusionTexture = material.glTFMaterial->occlusionTexture.value();

						if (occlusionTexture.index.has_value())
						{
							fillAssets(occlusionTexture.index.value(), SGLTF::SGLTFMaterial::EGT_OCCLUSION_TEXTURE);
							pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_OCCLUSION_TEXTURE;
						}
						
						/*
							if (occlusionTexture.texCoord.has_value())
							{
								;// TODO: the default is 0, but in no-0 value is present it is a relation between UV attribute with unique ID which is texCoord ID, so UV_<texCoord>
							}
						*/
					}

					if (material.glTFMaterial->emissiveTexture.has_value())
					{
						auto& emissiveTexture = material.glTFMaterial->emissiveTexture.value();

						if (emissiveTexture.index.has_value())
						{
							fillAssets(emissiveTexture.index.value(), SGLTF::SGLTFMaterial::EGT_EMISSIVE_TEXTURE);
							pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_EMISSIVE_TEXTURE;
						}
						
						/*
							if (emissiveTexture.texCoord.has_value())
							{
								;// TODO: the default is 0, but in no-0 value is present it is a relation between UV attribute with unique ID which is texCoord ID, so UV_<texCoord>
							}
						*/
					}
				}

				for (uint32_t i = 0u; i < samplerBindingsAmount; ++i)
				{
					(*cpuDS3Bindings)[i].binding = i;
					(*cpuDS3Bindings)[i].samplers = SAMPLERS.data() + i;
				}

				return core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(cpuDS3Bindings->begin(), cpuDS3Bindings->end());
			};

			//! camera UBO DS
			auto cpuDs1Layout = getDefaultAsset<ICPUDescriptorSetLayout, IAsset::ET_DESCRIPTOR_SET_LAYOUT>("nbl/builtin/descriptor_set_layout/basic_view_parameters", assetManager);		
			
			//! samplers and skinMatrices DS
			auto cpuDs3Layout = getCpuDs3Layout();
			
			auto cpuDescriptorSet3 = makeAndGetDS3set(IMAGE_VIEWS, cpuDs3Layout); 
			materialData.cpuMeshBuffer->setAttachedDescriptorSet(std::move(cpuDescriptorSet3));

			constexpr uint32_t PUSH_CONSTANTS_COUNT = 1u;
			asset::SPushConstantRange pushConstantRange[PUSH_CONSTANTS_COUNT]; 
			pushConstantRange[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
			pushConstantRange[0].offset = 0u;
			pushConstantRange[0].size = sizeof(CGLTFPipelineMetadata::SGLTFMaterialParameters);

			auto cpuPipelineLayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(pushConstantRange, pushConstantRange + PUSH_CONSTANTS_COUNT, nullptr, std::move(cpuDs1Layout), nullptr, std::move(cpuDs3Layout));
			return cpuPipelineLayout;
		}
		
		core::smart_refctd_ptr<ICPUDescriptorSet> CGLTFLoader::makeAndGetDS3set(std::array<core::smart_refctd_ptr<ICPUImageView>, SGLTF::SGLTFMaterial::EGT_COUNT>& cpuImageViews, core::smart_refctd_ptr<ICPUDescriptorSetLayout> cpuDescriptorSet3Layout)
		{
			auto cpuDescriptorSet3 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(core::smart_refctd_ptr<ICPUDescriptorSetLayout>(cpuDescriptorSet3Layout));
			
			for (uint16_t i = 0; i < SGLTF::SGLTFMaterial::EGT_COUNT; ++i)
			{
				auto cpuDescriptor = cpuDescriptorSet3->getDescriptors(i).begin();

				cpuDescriptor->desc = cpuImageViews[i];
				cpuDescriptor->image.imageLayout = EIL_UNDEFINED;
				cpuDescriptor->image.sampler = nullptr; //! Not needed, immutable (in DescriptorSet layout) samplers are used
			}

			return cpuDescriptorSet3;
		}
	}
}

#endif // _NBL_COMPILE_WITH_GLTF_LOADER_
