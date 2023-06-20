// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in Nabla.h

#include "CGLTFLoader.h"

#ifdef _NBL_COMPILE_WITH_GLTF_LOADER_

#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/utils/CDerivativeMapCreator.h"
#include "nbl/asset/utils/IMeshManipulator.h"

#include "simdjson/singleheader/simdjson.h"
#include <algorithm>

#include "nbl/core/execution.h"

using namespace nbl;
using namespace nbl::asset;

		enum WEIGHT_ENCODING
		{
			WE_UNORM8,
			WE_UNORM16,
			WE_SFLOAT,
			WE_COUNT
		};

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
			auto registerShader = [&](auto constexprStringType, IShader::E_SHADER_STAGE stage, const char* extraDefine=nullptr) -> void
			{
				auto fileSystem = assetManager->getSystem();

				auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
				{
					nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
					fileSystem->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
					if (future.wait())
						return future.copy();
					return nullptr;
				};

				core::smart_refctd_ptr<const system::IFile> glslFile = loadBuiltinData(decltype(constexprStringType)::value);
				auto glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
				memcpy(glsl->getPointer(),glslFile->getMappedPointer(),glsl->getSize());

				auto unspecializedShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), stage, asset::ICPUShader::E_CONTENT_TYPE::ECT_GLSL, stage != ICPUShader::ESS_VERTEX ? "?IrrlichtBAW glTFLoader FragmentShader?" : "?IrrlichtBAW glTFLoader VertexShader?");
				if (extraDefine)
					unspecializedShader = CGLSLCompiler::createOverridenCopy(unspecializedShader.get(),"%s",extraDefine);

				ICPUSpecializedShader::SInfo specInfo({},nullptr,"main");
				auto shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedShader),std::move(specInfo));

				SAssetBundle bundle(nullptr,{std::move(shader)});
				assetManager->changeAssetKey(bundle,decltype(constexprStringType)::value);
				assetManager->insertAssetIntoCache(bundle,IAsset::EM_IMMUTABLE);
			};

			/*
				The lambda registers either static
				and skinned version of the shader
			*/
			registerShader(VertexShaderUVCacheKey(),IShader::ESS_VERTEX);
			registerShader(VertexShaderColorCacheKey(),IShader::ESS_VERTEX);
			registerShader(VertexShaderNoUVColorCacheKey(),IShader::ESS_VERTEX);

			registerShader(VertexShaderSkinnedUVCacheKey(),IShader::ESS_VERTEX,"#define _NBL_SKINNING_ENABLED_\n");
			registerShader(VertexShaderSkinnedColorCacheKey(),IShader::ESS_VERTEX,"#define _NBL_SKINNING_ENABLED_\n");
			registerShader(VertexShaderSkinnedNoUVColorCacheKey(),IShader::ESS_VERTEX,"#define _NBL_SKINNING_ENABLED_\n");

			registerShader(FragmentShaderUVCacheKey(),IShader::ESS_FRAGMENT);
			registerShader(FragmentShaderColorCacheKey(),IShader::ESS_FRAGMENT);
			registerShader(FragmentShaderNoUVColorCacheKey(),IShader::ESS_FRAGMENT);

			
			//! texture DS
			ICPUDescriptorSetLayout::SBinding combinedSamplerBindings[SGLTF::SGLTFMaterial::EGT_COUNT];
			for (auto i=0u; i<SGLTF::SGLTFMaterial::EGT_COUNT; i++)
			{
				combinedSamplerBindings[i].binding = i;
				combinedSamplerBindings[i].type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
				combinedSamplerBindings[i].count = 1u;
				combinedSamplerBindings[i].stageFlags = IShader::ESS_FRAGMENT;
				combinedSamplerBindings[i].samplers = nullptr;
			}
			SAssetBundle bundle(nullptr,{core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(combinedSamplerBindings,combinedSamplerBindings+SGLTF::SGLTFMaterial::EGT_COUNT)});
			assetManager->changeAssetKey(bundle,DescriptorSetLayoutCacheKey);
			assetManager->insertAssetIntoCache(bundle,IAsset::EM_IMMUTABLE);
		}

		void CGLTFLoader::initialize()
		{
			IRenderpassIndependentPipelineLoader::initialize();
		}
		
		bool CGLTFLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
		{
			simdjson::dom::parser parser;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				system::IFile::success_t success;
				_file->read(success, jsonBuffer->getPointer(), 0u, jsonBuffer->getSize());
				if (!success)
					return false;
			}

			simdjson::dom::object tweets;
			auto error = parser.parse(reinterpret_cast<uint8_t*>(jsonBuffer->getPointer()), jsonBuffer->getSize()).get(tweets);

			if (error)
			{
				logger.log("Could not parse '" + _file->getFileName().string() + "' file!");
				return false;
			}

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
				@devsh Probably works now.
			*/
			SContext context(overrideAssetLoadParams, _file, _override, _hierarchyLevel);

			SGLTF glTF;
			if(!loadAndGetGLTF(glTF, context))
				return {};

			core::vector<core::smart_refctd_ptr<ICPUBuffer>> cpuBuffers;
			for (auto& glTFBuffer : glTF.buffers)
			{
				// FarFuture TODO: handle buffer embedded in glTF
				auto buffer_bundle = interm_getAssetInHierarchy(assetManager,glTFBuffer.uri.value(),context.loadContext.params,_hierarchyLevel+ICPUMesh::BUFFER_HIERARCHYLEVELS_BELOW,_override);
				if (buffer_bundle.getContents().empty())
					return {};

				auto cpuBuffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer_bundle.getContents().begin()[0]);
				cpuBuffers.emplace_back() = core::smart_refctd_ptr<ICPUBuffer>(cpuBuffer);
			}

			const auto imageViewHierarchyLevel = _hierarchyLevel+ICPUMesh::IMAGEVIEW_HIERARCHYLEVELS_BELOW;
			core::vector<core::smart_refctd_ptr<ICPUImageView>> cpuImageViews;
			{
				for (auto& glTFImage : glTF.images)
				{
					auto& cpuImageView = cpuImageViews.emplace_back();

					// FarFuture TODO: handle image embedded in glTF 
					// TODO: factor this out to be common for all PipelineLoaders https://github.com/Devsh-Graphics-Programming/Nabla/issues/270
					if (glTFImage.uri.has_value())
					{
						// TODO: THIS IS AN ABSOLUTELY WRONG CACHE PRE-PATH KEY TO USE!
						const std::string cpuImageViewCacheKey = getImageViewCacheKey(glTFImage.uri.value());

						cpuImageView = _override->findDefaultAsset<ICPUImageView>(cpuImageViewCacheKey,context.loadContext,imageViewHierarchyLevel).first;
						if (!cpuImageView)
						{
							auto image_bundle = interm_getAssetInHierarchy(assetManager,glTFImage.uri.value(),context.loadContext.params,imageViewHierarchyLevel,_override);
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
									context.loadContext.params.logger.log("GLTF: EXPECTED IMAGE ASSET TYPE!",system::ILogger::ELL_ERROR);
									return {};
								}
							}

							// TODO: this is wrong, it adds a loaded image view (the second switch case) to the cache again, move this insertion to the first switch case
							SAssetBundle samplerBundle = SAssetBundle(nullptr, { core::smart_refctd_ptr(cpuImageView) });
							_override->insertAssetIntoCache(samplerBundle,cpuImageViewCacheKey,context.loadContext,imageViewHierarchyLevel);
						}	
					}
					else
					{
						if (!glTFImage.mimeType.has_value() || !glTFImage.bufferView.has_value())
							return {};
						
						_NBL_DEBUG_BREAK_IF(true);
						return {}; // TODO FUTURE: load image where it's data is embeded in memory
					}
				}
			}
			
			core::vector<std::pair<core::smart_refctd_ptr<ICPUImageView>,core::smart_refctd_ptr<ICPUSampler>>> cpuTextures;
			const auto samplerHierarchyLevel = _hierarchyLevel+ICPUMesh::SAMPLER_HIERARCHYLEVELS_BELOW;
			for (auto& glTFTexture : glTF.textures)
			{
				auto& [imageView,sampler] = cpuTextures.emplace_back();
				if (glTFTexture.source.has_value())
					imageView = cpuImageViews[glTFTexture.source.value()];
				else
					imageView = _override->findDefaultAsset<ICPUImageView>("nbl/builtin/image_view/dummy2d",context.loadContext,imageViewHierarchyLevel).first;
				if (glTFTexture.sampler.has_value())
				{
					ICPUSampler::SParams samplerParams;
					using SGLTFSampler = SGLTF::SGLTFSampler;
					const auto& glTFSampler = glTF.samplers[glTFTexture.sampler.value()];
					switch (glTFSampler.magFilter)
					{
						case SGLTFSampler::STP_NEAREST:
							samplerParams.MaxFilter = ISampler::ETF_NEAREST;
							break;
						case SGLTFSampler::STP_LINEAR:
							samplerParams.MaxFilter = ISampler::ETF_LINEAR;
							break;
					}
					switch (glTFSampler.minFilter)
					{
						case SGLTFSampler::STP_NEAREST:
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
							break;
						case SGLTFSampler::STP_LINEAR:
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
							break;
						case SGLTFSampler::STP_NEAREST_MIPMAP_NEAREST:
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
							samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
							break;
						case SGLTFSampler::STP_LINEAR_MIPMAP_NEAREST:
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
							samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
							break;
						case SGLTFSampler::STP_NEAREST_MIPMAP_LINEAR:
							samplerParams.MinFilter = ISampler::ETF_NEAREST;
							samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
							break;
						case SGLTFSampler::STP_LINEAR_MIPMAP_LINEAR:
							samplerParams.MinFilter = ISampler::ETF_LINEAR;
							samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
							break;
					}
					switch (glTFSampler.wrapS)
					{
						case SGLTFSampler::STP_CLAMP_TO_EDGE:
							samplerParams.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
							break;
						case SGLTFSampler::STP_MIRRORED_REPEAT:
							samplerParams.TextureWrapU = ISampler::ETC_MIRROR;
							break;
						case SGLTFSampler::STP_REPEAT:
							samplerParams.TextureWrapU = ISampler::ETC_REPEAT;
							break;
					}
					switch (glTFSampler.wrapT)
					{
						case SGLTFSampler::STP_CLAMP_TO_EDGE:
							samplerParams.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
							break;
						case SGLTFSampler::STP_MIRRORED_REPEAT:
							samplerParams.TextureWrapV = ISampler::ETC_MIRROR;
							break;
						case SGLTFSampler::STP_REPEAT:
							samplerParams.TextureWrapV = ISampler::ETC_REPEAT;
							break;
					}
					sampler = getSampler(std::move(samplerParams),context.loadContext,_override);
				}
				else
					sampler = _override->findDefaultAsset<ICPUSampler>("nbl/builtin/sampler/default",context.loadContext,samplerHierarchyLevel).first;
			}

			// materials
			struct Material
			{
				CGLTFPipelineMetadata::SGLTFMaterialParameters pushConstants = {};
				core::smart_refctd_ptr<ICPUDescriptorSet> descriptorSet;
			};
			/*
				Assumes all supported textures are always present
				since vulkan doesnt support bindings with no/null descriptor,
				absent textures are filled with dummy 2D texture (while creating descriptor set)
			*/
			core::vector<Material> materials;
			for (auto i=0u; i<glTF.materials.size(); i++)
			{
				const auto& glTFMaterial = glTF.materials[i];
				auto& material = materials.emplace_back();
	
				material.descriptorSet = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(getDescriptorSetLayout(context));
				auto defaultImageView = _override->findDefaultAsset<ICPUImageView>("nbl/builtin/image_view/dummy2d",context.loadContext,0u).first;
				auto defaultSampler = _override->findDefaultAsset<ICPUSampler>("nbl/builtin/sampler/default",context.loadContext,0u).first;
				for (uint16_t i=0u; i<SGLTF::SGLTFMaterial::EGT_COUNT; ++i)
				{
					auto descriptorInfos = material.descriptorSet->getDescriptorInfos(i, IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
					descriptorInfos.begin()[0].desc = defaultImageView;
					descriptorInfos.begin()[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
					descriptorInfos.begin()[0].info.image.sampler = defaultSampler;
				}
				auto setImage = [&cpuTextures,&material](uint32_t globalTextureIndex, SGLTF::SGLTFMaterial::E_GLTF_TEXTURES localTextureIndex)
				{
					const auto& [imageView,sampler] = cpuTextures[globalTextureIndex];

					auto descriptorInfos = material.descriptorSet->getDescriptorInfos(localTextureIndex, IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
					descriptorInfos.begin()[0].desc = imageView;
					descriptorInfos.begin()[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
					descriptorInfos.begin()[0].info.image.sampler = sampler;
				};

				auto& pushConstants = material.pushConstants;
				if (glTFMaterial.pbrMetallicRoughness.has_value())
				{
					auto& pbrMetallicRoughness = glTFMaterial.pbrMetallicRoughness.value();
									
					if (pbrMetallicRoughness.baseColorTexture.has_value() && pbrMetallicRoughness.baseColorTexture.value().index.has_value())
					{
						pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_BASE_COLOR_TEXTURE;
						setImage(pbrMetallicRoughness.baseColorTexture.value().index.value(),SGLTF::SGLTFMaterial::EGT_BASE_COLOR_TEXTURE);
					}

					if (pbrMetallicRoughness.baseColorFactor.has_value())
					for (uint8_t i=0u; i<pbrMetallicRoughness.baseColorFactor.value().size(); ++i)
						pushConstants.metallicRoughness.baseColorFactor[i] = pbrMetallicRoughness.baseColorFactor.value()[i];

					if (pbrMetallicRoughness.metallicRoughnessTexture.has_value() && pbrMetallicRoughness.metallicRoughnessTexture.value().index.has_value())
					{
						pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_METALLIC_ROUGHNESS_TEXTURE;
						setImage(pbrMetallicRoughness.metallicRoughnessTexture.value().index.value(),SGLTF::SGLTFMaterial::EGT_METALLIC_ROUGHNESS_TEXTURE);
					}
	
					if (pbrMetallicRoughness.metallicFactor.has_value())
						pushConstants.metallicRoughness.metallicFactor = pbrMetallicRoughness.metallicFactor.value();

					if (pbrMetallicRoughness.roughnessFactor.has_value())
						pushConstants.metallicRoughness.roughnessFactor = pbrMetallicRoughness.roughnessFactor.value();
				}
				if (glTFMaterial.normalTexture.has_value() && glTFMaterial.normalTexture.value().index.has_value())
				{
					pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_NORMAL_TEXTURE;
					const auto normalTextureID = glTFMaterial.normalTexture.value().index.value();
					// TODO: CACHE THIS FFS!!!
					float scales[2] = {};
					auto imageView = CDerivativeMapCreator::createDerivativeMapViewFromNormalMap<false>(cpuTextures[normalTextureID].first->getCreationParameters().image.get(), scales);
					auto& sampler = cpuTextures[normalTextureID].second;

					auto descriptorInfos = material.descriptorSet->getDescriptorInfos(CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_NORMAL_TEXTURE, IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
					descriptorInfos.begin()[0].desc = imageView;
					descriptorInfos.begin()[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
					descriptorInfos.begin()[0].info.image.sampler = sampler;
				}
				if (glTFMaterial.occlusionTexture.has_value() && glTFMaterial.occlusionTexture.value().index.has_value())
				{
					pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_OCCLUSION_TEXTURE;
					setImage(glTFMaterial.occlusionTexture.value().index.value(),SGLTF::SGLTFMaterial::EGT_OCCLUSION_TEXTURE);
				}
				if (glTFMaterial.emissiveTexture.has_value() && glTFMaterial.emissiveTexture.value().index.has_value())
				{
					pushConstants.availableTextures |= CGLTFPipelineMetadata::SGLTFMaterialParameters::EGT_EMISSIVE_TEXTURE;
					setImage(glTFMaterial.emissiveTexture.value().index.value(),SGLTF::SGLTFMaterial::EGT_EMISSIVE_TEXTURE);
				}

				// Far TODO: you only need alphaCutOff in the shader push constants (not blend modes)
				pushConstants.alphaCutoff = 0.f;
				if (glTFMaterial.alphaCutoff.has_value())
					pushConstants.alphaCutoff = glTFMaterial.alphaCutoff.value();

				// Far TODO: set the blend modes in the pipeline instead
				CGLTFLoader::SGLTF::SGLTFMaterial::E_ALPHA_MODE alphaModeStream = decltype(alphaModeStream)::EAM_OPAQUE;
				if (glTFMaterial.alphaMode.has_value())
					alphaModeStream = glTFMaterial.alphaMode.value();
				switch (alphaModeStream)
				{
					case decltype(alphaModeStream)::EAM_OPAQUE:
						pushConstants.alphaMode = asset::CGLTFPipelineMetadata::EAM_OPAQUE;
						break;
					case decltype(alphaModeStream)::EAM_MASK:
						pushConstants.alphaMode = asset::CGLTFPipelineMetadata::EAM_MASK;
						break;
					case decltype(alphaModeStream)::EAM_BLEND:
						pushConstants.alphaMode = asset::CGLTFPipelineMetadata::EAM_BLEND;
						break;
				}

				if (glTFMaterial.emissiveFactor.has_value())
				for (uint8_t i=0u; i<glTFMaterial.emissiveFactor.value().size(); ++i)
					pushConstants.emissiveFactor[i] = glTFMaterial.emissiveFactor.value()[i];
			}
			
			core::smart_refctd_ptr<ICPUBuffer> vertexJointToSkeletonJoint,inverseBindPose;
			// cache
			struct MeshSkinPair
			{
				inline bool operator==(const MeshSkinPair& other) const = default;

				uint32_t mesh;
				uint32_t skin;
			};
			struct MeshSkinPairHash
			{
				inline size_t operator()(const MeshSkinPair& p) const
				{
					uint64_t val = p.skin;
					val = (val<<32ull)|p.mesh;
					return std::hash<uint64_t>()(val);
				}
			};
			core::unordered_set<MeshSkinPair,MeshSkinPairHash> meshSkinPairs;
			struct Skin
			{
				core::smart_refctd_ptr<ICPUSkeleton> skeleton = {};
				SBufferRange<ICPUBuffer> translationTable = {};
				SBufferRange<ICPUBuffer> inverseBindPose = {};
				ICPUSkeleton::joint_id_t root = ICPUSkeleton::invalid_joint_id;
				uint16_t jointCount;
			};
			core::vector<Skin> skins(glTF.skins.size());

			//
			const uint32_t nodeCount = glTF.nodes.size();
			struct SkeletonData
			{
				core::matrix3x4SIMD defaultNodeTransform; // _defaultTransforms
				std::string glTFNodeName;

				uint32_t skeletonID;
				uint32_t localJointID; // in range [0, jointsSize - 1]
				uint32_t localParentJointID; // for _parentJointIDsBinding

				uint32_t instanceID;
			};
			core::vector<core::smart_refctd_ptr<ICPUSkeleton>> skeletons;
			core::vector<SkeletonData> skeletonNodes(nodeCount);
			{
				core::vector<ICPUSkeleton::joint_id_t> globalParent(nodeCount,ICPUSkeleton::invalid_joint_id);
				// first pass over the nodes
				{
					core::vector<uint32_t> skeletonJointCount;
					// first flag parents and gather unique <mesh,skin> instances
					{
						for (uint32_t index=0u; index<nodeCount; ++index)
						{
							const auto& glTFnode = glTF.nodes[index];
							for (auto child : glTFnode.children)
								globalParent[child] = index;
							skeletonNodes[index].defaultNodeTransform = glTFnode.transformation.matrix;
							skeletonNodes[index].glTFNodeName = glTFnode.name.has_value() ? glTFnode.name.value():"NBL_IDENTITY";

							const uint32_t meshID = glTFnode.mesh.has_value() ? glTFnode.mesh.value() : 0xdeadbeef;
							if (meshID == 0xdeadbeef)
								continue;

							const uint32_t skinID = glTFnode.skin.has_value() ? glTFnode.skin.value() : 0xdeadbeef;
							meshSkinPairs.insert({ meshID,skinID });
						}
						// then figure out the remapping to ICPUSkeleton Nodes
						core::stack<uint32_t> dfsTraversalStack;
						for (uint32_t index=0u; index<nodeCount; ++index)
						{
							if (globalParent[index]==ICPUSkeleton::invalid_joint_id)
							{
								const uint32_t skeletonID = skeletonJointCount.size();
								uint32_t& jointID = skeletonJointCount.emplace_back() = 0u;
								//
								auto addNodeToSkeleton = [skeletonID,&jointID,&skeletonNodes,&globalParent,&glTF,&dfsTraversalStack](const uint32_t globalNodeID) -> void
								{
									const auto parent = globalParent[globalNodeID];

									skeletonNodes[globalNodeID].skeletonID = skeletonID;
									skeletonNodes[globalNodeID].localJointID = jointID++;
 									skeletonNodes[globalNodeID].localParentJointID = parent!=ICPUSkeleton::invalid_joint_id ? skeletonNodes[parent].localJointID:ICPUSkeleton::invalid_joint_id;
									for (auto child : glTF.nodes[globalNodeID].children)
										dfsTraversalStack.push(child);
								};
								addNodeToSkeleton(index);
								while (!dfsTraversalStack.empty())
								{
									const auto globalNodeID = dfsTraversalStack.top();
									dfsTraversalStack.pop();
									addNodeToSkeleton(globalNodeID);
								}
							}
						}
					}
					// create skeletons
					{
						core::vector<uint32_t> skeletonJointCountPrefixSum;
						skeletonJointCountPrefixSum.resize(skeletonJointCount.size());
						// now create buffer for skeletons
						SBufferBinding<ICPUBuffer> parentJointID = {0ull,core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ICPUSkeleton::joint_id_t)*nodeCount)};
						SBufferBinding<ICPUBuffer> defaultTransforms = {0ull,core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(core::matrix3x4SIMD)*nodeCount)};
						core::vector<const char*> names(nodeCount);
						// and fill them
						{
							std::exclusive_scan(skeletonJointCount.begin(),skeletonJointCount.end(),skeletonJointCountPrefixSum.begin(),0u);
							auto pParentJointID = reinterpret_cast<ICPUSkeleton::joint_id_t*>(parentJointID.buffer->getPointer());
							auto pDefaultTransforms = reinterpret_cast<core::matrix3x4SIMD*>(defaultTransforms.buffer->getPointer());
							for (uint32_t index=0u; index<nodeCount; ++index)
							{
								const auto& skeletonNode = skeletonNodes[index];
								const auto offset = skeletonJointCountPrefixSum[skeletonNode.skeletonID]+skeletonNode.localJointID;
								pParentJointID[offset] = skeletonNode.localParentJointID;
								pDefaultTransforms[offset] = skeletonNode.defaultNodeTransform;
								names[offset] = skeletonNode.glTFNodeName.data();
							}
						}
						for (auto i=0u; i<skeletonJointCount.size(); i++)
						{
							const auto offset = skeletonJointCountPrefixSum[i];
							auto parentJointIDBinding = parentJointID;
							auto defaultTransformsBinding = defaultTransforms;
							parentJointIDBinding.offset = offset*sizeof(ICPUSkeleton::joint_id_t);
							defaultTransformsBinding.offset = offset*sizeof(core::matrix3x4SIMD);
							const char* const* namesPtr = names.data()+offset;
							skeletons.emplace_back() = core::make_smart_refctd_ptr<ICPUSkeleton>(std::move(parentJointIDBinding),std::move(defaultTransformsBinding),namesPtr,namesPtr+skeletonJointCount[i]);
						}
					}
				}
			
				// size vertexJointToSkeletonJoint
				{
					uint32_t totalSkinJointRefs = 0u;
					for (const auto& skin : glTF.skins)
						totalSkinJointRefs += skin.joints.size();
					vertexJointToSkeletonJoint = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ICPUSkeleton::joint_id_t)*totalSkinJointRefs);
					inverseBindPose = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(core::matrix3x4SIMD)*totalSkinJointRefs);
				}
				// then go over skins
				uint32_t skinJointRefCount = 0u;
				for (auto index=0u; index<glTF.skins.size(); index++)
				{
					const auto& glTFSkin = glTF.skins[index];
					const auto jointCount = glTFSkin.joints.size();
					if (jointCount==0u)
						continue;

					// find LCM
					core::unordered_map<uint32_t,uint32_t> commonAncestors;
					// populate
					for (uint32_t node=glTFSkin.joints[0]; node!=ICPUSkeleton::invalid_joint_id; node=globalParent[node])
						commonAncestors.insert({node,0u});
					// trim
					for (const auto& joint : glTFSkin.joints)
					{
						// mark unvisited
						for (auto& ancestor : commonAncestors)
							ancestor.second = ~0u;
						// visit
						uint32_t level = 0u;
						for (uint32_t node=joint; node!=ICPUSkeleton::invalid_joint_id; node=globalParent[node])
						{
							auto found = commonAncestors.find(node);
							if (found!=commonAncestors.end())
								found->second = level;
							level++;
						}
						// remove unvisited
						for (auto it=commonAncestors.begin(); it!=commonAncestors.end();)
						{
							if (it->second == ~0u)
								it = commonAncestors.erase(it);
							else
								++it;
						}
					}
					// validate
					if (commonAncestors.empty())
					{
						context.loadContext.params.logger.log("GLTF: INVALID SKIN, NO COMMON ANCESTORS!",system::ILogger::ELL_ERROR);
						continue;
					}
				
					// find pivot node
					uint32_t globalRootNode;
					if (glTFSkin.skeleton.has_value()) //! we can explicitly point skin root node
					{
						globalRootNode = glTFSkin.skeleton.value();
						if (commonAncestors.find(globalRootNode)==commonAncestors.end())
						{
							context.loadContext.params.logger.log("GLTF: INVALID SKIN, EXPLICIT ROOT NOT IN COMMON ANCESTORS!", system::ILogger::ELL_ERROR);
							continue;
						}
					}
					else
					{
						uint32_t lowestLevel = ~0u;
						for (const auto& ancestor : commonAncestors)
						if (ancestor.second<lowestLevel)
						{
							globalRootNode = ancestor.first;
							lowestLevel = ancestor.second;
						}
					}

					skins[index].skeleton = skeletons[skeletonNodes[globalRootNode].skeletonID];
					skins[index].translationTable.offset = sizeof(ICPUSkeleton::joint_id_t) * skinJointRefCount;
					skins[index].translationTable.size = sizeof(ICPUSkeleton::joint_id_t) * jointCount;
					skins[index].translationTable.buffer = core::smart_refctd_ptr(vertexJointToSkeletonJoint);
					skins[index].inverseBindPose.offset = sizeof(core::matrix3x4SIMD) * skinJointRefCount;
					skins[index].inverseBindPose.size = sizeof(core::matrix3x4SIMD) * jointCount;
					skins[index].inverseBindPose.buffer = core::smart_refctd_ptr(inverseBindPose);
					skins[index].jointCount = jointCount;

					auto translationTableIt = reinterpret_cast<ICPUSkeleton::joint_id_t*>(skins[index].translationTable.buffer->getPointer())+skinJointRefCount;
					for (const auto& joint : glTFSkin.joints)
						*(translationTableIt++) = skeletonNodes[joint].localJointID;

					auto inverseBindPoseIt = reinterpret_cast<core::matrix3x4SIMD*>(skins[index].inverseBindPose.buffer->getPointer())+skinJointRefCount;
					const auto& accessorInverseBindMatricesID = glTFSkin.inverseBindMatrices.has_value() ? glTFSkin.inverseBindMatrices.value() : 0xdeadbeef;
					if (accessorInverseBindMatricesID!=0xdeadbeef)
					{
						const auto& glTFAccessor = glTF.accessors[accessorInverseBindMatricesID];
						if (!glTFAccessor.bufferView.has_value())
						{
							context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!",system::ILogger::ELL_ERROR);
							return false;
						}

						const auto& glTFBufferView = glTF.bufferViews[glTFAccessor.bufferView.value()];
						if (!glTFBufferView.buffer.has_value())
						{
							context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!",system::ILogger::ELL_ERROR);
							return false;
						}

						auto cpuBuffer = cpuBuffers[glTFBufferView.buffer.value()];
						const size_t globalIBPOffset = [&]()
						{
							const size_t bufferViewOffset = glTFBufferView.byteOffset.has_value() ? glTFBufferView.byteOffset.value() : 0u;
							const size_t relativeAccessorOffset = glTFAccessor.byteOffset.has_value() ? glTFAccessor.byteOffset.value() : 0u;

							return bufferViewOffset + relativeAccessorOffset;
						}();

						auto* inData = reinterpret_cast<core::matrix4SIMD*>(reinterpret_cast<uint8_t*>(cpuBuffer->getPointer()) + globalIBPOffset); //! glTF stores 4x4 IBP column_major matrices
						for (uint32_t j=0u; j<jointCount; ++j)
							inverseBindPoseIt[j] = core::transpose(inData[j]).extractSub3x4();
					}
					else
						std::fill_n(inverseBindPoseIt,jointCount,core::matrix3x4SIMD());

					skins[index].root = skeletonNodes[globalRootNode].localJointID;
					skinJointRefCount += jointCount;
				}
			}

			core::unordered_set<ICPURenderpassIndependentPipeline*> pipelineSet;
			core::vector<core::smart_refctd_ptr<ICPUMesh>> cpuMeshes;
			{
				// go over all meshes and create ICPUMeshes & ICPUMeshBuffers but without skins attached
				core::vector<core::smart_refctd_ptr<ICPUMesh>> meshesView;
				{
					for (const auto& glTFMesh : glTF.meshes)
					{
						auto& cpuMesh = meshesView.emplace_back() = core::make_smart_refctd_ptr<ICPUMesh>();

						for (const auto& glTFprimitive : glTFMesh.primitives)
						{
							std::remove_reference_t<decltype(glTFprimitive)> SGLTFPrimitive;

							auto cpuMeshBuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
							cpuMeshBuffer->setPositionAttributeIx(SAttributes::POSITION_ATTRIBUTE_LAYOUT_ID);

							using BufferViewReferencingBufferID = uint32_t;
							std::unordered_map<BufferViewReferencingBufferID, core::smart_refctd_ptr<ICPUBuffer>> idReferenceBindingBuffers;

							SVertexInputParams vertexInputParams;

							auto handleAccessor = [&](SGLTF::SGLTFAccessor& glTFAccessor, const std::optional<uint32_t> queryAttributeId = {}) -> bool
							{
								const E_FORMAT format = SGLTF::SGLTFAccessor::getFormat(glTFAccessor.componentType.value(), glTFAccessor.type.value());
								if (format == EF_UNKNOWN)
								{
									context.loadContext.params.logger.log("GLTF: COULD NOT SPECIFY NABLA FORMAT!",system::ILogger::ELL_ERROR);
									return false;
								}

								auto& glTFbufferView = glTF.bufferViews[glTFAccessor.bufferView.value()];
								const uint32_t attributeId = queryAttributeId.has_value() ? queryAttributeId.value() : 0xdeadbeef;
								const uint32_t& bufferBindingId = attributeId; //! glTF exporters are sometimes retarded setting relativeOffset more than 2048, so we go with single binding per attribute

								const uint32_t& bufferDataId = glTFbufferView.buffer.value();
								const auto& globalOffsetInBufferBindingResource = glTFbufferView.byteOffset.has_value() ? glTFbufferView.byteOffset.value() : 0u;
								const auto& relativeOffsetInBufferViewAttribute = glTFAccessor.byteOffset.has_value() ? glTFAccessor.byteOffset.value() : 0u;

								std::remove_reference_t<decltype(glTFbufferView)> SGLTFBufferView;

								auto setBufferBinding = [&](uint32_t target) -> void
								{
									asset::SBufferBinding<ICPUBuffer> bufferBinding;
									bufferBinding.offset = globalOffsetInBufferBindingResource + relativeOffsetInBufferViewAttribute;

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

										vertexInputParams.enabledAttribFlags |= core::createBitmask({ attributeId });
										vertexInputParams.attributes[attributeId].binding = bufferBindingId;
										vertexInputParams.attributes[attributeId].format = format;
										vertexInputParams.attributes[attributeId].relativeOffset = 0u;
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

							const E_PRIMITIVE_TOPOLOGY primitiveTopology = [&](uint32_t modeValue) -> E_PRIMITIVE_TOPOLOGY
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
									default:
										break;
								}
								return EPT_PATCH_LIST;
							}(glTFprimitive.mode.value());

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
								context.loadContext.params.logger.log("GLTF: COULD NOT DETECT POSITION ATTRIBUTE!",system::ILogger::ELL_ERROR);
								return false;
							}

							if (glTFprimitive.attributes.normal.has_value())
							{
								const size_t accessorID = glTFprimitive.attributes.normal.value();

								auto& glTFNormalAccessor = glTF.accessors[accessorID];
								cpuMeshBuffer->setNormalAttributeIx(SAttributes::NORMAL_ATTRIBUTE_LAYOUT_ID);
								if (!handleAccessor(glTFNormalAccessor, SAttributes::NORMAL_ATTRIBUTE_LAYOUT_ID))
									return {};
							}

							bool hasUV = false;
							if (glTFprimitive.attributes.texcoord.has_value())
							{
								const size_t accessorID = glTFprimitive.attributes.texcoord.value();

								hasUV = true;
								auto& glTFTexcoordXAccessor = glTF.accessors[accessorID];
								if (!handleAccessor(glTFTexcoordXAccessor, SAttributes::UV_ATTRIBUTE_LAYOUT_ID))
									return {};
							}
							bool hasColor = false;
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
										context.loadContext.params.logger.log("GLTF: JOINTS ACCESSOR MUST HAVE VEC4 TYPE!",system::ILogger::ELL_ERROR);
										return {};
									}

									// TODO: also support EF_R10G10B10A2_UINT if there's only max 3 vertex weights
									// you need to requantize (process) the vertex weights FIRST to know that
									const asset::E_FORMAT jointsFormat = [&]()
									{
										if (glTFJointsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_BYTE)
											return EF_R8G8B8A8_UINT;
										else if (glTFJointsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT)
											return EF_R16G16B16A16_UINT;
										return EF_UNKNOWN;
									}();

									if (jointsFormat == EF_UNKNOWN)
									{
										context.loadContext.params.logger.log("GLTF: DETECTED JOINTS BUFFER WITH INVALID COMPONENT TYPE!",system::ILogger::ELL_ERROR);
										return {};
									}

									if (!glTFJointsXAccessor.bufferView.has_value())
									{
										context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!",system::ILogger::ELL_ERROR);
										return {};
									}

									const auto& bufferViewID = glTFJointsXAccessor.bufferView.value();
									const auto& glTFBufferView = glTF.bufferViews[bufferViewID];

									if (!glTFBufferView.buffer.has_value())
									{
										context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!",system::ILogger::ELL_ERROR);
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
										context.loadContext.params.logger.log("GLTF: WEIGHTS ACCESSOR MUST HAVE VEC4 TYPE!",system::ILogger::ELL_ERROR);
										return {};
									}

									const asset::E_FORMAT weightsFormat = [&]()
									{
										if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_FLOAT)
											return EF_R32G32B32A32_SFLOAT;
										else if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_BYTE)
											return EF_R8G8B8A8_UINT; // TODO: UNORM
										else if (glTFWeightsXAccessor.componentType.value() == SGLTF::SGLTFAccessor::SCT_UNSIGNED_SHORT)
											return EF_R16G16B16A16_UINT; // TODO: UNORM
										else
											return EF_UNKNOWN;
									}();

									if (weightsFormat == EF_UNKNOWN)
									{
										context.loadContext.params.logger.log("GLTF: DETECTED WEIGHTS BUFFER WITH INVALID COMPONENT TYPE!",system::ILogger::ELL_ERROR);
										return {};
									}

									if (!glTFWeightsXAccessor.bufferView.has_value())
									{
										context.loadContext.params.logger.log("GLTF: NO BUFFER VIEW INDEX FOUND!",system::ILogger::ELL_ERROR);
										return {};
									}

									const auto& bufferViewID = glTFWeightsXAccessor.bufferView.value();
									const auto& glTFBufferView = glTF.bufferViews[bufferViewID];

									if (!glTFBufferView.buffer.has_value())
									{
										context.loadContext.params.logger.log("GLTF: NO BUFFER INDEX FOUND!",system::ILogger::ELL_ERROR);
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
							bool skinningEnabled = false;

							if (overrideJointsReference.size() && overrideWeightsReference.size())
							{
								if (overrideJointsReference.size() != overrideWeightsReference.size())
								{
									context.loadContext.params.logger.log("GLTF: JOINTS ATTRIBUTES VERTEX BUFFERS AMOUNT MUST BE EQUAL TO WEIGHTS ATTRIBUTES VERTEX BUFFERS AMOUNT!",system::ILogger::ELL_ERROR);
									return {};
								}

								if (overrideJointsReference.size() > 1u || overrideWeightsReference.size() > 1u)
								{
									if (!std::equal(std::begin(overrideJointsReference) + 1, std::end(overrideJointsReference), std::begin(overrideJointsReference), [](const OverrideReference& lhs, const OverrideReference& rhs) { return lhs.format == rhs.format && lhs.accessor->count.value() == rhs.accessor->count.value(); }))
									{
										context.loadContext.params.logger.log("GLTF: JOINTS ATTRIBUTES VERTEX BUFFERS MUST NOT HAVE VARIOUS DATA TYPE OR LENGTH!",system::ILogger::ELL_ERROR);
										return {};
									}

									if (!std::equal(std::begin(overrideWeightsReference) + 1, std::end(overrideWeightsReference), std::begin(overrideWeightsReference), [](const OverrideReference& lhs, const OverrideReference& rhs) { return lhs.format == rhs.format && lhs.accessor->count.value() == rhs.accessor->count.value(); }))
									{
										context.loadContext.params.logger.log("GLTF: WEIGHTS ATTRIBUTES VERTEX BUFFERS MUST NOT HAVE VARIOUS DATA TYPE OR LENGTH!",system::ILogger::ELL_ERROR);
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
													typename VertexInfluenceData::ComponentData& skinComponent = vertexInfluenceData.perVertexComponentsData[i];

													JointComponentT* vJoint = reinterpret_cast<JointComponentT*>(vJointsComponentDataRaw) + i;
													WeightCompomentT* vWeight = reinterpret_cast<WeightCompomentT*>(vWeightsComponentDataRaw) + i;

													skinComponent.joint = *vJoint;
													skinComponent.weight = *vWeight;
												}
											}

											std::vector<typename VertexInfluenceData::ComponentData> skinComponentUnlimitedStream;
											{
												for (const auto& vertexInfluenceData : vertexInfluenceDataContainer)
												for (const auto& skinComponent : vertexInfluenceData.perVertexComponentsData)
												{
													auto& data = skinComponentUnlimitedStream.emplace_back();

													data.joint = skinComponent.joint;
													data.weight = skinComponent.weight;
												}
											}

											//! sort, cache and keep only biggest influencers
											std::sort(std::begin(skinComponentUnlimitedStream), std::end(skinComponentUnlimitedStream), [&](const typename VertexInfluenceData::ComponentData& lhs, const typename VertexInfluenceData::ComponentData& rhs) { return lhs.weight < rhs.weight; });
											{
												auto iteratorEnd = skinComponentUnlimitedStream.begin() + (vertexInfluenceDataContainer.size() - 1u) * 4u;
												if (skinComponentUnlimitedStream.begin() != iteratorEnd)
													skinComponentUnlimitedStream.erase(skinComponentUnlimitedStream.begin(), iteratorEnd);

												std::sort(std::begin(skinComponentUnlimitedStream), std::end(skinComponentUnlimitedStream), [&](const typename VertexInfluenceData::ComponentData& lhs, const typename VertexInfluenceData::ComponentData& rhs) { return lhs.joint < rhs.joint; });
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
/*
											// just rely on format promotion to fix these up
											case 3u:
											{
												if constexpr (std::is_same<JointComponentT, uint8_t>::value)
													repackJointsFormat = EF_R8G8B8_UINT;
												else if (std::is_same<JointComponentT, uint16_t>::value)
													repackJointsFormat = EF_R16G16B16_UINT;

												if constexpr (std::is_same<WeightCompomentT, uint8_t>::value)
													repackWeightsFormat = EF_R8G8B8_UINT;
												else if (std::is_same<WeightCompomentT, uint16_t>::value)
													repackWeightsFormat = EF_R16G16B16_UINT;
												else if (std::is_same<WeightCompomentT, float>::value)
													repackWeightsFormat = EF_R32G32B32_SFLOAT;
											} break;
*/
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

										{
											const size_t repackJointsTexelByteSize = asset::getTexelOrBlockBytesize(repackJointsFormat);
											const size_t repackWeightsTexelByteSize = asset::getTexelOrBlockBytesize(repackWeightsFormat);

											auto vOverrideRepackedJointsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * repackJointsTexelByteSize);
											auto vOverrideRepackedWeightsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vCommonOverrideAttributesCount * repackWeightsTexelByteSize);

											memset(vOverrideRepackedJointsBuffer->getPointer(), 0, vOverrideRepackedJointsBuffer->getSize());
											memset(vOverrideRepackedWeightsBuffer->getPointer(), 0, vOverrideRepackedWeightsBuffer->getSize());
											{ //! pack buffers and quantize weights buffer
												constexpr uint16_t MAX_INFLUENCE_WEIGHTS_PER_VERTEX = 4;

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

#if 1 // TODO: rewrite this complex as F function
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

													for (uint16_t i = 0; i < quantRequest.encodeData.size(); ++i) //! quantization test
													{
														auto& encode = quantRequest.encodeData[i];
														auto* quantBuffer = std::get<typename QuantRequest::QUANT_BUFFER>(encode);
														auto* errorBuffer = std::get<typename QuantRequest::ERROR_BUFFER>(encode);
														const WEIGHT_ENCODING requestWeightEncoding = std::get<WEIGHT_ENCODING>(encode);
														const E_FORMAT requestQuantFormat = std::get<E_FORMAT>(encode);

														quantize(packedWeightsStream, quantBuffer, requestQuantFormat);
														core::vectorSIMDf quantsDecoded = decodeQuant(quantBuffer, requestQuantFormat);

														for (uint16_t i = 0; i < MAX_INFLUENCE_WEIGHTS_PER_VERTEX; ++i)
														{
															const auto& weightInput = packedWeightsStream.pointer[i];
															if (weightInput)
															{
																const typename QuantRequest::ERROR_TYPE& errorComponent = errorBuffer[i] = core::abs(quantsDecoded.pointer[i] - weightInput);

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
#endif
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

									auto setOverrideBufferBinding = [&](OverrideSkinningBuffers::Override& overrideData, uint16_t attributeID)
									{
										asset::SBufferBinding<ICPUBuffer> bufferBinding;
										bufferBinding.buffer = core::smart_refctd_ptr(overrideData.cpuBuffer);
										bufferBinding.offset = 0u;

										const uint32_t bufferBindingId = attributeID;

										cpuMeshBuffer->setVertexBufferBinding(std::move(bufferBinding), bufferBindingId);

										vertexInputParams.enabledBindingFlags |= core::createBitmask({ bufferBindingId });
										vertexInputParams.bindings[bufferBindingId].inputRate = EVIR_PER_VERTEX;
										vertexInputParams.bindings[bufferBindingId].stride = asset::getTexelOrBlockBytesize(overrideData.format);

										vertexInputParams.enabledAttribFlags |= core::createBitmask({ attributeID });
										vertexInputParams.attributes[attributeID].binding = bufferBindingId;
										vertexInputParams.attributes[attributeID].format = overrideData.format;
										vertexInputParams.attributes[attributeID].relativeOffset = 0;
									};

									cpuMeshBuffer->setJointIDAttributeIx(SAttributes::JOINTS_ATTRIBUTE_LAYOUT_ID);
									cpuMeshBuffer->setJointWeightAttributeIx(SAttributes::WEIGHTS_ATTRIBUTE_LAYOUT_ID);

									setOverrideBufferBinding(overrideSkinningBuffers.jointsAttributes, SAttributes::JOINTS_ATTRIBUTE_LAYOUT_ID);
									setOverrideBufferBinding(overrideSkinningBuffers.weightsAttributes, SAttributes::WEIGHTS_ATTRIBUTE_LAYOUT_ID);
								}

								skinningEnabled = true;
							}

							if (glTFprimitive.material.has_value())
							{
								const auto& material = materials[glTFprimitive.material.value()];
								memcpy(cpuMeshBuffer->getPushConstantsDataPtr(),&material.pushConstants,sizeof(material.pushConstants));
								cpuMeshBuffer->setAttachedDescriptorSet(core::smart_refctd_ptr(material.descriptorSet));
							}
							auto pipeline = getPipeline(context,primitiveTopology,vertexInputParams,skinningEnabled,hasUV,hasColor);
							pipelineSet.insert(pipeline.get());
							cpuMeshBuffer->setPipeline(std::move(pipeline));

							cpuMesh->getMeshBufferVector().push_back(std::move(cpuMeshBuffer));
						}
					}
				}

				// go over unique <mesh,skin> pairs and make a cpuMesh
				for (const auto& hash : meshSkinPairs)
				{
					const auto pair = static_cast<MeshSkinPair>(hash);

					auto& mesh = cpuMeshes.emplace_back() = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshesView[pair.mesh]->clone(1u)); // duplicate only mesh and meshbuffer

					if (pair.skin!=0xdeadbeefu) // has a skin
						for (auto& meshbuffer : mesh->getMeshBufferVector())
						{
							auto& skin = skins[pair.skin];
							const size_t jointCount = skin.jointCount;

							SBufferBinding<ICPUBuffer> inverseBindPoseBinding;
							inverseBindPoseBinding.buffer = core::smart_refctd_ptr(skin.inverseBindPose.buffer);
							inverseBindPoseBinding.offset = skin.inverseBindPose.offset;

							SBufferBinding<ICPUBuffer> jointAABBBufferBinding;
							jointAABBBufferBinding.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(jointCount*sizeof(core::aabbox3df));
							jointAABBBufferBinding.offset = 0u;

							auto* aabbPtr = reinterpret_cast<core::aabbox3df*>(jointAABBBufferBinding.buffer->getPointer());
							meshbuffer->setSkin(std::move(inverseBindPoseBinding),std::move(jointAABBBufferBinding),jointCount,meshbuffer->deduceMaxJointsPerVertex());
							nbl::asset::IMeshManipulator::calculateBoundingBox(meshbuffer.get(),aabbPtr);
						}
				}
			}

			// go over nodes one last time to record instances
			core::vector<CGLTFMetadata::Instance> instances;
			for (uint32_t index=0u; index<nodeCount; ++index)
			{
				const auto& glTFnode = glTF.nodes[index];

				const uint32_t meshID = glTFnode.mesh.has_value() ? glTFnode.mesh.value() : 0xdeadbeef;
				if (meshID == 0xdeadbeef)
				{
					skeletonNodes[index].instanceID = 0xdeadbeefu;
					continue;
				}

				const uint32_t skinID = glTFnode.skin.has_value() ? glTFnode.skin.value() : 0xdeadbeef;

				// record that as an instance 
				auto found = meshSkinPairs.find({meshID,skinID});
				assert(found!=meshSkinPairs.end());

				skeletonNodes[index].instanceID = instances.size();
				auto& instance = instances.emplace_back();
				if (skinID != 0xdeadbeefu) // has skin
				{
					instance.skeleton = skins[skinID].skeleton.get();
					instance.skinTranslationTable.buffer = skins[skinID].translationTable.buffer;
					instance.skinTranslationTable.offset = skins[skinID].translationTable.offset;
				}
				else
				{
					instance.skeleton = nullptr;
					instance.skinTranslationTable.buffer = nullptr;
					instance.skinTranslationTable.offset = 0xdeadbeefu;
				}
				instance.mesh = cpuMeshes[std::distance(meshSkinPairs.begin(),found)].get();
				instance.attachedToNode = skeletonNodes[index].localJointID;
			}

			//
			core::vector<CGLTFMetadata::Scene> scenes(glTF.scenes.size());
			{
				core::stack<uint32_t> dfsTraversalStack;
				for (auto index=0u; index!=scenes.size(); index++)
				{
					auto& instanceIDs = scenes[index].instanceIDs;
					auto addNodeToScene = [&glTF,&skeletonNodes,&dfsTraversalStack,&instanceIDs](const uint32_t globalNodeID) -> void
					{
						const auto instanceID = skeletonNodes[globalNodeID].instanceID;
						if (instanceID!=ICPUSkeleton::invalid_joint_id)
							instanceIDs.push_back(instanceID);
						for (auto child : glTF.nodes[globalNodeID].children)
							dfsTraversalStack.push(child);
					};
					for (auto node : glTF.scenes[index].nodes)
					{
						addNodeToScene(node);
						while (!dfsTraversalStack.empty())
						{
							const auto globalNodeID = dfsTraversalStack.top();
							dfsTraversalStack.pop();
							addNodeToScene(globalNodeID);
						}
					}
				}
			}

			core::smart_refctd_ptr<CGLTFMetadata> glTFMetadata = core::make_smart_refctd_ptr<CGLTFMetadata>(pipelineSet.size());
			{
				if (glTF.defaultScene.has_value())
					glTFMetadata->defaultSceneID = glTF.defaultScene.value();

				glTFMetadata->instances = std::move(instances);
				glTFMetadata->skeletons = std::move(skeletons);
				glTFMetadata->scenes = std::move(scenes);

				uint32_t i = 0u;
				for (auto& pipeline : pipelineSet)
					glTFMetadata->placeMeta(i++,pipeline,{core::smart_refctd_ptr(m_basicViewParamsSemantics)});
			}

			return SAssetBundle(std::move(glTFMetadata), cpuMeshes);
		}

		bool CGLTFLoader::loadAndGetGLTF(SGLTF& glTF, SContext& context)
		{
			simdjson::dom::parser parser;
			auto* _file = context.loadContext.mainFile;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				system::IFile::success_t success;
				_file->read(success, jsonBuffer->getPointer(), 0u, jsonBuffer->getSize());
				if (!success)
					return false;
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

			if (scene.error() != simdjson::error_code::NO_SUCH_FIELD)
				glTF.defaultScene = static_cast<uint32_t>(scene.get_uint64());

			if (scenes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				const auto& jsonScenes = scenes.get_array();
				for (const auto& jsonScene : jsonScenes)
				{
					auto& glTFScene = glTF.scenes.emplace_back();
					const auto& nodes = jsonScene.at_key("nodes");

					if(nodes.error() != simdjson::error_code::NO_SUCH_FIELD)
						for (const auto& node : nodes.get_array())
							glTFScene.nodes.push_back(static_cast<uint32_t>(node.get_uint64()));
				}
			}

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
						glTFBufferView.buffer = static_cast<uint32_t>(buffer.get_uint64().value());

					if (byteOffset.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteOffset = byteOffset.get_uint64().value();

					if (byteLength.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteLength = byteLength.get_uint64().value();

					if (byteStride.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.byteStride = static_cast<uint32_t>(byteStride.get_uint64().value());

					if (target.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBufferView.target = static_cast<uint32_t>(target.get_uint64().value());

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
						glTFTexture.sampler = static_cast<uint32_t>(sampler.get_uint64().value());

					if (source.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFTexture.source = static_cast<uint32_t>(source.get_uint64().value());

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
								glTFBaseColorTexture.index = static_cast<uint32_t>(index.get_uint64().value());

							if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFBaseColorTexture.texCoord = static_cast<uint32_t>(texCoord.get_uint64().value());
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
								glTFMetallicRoughnessTexture.index = static_cast<uint32_t>(index.get_uint64().value());

							if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
								glTFMetallicRoughnessTexture.texCoord = static_cast<uint32_t>(texCoord.get_uint64().value());
						}
					}

					if (normalTexture.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& glTFNormalTexture = glTFMaterial.normalTexture.emplace();
						const auto& normalTextureData = normalTexture.get_object();

						const auto& index = normalTextureData.at_key("index");
						const auto& texCoord = normalTextureData.at_key("texCoord");
						const auto& scale = normalTextureData.at_key("scale");

						if (index.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFNormalTexture.index = static_cast<uint32_t>(index.get_uint64().value());

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFNormalTexture.texCoord = static_cast<uint32_t>(texCoord.get_uint64().value());

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
							glTFOcclusionTexture.index = static_cast<uint32_t>(texCoord.get_uint64().value());

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFOcclusionTexture.texCoord = static_cast<uint32_t>(texCoord.get_uint64().value());

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
							glTFEmissiveTexture.index = static_cast<uint32_t>(texCoord.get_uint64().value());

						if (texCoord.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFEmissiveTexture.texCoord = static_cast<uint32_t>(texCoord.get_uint64().value());
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
										context.loadContext.params.logger.log("GLTF: UNSUPPORTED TARGET PATH!",system::ILogger::ELL_ERROR);
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
									context.loadContext.params.logger.log("GLTF: UNSUPPORTED INTERPOLATION!",system::ILogger::ELL_ERROR);
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
						context.loadContext.params.logger.log("GLTF: DETECTED TOO MANY JOINTS REFERENCES!",system::ILogger::ELL_ERROR);
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
						glTFAccessor.bufferView = static_cast<uint32_t>(bufferView.get_uint64().value());

					if (byteOffset.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.byteOffset = byteOffset.get_uint64().value();

					if (componentType.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.componentType = static_cast<SGLTF::SGLTFAccessor::SCompomentType>(componentType.get_uint64().value());

					if (normalized.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.normalized = normalized.get_bool().value();

					if (count.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFAccessor.count = static_cast<uint32_t>(count.get_uint64().value());

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
							context.loadContext.params.logger.log("GLTF: DETECTED UNSUPPORTED TYPE!",system::ILogger::ELL_ERROR);
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
						context.loadContext.params.logger.log("GLTF: COULD NOT DETECT ANY PRIMITIVE!",system::ILogger::ELL_ERROR);
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
							glTFPrimitive.indices = static_cast<uint32_t>(indices.get_uint64().value());

						if (material.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFPrimitive.material = static_cast<uint32_t>(material.get_uint64().value());

						if (mode.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFPrimitive.mode = static_cast<uint32_t>(mode.get_uint64().value());
						else
							glTFPrimitive.mode = 4;

						if (targets.error() != simdjson::error_code::NO_SUCH_FIELD)
							for (const auto& [targetKey, targetID] : targets.get_object())
								glTFPrimitive.targets.emplace()[targetKey.data()] = static_cast<uint32_t>(targetID.get_uint64().value());

						if (attributes.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							for (const auto& [attributeKey, accessorID] : attributes.get_object())
							{
								const auto requestedAccessor = accessorID.get_uint64().value();

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
										context.loadContext.params.logger.log("GLTF: LOADER DOESN'T SUPPORT MULTIPLE UV ATTRIBUTES!",system::ILogger::ELL_ERROR);
										return false;
									}

									glTFPrimitive.attributes.texcoord = requestedAccessor;
								}
								else if (attributeMap.first == "COLOR")
								{
									if (attributeMap.second >= 1u)
									{
										context.loadContext.params.logger.log("GLTF: LOADER DOESN'T SUPPORT MULTIPLE COLOR ATTRIBUTES!",system::ILogger::ELL_ERROR);
										return false;
									}

									glTFPrimitive.attributes.color = requestedAccessor;
								}
								else if (attributeMap.first == "JOINTS")
								{
									if (attributeMap.second >= glTFPrimitive.attributes.MAX_JOINTS_ATTRIBUTES)
									{
										context.loadContext.params.logger.log("GLTF: EXCEEDED 'MAX_JOINTS_ATTRIBUTES' FOR JOINTS ATTRIBUTES!",system::ILogger::ELL_ERROR);
										return false;
									}

									glTFPrimitive.attributes.joints[attributeMap.second] = requestedAccessor;
								}
								else if (attributeMap.first == "WEIGHTS")
								{
									if (attributeMap.second >= glTFPrimitive.attributes.MAX_WEIGHTS_ATTRIBUTES)
									{
										context.loadContext.params.logger.log("GLTF: EXCEEDED 'MAX_WEIGHTS_ATTRIBUTES' FOR JOINTS ATTRIBUTES!",system::ILogger::ELL_ERROR);
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
							glTFnode.camera = static_cast<uint32_t>(camera.get_uint64().value());

						if (children.error() != simdjson::error_code::NO_SUCH_FIELD)
							for (const auto& child : children)
								glTFnode.children.push_back(child.get_uint64().value());

						if (skin.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.skin = static_cast<uint32_t>(skin.get_uint64().value());

						if (matrix.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							const auto& matrixArray = matrix.get_array();
							core::matrix4SIMD tmpMatrix;

							for (uint32_t i = 0; i < matrixArray.size(); ++i)
								*(tmpMatrix.pointer() + i) = matrixArray.at(i).get_double().value();

							glTFnode.transformation.matrix = core::transpose(tmpMatrix).extractSub3x4();
						}
						else
						{
							struct SGLTFNTransformationTRS
							{
								core::vector3df_SIMD translation = {};							//!< The node's translation along the x, y, and z axes.
								core::vector3df_SIMD scale = core::vector3df_SIMD(1.f,1.f,1.f);	//!< The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
								core::quaternion rotation = {};									//!< The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
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
									trs.rotation.getPointer()[index++] = val.get_double().value();
							}

							if (scale.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								const auto& scaleArray = scale.get_array();

								size_t index = {};
								for (const auto& val : scaleArray)
									trs.scale[index++] = val.get_double().value();
							}

							glTFnode.transformation.matrix.setScaleRotationAndTranslation(trs.scale, trs.rotation, trs.translation);
						}

						if (mesh.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.mesh = static_cast<uint32_t>(mesh.get_uint64().value());

						if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.name = name.get_string().value();

						// TODO camera, skinning, etc HERE

						return glTFnode.validate();
					};

					if (!handleTheGLTFTree()) //! TODO more validations in future for glTF objects
					{
						context.loadContext.params.logger.log("GLTF: NODE VALIDATION FAILED!",system::ILogger::ELL_ERROR);
						return false;
					}
				}
			}

			return true;
		}

#endif // _NBL_COMPILE_WITH_GLTF_LOADER_
