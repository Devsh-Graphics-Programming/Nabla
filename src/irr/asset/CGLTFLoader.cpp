// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CGLTFLoader.h"

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "simdjson/singleheader/simdjson.h"
#include <filesystem>
#include "os.h"

#define VERT_SHADER_UV_CACHE_KEY "irr/builtin/shaders/loaders/gltf/vertex_uv.vert"
#define VERT_SHADER_COLOR_CACHE_KEY "irr/builtin/shaders/loaders/gltf/vertex_color.vert"
#define VERT_SHADER_NO_UV_COLOR_CACHE_KEY "irr/builtin/shaders/loaders/gltf/vertex_no_uv_color.vert"

#define FRAG_SHADER_UV_CACHE_KEY "irr/builtin/shaders/loaders/gltf/fragment_uv.frag"
#define FRAG_SHADER_COLOR_CACHE_KEY "irr/builtin/shaders/loaders/gltf/fragment_color.frag"
#define FRAG_SHADER_NO_UV_COLOR_CACHE_KEY "irr/builtin/shaders/loaders/gltf/fragment_no_uv_color.frag"

namespace irr
{
	namespace asset
	{
		namespace SAttributes
		{
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t POSITION_ATTRIBUTE_ID = 0;
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t NORMAL_ATTRIBUTE_ID = 1;
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t UV_ATTRIBUTE_BEGINING_ID = 2;			//!< those attributes are indexed
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t COLOR_ATTRIBUTE_BEGINING_ID = 5;		//!< those attributes are indexed
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t JOINTS_ATTRIBUTE_BEGINING_ID = 8;		//!< those attributes are indexed
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t WEIGHTS_ATTRIBUTE_BEGINING_ID = 12;	//!< those attributes are indexed

			_IRR_STATIC_INLINE_CONSTEXPR uint8_t MAX_UV_ATTRIBUTES = 3;				
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t MAX_COLOR_ATTRIBUTES = 3;
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t MAX_JOINTS_ATTRIBUTES = 4;
			_IRR_STATIC_INLINE_CONSTEXPR uint8_t MAX_WEIGHTS_ATTRIBUTES = 4;
		}

		/*
			Each glTF asset must have an asset property. 
			In fact, it's the only required top-level property
			for JSON to be a valid glTF.
		*/

		CGLTFLoader::CGLTFLoader(asset::IAssetManager* _m_assetMgr) 
			: assetManager(_m_assetMgr)
		{
			auto registerShader = [&](auto constexprStringType, ICPUSpecializedShader::E_SHADER_STAGE stage) -> void
			{
				auto shaderData = assetManager->getFileSystem()->loadBuiltinData<decltype(constexprStringType)>();
				auto unspecializedShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shaderData), asset::ICPUShader::buffer_contains_glsl);

				ICPUSpecializedShader::SInfo specInfo({}, nullptr, "main", stage, stage != ICPUSpecializedShader::ESS_VERTEX ? "?IrrlichtBAW glTFLoader FragmentShader?" : "?IrrlichtBAW glTFLoader VertexShader?");
				auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedShader), std::move(specInfo));

				auto insertShaderIntoCache = [&](const char* path)
				{
					asset::SAssetBundle bundle({ cpuShader });
					assetManager->changeAssetKey(bundle, path);
					assetManager->insertAssetIntoCache(bundle);
				};

				insertShaderIntoCache(decltype(constexprStringType)::value);
			};

			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_UV_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);
			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);
			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(VERT_SHADER_NO_UV_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_VERTEX);

			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_UV_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
			registerShader(IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(FRAG_SHADER_NO_UV_COLOR_CACHE_KEY) {}, ICPUSpecializedShader::ESS_FRAGMENT);
		}
		
		bool CGLTFLoader::isALoadableFileFormat(io::IReadFile* _file) const
		{
			simdjson::dom::parser parser;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				const auto beginPosition = _file->getPos();
				_file->read(jsonBuffer->getPointer(), jsonBuffer->getSize());
				_file->seek(beginPosition);
			}
			simdjson::dom::object tweets = parser.parse(reinterpret_cast<uint8_t*>(jsonBuffer->getPointer()), jsonBuffer->getSize());
			simdjson::dom::element element;

			if (tweets.at_key("asset").get(element) == simdjson::error_code::SUCCESS)
				if (element.at_key("version").get(element) == simdjson::error_code::SUCCESS)
					return true;

			return false;
		}

		asset::SAssetBundle CGLTFLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			SGLTF glTF;
			loadAndGetGLTF(glTF, _file);

			// TODO: having validated and loaded glTF data we can use it to create pipelines and data

			core::vector<core::smart_refctd_ptr<ICPUBuffer>> cpuBuffers;
			for (auto& glTFBuffer : glTF.buffers)
			{
				std::string relativeDirectory = std::string(assetManager->getFileSystem()->getFileDir(_file->getFileName()).c_str()) + "/";
				const asset::IAssetLoader::SAssetLoadParams params(0, nullptr, ECF_CACHE_EVERYTHING, relativeDirectory.c_str());
				auto buffer_bundle = assetManager->getAsset(glTFBuffer.uri.value(), params); // todo
				auto cpuBuffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer_bundle.getContents().begin()[0]);
				cpuBuffers.emplace_back() = core::smart_refctd_ptr<ICPUBuffer>(cpuBuffer);
			}
		
			core::vector<core::smart_refctd_ptr<CCPUMesh>> cpuMeshes;
			for (auto& glTFnode : glTF.nodes) 
			{
				auto cpuMesh = cpuMeshes.emplace_back();

				for (auto& glTFprimitive : glTFnode.glTFMesh.primitives) 
				{
					typedef std::remove_reference<decltype(glTFprimitive)>::type SGLTFPrimitive;
					
					auto cpuMeshBuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();

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

					typedef std::remove_reference<decltype(SGLTFPrimitive::accessors)::value_type::second_type>::type SGLTFAccessor;

					auto handleAccessor = [&](SGLTFAccessor& glTFAccessor, const std::optional<uint32_t> queryAttributeId = {})
					{
						auto getFormat = [&](uint32_t componentType, std::string type)
						{
							switch (componentType)
							{
								case SGLTFAccessor::SCT_BYTE:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R8_SINT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R8G8_SINT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R8G8B8_SINT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R8G8B8A8_SINT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R8G8B8A8_SINT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;

								case SGLTFAccessor::SCT_FLOAT:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R32_SFLOAT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R32G32_SFLOAT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R32G32B32_SFLOAT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R32G32B32A32_SFLOAT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R32G32B32A32_SFLOAT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;

								case SGLTFAccessor::SCT_SHORT:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R16_SINT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R16G16_SINT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R16G16B16_SINT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R16G16B16A16_SINT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R16G16B16A16_SINT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;

								case SGLTFAccessor::SCT_UNSIGNED_BYTE:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R8_UINT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R8G8_UINT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R8G8B8_UINT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R8G8B8A8_UINT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R8G8B8A8_UINT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;

								case SGLTFAccessor::SCT_UNSIGNED_INT:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R32_UINT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R32G32_UINT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R32G32B32_UINT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R32G32B32A32_UINT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R32G32B32A32_UINT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;

								case SGLTFAccessor::SCT_UNSIGNED_SHORT:
								{
									if (type == SGLTFAccessor::SType::SCALAR.data())
										return EF_R16_UINT;
									else if (type == SGLTFAccessor::SType::VEC2.data())
										return EF_R16G16_UINT;
									else if (type == SGLTFAccessor::SType::VEC3.data())
										return EF_R16G16B16_UINT;
									else if (type == SGLTFAccessor::SType::VEC4.data())
										return EF_R16G16B16A16_UINT;
									else if (type == SGLTFAccessor::SType::MAT2.data())
										return EF_R16G16B16A16_UINT;
									else if (type == SGLTFAccessor::SType::MAT3.data())
										return EF_UNKNOWN; // ?
									else if (type == SGLTFAccessor::SType::MAT4.data())
										return EF_UNKNOWN; // ?
								} break;
							}
						};

						const E_FORMAT format = getFormat(glTFAccessor.componentType.value(), glTFAccessor.type.value());

						auto& glTFbufferView = glTF.bufferViews[glTFAccessor.bufferView.value()];
						const uint32_t& bufferBindingId = glTFbufferView.buffer.value();
						const auto& globalOffsetInBufferBindingResource = glTFbufferView.byteOffset.value();
						const auto& relativeOffsetInBufferViewAttribute = glTFAccessor.byteOffset.value();

						typedef std::remove_reference<decltype(glTFbufferView)>::type SGLTFBufferView;

						auto setBufferBinding = [&](uint32_t target) -> void
						{
							asset::SBufferBinding<ICPUBuffer> bufferBinding;
							bufferBinding.offset = globalOffsetInBufferBindingResource;

							idReferenceBindingBuffers[bufferBindingId] = cpuBuffers[bufferBindingId];
							bufferBinding.buffer = idReferenceBindingBuffers[bufferBindingId];

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

									if (isDataInterleaved())
										vertexInputParams.bindings[bufferBindingId].stride = glTFbufferView.byteStride.value();
									else
										vertexInputParams.bindings[bufferBindingId].stride = getTexelOrBlockBytesize(format) * glTFAccessor.count.value(); // TODO: check it out

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

						setBufferBinding(glTFbufferView.target.value());
					};

					const E_PRIMITIVE_TOPOLOGY primitiveTopology = getMode(glTFprimitive.mode.value());
					primitiveAssemblyParams.primitiveType = primitiveTopology;

					if (glTFprimitive.indices.has_value())
					{
						auto& glTFIndexAccessor = glTFprimitive.accessors["INDEX"];
						handleAccessor(glTFIndexAccessor);

						switch (glTFIndexAccessor.componentType.value())
						{
							case SGLTFAccessor::SCT_UNSIGNED_SHORT:
							{
								cpuMeshBuffer->setIndexType(EIT_16BIT);
							} break;

							case SGLTFAccessor::SCT_UNSIGNED_INT:
							{
								cpuMeshBuffer->setIndexType(EIT_32BIT);
							} break;
						}

						cpuMeshBuffer->setIndexCount(glTFIndexAccessor.count.value());
					}

					auto statusPosition = glTFprimitive.accessors.find("POSITION");
					if (statusPosition != glTFprimitive.accessors.end())
					{
						auto& glTFPositionAccessor = glTFprimitive.accessors["POSITION"];
						handleAccessor(glTFPositionAccessor, SAttributes::POSITION_ATTRIBUTE_ID);

						cpuMeshBuffer->setIndexCount(glTFPositionAccessor.count.value());
					}
					else
						return {};

					auto statusNormal = glTFprimitive.accessors.find("NORMAL");
					if (statusNormal != glTFprimitive.accessors.end())
					{
						auto& glTFNormalAccessor = glTFprimitive.accessors["NORMAL"];
						handleAccessor(glTFNormalAccessor, SAttributes::NORMAL_ATTRIBUTE_ID);
					}

					for (uint32_t i = 0; i < SAttributes::MAX_UV_ATTRIBUTES; ++i)
					{
						auto statusTexcoord = glTFprimitive.accessors.find("TEXCOORD_" + std::to_string(i));
						if (statusTexcoord == glTFprimitive.accessors.end())
							break;
						else
						{
							hasUV = true;
							auto& glTFTexcoordXAccessor = glTFprimitive.accessors["TEXCOORD_" + std::to_string(i)];
							handleAccessor(glTFTexcoordXAccessor, SAttributes::UV_ATTRIBUTE_BEGINING_ID + i);
						}
					}

					for (uint32_t i = 0; i < SAttributes::MAX_COLOR_ATTRIBUTES; ++i)
					{
						auto statusColor = glTFprimitive.accessors.find("COLOR_" + std::to_string(i));
						if (statusColor == glTFprimitive.accessors.end())
							break;
						else
						{
							hasColor = true;
							auto& glTFColorXAccessor = glTFprimitive.accessors["COLOR_" + std::to_string(i)];
							handleAccessor(glTFColorXAccessor, SAttributes::COLOR_ATTRIBUTE_BEGINING_ID + i);
						}
					}

					for (uint32_t i = 0; i < SAttributes::MAX_JOINTS_ATTRIBUTES; ++i)
					{
						auto statusJoints = glTFprimitive.accessors.find("JOINTS_" + std::to_string(i));
						if (statusJoints == glTFprimitive.accessors.end())
							break;
						else
						{
							auto& glTFJointsXAccessor = glTFprimitive.accessors["JOINTS_" + std::to_string(i)];
							handleAccessor(glTFJointsXAccessor, SAttributes::JOINTS_ATTRIBUTE_BEGINING_ID + i);
						}
					}

					for (uint32_t i = 0; i < SAttributes::MAX_WEIGHTS_ATTRIBUTES; ++i)
					{
						auto statusWeights = glTFprimitive.accessors.find("WEIGHTS_" + std::to_string(i));
						if (statusWeights == glTFprimitive.accessors.end())
							break;
						else
						{
							auto& glTFWeightsXAccessor = glTFprimitive.accessors["WEIGHTS_" + std::to_string(i)];
							handleAccessor(glTFWeightsXAccessor, SAttributes::WEIGHTS_ATTRIBUTE_BEGINING_ID + i);
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
							if (bundle.isEmpty())
								return nullptr;
							auto assets = bundle.getContents();
							
							return core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assets.begin()[0]);
						};

						if (hasUV) // if both UV and Color defined - we use the UV
							return std::make_pair(loadShader(VERT_SHADER_UV_CACHE_KEY), loadShader(FRAG_SHADER_UV_CACHE_KEY));
						else if (hasColor)
							return std::make_pair(loadShader(VERT_SHADER_COLOR_CACHE_KEY), loadShader(FRAG_SHADER_COLOR_CACHE_KEY));
						else
							return std::make_pair(loadShader(VERT_SHADER_NO_UV_COLOR_CACHE_KEY), loadShader(VERT_SHADER_NO_UV_COLOR_CACHE_KEY));
					};

					auto [cpuVertexShader, cpuFragmentShader] = getShaders(hasUV, hasColor);
					size_t ds0_samplerBinding = 0, ds1_uboBinding = 0;

					asset::ICPUDescriptorSetLayout::SBinding cpuSamplerBinding;

					cpuSamplerBinding.binding = ds0_samplerBinding;
					cpuSamplerBinding.type = EDT_COMBINED_IMAGE_SAMPLER;
					cpuSamplerBinding.count = 1u;
					cpuSamplerBinding.stageFlags = static_cast<ICPUSpecializedShader::E_SHADER_STAGE>(ICPUSpecializedShader::ESS_FRAGMENT);
					cpuSamplerBinding.samplers = nullptr;

					ICPUDescriptorSetLayout::SBinding cpuUboBinding;
					cpuUboBinding.count = 1u;
					cpuUboBinding.binding = ds1_uboBinding;
					cpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
					cpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

					// TODO
				}
			}

			/*
			
					TODO: bottom to change

					put the bellows to the top to make it easy to load and change the way of loading
			

			for (auto& [key, value] : tweets)
			{
				if (key == "asset")
				{
					tweets.at_key("asset").at_key("version").get(element);
					header.version = std::stoi(element.get_string().value().data());

					auto& minVersion = value.at_key("minVersion");
					if (minVersion.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						header.minVersion = minVersion.get_uint64().value();
						if (header.minVersion.value() > header.version)
							return {};
					}

					auto& generator = value.at_key("generator");
					if (generator.error() != simdjson::error_code::NO_SUCH_FIELD)
						header.generator = generator.get_string().value().data();

					auto& copyright = value.at_key("copyright");
					if (copyright.error() != simdjson::error_code::NO_SUCH_FIELD)
						header.copyright = copyright.get_string().value().data();
				}

				
					Buffers and buffer views do not contain type information.
					They simply define the raw data for retrieval from the file.
					Objects within the glTF file (meshes, skins, animations) access buffers
					or buffer views via accessors.
				

				else if (key == "buffers")
				{
					for (auto& buffer : value)
					{
						auto& byteLength = buffer.at_key("byteLength");
						if (byteLength.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto byteLengthVal = byteLength.get_uint64().value();
							if (byteLengthVal < 1)
								continue;
						}
						else
							continue;

						auto& uri = buffer.at_key("uri");
						if (uri.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							std::string_view uriBin = uri.get_string().value();

							const asset::IAssetLoader::SAssetLoadParams params;
							auto buffer_bundle = assetManager->getAsset(rootAssetDirectory + uriBin.data(), params);
							auto buffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer_bundle.getContents().begin()[0]);

							// put
						}
						else
							continue;
					}
				}

				else if (key == "bufferViews")
				{
					for (auto& bufferView : value)
					{
						asset::SBufferBinding<ICPUBuffer> bufferBinding;

						auto& buffer = bufferView.at_key("buffer");
						if (buffer.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& bufferID = buffer.get_uint64().value();
						}
						else
							continue;
					}
				}

				
					 Meshes are defined as arrays of primitives.
					 Primitives correspond to the data required for GPU draw calls
				

				else if (key == "meshes")
				{
					for (auto& mesh : value)
					{
						auto& primitives = mesh.at_key("primitives");
						if (primitives.error() != simdjson::error_code::NO_SUCH_FIELD)
							for (auto& primitive : primitives)
							{
								auto& attributes = primitive.at_key("attributes");
								if (attributes.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& position = attributes.at_key("POSITION");
									auto& normal = attributes.at_key("NORMAL");
									auto& tangent = attributes.at_key("TANGENT");
									auto& texcoord0 = attributes.at_key("TEXCOORD_0");
									auto& texcoord1 = attributes.at_key("TEXCOORD_1");
									auto& color0 = attributes.at_key("COLOR_0");
									auto& joint0 = attributes.at_key("JOINTS_0");
									auto& weight0 = attributes.at_key("WEIGHTS_0");

									// TODO
								}
								else
									continue;
							}
						else
							continue;
					}
				}

				else if (key == "nodes")
				{
					for (auto& node : value)
					{

					}
				}

				
					All large data for meshes, skins, and animations is stored in buffers and retrieved via accessors.
					An accessor defines a method for retrieving data as typed arrays from within a bufferView.
				

				else if (key == "accessors")
				{
					for (auto& accessor : value)
					{
						auto& componentType = accessor.at_key("componentType");
						if (componentType.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& type = componentType.get_uint64().value();

							switch (type)
							{
								case 5120: // BYTE
								{

								} break;

								case 5121: // UNSIGNED_BYTE
								{

								} break;

								case 5122: // SHORT
								{

								} break;

								case 5123: // UNSIGNED_SHORT
								{

								} break;

								case 5124: // UNSIGNED_INT
								{

								} break;

								case 5125: // FLOAT
								{

								} break;

								case 5126:
								{

								} break;

								default:
								{
									return {}; // TODO
								} break;
							}
						}
						else
							continue;

						auto& count = accessor.at_key("count");
						if (count.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& countVal = count.get_uint64().value();
							if (countVal < 1)
								continue;
						}
						else
							continue;


						auto& type = accessor.at_key("type");
						if (type.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& typeVal = type.get_string().value();

							if (typeVal.data() == "SCALAR")
							{

							}
							else if (typeVal.data() == "VEC2")
							{

							}
							else if (typeVal.data() == "VEC3")
							{

							}
							else if (typeVal.data() == "VEC4")
							{

							}
							else if (typeVal.data() == "MAT2")
							{

							}
							else if (typeVal.data() == "MAT3")
							{

							}
							else if (typeVal.data() == "MAT4")
							{

							}
						}
					}
				}

		
					A texture is defined by an image resource, denoted by
					the source property and a sampler index (sampler).
				

				else if (key == "textures")
				{
					for (auto& texture : value)
					{
						auto& sampler = texture.at_key("sampler");
						if (sampler.error() != simdjson::error_code::NO_SUCH_FIELD)
							sampler.get_uint64().value(); // TODO

						auto& source = texture.at_key("source");
						if (source.error() != simdjson::error_code::NO_SUCH_FIELD)
							source.get_uint64().value(); // TODO
					}
				}

				
					Images referred to by textures are stored in the images.
				

				else if (key == "images")
				{
					for (auto& image : value)
					{
						auto& uri = image.at_key("uri");
						if (uri.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							std::string_view uriImage = uri.get_string().value();

							const asset::IAssetLoader::SAssetLoadParams params;
							auto image_bundle = assetManager->getAsset(rootAssetDirectory + uriImage.data(), params);
							auto image = core::smart_refctd_ptr_static_cast<ICPUImage>(image_bundle.getContents().begin()[0]);
						}
						else
							continue;
					}
				}

				
					Each sampler specifies filter and wrapping options corresponding to the GL types.
				

				else if (key == "samplers")
				{
					for (auto& sampler : value)
					{
						auto& magFilter = sampler.at_key("magFilter");
						if (magFilter.error() != simdjson::error_code::NO_SUCH_FIELD)
							magFilter.get_uint64().value(); // TODO

						auto& minFilter = sampler.at_key("minFilter");
						if (minFilter.error() != simdjson::error_code::NO_SUCH_FIELD)
							minFilter.get_uint64().value(); // TODO

						auto& wrapS = sampler.at_key("wrapS");
						if (wrapS.error() != simdjson::error_code::NO_SUCH_FIELD)
							wrapS.get_uint64().value(); // TODO

						auto& wrapT = sampler.at_key("wrapT");
						if (wrapT.error() != simdjson::error_code::NO_SUCH_FIELD)
							wrapT.get_uint64().value(); // TODO
					}
				}

		
					There are materials using a common set of parameters that are based on widely 
					used material representations from Physically-Based Rendering (PBR).

				else if (key == "materials")
				{
					// TODO
				}

					A camera defines the projection matrix that transforms from view to clip coordinates.
				

				else if (key == "cameras")
				{
					for (auto& camera : value)
					{
						auto& type = camera.at_key("type");
						if (type.error() == simdjson::error_code::NO_SUCH_FIELD)
							continue;
						else
						{
							auto& typeVal = type.get_string().value();

							if (typeVal == "perspective")
							{
								auto& perspective = camera.at_key("perspective");
								if (perspective.error() == simdjson::error_code::NO_SUCH_FIELD)
									continue;

								auto& yfov = perspective.at_key("yfov");
								if (yfov.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& yfovVal = yfov.get_double().value();
									if (yfovVal <= 0)
										continue;
								}
								else
									continue;

								auto& znear = perspective.at_key("znear");
								if (znear.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& znearVal = znear.get_double().value();
									if (znearVal <= 0)
										continue;
								}
								else
									continue;

							}
							else if (typeVal == "orthographic")
							{
								auto& orthographic = camera.at_key("orthographic");
								if (orthographic.error() == simdjson::error_code::NO_SUCH_FIELD)
									continue;

								auto& xmag = orthographic.at_key("xmag");
								if (xmag.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& xmagVal = xmag.get_double().value();
								}
								else
									continue;

								auto& ymag = orthographic.at_key("ymag");
								if (ymag.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& ymagVal = ymag.get_double().value();
								}
								else
									continue;

								auto& znear = orthographic.at_key("znear");
								if (znear.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& znearVal = znear.get_double().value();
									if (znearVal < 0)
										continue;
								}
								else
									continue;

								auto& zfar = orthographic.at_key("znear");
								if (zfar.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& zfarVal = zfar.get_double().value();
									if (zfarVal <= 0)
										continue;
								}
								else
									continue;
							}
							else
								continue;
						}
					}
				}

				else if (key == "scenes")
				{
					for (auto& scene : value)
					{
						// TODO
					}
				}

				else if (key == "scene")
				{
					auto& sceneID = value.get_uint64().value(); 
				}
			}

			*/

			return {};
		}

		void CGLTFLoader::loadAndGetGLTF(SGLTF& glTF, io::IReadFile* _file)
		{
			simdjson::dom::parser parser;

			auto jsonBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
			{
				const auto beginPosition = _file->getPos();
				_file->read(jsonBuffer->getPointer(), jsonBuffer->getSize());
				_file->seek(beginPosition);
			}

			simdjson::dom::object tweets = parser.parse(reinterpret_cast<uint8_t*>(jsonBuffer->getPointer()), jsonBuffer->getSize());
			simdjson::dom::element element;

			//std::filesystem::path filePath(_file->getFileName().c_str());
			//const std::string rootAssetDirectory = std::filesystem::absolute(filePath.remove_filename()).u8string();

			auto& extensionsUsed = tweets.at_key("extensionsUsed");
			auto& extensionsRequired = tweets.at_key("extensionsRequired");
			auto& accessors = tweets.at_key("accessors");
			auto& animations = tweets.at_key("animations");
			auto& asset = tweets.at_key("asset");
			auto& buffers = tweets.at_key("buffers");
			auto& bufferViews = tweets.at_key("bufferViews");
			auto& cameras = tweets.at_key("cameras");
			auto& images = tweets.at_key("images");
			auto& materials = tweets.at_key("materials");
			auto& meshes = tweets.at_key("meshes");
			auto& nodes = tweets.at_key("nodes");
			auto& samplers = tweets.at_key("samplers");
			auto& scene = tweets.at_key("scene");
			auto& scenes = tweets.at_key("scenes");
			auto& skins = tweets.at_key("skins");
			auto& textures = tweets.at_key("textures");
			auto& extensions = tweets.at_key("extensions");
			auto& extras = tweets.at_key("extras");

			if (buffers.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& jsonBuffers = buffers.get_array();
				for (auto& jsonBuffer : jsonBuffers)
				{
					auto& glTFBuffer = glTF.buffers.emplace_back();

					auto& uri = jsonBuffer.at_key("uri");
					auto& name = jsonBuffer.at_key("name");
					auto& extensions = jsonBuffer.at_key("extensions");
					auto& extras = jsonBuffer.at_key("extras");

					if (uri.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBuffer.uri = uri.get_string().value();

					if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
						glTFBuffer.name = name.get_string().value();
				}
			}

			if (bufferViews.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& jsonBufferViews = bufferViews.get_array();
				for (auto& jsonBufferView : jsonBufferViews)
				{
					auto& glTFBufferView = glTF.bufferViews.emplace_back();

					auto& buffer = jsonBufferView.at_key("buffer");
					auto& byteOffset = jsonBufferView.at_key("byteOffset");
					auto& byteLength = jsonBufferView.at_key("byteLength");
					auto& byteStride = jsonBufferView.at_key("byteStride");
					auto& target = jsonBufferView.at_key("target");
					auto& name = jsonBufferView.at_key("name");
					auto& extensions = jsonBufferView.at_key("extensions");
					auto& extras = jsonBufferView.at_key("extras");

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

			if (nodes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& nData = nodes.get_array();
				for (size_t iteratorID = 0; iteratorID < nData.size(); ++iteratorID)
				{
					// TODO: fill the node and get down through the tree (mesh, primitives, attributes, buffer views, materials, etc) till the end.

					auto handleTheGLTFTree = [&]()
					{
						auto& glTFnode = glTF.nodes.emplace_back();
						auto& jsonNode = nData.at(iteratorID);

						auto& camera = jsonNode.at_key("camera");
						auto& children = jsonNode.at_key("children");
						auto& skin = jsonNode.at_key("skin");
						auto& matrix = jsonNode.at_key("matrix");
						auto& mesh = jsonNode.at_key("mesh");
						auto& rotation = jsonNode.at_key("rotation");
						auto& scale = jsonNode.at_key("scale");
						auto& translation = jsonNode.at_key("translation");
						auto& weights = jsonNode.at_key("weights");
						auto& name = jsonNode.at_key("name");
						auto& extensions = jsonNode.at_key("extensions");
						auto& extras = jsonNode.at_key("extras");

						if (camera.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.camera = camera.get_uint64().value();

						if (children.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							glTFnode.children.emplace();
							for (auto& child : children)
								glTFnode.children.value().emplace_back() = child.get_uint64().value();
						}

						if (skin.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.skin = skin.get_uint64().value();

						if (matrix.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& matrixArray = matrix.get_array();
							core::matrix4SIMD tmpMatrix;

							for (uint32_t i = 0; i < matrixArray.size(); ++i)
								*(tmpMatrix.pointer() + i) = matrixArray.at(i).get_double().value();

							// TODO tmpMatrix (coulmn major) to row major (currentNode.matrix)

							glTFnode.transformation.matrix = tmpMatrix;
						}
						else
						{
							if (translation.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								auto& translationArray = translation.get_array();
								for (auto& val : translationArray)
								{
									size_t index = &val - &(*translationArray.begin());
									glTFnode.transformation.trs.translation[index] = val.get_double().value();
								}
							}

							if (rotation.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								auto& rotationArray = rotation.get_array();
								for (auto& val : rotationArray)
								{
									size_t index = &val - &(*rotationArray.begin());
									glTFnode.transformation.trs.rotation[index] = val.get_double().value();
								}
							}

							if (scale.error() != simdjson::error_code::NO_SUCH_FIELD)
							{
								auto& scaleArray = scale.get_array();
								for (auto& val : scaleArray)
								{
									size_t index = &val - &(*scaleArray.begin());
									glTFnode.transformation.trs.scale[index] = val.get_double().value();
								}
							}
						}

						if (mesh.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.mesh = mesh.get_uint64().value();

						if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
							glTFnode.name = name.get_string().value();

						// TODO camera, skinning, etc HERE

						if (glTFnode.validate())
						{
							auto& mData = meshes.get_array();
							for (size_t iteratorID = 0; iteratorID < mData.size(); ++iteratorID)
							{
								auto& jsonMesh = meshes.get_array().at(iteratorID);

								if (jsonMesh.error() != simdjson::error_code::NO_SUCH_FIELD)
								{
									auto& glTFMesh = glTFnode.glTFMesh;

									auto& primitives = jsonMesh.at_key("primitives");
									auto& weights = jsonMesh.at_key("weights");
									auto& name = jsonMesh.at_key("name");
									auto& extensions = jsonMesh.at_key("extensions");
									auto& extras = jsonMesh.at_key("extras");

									if (primitives.error() == simdjson::error_code::NO_SUCH_FIELD)
										return false;

									auto& pData = primitives.get_array();
									for (size_t iteratorID = 0; iteratorID < pData.size(); ++iteratorID)
									{
										auto& glTFPrimitive = glTFMesh.primitives.emplace_back();
										auto& jsonPrimitive = pData.at(iteratorID);

										auto& attributes = jsonPrimitive.at_key("attributes");
										auto& indices = jsonPrimitive.at_key("indices");
										auto& material = jsonPrimitive.at_key("material");
										auto& mode = jsonPrimitive.at_key("mode");
										auto& targets = jsonPrimitive.at_key("targets");
										auto& extensions = jsonPrimitive.at_key("extensions");
										auto& extras = jsonPrimitive.at_key("extras");

										if (indices.error() != simdjson::error_code::NO_SUCH_FIELD)
											glTFPrimitive.indices = indices.get_uint64().value();

										if (material.error() != simdjson::error_code::NO_SUCH_FIELD)
											glTFPrimitive.material = material.get_uint64().value();

										if (mode.error() != simdjson::error_code::NO_SUCH_FIELD)
											glTFPrimitive.mode = mode.get_uint64().value();
										else
											glTFPrimitive.mode = 4;

										if (targets.error() != simdjson::error_code::NO_SUCH_FIELD)
											for (auto& [targetKey, targetID] : targets.get_object())
												glTFPrimitive.targets.emplace()[targetKey.data()] = targetID.get_uint64().value();

										auto insertAccessorIntoGLTFCache = [&](const std::string_view& cacheKey, const uint32_t accessorID)
										{
											auto& jsonAccessor = accessors.get_array().at(accessorID);

											if (jsonAccessor.error() != simdjson::NO_SUCH_FIELD)
											{
												auto& glTFAccessor = glTFPrimitive.accessors[cacheKey.data()];

												auto& bufferView = jsonAccessor.at_key("bufferView");
												auto& byteOffset = jsonAccessor.at_key("byteOffset");
												auto& componentType = jsonAccessor.at_key("componentType");
												auto& normalized = jsonAccessor.at_key("normalized");
												auto& count = jsonAccessor.at_key("count");
												auto& type = jsonAccessor.at_key("type");
												auto& max = jsonAccessor.at_key("max");
												auto& min = jsonAccessor.at_key("min");
												auto& sparse = jsonAccessor.at_key("sparse");
												auto& name = jsonAccessor.at_key("name");
												auto& extensions = jsonAccessor.at_key("extensions");
												auto& extras = jsonAccessor.at_key("extras");

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
													glTFAccessor.type = type.get_string().value();

												if (max.error() != simdjson::error_code::NO_SUCH_FIELD)
												{
													glTFAccessor.max.emplace();
													auto& maxArray = max.get_array();
													for (uint32_t i = 0; i < maxArray.size(); ++i)
														glTFAccessor.max.value().push_back(maxArray.at(i).get_double().value());
												}

												if (min.error() != simdjson::error_code::NO_SUCH_FIELD)
												{
													glTFAccessor.min.emplace();
													auto& minArray = min.get_array();
													for (uint32_t i = 0; i < minArray.size(); ++i)
														glTFAccessor.min.value().push_back(minArray.at(i).get_double().value());
												}

												/*
													TODO: in future

													if (sparse.error() != simdjson::error_code::NO_SUCH_FIELD)
														glTFAccessor.sparse = ;
												*/

												if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
													glTFAccessor.name = count.get_string().value();

												if (!glTFAccessor.validate())
													return false;
											}
											else
												return false; // todo
										};

										if (attributes.error() != simdjson::error_code::NO_SUCH_FIELD)
										{
											if (glTFPrimitive.indices.has_value())
												insertAccessorIntoGLTFCache("INDEX", glTFPrimitive.indices.value());

											for (auto& [attributeKey, attributeID] : attributes.get_object())
												insertAccessorIntoGLTFCache(attributeKey, attributeID.get_uint64().value());
										}
										else
											return false;
									}

									// weights - TODO in future

									if (name.error() != simdjson::error_code::NO_SUCH_FIELD)
										glTFMesh.name = name.get_string().value();
								}
								else
								{
									/*
										A node doesnt have a mesh -> it is valid by the documentation of the glTF, but I think the
										loader should do continue, delete the node and handle next node or we should provide the defaults
									*/

									return false;
								}
							}
						}
						else
							return false;
					};

					if (!handleTheGLTFTree())
					{
						glTF.nodes.pop_back();
						continue;
					}
				}
			}
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
