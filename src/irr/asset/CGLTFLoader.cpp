// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CGLTFLoader.h"

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "simdjson/singleheader/simdjson.h"
#include <filesystem>
#include "os.h"

namespace irr
{
	namespace asset
	{
		/*
			Each glTF asset must have an asset property. 
			In fact, it's the only required top-level property
			for JSON to be a valid glTF.
		*/

		bool CGLTFLoader::isALoadableFileFormat(io::IReadFile* _file) const
		{
			simdjson::dom::parser parser;
			simdjson::dom::object tweets = parser.load(_file->getFileName().c_str());
			simdjson::dom::element element;

			if (tweets.at_key("asset").get(element) == simdjson::error_code::SUCCESS)
				if (element.at_key("version").get(element) == simdjson::error_code::SUCCESS)
					return true;

			return false;
		}

		asset::SAssetBundle CGLTFLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			SGLTFData glTfData;

			CGLTFHeader header;
			simdjson::dom::parser parser;
			simdjson::dom::object tweets = parser.load(_file->getFileName().c_str());
			simdjson::dom::element element;

			std::filesystem::path filePath(_file->getFileName().c_str());
			const std::string rootAssetDirectory = std::filesystem::absolute(filePath.remove_filename()).u8string();

			constexpr uint8_t POSITION_ATTRIBUTE = 0;
			constexpr uint8_t NORMAL_ATTRIBUTE = 3;

			auto meshBuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
			meshBuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);
			meshBuffer->setNormalnAttributeIx(NORMAL_ATTRIBUTE);

			auto mbVertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
			auto mbFragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();

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

			if (meshes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& mData = meshes.get_array();
				for (size_t iteratorID = 0; iteratorID < mData.size(); ++iteratorID)
				{
					auto& mesh = mData.at(iteratorID);
					auto& primitives = mesh.at_key("primitives");
					auto& weights = mesh.at_key("weights");
					auto& name = mesh.at_key("name");
					auto& extensions = mesh.at_key("extensions");
					auto& extras = mesh.at_key("extras");

					if (primitives.error() == simdjson::error_code::NO_SUCH_FIELD)
						return {};

					auto& pData = primitives.get_array();
					for (size_t iteratorID = 0; iteratorID < pData.size(); ++iteratorID)
					{
						auto& primitive = pData.at(iteratorID);
						auto& attributes = primitive.at_key("attributes");
						auto& indices = primitive.at_key("indices");
						auto& material = primitive.at_key("material");
						auto& mode = primitive.at_key("mode");
						auto& targets = primitive.at_key("targets");
						auto& extensions = primitive.at_key("extensions");
						auto& extras = primitive.at_key("extras");

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
							return {};

						auto getMode = [&]() -> E_PRIMITIVE_TOPOLOGY
						{
							switch (mode.get_uint64().value())
							{
								case SGLTFPT_POINTS:
									return EPT_POINT_LIST;
								case SGLTFPT_LINES:
									return EPT_LINE_LIST;
								case SGLTFPT_LINE_LOOP:
									return EPT_LINE_LIST_WITH_ADJACENCY; // check it
								case SGLTFPT_LINE_STRIP: 
									return EPT_LINE_STRIP;
								case SGLTFPT_TRIANGLES: 
									return EPT_TRIANGLE_LIST;
								case SGLTFPT_TRIANGLE_STRIP:
									return EPT_TRIANGLE_STRIP;
								case SGLTFPT_TRIANGLE_FAN: 
									return EPT_TRIANGLE_STRIP_WITH_ADJACENCY; // check it
								default:
									return {};
							}
						};

						const E_PRIMITIVE_TOPOLOGY primitiveTopology = mode.error() == simdjson::error_code::NO_SUCH_FIELD ? EPT_TRIANGLE_LIST : getMode();
					}
				}
			}

			if (bufferViews.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& bVData = bufferViews.get_array();
				for (size_t iteratorID = 0; iteratorID < bVData.size(); ++iteratorID)
				{
					auto& bufferView = bVData.at(iteratorID);
					auto& bufferViewBufferID = bufferView.at_key("buffer");

					if (bufferViewBufferID.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& bufferViewBufferIDVal = bufferViewBufferID.get_uint64().value();
						auto& buffer = tweets.at_key("buffers").at(bufferViewBufferIDVal);

						auto& bufferUri = buffer.at_key("uri");
						auto& bufferName = buffer.at_key("name");
						auto& bufferExtensions = buffer.at_key("extensions");
						auto& bufferExtras = buffer.at_key("extras");

						if (bufferUri.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							std::string_view uriBin = bufferUri.get_string().value();

							const asset::IAssetLoader::SAssetLoadParams params;
							auto buffer_bundle = assetManager->getAsset(rootAssetDirectory + uriBin.data(), params);
							auto cpuBuffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buffer_bundle.getContents().begin()[0]);

							auto& bufferByteOffset = bufferView.at_key("byteOffset");

							SBufferBinding<ICPUBuffer> bufferBinding;
							bufferBinding.offset = bufferByteOffset.error() == simdjson::error_code::NO_SUCH_FIELD ? 0 : bufferByteOffset.get_uint64().value();
							bufferBinding.buffer = cpuBuffer;

							meshBuffer->setVertexBufferBinding(std::move(bufferBinding), iteratorID);

							auto& bufferViewByteLength = bufferView.at_key("byteLength");
							auto& bufferViewByteStride = bufferView.at_key("byteStride");
							auto& bufferViewTarget = bufferView.at_key("target");
							auto& bufferViewName = bufferView.at_key("name");
							auto& bufferViewExtensions = bufferView.at_key("extensions");
							auto& bufferViewExtras = bufferView.at_key("extras");

							// TODO

						}
					}
					else
						continue;
				}
			}

			if (accessors.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& acData = accessors.get_array();
				for (size_t iteratorID = 0; iteratorID < acData.size(); ++iteratorID)
				{
					auto& accessor = acData.at(iteratorID);
					auto& accessorBufferView = accessor.at_key("bufferView");
					auto& accessorByteOffset = accessor.at_key("byteOffset");
					auto& accessorComponentType = accessor.at_key("componentType");
					auto& accessorCount = accessor.at_key("count");
					auto& accessorType = accessor.at_key("type");
					auto& accessorMax = accessor.at_key("max");
					auto& accessorMin = accessor.at_key("min");
					auto& accessorSparse = accessor.at_key("sparse");
					auto& accessorName = accessor.at_key("name");
					auto& accessorExtensions = accessor.at_key("extensions");
					auto& accessorExtras = accessor.at_key("extras");

					if (accessorComponentType.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& type = accessorComponentType.get_uint64().value();

						switch (type)
						{
							case SGLTFAccessor::SCT_BYTE:
							{

							} break;

							case SGLTFAccessor::SCT_UNSIGNED_BYTE:
							{

							} break;

							case SGLTFAccessor::SCT_SHORT:
							{

							} break;

							case SGLTFAccessor::SCT_UNSIGNED_SHORT:
							{

							} break;

							case SGLTFAccessor::SCT_UNSIGNED_INT:
							{

							} break;

							case SGLTFAccessor::SCT_FLOAT:
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

					if (accessorCount.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& countVal = accessorCount.get_uint64().value();
						if (countVal < 1)
							continue;
					}
					else
						continue;

					if (accessorType.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& typeVal = accessorType.get_string().value();

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

			if (nodes.error() != simdjson::error_code::NO_SUCH_FIELD)
			{
				auto& nData = nodes.get_array();
				for (size_t iteratorID = 0; iteratorID < nData.size(); ++iteratorID)
				{
					auto& node = nData.at(iteratorID);

					auto& camera = node.at_key("camera");
					auto& children = node.at_key("children");
					auto& skin = node.at_key("skin");
					auto& matrix = node.at_key("matrix");
					auto& mesh = node.at_key("mesh");
					auto& rotation = node.at_key("rotation");
					auto& scale = node.at_key("scale");
					auto& translation = node.at_key("translation");
					auto& weights = node.at_key("weights");
					auto& name = node.at_key("name");
					auto& extensions = node.at_key("extensions");
					auto& extras = node.at_key("extras");

					auto& currentNode = glTfData.nodes.emplace_back();

					if (camera.error() != simdjson::error_code::NO_SUCH_FIELD)
						currentNode.camera = camera.get_uint64().value();

					if (children.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						currentNode.children = {};
						for (auto& child : children)
							currentNode.children.value().emplace_back() = child.get_uint64().value();
					}

					if (skin.error() != simdjson::error_code::NO_SUCH_FIELD)
						currentNode.skin = skin.get_uint64().value();

					if (matrix.error() != simdjson::error_code::NO_SUCH_FIELD)
					{
						auto& matrixArray = matrix.get_array();
						core::matrix4SIMD tmpMatrix;

						memcpy(tmpMatrix.pointer(), &(*matrixArray.begin()), matrixArray.size() * sizeof(float)); // TODO check it out
						// TODO tmpMatrix (coulmn major) to row major (currentNode.matrix)
					}
					else
					{
						if (translation.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& translationArray = translation.get_array();
							for (auto& val : translationArray)
							{
								size_t index = &val - &(*translationArray.begin());
								currentNode.transformation.trs.translation[index] = val.get_double().value();
							}
						}

						if (rotation.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& rotationArray = rotation.get_array();
							for (auto& val : rotationArray)
							{
								size_t index = &val - &(*rotationArray.begin());
								currentNode.transformation.trs.rotation[index] = val.get_double().value();
							}
						}

						if (scale.error() != simdjson::error_code::NO_SUCH_FIELD)
						{
							auto& scaleArray = scale.get_array();
							for (auto& val : scaleArray)
							{
								size_t index = &val - &(*scaleArray.begin());
								currentNode.transformation.trs.scale[index] = val.get_double().value();
							}
						}
					}
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
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
