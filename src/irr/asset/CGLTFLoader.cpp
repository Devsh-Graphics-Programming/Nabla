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
			CGLTFHeader header;
			simdjson::dom::parser parser;
			simdjson::dom::object tweets = parser.load(_file->getFileName().c_str());
			simdjson::dom::element element;

			std::filesystem::path filePath(_file->getFileName().c_str());
			const std::string rootAssetDirectory = std::filesystem::absolute(filePath.remove_filename()).u8string();

			constexpr uint8_t POSITION_ATTRIBUTE = 0;
			constexpr uint8_t NORMAL_ATTRIBUTE = 3;

			auto mesh = core::make_smart_refctd_ptr<CCPUMesh>();
			auto mbVertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
			auto mbFragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();

			for (auto& [key, value] : tweets)
			{
				auto meshbuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
				meshbuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);
				meshbuffer->setNormalnAttributeIx(NORMAL_ATTRIBUTE);

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

				/*
					Buffers and buffer views do not contain type information.
					They simply define the raw data for retrieval from the file.
					Objects within the glTF file (meshes, skins, animations) access buffers
					or buffer views via accessors.
				*/

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
							buffer.get_uint64().value(); // TODO
						else
							continue;
					}
				}

				/*
					 Meshes are defined as arrays of primitives.
					 Primitives correspond to the data required for GPU draw calls
				*/

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

				/*
					All large data for meshes, skins, and animations is stored in buffers and retrieved via accessors.
					An accessor defines a method for retrieving data as typed arrays from within a bufferView.
				*/

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

				/*
					A texture is defined by an image resource, denoted by
					the source property and a sampler index (sampler).
				*/

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

				/*
					Images referred to by textures are stored in the images.
				*/

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

				/*
					Each sampler specifies filter and wrapping options corresponding to the GL types.
				*/

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

				/*
					There are materials using a common set of parameters that are based on widely 
					used material representations from Physically-Based Rendering (PBR).
				*/

				else if (key == "materials")
				{
					// TODO
				}

				/*
					A camera defines the projection matrix that transforms from view to clip coordinates.
				*/

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

			return {};
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
