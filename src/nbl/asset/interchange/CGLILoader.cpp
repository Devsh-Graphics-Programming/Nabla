/*
MIT License
Copyright (c) 2019 AnastaZIuk
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "CGLILoader.h"

#ifdef _NBL_COMPILE_WITH_GLI_LOADER_

#include "os.h"

#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

#ifdef _NBL_COMPILE_WITH_GLI_
#include "gli/gli.hpp"
#else
#error "It requires GLI library"
#endif

namespace nbl
{
	namespace asset
	{
		static inline std::pair<E_FORMAT, ICPUImageView::SComponentMapping> getTranslatedGLIFormat(const gli::texture& texture, const gli::gl& glVersion);
		static inline void assignGLIDataToRegion(void* regionData, const gli::texture& texture, const uint16_t layer, const uint16_t face, const uint16_t level, const uint64_t sizeOfData);
		static inline bool performLoadingAsIFile(gli::texture& texture, system::IFile* file, system::ISystem* sys);

		asset::SAssetBundle CGLILoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			gli::texture texture;
			

			if (!performLoadingAsIFile(texture, _file, m_system.get()))
				return {};

		    const gli::gl glVersion(gli::gl::PROFILE_GL33);
			const auto target = glVersion.translate(texture.target());
			const auto format = getTranslatedGLIFormat(texture, glVersion);
			IImage::E_TYPE imageType;
			IImageView<ICPUImage>::E_TYPE imageViewType;

			if (format.first == EF_UNKNOWN)
				return {};

			switch (texture.target())
			{
				case gli::TARGET_1D:
				{
					imageType = IImage::ET_1D;
					imageViewType = ICPUImageView::ET_1D;
					break;
				}
				case gli::TARGET_1D_ARRAY:
				{
					imageType = IImage::ET_1D;
					imageViewType = ICPUImageView::ET_1D_ARRAY;
					break;
				}
				case gli::TARGET_2D:
				{
					imageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_2D;
					break;
				}
				case gli::TARGET_2D_ARRAY:
				{
					imageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_2D_ARRAY;
					break;
				}
				case gli::TARGET_3D:
				{
					imageType = IImage::ET_3D;
					imageViewType = ICPUImageView::ET_3D;
					break;
				}
				case gli::TARGET_CUBE:
				{
					imageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_CUBE_MAP;
					break;
				}
				case gli::TARGET_CUBE_ARRAY:
				{
					imageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_CUBE_MAP_ARRAY;
					break;
				}
				default:
				{
					imageViewType = ICPUImageView::ET_COUNT;
					assert(0);
					break;
				}
			}

			const bool isItACubemap = doesItHaveFaces(imageViewType);
			const bool layersFlag = doesItHaveLayers(imageViewType);

			const auto texelBlockDimension = asset::getBlockDimensions(format.first);
			const auto texelBlockByteSize = asset::getTexelOrBlockBytesize(format.first);
			auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texture.size());
			auto data = reinterpret_cast<uint8_t*>(texelBuffer->getPointer());

			ICPUImage::SCreationParams imageInfo;
			imageInfo.format = format.first;
			imageInfo.type = imageType;
			imageInfo.flags = isItACubemap ? ICPUImage::E_CREATE_FLAGS::ECF_CUBE_COMPATIBLE_BIT : static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
			imageInfo.samples = ICPUImage::ESCF_1_BIT;
			imageInfo.extent.width = texture.extent().x;
			imageInfo.extent.height = texture.extent().y;
			imageInfo.extent.depth = texture.extent().z;
			imageInfo.mipLevels = texture.levels();
			imageInfo.arrayLayers = texture.faces() * texture.layers();

			auto image = ICPUImage::create(std::move(imageInfo));

			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(imageInfo.mipLevels);

			auto getFullSizeOfRegion = [&](const uint16_t mipLevel) -> uint64_t
			{
				return texture.size(mipLevel) * imageInfo.arrayLayers;
			};

			auto getFullSizeOfLayer = [&](const uint16_t mipLevel) -> uint64_t
			{
				return texture.size(mipLevel);
			};

			{
				uint16_t regionIndex = {};
				uint64_t offset = {};
				for (auto region = regions->begin(); region != regions->end(); ++region)
				{
					region->imageExtent.width = texture.extent(regionIndex).x;
					region->imageExtent.height = texture.extent(regionIndex).y;
					region->imageExtent.depth = texture.extent(regionIndex).z;
					region->bufferRowLength = region->imageExtent.width;
					region->bufferImageHeight = 0u;
					region->imageSubresource.mipLevel = regionIndex;
					region->imageSubresource.layerCount = imageInfo.arrayLayers;
					region->imageSubresource.baseArrayLayer = 0;
					region->bufferOffset = offset;

					offset += getFullSizeOfRegion(regionIndex);
					++regionIndex;
				}
			}

			auto getCurrentGliLayerAndFace = [&](uint16_t layer)
			{
				static uint16_t gliLayer, gliFace;

				if (isItACubemap)
				{
					gliFace = layer % 6;

					if (layersFlag)
						gliLayer = layer / 6;
					else
						gliLayer = 0;
				}
				else
				{
					gliFace = 0;
					gliLayer = layer;
				}

				return std::make_pair(gliLayer, gliFace);
			};

			uint64_t tmpDataSizePerRegionSum = {};
			for (uint16_t mipLevel = 0; mipLevel < imageInfo.mipLevels; ++mipLevel)
			{
				const auto layerSize = getFullSizeOfLayer(mipLevel);
				for (uint16_t layer = 0; layer < imageInfo.arrayLayers; ++layer)
				{
					const auto layersData = getCurrentGliLayerAndFace(layer);
					const auto gliLayer = layersData.first;
					const auto gliFace = layersData.second;

					assignGLIDataToRegion((reinterpret_cast<uint8_t*>(data) + tmpDataSizePerRegionSum + (layer * layerSize)), texture, gliLayer, gliFace, mipLevel, layerSize);
				}
				tmpDataSizePerRegionSum += getFullSizeOfRegion(mipLevel);
			}

			image->setBufferAndRegions(std::move(texelBuffer), regions);

			if (imageInfo.format == asset::EF_R8_SRGB)
				image = IImageAssetHandlerBase::convertR8ToR8G8B8Image(image);

			ICPUImageView::SCreationParams imageViewInfo;
			imageViewInfo.image = std::move(image);
			imageViewInfo.format = imageViewInfo.image->getCreationParameters().format;
			imageViewInfo.viewType = imageViewType;
			imageViewInfo.components = format.second;
			imageViewInfo.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			imageViewInfo.subresourceRange.baseArrayLayer = 0u;
			imageViewInfo.subresourceRange.baseMipLevel = 0u;
			imageViewInfo.subresourceRange.layerCount = imageInfo.arrayLayers;
			imageViewInfo.subresourceRange.levelCount = imageInfo.mipLevels;

			auto imageView = ICPUImageView::create(std::move(imageViewInfo));

			return SAssetBundle(nullptr,{std::move(imageView)});
		}

		bool performLoadingAsIFile(gli::texture& texture, system::IFile* file, system::ISystem* sys)
		{
			const auto fileName = file->getFileName().string();
			core::vector<char> memory(file->getSize());
			const auto sizeOfData = memory.size();

			system::ISystem::future_t<uint32_t> future;
			sys->readFile(future, file, memory.data(), 0, sizeOfData);
			future.get();

			if (fileName.rfind(".dds") != std::string::npos)
				texture = gli::load_dds(memory.data(), sizeOfData);
			else if (fileName.rfind(".kmg") != std::string::npos)
				texture = gli::load_kmg(memory.data(), sizeOfData);
			else if (fileName.rfind(".ktx") != std::string::npos)
				texture = gli::load_ktx(memory.data(), sizeOfData);

			if (!texture.empty())
				return true;
			else
			{
				os::Printer::log("LOADING GLI: failed to load the file", file->getFileName().string(), ELL_ERROR);
				return false;
			}
		}

		bool CGLILoader::isALoadableFileFormat(system::IFile* _file) const
		{
			const auto fileName = std::string(_file->getFileName().string());

			constexpr auto ddsMagic = 0x20534444;
			constexpr std::array<uint8_t, 12> ktxMagic = { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
			constexpr std::array<uint8_t, 16> kmgMagic = { 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55 };

			system::ISystem::future_t<uint32_t> future;
			if (fileName.rfind(".dds") != std::string::npos)
			{
				std::remove_const<decltype(ddsMagic)>::type tmpBuffer;
				m_system->readFile(future, _file, &tmpBuffer, 0, sizeof(ddsMagic));
				future.get();
				if (*reinterpret_cast<decltype(ddsMagic)*>(&tmpBuffer) == ddsMagic)
				{
					return true;
				}
				else
					os::Printer::log("LOAD GLI: Invalid (non-DDS) file!", ELL_ERROR);
			}
			else if (fileName.rfind(".kmg") != std::string::npos)
			{
				std::remove_const<decltype(kmgMagic)>::type tmpBuffer;
				m_system->readFile(future, _file, tmpBuffer.data(), 0, sizeof(kmgMagic[0]) * kmgMagic.size());
				future.get();
				if (tmpBuffer == kmgMagic)
				{
					return true;
				}
				else
					os::Printer::log("LOAD GLI: Invalid (non-KMG) file!", ELL_ERROR);
			}
			else if (fileName.rfind(".ktx") != std::string::npos)
			{
				std::remove_const<decltype(ktxMagic)>::type tmpBuffer;
				m_system->readFile(future, _file, tmpBuffer.data(), 0, sizeof(ktxMagic[0]) * ktxMagic.size());
				future.get();
				if (tmpBuffer == ktxMagic)
				{
					return true;
				}
				else
					os::Printer::log("LOAD GLI: Invalid (non-KTX) file!", ELL_ERROR);
			}
			
			return false;
		}

		inline std::pair<E_FORMAT, ICPUImageView::SComponentMapping> getTranslatedGLIFormat(const gli::texture& texture, const gli::gl& glVersion)
		{
			using namespace gli;
			gli::gl::format formatToTranslate = glVersion.translate(texture.format(), texture.swizzles());
			ICPUImageView::SComponentMapping compomentMapping;

			static const core::unordered_map<gli::gl::swizzle, ICPUImageView::SComponentMapping::E_SWIZZLE> swizzlesMappingAPI =
			{
				std::make_pair(gl::SWIZZLE_RED, ICPUImageView::SComponentMapping::ES_R),
				std::make_pair(gl::SWIZZLE_GREEN, ICPUImageView::SComponentMapping::ES_G),
				std::make_pair(gl::SWIZZLE_BLUE, ICPUImageView::SComponentMapping::ES_B),
				std::make_pair(gl::SWIZZLE_ALPHA, ICPUImageView::SComponentMapping::ES_A),
				std::make_pair(gl::SWIZZLE_ONE, ICPUImageView::SComponentMapping::ES_ONE),
				std::make_pair(gl::SWIZZLE_ZERO, ICPUImageView::SComponentMapping::ES_ZERO)
			};

			auto getMappedSwizzle = [&](const gli::gl::swizzle& currentSwizzleToCheck)
			{
				for (auto& mappedSwizzle : swizzlesMappingAPI)
					if (currentSwizzleToCheck == mappedSwizzle.first)
						return mappedSwizzle.second;
				return ICPUImageView::SComponentMapping::ES_IDENTITY;
			};

			compomentMapping.r = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.r));
			compomentMapping.g = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.g));
			compomentMapping.b = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.b));
			compomentMapping.a = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.a));

			auto getTranslatedFinalFormat = [&](const E_FORMAT& format = EF_UNKNOWN, const std::string_view &specialErrorOnUnknown = "Unsupported format!")
			{
				if(format == EF_UNKNOWN)
					os::Printer::log(("LOAD GLI: " + std::string(specialErrorOnUnknown)).c_str(), ELL_ERROR);

				return std::make_pair(format, compomentMapping);
			};

			constexpr auto alphaPVRTCError = "Vulkan isn't able to distinguish between alpha and non alpha containing PVRTC";
			constexpr auto vulcanVertexFormatsError = "Vulkan's vertex formats are disallowed for textures";

			switch (formatToTranslate.Internal)
			{
				// "///" means a format doesn't fit in Power-Of-Two for a texel/block (there is an exception in Power-Of-Two rule - 24bit RGB basic format)

				case gl::INTERNAL_RGB_UNORM: return getTranslatedFinalFormat(EF_R8G8B8_UNORM);			//GL_RGB
				case gl::INTERNAL_BGR_UNORM: return getTranslatedFinalFormat(EF_B8G8R8_UNORM);			//GL_BGR
				case gl::INTERNAL_RGBA_UNORM: return getTranslatedFinalFormat(EF_R8G8B8A8_UNORM);		//GL_RGBA
				case gl::INTERNAL_BGRA_UNORM: return getTranslatedFinalFormat(EF_B8G8R8A8_UNORM);		//GL_BGRA
				case gl::INTERNAL_BGRA8_UNORM: return getTranslatedFinalFormat(EF_B8G8R8A8_UNORM);		//GL_BGRA8_EXT

				// unorm formats
				case gl::INTERNAL_R8_UNORM: return getTranslatedFinalFormat(EF_R8_UNORM);			//GL_R8
				case gl::INTERNAL_RG8_UNORM: return getTranslatedFinalFormat(EF_R8G8_UNORM);		//GL_RG8
				case gl::INTERNAL_RGB8_UNORM: return getTranslatedFinalFormat(EF_R8G8B8_UNORM);		//GL_RGB8
				case gl::INTERNAL_RGBA8_UNORM: return getTranslatedFinalFormat(EF_R8G8B8A8_UNORM);		//GL_RGBA8

				case gl::INTERNAL_R16_UNORM: return getTranslatedFinalFormat(EF_R16_UNORM);		//GL_R16
				case gl::INTERNAL_RG16_UNORM: return getTranslatedFinalFormat(EF_R16G16_UNORM);		//GL_RG16
				///case gl::INTERNAL_RGB16_UNORM: return getTranslatedFinalFormat(EF_R16G16B16_UNORM);		//GL_RGB16
				case gl::INTERNAL_RGBA16_UNORM: return getTranslatedFinalFormat(EF_R16G16B16A16_UNORM);		//GL_RGBA16

				case gl::INTERNAL_RGB10A2_UNORM: return getTranslatedFinalFormat(EF_A2R10G10B10_UNORM_PACK32);	//GL_RGB10_A2
				case gl::INTERNAL_RGB10A2_SNORM_EXT: return getTranslatedFinalFormat(EF_A2R10G10B10_SNORM_PACK32);

				// snorm formats
				case gl::INTERNAL_R8_SNORM: return getTranslatedFinalFormat(EF_R8_SNORM);			//GL_R8_SNORM
				case gl::INTERNAL_RG8_SNORM: return getTranslatedFinalFormat(EF_R8G8_SNORM);		//GL_RG8_SNORM
				case gl::INTERNAL_RGB8_SNORM: return getTranslatedFinalFormat(EF_R8G8B8_SNORM);		//GL_RGB8_SNORM
				case gl::INTERNAL_RGBA8_SNORM: return getTranslatedFinalFormat(EF_R8G8B8A8_SNORM);		//GL_RGBA8_SNORM

				case gl::INTERNAL_R16_SNORM: return getTranslatedFinalFormat(EF_R16_SNORM);		//GL_R16_SNORM
				case gl::INTERNAL_RG16_SNORM: return getTranslatedFinalFormat(EF_R16G16_SNORM);		//GL_RG16_SNORM
				///case gl::INTERNAL_RGB16_SNORM: return getTranslatedFinalFormat(EF_R16G16B16_SNORM);		//GL_RGB16_SNORM
				case gl::INTERNAL_RGBA16_SNORM: return getTranslatedFinalFormat(EF_R16G16B16A16_SNORM);		//GL_RGBA16_SNORM

				// unsigned integer formats
				case gl::INTERNAL_R8U: return getTranslatedFinalFormat(EF_R8_UINT);				//GL_R8UI
				case gl::INTERNAL_RG8U: return getTranslatedFinalFormat(EF_R8G8_UINT);				//GL_RG8UI
				case gl::INTERNAL_RGB8U: return getTranslatedFinalFormat(EF_R8G8B8_UINT);			//GL_RGB8UI
				case gl::INTERNAL_RGBA8U: return getTranslatedFinalFormat(EF_R8G8B8A8_UINT);			//GL_RGBA8UI

				case gl::INTERNAL_R16U: return getTranslatedFinalFormat(EF_R16_UINT);				//GL_R16UI
				case gl::INTERNAL_RG16U: return getTranslatedFinalFormat(EF_R16G16_UINT);			//GL_RG16UI
				///case gl::INTERNAL_RGB16U: return getTranslatedFinalFormat(EF_R16G16B16_UINT);			//GL_RGB16UI
				case gl::INTERNAL_RGBA16U: return getTranslatedFinalFormat(EF_R16G16B16A16_UINT);			//GL_RGBA16UI

				case gl::INTERNAL_R32U: return getTranslatedFinalFormat(EF_R32_UINT);				//GL_R32UI
				case gl::INTERNAL_RG32U: return getTranslatedFinalFormat(EF_R32G32_UINT);			//GL_RG32UI
				///case gl::INTERNAL_RGB32U: return getTranslatedFinalFormat(EF_R32G32B32_UINT);			//GL_RGB32UI
				case gl::INTERNAL_RGBA32U: return getTranslatedFinalFormat(EF_R32G32B32A32_UINT);			//GL_RGBA32UI

				case gl::INTERNAL_RGB10A2U: return getTranslatedFinalFormat(EF_A2R10G10B10_UINT_PACK32);			//GL_RGB10_A2UI
				case gl::INTERNAL_RGB10A2I_EXT: return getTranslatedFinalFormat(EF_A2R10G10B10_SINT_PACK32);	//GL_RGB10_A2I

				// signed integer formats
				case gl::INTERNAL_R8I: return getTranslatedFinalFormat(EF_R8_SINT);				//GL_R8I
				case gl::INTERNAL_RG8I: return getTranslatedFinalFormat(EF_R8G8_SINT);				//GL_RG8I
				case gl::INTERNAL_RGB8I: return getTranslatedFinalFormat(EF_R8G8B8_SINT);			//GL_RGB8I
				case gl::INTERNAL_RGBA8I: return getTranslatedFinalFormat(EF_R8G8B8A8_SINT);			//GL_RGBA8I

				case gl::INTERNAL_R16I: return getTranslatedFinalFormat(EF_R16_SINT);				//GL_R16I
				case gl::INTERNAL_RG16I: return getTranslatedFinalFormat(EF_R16G16_SINT);			//GL_RG16I
				///case gl::INTERNAL_RGB16I: return getTranslatedFinalFormat(EF_R16G16B16_SINT);			//GL_RGB16I
				case gl::INTERNAL_RGBA16I: return getTranslatedFinalFormat(EF_R16G16B16A16_SINT);			//GL_RGBA16I

				case gl::INTERNAL_R32I: return getTranslatedFinalFormat(EF_R32_SINT);				//GL_R32I
				case gl::INTERNAL_RG32I: return getTranslatedFinalFormat(EF_R32G32_SINT);			//GL_RG32I
				///case gl::INTERNAL_RGB32I: return getTranslatedFinalFormat(EF_R32G32B32_SINT);			//GL_RGB32I
				case gl::INTERNAL_RGBA32I: return getTranslatedFinalFormat(EF_R32G32B32A32_SINT);			//GL_RGBA32I

				// Floating formats
				case gl::INTERNAL_R16F: return getTranslatedFinalFormat(EF_R16_SFLOAT);				//GL_R16F
				case gl::INTERNAL_RG16F: return getTranslatedFinalFormat(EF_R16G16_SFLOAT);			//GL_RG16F
				///case gl::INTERNAL_RGB16F: return getTranslatedFinalFormat(EF_R16G16B16_SFLOAT);			//GL_RGB16F
				case gl::INTERNAL_RGBA16F: return getTranslatedFinalFormat(EF_R16G16B16A16_SFLOAT);			//GL_RGBA16F

				case gl::INTERNAL_R32F: return getTranslatedFinalFormat(EF_R32_SFLOAT);				//GL_R32F
				case gl::INTERNAL_RG32F: return getTranslatedFinalFormat(EF_R32G32_SFLOAT);			//GL_RG32F
				///case gl::INTERNAL_RGB32F: return getTranslatedFinalFormat(EF_R32G32B32_SFLOAT);			//GL_RGB32F
				case gl::INTERNAL_RGBA32F: return getTranslatedFinalFormat(EF_R32G32B32A32_SFLOAT);			//GL_RGBA32F

				case gl::INTERNAL_R64F_EXT: return getTranslatedFinalFormat(EF_R64_SFLOAT);			//GL_R64F
				case gl::INTERNAL_RG64F_EXT: return getTranslatedFinalFormat(EF_R64G64_SFLOAT);		//GL_RG64F
				///case gl::INTERNAL_RGB64F_EXT: return getTranslatedFinalFormat(EF_R64G64B64_SFLOAT);		//GL_RGB64F
				case gl::INTERNAL_RGBA64F_EXT: return getTranslatedFinalFormat(EF_R64G64B64A64_SFLOAT);		//GL_RGBA64F

				// sRGB formats
				case gl::INTERNAL_SR8: return getTranslatedFinalFormat(EF_R8_SRGB);				//GL_SR8_EXT
				case gl::INTERNAL_SRG8: return getTranslatedFinalFormat(EF_R8G8_SRGB);				//GL_SRG8_EXT
				case gl::INTERNAL_SRGB8: return getTranslatedFinalFormat(EF_R8G8B8_SRGB);			//GL_SRGB8
				case gl::INTERNAL_SRGB8_ALPHA8: return getTranslatedFinalFormat(EF_R8G8B8A8_SRGB);		//GL_SRGB8_ALPHA8

				// Packed formats
				case gl::INTERNAL_RGB9E5: return getTranslatedFinalFormat(EF_E5B9G9R9_UFLOAT_PACK32);			//GL_RGB9_E5
				case gl::INTERNAL_RG11B10F: return getTranslatedFinalFormat(EF_B10G11R11_UFLOAT_PACK32);			//GL_R11F_G11F_B10F
				case gl::INTERNAL_RG3B2: return getTranslatedFinalFormat(/*we dont have such format*/);			//GL_R3_G3_B2
				case gl::INTERNAL_R5G6B5: return getTranslatedFinalFormat(EF_R5G6B5_UNORM_PACK16);			//GL_RGB565
				case gl::INTERNAL_RGB5A1: return getTranslatedFinalFormat(EF_R5G5B5A1_UNORM_PACK16);			//GL_RGB5_A1
				case gl::INTERNAL_RGBA4: return getTranslatedFinalFormat(EF_R4G4B4A4_UNORM_PACK16);			//GL_RGBA4

				case gl::INTERNAL_RG4_EXT: return getTranslatedFinalFormat(EF_R4G4_UNORM_PACK8);

				// Luminance Alpha formats
				case gl::INTERNAL_LA4: return getTranslatedFinalFormat(EF_R4G4B4A4_UNORM_PACK16);				//GL_LUMINANCE4_ALPHA4
				case gl::INTERNAL_L8: return getTranslatedFinalFormat(EF_R8G8B8_SRGB);				//GL_LUMINANCE8
				case gl::INTERNAL_A8: return getTranslatedFinalFormat(EF_R8G8B8_SRGB);				//GL_ALPHA8
				case gl::INTERNAL_LA8: return getTranslatedFinalFormat(EF_R8G8B8A8_SRGB);				//GL_LUMINANCE8_ALPHA8
				case gl::INTERNAL_L16: return getTranslatedFinalFormat(EF_R16G16B16_UNORM);				//GL_LUMINANCE16
				case gl::INTERNAL_A16: return getTranslatedFinalFormat(EF_R16G16B16_UNORM);				//GL_ALPHA16
				case gl::INTERNAL_LA16: return getTranslatedFinalFormat(EF_R8G8B8A8_SRGB);				//GL_LUMINANCE16_ALPHA16

				// Depth formats
				case gl::INTERNAL_D16: return getTranslatedFinalFormat(EF_D16_UNORM);				//GL_DEPTH_COMPONENT16
				case gl::INTERNAL_D24: return getTranslatedFinalFormat(EF_X8_D24_UNORM_PACK32);				//GL_DEPTH_COMPONENT24
				case gl::INTERNAL_D16S8_EXT: return getTranslatedFinalFormat(EF_D16_UNORM_S8_UINT);
				case gl::INTERNAL_D24S8: return getTranslatedFinalFormat(EF_D24_UNORM_S8_UINT);			//GL_DEPTH24_STENCIL8
				case gl::INTERNAL_D32: return getTranslatedFinalFormat(EF_D32_SFLOAT_S8_UINT /*?*/);				//GL_DEPTH_COMPONENT32
				case gl::INTERNAL_D32F: return getTranslatedFinalFormat(/*EF_D32_SFLOAT signed probably won't fit it?*/);				//GL_DEPTH_COMPONENT32F
				case gl::INTERNAL_D32FS8X24: return getTranslatedFinalFormat(EF_D32_SFLOAT_S8_UINT /*?*/);		//GL_DEPTH32F_STENCIL8
				case gl::INTERNAL_S8_EXT: return getTranslatedFinalFormat(EF_S8_UINT);			//GL_STENCIL_INDEX8

				// Compressed formats
				case gl::INTERNAL_RGB_DXT1: return getTranslatedFinalFormat(EF_BC1_RGB_UNORM_BLOCK);						//GL_COMPRESSED_RGB_S3TC_DXT1_EXT
				case gl::INTERNAL_RGBA_DXT1: return getTranslatedFinalFormat(EF_BC1_RGBA_UNORM_BLOCK);					//GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
				case gl::INTERNAL_RGBA_DXT3: return getTranslatedFinalFormat(EF_BC2_UNORM_BLOCK);					//GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
				case gl::INTERNAL_RGBA_DXT5: return getTranslatedFinalFormat(EF_BC3_UNORM_BLOCK);					//GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
				case gl::INTERNAL_R_ATI1N_UNORM: return getTranslatedFinalFormat(EF_BC4_UNORM_BLOCK);				//GL_COMPRESSED_RED_RGTC1
				case gl::INTERNAL_R_ATI1N_SNORM: return getTranslatedFinalFormat(EF_BC4_SNORM_BLOCK);				//GL_COMPRESSED_SIGNED_RED_RGTC1
				case gl::INTERNAL_RG_ATI2N_UNORM: return getTranslatedFinalFormat(EF_BC5_UNORM_BLOCK);				//GL_COMPRESSED_RG_RGTC2
				case gl::INTERNAL_RG_ATI2N_SNORM: return getTranslatedFinalFormat(EF_BC5_SNORM_BLOCK);				//GL_COMPRESSED_SIGNED_RG_RGTC2
				case gl::INTERNAL_RGB_BP_UNSIGNED_FLOAT: return getTranslatedFinalFormat(EF_BC6H_UFLOAT_BLOCK);		//GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
				case gl::INTERNAL_RGB_BP_SIGNED_FLOAT: return getTranslatedFinalFormat(EF_BC6H_SFLOAT_BLOCK);			//GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT
				case gl::INTERNAL_RGB_BP_UNORM: return getTranslatedFinalFormat(EF_BC7_UNORM_BLOCK);					//GL_COMPRESSED_RGBA_BPTC_UNORM
				case gl::INTERNAL_RGB_PVRTC_4BPPV1: return getTranslatedFinalFormat(EF_UNKNOWN, alphaPVRTCError);				//GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG
				case gl::INTERNAL_RGB_PVRTC_2BPPV1: return getTranslatedFinalFormat(EF_UNKNOWN, alphaPVRTCError);				//GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_4BPPV1: return getTranslatedFinalFormat(EF_PVRTC1_4BPP_UNORM_BLOCK_IMG);			//GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_2BPPV1: return getTranslatedFinalFormat(EF_PVRTC1_2BPP_UNORM_BLOCK_IMG);			//GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_4BPPV2: return getTranslatedFinalFormat(EF_PVRTC1_4BPP_UNORM_BLOCK_IMG);			//GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG
				case gl::INTERNAL_RGBA_PVRTC_2BPPV2: return getTranslatedFinalFormat(EF_PVRTC1_2BPP_UNORM_BLOCK_IMG);			//GL_COMPRESSED_RGBA_PVRTC_2BPPV2_IMG
				case gl::INTERNAL_ATC_RGB: return getTranslatedFinalFormat();						//GL_ATC_RGB_AMD
				case gl::INTERNAL_ATC_RGBA_EXPLICIT_ALPHA: return getTranslatedFinalFormat();		//GL_ATC_RGBA_EXPLICIT_ALPHA_AMD
				case gl::INTERNAL_ATC_RGBA_INTERPOLATED_ALPHA: return getTranslatedFinalFormat();	//GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD

				case gl::INTERNAL_RGB_ETC: return getTranslatedFinalFormat(EF_ETC2_R8G8B8_UNORM_BLOCK);						//GL_COMPRESSED_RGB8_ETC1
				case gl::INTERNAL_RGB_ETC2: return getTranslatedFinalFormat(EF_ETC2_R8G8B8_UNORM_BLOCK);						//GL_COMPRESSED_RGB8_ETC2
				case gl::INTERNAL_RGBA_PUNCHTHROUGH_ETC2: return getTranslatedFinalFormat(EF_ETC2_R8G8B8A1_UNORM_BLOCK);		//GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2
				case gl::INTERNAL_RGBA_ETC2: return getTranslatedFinalFormat(EF_ETC2_R8G8B8A8_UNORM_BLOCK);					//GL_COMPRESSED_RGBA8_ETC2_EAC
				case gl::INTERNAL_R11_EAC: return getTranslatedFinalFormat(EF_EAC_R11_UNORM_BLOCK);						//GL_COMPRESSED_R11_EAC
				case gl::INTERNAL_SIGNED_R11_EAC: return getTranslatedFinalFormat(EF_EAC_R11_SNORM_BLOCK);				//GL_COMPRESSED_SIGNED_R11_EAC
				case gl::INTERNAL_RG11_EAC: return getTranslatedFinalFormat(EF_EAC_R11G11_UNORM_BLOCK);						//GL_COMPRESSED_RG11_EAC
				case gl::INTERNAL_SIGNED_RG11_EAC: return getTranslatedFinalFormat(EF_EAC_R11G11_SNORM_BLOCK);				//GL_COMPRESSED_SIGNED_RG11_EAC

				case gl::INTERNAL_RGBA_ASTC_4x4: return getTranslatedFinalFormat(EF_ASTC_4x4_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_4x4_KHR
				case gl::INTERNAL_RGBA_ASTC_5x4: return getTranslatedFinalFormat(EF_ASTC_5x4_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_5x4_KHR
				case gl::INTERNAL_RGBA_ASTC_5x5: return getTranslatedFinalFormat(EF_ASTC_5x5_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_5x5_KHR
				case gl::INTERNAL_RGBA_ASTC_6x5: return getTranslatedFinalFormat(EF_ASTC_6x5_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_6x5_KHR
				case gl::INTERNAL_RGBA_ASTC_6x6: return getTranslatedFinalFormat(EF_ASTC_6x6_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_6x6_KHR
				case gl::INTERNAL_RGBA_ASTC_8x5: return getTranslatedFinalFormat(EF_ASTC_8x5_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_8x5_KHR
				case gl::INTERNAL_RGBA_ASTC_8x6: return getTranslatedFinalFormat(EF_ASTC_8x6_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_8x6_KHR
				case gl::INTERNAL_RGBA_ASTC_8x8: return getTranslatedFinalFormat(EF_ASTC_8x8_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_8x8_KHR
				case gl::INTERNAL_RGBA_ASTC_10x5: return getTranslatedFinalFormat(EF_ASTC_10x5_UNORM_BLOCK); 				//GL_COMPRESSED_RGBA_ASTC_10x5_KHR
				case gl::INTERNAL_RGBA_ASTC_10x6: return getTranslatedFinalFormat(EF_ASTC_10x6_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_10x6_KHR
				case gl::INTERNAL_RGBA_ASTC_10x8: return getTranslatedFinalFormat(EF_ASTC_10x8_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_10x8_KHR
				case gl::INTERNAL_RGBA_ASTC_10x10: return getTranslatedFinalFormat(EF_ASTC_10x10_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_10x10_KHR
				case gl::INTERNAL_RGBA_ASTC_12x10: return getTranslatedFinalFormat(EF_ASTC_12x10_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_12x10_KHR
				case gl::INTERNAL_RGBA_ASTC_12x12: return getTranslatedFinalFormat(EF_ASTC_12x12_UNORM_BLOCK);				//GL_COMPRESSED_RGBA_ASTC_12x12_KHR

				// sRGB formats
				case gl::INTERNAL_SRGB_DXT1: return getTranslatedFinalFormat(EF_BC1_RGB_SRGB_BLOCK);					//GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT1: return getTranslatedFinalFormat(EF_BC1_RGBA_SRGB_BLOCK);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT3: return getTranslatedFinalFormat(EF_BC2_SRGB_BLOCK);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT5: return getTranslatedFinalFormat(EF_BC3_SRGB_BLOCK);				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
				case gl::INTERNAL_SRGB_BP_UNORM: return getTranslatedFinalFormat(EF_BC7_SRGB_BLOCK);				//GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM
				case gl::INTERNAL_SRGB_PVRTC_2BPPV1: return getTranslatedFinalFormat(EF_UNKNOWN, alphaPVRTCError);			//GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT
				case gl::INTERNAL_SRGB_PVRTC_4BPPV1: return getTranslatedFinalFormat(EF_UNKNOWN, alphaPVRTCError);			//GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_2BPPV1: return getTranslatedFinalFormat(EF_PVRTC1_2BPP_UNORM_BLOCK_IMG);		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_4BPPV1: return getTranslatedFinalFormat(EF_PVRTC1_4BPP_UNORM_BLOCK_IMG);		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_2BPPV2: return getTranslatedFinalFormat(EF_PVRTC1_2BPP_UNORM_BLOCK_IMG);		//COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_4BPPV2: return getTranslatedFinalFormat(EF_PVRTC1_4BPP_UNORM_BLOCK_IMG);		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG
				case gl::INTERNAL_SRGB8_ETC2: return getTranslatedFinalFormat(EF_ETC2_R8G8B8_SRGB_BLOCK);						//GL_COMPRESSED_SRGB8_ETC2
				case gl::INTERNAL_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: return getTranslatedFinalFormat(EF_ETC2_R8G8B8A1_SRGB_BLOCK);	//GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
				case gl::INTERNAL_SRGB8_ALPHA8_ETC2_EAC: return getTranslatedFinalFormat(EF_ETC2_R8G8B8A8_SRGB_BLOCK);			//GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_4x4: return getTranslatedFinalFormat(EF_ASTC_4x4_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_5x4: return getTranslatedFinalFormat(EF_ASTC_5x4_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_5x5: return getTranslatedFinalFormat(EF_ASTC_5x5_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_6x5: return getTranslatedFinalFormat(EF_ASTC_6x5_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_6x6: return getTranslatedFinalFormat(EF_ASTC_6x6_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x5: return getTranslatedFinalFormat(EF_ASTC_8x5_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x6: return getTranslatedFinalFormat(EF_ASTC_8x6_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x8: return getTranslatedFinalFormat(EF_ASTC_8x8_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x5: return getTranslatedFinalFormat(EF_ASTC_10x5_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x6: return getTranslatedFinalFormat(EF_ASTC_10x6_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x8: return getTranslatedFinalFormat(EF_ASTC_10x8_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x10: return getTranslatedFinalFormat(EF_ASTC_10x10_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_12x10: return getTranslatedFinalFormat(EF_ASTC_12x10_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_12x12: return getTranslatedFinalFormat(EF_ASTC_12x12_SRGB_BLOCK);		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR

				case gl::INTERNAL_R8_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_R8_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RG8_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RG8_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB8_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB8_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGBA8_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGBA8_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB10A2_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB10A2_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_R16_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_R16_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RG16_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RG16_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB16_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGB16_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGBA16_USCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				case gl::INTERNAL_RGBA16_SSCALED_GTC: return getTranslatedFinalFormat(EF_UNKNOWN, vulcanVertexFormatsError);
				default: 
					assert(0);
					return std::make_pair(EF_UNKNOWN, ICPUImageView::SComponentMapping{});
			}
		}

		inline void assignGLIDataToRegion(void* regionData, const gli::texture& texture, const uint16_t layer, const uint16_t face, const uint16_t level, const uint64_t sizeOfData)
		{
			const void* ptrToBegginingOfData = texture.data(layer, face, level);
			memcpy(regionData, ptrToBegginingOfData, sizeOfData);
		}
	}
}

#endif // _NBL_COMPILE_WITH_GLI_LOADER_
