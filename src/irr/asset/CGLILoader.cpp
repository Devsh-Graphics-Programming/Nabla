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

#ifdef _IRR_COMPILE_WITH_GLI_LOADER_

#ifdef _IRR_COMPILE_WITH_GLI_
#include "gli/gli.hpp"
#else
#error "It requires GLI library"
#endif

namespace irr
{
	namespace asset
	{
		asset::SAssetBundle CGLILoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			SContext ctx(_file->getSize());
			ctx.file = _file;

			const char* filename = _file->getFileName().c_str();
			gli::texture texture = gli::load(filename);
			if (texture.empty())
			{
				os::Printer::log("Failed to load texture at path ", filename, ELL_ERROR);
				return {};
			}
			
		    const gli::gl glVersion(gli::gl::PROFILE_GL33);
			const GLenum target = glVersion.translate(texture.target());
			const auto format = getTranslatedGLIFormat(texture, glVersion);
			const uint32_t texelFormatByteSize = getTexelOrBlockBytesize(format.first);
			IImage::E_TYPE baseImageType;
			IImageView<ICPUImage>::E_TYPE imageViewType;

			switch (target)
			{
				case gli::TARGET_1D:
				{
					baseImageType = IImage::ET_1D;
					imageViewType = ICPUImageView::ET_1D;
					break;
				}
				case gli::TARGET_1D_ARRAY:
				{
					baseImageType = IImage::ET_1D;
					imageViewType = ICPUImageView::ET_1D_ARRAY;
					break;
				}
				case gli::TARGET_2D:
				{
					baseImageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_2D;
					break;
				}
				case gli::TARGET_2D_ARRAY:
				{
					baseImageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_2D_ARRAY;
					break;
				}
				case gli::TARGET_3D:
				{
					baseImageType = IImage::ET_3D;
					imageViewType = ICPUImageView::ET_3D;
					break;
				}
				case gli::TARGET_CUBE:
				{
					baseImageType = IImage::ET_2D;
					imageViewType = ICPUImageView::ET_CUBE_MAP;
					break;
				}
				case gli::TARGET_CUBE_ARRAY:
				{
					baseImageType = IImage::ET_2D;
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

			std::pair<core::smart_refctd_dynamic_array<ICPUImage>, core::smart_refctd_dynamic_array<ICPUImageView>> bundle
			= std::make_pair																							
			(
				core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage>>(texture.levels()),
				core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImageView>>(texture.levels())
			);

			for (uint16_t mipmapLevel = 0; mipmapLevel < bundle.first->size(); ++mipmapLevel)
			{
				const auto totalFaces = texture.layers() * texture.faces();
				glm::tvec3<GLsizei> extent(texture.extent(mipmapLevel));

				ICPUImage::SCreationParams baseImageInfo;
				baseImageInfo.format = format.first;
				baseImageInfo.type = baseImageType;
				baseImageInfo.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u);
				baseImageInfo.samples = ICPUImage::ESCF_1_BIT;
				baseImageInfo.extent.width = extent[0];
				baseImageInfo.extent.height = extent[1];
				baseImageInfo.extent.width = extent[2];
				baseImageInfo.mipLevels = texture.levels();
				baseImageInfo.arrayLayers = totalFaces;

				auto& baseImage = core::make_smart_refctd_ptr<ICPUImage>(*(bundle.first->begin() + mipmapLevel));
				baseImage = ICPUImage::create(std::move(baseImageInfo));

				ICPUImageView::SCreationParams imageViewInfo;
				imageViewInfo.image = baseImage;
				imageViewInfo.format = format.first;
				imageViewInfo.viewType = imageViewType;
				imageViewInfo.components = format.second;
				imageViewInfo.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
				imageViewInfo.subresourceRange.baseArrayLayer = 0u;
				imageViewInfo.subresourceRange.baseMipLevel = 0u;
				imageViewInfo.subresourceRange.layerCount = totalFaces;
				imageViewInfo.subresourceRange.levelCount = 1u;

				core::vector<core::smart_refctd_ptr<ICPUBuffer>> texelBuffers;
				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(totalFaces);

				// do stuff

				for (std::size_t layer = 0; layer < texture.layers(); ++layer)					// if it is an array of textures of certain type 
					for (std::size_t face = 0; face < texture.faces(); ++face)					// if it is a cube, otherwise there is only single face
					{
						auto index = (layer * texture.layers()) + face;
						texelBuffers.emplace_back(core::make_smart_refctd_ptr<ICPUBuffer>(baseImage->getImageDataSizeInBytes())); // I need to put data here, but
						// core::smart_refctd_ptr<CCustomAllocatorCPUBuffer<>> buffer = CCustomAllocatorCPUBuffer<>(XSIZEX, XDATAX);	is inaccessable!

						ICPUImage::SBufferCopy& region = *(regions->begin() + index);
						region.imageSubresource.mipLevel = 0u;
						region.imageSubresource.baseArrayLayer = 0u;
						region.imageSubresource.layerCount = 1u;
						region.bufferOffset = 0u;
						region.bufferRowLength = extent[0] * texelFormatByteSize;
						region.bufferImageHeight = 0u; //tightly packed
						region.imageOffset = { 0u, 0u, 0u };
						region.imageExtent.width = extent[0];
						region.imageExtent.height = extent[1];
						region.imageExtent.depth = extent[2];
					}

				auto& viewImage = core::make_smart_refctd_ptr<ICPUImageView>(*(bundle.second->begin() + mipmapLevel));
				// TODO need to set regions on viewImage as subresource of certain mipmap
				viewImage = ICPUImageView::create(std::move(imageViewInfo));
			}

			ctx.file = nullptr;

			//return SAssetBundle( ? );
		}

		bool CGLILoader::isALoadableFileFormat(io::IReadFile* _file) const
		{
			return true; // gli provides a function to load files, but we can check files' signature actually if needed
		}

		inline std::pair<E_FORMAT, ICPUImageView::SComponentMapping> CGLILoader::getTranslatedGLIFormat(const gli::texture& texture, const gli::gl& glVersion)
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
			};

			compomentMapping.r = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.r));
			compomentMapping.g = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.g));
			compomentMapping.b = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.b));
			compomentMapping.a = getMappedSwizzle(static_cast<gli::gl::swizzle>(formatToTranslate.Swizzles.a));

			auto getTranslatedFinalFormat = [&](const E_FORMAT& format = EF_UNKNOWN)
			{
				return std::make_pair(format, compomentMapping);
			};

			// TODO - fill formats
			switch (formatToTranslate.Internal)
			{
				case gl::INTERNAL_RGB_UNORM: return getTranslatedFinalFormat(EF_R8G8B8_UNORM);			//GL_RGB
				case gl::INTERNAL_BGR_UNORM: return getTranslatedFinalFormat();			//GL_BGR
				case gl::INTERNAL_RGBA_UNORM: return getTranslatedFinalFormat();		//GL_RGBA
				case gl::INTERNAL_BGRA_UNORM: return getTranslatedFinalFormat();		//GL_BGRA
				case gl::INTERNAL_BGRA8_UNORM: return getTranslatedFinalFormat();		//GL_BGRA8_EXT

				// unorm formats
				case gl::INTERNAL_R8_UNORM: return getTranslatedFinalFormat();			//GL_R8
				case gl::INTERNAL_RG8_UNORM: return getTranslatedFinalFormat();		//GL_RG8
				case gl::INTERNAL_RGB8_UNORM: return getTranslatedFinalFormat();		//GL_RGB8
				case gl::INTERNAL_RGBA8_UNORM: return getTranslatedFinalFormat();		//GL_RGBA8

				case gl::INTERNAL_R16_UNORM: return getTranslatedFinalFormat();		//GL_R16
				case gl::INTERNAL_RG16_UNORM: return getTranslatedFinalFormat();		//GL_RG16
				case gl::INTERNAL_RGB16_UNORM: return getTranslatedFinalFormat();		//GL_RGB16
				case gl::INTERNAL_RGBA16_UNORM: return getTranslatedFinalFormat();		//GL_RGBA16

				case gl::INTERNAL_RGB10A2_UNORM: return getTranslatedFinalFormat();	//GL_RGB10_A2
				case gl::INTERNAL_RGB10A2_SNORM_EXT: return getTranslatedFinalFormat();

				// snorm formats
				case gl::INTERNAL_R8_SNORM: return getTranslatedFinalFormat();			//GL_R8_SNORM
				case gl::INTERNAL_RG8_SNORM: return getTranslatedFinalFormat();		//GL_RG8_SNORM
				case gl::INTERNAL_RGB8_SNORM: return getTranslatedFinalFormat();		//GL_RGB8_SNORM
				case gl::INTERNAL_RGBA8_SNORM: return getTranslatedFinalFormat();		//GL_RGBA8_SNORM

				case gl::INTERNAL_R16_SNORM: return getTranslatedFinalFormat();		//GL_R16_SNORM
				case gl::INTERNAL_RG16_SNORM: return getTranslatedFinalFormat();		//GL_RG16_SNORM
				case gl::INTERNAL_RGB16_SNORM: return getTranslatedFinalFormat();		//GL_RGB16_SNORM
				case gl::INTERNAL_RGBA16_SNORM: return getTranslatedFinalFormat();		//GL_RGBA16_SNORM

				// unsigned integer formats
				case gl::INTERNAL_R8U: return getTranslatedFinalFormat();				//GL_R8UI
				case gl::INTERNAL_RG8U: return getTranslatedFinalFormat();				//GL_RG8UI
				case gl::INTERNAL_RGB8U: return getTranslatedFinalFormat();			//GL_RGB8UI
				case gl::INTERNAL_RGBA8U: return getTranslatedFinalFormat();			//GL_RGBA8UI

				case gl::INTERNAL_R16U: return getTranslatedFinalFormat();				//GL_R16UI
				case gl::INTERNAL_RG16U: return getTranslatedFinalFormat();			//GL_RG16UI
				case gl::INTERNAL_RGB16U: return getTranslatedFinalFormat();			//GL_RGB16UI
				case gl::INTERNAL_RGBA16U: return getTranslatedFinalFormat();			//GL_RGBA16UI

				case gl::INTERNAL_R32U: return getTranslatedFinalFormat();				//GL_R32UI
				case gl::INTERNAL_RG32U: return getTranslatedFinalFormat();			//GL_RG32UI
				case gl::INTERNAL_RGB32U: return getTranslatedFinalFormat();			//GL_RGB32UI
				case gl::INTERNAL_RGBA32U: return getTranslatedFinalFormat();			//GL_RGBA32UI

				case gl::INTERNAL_RGB10A2U: return getTranslatedFinalFormat();			//GL_RGB10_A2UI
				case gl::INTERNAL_RGB10A2I_EXT: return getTranslatedFinalFormat();

				// signed integer formats
				case gl::INTERNAL_R8I: return getTranslatedFinalFormat();				//GL_R8I
				case gl::INTERNAL_RG8I: return getTranslatedFinalFormat();				//GL_RG8I
				case gl::INTERNAL_RGB8I: return getTranslatedFinalFormat();			//GL_RGB8I
				case gl::INTERNAL_RGBA8I: return getTranslatedFinalFormat();			//GL_RGBA8I

				case gl::INTERNAL_R16I: return getTranslatedFinalFormat();				//GL_R16I
				case gl::INTERNAL_RG16I: return getTranslatedFinalFormat();			//GL_RG16I
				case gl::INTERNAL_RGB16I: return getTranslatedFinalFormat();			//GL_RGB16I
				case gl::INTERNAL_RGBA16I: return getTranslatedFinalFormat();			//GL_RGBA16I

				case gl::INTERNAL_R32I: return getTranslatedFinalFormat();				//GL_R32I
				case gl::INTERNAL_RG32I: return getTranslatedFinalFormat();			//GL_RG32I
				case gl::INTERNAL_RGB32I: return getTranslatedFinalFormat();			//GL_RGB32I
				case gl::INTERNAL_RGBA32I: return getTranslatedFinalFormat();			//GL_RGBA32I

				// Floating formats
				case gl::INTERNAL_R16F: return getTranslatedFinalFormat();				//GL_R16F
				case gl::INTERNAL_RG16F: return getTranslatedFinalFormat();			//GL_RG16F
				case gl::INTERNAL_RGB16F: return getTranslatedFinalFormat();			//GL_RGB16F
				case gl::INTERNAL_RGBA16F: return getTranslatedFinalFormat();			//GL_RGBA16F

				case gl::INTERNAL_R32F: return getTranslatedFinalFormat();				//GL_R32F
				case gl::INTERNAL_RG32F: return getTranslatedFinalFormat();			//GL_RG32F
				case gl::INTERNAL_RGB32F: return getTranslatedFinalFormat();			//GL_RGB32F
				case gl::INTERNAL_RGBA32F: return getTranslatedFinalFormat();			//GL_RGBA32F

				case gl::INTERNAL_R64F_EXT: return getTranslatedFinalFormat();			//GL_R64F
				case gl::INTERNAL_RG64F_EXT: return getTranslatedFinalFormat();		//GL_RG64F
				case gl::INTERNAL_RGB64F_EXT: return getTranslatedFinalFormat();		//GL_RGB64F
				case gl::INTERNAL_RGBA64F_EXT: return getTranslatedFinalFormat();		//GL_RGBA64F

				// sRGB formats
				case gl::INTERNAL_SR8: return getTranslatedFinalFormat();				//GL_SR8_EXT
				case gl::INTERNAL_SRG8: return getTranslatedFinalFormat();				//GL_SRG8_EXT
				case gl::INTERNAL_SRGB8: return getTranslatedFinalFormat();			//GL_SRGB8
				case gl::INTERNAL_SRGB8_ALPHA8: return getTranslatedFinalFormat();		//GL_SRGB8_ALPHA8

				// Packed formats
				case gl::INTERNAL_RGB9E5: return getTranslatedFinalFormat();			//GL_RGB9_E5
				case gl::INTERNAL_RG11B10F: return getTranslatedFinalFormat();			//GL_R11F_G11F_B10F
				case gl::INTERNAL_RG3B2: return getTranslatedFinalFormat();			//GL_R3_G3_B2
				case gl::INTERNAL_R5G6B5: return getTranslatedFinalFormat();			//GL_RGB565
				case gl::INTERNAL_RGB5A1: return getTranslatedFinalFormat();			//GL_RGB5_A1
				case gl::INTERNAL_RGBA4: return getTranslatedFinalFormat();			//GL_RGBA4

				case gl::INTERNAL_RG4_EXT: return getTranslatedFinalFormat();

				// Luminance Alpha formats
				case gl::INTERNAL_LA4: return getTranslatedFinalFormat();				//GL_LUMINANCE4_ALPHA4
				case gl::INTERNAL_L8: return getTranslatedFinalFormat();				//GL_LUMINANCE8
				case gl::INTERNAL_A8: return getTranslatedFinalFormat();				//GL_ALPHA8
				case gl::INTERNAL_LA8: return getTranslatedFinalFormat();				//GL_LUMINANCE8_ALPHA8
				case gl::INTERNAL_L16: return getTranslatedFinalFormat();				//GL_LUMINANCE16
				case gl::INTERNAL_A16: return getTranslatedFinalFormat();				//GL_ALPHA16
				case gl::INTERNAL_LA16: return getTranslatedFinalFormat();				//GL_LUMINANCE16_ALPHA16

				// Depth formats
				case gl::INTERNAL_D16: return getTranslatedFinalFormat();				//GL_DEPTH_COMPONENT16
				case gl::INTERNAL_D24: return getTranslatedFinalFormat();				//GL_DEPTH_COMPONENT24
				case gl::INTERNAL_D16S8_EXT: return getTranslatedFinalFormat();
				case gl::INTERNAL_D24S8: return getTranslatedFinalFormat();			//GL_DEPTH24_STENCIL8
				case gl::INTERNAL_D32: return getTranslatedFinalFormat();				//GL_DEPTH_COMPONENT32
				case gl::INTERNAL_D32F: return getTranslatedFinalFormat();				//GL_DEPTH_COMPONENT32F
				case gl::INTERNAL_D32FS8X24: return getTranslatedFinalFormat();		//GL_DEPTH32F_STENCIL8
				case gl::INTERNAL_S8_EXT: return getTranslatedFinalFormat();			//GL_STENCIL_INDEX8

				// Compressed formats
				case gl::INTERNAL_RGB_DXT1: return getTranslatedFinalFormat();						//GL_COMPRESSED_RGB_S3TC_DXT1_EXT
				case gl::INTERNAL_RGBA_DXT1: return getTranslatedFinalFormat();					//GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
				case gl::INTERNAL_RGBA_DXT3: return getTranslatedFinalFormat();					//GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
				case gl::INTERNAL_RGBA_DXT5: return getTranslatedFinalFormat();					//GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
				case gl::INTERNAL_R_ATI1N_UNORM: return getTranslatedFinalFormat();				//GL_COMPRESSED_RED_RGTC1
				case gl::INTERNAL_R_ATI1N_SNORM: return getTranslatedFinalFormat();				//GL_COMPRESSED_SIGNED_RED_RGTC1
				case gl::INTERNAL_RG_ATI2N_UNORM: return getTranslatedFinalFormat();				//GL_COMPRESSED_RG_RGTC2
				case gl::INTERNAL_RG_ATI2N_SNORM: return getTranslatedFinalFormat();				//GL_COMPRESSED_SIGNED_RG_RGTC2
				case gl::INTERNAL_RGB_BP_UNSIGNED_FLOAT: return getTranslatedFinalFormat();		//GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
				case gl::INTERNAL_RGB_BP_SIGNED_FLOAT: return getTranslatedFinalFormat();			//GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT
				case gl::INTERNAL_RGB_BP_UNORM: return getTranslatedFinalFormat();					//GL_COMPRESSED_RGBA_BPTC_UNORM
				case gl::INTERNAL_RGB_PVRTC_4BPPV1: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG
				case gl::INTERNAL_RGB_PVRTC_2BPPV1: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_4BPPV1: return getTranslatedFinalFormat();			//GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_2BPPV1: return getTranslatedFinalFormat();			//GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG
				case gl::INTERNAL_RGBA_PVRTC_4BPPV2: return getTranslatedFinalFormat();			//GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG
				case gl::INTERNAL_RGBA_PVRTC_2BPPV2: return getTranslatedFinalFormat();			//GL_COMPRESSED_RGBA_PVRTC_2BPPV2_IMG
				case gl::INTERNAL_ATC_RGB: return getTranslatedFinalFormat();						//GL_ATC_RGB_AMD
				case gl::INTERNAL_ATC_RGBA_EXPLICIT_ALPHA: return getTranslatedFinalFormat();		//GL_ATC_RGBA_EXPLICIT_ALPHA_AMD
				case gl::INTERNAL_ATC_RGBA_INTERPOLATED_ALPHA: return getTranslatedFinalFormat();	//GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD

				case gl::INTERNAL_RGB_ETC: return getTranslatedFinalFormat();						//GL_COMPRESSED_RGB8_ETC1
				case gl::INTERNAL_RGB_ETC2: return getTranslatedFinalFormat();						//GL_COMPRESSED_RGB8_ETC2
				case gl::INTERNAL_RGBA_PUNCHTHROUGH_ETC2: return getTranslatedFinalFormat();		//GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2
				case gl::INTERNAL_RGBA_ETC2: return getTranslatedFinalFormat();					//GL_COMPRESSED_RGBA8_ETC2_EAC
				case gl::INTERNAL_R11_EAC: return getTranslatedFinalFormat();						//GL_COMPRESSED_R11_EAC
				case gl::INTERNAL_SIGNED_R11_EAC: return getTranslatedFinalFormat();				//GL_COMPRESSED_SIGNED_R11_EAC
				case gl::INTERNAL_RG11_EAC: return getTranslatedFinalFormat();						//GL_COMPRESSED_RG11_EAC
				case gl::INTERNAL_SIGNED_RG11_EAC: return getTranslatedFinalFormat();				//GL_COMPRESSED_SIGNED_RG11_EAC

				case gl::INTERNAL_RGBA_ASTC_4x4: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_4x4_KHR
				case gl::INTERNAL_RGBA_ASTC_5x4: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_5x4_KHR
				case gl::INTERNAL_RGBA_ASTC_5x5: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_5x5_KHR
				case gl::INTERNAL_RGBA_ASTC_6x5: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_6x5_KHR
				case gl::INTERNAL_RGBA_ASTC_6x6: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_6x6_KHR
				case gl::INTERNAL_RGBA_ASTC_8x5: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_8x5_KHR
				case gl::INTERNAL_RGBA_ASTC_8x6: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_8x6_KHR
				case gl::INTERNAL_RGBA_ASTC_8x8: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_8x8_KHR
				case gl::INTERNAL_RGBA_ASTC_10x5: return getTranslatedFinalFormat(); 				//GL_COMPRESSED_RGBA_ASTC_10x5_KHR
				case gl::INTERNAL_RGBA_ASTC_10x6: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_10x6_KHR
				case gl::INTERNAL_RGBA_ASTC_10x8: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_10x8_KHR
				case gl::INTERNAL_RGBA_ASTC_10x10: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_10x10_KHR
				case gl::INTERNAL_RGBA_ASTC_12x10: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_12x10_KHR
				case gl::INTERNAL_RGBA_ASTC_12x12: return getTranslatedFinalFormat();				//GL_COMPRESSED_RGBA_ASTC_12x12_KHR

				// sRGB formats
				case gl::INTERNAL_SRGB_DXT1: return getTranslatedFinalFormat();					//GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT1: return getTranslatedFinalFormat();				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT3: return getTranslatedFinalFormat();				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
				case gl::INTERNAL_SRGB_ALPHA_DXT5: return getTranslatedFinalFormat();				//GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
				case gl::INTERNAL_SRGB_BP_UNORM: return getTranslatedFinalFormat();				//GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM
				case gl::INTERNAL_SRGB_PVRTC_2BPPV1: return getTranslatedFinalFormat();			//GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT
				case gl::INTERNAL_SRGB_PVRTC_4BPPV1: return getTranslatedFinalFormat();			//GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_2BPPV1: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_4BPPV1: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_2BPPV2: return getTranslatedFinalFormat();		//COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG
				case gl::INTERNAL_SRGB_ALPHA_PVRTC_4BPPV2: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG
				case gl::INTERNAL_SRGB8_ETC2: return getTranslatedFinalFormat();						//GL_COMPRESSED_SRGB8_ETC2
				case gl::INTERNAL_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: return getTranslatedFinalFormat();	//GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
				case gl::INTERNAL_SRGB8_ALPHA8_ETC2_EAC: return getTranslatedFinalFormat();			//GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_4x4: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_5x4: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_5x5: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_6x5: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_6x6: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x5: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x6: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_8x8: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x5: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x6: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x8: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_10x10: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_12x10: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR
				case gl::INTERNAL_SRGB8_ALPHA8_ASTC_12x12: return getTranslatedFinalFormat();		//GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR

				case gl::INTERNAL_R8_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_R8_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RG8_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RG8_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB8_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB8_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGBA8_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGBA8_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB10A2_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB10A2_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_R16_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_R16_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RG16_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RG16_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB16_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGB16_SSCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGBA16_USCALED_GTC: return getTranslatedFinalFormat();
				case gl::INTERNAL_RGBA16_SSCALED_GTC: return getTranslatedFinalFormat();
				default: assert(0);
			}
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLI_LOADER_