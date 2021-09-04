// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_COMMON_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_COMMON_H_INCLUDED__

#include "BuildConfigOptions.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/asset/format/EFormat.h"
namespace nbl::video
{

struct SOpenGLBarrierHelper
{
	public:
		SOpenGLBarrierHelper(const COpenGLFeatureMap* features) : TransformFeedbackBit(0), // we dont expose transform feedback
			QueryBufferBit(features->isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_query_buffer_object)||features->Version>=440 ? GL_QUERY_BUFFER_BARRIER_BIT:0),
			LateFragmentAccessBits(QueryBufferBit|GL_FRAMEBUFFER_BARRIER_BIT),
			FragmentShaderAndAfterAccessBits(DescriptorAccessBits|LateFragmentAccessBits), EarlyFragmentTestsAndAfterAccessBits(FragmentShaderAndAfterAccessBits),
			ProgrammablePrimitivePipelineAndAfterAccessBits(TransformFeedbackBit|EarlyFragmentTestsAndAfterAccessBits),
			InputAssemblyAndAfterAccessBits(GL_ELEMENT_ARRAY_BARRIER_BIT|GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT|ProgrammablePrimitivePipelineAndAfterAccessBits),
			AllGraphicsBits(GL_COMMAND_BARRIER_BIT|InputAssemblyAndAfterAccessBits), AllBarrierBits(AllGraphicsBits|ComputeAccessBits|TransferBits|HostBits)
		{
		}

		inline GLbitfield pipelineStageFlagsToMemoryBarrierBits(asset::E_PIPELINE_STAGE_FLAGS srcflags, asset::E_PIPELINE_STAGE_FLAGS dstflags) const
		{
			// stuff GL backend could expose but we dont support
			assert(srcflags&asset::EPSF_TRANSFORM_FEEDBACK_BIT_EXT==0);
			assert(srcflags&asset::EPSF_CONDITIONAL_RENDERING_BIT_EXT==0);
			assert(srcflags<asset::EPSF_SHADING_RATE_IMAGE_BIT_NV);

			constexpr GLbitfield HostBits = GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT;

			constexpr uint32_t PipelineStageCount = 14u;

			// Why are all the Pipeline stages except top,indirect command and vertex input are declaring `AllBarrierBits` ?
			// Because each of these stages can write a buffer or an image
			// In turn the buffer can be used as anything
			// (TBO,UBO,SSBO,IndirectCommand,QueryBuffer and Vertex Input/Index Buffer, Transfer Read/Write, Mapping)
			// as well as the image (Sampled Image, Storage Image, Framebuffer attachment, texture update)
			GLbitfield srcbits = 0;
			if (srcflags & asset::EPSF_ALL_GRAPHICS_BIT)
				srcbits = AllBarrierBits;
			else if (srcflags & asset::EPSF_ALL_COMMANDS_BIT)
				srcbits = AllBarrierBits;
			else
			{
				/*
				const GLbitfield producerBits[PipelineStageCount] = { // src stage mask
					0, // stages before top of pipe can't possibly create anything
					0, // indirect command execute doesn't produce any writes 
					0, // vertex input stage is also readonly
					AllBarrierBits, // vertex shader stage
					AllBarrierBits, // control shader stage
					AllBarrierBits, // evaluation shader stage
					AllBarrierBits, // geometry shader stage
					AllBarrierBits, // fragment shader stage
					AllBarrierBits, // early fragment test stage
					AllBarrierBits, // late fragment test stage
					AllBarrierBits, // color attachment output stage
					AllBarrierBits, // compute shader stage
					AllBarrierBits, // transfer could have stored into anything
					AllBarrierBits // bottom of pipe could have produced everything as it logically follows all of them
				};
				for (uint32_t i=0u; i<PipelineStageCount; ++i)
				if ((srcflags>>i)&0x1u)
					srcbits |= producerBits[i];
				*/
				if (srcflags&(asset::EPSF_HOST_BIT-asset::EPSF_VERTEX_SHADER_BIT))
					srcbits = AllBarrierBits;
				else if (srcflags & asset::EPSF_HOST_BIT)
					srcbits = AllBarrierBits^ImageTransferBits; // you cannot map the memory of an image in OpenGL
			}

			const GLbitfield consumerBits[PipelineStageCount] = { // dst stage mask
				AllBarrierBits, // every later stage could consume anything
				AllGraphicsBits|ComputeAccessBits, // every later shader count consume anything
				InputAssemblyAndAfterAccessBits, // vertex input stage is later than command stage
				ProgrammablePrimitivePipelineAndAfterAccessBits, // vertex shader stage is later than vertex input stage
				ProgrammablePrimitivePipelineAndAfterAccessBits, // control shader stage is later than vertex shader stage, but cannot consume any less resource types
				ProgrammablePrimitivePipelineAndAfterAccessBits, // evaluation shader stage is later than control shader stage, but cannot consume less resource types
				ProgrammablePrimitivePipelineAndAfterAccessBits, // geometry shader stage is later than evaluation shader stage, but cannot consume any less resource types
				FragmentShaderAndAfterAccessBits, // fragment shader stage needs to include all the bits late fragment test needs
				EarlyFragmentTestsAndAfterAccessBits, // early fragment test stage is later than geometry shader stage and ergo no longer writes into transform feedback can no longer occur
				LateFragmentAccessBits, // late fragment test stage needs access to framebuffer and query buffer
				GL_FRAMEBUFFER_BARRIER_BIT, // color attachment output stage only writes to framebuffer
				DescriptorAccessBits, // compute shader stage is later than command stage
				TransferBits,
				0 // bottom of pipe reads nothing
			};

			GLbitfield dstbits = 0;
			if (dstflags & asset::EPSF_HOST_BIT)
				dstbits |= HostBits;
			if (dstflags & asset::EPSF_ALL_GRAPHICS_BIT)
				dstbits |= AllGraphicsBits;
			if (dstflags & asset::EPSF_ALL_COMMANDS_BIT)
				dstbits |= ComputeAccessBits|AllGraphicsBits|TransferBits; // OpenGL queue can do everything
			for (uint32_t i = 0u; i < PipelineStageCount; ++i)
				if (dstflags & (1u << i))
					dstbits |= consumerBits[i];

			return srcbits&dstbits;
		}

		inline GLbitfield accessFlagsToMemoryBarrierBits(const asset::SMemoryBarrier& barrier) const
		{
			constexpr uint32_t AccessBitCount = 28u;
			const GLbitfield srcBits[AccessBitCount] = {
				0, // command processing cannot make any writes
				0, // index buffer reading cannot make any writes
				0, // vertex buffer reading cannot make any writes
				0, // UBO reading cannot make any writes
				0, // input attachment reading cannot make any writes
				0, // UBO/UTB/sampler/SSBO/STB,Image reading cannot make any writes
				AllBarrierBits, // writing to the above, can result in any usage of the result whatsoever
				0, // FBO attachment reading cannot make any writes
				ImageDescriptorAccessBits|ImageTransferBits, // FBO attachment writing can be later used in image accesses, framebuffer and texture transfers
				0, // FBO attachment reading cannot make any writes
				ImageDescriptorAccessBits|ImageTransferBits, // FBO attachment writing can be later used in image accesses, framebuffer and texture transfers
				0, // reads dont produce writes
				AllBarrierBits, // writing to a buffer or an image can result in any usage of the result whatsoever
				0, // reads dont produce writes
				AllBarrierBits^ImageTransferBits, // writing to a buffer can result in any usage except as a framebuffer/texture (but can as a sampler or image due to BufferViews)
				0, // reads dont produce writes
				AllBarrierBits, // writing to a buffer or an image can result in any usage of the result whatsoever
				0, // we dont support transform feedback
				0, // we dont support transform feedback
				0, // we dont support transform feedback
				0, // we dont support conditional rendering extension
				GL_FRAMEBUFFER_BARRIER_BIT,
				0, // OpenGL doesn't have support for acceleration structures
				0, // OpenGL doesn't have support for acceleration structures
				0, // we don't support shading rate extension
				0, // we don't support fragment density map extension
				0, // command processing cannot make any writes
				0 // We don't support the NV commandbuffer/token extension
			};
			const GLbitfield dstBits[AccessBitCount] = {
				GL_COMMAND_BARRIER_BIT,
				GL_ELEMENT_ARRAY_BARRIER_BIT,
				GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT,
				GL_UNIFORM_BARRIER_BIT,
				GL_FRAMEBUFFER_BARRIER_BIT|ImageDescriptorAccessBits, // input attachment read, no idea what should be here TODO
				DescriptorAccessBits,
				DescriptorAccessBits,
				GL_FRAMEBUFFER_BARRIER_BIT,
				GL_FRAMEBUFFER_BARRIER_BIT,
				GL_FRAMEBUFFER_BARRIER_BIT,
				GL_FRAMEBUFFER_BARRIER_BIT,
				TransferBits,
				TransferBits,
				GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT,
				GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT,
				AllBarrierBits,
				AllBarrierBits,
				0, // we dont support transform feedback
				0, // we dont support transform feedback
				0, // we dont support transform feedback
				0, // we dont support conditional rendering extension
				GL_FRAMEBUFFER_BARRIER_BIT,
				0, // OpenGL doesn't have support for acceleration structures
				0, // OpenGL doesn't have support for acceleration structures
				0, // we don't support shading rate extension
				0, // we don't support fragment density map extension
				0, // command processing cannot make any writes
				0 // We don't support the NV commandbuffer/token extension
			};

			GLbitfield srcBarrier=0, dstBarrier=0;
			for (uint32_t i=0u; i<AccessBitCount; ++i)
			{
				const auto flag = 0x1u<<i;
				if (barrier.srcAccessMask&flag)
					srcBarrier |= srcBits[i];
				if (barrier.dstAccessMask&flag)
					dstBarrier |= dstBits[i];
			}

			return srcBarrier&dstBarrier;
		}

		//
		static inline constexpr GLbitfield HostBits = GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT;

		//
		static inline constexpr GLbitfield BufferTransferBits = GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT;
		static inline constexpr GLbitfield ImageTransferBits = GL_TEXTURE_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT; // framebuffer because of glReadPixels for GetSubImage emulation
		static inline constexpr GLbitfield TransferBits = BufferTransferBits | ImageTransferBits;

		static inline constexpr GLbitfield ImageDescriptorAccessBits = GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
		static inline constexpr GLbitfield DescriptorAccessBits = GL_UNIFORM_BARRIER_BIT | ImageDescriptorAccessBits | GL_SHADER_STORAGE_BARRIER_BIT;
		// we include the image bits because of `samplerBuffer` and `imageBuffer`
		static inline constexpr GLbitfield BufferDescriptorAccessBits = DescriptorAccessBits;

		//
		static inline constexpr GLbitfield ComputeAccessBits = GL_COMMAND_BARRIER_BIT | DescriptorAccessBits;
		

		// we dont expose transform feedback
		const GLbitfield TransformFeedbackBit,QueryBufferBit;
	
		const GLbitfield LateFragmentAccessBits;
		const GLbitfield FragmentShaderAndAfterAccessBits;
		const GLbitfield EarlyFragmentTestsAndAfterAccessBits;
		const GLbitfield ProgrammablePrimitivePipelineAndAfterAccessBits;
		const GLbitfield InputAssemblyAndAfterAccessBits;
		const GLbitfield AllGraphicsBits;

		// we dont expose atomic counters
		const GLbitfield AllBarrierBits;
};

inline GLenum	getSizedOpenGLFormatFromOurFormat(IOpenGL_FunctionTable* gl, asset::E_FORMAT format)
{
	using namespace asset;
	switch (format)
	{
		case EF_A1R5G5B5_UNORM_PACK16:
			return GL_RGB5_A1;
			break;
		case EF_R5G6B5_UNORM_PACK16:
			return GL_RGB565;
			break;
			// Floating Point texture formats. Thanks to Patryk "Nadro" Nadrowski.
		case EF_B10G11R11_UFLOAT_PACK32:
			return GL_R11F_G11F_B10F;
			break;
		case EF_R16_SFLOAT:
			return GL_R16F;
			break;
		case EF_R16G16_SFLOAT:
			return GL_RG16F;
			break;
		case EF_R16G16B16_SFLOAT:
			return GL_RGB16F;
		case EF_R16G16B16A16_SFLOAT:
			return GL_RGBA16F;
			break;
		case EF_R32_SFLOAT:
			return GL_R32F;
			break;
		case EF_R32G32_SFLOAT:
			return GL_RG32F;
			break;
		case EF_R32G32B32_SFLOAT:
			return GL_RGB32F;
			break;
		case EF_R32G32B32A32_SFLOAT:
			return GL_RGBA32F;
			break;
		case EF_R8_UNORM:
			return GL_R8;
			break;
		case EF_R8_SRGB:
			if (!gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_R8))
				return GL_SR8_EXT;
			break;
		case EF_R8G8_UNORM:
			return GL_RG8;
			break;
		case EF_R8G8_SRGB:
			if (!gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_RG8))
				return GL_SRG8_EXT;
			break;
		case EF_R8G8B8_UNORM:
			return GL_RGB8;
			break;
		case EF_B8G8R8A8_UNORM:
			return GL_RGBA8;
			break;
		case EF_B8G8R8A8_SRGB:
			return GL_SRGB8_ALPHA8;
			break;
		case EF_R8G8B8A8_UNORM:
			return GL_RGBA8;
			break;
		case EF_R8_UINT:
			return GL_R8UI;
			break;
		case EF_R8G8_UINT:
			return GL_RG8UI;
			break;
		case EF_R8G8B8_UINT:
			return GL_RGB8UI;
			break;
		case EF_B8G8R8A8_UINT:
			return GL_RGBA8UI;
			break;
		case EF_R8G8B8A8_UINT:
			return GL_RGBA8UI;
			break;
		case EF_R8_SINT:
			return GL_R8I;
			break;
		case EF_R8G8_SINT:
			return GL_RG8I;
			break;
		case EF_R8G8B8_SINT:
			return GL_RGB8I;
			break;
		case EF_B8G8R8A8_SINT:
			return GL_RGBA8I;
			break;
		case EF_R8G8B8A8_SINT:
			return GL_RGBA8I;
			break;
		case EF_R8_SNORM:
			return GL_R8_SNORM;
			break;
		case EF_R8G8_SNORM:
			return GL_RG8_SNORM;
			break;
		case EF_R8G8B8_SNORM:
			return GL_RGB8_SNORM;
			break;
		case EF_B8G8R8A8_SNORM:
			return GL_RGBA8_SNORM;
			break;
		case EF_R8G8B8A8_SNORM:
			return GL_RGBA8_SNORM;
			break;
		case EF_R16_UNORM:
			return GL_R16;
			break;
		case EF_R16G16_UNORM:
			return GL_RG16;
			break;
		case EF_R16G16B16_UNORM:
			return GL_RGB16;
			break;
		case EF_R16G16B16A16_UNORM:
			return GL_RGBA16;
			break;
		case EF_R16_UINT:
			return GL_R16UI;
			break;
		case EF_R16G16_UINT:
			return GL_RG16UI;
			break;
		case EF_R16G16B16_UINT:
			return GL_RGB16UI;
			break;
		case EF_R16G16B16A16_UINT:
			return GL_RGBA16UI;
			break;
		case EF_R16_SINT:
			return GL_R16I;
			break;
		case EF_R16G16_SINT:
			return GL_RG16I;
			break;
		case EF_R16G16B16_SINT:
			return GL_RGB16I;
			break;
		case EF_R16G16B16A16_SINT:
			return GL_RGBA16I;
			break;
		case EF_R16_SNORM:
			return GL_R16_SNORM;
			break;
		case EF_R16G16_SNORM:
			return GL_RG16_SNORM;
			break;
		case EF_R16G16B16_SNORM:
			return GL_RGB16_SNORM;
			break;
		case EF_R16G16B16A16_SNORM:
			return GL_RGBA16_SNORM;
			break;
		case EF_R32_UINT:
			return GL_R32UI;
			break;
		case EF_R32G32_UINT:
			return GL_RG32UI;
			break;
		case EF_R32G32B32_UINT:
			return GL_RGB32UI;
			break;
		case EF_R32G32B32A32_UINT:
			return GL_RGBA32UI;
			break;
		case EF_R32_SINT:
			return GL_R32I;
			break;
		case EF_R32G32_SINT:
			return GL_RG32I;
			break;
		case EF_R32G32B32_SINT:
			return GL_RGB32I;
			break;
		case EF_R32G32B32A32_SINT:
			return GL_RGBA32I;
			break;
		case EF_A2B10G10R10_UNORM_PACK32:
			return GL_RGB10_A2;
			break;
		case EF_A2B10G10R10_UINT_PACK32:
			return GL_RGB10_A2UI;
			break;
		case EF_R8G8B8_SRGB:
			return GL_SRGB8;
			break;
		case EF_R8G8B8A8_SRGB:
			return GL_SRGB8_ALPHA8;
			break;
		case EF_BC1_RGB_UNORM_BLOCK:
			return GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
			break;
		case EF_BC1_RGBA_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
			break;
		case EF_BC2_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
			break;
		case EF_BC3_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
			break;
		case EF_BC1_RGB_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
			break;
		case EF_BC1_RGBA_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
			break;
		case EF_BC2_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
			break;
		case EF_BC3_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
			break;
		case EF_BC7_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_BPTC_UNORM;
		case EF_BC7_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM;
		case EF_BC6H_SFLOAT_BLOCK:
			return GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
		case EF_BC6H_UFLOAT_BLOCK:
			return GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
		case EF_ETC2_R8G8B8_UNORM_BLOCK:
			return GL_COMPRESSED_RGB8_ETC2;
		case EF_ETC2_R8G8B8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ETC2;
		case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
			return GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
		case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2;
		case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA8_ETC2_EAC;
		case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC;
		case EF_EAC_R11G11_UNORM_BLOCK:
			return GL_COMPRESSED_RG11_EAC;
		case EF_EAC_R11G11_SNORM_BLOCK:
			return GL_COMPRESSED_SIGNED_RG11_EAC;
		case EF_EAC_R11_UNORM_BLOCK:
			return GL_COMPRESSED_R11_EAC;
		case EF_EAC_R11_SNORM_BLOCK:
			return GL_COMPRESSED_SIGNED_R11_EAC;
		case EF_ASTC_4x4_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
		case EF_ASTC_5x4_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
		case EF_ASTC_5x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_5x5_KHR;
		case EF_ASTC_6x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_6x5_KHR;
		case EF_ASTC_6x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_6x6_KHR;
		case EF_ASTC_8x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x5_KHR;
		case EF_ASTC_8x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x6_KHR;
		case EF_ASTC_8x8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x8_KHR;
		case EF_ASTC_10x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x5_KHR;
		case EF_ASTC_10x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_10x8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_10x10_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_12x10_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_12x12_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_4x4_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR;
		case EF_ASTC_5x4_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR;
		case EF_ASTC_5x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR;
		case EF_ASTC_6x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR;
		case EF_ASTC_6x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR;
		case EF_ASTC_8x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR;
		case EF_ASTC_8x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR;
		case EF_ASTC_8x8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR;
		case EF_ASTC_10x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR;
		case EF_ASTC_10x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR;
		case EF_ASTC_10x8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR;
		case EF_ASTC_10x10_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR;
		case EF_ASTC_12x10_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR;
		case EF_ASTC_12x12_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR;
		case EF_E5B9G9R9_UFLOAT_PACK32:
			return GL_RGB9_E5;
			break;
		case EF_D16_UNORM:
			return GL_DEPTH_COMPONENT16;
			break;
		case EF_X8_D24_UNORM_PACK32:
			return GL_DEPTH_COMPONENT24;
			break;
		case EF_D24_UNORM_S8_UINT:
			return GL_DEPTH24_STENCIL8;
			break;
		case EF_D32_SFLOAT:
			return GL_DEPTH_COMPONENT32F;
			break;
		case EF_D32_SFLOAT_S8_UINT:
			return GL_DEPTH32F_STENCIL8;
			break;
		case EF_S8_UINT:
			return GL_STENCIL_INDEX8;
			break;
		default:
			break;
	}
	return GL_INVALID_ENUM;
}

#if 0
inline asset::E_FORMAT	getOurFormatFromSizedOpenGLFormat(GLenum sizedFormat)
{
	using namespace asset;
	switch (sizedFormat)
	{
		case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
			return EF_ASTC_4x4_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:
			return EF_ASTC_5x4_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:
			return EF_ASTC_5x5_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:
			return EF_ASTC_6x5_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:
			return EF_ASTC_6x6_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:
			return EF_ASTC_8x5_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:
			return EF_ASTC_8x6_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
			return EF_ASTC_8x8_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:
			return EF_ASTC_10x5_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:
			return EF_ASTC_10x6_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:
			return EF_ASTC_10x8_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:
			return EF_ASTC_10x10_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:
			return EF_ASTC_12x10_UNORM_BLOCK;
		case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:
			return EF_ASTC_12x12_UNORM_BLOCK;

		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
			return EF_ASTC_4x4_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
			return EF_ASTC_5x4_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
			return EF_ASTC_5x5_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
			return EF_ASTC_6x5_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
			return EF_ASTC_6x6_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
			return EF_ASTC_8x5_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
			return EF_ASTC_8x6_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
			return EF_ASTC_8x8_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
			return EF_ASTC_10x5_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
			return EF_ASTC_10x6_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
			return EF_ASTC_10x8_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
			return EF_ASTC_10x10_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
			return EF_ASTC_12x10_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
			return EF_ASTC_12x12_SRGB_BLOCK;

			/*case asset::EF_BC1_RGB_SRGB_BLOCK:
				return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
				break;
			case asset::EF_BC1_RGBA_SRGB_BLOCK:
				return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
				break;
			case asset::EF_BC2_SRGB_BLOCK:
				return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
				break;
			case asset::EF_BC3_SRGB_BLOCK:
				return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
				break;*/
		case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
			return asset::EF_BC1_RGB_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
			return asset::EF_BC1_RGBA_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
			return asset::EF_BC2_SRGB_BLOCK;
		case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
			return asset::EF_BC3_SRGB_BLOCK;
		case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
			return asset::EF_BC1_RGB_UNORM_BLOCK;
			break;
		case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
			return asset::EF_BC1_RGBA_UNORM_BLOCK;
			break;
		case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
			return asset::EF_BC2_UNORM_BLOCK;
			break;
		case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
			return asset::EF_BC3_UNORM_BLOCK;
			break;
		case GL_COMPRESSED_RGBA_BPTC_UNORM:
			return asset::EF_BC7_UNORM_BLOCK;
		case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
			return asset::EF_BC7_SRGB_BLOCK;
		case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
			return asset::EF_BC6H_SFLOAT_BLOCK;
		case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
			return asset::EF_BC6H_UFLOAT_BLOCK;
		case GL_COMPRESSED_RGB8_ETC2:
			return asset::EF_ETC2_R8G8B8_UNORM_BLOCK;
		case GL_COMPRESSED_SRGB8_ETC2:
			return asset::EF_ETC2_R8G8B8_SRGB_BLOCK;
		case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
			return asset::EF_ETC2_R8G8B8A1_UNORM_BLOCK;
		case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
			return asset::EF_ETC2_R8G8B8A1_SRGB_BLOCK;
		case GL_COMPRESSED_RGBA8_ETC2_EAC:
			return asset::EF_ETC2_R8G8B8A8_UNORM_BLOCK;
		case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
			return asset::EF_ETC2_R8G8B8A8_SRGB_BLOCK;
		case GL_COMPRESSED_RG11_EAC:
			return asset::EF_EAC_R11G11_UNORM_BLOCK;
		case GL_COMPRESSED_SIGNED_RG11_EAC:
			return asset::EF_EAC_R11G11_SNORM_BLOCK;
		case GL_COMPRESSED_R11_EAC:
			return asset::EF_EAC_R11_UNORM_BLOCK;
		case GL_COMPRESSED_SIGNED_R11_EAC:
			return asset::EF_EAC_R11_SNORM_BLOCK;
		case GL_STENCIL_INDEX8:
			return asset::EF_S8_UINT;
			break;
		case GL_RGBA2:
			///return asset::EF_8BIT_PIX;
			break;
		case GL_R3_G3_B2:
			///return asset::EF_8BIT_PIX;
			break;
		case GL_R8:
			return asset::EF_R8_UNORM;
			break;
		case GL_R8I:
			return asset::EF_R8_SINT;
			break;
		case GL_R8UI:
			return asset::EF_R8_UINT;
			break;
		case GL_R8_SNORM:
			return asset::EF_R8_SNORM;
			break;
		case GL_RGB4:
			///return asset::EF_16BIT_PIX;
			break;
		case GL_RGB5:
			///return asset::EF_;
			break;
		case GL_DEPTH_COMPONENT16:
			return asset::EF_D16_UNORM;
			break;
		case GL_RGBA4:
			return asset::EF_R4G4B4A4_UNORM_PACK16;
			break;
		case GL_RGB5_A1:
			return asset::EF_R5G5B5A1_UNORM_PACK16;
			break;
		case GL_RG8:
			return asset::EF_R8G8_UNORM;
			break;
		case GL_SR8_EXT:
			return asset::EF_R8_SRGB;
			break;
		case GL_RG8I:
			return asset::EF_R8G8_SINT;
			break;
		case GL_RG8UI:
			return asset::EF_R8G8_UINT;
			break;
		case GL_RG8_SNORM:
			return asset::EF_R8G8_SNORM;
			break;
		case GL_R16:
			return asset::EF_R16_UNORM;
			break;
		case GL_R16I:
			return asset::EF_R16_SINT;
			break;
		case GL_R16UI:
			return asset::EF_R16_UINT;
			break;
		case GL_R16_SNORM:
			return asset::EF_R16_SNORM;
			break;
		case GL_R16F:
			return asset::EF_R16_SFLOAT;
			break;
		case GL_DEPTH_COMPONENT24:
			return asset::EF_X8_D24_UNORM_PACK32;
			break;
		case GL_RGB8:
			return asset::EF_R8G8B8_UNORM;
			break;
		case GL_SRG8_EXT:
			return asset::EF_R8G8_SRGB;
			break;
		case GL_RGB8I:
			return asset::EF_R8G8B8_SINT;
			break;
		case GL_RGB8UI:
			return asset::EF_R8G8B8_UINT;
			break;
		case GL_RGB8_SNORM:
			return asset::EF_R8G8B8_SNORM;
			break;
		case GL_SRGB8:
			return asset::EF_R8G8B8_SRGB;
			break;
		case GL_RGB10:
			///return asset::EF_;
			break;
		case GL_DEPTH24_STENCIL8:
			return asset::EF_D24_UNORM_S8_UINT;
			break;
		case GL_DEPTH_COMPONENT32:
			///return asset::EF_DEPTH32;
			break;
		case GL_DEPTH_COMPONENT32F:
			return asset::EF_D32_SFLOAT;
			break;
		case GL_RGBA8:
			return asset::EF_R8G8B8A8_UNORM;
			break;
		case GL_RGBA8I:
			return asset::EF_R8G8B8A8_SINT;
			break;
		case GL_RGBA8UI:
			return asset::EF_R8G8B8A8_UINT;
			break;
		case GL_RGBA8_SNORM:
			return asset::EF_R8G8B8A8_SNORM;
			break;
		case GL_SRGB8_ALPHA8:
			return asset::EF_R8G8B8A8_SRGB;
			break;
		case GL_RGB10_A2:
			return asset::EF_A2B10G10R10_UNORM_PACK32;
			break;
		case GL_RGB10_A2UI:
			return asset::EF_A2B10G10R10_UINT_PACK32;
			break;
		case GL_R11F_G11F_B10F:
			return asset::EF_B10G11R11_UFLOAT_PACK32;
			break;
		case GL_RGB9_E5:
			return asset::EF_E5B9G9R9_UFLOAT_PACK32;
			break;
		case GL_RG16:
			return asset::EF_R16G16_UNORM;
			break;
		case GL_RG16I:
			return asset::EF_R16G16_SINT;
			break;
		case GL_RG16UI:
			return asset::EF_R16G16_UINT;
			break;
		case GL_RG16F:
			return asset::EF_R16G16_SFLOAT;
			break;
		case GL_R32I:
			return asset::EF_R32G32_SINT;
			break;
		case GL_R32UI:
			return asset::EF_R32G32_UINT;
			break;
		case GL_R32F:
			return asset::EF_R32_SFLOAT;
			break;
		case GL_RGB12:
			///return asset::EF_;
			break;
		case GL_DEPTH32F_STENCIL8:
			return asset::EF_D32_SFLOAT_S8_UINT;
			break;
		case GL_RGBA12:
			///return asset::EF_;
			break;
		case GL_RGB16:
			return asset::EF_R16G16B16_UNORM;
			break;
		case GL_RGB16I:
			return asset::EF_R16G16B16_SINT;
			break;
		case GL_RGB16UI:
			return asset::EF_R16G16B16_UINT;
			break;
		case GL_RGB16_SNORM:
			return asset::EF_R16G16B16_SNORM;
			break;
		case GL_RGB16F:
			return asset::EF_R16G16B16_SFLOAT;
			break;
		case GL_RGBA16:
			return asset::EF_R16G16B16A16_UNORM;
			break;
		case GL_RGBA16I:
			return asset::EF_R16G16B16A16_SINT;
			break;
		case GL_RGBA16UI:
			return asset::EF_R16G16B16A16_UINT;
			break;
		case GL_RGBA16F:
			return asset::EF_R16G16B16A16_SFLOAT;
			break;
		case GL_RG32I:
			return asset::EF_R32G32_SINT;
			break;
		case GL_RG32UI:
			return asset::EF_R32G32_UINT;
			break;
		case GL_RG32F:
			return asset::EF_R32G32_SFLOAT;
			break;
		case GL_RGB32I:
			return asset::EF_R32G32B32_SINT;
			break;
		case GL_RGB32UI:
			return asset::EF_R32G32B32_UINT;
			break;
		case GL_RGB32F:
			return asset::EF_R32G32B32_SFLOAT;
			break;
		case GL_RGBA32I:
			return asset::EF_R32G32B32A32_SINT;
			break;
		case GL_RGBA32UI:
			return asset::EF_R32G32B32A32_UINT;
			break;
		case GL_RGBA32F:
			return asset::EF_R32G32B32A32_SFLOAT;
			break;
		default:
			break;
	}
	return asset::EF_UNKNOWN;
}
#endif
static GLenum formatEnumToGLenum(IOpenGL_FunctionTable* gl, asset::E_FORMAT fmt)
{
    using namespace asset;
    switch (fmt)
    {
		case EF_R16_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16B16_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
			return GL_HALF_FLOAT;
		case EF_R32_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R32G32B32_SFLOAT:
		case EF_R32G32B32A32_SFLOAT:
			return GL_FLOAT;
		case EF_R8_UNORM:
		case EF_R8_UINT:
		case EF_R8G8_UNORM:
		case EF_R8G8_UINT:
		case EF_R8G8B8_UNORM:
		case EF_R8G8B8_UINT:
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_UINT:
		case EF_R8_USCALED:
		case EF_R8G8_USCALED:
		case EF_R8G8B8_USCALED:
		case EF_R8G8B8A8_USCALED:
		case EF_B8G8R8A8_UNORM:
			return GL_UNSIGNED_BYTE;
		case EF_R8_SRGB:
			if (gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_R8))
				return GL_UNSIGNED_BYTE;
			break;
		case EF_R8G8_SRGB:
			if (gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_RG8))
				return GL_UNSIGNED_BYTE;
			break;
		case EF_R8_SNORM:
		case EF_R8_SINT:
		case EF_R8G8_SNORM:
		case EF_R8G8_SINT:
		case EF_R8G8B8_SNORM:
		case EF_R8G8B8_SINT:
		case EF_R8G8B8A8_SNORM:
		case EF_R8G8B8A8_SINT:
		case EF_R8_SSCALED:
		case EF_R8G8_SSCALED:
		case EF_R8G8B8_SSCALED:
		case EF_R8G8B8A8_SSCALED:
			return GL_BYTE;
		case EF_R16_UNORM:
		case EF_R16_UINT:
		case EF_R16_USCALED:
		case EF_R16G16_UNORM:
		case EF_R16G16_UINT:
		case EF_R16G16_USCALED:
		case EF_R16G16B16_UNORM:
		case EF_R16G16B16_UINT:
		case EF_R16G16B16_USCALED:
		case EF_R16G16B16A16_UNORM:
		case EF_R16G16B16A16_UINT:
		case EF_R16G16B16A16_USCALED:
			return GL_UNSIGNED_SHORT;
		case EF_R16_SNORM:
		case EF_R16_SINT:
		case EF_R16_SSCALED:
		case EF_R16G16_SNORM:
		case EF_R16G16_SINT:
		case EF_R16G16_SSCALED:
		case EF_R16G16B16_SNORM:
		case EF_R16G16B16_SINT:
		case EF_R16G16B16_SSCALED:
		case EF_R16G16B16A16_SNORM:
		case EF_R16G16B16A16_SINT:
		case EF_R16G16B16A16_SSCALED:
			return GL_SHORT;
		case EF_R32_UINT:
		case EF_R32G32_UINT:
		case EF_R32G32B32_UINT:
		case EF_R32G32B32A32_UINT:
			return GL_UNSIGNED_INT;
		case EF_R32_SINT:
		case EF_R32G32_SINT:
		case EF_R32G32B32_SINT:
		case EF_R32G32B32A32_SINT:
			return GL_INT;
		case EF_A2R10G10B10_UNORM_PACK32:
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_A2B10G10R10_USCALED_PACK32:
		case EF_A2B10G10R10_UINT_PACK32:
			return GL_UNSIGNED_INT_2_10_10_10_REV;
		case EF_A2R10G10B10_SNORM_PACK32:
		case EF_A2B10G10R10_SNORM_PACK32:
		case EF_A2B10G10R10_SSCALED_PACK32:
		case EF_A2B10G10R10_SINT_PACK32:
			return GL_INT_2_10_10_10_REV;
		case EF_R64_SFLOAT:
		case EF_R64G64_SFLOAT:
		case EF_R64G64B64_SFLOAT:
		case EF_R64G64B64A64_SFLOAT:
			return GL_DOUBLE;
		case EF_E5B9G9R9_UFLOAT_PACK32:
			return GL_UNSIGNED_INT_5_9_9_9_REV;
		case EF_D24_UNORM_S8_UINT:
			return GL_UNSIGNED_INT_24_8;
		case EF_D32_SFLOAT_S8_UINT:
			return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
		case EF_B10G11R11_UFLOAT_PACK32:
			return GL_UNSIGNED_INT_10F_11F_11F_REV;
		case EF_B4G4R4A4_UNORM_PACK16:
		case EF_R4G4B4A4_UNORM_PACK16:
			return GL_UNSIGNED_SHORT_4_4_4_4;
		case EF_R5G6B5_UNORM_PACK16:
			return GL_UNSIGNED_SHORT_5_6_5;
		case EF_B5G6R5_UNORM_PACK16:
			return GL_UNSIGNED_SHORT_5_6_5_REV;
		case EF_R5G5B5A1_UNORM_PACK16:
		case EF_B5G5R5A1_UNORM_PACK16:
			return GL_UNSIGNED_SHORT_5_5_5_1;
		case EF_A1R5G5B5_UNORM_PACK16:
			return GL_UNSIGNED_SHORT_1_5_5_5_REV;
		default:
			return GL_INVALID_ENUM;
    }
	return GL_INVALID_ENUM;
}


//! Get opengl values for the GPU texture storage
inline void getOpenGLFormatAndParametersFromColorFormat(IOpenGL_FunctionTable* gl, asset::E_FORMAT format, GLenum& colorformat, GLenum& type, const system::logger_opt_ptr logger = nullptr)
{
	using namespace asset;
	// default
	colorformat = GL_INVALID_ENUM;
	type = GL_INVALID_ENUM;

	switch (format)
	{
		case asset::EF_R4G4B4A4_UNORM_PACK16:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_SHORT_4_4_4_4;
			break;
		case asset::EF_B4G4R4A4_UNORM_PACK16:
			colorformat = GL_BGRA;
			type = GL_UNSIGNED_SHORT_4_4_4_4;
			break;
		case asset::EF_R5G6B5_UNORM_PACK16:
			colorformat = GL_RGB;
			type = GL_UNSIGNED_SHORT_5_6_5;
			break;
		case asset::EF_B5G6R5_UNORM_PACK16:
			colorformat = GL_RGB;
			type = GL_UNSIGNED_SHORT_5_6_5_REV;
			break;
		case asset::EF_R5G5B5A1_UNORM_PACK16:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_SHORT_5_5_5_1;
			break;
		case asset::EF_B5G5R5A1_UNORM_PACK16:
			colorformat = GL_BGRA;
			type = GL_UNSIGNED_SHORT_5_5_5_1;
			break;
		case asset::EF_A1R5G5B5_UNORM_PACK16:
			colorformat = GL_BGRA_EXT;
			type = GL_UNSIGNED_SHORT_1_5_5_5_REV;
			break;
		case asset::EF_R8_UNORM:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8_SNORM:
		{
			colorformat = GL_RED;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8_UINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8_SINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8_SRGB:
		{
			if (!gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_R8))
				break;
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8_UNORM:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8_SNORM:
		{
			colorformat = GL_RG;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8G8_UINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8_SINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8G8_SRGB:
		{
			if (!gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_sRGB_RG8))
				break;
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8B8_UNORM:
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;
			break;
		case asset::EF_R8G8B8_SNORM:
			colorformat = GL_RGB;
			type = GL_BYTE;
			break;
		case asset::EF_R8G8B8_UINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8B8_SINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8G8B8_SRGB:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8B8A8_UNORM:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
			break;
		case asset::EF_R8G8B8A8_SNORM:
			colorformat = GL_RGBA;
			type = GL_BYTE;
			break;
		case asset::EF_R8G8B8A8_UINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_R8G8B8A8_SINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_BYTE;
		}
		break;
		case asset::EF_R8G8B8A8_SRGB:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_B8G8R8A8_UNORM:
			colorformat = GL_BGRA_EXT;
			type = GL_UNSIGNED_BYTE;
			break;
		case asset::EF_B8G8R8A8_SNORM:
			colorformat = GL_BGRA_EXT;
			type = GL_BYTE;
			break;
		case asset::EF_B8G8R8A8_UINT:
			colorformat = GL_BGRA_INTEGER;
			type = GL_UNSIGNED_BYTE;
			break;
		case asset::EF_B8G8R8A8_SINT:
			colorformat = GL_BGRA_INTEGER;
			type = GL_BYTE;
			break;
		case asset::EF_B8G8R8A8_SRGB:
			colorformat = GL_BGRA_EXT;
			type = GL_UNSIGNED_BYTE;
			break;
		case asset::EF_A8B8G8R8_UNORM_PACK32:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_INT_8_8_8_8_REV;
			break;
		case asset::EF_A8B8G8R8_SNORM_PACK32:
			colorformat = GL_RGBA;
			type = GL_BYTE;
			break;
		case asset::EF_A8B8G8R8_UINT_PACK32:
			colorformat = GL_RGBA_INTEGER;
			type = GL_UNSIGNED_INT_8_8_8_8_REV;
			break;
		case asset::EF_A8B8G8R8_SINT_PACK32:
			colorformat = GL_RGBA_INTEGER;
			type = GL_BYTE;
			break;
		case asset::EF_A8B8G8R8_SRGB_PACK32:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_INT_8_8_8_8_REV;
			break;
		case EF_A2R10G10B10_UNORM_PACK32:
			colorformat = GL_BGRA;
			type = GL_UNSIGNED_INT_2_10_10_10_REV;
			break;
		case EF_A2R10G10B10_SNORM_PACK32:
			colorformat = GL_BGRA;
			type = GL_INT_2_10_10_10_REV;
			break;
		case EF_A2R10G10B10_UINT_PACK32:
			colorformat = GL_BGRA_INTEGER;
			type = GL_UNSIGNED_INT_2_10_10_10_REV;
			break;
		case EF_A2R10G10B10_SINT_PACK32:
			colorformat = GL_BGRA_INTEGER;
			type = GL_INT_2_10_10_10_REV;
			break;
		case EF_A2B10G10R10_UNORM_PACK32:
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_INT_2_10_10_10_REV;
			break;
		case EF_A2B10G10R10_SNORM_PACK32:
			colorformat = GL_RGBA;
			type = GL_INT_2_10_10_10_REV;
			break;
		case EF_A2B10G10R10_UINT_PACK32:
			colorformat = GL_RGBA_INTEGER;
			type = GL_UNSIGNED_INT_2_10_10_10_REV;
			break;
		case EF_A2B10G10R10_SINT_PACK32:
			colorformat = GL_RGBA_INTEGER;
			type = GL_INT_2_10_10_10_REV;
			break;
		case asset::EF_R16_UNORM:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16_SNORM:
		{
			colorformat = GL_RED;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16_UINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16_SINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16_SFLOAT:
		{
			colorformat = GL_RED;
			type = GL_HALF_FLOAT;
		}
		break;
		case asset::EF_R16G16_UNORM:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16_SNORM:
		{
			colorformat = GL_RG;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16_UINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16_SINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16_SFLOAT:
		{
			colorformat = GL_RG;
			type = GL_HALF_FLOAT;
		}
		break;
		case asset::EF_R16G16B16_SNORM:
		{
			colorformat = GL_RGB;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16B16_UNORM:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16B16_UINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16B16_SINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16B16_SFLOAT:
			colorformat = GL_RGB;
			type = GL_HALF_FLOAT;
			break;
		case asset::EF_R16G16B16A16_SNORM:
		{
			colorformat = GL_RGBA;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16B16A16_UNORM:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16B16A16_UINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_R16G16B16A16_SINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_SHORT;
		}
		break;
		case asset::EF_R16G16B16A16_SFLOAT:
		{
			colorformat = GL_RGBA;
			type = GL_HALF_FLOAT;
		}
		break;
		case asset::EF_R32_UINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_UNSIGNED_INT;
		}
		break;
		case asset::EF_R32_SINT:
		{
			colorformat = GL_RED_INTEGER;
			type = GL_INT;
		}
		break;
		case asset::EF_R32_SFLOAT:
		{
			colorformat = GL_RED;
			type = GL_FLOAT;
		}
		break;
		case asset::EF_R32G32_UINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_UNSIGNED_INT;
		}
		break;
		case asset::EF_R32G32_SINT:
		{
			colorformat = GL_RG_INTEGER;
			type = GL_INT;
		}
		break;
		case asset::EF_R32G32_SFLOAT:
		{
			colorformat = GL_RG;
			type = GL_FLOAT;
		}
		break;
		case asset::EF_R32G32B32_UINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_UNSIGNED_INT;
		}
		break;
		case asset::EF_R32G32B32_SINT:
		{
			colorformat = GL_RGB_INTEGER;
			type = GL_INT;
		}
		break;
		case asset::EF_R32G32B32_SFLOAT:
		{
			colorformat = GL_RGB;
			type = GL_FLOAT;
		}
		break;
		case asset::EF_R32G32B32A32_UINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_UNSIGNED_INT;
		}
		break;
		case asset::EF_R32G32B32A32_SINT:
		{
			colorformat = GL_RGBA_INTEGER;
			type = GL_INT;
		}
		break;
		case asset::EF_R32G32B32A32_SFLOAT:
		{
			colorformat = GL_RGBA;
			type = GL_FLOAT;
		}
		break;
		case asset::EF_B10G11R11_UFLOAT_PACK32:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_INT_10F_11F_11F_REV;
		}
		break;
		case asset::EF_E5B9G9R9_UFLOAT_PACK32:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_INT_5_9_9_9_REV;
		}
		break;
		/// this is totally wrong but safe - most probs have to reupload
		case asset::EF_D16_UNORM:
		{
			colorformat = GL_DEPTH;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_X8_D24_UNORM_PACK32:
		{
			colorformat = GL_DEPTH;
			type = GL_UNSIGNED_SHORT;
		}
		break;
		case asset::EF_D32_SFLOAT:
		{
			colorformat = GL_DEPTH;
			type = GL_FLOAT;
		}
		break;
		case asset::EF_S8_UINT:
		{
			colorformat = GL_STENCIL;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_D24_UNORM_S8_UINT:
		{
			colorformat = GL_DEPTH_STENCIL;
			type = GL_UNSIGNED_INT_24_8;
		}
		break;
		case asset::EF_D32_SFLOAT_S8_UINT:
		{
			colorformat = GL_DEPTH_STENCIL;
			type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
		}
		break;
#if 0
		case asset::EF_BC1_RGB_UNORM_BLOCK:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC1_RGBA_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC2_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC3_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC1_RGB_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC1_RGBA_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC2_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC3_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC7_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC7_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_BC6H_SFLOAT_BLOCK:
		{
			colorformat = GL_RGB;
			type = GL_HALF_FLOAT;
		}
		break;
		case asset::EF_BC6H_UFLOAT_BLOCK:
		{
			colorformat = GL_RGB;
			type = GL_HALF_FLOAT;
		}
		break;
		case asset::EF_ETC2_R8G8B8_UNORM_BLOCK:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_ETC2_R8G8B8_SRGB_BLOCK:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_ETC2_R8G8B8A1_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_ETC2_R8G8B8A1_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_ETC2_R8G8B8A8_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_ETC2_R8G8B8A8_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_EAC_R11G11_UNORM_BLOCK:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_EAC_R11G11_SNORM_BLOCK:
		{
			colorformat = GL_RG;
			type = GL_BYTE;
		}
		break;
		case asset::EF_EAC_R11_UNORM_BLOCK:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case asset::EF_EAC_R11_SNORM_BLOCK:
		{
			colorformat = GL_RED;
			type = GL_BYTE;
		}
		break;
		case EF_ASTC_4x4_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_5x4_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_5x5_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_6x5_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_6x6_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x5_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x6_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x8_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x5_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x6_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x8_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x10_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_12x10_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_12x12_UNORM_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_4x4_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_5x4_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_5x5_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_6x5_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_6x6_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x5_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x6_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_8x8_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x5_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x6_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x8_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_10x10_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_12x10_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
		case EF_ASTC_12x12_SRGB_BLOCK:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
		}
		break;
#endif
		default:
			break;
	}

	if (colorformat == GL_INVALID_ENUM || type == GL_INVALID_ENUM)
		logger.log("Unsupported upload format", system::ILogger::ELL_ERROR);
}

}


#endif