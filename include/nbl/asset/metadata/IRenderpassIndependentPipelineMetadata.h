// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_METADATA_H_INCLUDED__

/*
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/asset/ICPUSpecializedShader.h"
*/
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/IDescriptor.h"

#include <compare>

//#include "nbl/asset/utils/IBuiltinIncludeLoader.h"

//#include "nbl/asset/asset_utils.h"

namespace nbl
{
namespace asset
{

//! A class to derive loader-specific pipeline metadata objects from
/**
	Pipelines may sometimes require external inputs from outside of the resourced they were built with, for total flexibility
	we cannot standardise "conventions" of shader inputs like in game engines or old-style frameworks.

	But we can provide useful metadata from the loader.
*/
class IRenderpassIndependentPipelineMetadata : public core::Interface
{
	public:
		//! A common struct to unify the metadata declarations.
		/**
			When a pipeline or meshbuffer asset require some inputs to work correctly,
			they can put this info in the metadata in a custom way for every loader.

			Most meshbuffers or graphics pipelines will need to know about the model's world view projection matrix,
			as well as the inverse transpose of the world matrix, and the camera world position in case of lighting.

			However more advanced loaders such as glTF may want to let us know that there is a texture being used as
			an environment map, and with this knowledge we could know that we could, for example change it or provide it
			to match the object with the scene.

			Remember that we always have the shader introspector which can give us the information about all the descriptors
			that a shader uses, but it won't give us the semantics, or in simple english, the meaning why they are being used.

			(@see ICPURenderpassIndependentPipeilne and @see ICPUComputePipeline)
		*/
		struct ShaderInput
		{
			struct DescriptorCommon
			{
				uint32_t set;
				uint32_t binding;

				auto operator<=>(const DescriptorCommon&) const = default;
			};

			struct CombinedImageSampler : DescriptorCommon
			{
				IImageView<ICPUImage>::E_TYPE viewType;
				// TODO: some info about format class

				auto operator<=>(const CombinedImageSampler&) const = default;
			};
			struct StorageImage : DescriptorCommon
			{
				E_FORMAT format;

				auto operator<=>(const StorageImage&) const = default;
			};
			struct TexelBuffer : DescriptorCommon
			{
				// relative to the start of the IBufferView
				uint32_t relByteoffset;
				// TODO: some info about format class

				auto operator<=>(const TexelBuffer&) const = default;
			};
			struct StorageTexelBuffer : DescriptorCommon
			{
				// relative to the start of the IBufferView
				uint32_t relByteoffset;
				E_FORMAT format;

				auto operator<=>(const StorageTexelBuffer&) const = default;
			};
			struct Buffer : DescriptorCommon
			{
				// relative to the offset of the descriptor when bound (true byteoffset = static descriptor-set defined + dynamic [if enabled] + this value)
				uint32_t relByteoffset;
                uint32_t bytesize;

				auto operator<=>(const Buffer&) const = default;
			};
			struct PushConstant
			{
				uint32_t byteOffset;

				auto operator<=>(const PushConstant&) const = default;
			};
			enum class E_TYPE: uint8_t 
			{
				ET_COMBINED_IMAGE_SAMPLER = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER),
				ET_STORAGE_IMAGE = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_STORAGE_IMAGE),
				ET_UNIFORM_TEXEL_BUFFER = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER),
				ET_STORAGE_TEXEL_BUFFER = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER),
				ET_UNIFORM_BUFFER = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_UNIFORM_BUFFER),
				ET_STORAGE_BUFFER = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_STORAGE_BUFFER),
				ET_INPUT_ATTACHMENT = static_cast<uint8_t>(IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT),
				ET_PUSH_CONSTANT = 11
			};
			E_TYPE type;
			IShader::E_SHADER_STAGE shaderAccessFlags;
			union
			{
				CombinedImageSampler	combinedImageSampler;
				StorageImage			storageImage;
				TexelBuffer				texelBuffer;
				StorageTexelBuffer		storageTexelBuffer;
				Buffer					uniformBufferObject;
				Buffer					storageBufferObject;
				PushConstant			pushConstant;
			};

			inline bool operator!=(const ShaderInput& other) const
			{
				return !std::memcmp(this, &other, sizeof(other));
			}
		};

		//! A non exhaustive list of commonly used shader input semantics
		enum E_COMMON_SHADER_INPUT
		{
			//! core::matrix4SIMD giving the total projection onto the screen from model-space coordinates
			ECSI_WORLD_VIEW_PROJ,
			//! core::matrix4SIMD giving the mapping from view-space into the pre-divide NDC space
			ECSI_PROJ,
			//! core::matrix3x4SIMD giving the view-space transformation from model-space coordinates
			ECSI_WORLD_VIEW,
			//! core::matrix3x4SIMD giving the view-space transformation from world-space
			ECSI_VIEW,
			//! core::matrix3x4SIMD giving the world-space transformation from model-space (last column is object world-space-position)
			ECSI_WORLD,
			//! core::matrix4SIMD giving the total projection to model-space coordinates from screen-space
			ECSI_WORLD_VIEW_PROJ_INVERSE,
			//! core::matrix4SIMD giving the mapping from the pre-divide NDC space into view-space
			ECSI_PROJ_INVERSE,
			//! core::matrix3x4SIMD giving the model-space transformation from view-space coordinates
			ECSI_WORLD_VIEW_INVERSE,
			//! core::matrix3x4SIMD giving the world-space transformation from view-space (last column is camera world-space-position)
			ECSI_VIEW_INVERSE,
			//! core::matrix3x4SIMD giving the model-space transformation from world-space
			ECSI_WORLD_INVERSE,
			//! transpose of core::matrix4SIMD giving the total projection to model-space coordinates from screen-space
			ECSI_WORLD_VIEW_PROJ_INVERSE_TRANSPOSE,
			//! transpose of core::matrix4SIMD giving the mapping from the pre-divide NDC space into view-space
			ECSI_PROJ_INVERSE_TRANSPOSE,
			//! transpose of core::matrix3x4SIMD giving the model-space transformation from view-space coordinates (upper 3x3 matrix can be used instead of `gl_NormalMatrix`)
			ECSI_WORLD_VIEW_INVERSE_TRANSPOSE,
			//! transpose of core::matrix3x4SIMD giving the world-space transformation from view-space (last row is camera world-space-position)
			ECSI_VIEW_INVERSE_TRANSPOSE,
			//! transpose of core::matrix3x4SIMD giving the model-space transformation from world-space (upper 3x3 matrix can transform model space normals to world space)
			ECSI_WORLD_INVERSE_TRANSPOSE,

			//! a simple non-filtered environment map as a cubemap
			ECSI_ENVIRONMENT_CUBEMAP,

			//! For internal 
			ECSI_COUNT
		};
		//! Tie the semantics to inputs
		struct ShaderInputSemantic
		{
			E_COMMON_SHADER_INPUT type;
			ShaderInput descriptorSection;

			inline bool operator!=(const ShaderInputSemantic& other) const
			{
				return type != other.type || descriptorSection != other.descriptorSection;
			}
		};
		core::SRange<const ShaderInputSemantic> m_inputSemantics;

		inline bool operator!=(const IRenderpassIndependentPipelineMetadata& other) const
		{
			if (m_inputSemantics.empty())
				return false;

			bool status = false;
			for (size_t i = 0; i < m_inputSemantics.size(); ++i)
				status = m_inputSemantics.begin()[i] != other.m_inputSemantics.begin()[i];

			return status;
		}

	protected:
		IRenderpassIndependentPipelineMetadata() : m_inputSemantics(nullptr,nullptr) {}
		IRenderpassIndependentPipelineMetadata(core::SRange<const ShaderInputSemantic>&& _inputSemantics) : m_inputSemantics(std::move(_inputSemantics)) {}
		virtual ~IRenderpassIndependentPipelineMetadata() = default;

		//!
		inline IRenderpassIndependentPipelineMetadata& operator=(IRenderpassIndependentPipelineMetadata&& other)
		{
			m_inputSemantics = std::move(other.m_inputSemantics);
			return *this;
		}
};


}
}

#endif
