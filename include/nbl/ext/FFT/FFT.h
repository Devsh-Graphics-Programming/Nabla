// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FFT_INCLUDED_
#define _NBL_EXT_FFT_INCLUDED_

#include "nabla.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUShader.h"


namespace nbl
{
namespace ext
{
namespace FFT
{

typedef uint32_t uint;
struct alignas(16) uvec3 {
	uint x,y,z;
};
struct alignas(16) uvec4 {
	uint x,y,z,w;
};
#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl";

class FFT : public core::TotalInterface
{
	public:
		struct Parameters_t alignas(16) : nbl_glsl_ext_FFT_Parameters_t {
		};

		enum class Direction : uint8_t {
			X = 0,
			Y = 1,
			Z = 2,
		};
		
		enum class PaddingType : uint8_t {
			CLAMP_TO_EDGE = 0,
			FILL_WITH_ZERO = 1,
			// TODO: mirror?
		};

		enum class DataType {
			SSBO,
			TEXTURE2D
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupDims[3];
			uint32_t workGroupCount[3];
		};

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;

		// returns dispatch size and fills the uniform data
		static inline DispatchInfo_t buildParameters(
			asset::VkExtent3D const & paddedInputDimensions,
			Direction direction)
		{
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			DispatchInfo_t ret = {};

			ret.workGroupDims[0] = DEFAULT_WORK_GROUP_SIZE;
			ret.workGroupDims[1] = 1;
			ret.workGroupDims[2] = 1;

			if(direction == Direction::X)
			{
				ret.workGroupCount[0] = 1;
				ret.workGroupCount[1] = paddedInputDimensions.height;
				ret.workGroupCount[2] = paddedInputDimensions.depth;
			}
			else if(direction == Direction::Y) {
				ret.workGroupCount[0] = paddedInputDimensions.width;
				ret.workGroupCount[1] = 1;
				ret.workGroupCount[2] = paddedInputDimensions.depth;
			}
			else if(direction == Direction::Z)
			{
				ret.workGroupCount[0] = paddedInputDimensions.width;
				ret.workGroupCount[1] = paddedInputDimensions.height;
				ret.workGroupCount[2] = 1;
			}

			return ret;
		}

		
		static inline asset::VkExtent3D padDimensionToNextPOT(asset::VkExtent3D dimension, asset::VkExtent3D const & minimum_dimension = asset::VkExtent3D{ 1, 1, 1 })
		{
			if(dimension.width < minimum_dimension.width)
				dimension.width = minimum_dimension.width;
			if(dimension.height < minimum_dimension.height)
				dimension.height = minimum_dimension.height;
			if(dimension.depth < minimum_dimension.depth)
				dimension.depth = minimum_dimension.depth;

			dimension.width = core::roundUpToPoT(dimension.width);
			dimension.height = core::roundUpToPoT(dimension.height);
			dimension.depth = core::roundUpToPoT(dimension.depth);

			return dimension;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		static core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultDescriptorSetLayout(video::IVideoDriver* driver, DataType inputType);
		
		//
		static core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::IVideoDriver* driver, DataType inputType);
		
		//
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & paddedInputDimensions, uint32_t numChannels)
		{
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			return (paddedInputDimensions.width * paddedInputDimensions.height * paddedInputDimensions.depth * numChannels) * (sizeof(float) * 2);
		}
		
		static core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(video::IVideoDriver* driver, DataType inputType, uint32_t maxDimensionSize);

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_DESCRIPTOR_COUNT = 2u;
		static inline void updateDescriptorSet(
			video::IVideoDriver * driver,
			video::IGPUDescriptorSet * set,
			core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
		{
			video::IGPUDescriptorSet::SDescriptorInfo pInfos[MAX_DESCRIPTOR_COUNT];
			video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[MAX_DESCRIPTOR_COUNT];

			for (auto i=0; i< MAX_DESCRIPTOR_COUNT; i++)
			{
				pWrites[i].dstSet = set;
				pWrites[i].arrayElement = 0u;
				pWrites[i].count = 1u;
				pWrites[i].info = pInfos+i;
			}

			// Input Buffer 
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[0].count = 1;
			pInfos[0].desc = inputBufferDescriptor;
			pInfos[0].buffer.size = inputBufferDescriptor->getSize();
			pInfos[0].buffer.offset = 0u;

			// Output Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[1].count = 1;
			pInfos[1].desc = outputBufferDescriptor;
			pInfos[1].buffer.size = outputBufferDescriptor->getSize();
			pInfos[1].buffer.offset = 0u;

			driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
		}

		static inline void updateDescriptorSet(
			video::IVideoDriver * driver,
			video::IGPUDescriptorSet * set,
			core::smart_refctd_ptr<video::IGPUImageView> inputImageDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor,
			asset::ISampler::E_TEXTURE_CLAMP textureWrap)
		{
			auto sampler = getSampler(driver,textureWrap);

			video::IGPUDescriptorSet::SDescriptorInfo pInfos[MAX_DESCRIPTOR_COUNT];
			video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[MAX_DESCRIPTOR_COUNT];

			for (auto i = 0; i < MAX_DESCRIPTOR_COUNT; i++)
			{
				pWrites[i].dstSet = set;
				pWrites[i].arrayElement = 0u;
				pWrites[i].count = 1u;
				pWrites[i].info = pInfos+i;
			}

			// Input Buffer 
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
			pWrites[0].count = 1;
			pInfos[0].desc = inputImageDescriptor;
			pInfos[0].image.sampler = sampler;
			pInfos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);;

			// Output Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[1].count = 1;
			pInfos[1].desc = outputBufferDescriptor;
			pInfos[1].buffer.size = outputBufferDescriptor->getSize();
			pInfos[1].buffer.offset = 0u;

			driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
		}

		static inline void dispatchHelper(
			video::IVideoDriver* driver,
			const DispatchInfo_t& dispatchInfo,
			bool issueDefaultBarrier=true)
		{
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

		static inline void pushConstants(
			video::IVideoDriver* driver,
			video::IGPUPipelineLayout * pipelineLayout,
			asset::VkExtent3D const & inputDimension,
			asset::VkExtent3D const & paddedInputDimension,
			Direction direction,
			bool isInverse, 
			uint32_t numChannels,
			PaddingType paddingType = PaddingType::CLAMP_TO_EDGE)
		{

			uint8_t isInverse_u8 = isInverse;
			uint8_t direction_u8 = static_cast<uint8_t>(direction);
			uint8_t paddingType_u8 = static_cast<uint8_t>(paddingType);
			
			uint32_t packed = (direction_u8 << 16u) | (isInverse_u8 << 8u) | paddingType_u8;

			Parameters_t params = {};
			params.dimension.x = inputDimension.width;
			params.dimension.y = inputDimension.height;
			params.dimension.z = inputDimension.depth;
			params.dimension.w = packed;
			params.padded_dimension.x = paddedInputDimension.width;
			params.padded_dimension.y = paddedInputDimension.height;
			params.padded_dimension.z = paddedInputDimension.depth;
			params.padded_dimension.w = numChannels;

			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &params);
		}

		static void defaultBarrier();

	private:
		FFT() = delete;
		//~FFT() = delete;

		static core::smart_refctd_ptr<video::IGPUSampler> getSampler(video::IVideoDriver* driver, asset::ISampler::E_TEXTURE_CLAMP textureWrap);
};


}
}
}

#endif
