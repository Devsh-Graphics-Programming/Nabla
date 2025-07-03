#include "nbl/asset/ECommonEnums.h"

namespace nbl::asset
{

constexpr static int32_t findLSB(size_t val)
{
	if constexpr(std::is_constant_evaluated())
	{
		for (size_t ix=0ull; ix<sizeof(size_t)*8; ix++)
			if ((0x1ull << ix) & val) return ix;
		return ~0u;
	} else
	{
		return hlsl::findLSB(val);
	}
}

core::bitflag<PIPELINE_STAGE_FLAGS> allPreviousStages(core::bitflag<PIPELINE_STAGE_FLAGS> stages)
{
	struct PerStagePreviousStages
	{
		public:
			constexpr PerStagePreviousStages()
			{
				// set all stage to have itself as their previous stages
				for (auto i = 0; i < std::numeric_limits<PIPELINE_STAGE_FLAGS>::digits; i++)
					data[i] = static_cast<PIPELINE_STAGE_FLAGS>(i);

				add(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT);

				add(PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT);

				// graphics primitive pipeline
				PIPELINE_STAGE_FLAGS primitivePrevStage = PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT;
				for (auto pipelineStage : {PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT, PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT, PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT, PIPELINE_STAGE_FLAGS::TESSELLATION_CONTROL_SHADER_BIT, PIPELINE_STAGE_FLAGS::TESSELLATION_EVALUATION_SHADER_BIT, PIPELINE_STAGE_FLAGS::GEOMETRY_SHADER_BIT, PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT, PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT, PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT, PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT})
				{
					if (pipelineStage == PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT)
						primitivePrevStage |= PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT;
					add(pipelineStage, primitivePrevStage);
					primitivePrevStage |= pipelineStage;
				}


			}
			constexpr const auto& operator[](const size_t ix) const {return data[ix];}

		private:

			constexpr void add(PIPELINE_STAGE_FLAGS stageFlag, PIPELINE_STAGE_FLAGS previousStageFlags)
			{
				const auto bitIx = findLSB(static_cast<size_t>(stageFlag));
				data[bitIx] |= previousStageFlags;
			}

			PIPELINE_STAGE_FLAGS data[std::numeric_limits<std::underlying_type_t<PIPELINE_STAGE_FLAGS>>::digits] = {};
	};

	constexpr PerStagePreviousStages bitToAccess = {};

	core::bitflag<PIPELINE_STAGE_FLAGS> retval = PIPELINE_STAGE_FLAGS::NONE;
	while (bool(stages.value))
	{
		const auto bitIx = findLSB(static_cast<size_t>(stages.value));
		retval |= bitToAccess[bitIx];
		stages ^= static_cast<PIPELINE_STAGE_FLAGS>(0x1u<<bitIx);
	}

	return retval;
}

core::bitflag<PIPELINE_STAGE_FLAGS> allLaterStages(core::bitflag<PIPELINE_STAGE_FLAGS> stages)
{
	struct PerStageLaterStages
	{
		public:
			constexpr PerStageLaterStages()
			{
				// set all stage to have itself as their next stages
				for (auto i = 0; i < std::numeric_limits<PIPELINE_STAGE_FLAGS>::digits; i++)
					data[i] = static_cast<PIPELINE_STAGE_FLAGS>(i);

				add(PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT, PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT);
				add(PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT, PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT);

				// graphics primitive pipeline
				PIPELINE_STAGE_FLAGS laterStage = PIPELINE_STAGE_FLAGS::NONE;
				const auto graphicsPrimitivePipelineOrders = std::array{ PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT, PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT, PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT, PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT, PIPELINE_STAGE_FLAGS::TESSELLATION_CONTROL_SHADER_BIT, PIPELINE_STAGE_FLAGS::TESSELLATION_EVALUATION_SHADER_BIT, PIPELINE_STAGE_FLAGS::GEOMETRY_SHADER_BIT, PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT, PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT, PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT, PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT };
				for (auto iter = graphicsPrimitivePipelineOrders.rbegin(); iter < graphicsPrimitivePipelineOrders.rend(); iter++)
				{
					const auto pipelineStage = *iter;
					add(pipelineStage, laterStage);
					laterStage |= pipelineStage;
				}

				add(PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT, PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT);
			}
			constexpr const auto& operator[](const size_t ix) const {return data[ix];}

		private:

			constexpr void add(PIPELINE_STAGE_FLAGS stageFlag, PIPELINE_STAGE_FLAGS laterStageFlags)
			{
				const auto bitIx = findLSB(static_cast<size_t>(stageFlag));
				data[bitIx] |= laterStageFlags;
			}

			PIPELINE_STAGE_FLAGS data[std::numeric_limits<std::underlying_type_t<PIPELINE_STAGE_FLAGS>>::digits] = {};
	};

	constexpr PerStageLaterStages bitToAccess = {};

	core::bitflag<PIPELINE_STAGE_FLAGS> retval = PIPELINE_STAGE_FLAGS::NONE;
	while (bool(stages.value))
	{
		const auto bitIx = findLSB(static_cast<size_t>(stages.value));
		retval |= bitToAccess[bitIx];
		stages ^= static_cast<PIPELINE_STAGE_FLAGS>(0x1u<<bitIx);
	}

	return retval;
}

core::bitflag<ACCESS_FLAGS> allAccessesFromStages(core::bitflag<PIPELINE_STAGE_FLAGS> stages)
{
	struct PerStageAccesses
	{
		public:
			constexpr PerStageAccesses()
			{
        init(PIPELINE_STAGE_FLAGS::HOST_BIT,ACCESS_FLAGS::HOST_READ_BIT|ACCESS_FLAGS::HOST_WRITE_BIT);

        constexpr auto TransferRW = ACCESS_FLAGS::TRANSFER_READ_BIT|ACCESS_FLAGS::TRANSFER_WRITE_BIT;
        init(PIPELINE_STAGE_FLAGS::COPY_BIT,TransferRW);
        init(PIPELINE_STAGE_FLAGS::CLEAR_BIT,ACCESS_FLAGS::TRANSFER_WRITE_BIT);

        constexpr auto MicromapRead = ACCESS_FLAGS::SHADER_READ_BITS;//|ACCESS_FLAGS::MICROMAP_READ_BIT;
//                init(PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT,MicromapRead|ACCESS_FLAGS::MICROMAP_WRITE_BIT); // can micromaps be built indirectly?
        
        constexpr auto AccelerationStructureRW = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT|ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
        init(PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT,TransferRW|AccelerationStructureRW);
        init(PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT|MicromapRead|AccelerationStructureRW);

        init(PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT,ACCESS_FLAGS::COMMAND_PREPROCESS_READ_BIT|ACCESS_FLAGS::COMMAND_PREPROCESS_WRITE_BIT);
        init(PIPELINE_STAGE_FLAGS::CONDITIONAL_RENDERING_BIT,ACCESS_FLAGS::CONDITIONAL_RENDERING_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT,ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT);

        constexpr auto ShaderRW = ACCESS_FLAGS::SHADER_READ_BITS|ACCESS_FLAGS::SHADER_WRITE_BITS;
        constexpr auto AllShaderStagesRW = ShaderRW^(ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT|ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,AllShaderStagesRW);
        init(PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT,ACCESS_FLAGS::INDEX_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT,ACCESS_FLAGS::VERTEX_ATTRIBUTE_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT,AllShaderStagesRW);
        init(PIPELINE_STAGE_FLAGS::TESSELLATION_CONTROL_SHADER_BIT,AllShaderStagesRW);
        init(PIPELINE_STAGE_FLAGS::TESSELLATION_EVALUATION_SHADER_BIT,AllShaderStagesRW);
        init(PIPELINE_STAGE_FLAGS::GEOMETRY_SHADER_BIT,AllShaderStagesRW);
//                init(PIPELINE_STAGE_FLAGS::TASK_SHADER_BIT,AllShaderStagesRW);
//                init(PIPELINE_STAGE_FLAGS::MESH_SHADER_BIT,AllShaderStagesRW);
        init(PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT,ACCESS_FLAGS::FRAGMENT_DENSITY_MAP_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT,ACCESS_FLAGS::SHADING_RATE_ATTACHMENT_READ_BIT);
        constexpr auto DepthStencilRW = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT|ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        init(PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,DepthStencilRW);
        init(PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,AllShaderStagesRW|ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT);
        init(PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,DepthStencilRW);
        init(PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,ACCESS_FLAGS::COLOR_ATTACHMENT_READ_BIT|ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT);

        init(PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,AllShaderStagesRW|ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT);

        init(PIPELINE_STAGE_FLAGS::RESOLVE_BIT,TransferRW);
        init(PIPELINE_STAGE_FLAGS::BLIT_BIT,TransferRW);

//                init(PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT,ACCESS_FLAGS::VIDEO_DECODE_READ_BIT|ACCESS_FLAGS::VIDEO_DECODE_WRITE_BIT);
//                init(PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT,ACCESS_FLAGS::VIDEO_ENCODE_READ_BIT|ACCESS_FLAGS::VIDEO_ENCODE_WRITE_BIT);
//                init(PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT,ACCESS_FLAGS::OPTICAL_FLOW_READ_BIT|ACCESS_FLAGS::OPTICAL_FLOW_WRITE_BIT);
			}
			constexpr const auto& operator[](const size_t ix) const {return data[ix];}

		private:
				
			constexpr void init(PIPELINE_STAGE_FLAGS stageFlag, ACCESS_FLAGS accessFlags)
			{
				const auto bitIx = findLSB(static_cast<size_t>(stageFlag));
				data[bitIx] = accessFlags;
			}

			ACCESS_FLAGS data[32] = {};
	};
	constexpr PerStageAccesses bitToAccess = {};

	// TODO: add logically later or previous stages to make sure all other accesses remain valid
	// or ideally expand the stages before calling `allAccessesFromStages` (TODO: add a `allLaterStages` and `allPreviouStages` basically)

	core::bitflag<ACCESS_FLAGS> retval = ACCESS_FLAGS::NONE;
	while (bool(stages.value))
	{
		const auto bitIx = findLSB(static_cast<size_t>(stages.value));
		retval |= bitToAccess[bitIx];
		stages ^= static_cast<PIPELINE_STAGE_FLAGS>(0x1u<<bitIx);
	}

	return retval;
}

core::bitflag<PIPELINE_STAGE_FLAGS> allStagesFromAccesses(core::bitflag<ACCESS_FLAGS> accesses)
{
	struct PerAccessStages
	{
		public:
			constexpr PerAccessStages()
			{
        init(ACCESS_FLAGS::HOST_READ_BIT,PIPELINE_STAGE_FLAGS::HOST_BIT);
        init(ACCESS_FLAGS::HOST_WRITE_BIT,PIPELINE_STAGE_FLAGS::HOST_BIT);

        init(ACCESS_FLAGS::TRANSFER_READ_BIT,PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS^PIPELINE_STAGE_FLAGS::CLEAR_BIT);
        init(ACCESS_FLAGS::TRANSFER_WRITE_BIT,PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);

        constexpr auto MicromapAccelerationStructureBuilds = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;//|PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT;
//                init(ACCESS_FLAGS::MICROMAP_READ_BIT,MicromapAccelerationStructureBuilds);
//                init(ACCESS_FLAGS::MICROMAP_WRITE_BIT,PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT);
        
        constexpr auto AllShaders = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT|PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS|PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT|PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
        constexpr auto AccelerationStructureOperations = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT|PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
        init(ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT,AccelerationStructureOperations|AllShaders);
        init(ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT,AccelerationStructureOperations);

        init(ACCESS_FLAGS::COMMAND_PREPROCESS_READ_BIT,PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT);
        init(ACCESS_FLAGS::COMMAND_PREPROCESS_WRITE_BIT,PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT);
        init(ACCESS_FLAGS::CONDITIONAL_RENDERING_READ_BIT,PIPELINE_STAGE_FLAGS::CONDITIONAL_RENDERING_BIT);
        init(ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT,PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT|PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT);

        init(ACCESS_FLAGS::UNIFORM_READ_BIT,AllShaders);
        init(ACCESS_FLAGS::SAMPLED_READ_BIT,AllShaders);//|PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT);
        init(ACCESS_FLAGS::STORAGE_READ_BIT,AllShaders|MicromapAccelerationStructureBuilds);
        init(ACCESS_FLAGS::STORAGE_WRITE_BIT,AllShaders);

        init(ACCESS_FLAGS::INDEX_READ_BIT,PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT);
        init(ACCESS_FLAGS::VERTEX_ATTRIBUTE_READ_BIT,PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT);

        init(ACCESS_FLAGS::FRAGMENT_DENSITY_MAP_READ_BIT,PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT);
        init(ACCESS_FLAGS::SHADING_RATE_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT);
        constexpr auto FragmentTests = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT;
        init(ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT,FragmentTests);
        init(ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,FragmentTests);
        init(ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT);
        init(ACCESS_FLAGS::COLOR_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT);
        init(ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT);

        init(ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT,PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT);

//                init(ACCESS_FLAGS::VIDEO_DECODE_READ_BIT,PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_DECODE_WRITE_BIT,PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_ENCODE_READ_BIT,PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_ENCODE_WRITE_BIT,PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT);
//                init(ACCESS_FLAGS::OPTICAL_FLOW_READ_BIT,PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT);
//                init(ACCESS_FLAGS::OPTICAL_FLOW_WRITE_BIT,PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT);
			}
			constexpr const auto& operator[](const size_t ix) const {return data[ix];}

		private:
			constexpr void init(ACCESS_FLAGS accessFlags, PIPELINE_STAGE_FLAGS stageFlags)
			{
				const auto bitIx = findLSB(static_cast<size_t>(accessFlags));
				data[bitIx] = stageFlags;
			}

			PIPELINE_STAGE_FLAGS data[32] = {};
	};
	constexpr PerAccessStages bitToStage = {};

	core::bitflag<PIPELINE_STAGE_FLAGS> retval = PIPELINE_STAGE_FLAGS::NONE;
	while (bool(accesses.value))
	{
		const auto bitIx = findLSB(static_cast<size_t>(accesses.value));
		retval |= bitToStage[bitIx];
		accesses ^= static_cast<ACCESS_FLAGS>(0x1u<<bitIx);
	}

	return retval;
}
}

