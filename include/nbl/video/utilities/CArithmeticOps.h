// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_ARITHMETIC_OPS_H_INCLUDED_
#define _NBL_VIDEO_C_ARITHMETIC_OPS_H_INCLUDED_


#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"

#include "nbl/builtin/hlsl/scan/declarations.hlsl"


namespace nbl::video
{
class CArithmeticOps : public core::IReferenceCounted
{
    public:
		// Only 4 byte wide data types supported due to need to trade the via shared memory,
		// different combinations of data type and operator have different identity elements.
		// `EDT_INT` and `EO_MIN` will have `INT_MAX` as identity, while `EDT_UINT` would have `UINT_MAX`  
		enum E_DATA_TYPE : uint8_t
		{
			EDT_UINT=0u,
			EDT_INT,
			EDT_FLOAT,
			EDT_COUNT
		};
		enum E_OPERATOR : uint8_t
		{
			EO_AND = 0u,
			EO_XOR,
			EO_OR,
			EO_ADD,
			EO_MUL,
			EO_MIN,
			EO_MAX,
			EO_COUNT
		};

		// This struct is only for managing where to store intermediate results of the scans
		struct Parameters : nbl::hlsl::scan::Parameters_t // this struct and its methods are also available in HLSL
		{
			static inline constexpr uint32_t MaxLevels = NBL_BUILTIN_MAX_LEVELS;

			Parameters()
			{
				std::fill_n(lastElement,MaxLevels/2+1,0u);
				std::fill_n(temporaryStorageOffset,MaxLevels/2,0u);
			}
			// build the constant tables for each level given the number of elements to scan and workgroupSize
			Parameters(const uint32_t _elementCount, const uint32_t workgroupSize) : Parameters()
			{
				assert(_elementCount!=0u && "Input element count can't be 0!");
				const auto maxReductionLog2 = hlsl::findMSB(workgroupSize)*(MaxLevels/2u+1u);
				assert(maxReductionLog2>=32u||((_elementCount-1u)>>maxReductionLog2)==0u && "Can't scan this many elements with such small workgroups!");

				lastElement[0u] = _elementCount-1u;
				for (topLevel=0u; lastElement[topLevel]>=workgroupSize;)
					temporaryStorageOffset[topLevel-1u] = lastElement[++topLevel] = std::ceil(lastElement[topLevel] / double(workgroupSize));
				
				std::exclusive_scan(temporaryStorageOffset,temporaryStorageOffset+sizeof(temporaryStorageOffset)/sizeof(uint32_t),temporaryStorageOffset,0u);
			}
            // given already computed tables of lastElement indice	s per level, number of levels, and storage offsets, tell us total auxillary buffer size needed
			inline uint32_t getScratchSize(uint32_t ssboAlignment=256u)
			{
				uint32_t uint_count = 1u; // reduceResult field
				uint_count += MaxLevels; // workgroup enumerator
				uint_count += temporaryStorageOffset[MaxLevels/2u-1u]; // last scratch offset
				uint_count += lastElement[topLevel]+1u; // starting from the last temporary storage offset, we also need slots equal to the top level's elementCount (add 1u because 0u based)
				return core::roundUp<uint32_t>(uint_count*sizeof(uint32_t),ssboAlignment);
			}

			inline uint32_t getWorkgroupEnumeratorSize()
			{
				return MaxLevels * sizeof(uint32_t);
			}
		};
        
        // the default scheduler we provide works as described above in the big documentation block
		struct SchedulerParameters : nbl::hlsl::scan::DefaultSchedulerParameters_t  // this struct and its methods are also available in HLSL
		{
			SchedulerParameters()
			{
				std::fill_n(cumulativeWorkgroupCount,Parameters::MaxLevels,0u);
				std::fill_n(workgroupFinishFlagsOffset, Parameters::MaxLevels, 0u);
				std::fill_n(lastWorkgroupSetCountForLevel,Parameters::MaxLevels,0u);
			}
            
            // given the number of elements and workgroup size, figure out how many atomics we need
            // also account for the fact that we will want to use the same scratch buffer both for the
            // scheduler's atomics and the aux data storage
			SchedulerParameters(Parameters& outScanParams, const uint32_t _elementCount, const uint32_t workgroupSize) : SchedulerParameters()
			{
				outScanParams = Parameters(_elementCount,workgroupSize);
				const auto topLevel = outScanParams.topLevel;

				std::copy_n(outScanParams.lastElement+1u,topLevel,cumulativeWorkgroupCount);
				std::reverse_copy(cumulativeWorkgroupCount,cumulativeWorkgroupCount+topLevel,cumulativeWorkgroupCount+topLevel+1u);
				cumulativeWorkgroupCount[topLevel] = 1u; // the top level will always end up with 1 workgroup to do the final reduction
				for (auto i = 0u; i <= topLevel; i++) {
					workgroupFinishFlagsOffset[i] = ((cumulativeWorkgroupCount[i] - 1u) >> hlsl::findMSB(workgroupSize)) + 1; // RECHECK: findMSB(511) == 8u !! Here we assume it's 9u !!
					lastWorkgroupSetCountForLevel[i] = (cumulativeWorkgroupCount[i] - 1u) & (workgroupSize - 1u);
				}

				const auto wgFinishedFlagsSize = std::accumulate(workgroupFinishFlagsOffset, workgroupFinishFlagsOffset + Parameters::MaxLevels, 0u);
				std::exclusive_scan(workgroupFinishFlagsOffset, workgroupFinishFlagsOffset + Parameters::MaxLevels, workgroupFinishFlagsOffset, 0u);
				for (auto i=0u; i<sizeof(Parameters::temporaryStorageOffset)/sizeof(uint32_t); i++)
					outScanParams.temporaryStorageOffset[i] += wgFinishedFlagsSize;
				
				std::inclusive_scan(cumulativeWorkgroupCount, cumulativeWorkgroupCount + Parameters::MaxLevels, cumulativeWorkgroupCount);
			}
		};
        
        // push constants of the default direct scan pipeline provide both aux memory offset params and scheduling params
		struct DefaultPushConstants
		{
			Parameters scanParams;
			SchedulerParameters schedulerParams;
		};
        
		struct DispatchInfo
		{
			DispatchInfo() : wg_count(0u)
			{
			}
            
            // in case we scan very few elements, you don't want to launch workgroups that wont do anything
			DispatchInfo(const IPhysicalDevice::SLimits& limits, const uint32_t elementCount, const uint32_t workgroupSize)
			{
				constexpr auto workgroupSpinningProtection = 4u; // to prevent first workgroup starving/idling on level 1 after finishing level 0 early
				wg_count = limits.computeOptimalPersistentWorkgroupDispatchSize(elementCount,workgroupSize,workgroupSpinningProtection);
				assert(wg_count >= elementCount / workgroupSize^2 + 1 && "Too few workgroups! The workgroup count must be at least the elementCount/(wgSize^2)");
			}

			uint32_t wg_count;
		};

		CArithmeticOps(core::smart_refctd_ptr<ILogicalDevice>&& device) : CArithmeticOps(std::move(device),core::roundDownToPoT(device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)) {}
		
        CArithmeticOps(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t workgroupSize) : m_device(std::move(device)), m_workgroupSize(workgroupSize)
		{
			assert(core::isPoT(m_workgroupSize));
			const asset::SPushConstantRange pc_range[] = {asset::IShader::ESS_COMPUTE,0u,sizeof(DefaultPushConstants)};
			const IGPUDescriptorSetLayout::SBinding bindings[2] = {
				{ 0u, asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_COMPUTE, 1u, nullptr }, // main buffer
				{ 1u, asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_COMPUTE, 1u, nullptr } // scratch
			};

			m_ds_layout = m_device->createDescriptorSetLayout(bindings);
			assert(m_ds_layout && "CArithmeticOps Descriptor Set Layout was not created successfully");
			m_pipeline_layout = m_device->createPipelineLayout({ pc_range }, core::smart_refctd_ptr(m_ds_layout));
			assert(m_pipeline_layout && "CArithmeticOps Pipeline Layout was not created successfully");
		}

		inline auto getDefaultDescriptorSetLayout() const { return m_ds_layout.get(); }

		inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

		inline uint32_t getWorkgroupSize() const {return m_workgroupSize;}

		inline void buildParameters(const uint32_t elementCount, DefaultPushConstants& pushConstants, DispatchInfo& dispatchInfo)
		{
			pushConstants.schedulerParams = SchedulerParameters(pushConstants.scanParams,elementCount,m_workgroupSize);
			dispatchInfo = DispatchInfo(m_device->getPhysicalDevice()->getLimits(),elementCount,m_workgroupSize);
		}

		static inline void updateDescriptorSet(ILogicalDevice* device, IGPUDescriptorSet* set, const asset::SBufferRange<IGPUBuffer>& input_range, const asset::SBufferRange<IGPUBuffer>& scratch_range)
		{
			IGPUDescriptorSet::SDescriptorInfo infos[2];
			infos[0].desc = input_range.buffer;
			infos[0].info.buffer = {input_range.offset,input_range.size};
			infos[1].desc = scratch_range.buffer;
			infos[1].info.buffer = {scratch_range.offset,scratch_range.size};

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[2];
			for (auto i=0u; i<2u; i++)
			{
				writes[i] = { .dstSet = set,.binding = i,.arrayElement = 0u,.count = 1u,.info = infos+i };
			}

			device->updateDescriptorSets(2, writes, 0u, nullptr);
		}

		static inline void dispatchHelper(
			IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* layout, const DefaultPushConstants& pushConstants, const DispatchInfo& dispatchInfo,
			std::span<const IGPUCommandBuffer::SPipelineBarrierDependencyInfo> bufferBarriers
		)
		{
			cmdbuf->pushConstants(layout, asset::IShader::ESS_COMPUTE, 0u, sizeof(DefaultPushConstants), &pushConstants);
			if (bufferBarriers.size() > 0)
			{
				for (uint32_t i = 0; i < bufferBarriers.size(); i++)
				{
					cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, bufferBarriers[i]);
				}
			}
			cmdbuf->dispatch(dispatchInfo.wg_count, 1u, 1u);
		}

        inline core::smart_refctd_ptr<asset::ICPUShader> createBaseShader(const char* shaderFile, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchElCount) const
        {
            auto system = m_device->getPhysicalDevice()->getSystem();
            core::smart_refctd_ptr<const system::IFile> hlsl;
            {
                auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
                {
                    nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
                    system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
                    if (future.wait())
                        return future.copy();
                    return nullptr;
                };

                // hlsl = loadBuiltinData("nbl/builtin/hlsl/scan/indirect_reduce.hlsl"); ??
                hlsl = loadBuiltinData(shaderFile);
            }
            
            auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(hlsl->getSize());
            memcpy(buffer->getPointer(), hlsl->getMappedPointer(), hlsl->getSize());
            auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_HLSL, "????");

            // REVIEW: Probably combine the commons parts of CReduce and CScanner
            const char* storageType = nullptr;
            switch (dataType)
            {
                case EDT_UINT:
                    storageType = "uint32_t";
                    break;
                case EDT_INT:
                    storageType = "int32_t";
                    break;
                case EDT_FLOAT:
                    storageType = "float32_t";
                    break;
                default:
                    assert(false);
                    break;
            }

            const char* binop = nullptr;
            switch (op)
            {
                case EO_AND:
                    binop = "bit_and";
                    break;
                case EO_OR:
                    binop = "bit_or";
                    break;
                case EO_XOR:
                    binop = "bit_xor";
                    break;
                case EO_ADD:
                    binop = "plus";
                    break;
                case EO_MUL:
                    binop = "multiplies";
                    break;
                case EO_MAX:
                    binop = "maximum";
                    break;
                case EO_MIN:
                    binop = "minimum";
                    break;
                default:
                    assert(false);
                    break;
            }
            
			return asset::CHLSLCompiler::createOverridenCopy(cpushader.get(), "#define WORKGROUP_SIZE %d\ntypedef %s Storage_t;\n#define BINOP nbl::hlsl::%s\n#define SCRATCH_EL_CNT %d\n", m_workgroupSize, storageType, binop, scratchElCount);
        }

		inline ILogicalDevice* getDevice() const {return m_device.get();}
    protected:
        // REVIEW: Does it need an empty destructor?

		core::smart_refctd_ptr<ILogicalDevice> m_device;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_ds_layout;
		core::smart_refctd_ptr<IGPUPipelineLayout> m_pipeline_layout;
		
		const uint32_t m_workgroupSize;
};

}

#endif