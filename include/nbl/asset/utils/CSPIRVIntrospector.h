// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_
#define _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_


#include "nbl/core/declarations.h"

#include <cstdint>
#include <memory>

#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
//#include "nbl/asset/utils/ShaderRes.h"
#include "nbl/asset/utils/CGLSLCompiler.h"

#include "nbl/core/definitions.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
	class Resource;
    struct SPIRType;
}


// wszystkie struktury w CIntrospecionData powininny u¿ywaæ bit flagi, ozaczaj¹cej shader stage (core::unordered_map)
// CStageIntrospecionData nie powinien u¿ywaæ bit flagi, ozaczaj¹cej shader stage (core::vector)

// hashowane s¹ tylko set i binding
// dla spec constant tylko specConstantID
// validacja kolizji (dla SSpecConstants mo¿e siê jedynie ró¿niæ name)
// ogarn¹æ sytuacje gdy jeden descriptor binding ma wiêcej arrayElementCount ni¿ w SPIR-V
// w `CStageIntrospectionData` powinien byæ trzymana struktura `SIntrospectionParams`

// TODO: put this somewhere useful in a separate header
namespace nbl::core
{

template<typename T, size_t Extent=std::dynamic_extent>
struct based_span
{
	public:
		constexpr based_span()
		{
			static_assert(sizeof(based_span<T,Extent>)==sizeof(std::span<T,Extent>));
		}

		constexpr explicit based_span(T* basePtr, std::span<T,Extent> span) : m_offset(span.data()-basePtr), m_size(span.size()) {}
		constexpr explicit based_span(size_t offset, size_t size) : m_offset(offset), m_size(size) {}

		inline std::span<T,Extent> operator()(T* newBase) const {return {newBase+m_offset,m_size};}

	private:
		size_t m_offset = ~0ull;
		size_t m_size = 0ull;
};

}

namespace nbl::asset
{

class NBL_API2 CSPIRVIntrospector : public core::Uncopyable
{
	public:
		constexpr static inline uint16_t MaxPushConstantsSize = 256;
		//
		class CIntrospectionData : public core::IReferenceCounted
		{
			public:
				struct SArrayInfo
				{
					union
					{
						uint32_t value = 0;
						struct
						{
							uint32_t specID : 31;
							uint32_t isSpecConstant : 1;
						};
					};

					// illegal for push constant block members
					inline bool isRuntimeSized() const { return value == 0; }
				};
				struct SDescriptorInfo
				{
					inline bool operator<(const SDescriptorInfo& _rhs) const
					{
						return binding<_rhs.binding;
					}

					uint32_t binding = ~0u;
					std::span<SArrayInfo> count = {};
					IDescriptor::E_TYPE type = IDescriptor::E_TYPE::ET_COUNT;
				};
		};
		// Forward declare for friendship
		class CPipelineIntrospectionData;
		class CStageIntrospectionData : public CIntrospectionData
		{
			public:
				//! General
				enum class VAR_TYPE : uint8_t
				{
					UNKNOWN_OR_STRUCT,
					U64,
					I64,
					U32,
					I32,
					U16,
					I16,
					U8,
					I8,
					F64,
					F32,
					F16
				};
				struct SInterface
				{
					uint32_t location;
					uint32_t elements; // of array
					VAR_TYPE basetype;

					inline bool operator<(const SInterface& _rhs) const
					{
						return location < _rhs.location;
					}
				};
				struct SInputInterface final : SInterface {};
				struct SOutputInterface : SInterface {};
				struct SFragmentOutputInterface final : SOutputInterface
				{
					//! for dual source blending
					uint8_t colorIndex;
				};
				struct STypeInfo
				{
					inline bool isScalar() const {return lastRow==0 && lastCol==0;}
					inline bool isVector() const {return lastRow>0 && lastCol==0;}
					inline bool isMatrix() const {return lastRow>0 && stride>0;}

					uint16_t lastRow : 2 = 0;
					uint16_t lastCol : 2 = 0;
					//! rowMajor=false implies col-major
					uint16_t rowMajor : 1 = 0;
					//! stride==0 implies not matrix
					uint16_t stride : 11 = 0;
					VAR_TYPE type = VAR_TYPE::UNKNOWN_OR_STRUCT;
					uint8_t restrict_ : 1 = false;
					uint8_t volatile_ : 1 = false;
					uint8_t coherent : 1 = false;
					uint8_t readonly : 1 = false;
					uint8_t writeonly : 1 = false;
				};
				//
				template<typename T, bool Mutable>
				using span_t = std::conditional_t<Mutable,core::based_span<T>,std::span<const T>>;
				//
				template<bool Mutable=false>
				struct SMemoryBlock
				{
					struct SMember
					{
						// TODO [Przemek]: doesn't work when `Mutable = true`, fix
						/*inline std::enable_if_t<!Mutable,std::string_view> getName() const
						{
							return std::string_view(name.data(),name.size());
						}*/

						//! children
						span_t<char, Mutable> name = {};
						span_t<SMember,Mutable> members = {};
						//! self
						
						uint32_t offset = 0;
						span_t<SArrayInfo,Mutable> count = {};
						// only relevant if `count.isArray()`
						uint32_t stride = 0;
						// This is the size of the entire member, so for an array it includes everything
						uint32_t size = 0;
						STypeInfo typeInfo = {};
					};

					span_t<char, Mutable> name = {};
					decltype(SMember::members) members;
				};
				//! Maybe one day in the future they'll use memory blocks, but not now
				template<bool Mutable=false>
				struct SSpecConstant final
				{
					span_t<char,Mutable> name = {};
					union {
						uint64_t u64;
						int64_t i64;
						uint32_t u32;
						int32_t i32;
						double f64;
						float f32;
					} defaultValue;
					uint32_t id;
					uint32_t byteSize;
					VAR_TYPE type;
				};

				//! Push Constants
				template<bool Mutable=false>
				struct SPushConstantInfo : SMemoryBlock<Mutable>
				{
					span_t<char,Mutable> name = {};
					// believe it or not you can declare an empty PC block
					bool present = false;
				};

				//! Descriptors
				template<bool Mutable=false>
				struct SDescriptorVarInfo final : SDescriptorInfo
				{
					struct SCombinedImageSampler final
					{
						IImageView<ICPUImage>::E_TYPE viewType : 3;
						uint8_t multisample : 1;
						uint8_t shadow : 1;
					};
					struct SStorageImage final
					{
						// `EF_UNKNOWN` means that Shader will use the StoreWithoutFormat or LoadWithoutFormat capability
						E_FORMAT format = EF_UNKNOWN;
						IImageView<ICPUImage>::E_TYPE viewType : 3;
						uint8_t shadow : 1;
					};
					struct SUniformTexelBuffer final
					{
					};
					struct SStorageTexelBuffer final
					{
					};
					struct SUniformBuffer final : SMemoryBlock<true>
					{
						size_t size = 0;
					};
					struct SStorageBuffer final : SMemoryBlock<true>
					{
						template<bool C=!Mutable>
						inline std::enable_if_t<C,bool> isLastMemberRuntimeSized() const
						{
							if (this->members.empty())
								return false;
							return this->members.back().count.isRuntimeSized;
						}
						template<bool C=!Mutable>
						inline std::enable_if_t<C,size_t> getRuntimeSize(size_t lastMemberElementCount) const
						{
							if (isLastMemberRuntimeSized)
							{
								const auto& lastMember = this->members.back();
								assert(!lastMember.count.isSpecConstantID);
								return sizeWithoutLastMember+lastMemberElementCount*lastMember.stride;
							}
							return sizeWithoutLastMember;
						}

						//! Use `getRuntimeSize` for size of the struct with assumption of passed number of elements.
						//! Need special handling if last member is rutime-sized array (e.g. buffer SSBO `buffer { float buf[]; }`)
						size_t sizeWithoutLastMember;
					};
					struct SInputAttachment final
					{
						uint32_t index;
					};

					//! Note: for SSBOs and UBOs it's the block name
					span_t<char,Mutable> name = {};
					//
					union
					{
						SCombinedImageSampler combinedImageSampler;
						SStorageImage storageImage;
						SUniformTexelBuffer uniformTexelBuffer;
						SStorageTexelBuffer storageTexelBuffer;
						SStorageBuffer uniformBuffer;
						SStorageBuffer storageBuffer;
						SInputAttachment inputAttachment;
						// TODO: acceleration structure?
					};
				};

				//! For the Factory Creation
				struct SParams
				{
					std::string entryPoint;
					core::smart_refctd_ptr<const ICPUShader> shader;

					bool operator==(const SParams& rhs) const
					{
						if (entryPoint != rhs.entryPoint)
							return false;
						if (!rhs.shader)
							return false;
						if (shader->getStage() != rhs.shader->getStage())
							return false;
						if (shader->getContentType() != rhs.shader->getContentType())
							return false;
						if (shader->getContent()->getSize() != rhs.shader->getContent()->getSize())
							return false;
						return memcmp(shader->getContent()->getPointer(), rhs.shader->getContent()->getPointer(), shader->getContent()->getSize()) == 0;
					}
				};
				inline const auto& getParams() const {return m_params;}

				//! TODO: Add getters for all the other members!
				inline const auto& getDescriptorSetInfo(const uint8_t set) const {return m_descriptorSetBindings[set];}

				/*inline bool canSpecializationlesslyCreateDescSetFrom() const
				{
					for (const auto& descSet : m_descriptorSetBindings)
					{
						auto found = std::find_if(descSet.begin(),descSet.end(),[](const SDescriptorVarInfo<>& bnd)->bool{ return bnd.count.isSpecConstant();});
						if (found!=descSet.end())
							return false;
					}
					return true;
				}*/

				// all members are set-up outside the ctor
				inline CStageIntrospectionData() {}

				// We don't need to do anything, all the data was allocated from vector pools
				inline ~CStageIntrospectionData() {}

			protected:
				friend CSPIRVIntrospector;

				using creation_member_t = SMemoryBlock<true>::SMember;
				template<class Pre>
				inline void visitMemoryBlockPreOrderDFS(SMemoryBlock<true>& block, Pre& pre)
				{
					std::stack<creation_member_t*> s;
					auto pushAllMembers = [](const auto& parent)->void
					{
						for (const auto& m : parent.members(m_memberPool.data()))
							s.push(&m);
					};
					pushAllMembers(block);
					while (!s.empty())
					{
						const auto& m = s.top();
						pre(*m);
						s.pop();
						pushAllMembers(*m);
					}
				}
				//template<class Pre>
				//inline void visitMemoryBlockPreOrderBFS(SMemoryBlock<true>& block, Pre& pre)
				//{
				//	std::queue<creation_member_t*> q;
				//	// TODO: pushAllMembers
				//	while (!s.empty())
				//	{
				//		const auto& m = q.front();
				//		pre(*m);
				//		q.pop();
				//		pushAllMembers(*m);
				//	}
				//}

				SDescriptorVarInfo<>& addResource_common(const spirv_cross::Compiler& comp, const spirv_cross::Resource& r, IDescriptor::E_TYPE restype);
				
				// Parameters it was created with
				SParams m_params;
				//! Sorted by `id`
				core::vector<SSpecConstant<>> m_specConstants; // TODO: maybe unordered_set?
				//! Sorted by `location`
				core::vector<SInputInterface> m_input;
				std::variant<
					core::vector<SFragmentOutputInterface>, // when `params.shader->getStage()==ESS_FRAGMENT`
					core::vector<SOutputInterface> // otherwise
				> m_output;
				//!
				SPushConstantInfo<> m_pushConstants;
				//! Each vector is sorted by `binding`
				core::vector<SDescriptorVarInfo<>> m_descriptorSetBindings[4];
				// The idea is that we construct with based_span (so we can add `.data()` when accessing)
				// then just convert from `SMemoryBlock<core::based_span>` to `SMemoryBlock<std::span>`
				// in-place when filling in the `descriptorSetBinding` vector which we pass to ctor
				core::vector<char> m_stringPool;
				core::vector<SArrayInfo> m_arraySizePool;
				core::vector<SMemoryBlock<>> m_memberPool;
		};
		class CPipelineIntrospectionData final : public CIntrospectionData
		{
			public:
				struct SDescriptorInfo final : CIntrospectionData::SDescriptorInfo
				{
					// Which shader stages touch it
					core::bitflag<ICPUShader::E_SHADER_STAGE> stageMask = ICPUShader::ESS_UNKNOWN;
				};
				//
				inline CPipelineIntrospectionData()
				{
					std::fill(m_pushConstantBytes.begin(),m_pushConstantBytes.end(),ICPUShader::ESS_UNKNOWN);
				}

				// returns true if successfully added all the info to self, false if incompatible with what's already in our pipeline or incomplete (e.g. missing spec constants)
				NBL_API2 bool merge(const CStageIntrospectionData* stageData, const ICPUShader::SSpecInfoBase::spec_constant_map_t* specConstants=nullptr);

				//
				NBL_API2 core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection();
				NBL_API2 core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(const uint32_t setID);
				NBL_API2 core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection();

			protected:
				// ESS_UNKNOWN on a byte means its not declared in any shader merged so far
				std::array<core::bitflag<ICPUShader::E_SHADER_STAGE>,MaxPushConstantsSize> m_pushConstantBytes;
				//
				struct Hash
				{
					inline size_t operator()(const SDescriptorInfo& item) const {return item.binding;}
				};
				struct KeyEqual
				{
					inline bool operator()(const SDescriptorInfo& lhs, const SDescriptorInfo& rhs) const
					{
						return lhs.binding==rhs.binding;
					}
				};
				core::unordered_set<SDescriptorInfo,Hash,KeyEqual> m_descriptorSetBindings[4];
		};

		// 
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::debugInfoFlags enabling EDIF_SOURCE_BIT implicitly or explicitly, with no `spirvOptimizer` used in order to include names in introspection data
		inline core::smart_refctd_ptr<const CStageIntrospectionData> introspect(const CStageIntrospectionData::SParams& params, bool insertToCache=true)
		{
			if (!params.shader)
				return nullptr;
    
			if (params.shader->getContentType()!=IShader::E_CONTENT_TYPE::ECT_SPIRV)
				return nullptr;

			// TODO: templated find!
			//auto introspectionData = m_introspectionCache.find(params);
			//if (introspectionData != m_introspectionCache.end())
			//	return *introspectionData;

			auto introspection = doIntrospection(params);

			//if (insertToCache)
			//	m_introspectionCache.insert(introspectionData,introspection);

			return introspection;
		}
		
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info, core::smart_refctd_ptr<ICPUPipelineLayout>&& layout=nullptr)
		{
			if (info.shader->getStage()!=IShader::ESS_COMPUTE)
				return nullptr;

			// TODO: 
			// 1. find or perform introspection using `info`
			// 2. if `layout` then just check for compatiblity
			// 3. if `!layout` then create `CPipelineIntrospectionData` from the stage introspection and create a Layout

			ICPUComputePipeline::SCreationParams params = {{.layout = layout.get()}};
			params.shader = info;
			return ICPUComputePipeline::create(params);
		}

#if 0 // wait until Renderpass Indep completely gone and Graphics Pipeline is used in a new way
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(const std::span<const ICPUShader::SSpecInfo> _infos);
		struct CShaderStages
		{
			const CStageIntrospectionData* vertex = nullptr;
			const CStageIntrospectionData* fragment = nullptr;
			const CStageIntrospectionData* control = nullptr;
			const CStageIntrospectionData* evaluation = nullptr;
			const CStageIntrospectionData* geometry = nullptr;
		}
		core::smart_refctd_ptr<ICPUGraphicsPipeline> createApproximateGraphicsPipeline(const CShaderStages& shaderStages);
#endif	
	private:
		NBL_API2 core::smart_refctd_ptr<const CStageIntrospectionData> doIntrospection(const CStageIntrospectionData::SParams& params);
		void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, CStageIntrospectionData::SMemoryBlock<true>& _res, uint32_t _blockBaseTypeID, uint32_t _varID/*, const mapId2SpecConst_t& _sortedId2sconst*/) const;
		size_t calcBytesizeforType(spirv_cross::Compiler& comp, const spirv_cross::SPIRType& type) const;

		struct KeyHasher
		{
			using is_transparent = void;

			inline size_t operator()(const CStageIntrospectionData::SParams& params) const
			{
				auto stringViewHasher = std::hash<std::string_view>();

				auto code = std::string_view(reinterpret_cast<const char*>(params.shader->getContent()->getPointer()),params.shader->getContent()->getSize());
				size_t hash = stringViewHasher(code);

				core::hash_combine<std::string_view>(hash, std::string_view(params.entryPoint));
				core::hash_combine<uint32_t>(hash, static_cast<uint32_t>(params.shader->getStage()));

				return hash;
			}
			inline size_t operator()(const core::smart_refctd_ptr<const CStageIntrospectionData>& data) const
			{
				return operator()(data->getParams());
			}
		};
		struct KeyEquals
		{
			using is_transparent = void;

			inline bool operator()(const CStageIntrospectionData::SParams& lhs, const core::smart_refctd_ptr<const CStageIntrospectionData>& rhs) const
			{
				return lhs==rhs->getParams();
			}
			inline bool operator()(const core::smart_refctd_ptr<const CStageIntrospectionData>& lhs, const CStageIntrospectionData::SParams& rhs) const
			{
				return lhs->getParams()==rhs;
			}
			inline bool operator()(const core::smart_refctd_ptr<const CStageIntrospectionData>& lhs, const core::smart_refctd_ptr<const CStageIntrospectionData>& rhs) const
			{
				return operator()(lhs,rhs->getParams());
			}
		};

		//using ParamsToDataMap = core::unordered_set<core::smart_refctd_ptr<const CStageIntrospectionData>,KeyHasher,KeyEquals>;
		//ParamsToDataMap m_introspectionCache;
};

} // nbl::asset
#endif
