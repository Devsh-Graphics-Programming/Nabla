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
#include "nbl/asset/utils/ShaderRes.h"
#include "nbl/asset/utils/CGLSLCompiler.h"


#include "nbl/core/definitions.h"


namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}

// podzielic CIntrospectionData na dwie klasy
// jedna bez inputOutput i bez push constant blocka `CIntrospectionData`
// druga dziedziczy z pierwszej i dodaje te 2 rzeczy `CStageIntrospectionData`

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
struct based_span : private std::span<T,Extent>
{
	constexpr based_span() : span<T,Extent>() {}

	constexpr explicit based_span(T* basePtr, std::span<T,Extent> span) : span<T,Extent>(span.begin()-basePtr,span.end()-basePtr) {}

	inline std::span<T,Extent> operator()(T* newBase) const {return std::span<T,Extent>(begin()+newBase,end()+newBase);}
};

}

namespace nbl::asset
{

class NBL_API2 CSPIRVIntrospector : public core::Uncopyable
{
	public:
		class NBL_API2 CIntrospectionData : public core::IReferenceCounted
		{
			public:
				struct SDescriptorInfo
				{
					uint32_t binding;
				};
				struct SCombinedImageSampler final : SDescriptorInfo
				{
					IImageView<ICPUImage>::E_TYPE viewType : 3;
					uint8_t multisample : 1;
					uint8_t shadow : 1;
				};
				struct SStorageImage final : SDescriptorInfo
				{
					// `EF_UNKNOWN` means that Shader will use the StoreWithoutFormat or LoadWithoutFormat capability
					E_FORMAT format = EF_UNKNOWN;
					IImageView<ICPUImage>::E_TYPE viewType : 3;
					uint8_t shadow : 1;
				};
				struct SUniformTexelBuffer final : SDescriptorInfo
				{
				};
				struct SStorageTexelBuffer final : SDescriptorInfo
				{
				};
				struct SInputAttachment final : SDescriptorInfo
				{
					uint32_t index;
				};
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
					uint32_t stride = 0;

					inline bool isArray() const {return stride;}
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
					VAR_TYPE type = UNKNOWN_OR_STRUCT;
					uint8_t restrict_ : 1 = false;
					uint8_t volatile_ : 1 = false;
					uint8_t coherent : 1 = false;
					uint8_t readonly : 1 = false;
					uint8_t writeonly : 1 = false;
				};
				//
				template<template<typename> class span_t>
				constexpr static inline bool IsSpan = std::is_same_v<span_t<void>,std::span<void>>;
				//
				template<template<typename> class span_t>
				struct SMemoryBlock
				{
					struct SMember
					{
						inline std::enable_if_t<IsSpan<span_t>,std::string_view> getName() const
						{
							return std::string_view(name.data(),name.size());
						}

						span_t<SMember> members;
						// self
						span_t<char> name;
						SArrayInfo count = {};
						uint32_t offset = 0;
						// This is the size of the entire member, so for an array it includes everything
						uint32_t size = 0;
						STypeInfo typeInfo = {};
					};

					decltype(SMember::members) members;
				};
				//
				template<template<typename> class span_t>
				struct SUniformBuffer final : SDescriptorInfo, SMemoryBlock<span_t>
				{
					size_t size = 0;
				};
				template<template<typename> class span_t>
				struct SStorageBuffer final : SDescriptorInfo, SMemoryBlock<span_t>
				{
					inline std::enable_if_t<IsSpan<span_t>,bool> isLastMemberRuntimeSized() const
					{
						if (members.empty())
							return false;
						return members.back().count.value==0;
					}
					inline std::enable_if_t<IsSpan<span_t>,size_t> getRuntimeSize(size_t lastMemberElementCount) const
					{
						if (isLastMemberRuntimeSized)
						{
							const auto& lastMember = members.back();
							assert(members.back().count.isArray());
							return sizeWithoutLastMember+lastMemberElementCount*lastMember.count.stride;
						}
						return sizeWithoutLastMember;
					}

					//! Use `getRuntimeSize` for size of the struct with assumption of passed number of elements.
					//! Need special handling if last member is rutime-sized array (e.g. buffer SSBO `buffer { float buf[]; }`)
					size_t sizeWithoutLastMember;
				};
				// DO NOT CHANGE THE ORDER! Or you'll mess up `getDescriptorType(const SDescriptorVariant&)`
				using SDescriptorVariant = std::variant<
					SCombinedImageSampler,
					SStorageImage,
					SUniformTexelBuffer,
					SStorageTexelBuffer,
					SInputAttachment,
					SUniformBuffer<std::span>,
					SStorageBuffer<std::span>
				>;
				static inline IDescriptor::E_TYPE getDescriptorType(const SDescriptorVariant& v)
				{
					return static_cast<IDescriptor::E_TYPE>(v.index());
				}

				inline const auto& getDescriptorSetInfo(const uint8_t set) const {return m_descriptorSetBindings[set];}


				//! Push constants uniform block
				struct {
					bool present;
					core::string name;
					SShaderPushConstant info;
				} pushConstant;

				bool canSpecializationlesslyCreateDescSetFrom() const
				{
					for (const auto& descSet : m_descriptorSetBindings)
					{
						auto found = std::find_if(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& bnd) { return bnd.descCountIsSpecConstant; });
						if (found != descSet.end())
							return false;
					}
					return true;
				}

			protected:
				using final_member_t = SMemoryBlock<std::span>::SMember;
				inline CIntrospectionData(core::vector<SDescriptorVariant>* _descriptorSetBindings, core::vector<const char>&& _stringPool, core::vector<final_member_t>&& _memberPool) :
					m_stringPool(std::move(_stringPool)), m_memberPool(std::move(_memberPool))
				{
					for (auto i=0; i<4; i++)
						m_descriptorSetBindings[i] = std::move(_descriptorSetBindings[i]);
				}
				// We don't need to do anything, all the data was allocated from vector pools
				inline ~CIntrospectionData() {}

				using creation_member_t = SMemoryBlock<core::based_span>::SMember;
				template<class Pre>
				inline void visitMemoryBlockPreOrderDFS(SMemoryBlock<core::based_span>& block, Pre& pre)
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
				template<class Pre>
				inline void visitMemoryBlockPreOrderBFS(SMemoryBlock<core::based_span>& block, Pre& pre)
				{
					std::queue<creation_member_t*> q;
					// TODO: pushAllMembers
					while (!s.empty())
					{
						const auto& m = q.front();
						pre(*m);
						q.pop();
						pushAllMembers(*m);
					}
				}
				
				//! Each vector is sorted by `binding`
				core::vector<SDescriptorVariant> m_descriptorSetBindings[4];
				// The idea is that we construct with based_span (so we can add `.data()` when accessing)
				// then just convert from `SMemoryBlock<core::based_span>` to `SMemoryBlock<std::span>`
				// in-place when filling in the `descriptorSetBinding` vector which we pass to ctor
				core::vector<const char> m_stringPool;
				core::vector<final_member_t> m_memberPool;
		};
		class CStageIntrospectionData final : public CIntrospectionData
		{
			public:
				struct SParams
				{
					std::string entryPoint;
					core::smart_refctd_ptr<const ICPUShader> cpuShader;

					bool operator==(const SParams& rhs) const
					{
						if (entryPoint != rhs.entryPoint)
							return false;
						if (!rhs.cpuShader)
							return false;
						if (cpuShader->getStage() != rhs.cpuShader->getStage())
							return false;
						if (cpuShader->getContentType() != rhs.cpuShader->getContentType())
							return false;
						if (cpuShader->getContent()->getSize() != rhs.cpuShader->getContent()->getSize())
							return false;
						return memcmp(cpuShader->getContent()->getPointer(), rhs.cpuShader->getContent()->getPointer(), cpuShader->getContent()->getSize()) == 0;
					}
				};
				struct SSpecConstant
				{
					// TODO: change to std::string_view to big pool allocated thing later
					std::string name;
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
				struct SInterface
				{
					uint32_t location;
					uint32_t elements; // of array
					VAR_TYPE basetype;

					inline bool operator<(const SInterface& _rhs) const
					{
						return location<_rhs.location;
					}
				};
				struct SInputInterface final : SInterface {};
				struct SOutputInterface : SInterface {};
				struct SFragmentOutputInterface final : SOutputInterface
				{
					//! for dual source blending
					uint8_t colorIndex;
				};

				// Parameters it was created with
				const SParams params;
				//! Sorted by `id`
				core::vector<SSpecConstant> specConstants;
				//! Sorted by `location`
				core::vector<SInputInterface> input;
				std::variant<
					core::vector<SFragmentOutputInterface>, // when `params.cpuShader->getStage()==ESS_FRAGMENT`
					core::vector<SOutputInterface> // otherwise
				> output;
		};

		// 
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::debugInfoFlags enabling EDIF_SOURCE_BIT implicitly or explicitly, with no `spirvOptimizer` used in order to include names in introspection data
		// powinna zwracac CStageIntrospectionData
		core::smart_refctd_ptr<const CIntrospectionData> introspect(const SIntrospectionParams& params, bool insertToCache = true);

		// 
		//core::smart_refctd_ptr<const CIntrospectionData> merge(const std::span<const CStageIntrospectionData>& asdf, const ICPUShader::SSPecInfo::spec_constant_map_t& = {});

		// When the methods take a span of shaders, they are computing things for an imaginary pipeline that includes **all** of them
		// przeniesc do CIntrospectionData
		std::pair<bool/*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, const std::span<const ICPUShader::SSpecInfo> _infos);

		//
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info)
		//TODO: inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(CStageIntrospectionData* asdf)
		{
			if (info.shader->getStage()!=IShader::ESS_COMPUTE)
				return nullptr;

			core::smart_refctd_ptr<const CIntrospectionData> introspection = nullptr;
			
			//TODO: zamiast tego mergujemy `CStageIntrospectionData` w `CIntrospectionData` u¿ywaj¹c `merge`
			if (!introspectAllShaders(&introspection,{&info,1}))
				return nullptr;

			auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection,{&info,1});
			ICPUComputePipeline::SCreationParams params = {{.layout = layout.get()}};
			params.shader = info;
			return ICPUComputePipeline::create(params);
		}

		//
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
	
	private:
		//TODO: przenieœæ jako members do CIntrospectionData
		core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl();
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(const uint32_t setID);
		core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl();

		core::smart_refctd_ptr<CStageIntrospectionData> introspectShader(const ICPUShader::SSpecInfo _infos);

		core::smart_refctd_ptr<const CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const std::string& entryPoint, const IShader::E_SHADER_STAGE stage) const;
		void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const;
		size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

	private:

		struct KeyHasher
		{
			size_t operator()(const SIntrospectionParams& param) const 
			{
				auto stringViewHasher = std::hash<std::string_view>();

				auto code = std::string_view(reinterpret_cast<const char*>(param.cpuShader->getContent()->getPointer()), param.cpuShader->getContent()->getSize());
				size_t hash = stringViewHasher(code);

				core::hash_combine<std::string_view>(hash, std::string_view(param.entryPoint));
				core::hash_combine<uint32_t>(hash, static_cast<uint32_t>(param.cpuShader->getStage()));

				return hash;
			}
		};

		using ParamsToDataMap = core::unordered_map<SIntrospectionParams,core::smart_refctd_ptr<const CIntrospectionData>, KeyHasher>;
		// using ParamsToDataMap = core::unordered_set<core::smart_refctd_ptr<const CStageIntrospectionData>, KeyHasher, KeyEquals>;
		ParamsToDataMap m_introspectionCache;
};

} // nbl::asset
#endif
