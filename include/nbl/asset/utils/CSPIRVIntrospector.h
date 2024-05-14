// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_
#define _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_

// TODO: 
// - test input / output

#include "nbl/core/declarations.h"

#include <cstdint>
#include <memory>

#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/utils/CGLSLCompiler.h"

#include "nbl/core/definitions.h"
#include "nbl/core/based_offset.h"
#include "nbl/core/based_span.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
	class Resource;
    struct SPIRType;
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
				//! NOTE: Whenever its used in a span, the extents are recorded Least Significant Stride to Most
				//! So `var[Z][Y][X]` is stored as `{X,Y,Z}`
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
					inline bool isRuntimeSized() const {return !isSpecConstant && value==0;}
				};

				struct SDescriptorArrayInfo
				{
					enum class DESCRIPTOR_COUNT : uint32_t
					{
						STATIC = 0,
						SPEC_CONSTANT = 1,
						RUNTIME = 2
					};
					// spec constant ID if `countMode==SPEC_CONSTANT` otherwise a lower bound on binding array size
					union
					{
						uint32_t count : 30 = 0;
						uint32_t specID : 30;
					};

					DESCRIPTOR_COUNT countMode : 2;
				};

				struct SDescriptorInfo
				{
					inline bool operator<(const SDescriptorInfo& _rhs) const
					{
						return binding<_rhs.binding;
					}

					uint32_t binding = ~0u;
					IDescriptor::E_TYPE type = IDescriptor::E_TYPE::ET_COUNT;
				};
		};
		// Forward declare for friendship
		class CPipelineIntrospectionData;
		class NBL_API2 CStageIntrospectionData : public CIntrospectionData
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

				inline bool canSpecializationlesslyCreateDescSetFrom() const
				{
					for (const auto& descSet : m_descriptorSetBindings)
					{
						auto found = std::find_if(descSet.begin(), descSet.end(), [](const SDescriptorVarInfo<false>& bnd)->bool { return bnd.count.countMode == SDescriptorArrayInfo::DESCRIPTOR_COUNT::SPEC_CONSTANT; });
						if (found != descSet.end())
							return false;
					}
					return true;
				}

				struct SInterface
				{
					uint32_t location;
					uint32_t elements; // of array
					VAR_TYPE baseType;

					inline bool operator==(const SInterface& rhs) const
					{
						return std::memcmp(this, &rhs, sizeof(SInterface));
					}
				};
				struct SInputInterface final : SInterface {};
				struct SOutputInterface : SInterface {};
				struct SFragmentOutputInterface final : SOutputInterface
				{
					//! for dual source blending
					uint8_t colorIndex;
				};
				//! TODO: certain things can only be applied on top of members, like:
				//! - stride
				struct STypeInfo
				{
					inline bool isScalar() const {return lastRow==0 && lastCol==0;}
					inline bool isVector() const {return lastRow>0 && lastCol==0;}
					inline bool isMatrix() const {return lastRow>0 && stride>0;}

					uint16_t lastRow : 2 = 0;
					uint16_t lastCol : 2 = 0;
					//! rowMajor=false implies col-major
					uint16_t rowMajor : 1 = true; // have it both in SMemberInfo and here, two different ways to get decorations
					//! stride==0 implies not matrix
					uint16_t stride : 11 = 0; // move to SMemberInfo
					VAR_TYPE type : 6 = VAR_TYPE::UNKNOWN_OR_STRUCT;
					uint8_t restrict_ : 1 = false; // have it both in SMemberInfo and here, two different ways to get decorations
					uint8_t aliased : 1 = false; // have it both in SMemberInfo and here, two different ways to get decorations
				};
				//
				template<typename T, bool Mutable>
				using ptr_t = std::conditional_t<Mutable,core::based_offset<T>,const T*>;
				//
				template<typename T, bool Mutable>
				using span_t = std::conditional_t<Mutable,core::based_span<T>,std::span<const T>>;
				//
				template<bool Mutable=false>
				struct SType
				{
					public:
						inline bool isArray() const {return !count.empty();}

						struct MatrixInfo
						{
							uint8_t rowMajor : 1;
							uint8_t rows : 3;
							uint8_t columns : 3;
						};

						//! children
						//! TODO: replace these 5 SoA arrays with some struct similar to `STypeInfo` that has the per-member decorations, call it `SMemberInfo`
						using member_type_t = ptr_t<SType<Mutable>,Mutable>;
						inline ptr_t<member_type_t,Mutable> memberTypes() const {return reinterpret_cast<const ptr_t<member_type_t,Mutable>&>(memberInfoStorage);}
						using member_name_t = span_t<char,Mutable>;
						inline ptr_t<member_name_t,Mutable> memberNames() const
						{
							if constexpr (Mutable)
								return (memberTypes()+memberCount).byte_offset();
							else
								return reinterpret_cast<const member_name_t*>(memberTypes()+memberCount);
						}

						//! SPIR-V is a bit dumb and types of the same struct can have variable size depending on std140, std430, or scalar layouts
						//! it would have been much easier if the layout was baked into the struct, so there'd be a separate type
						//! (like a template instantation) of each free-floating struct in different layouts or at least like cv-qualifiers baked into a type.
						using member_size_t = uint32_t;
						// This is the size of the entire member, so for an array it includes everything
						inline ptr_t<member_size_t,Mutable> memberSizes() const
						{
							if constexpr (Mutable)
								return (memberNames()+memberCount).byte_offset();
							else
								return reinterpret_cast<const member_size_t*>(memberNames()+memberCount);
						}
						using member_offset_t = member_size_t;
						inline ptr_t<member_offset_t,Mutable> memberOffsets() const {return memberSizes()+memberCount;}
						using member_stride_t = uint32_t;
						// `memberStrides[i]` only relevant if `memberTypes[i]->isArray()`
						inline ptr_t<member_stride_t,Mutable> memberStrides() const {return memberOffsets()+memberCount;}
						using member_matrix_info_t = MatrixInfo;
						inline ptr_t<member_matrix_info_t,Mutable> memberMatrixInfos() const {return reinterpret_cast<ptr_t<member_matrix_info_t,Mutable>&>(memberStrides()+memberCount); }

						constexpr static inline size_t StoragePerMember = sizeof(member_type_t)+sizeof(member_name_t)+sizeof(member_size_t)+sizeof(member_offset_t)+sizeof(member_stride_t)+sizeof(member_matrix_info_t);

						//! self
						span_t<char,Mutable> typeName = {};
						span_t<SArrayInfo,Mutable> count = {};
						STypeInfo info = {};

					//private:
						uint32_t memberCount = 0;
						ptr_t<char,Mutable> memberInfoStorage = {};
				};
				template<bool Mutable=false>
				using type_ptr = SType<Mutable>::member_type_t;
				template<bool Mutable=false>
				struct SMemoryBlock
				{
					type_ptr<Mutable> type = nullptr;
				};
				template<bool Mutable, class Pre>
				inline void visitMemoryBlockPreOrderDFS(SMemoryBlock<Mutable>& block, Pre& pre)
				{
					auto* const basePtr = m_memPool.data();

					std::stack<type_ptr<Mutable>> stk;
					if (block.type)
						stk.push(block.type);
					while (!stk.empty())
					{
						const auto& entry = stk.top();
						std::conditional_t<Mutable,SType<true>,const SType<false>>* type;
						if constexpr (Mutable)
							type = entry(basePtr);
						else
							type = entry;

						stk.pop();

						const type_ptr<Mutable>* members;
						if constexpr(Mutable)
							members = type->memberTypes()(basePtr);
						else
							members = type->memberTypes();
						for (auto i=0u; i<type->memberCount; i++)
							stk.push(members[i]);

						pre(type);
					}
				}
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
					uint8_t offset = 0u;
					uint8_t size = 0u;

					// believe it or not you can declare an empty PC block
					inline bool present() const {return SMemoryBlock<Mutable>::type;}
				};

				//! Descriptors
				template<bool Mutable=false>
				struct SDescriptorVarInfo final : SDescriptorInfo
				{
					struct SRWDescriptor
					{
						uint8_t readonly : 1 = false;
						uint8_t writeonly : 1 = false;
					};
					struct SImage
					{
						IImageView<ICPUImage>::E_TYPE viewType : 3 = IImageView<ICPUImage>::E_TYPE::ET_2D;
						uint8_t shadow : 1 = false;
					};
					struct SCombinedImageSampler final : SImage
					{
						uint8_t multisample : 1;
					};
					struct SStorageImage final : SRWDescriptor, SImage
					{
						// `EF_UNKNOWN` means that Shader will use the StoreWithoutFormat or LoadWithoutFormat capability
						E_FORMAT format = EF_UNKNOWN;
					};
					struct SUniformTexelBuffer final
					{
					};
					struct SStorageTexelBuffer final : SRWDescriptor
					{
					};
					struct SUniformBuffer final : SMemoryBlock<Mutable>
					{
						size_t size = 0;
					};
					struct SStorageBuffer final : SRWDescriptor, SMemoryBlock<Mutable>
					{
						template<bool C=!Mutable>
						inline std::enable_if_t<C,bool> isLastMemberRuntimeSized() const
						{
							if (type->memberCount)
								return type->memberTypes()[type->memberCount-1].count.front().isRuntimeSized();
							return false;
						}
						template<bool C=!Mutable>
						inline std::enable_if_t<C,size_t> getRuntimeSize(const size_t lastMemberElementCount) const
						{
							if (isLastMemberRuntimeSized())
							{
								const auto& lastMember = type->memberTypes()[type->memberCount-1];
								assert(!lastMember.count.front().isSpecConstantID);
								return sizeWithoutLastMember+lastMemberElementCount*type->memberStrides()[type->memberCount-1];
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

						// `DescriptorType descriptor[1];` can be treated as `DescriptorType descriptor;`
					inline bool isArray() const {return count.countMode != SDescriptorArrayInfo::DESCRIPTOR_COUNT::STATIC || count.count > 1u;}
					inline bool isRunTimeSized() const {return count.countMode == SDescriptorArrayInfo::DESCRIPTOR_COUNT::RUNTIME;}

					//! Note: for SSBOs and UBOs it's the block name
					span_t<char,Mutable> name = {};
					SDescriptorArrayInfo count;
					uint8_t restrict_ : 1 = false;
					uint8_t aliased : 1 = false;
					
					//
					union
					{
						SCombinedImageSampler combinedImageSampler;
						SStorageImage storageImage;
						SUniformTexelBuffer uniformTexelBuffer;
						SStorageTexelBuffer storageTexelBuffer;
						SUniformBuffer uniformBuffer;
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
				inline const auto& getDescriptorSetInfo(const uint8_t set) const {return m_descriptorSetBindings[set];}
				inline const auto& getInputs() const { return m_input; }
				inline const std::span<const SFragmentOutputInterface> getFragmentShaderOutputs() const
				{
					if (m_shaderStage != IShader::ESS_FRAGMENT)
						return {};

					const auto& outputInterface = std::get<core::vector<SFragmentOutputInterface>>(m_output);
					return { outputInterface.data(), outputInterface.size() };
				}
				inline const std::span<const SOutputInterface> getShaderOutputs() const
				{
					if (m_shaderStage == IShader::ESS_UNKNOWN || m_shaderStage == IShader::ESS_FRAGMENT)
						return {};

					const auto& outputInterface = std::get<core::vector<SOutputInterface>>(m_output);
					return { outputInterface.data(), outputInterface.size() };
				}
				inline const auto& getPushConstants() const {return m_pushConstants;}
				inline const auto& getSpecConstants() const {return m_specConstants;}

				void debugPrint(system::ILogger* logger) const;

				// all members are set-up outside the ctor
				inline CStageIntrospectionData() {}

				// We don't need to do anything, all the data was allocated from vector pools
				inline ~CStageIntrospectionData() {}

			protected:
				friend CSPIRVIntrospector;

				//! Only call these during construction!
				inline size_t allocOffset(const size_t bytes) // TODO: move to cpp
				{
					const size_t off = m_memPool.size();
					m_memPool.resize(off+bytes);
					return off;
				}
				template<typename T>
				inline core::based_span<T> alloc(const size_t count)
				{
					const auto off = allocOffset(sizeof(T)*count);
					return {off,count};
				}
				inline core::based_span<SArrayInfo> addCounts(const size_t count, const uint32_t* sizes, const bool* size_is_literal) // TODO: move to cpp
				{
					if (count == 0)
						return {};

					const auto range = alloc<SArrayInfo>(count);
					auto arraySizes = range(m_memPool.data());
					for (size_t i=0; i<count; i++)
					{
						// the API for this spec constant checking is truly messed up
						if (size_is_literal[i])
							arraySizes[i].value = sizes[i];
						else
						{
							arraySizes[i].specID = sizes[i]; // ID of spec constant if size is spec constant
							arraySizes[i].isSpecConstant = true;
						}
					}

					return range;
				}
				inline SDescriptorArrayInfo addDescriptorCount(const uint32_t size, const bool sizeIsLiteral)
				{
					SDescriptorArrayInfo descriptorArraySize;

					// the API for this spec constant checking is truly messed up
					if (sizeIsLiteral)
					{
						descriptorArraySize.count = size;
						descriptorArraySize.countMode = size == 0u ? SDescriptorArrayInfo::DESCRIPTOR_COUNT::RUNTIME : SDescriptorArrayInfo::DESCRIPTOR_COUNT::STATIC;
					}
					else
					{
						descriptorArraySize.specID = size; // ID of spec constant if size is spec constant
						descriptorArraySize.countMode = SDescriptorArrayInfo::DESCRIPTOR_COUNT::SPEC_CONSTANT;
					}

					return descriptorArraySize;
				}
				inline core::based_span<char> addString(const std::string_view str)
				{
					const auto range = alloc<char>(str.size()+1);
					const auto out = range(m_memPool.data());
					memcpy(out.data(),str.data(),str.size());
					out.back() = '\0';
					return range;
				}
				SDescriptorVarInfo<true>* addResource(const spirv_cross::Compiler& comp, const spirv_cross::Resource& r, IDescriptor::E_TYPE restype);
				inline core::based_offset<SType<true>> addType(const size_t memberCount)
				{
					const auto memberStorage = allocOffset(SType<true>::StoragePerMember*memberCount);
					auto retval = alloc<SType<true>>(1);
					// allocated everything before touching any data stores, to avoid pointer invalidation
					auto pType = retval(m_memPool.data()).data();
					pType->memberCount = memberCount;
					pType->memberInfoStorage = {memberStorage};
					return {retval.byte_offset()};
				}
				void shaderMemBlockIntrospection(const spirv_cross::Compiler& comp, SMemoryBlock<true>* root, const spirv_cross::Resource& r);
				void finalize(IShader::E_SHADER_STAGE stage);

				//! debug
				static void printExtents(std::ostringstream& out, const SArrayInfo& count);
				static void printType(std::ostringstream& out, const SType<false>* counts, const uint32_t depth=0);
				
				IShader::E_SHADER_STAGE m_shaderStage = IShader::E_SHADER_STAGE::ESS_UNKNOWN;

				// Parameters it was created with
				SParams m_params;
				struct SpecConstantKeyEqual
				{
					inline bool operator()(const CStageIntrospectionData::SSpecConstant<>& lhs, const CStageIntrospectionData::SSpecConstant<>& rhs) const
					{ 
						return lhs.id == rhs.id;
					}
				};
				struct SpecConstantHash
				{
					inline uint32_t operator()(const CStageIntrospectionData::SSpecConstant<>&pc) const { return std::hash<size_t>()(pc.id); }
				};
				using SpecConstantsSet = core::unordered_set<SSpecConstant<>, SpecConstantHash, SpecConstantKeyEqual>;
				SpecConstantsSet m_specConstants;

				struct InputInterfaceHash
				{
					inline uint32_t operator()(const CStageIntrospectionData::SInputInterface& input) const
					{
						// TODO: combine hashes of ints maybe?
						std::stringstream ss;
						ss << input.location << input.elements << static_cast<uint32_t>(input.baseType);

						return std::hash<std::string_view>()(std::string_view(ss.str()));
					}
				};
				struct InputInterfaceKeyEqual
				{
					inline bool operator()(const CStageIntrospectionData::SInputInterface& lhs, const CStageIntrospectionData::SInputInterface& rhs) const
					{
						return std::memcmp(&lhs, &rhs, sizeof(CStageIntrospectionData::SInputInterface)) == 0;
					}
				};

				using InputInterfaceMap = core::unordered_set<SInputInterface, InputInterfaceHash, InputInterfaceKeyEqual>;
				InputInterfaceMap m_input;
				std::variant<
					core::vector<SFragmentOutputInterface>, // when `params.shader->getStage()==ESS_FRAGMENT`
					core::vector<SOutputInterface> // otherwise
				> m_output;
				//!
				SPushConstantInfo<> m_pushConstants = {};
				//! Each vector is sorted by `binding`
				constexpr static inline uint8_t DESCRIPTOR_SET_COUNT = 4;
				core::vector<SDescriptorVarInfo<>> m_descriptorSetBindings[DESCRIPTOR_SET_COUNT];
				// The idea is that we construct with based_span (so we can add `.data()` when accessing)
				// then just convert from `SMemoryBlock<true>` to `SMemoryBlock<false>`
				// in-place when filling in the `descriptorSetBinding` vector which we pass to ctor
				core::vector<char> m_memPool;
		};
		class NBL_API2 CPipelineIntrospectionData final : public CIntrospectionData
		{
			public:
				struct SDescriptorInfo final : CIntrospectionData::SDescriptorInfo
				{
					inline bool isArray() const {return count != 1u;}
					inline bool isRuntimeSized() const {return isRuntimeSizedFlag;}

					uint32_t count : 31 = 0;
					uint32_t isRuntimeSizedFlag : 1;
					// Which shader stages touch it
					core::bitflag<ICPUShader::E_SHADER_STAGE> stageMask = ICPUShader::ESS_UNKNOWN;
				};
				//
				inline CPipelineIntrospectionData()
				{
					std::fill(m_pushConstantBytes.begin(),m_pushConstantBytes.end(),ICPUShader::ESS_UNKNOWN);
					std::fill(m_highestBindingNumbers.begin(), m_highestBindingNumbers.end(), HighestBindingData());
				}

				// returns true if successfully added all the info to self, false if incompatible with what's already in our pipeline or incomplete (e.g. missing spec constants)
				bool merge(const CStageIntrospectionData* stageData, const ICPUShader::SSpecInfoBase::spec_constant_map_t* specConstants=nullptr);

				//
				core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection(core::smart_refctd_ptr<const CStageIntrospectionData>& introspection);
				core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(const uint32_t setID);
				core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection(core::smart_refctd_ptr<const CStageIntrospectionData>& introspection);

			protected:
				// ESS_UNKNOWN on a byte means its not declared in any shader merged so far
				std::array<core::bitflag<ICPUShader::E_SHADER_STAGE>,MaxPushConstantsSize> m_pushConstantBytes;
				//
				struct Hash
				{
					inline size_t operator()(const SDescriptorInfo& item) const {return std::hash<size_t>()(item.binding);}
				};
				struct KeyEqual
				{
					inline bool operator()(const SDescriptorInfo& lhs, const SDescriptorInfo& rhs) const
					{
						return lhs.binding==rhs.binding;
					}
				};
				using DescriptorSetBindings = core::unordered_set<SDescriptorInfo, Hash, KeyEqual>;
				DescriptorSetBindings m_descriptorSetBindings[4];

				struct HighestBindingData
				{
					bool isRunTimeSized = false;
					int32_t binding = -1;
				};
				using HighestBindingArray = std::array<HighestBindingData, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT>;
				HighestBindingArray m_highestBindingNumbers;
		};

		// 
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::debugInfoFlags enabling EDIF_SOURCE_BIT implicitly or explicitly, with no `spirvOptimizer` used in order to include names in introspection data
		inline core::smart_refctd_ptr<const CStageIntrospectionData> introspect(const CStageIntrospectionData::SParams& params, bool insertToCache=true)
		{
			if (!params.shader)
				return nullptr;
    
			if (params.shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_SPIRV)
				return nullptr;

			// TODO: templated find!
			auto introspectionData = m_introspectionCache.find(params);
			if (introspectionData != m_introspectionCache.end())
				return *introspectionData;

			auto introspection = doIntrospection(params);

			if (insertToCache)
				m_introspectionCache.insert(introspectionData,introspection);

			return introspection;
		}

		//! creates pipeline for a single ICPUShader
		core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info, core::smart_refctd_ptr<ICPUPipelineLayout>&& layout = nullptr);

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
		core::smart_refctd_ptr<const CStageIntrospectionData> doIntrospection(const CStageIntrospectionData::SParams& params);
		size_t calcBytesizeForType(spirv_cross::Compiler& comp, const spirv_cross::SPIRType& type) const;
		// TODO: hash map instead
		using OutputVecT = core::vector<CSPIRVIntrospector::CStageIntrospectionData::SOutputInterface>;
		using FragmentOutputVecT = core::vector<CSPIRVIntrospector::CStageIntrospectionData::SFragmentOutputInterface>;

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

		using ParamsToDataMap = core::unordered_set<core::smart_refctd_ptr<const CStageIntrospectionData>,KeyHasher,KeyEquals>;
		ParamsToDataMap m_introspectionCache;
};

} // nbl::asset
#endif
