// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_NODE_POOL_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"
#include "nbl/core/alloc/refctd_memory_resource.h"

#include <type_traits>


namespace nbl::asset::material_compiler3
{

// Class to manage all nodes' backing and hand them out as `uint32_t` handles
class CNodePool : public core::IReferenceCounted
{
		struct Config
		{
			using AddressAllocator = core::LinearAddressAllocator<uint32_t>;
			using HandleValue = uint32_t;
			constexpr static inline bool ThreadSafe = false;
		};

	public:
		template<typename T, uint16_t N, uint16_t M>
		static inline void printMatrix(std::ostringstream& sstr, const hlsl::matrix<T,N,M>& m)
		{
			for (uint16_t i=0; i<N; i++)
			{
				if (i)
					sstr << "\\n";
				for (uint16_t j=0; j<M; j++)
				{
					if (j)
						sstr << ",";
					sstr << std::to_string(m[i][j]);
				}
			}
		}

		//
		using obj_pool_type = nbl::core::CObjectPool<Config>;

		//
		inline obj_pool_type& getObjectPool() {return m_composed;}
		inline const obj_pool_type& getObjectPool() const {return m_composed;}
		
		//
		struct SParameter
		{
			inline operator bool() const
			{
				return abs(scale)<std::numeric_limits<float>::infinity() && (!view || viewChannel<getFormatChannelCount(view->getCreationParameters().format));
			}
			inline bool operator!=(const SParameter& other) const
			{
				if (scale!=other.scale)
					return true;
				if (viewChannel!=other.viewChannel)
					return true;
				// don't compare paddings!
				if (view!=other.view)
					return true;
				return sampler!=other.sampler;
			}
			inline bool operator==(const SParameter& other) const {return !operator!=(other);}

			void printDot(std::ostringstream& sstr, const core::string& selfID) const;

			// at this stage we store the multipliers in highest precision
			float scale = std::numeric_limits<float>::infinity();
			// rest are ignored if the view is null
			uint8_t viewChannel : 2 = 0;
			uint8_t padding[3] = {0,0,0}; // TODO: padding stores metadata, shall we exclude from assignment and copy operators?
			core::smart_refctd_ptr<const ICPUImageView> view = {};
			// lodbias and clamp shadow comparison functions, anisotropy and minFilter are ignored
			// NOTE: could take only things that matter from the sampler and pack the viewChannel and reduce padding
			ICPUSampler::SParams sampler = {};
		};
		// In the forest, this is not a node, we'll deduplicate later
		template<uint8_t Count>
		struct SParameterSet
		{
			inline operator bool() const
			{
				for (uint8_t i=0; i<Count; i++)
				if (!params[i])
					return false;
				return true;
			}
			// Ignored if no modulator textures and isotropic BxDF
			uint8_t& uvSlot() {return params[0].padding[0];}
			const uint8_t& uvSlot() const {return params[0].padding[0];}
			// Note: the padding abuse
			static_assert(sizeof(SParameter::padding)>0);

			template<typename StringConstIterator=const core::string*>
			inline void printDot(std::ostringstream& sstr, const core::string& selfID, StringConstIterator paramNameBegin={}, const bool uvRequired=false) const
			{
				CNodePool::printDotParameterSet<Count,StringConstIterator>(*this,Count,sstr,selfID,std::forward<StringConstIterator>(paramNameBegin),uvRequired);
			}

			// identity transform by default, ignored if no UVs
			// NOTE: a transform could be applied per-param, whats important that the UV slot remains the smae across all of them.
			hlsl::float32_t2x3 uvTransform = hlsl::float32_t2x3(
				1,0,0,
				0,1,0
			);
			SParameter params[Count] = {};

			// to make sure there will be no padding inbetween
			static_assert(alignof(SParameter)>=alignof(hlsl::float32_t2x3));
		};

		//
		class INode
		{
			public:
				virtual const std::string_view getTypeName() const = 0;
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = obj_pool_type::template typed_pointer_type<T>;

		// Debug Info node
		class CDebugInfo : public obj_pool_type::IVariableSize, public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::asset::material_compiler3::CNodePool::CDebugInfo";}
				inline uint32_t getSize() const {return calc_size(nullptr,m_size);}
				
				static inline uint32_t calc_size(const void* data, const uint32_t size)
				{
					return core::alignUp(sizeof(CDebugInfo)+size,alignof(CDebugInfo));
				}
				static inline uint32_t calc_size(const std::string_view& view)
				{
					return calc_size(nullptr,view.length()+1);
				}
				inline CDebugInfo(const void* data, const uint32_t size) : m_size(size)
				{
					if (data)
						memcpy(std::launder(this+1),data,m_size);
				}
				inline CDebugInfo(const std::string_view& view) : CDebugInfo(nullptr,view.length()+1)
				{
					auto* out = std::launder(reinterpret_cast<char*>(this+1));
					if (m_size>1)
						memcpy(out,view.data(),m_size);
					out[m_size-1] = 0;
				}

				inline const std::span<const uint8_t> data() const
				{
					return {reinterpret_cast<const uint8_t*>(this+1),m_size};
				}

			protected:
				const uint32_t m_size;
		};

		// Why are all of these kept together and forced to fetch from the same UV ?
		// Because they're supposed to be filtered together with the knowledge of the NDF
		// TODO: should really be 5 parameters (2+3) cause of rotatable anisotropic roughness
		struct SBasicNDFParams : SParameterSet<4>
		{
			inline auto getDerivMap() {return std::span<SParameter,2>(params,2);}
			inline auto getDerivMap() const {return std::span<const SParameter,2>(params,2);}
			inline auto getRougness() {return std::span<SParameter,2>(params+2,2);}
			inline auto getRougness() const {return std::span<const SParameter,2>(params+2,2);}
					
			inline SBasicNDFParams()
			{
				// initialize with constant flat deriv map and smooth roughness
				for (auto& param : params)
					param.scale = 0.f;
			}

			// The usage of a normal modifier implies potential anisotropic roughness when filtering (CLEAR, CLEAN, Neural), so all 4 (or 5) parameters should come from a texture.
			// When normal modifier is not used, the roughness can still come from a texture but can be isotropic or anisotropic. Weird combos will require making tiny textures when converting from AST.
			enum class EParamType : uint8_t
			{
				TotallyMapped,
				AnisotropicMapped,
				IsotropicMapped,
				AnisotropicConstant,
				IsotropicConstant
			};
			// This is about how we load our data into the NDF not whether the NDF is really isotropic
			inline EParamType determineParamType() const
			{
				// a derivative map from a texture allows for anisotropic NDFs at higher mip levels when pre-filtered properly
				for (auto i=0; i<2; i++)
				if (getDerivMap()[i].scale!=0.f && getDerivMap()[i].view)
					return EParamType::TotallyMapped;
				const auto roughness = getRougness();
				// having one roughness be mapped and another not mapped, isn't very useful in any renderer
				const bool roughnessIsMapped = roughness[0].scale!=0.f && roughness[0].view || roughness[1].scale!=0.f && roughness[1].view;
				// if roughness inputs are not equal (same scale, same texture) then NDF can be anisotropic in places
				if (roughness[0]!=roughness[1])
				{
					return roughnessIsMapped ? EParamType::AnisotropicMapped:EParamType::AnisotropicConstant;
				}
				else if (roughnessIsMapped)
				{
					return EParamType::IsotropicMapped;
				}
				else
					return EParamType::IsotropicConstant;
			}

			// conservative check, checks if we can optimize certain things this way
			inline bool definitelyIsotropic() const
			{
				switch (determineParamType())
				{
					case EParamType::IsotropicMapped: [[fallthrough]];
					case EParamType::IsotropicConstant:
						break;
					default:
						return false;
				}
				// if a reference stretch is used, stretched triangles can turn the distribution anisotropic
				return stretchInvariant();
			}
			// whether the derivative map and roughness is constant regardless of UV-space texture stretching
			inline bool stretchInvariant() const {return !(abs(hlsl::determinant(reference))>std::numeric_limits<float>::min());}

			void printDot(std::ostringstream& sstr, const core::string& selfID) const;

			// Ignored if not invertible, otherwise its the reference "stretch" (UV derivatives) at which identity roughness and normalmapping occurs
			hlsl::float32_t2x2 reference = hlsl::float32_t2x2(0,0,0,0);
		};

		//
		template<typename T>
		inline const std::string_view getTypeName(const typed_pointer_type<T> h) const
		{
			const auto* node = getObjectPool().deref<const T>(h);
			return node ? node->getTypeName():"nullptr";
		}
		

	protected:
		inline CNodePool(typename obj_pool_type::creation_params_type&& params) : m_composed(std::move(params)) {}

		template<uint8_t Count>
		friend struct SParameterSet;
		// Use `_count` instead of `Count` because of how wonkily this stuff gets used
		template<uint8_t Count, typename StringConstIterator=const core::string*>
		static inline void printDotParameterSet(const SParameterSet<Count>& _set, const uint8_t _count, std::ostringstream& sstr, const core::string& selfID, StringConstIterator paramNameBegin={}, const bool uvRequired=false)
		{
			bool imageUsed = false;
			for (uint8_t i=0; i<_count; i++)
			{
				const auto paramID = selfID+"_param"+std::to_string(i);
				if (_set.params[i].view)
					imageUsed = true;
				_set.params[i].printDot(sstr,paramID);
				sstr << "\n\t" << selfID << " -> " << paramID;
				if (paramNameBegin)
					sstr <<" [label=\"" << *(paramNameBegin++) << "\"]";
				else
					sstr <<" [label=\"Param " << std::to_string(i) <<"\"]";
			}
			if (uvRequired || imageUsed)
			{
				const auto uvTransformID = selfID+"_uvTransform";
				sstr << "\n\t" << uvTransformID << " [label=\"uvSlot = " << std::to_string(_set.uvSlot()) << "\\n";
				printMatrix(sstr,*reinterpret_cast<const decltype(_set.uvTransform)*>(_set.params+_count));
				sstr << "\"]";
				sstr << "\n\t" << selfID << " -> " << uvTransformID << "[label=\"UV Transform\"]";
			}
		}

		obj_pool_type m_composed;
};

inline void CNodePool::SParameter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	sstr << "\n\t" << selfID << "[label=\"scale = " << std::to_string(scale);
	if (view)
	{
		sstr << "\\nchannel = " << std::to_string(viewChannel);
		const auto& viewParams = view->getCreationParameters();
		sstr << "\\nWraps = {" << sampler.TextureWrapU;
		if (viewParams.viewType!=ICPUImageView::ET_1D && viewParams.viewType!=ICPUImageView::ET_1D_ARRAY)
			sstr << "," << sampler.TextureWrapV;
		if (viewParams.viewType==ICPUImageView::ET_3D)
			sstr << "," << sampler.TextureWrapW;
		sstr << "}\\nBorder = " << sampler.BorderColor;
		// don't bother printing the rest, we really don't care much about those
	}
	sstr << "\"]";
	// TODO: do specialized printing for image views (they need to be gathered into a view set -> need a printing context struct)
	/*
	struct SDotPrintContext
	{
		std::ostringstream* sstr;
		core::unordered_map<ICPUImageView*,core::blake3_hash>* usedViews;
		uint16_t indentation = 0;
	};
	*/
	if (view)
		sstr << "\n\t" << selfID << " -> _view_" << std::to_string(reinterpret_cast<const uint64_t&>(view));
}

inline void CNodePool::SBasicNDFParams::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	constexpr const char* paramSemantics[] = {
		"dh/du",
		"dh/dv",
		"alpha_u",
		"alpha_v"
	};
	SParameterSet<4>::printDot(sstr,selfID,paramSemantics,!definitelyIsotropic());
	if (!stretchInvariant())
	{
		const auto referenceID = selfID+"_reference";
		sstr << "\n\t" << referenceID << " [label=\"";
		printMatrix(sstr,reference);
		sstr << "\"]";
		sstr << "\n\t" << selfID << " -> " << referenceID << " [label=\"Stretch Reference\"]";
	}
}

// specialization of parameter hashing
template<typename Dummy>
struct core::blake3_hasher::update_impl<CNodePool::SParameter,Dummy>
{
	using input_t = asset::material_compiler3::CNodePool::SParameter;

	static inline void __call(blake3_hasher& hasher, const input_t& param)
	{
		hasher << param.scale;
		if (!param.view)
			return;
		const auto& viewParams = param.view->getCreationParameters();
		// TODO: hash it like CAssetConverter
		{
			hasher << ptrdiff_t(param.view.get());
		}
		// in the future this might change
		hasher << param.viewChannel;
		const auto& sampler = param.sampler;
		hasher << param.sampler.BorderColor;
		hasher << param.sampler.MaxFilter;
		using view_type_e = asset::IImageView<asset::ICPUImage>::E_TYPE;
		switch (viewParams.viewType)
		{
			case view_type_e::ET_3D:
				hasher << param.sampler.TextureWrapW;
				[[fallthrough]];
			case view_type_e::ET_2D: [[fallthrough]];
			case view_type_e::ET_2D_ARRAY: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP_ARRAY:
				hasher << param.sampler.TextureWrapV;
				[[fallthrough]];
			default:
				hasher << param.sampler.TextureWrapU;
				break;
		}
	}
};
template<uint8_t Count, typename Dummy>
struct core::blake3_hasher::update_impl<CNodePool::SParameterSet<Count>,Dummy>
{
	using input_t = asset::material_compiler3::CNodePool::SParameterSet<Count>;

	static inline void __call(blake3_hasher& hasher, const input_t& input)
	{
		bool noTextures = true;
		for (uint8_t i=0; i<Count; i++)
		if (input.params[i].view)
		{
			noTextures = false;
			break;
		}
		if (noTextures)
			return;
		hasher << input.uvTransform;
		hasher << input.uvSlot();
		for (uint8_t i=0; i<Count; i++)
			hasher << input.params[i];
	}
};
template<typename Dummy>
struct core::blake3_hasher::update_impl<CNodePool::SBasicNDFParams,Dummy>
{
	using input_t = asset::material_compiler3::CNodePool::SBasicNDFParams;

	static inline void __call(blake3_hasher& hasher, const input_t& input)
	{
		using type_e = input_t::EParamType;
		const type_e type = input.determineParamType();
		update_impl<uint8_t>::__call(hasher,static_cast<uint8_t>(type));
		update_impl<asset::material_compiler3::CNodePool::SParameterSet<4>>::__call(hasher,*this);
		// reference stretch can be applied on non-mapped NDFs too
		if (!input.stretchInvariant())
			hasher << input.reference;
	}
};

} // namespace nbl::asset::material_compiler3
#endif