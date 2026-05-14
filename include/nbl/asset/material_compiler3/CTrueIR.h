// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_


#include "nbl/system/ILogger.h"
#include "nbl/system/to_string.h"

#include "nbl/asset/material_compiler3/CNodePool.h"
#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material's interface layer, but here they're in "Accumulator Form"
// They appear "flipped upside down", its expected our backends will evaluate contributors first, and then bother with the attenuators. 
class CTrueIR : public CNodePool // TODO: turn into an asset!
{
	    using block_allocator_type = CNodePool::obj_pool_type::block_allocator_type;
		template<typename T>
		using _typed_pointer_type = CNodePool::obj_pool_type::mem_pool_type::typed_pointer_type<T>;

	public:
		// constructor
		using creation_params_type = typename obj_pool_type::creation_params_type;
		static inline core::smart_refctd_ptr<CTrueIR> create(creation_params_type&& params)
		{
			if (params.composed.blockSizeKBLog2<4)
				return nullptr;
			return core::smart_refctd_ptr<CTrueIR>(new CTrueIR(std::move(params)),core::dont_grab);
		}		

		//! Stuff thats kind-of common to CFrontendIR but doesn't make sense in CNodePool
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
				if (view)
				{
					if (viewChannel!=other.viewChannel)
						return true;
					if (wrapU!=other.wrapU)
						return true;
					if (wrapV!=other.wrapV)
						return true;
					if (wrapW!=other.wrapW)
						return true;
					if (borderColor!=other.borderColor)
						return true;
					if (linearMagnification!=other.linearMagnification)
						return true;
				}
				// don't compare paddings!
				return view!=other.view;
			}
			inline bool operator==(const SParameter& other) const {return !operator!=(other);}

			void printDot(std::ostringstream& sstr, const core::string& selfID) const;

			// at this stage we store the multipliers in highest precision
			float scale = std::numeric_limits<float>::infinity();
			// rest are ignored if the view is null, only take the things that matter from ISampler
			static_assert(std::is_same_v<std::underlying_type_t<hlsl::TextureClamp>,uint16_t>);
			uint16_t viewChannel : 2 = 0;
			uint16_t linearMagnification : 1 = true;
			hlsl::TextureClamp wrapU : 3 = hlsl::TextureClamp::ETC_REPEAT;
			hlsl::TextureClamp wrapV : 3 = hlsl::TextureClamp::ETC_REPEAT;
			hlsl::TextureClamp wrapW : 3 = hlsl::TextureClamp::ETC_REPEAT;
			uint16_t borderColor : 3 = ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_BLACK;
			// 1 bit left over
			uint8_t padding[2] = {0,0}; // TODO: padding stores metadata, shall we exclude from assignment and copy operators?
			core::smart_refctd_ptr<const ICPUImageView> view = {};
		};
		// We worry about keeping parameters in the same image view later (the backend)
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
				CTrueIR::printDotParameterSet<Count,StringConstIterator>(*this,Count,sstr,selfID,std::forward<StringConstIterator>(paramNameBegin),uvRequired);
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
		// Why are all of these kept together and forced to fetch from the same UV ?
		// Because they're supposed to be filtered together with the knowledge of the NDF
		// TODO: should really be 5 parameters (2+3) cause of rotatable anisotropic roughness
		struct SBasicNDFParams : SParameterSet<4>
		{
			enum class EDistribution : uint8_t
			{
				GGX = 0,
				Beckmann = 1,
				Invalid = 255
			};
			inline EDistribution& getDistribution() {return reinterpret_cast<EDistribution&>(params[1].padding[0]);}
			inline const EDistribution& getDistribution() const {return reinterpret_cast<const EDistribution&>(params[1].padding[0]);}

			inline auto getDerivMap() {return std::span<SParameter,2>(params,2);}
			inline auto getDerivMap() const {return std::span<const SParameter,2>(params,2);}
			inline auto getRougness() {return std::span<SParameter,2>(params+2,2);}
			inline auto getRougness() const {return std::span<const SParameter,2>(params+2,2);}
					
			inline SBasicNDFParams()
			{
				getDistribution() = EDistribution::Invalid;
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

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				enum class EFinalType : uint8_t
				{
					COrientedLayer=0,
					CCorellatedTransmission,
					CContributorSum,
					CWeightedContributor,
					CEmitter,
					CDeltaTransmission,
					CSpectralVariable,
					COrenNayar,
					CCookTorrance,
					CFactorCombiner,
					CBeer,
					CFresnel,
					CThinInfiniteScatterCorrection
				};
				virtual EFinalType getFinalType() const = 0;

				const auto& getHash() const {return hash;}

				// only call once the nodes underneath are linked up (because it doesn't call recursively), returning empty hash means error/invalid node
				inline core::blake3_hash_t computeHash(const obj_pool_type& pool) const
				{
					core::blake3_hasher hasher = {};
					// always put the node type into the hash
					hasher << static_cast<uint8_t>(getFinalType());
					if (!computeHash_impl(pool,hasher))
 						return {};
					return hasher.operator core::blake3_hash_t();
				}

				// Only sane child count allowed
				virtual uint8_t getChildCount() const = 0;
				inline _typed_pointer_type<const INode> getChildHandle(const uint8_t ix)
				{
					if (ix < getChildCount())
						return getChildHandle_impl(ix);
					return {};
				}
				inline _typed_pointer_type<const INode> getChildHandle(const uint8_t ix) const
				{
					auto retval = const_cast<INode*>(this)->getChildHandle(ix);
					return retval;
				}

				virtual inline std::string_view getChildName_impl(const uint8_t ix) const { return ""; }
				virtual inline void printDot(std::ostringstream& sstr, const core::string& selfID) const {}

			protected:
				friend class CTrueIR;
				// child managment
				virtual inline _typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const { assert(false); return {}; }
				inline void setChild(const uint8_t ix, _typed_pointer_type<INode> newChild)
				{
					assert(ix < getChildCount());
					setChild_impl(ix, newChild);
				}
				virtual inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) { assert(false); }

				inline bool recomputeHash(const obj_pool_type& pool)
				{
					hash = computeHash(pool);
					return hash!=core::blake3_hash_t::EmptyInput();
				}
				virtual bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const = 0;
#define HASH_REQUIREDS_HASH(HANDLE) { \
					if (const auto hash=pool.deref(HANDLE)->getHash(); hash==core::blake3_hash_t::EmptyInput()) \
						return false; \
					else \
						hasher << hash; \
				}
#define HASH_OPTIONALS_HASH(HANDLE) if (HANDLE) {HASH_REQUIREDS_HASH(HANDLE);} else {hasher << core::blake3_hash_t::EmptyInput();}

				virtual _typed_pointer_type<INode> copy(CTrueIR* ir) const = 0;
#define COPY_DEFAULT_IMPL inline _typed_pointer_type<INode> copy(CTrueIR* ir) const override final \
				{ \
					return CNodePool::copyNode<std::remove_const_t<std::remove_pointer_t<decltype(this)> > >(this,ir); \
				}

				// Each node is final and immutable, has a precomputed hash for the whole subtree beneath it.
				// Debug info does not form part of the hash, so can get wildly replaced.
				core::blake3_hash_t hash = core::blake3_hash_t::EmptyInput();
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = _typed_pointer_type<T>;
		// Contributors are things which can either be importance sampled to continue a path or impart a contribution ot the integral
		// We don't make the contributor hold its factor so that we still have a separate hash value and good codegen (all identical contributors go to same function call
		class IContributor : public INode
		{
			public:
				virtual bool isEmitter() const = 0;
		};
		// this is for a term in a mul or add expression
		class IFactor : public INode
		{
			protected:
				uint64_t padding = 0;
		};
#define TYPE_NAME_STR(NAME) "nbl::asset::material_compiler3::CTrueIR::"#NAME
		// we step away from our usual way of doing things via linked lists, because this is simpler to rearrange
		class CFactorCombiner final : public obj_pool_type::IVariableSize, public IFactor
		{
			public:
				inline EFinalType getFinalType() const override {return EFinalType::CFactorCombiner;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CFactorCombiner);}

				// There's only two ops we support
				enum Type : uint8_t
				{
					Mul = 0,
					Add = 1
				};
				struct SState
				{
					uint64_t type : 1 = Type::Mul;
					uint64_t childCount : 6 = 0;
					// which factors get `1-x` for an Add node or `-x` for a Mul node before getting used
					uint64_t childIxComplementMask : 57 = 0x0u;
				};
				static_assert(sizeof(SState) == sizeof(IFactor::padding));
				
				//
				static inline uint32_t calc_size(const SState state)
				{
					return sizeof(CFactorCombiner)+sizeof(child[0])*(state.childCount-1);
				}
				inline CFactorCombiner(const SState state)
				{
					padding = std::bit_cast<uint64_t>(state);
				}

				//
				inline SState getState() const {return std::bit_cast<SState>(padding);}
				//
				inline void setComplementMask(const uint64_t mask)
				{
					assert(mask < (0x1<<57));
					auto state = getState();
					state.childIxComplementMask = mask;
					padding = std::bit_cast<uint64_t>(state);
				}

				inline uint8_t getChildCount() const override final { return getState().childCount; }

				// Only sane child count allowed
				inline void setChildHandle(const uint8_t ix, const typed_pointer_type<const IFactor> handle)
				{
					if (ix < getState().childCount)
						child[ix] = handle;
				}

			protected:
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					if (getState().childCount==0)
						return false;
					hasher << padding; // hash whole state at once
					for (uint16_t i=0; i<getState().childCount; i++)
					{
						if (!child[i])
							return false;
						HASH_REQUIREDS_HASH(child[i]);
					}
					return true;
				}

				inline typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const override final { return child[ix]; }
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) override final { child[ix] = block_allocator_type::_static_cast<IFactor>(newChild); }

				inline _typed_pointer_type<INode> copy(CTrueIR* ir) const override final
				{
					auto& pool = ir->getObjectPool();
					const auto copyH = pool.emplace<std::remove_const_t<std::remove_pointer_t<decltype(this)> > >(getState());
					if (auto* const copy = pool.deref(copyH); copyH)
						*copy = *this;
					return copyH;
				}

				typed_pointer_type<const IFactor> child[1] = {{}};
		};
		// Note that this is not a root node, its a flipped leaf!
		class CWeightedContributor final : public obj_pool_type::INonTrivial, public INode
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					if (!contributor)
						return false;
					HASH_REQUIREDS_HASH(contributor);
					HASH_OPTIONALS_HASH(factor);
					// TODO: else where it hashes with a premade hash of a `IFactor = CConstantFactor` of monochrome 1.0 scalar
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CWeightedContributor;}

				inline uint8_t getChildCount() const override final { return 2; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CWeightedContributor);}
				inline std::string_view getChildName_impl(const uint8_t ix) const override final { return ix ? "factor" : "contributor"; }


				typed_pointer_type<const IContributor> contributor = {};
				// if null then assumed to be 1
				typed_pointer_type<const IFactor> factor = {};

		    protected:
			    COPY_DEFAULT_IMPL

				inline typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const override final
			    {
					if (ix)
						return factor;
					return contributor;
			    }
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) override final
			    {
					if (ix)
			            factor = block_allocator_type::_static_cast<IFactor>(newChild);
					else
						contributor = block_allocator_type::_static_cast<IContributor>(newChild);
			    }
		};
		// One BRDF or BTDF component of a layer is represented as
		// 	   f(w_i,w_o) = Sum_i^N Product_j^{N_i} h_{ij}(w_i,w_o) l_i(w_i,w_o)
		class CContributorSum final : public obj_pool_type::INonTrivial, public INode
		{

				// CANONICALIZATION NOTE: The conntributors shall be ordered in following order:
				// - according to their type, emitters first, then BxDFs, within those
				//	+ Emitters with IES profile, then without 
				//	+ BxDFs by their type
				//		* within the same type, by the parameters (with/without bump, then rest
				// - all things being equal, order by hash
				// For factors we want to order from lowest spectrality to highest, the computational expense.
				// Due to Schussler and Yining, BxDFs are highly unlikely to evaluate to 0 but they can produce invalid samples.
				// Also BxDFs and Emitters (which only hold IES profiles) are monochromatic so accumulating their contribution first uses least registers.
				// Fresnel, Beer extinction and other analytic modifiers never produce 0s so should be evaluated last.
				// Function factors which have to be evaluated via composition go even later, but within their dimension category.
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					if (product)
					{
						HASH_REQUIREDS_HASH(product);
						HASH_OPTIONALS_HASH(rest);
					}
					else if (rest) // this is an invalid combo, because `rest->product` could be hoisted in place of `product`
						return false;
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CContributorSum;}

				inline uint8_t getChildCount() const override final { return 2; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CContributorSum);}
				inline std::string_view getChildName_impl(const uint8_t ix) const override final { return ix ? "rest" : "product"; }

				// the product is ...
				typed_pointer_type<const CWeightedContributor> product = {};
				// the rest node is ...
				_typed_pointer_type<const CContributorSum> rest = {};

		    protected:
			    COPY_DEFAULT_IMPL

				inline typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const override final
				{
					if (ix)
						return rest;
					return product;
				}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) override final
				{
					if (ix)
						rest = block_allocator_type::_static_cast<CContributorSum>(newChild);
					else
						product = block_allocator_type::_static_cast<CWeightedContributor>(newChild);
				}
		};

		// For codegen, we can compute total uncorrelated layering by convolving every `h_{ij}(w_i,w_o) l_i(w_i,w_o)` term in the layer above with layer below
		class COrientedLayer;
		// Corellated layering is a far far far TODO, all it means that certain convolutions don't happen - certain BxDFs don't layer over each other (tricky to express in AST and IR)
		class CCorellatedTransmission final : public obj_pool_type::INonTrivial, public INode
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					// invalid combo, null btdf prevents you crossing over to the next layer and hitting current from below
					// also if there's no btdf it makes no sense to have a `next` sibling.
					if (!btdf)
						return false;
					HASH_REQUIREDS_HASH(btdf);
					HASH_OPTIONALS_HASH(brdfBottom);
					HASH_OPTIONALS_HASH(coated);
					HASH_OPTIONALS_HASH(next);
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CCorellatedTransmission;}

				inline uint8_t getChildCount() const override final { return 3; }	// TODO: or 4?

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCorellatedTransmission);}
				inline std::string_view getChildName_impl(const uint8_t ix) const override final { return ix ? (ix > 1 ? "next" : "brdfBottom") : "btdf"; }

				// you can set the children later
				inline CCorellatedTransmission() = default;

				// Obligatory if you don't want transmission then don't put a valid handle in `COrientedLayer::firstTransmission`
				// Also shouldn't contain `CDeltaTransmission` in the BTDF unless `coated` is null (TODO a check for that)
				typed_pointer_type<const CContributorSum> btdf = {};
				// because the layer is oriented, these last two members must be null when the coating stops
				typed_pointer_type<const CContributorSum> brdfBottom = {};
				_typed_pointer_type<const COrientedLayer> coated = {};
				// optional, indicates a "sibling" transmission thats next to this one
				_typed_pointer_type<const CCorellatedTransmission> next = {};

		    protected:
			    COPY_DEFAULT_IMPL

		        inline typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const override final
				{
					if (ix > 1)
						return next;
					if (ix)
						return brdfBottom;
					return btdf;
				}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) override final
				{
					if (ix > 1)
						next = block_allocator_type::_static_cast<CCorellatedTransmission>(newChild);
					else if (ix)
						brdfBottom = block_allocator_type::_static_cast<CContributorSum>(newChild);
					else
						btdf = block_allocator_type::_static_cast<CContributorSum>(newChild);
				}
		};
		// The oriented layer is a layer with already all the Etas reciprocated, etc.
		class COrientedLayer final : public obj_pool_type::INonTrivial, public INode
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					HASH_OPTIONALS_HASH(brdfTop);
					HASH_OPTIONALS_HASH(firstTransmission);
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::COrientedLayer;}

				inline uint8_t getChildCount() const override final { return 2; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrientedLayer);}

				// you can set the children later
				inline COrientedLayer() = default;

				// These are same as the frontend's except that all the etas are oriented and reciprocated
				typed_pointer_type<const CContributorSum> brdfTop = {};
				// this node must be non-null until the last layer
				typed_pointer_type<const CCorellatedTransmission> firstTransmission = {};

		    protected:
			    COPY_DEFAULT_IMPL

				inline typed_pointer_type<const INode> getChildHandle_impl(const uint8_t ix) const override final
				{
					if (ix)
						return firstTransmission;
					else
						return brdfTop;
				}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<const INode> newChild) override final
				{
					if (ix)
						firstTransmission = block_allocator_type::_static_cast<CCorellatedTransmission>(newChild);
					else
						brdfTop = block_allocator_type::_static_cast<CContributorSum>(newChild);
				}
		};
		//
		class IFactorLeaf : public IFactor
		{
			public:
				inline bool isScalar() const {return getSpectralBins()==1;}

				virtual uint8_t getSpectralBins() const = 0;
		};
		//
		class ISpectralVariable
		{
				inline SParameter& getParameter(const uint8_t i) {return pWonky()->params[i];}

			public:
				enum class ESemantics : uint8_t
				{
					NoneUndefined = 0,
					// 3 knots, their wavelengths are implied and fixed at color primaries
					Fixed3_SRGB = 1,
					Fixed3_DCI_P3 = 2,
					Fixed3_BT2020 = 3,
					Fixed3_AdobeRGB = 4,
					Fixed3_AcesCG = 5,
					// Ideas: each node is described by (wavelength,value) pair
					// PairsLinear = 5, // linear interpolation
					// PairsLogLinear = 5, // linear interpolation in wavelenght log space
				};

				// essential!
				inline uint8_t getKnotCount() const
				{
					static_assert(sizeof(SParameter::padding)>1);
					return pWonky()->params[0].padding[1];
				}

				// encapsulation due to padding abuse
				inline auto& uvTransform() {return pWonky()->uvTransform;}
				inline const auto& uvTransform() const {return pWonky()->uvTransform;}

				inline uint8_t& uvSlot() {return pWonky()->uvSlot();}
				inline const uint8_t& uvSlot() const {return pWonky()->uvSlot();}
				
				inline const SParameter& getParameter(const uint8_t i) const {assert(i<getKnotCount()); return pWonky()->params[i];}
				inline void setParameter(const uint8_t i, const SParameter& value)
				{
					assert(i<getKnotCount());
					auto& param = getParameter(i);
					uint8_t backup[sizeof(param.padding)];
					std::copy_n(param.padding,sizeof(param.padding),backup);
					param = value;
					std::copy_n(backup,sizeof(param.padding),param.padding);
				}

				inline ESemantics getSemantics() const
				{
					if (getKnotCount()>1)
						return static_cast<ESemantics>(getParameter(1).padding[0]);
					else
						return ESemantics::NoneUndefined;
				}

				inline void setSemantics(const ESemantics value)
				{
					if (getKnotCount()>1)
						getParameter(1).padding[0] = static_cast<uint8_t>(value);
				}

			protected:
				inline ISpectralVariable() = default;
				// delete all these so we dont implicitly copy
				inline ISpectralVariable(const ISpectralVariable&) = delete;
				inline ISpectralVariable(ISpectralVariable&&) = delete;
				inline ISpectralVariable& operator=(const ISpectralVariable&) = delete;
				inline ISpectralVariable& operator=(ISpectralVariable&&) = delete;
				// fill out the params later
				inline void init(const uint8_t knotCount)
				{
					std::uninitialized_default_construct_n(pWonky(),1);
					if (knotCount>1)
						std::uninitialized_default_construct_n(pWonky()->params+1,knotCount-1);
					// back up the count
					static_assert(sizeof(SParameter::padding)>1);
					getParameter(0).padding[1] = knotCount;
					// set it correctly for monochrome
					setSemantics(ESemantics::NoneUndefined);
				}
				inline void init(const ISpectralVariable& other)
				{
					const auto* const src = other.pWonky();
					auto* const dst = pWonky();
					std::uninitialized_copy_n(src,1,dst);
					const size_t count = other.getKnotCount();
					if (count>1)
						std::uninitialized_copy_n(src->params+1,count-1,dst->params+1);
				}
				inline void init(const uint8_t knotCount, const ISpectralVariable& other)
				{
					init(knotCount);
					const auto count = hlsl::min(other.getKnotCount(),knotCount);
					for (uint8_t c=0; c<count; c++)
						getParameter(c) = other.getParameter(c);
					// restore
					getParameter(0).padding[1] = knotCount;
				}
				inline ~ISpectralVariable() = default;
				
				bool valid(const system::logger_opt_ptr logger) const;

				virtual inline SParameterSet<1>* pWonky() = 0;
				inline const SParameterSet<1>* pWonky() const {return const_cast<const SParameterSet<1>*>(const_cast<ISpectralVariable*>(this)->pWonky());}
		};
		// This node could also represent non directional emission, but we have another node for that
		template<class OtherBase> requires std::is_base_of_v<ISpectralVariable,OtherBase>
		class alignas(SParameterSet<1>) CSpectralVariable final : public obj_pool_type::IVariableSize, public OtherBase
		{
				using this_t = CSpectralVariable<OtherBase>;

			protected:
				inline ~CSpectralVariable()
				{
					const auto count = ISpectralVariable::getKnotCount();
					std::destroy_n(pWonky(),1);
					if (count>1)
						std::destroy_n(pWonky()->params+1,count-1);
				}

				inline SParameterSet<1>* pWonky() override final {return reinterpret_cast<SParameterSet<1>*>(this+1);}

			public:
				inline CSpectralVariable(const uint8_t knotCount) : OtherBase() {ISpectralVariable::init(knotCount);}
				inline CSpectralVariable(const ISpectralVariable& other) : OtherBase() {ISpectralVariable::init(other);}
				inline CSpectralVariable(const uint8_t knotCount, const ISpectralVariable& other) : OtherBase() {ISpectralVariable::init(knotCount,other);}
				
				//
				inline _typed_pointer_type<this_t> copy(obj_pool_type& pool) const
				{
					// need to coax the compiler into compiling
					const ISpectralVariable& rThis = *this;
					return pool.emplace<this_t>(rThis);
				}

				//
				static inline uint32_t calc_size(const uint8_t knotCount)
				{
					return sizeof(this_t)+sizeof(SParameterSet<1>)+sizeof(SParameter)*(knotCount-1);
				}
				// for copying
				static inline uint32_t calc_size(const ISpectralVariable& other)
				{
					return calc_size(other.getKnotCount());
				}
				static inline uint32_t calc_size(const uint8_t knotCount, const ISpectralVariable& other)
				{
					return calc_size(knotCount);
				}
				
				// TODO: improve the token pasting here
				inline const std::string_view getTypeName() const override final {return TYPE_NAME_STR(CSpectralVariable<SpectralBins>);}
		};
		//
		class ISpectralVariableFactor : public ISpectralVariable, public IFactorLeaf
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override final
				{
					const auto count = getKnotCount();
					hasher << count;
					{
						const ESemantics semantics = getSemantics();
						if (getKnotCount()>1 && semantics==ESemantics::NoneUndefined)
							return false;
						hasher << semantics;
					}
					bool hasTextures = false;
					const auto* const wonky = pWonky();
					for (uint8_t i=0; i<count; i++)
					{
						// fear not, without a view, nothing excpt for scale will hash
						hasher << wonky->params[i];
						hasTextures = hasTextures || wonky->params[i].view;
					}
					if (hasTextures)
					{
						hasher << wonky->uvTransform;
						hasher << wonky->uvSlot();
					}
					return true;
				}

			public:
				inline EFinalType getFinalType() const override final
				{
					return EFinalType::CSpectralVariable;
				}

				inline uint8_t getChildCount() const override final { return 0; }

				//
				inline uint8_t getSpectralBins() const override final {return getKnotCount();}

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override final;

		    protected:
				inline _typed_pointer_type<INode> copy(CTrueIR* ir) const override final
				{
					return static_cast<const CSpectralVariableFactor*>(this)->copy(ir->getObjectPool());
				}
		};
		using CSpectralVariableFactor = CSpectralVariable<ISpectralVariableFactor>;
		
		// Unit Radiance emitter modulated by an IES profile
		class CEmitter final : public obj_pool_type::INonTrivial, public IContributor
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << profileTransform;
					if (profile.view)
					{
						// we ignore most of the sampler, needs to be set always the same
						auto copy = profile;
						copy.wrapU = hlsl::TextureClamp::ETC_CLAMP_TO_EDGE;
						copy.wrapV = hlsl::TextureClamp::ETC_CLAMP_TO_EDGE;
						copy.wrapW = hlsl::TextureClamp::ETC_CLAMP_TO_EDGE;
						copy.linearMagnification = true;
						copy.borderColor = ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_TRANSPARENT_BLACK;
						// there's a limited set of symmetries we can exploit in our IES tabulations, TODO: check which (probably not REPEAT)
						hasher << copy;
					}
					else
						hasher << profile.scale;
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CEmitter;}

				inline uint8_t getChildCount() const override final { return 0; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CEmitter);}

				inline bool isEmitter() const override {return true;}

				// you can set the members later
				inline CEmitter() = default;

				// This can be anything like an IES profile, if invalid, there's no directionality to the emission
				// `profile.scale` can still be used to influence the light strength without influencing NEE light picking probabilities
				SParameter profile = {};
				hlsl::float32_t3x3 profileTransform = hlsl::float32_t3x3(
					1,0,0,
					0,1,0,
					0,0,1
				);
				// TODO: semantic flags/metadata (symmetries of the profile)

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override final;

		    protected:
			    COPY_DEFAULT_IMPL
		};

		// To use a bump map, the Material needs to be provided UVs (which can or can not have associated tangents and smooth normals), but that's the responsibility of backend.
		// The bump map just needs to be provided a TBN basis thats orthogonal so that normals or derivatives can be transformed.
		// Normal maps will use Olano so won't be assumed to be normalized (they wouldn't be anyway due to interpolation), and derivative maps are never normalized and have huge range.
		// Depending on the mode (deformation stretch enabled or not) the TB of the matrix will be normalized before the multiplication so that TBN is orthonormal and does not impart a UV stretch.
		// After multiplication the normal will be normalized again, which is when we should apply Schussler or Yining.
		//  
		// Schussler and Yining require we define a tangential microfacet from the perturbed normal `P` but differently:
		// - Schussler wants normalize({-P.x,-P.y,0}) which is totally tangential and orthogonal to the geometric normal
		// - Yining wants normalize({-P.xy*P.z,1-P.z*P.z}) which is orthogonal to the perturbed normal
		// Both give an NDF with average normal projected onto the geometric normal being colinear to the geometric normal.
		// Each microfacet has different pros and cons:
		// - Tangential microfacet can use 100% perfect mirror material because its impossible to get 1-order scattering from it (camera and light can't see it at the same time)
		// - Mirror Tangential microfacet makes 2-nd order scattering tractable, just evaluate the BRDF 3 times with original L and V, then once with reflected L and reflected V
		// - for Lambertian BRDF this is only 2 evaluations because reflected V doesn't change anything
		// - the dot products with reflected L or V can be computed quickly from the regular ones, but `H` always changes in this case
		// - Orthogonal microfacet makes sure zero masking is achieved when LdotP=1 and appearance is 100% preserved in this case
		// - Orthogonal microfacet can be seen directly and therefore it needs to have the same BxDF as the perturbed microfacet
		// - The above requirements makes 2-nd order scattering intractable for orthogonal microfacets without random walks in the surface profile
		// - We only need to evaluate the BxDF twice with Yinning and all products with perturbed normal can be obtained similarly to obtaining the reflected L and V for schussler (permutations and scaling)
		// Both approaches are imperfect and in the limit will cause "fresnel lens" like appearance with Cook Torrance low roughness materials, you'll see a blend between classical bump mapped appearance and:
		// - for 2nd-order Schussler a reflection of the objects by a flipped normal {-P.xy,P.z} due to mirror microfacet making a "virtual normal" behind the mirror 
		// - for 1st-order Yinning a reflection of the objects by an orthogonal normal which approaches the geometric normal as the perturbed normal gets more extreme
		// This means that a tangential microfacet will show no blend/retroreflection if both L and V are behind the tangential microfacet.
		// Whereas an orthogonal microfacet will show no blend/retroreflection if both L and V are below the orthogonal microfacet, so tangential ensures this property for half the hemisphere.
		// But with extreme normal perturbation, the appearance of Schussler will match that of a very deep V-groove losing a lot of energy, but Yining will look like a flat surface of the same material.
		// This is because Schussler has no bound on the ratio of the projected microsurface area on the geometric surface, whereas Yining bounds it to be <= 2.0 
		// 
		// There's also a Schussler variant where the tangential microfacet has the same BRDF as the perturbed one, with only 1-st order scattering. This is nice because diffuse doesn't become a special case.
		// Then with this you can actually see the tangential microfacet (smooth mirror makes it black with only 1st order), which works better for BSDF than 2nd order scattering with a mirror facet.
		// Note that transmissive or smooth glass tangential face could never work, because it presents same issues as a non-tangential mirror facet (V and L can interact directly in 1-st order).
		// Mirror facet would cause you to see refractions from a reflected V, which would be weird when the VdotP->0 and you'd expect 100% fresnel reflection. Worse yet you'd see them in the "wrong" direction.
		// Having a tangential microfacet with same BSDF does blend towards transmission as `VdotP->0`, but at least the refractive rays correctly have `L.xy = -V.xy` and are going "forward" although
		// they are bending upwards vs. the transmitted direction, not downwards like they would against the macro surface or the perturbed surface before VdotP==0.
		// 
		// Both this and Yining are faster because of the 2 BxDF evaluations instead of 3, and they only differ by the inter-facet shadowing and masking functions and the tangent facet normal is computed.
		// However an orthogonal microfacet will work better for BSDFs because the facet tangents and normals form a square and not a rhombus, which means that a V to the "left" of the perturbed normal
		// will always be to the "right" of the orthogonal microfacet, meaning we'll get a consistent refraction (both L=refract(V,facet) will curve away from -V in the same direction.
		class IBxDF : public IContributor
		{
			public:
				inline bool isEmitter() const override {return false;}
		};
		class CDeltaTransmission final : public obj_pool_type::INonTrivial, public IBxDF
		{
				// nothing to do
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override {return true;}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CDeltaTransmission;}

				inline uint8_t getChildCount() const override final { return 0; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CDeltaTransmission);}

				inline CDeltaTransmission() = default;

		    protected:
			    COPY_DEFAULT_IMPL
		};
		class IBxDFWithNDF : public IBxDF
		{
			protected:
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << ndfParams;
					return true;
				}

			public:
				SBasicNDFParams ndfParams = {};
		};
		class COrenNayar final : public obj_pool_type::INonTrivial, public IBxDFWithNDF
		{
			public:
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					if (ndfParams.getDistribution()!=SBasicNDFParams::EDistribution::Invalid)
						return false;
					IBxDFWithNDF::computeHash_impl(pool,hasher);
					return true;
				}

				inline EFinalType getFinalType() const override {return EFinalType::COrenNayar;}

				inline uint8_t getChildCount() const override final { return 0; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrenNayar);}

				inline COrenNayar() = default;

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override final;

		    protected:
			    COPY_DEFAULT_IMPL
		};
		class CCookTorrance final : public obj_pool_type::INonTrivial, public IBxDFWithNDF
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					if (ndfParams.getDistribution()>SBasicNDFParams::EDistribution::Beckmann)
						return false;
					IBxDFWithNDF::computeHash_impl(pool,hasher);
					hasher << static_cast<uint8_t>(ndfParams.getDistribution());
					hasher << isEtaReciprocal();
					HASH_OPTIONALS_HASH(orientedRealEta);
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CCookTorrance;}

				inline uint8_t getChildCount() const override final { return 1; }

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCookTorrance);}
				inline std::string_view getChildName_impl(const uint8_t ix) const override final { return "orientedRealEta"; }

				inline CCookTorrance() = default;

				//
				inline bool isEtaReciprocal() const {return ndfParams.params[2].padding[0];}
				inline void setEtaReciprocal(const bool value) {ndfParams.params[2].padding[0] = value;}

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override final;

				// BTDF ONLY! We need this eta to compute the refractions of `L` when importance sampling and the Jacobian during H to L generation for rough dielectrics
				// It does not mean we compute the Fresnel weights though! You might ask why we don't do that given that state of the art importance sampling
				// (at time of writing) is to decide upon reflection vs. refraction after the microfacet normal `H` is already sampled,
				// producing an estimator with just Masking and Shadowing function ratios. The reason is because we can simplify our IR by separating out
				// BRDFs and BTDFs components into separate expressions, and also importance sample much better. 
				typed_pointer_type<const CSpectralVariableFactor> orientedRealEta = {};

		    protected:
			    COPY_DEFAULT_IMPL

				inline typed_pointer_type<INode> getChildHandle_impl(const uint8_t ix) const override final	{ return block_allocator_type::_static_cast<INode>(orientedRealEta); }
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<INode> newChild) override final { orientedRealEta = block_allocator_type::_static_cast<CSpectralVariableFactor>(newChild); }
		};
		//! Basic factor nodes
		// Effective transparency = exp2(log2(perpTransmittance)*thickness/dot(refract(V,X,eta),X)) = exp2(log2(perpTransmittance)*thickness*inversesqrt(1.f+(LdotX-1)*rcpEta))
		// Eta and `LdotX` is taken from the contributor BxDF node. With refractions from Dielectrics, we get just `1/LdotX`, for Delta Transmission we get `1/VdotN` since its the same.
		// Note: its allowed to apply Beer directly on BRDF as well as BTDF to simulate foggy extinction on the top layer
		class CBeer final : public obj_pool_type::INonTrivial, public IFactorLeaf
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << channels;
					HASH_REQUIREDS_HASH(perpTransmittance);
					HASH_REQUIREDS_HASH(thickness);
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CBeer;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CBeer);}

				inline std::string_view getChildName_impl(const uint8_t ix) const override final {return ix ? "Thickness":"Perpendicular\\nTransmittance";}

				inline uint8_t getSpectralBins() const override {return channels;}

				// cannot be null, otherwise no point being there as term will multiply to 0
				typed_pointer_type<const IFactor> perpTransmittance = {};
				// cannot be null, otherwise its always exp2(0) and term will always be 1
				typed_pointer_type<const IFactor> thickness = {};
				// can be worked out by analyzing what we point to, but not needed
				uint8_t channels = 3;
		};
		class CFresnel final : public obj_pool_type::INonTrivial, public IFactorLeaf
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << reciprocateEtas;
					hasher << channels;
					if (orientedImagEta)
					{
						HASH_OPTIONALS_HASH(orientedRealEta);
						hasher << orientedImagEta;
					}
					else
					{
						HASH_REQUIREDS_HASH(orientedRealEta);
					}
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CFresnel;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CFresnel);}

				inline std::string_view getChildName_impl(const uint8_t ix) const override final {return ix ? "Imaginary":"Real";}

				inline uint8_t getSpectralBins() const override {return channels;}

				// cannot be null or a constant of 1 while imaginary is null
				typed_pointer_type<const IFactor> orientedRealEta = {};
				// If null, then treated as a 0 (important to optimize those to 0). MUST be null for BTDFs!
				typed_pointer_type<const IFactor> orientedImagEta = {};
				// easier on the codegen
				uint8_t reciprocateEtas : 1 = false;
				// can be worked out by analyzing what we point to, but not needed
				uint8_t channels : 7 = 3;
		};
		class CThinInfiniteScatterCorrection final : public obj_pool_type::INonTrivial, public IFactorLeaf
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << channels;
					HASH_REQUIREDS_HASH(reflectanceTop);
					HASH_OPTIONALS_HASH(extinction);
					if (reflectanceBottom)
						hasher << reflectanceBottom;
					else
						hasher << reflectanceTop;
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CThinInfiniteScatterCorrection;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CThinInfiniteScatterCorrection);}

				inline std::string_view getChildName_impl(const uint8_t ix) const override final {return ix ? (ix>1 ? "reflectanceBottom":"extinction"):"reflectanceTop";}

				inline uint8_t getSpectralBins() const override {return channels;}

				// cannot be null otherwise no point being there
				typed_pointer_type<const IFactor> reflectanceTop = {};
				// optional, if null then treated as E=1.0
				typed_pointer_type<const IFactor> extinction = {};
				// optional, if null then `reflectanceTop` used in its place
				typed_pointer_type<const IFactor> reflectanceBottom = {};
				// can be worked out by analyzing what we point to, but not needed
				uint8_t channels = 3;
		};
#undef TYPE_NAME_STR
#undef HASH_THE_HASH
		
		//
		struct SBasicNodes final
		{
			public:
				inline bool operator==(const SBasicNodes& other) const = default;

				typed_pointer_type<const CContributorSum> blackHoleBxDF = {};
				typed_pointer_type<const CSpectralVariableFactor> scalarNegation = {};
				// these are never meant to be hashed and inserted into `m_uniqueNodes`
				typed_pointer_type<const CContributorSum> errorBxDF = {};
				typed_pointer_type<const COrientedLayer> errorLayer = {};

			private:
				friend class CTrueIR;
				NBL_API2 SBasicNodes(CTrueIR* ir);
		};
		const SBasicNodes& getBasicNodes() const {return m_basicNodes;}
		
		//
		template<typename T> requires std::is_base_of_v<INode,T>
		inline typed_pointer_type<const T> hashNCache(const typed_pointer_type<T> origH)
		{
			auto* const orig = getObjectPool().deref(origH);
			if (orig)
			if (orig->getHash()!=core::blake3_hash_t::EmptyInput() || orig->recomputeHash(getObjectPool()))
			{
				auto& slot = m_uniqueNodes[orig->getHash()];
				if (slot)
					return CNodePool::obj_pool_type::block_allocator_type::_static_cast<const T>(slot);
				slot = origH;
				return origH;
			}
			return {};
		}

		// Each material comes down to this, this is the only struct we don't de-duplicate
		struct SMaterial
		{
			//
			struct SMetadata
			{
				// Stats needed by a renderer to skip loading parts of a material or remove expensive code altogether
				enum class ECapabilityBits : uint16_t
				{
					None = 0,
					// if any such contributor present
					NotBlackhole = 0x1u<<0, // actually have a material
					NonDelta = 0x1u<<1, // can evaluate against point lights (or other samplings)
					DeltaTransmissive = 0x1u<<2, // can use stochastic transparency for closest hit rays and blending for anyhit 
					NonSpatiallyVaryingEmissive = 0x1u<<3, // definitely register for NEE
					SpatiallyVaryingEmissive = 0x1u<<4, // maybe register for NEE but needs different kind of NEE
					// TODO: 5,6,7 left
					// Bits that help us remove expensive code from impl
					DerivativeMap = 0x1u<<8,
					DirectionallyVaryingEmissive = 0x1u<<9, // IES profile
					Lambertian = 0x1u<<10,
					OrenNayar = 0x1u<<11,
					GGX = 0x1u<<12,
					AnisotropicGGX = 0x1u<<13,
					Beckmann = 0x1u<<14,
					AnisotropicBeckmann = 0x1u<<15,
				};
				//
				constexpr static inline uint8_t MaxUVSlots = 32;
				std::bitset<MaxUVSlots> usedUVSlots = {};
				// the tangent frames are a subset of used UV slots, unless there's an anisotropic BRDF involved
				std::bitset<MaxUVSlots> usedTangentFrames = {};
				//
				core::bitflag<ECapabilityBits> capabilities = {};
			};
			//
			struct SOriented
			{
				// null means no material
				typed_pointer_type<const COrientedLayer> root = {};
				//
				SMetadata metadata = {};
			};
			SOriented front = {};
			SOriented back = {};
			// TODO: more detailed debug info
			CNodePool::typed_pointer_type<CNodePool::CDebugInfo> debugInfo = {};
		};
		inline std::span<const SMaterial> getMaterials() const {return m_materials;}
		
		struct SMaterialHandle
		{
			constexpr static inline uint32_t Invalid = ~0u;
			explicit inline operator bool() const {return value!=Invalid;}

			uint32_t value = Invalid;
		};
		constexpr static inline SMaterialHandle BlackholeMaterialHandle = { 0u };
		//
		inline void reset()
		{
			getObjectPool().reset();
			m_materials.clear();
			m_uniqueNodes.clear();
			// remake reinsert the unique nodes
			const SBasicNodes tmp = SBasicNodes(this);
			assert(m_basicNodes==tmp);
			// create the `BlackholeMaterialHandle`
			m_materials = {SMaterial{.debugInfo=getObjectPool().emplace<CNodePool::CDebugInfo>("CTrueIR's BlackHole Material")}};
		}

		//! Utitilies for copying from other IR into ours while optimizing
		// This one doesn't touch your coated nodes, but only the first coating
		inline bool rewriteSingleLayer(typed_pointer_type<const COrientedLayer>& singleLayer, SMaterial::SMetadata& metadata, CTrueIR* srcIR)
		{
			SRewriteSession session = SRewriteArgs{.src=srcIR,.dst=this,.pMetadata=&metadata};
			return session.rewriteSingleLayer(singleLayer);
		}
		inline bool rewrite(SMaterial::SOriented& material, CTrueIR* srcIR)
		{
			SRewriteSession session = SRewriteArgs{.src=srcIR,.dst=this,.pMetadata=&material.metadata};
			return session.rewrite(material.root);
		}
		inline bool rewrite(SMaterial& material, CTrueIR* srcIR)
		{
			SRewriteSession session = SRewriteArgs{.src=srcIR,.dst=this};
			if (material.front.root)
			{
				session.changeMetadata(&material.front.metadata);
				session.rewrite(material.front.root);
			}
			if (material.back.root)
			{
				session.changeMetadata(&material.back.metadata);
				session.rewrite(material.back.root);
			}
			if (material.debugInfo && srcIR!=this)
			{
				// TODO: copy the debug info across into new pool
				assert(false);
				material = {};
				return false;
			}
			return true;
		}

		uint32_t deepCopy(typed_pointer_type<const INode>* out, const std::span<const typed_pointer_type<const INode>> orig, const CTrueIR* srcIR=nullptr);
		
		// TODO: Optimization passes on the IR
		// It is the backend's job to handle:
		// - constant encoding precision (scale factors, UV matrices, IoRs)
		// - multiscatter compensation
		// - compilation failure to unsupported complex layering
		// - compilation failure to unsupported complex layering
		SMaterialHandle addMaterial(SMaterial material, CTrueIR* srcIR=nullptr)
		{
			if (!srcIR)
				srcIR = this;
			if (rewrite(material,srcIR))
			{
				m_materials.push_back(material);
				return {.value=static_cast<uint32_t>(m_materials.size()-1)};
			}
			else
				return {};
		}

		// For Debug Visualization
		struct SDotPrinter final
		{
			public:
				inline SDotPrinter() = default;
				inline SDotPrinter(const CTrueIR* ir) : m_ir(ir) {}
				// assign in reverse because we want materials to print in order
				inline SDotPrinter(const CTrueIR* ir, std::span<const SMaterial> roots) : m_ir(ir)//, layerStack(roots.rbegin(),roots.rend())
				{
					// should probably size it better, if I knew total node count allocated or live
					visitedNodes.reserve(roots.size()<<4);
					layerStack.reserve(roots.size());
					for (const auto& m : roots)
					{
						layerStack.push_back(m.front.root);
						layerStack.push_back(m.back.root);
					}
				}

				inline void reset(const CTrueIR* ir)
				{
					visitedNodes.clear();
					layerStack.clear();
					nodeStack.clear();
					m_ir = ir;
				}

				NBL_API2 void operator()(std::ostringstream& output);
				inline core::string operator()()
				{
					std::ostringstream tmp;
					operator()(tmp);
					return tmp.str();
				}
			
				core::unordered_set<typed_pointer_type<const INode>> visitedNodes;
				// TODO: track layering depth and indent accordingly?
				core::vector<typed_pointer_type<const COrientedLayer>> layerStack;
				core::vector<typed_pointer_type<const INode>> nodeStack;

			private:
				const CTrueIR* m_ir;
		};
		
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
				printMatrix(sstr,_set.uvTransform);
				sstr << "\"]";
				sstr << "\n\t" << selfID << " -> " << uvTransformID << "[label=\"UV Transform\"]";
			}
		}

	protected:
		struct SRewriteArgs final
		{
			inline bool needDeepCopy() const {return src!=dst;}

			const CTrueIR* const src;
			CTrueIR* const dst;
			const SMaterial::SMetadata* pMetadata;
		};
		struct SRewriteSession final
		{
			public:
				inline SRewriteSession(const SRewriteArgs& _args) : args(_args) {}
				NBL_API2 ~SRewriteSession();

				inline void changeMetadata(SMaterial::SMetadata* pMetadata) {args.pMetadata = pMetadata;}

				NBL_API2 bool rewrite(typed_pointer_type<const COrientedLayer>& oriented);
				NBL_API2 bool rewriteSingleLayer(typed_pointer_type<const COrientedLayer>& oriented);

			private:
				template <typename T, typename... FuncArgs>
				inline typed_pointer_type<T> emplace(FuncArgs&&... args)
				{
					const auto retval =  args.dst->getObjectPool().emplace<T,FuncArgs...>(1u,std::forward<FuncArgs>(args)...);
					if (retval)
						createdNodes.push_back(retval);
					return retval;
				}

				SRewriteArgs args;
				core::vector<typed_pointer_type<INode>> createdNodes;
				bool success = true;
		};

		inline core::string getNodeID(const typed_pointer_type<const INode> handle) const { return core::string("_") + std::to_string(handle.value); }
		inline core::string getLabelledNodeID(const typed_pointer_type<const INode> handle) const
		{
			const INode* node = getObjectPool().deref(handle);
			assert(node);
			core::string retval = getNodeID(handle);
			retval += " [label=\"";
			retval += node->getTypeName();
			retval += "\\n" + system::to_string(node->getHash());
			// maybe label suffix?
			retval += "\"]";
			return retval;
		}

		NBL_API2 CTrueIR(creation_params_type&& params);

		core::vector<SMaterial> m_materials;
		// TODO: either we put the typeid in the hash, or we have a type of hashmap per type
		core::unordered_map<core::blake3_hash_t,typed_pointer_type<const INode>> m_uniqueNodes;
		friend struct SBasicNodes;
		const SBasicNodes m_basicNodes;
};

template class CTrueIR::CSpectralVariable<CTrueIR::ISpectralVariableFactor>;

NBL_ENUM_ADD_BITWISE_OPERATORS(CTrueIR::SMaterial::SMetadata::ECapabilityBits)

}

//
namespace nbl::system::impl
{
template<>
struct system::impl::to_string_helper<asset::material_compiler3::CTrueIR::ISpectralVariable::ESemantics>
{
	using type = asset::material_compiler3::CTrueIR::ISpectralVariable::ESemantics;

	static inline std::string __call(const type value)
	{
		switch (value)
		{
			case type::NoneUndefined:
				return "NoneUndefined";
			case type::Fixed3_SRGB:
				return "Fixed3_SRGB";
			case type::Fixed3_DCI_P3:
				return "Fixed3_DCI_P3";
			case type::Fixed3_BT2020:
				return "Fixed3_BT2020";
			case type::Fixed3_AdobeRGB:
				return "Fixed3_AdobeRGB";
			case type::Fixed3_AcesCG:
				return "Fixed3_AcesCG";
			default:
				break;
		}
		return "";
	}
};
}

//
namespace nbl::asset::material_compiler3
{

inline bool CTrueIR::ISpectralVariable::valid(const system::logger_opt_ptr logger) const
{
	const auto knotCount = getKnotCount();
	// non-monochrome spectral variable 
	if (const auto semantic=getSemantics(); knotCount>1)
	switch (semantic)
	{
		case ESemantics::Fixed3_SRGB: [[fallthrough]];
		case ESemantics::Fixed3_DCI_P3: [[fallthrough]];
		case ESemantics::Fixed3_BT2020: [[fallthrough]];
		case ESemantics::Fixed3_AdobeRGB: [[fallthrough]];
		case ESemantics::Fixed3_AcesCG:
			if (knotCount!=3)
			{
				logger.log("Semantic %s is only usable with 3 knots, this has %d knots",system::ILogger::ELL_ERROR,system::to_string(semantic).c_str(),knotCount);
				return false;
			}
			break;
		default:
			logger.log("Invalid Semantic %s",system::ILogger::ELL_ERROR,system::to_string(semantic).c_str());
			return false;
	}
	for (auto i=0u; i<knotCount; i++)
	if (!getParameter(i))
	{
		logger.log("Knot %u parameters invalid",system::ILogger::ELL_ERROR,i);
		return false;
	}
	return true;
}

inline void CTrueIR::SParameter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	sstr << "\n\t" << selfID << "[label=\"scale = " << std::to_string(scale);
	if (view)
	{
		sstr << "\\nchannel = " << std::to_string(viewChannel);
		const auto& viewParams = view->getCreationParameters();
		sstr << "\\nWraps = {" << wrapU;
		if (viewParams.viewType!=ICPUImageView::ET_1D && viewParams.viewType!=ICPUImageView::ET_1D_ARRAY)
			sstr << "," << wrapV;
		if (viewParams.viewType==ICPUImageView::ET_3D)
			sstr << "," << wrapW;
		// TODO: conditionally print this if wrap modes require it
		sstr << "}\\nBorder = " << borderColor;
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

inline void CTrueIR::SBasicNDFParams::printDot(std::ostringstream& sstr, const core::string& selfID) const
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
struct core::blake3_hasher::update_impl<CTrueIR::SParameter,Dummy>
{
	using input_t = asset::material_compiler3::CTrueIR::SParameter;

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
		hasher << param.linearMagnification;
		bool wrapModeUsesBorder = false;
		auto hashWrap = [&](const hlsl::TextureClamp wrap)->void
		{
			hasher << wrap;
			switch (wrap)
			{
				case hlsl::TextureClamp::ETC_CLAMP_TO_BORDER:
					wrapModeUsesBorder = true;
					break;
				default:
					break;
			}
		};
		using view_type_e = asset::IImageView<asset::ICPUImage>::E_TYPE;
		switch (viewParams.viewType)
		{
			case view_type_e::ET_3D:
				hashWrap(param.wrapW);
				[[fallthrough]];
			case view_type_e::ET_2D: [[fallthrough]];
			case view_type_e::ET_2D_ARRAY: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP_ARRAY:
				hashWrap(param.wrapV);
				[[fallthrough]];
			default:
				hashWrap(param.wrapU);
				break;
		}
		if (wrapModeUsesBorder)
			hasher << param.borderColor;
	}
};
template<uint8_t Count, typename Dummy>
struct core::blake3_hasher::update_impl<CTrueIR::SParameterSet<Count>,Dummy>
{
	using input_t = asset::material_compiler3::CTrueIR::SParameterSet<Count>;

	static inline void __call(blake3_hasher& hasher, const input_t& input)
	{
		bool noTextures = true;
		for (uint8_t i=0; i<Count; i++)
		{
			// fear not, without a view, nothing excpt for scale will hash
			hasher << input.params[i];
			if (input.params[i].view)
				noTextures = false;
		}
		if (noTextures)
			return;
		hasher << input.uvTransform;
		hasher << input.uvSlot();
	}
};
template<typename Dummy>
struct core::blake3_hasher::update_impl<CTrueIR::SBasicNDFParams,Dummy>
{
	using input_t = asset::material_compiler3::CTrueIR::SBasicNDFParams;

	static inline void __call(blake3_hasher& hasher, const input_t& input)
	{
		using type_e = input_t::EParamType;
		const type_e type = input.determineParamType();
		update_impl<uint8_t>::__call(hasher,static_cast<uint8_t>(type));
		update_impl<asset::material_compiler3::CTrueIR::SParameterSet<4>>::__call(hasher,input);
		hasher << input.getDistribution();
		// reference stretch can be applied on non-mapped NDFs too
		if (!input.stretchInvariant())
			hasher << input.reference;
	}
};

} // namespace nbl::asset::material_compiler3

#endif
