// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_


#include "nbl/system/ILogger.h"

#include "nbl/asset/material_compiler3/CNodePool.h"
#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/ICPUImageView.h"


namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material's interface layer, but here they're in "Accumulator Form"
// They appear "flipped upside down", its expected our backends will evaluate contributors first, and then bother with the attenuators. 
class CTrueIR : public CNodePool // TODO: turn into an asset!
{
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
					CParameters_1,
					CParameters_2,
					CParameters_3,
					CParameters_4,
					COrenNayar,
					CCookTorrance,
					CFactorCombiner,
					CConstantFactor
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

			protected:
				friend class CTrueIR;
				inline bool recomputeHash(const obj_pool_type& pool)
				{
					hash = computeHash(pool);
					return hash!=core::blake3_hash_t{};
				}
				virtual bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const = 0;
#define HASH_REQUIREDS_HASH(HANDLE) { \
					if (const auto hash=pool.deref(HANDLE)->getHash(); hash==core::blake3_hash_t{}) \
						return false; \
					else \
						hasher << hash; \
				}
#define HASH_OPTIONALS_HASH(HANDLE) if (HANDLE) {HASH_REQUIREDS_HASH(HANDLE);} else {hasher << core::blake3_hash_t{};}

				// Each node is final and immutable, has a precomputed hash for the whole subtree beneath it.
				// Debug info does not form part of the hash, so can get wildly replaced.
				core::blake3_hash_t hash;
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
		// we step away from our usual way of doing things via linked lists, because this is simpler to rearrange
		class CFactorCombiner final : public obj_pool_type::IVariableSize, public IFactor
		{
			public:
				inline EFinalType getFinalType() const override {return EFinalType::CFactorCombiner;}

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
				inline SState getState() const {return std::bit_cast<SState>(padding);}

				// Only sane child count allowed
				inline typed_pointer_type<const IFactor> getChildHandle(const uint8_t ix)
				{
					if (ix<getState().childCount)
						return child[ix];
					return {};
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

				friend class CTrueIR;
				typed_pointer_type<const IFactor> child[1] = {{}};
		};
#define TYPE_NAME_STR(NAME) "nbl::asset::material_compiler3::CTrueIR::"#NAME
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

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CWeightedContributor);}


				typed_pointer_type<const IContributor> contributor = {};
				// if null then assumed to be 1
				typed_pointer_type<const IFactor> factor = {};
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

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CContributorSum);}


				// the product is ...
				typed_pointer_type<const CWeightedContributor> product = {};
				// the rest node is ...
				_typed_pointer_type<const CContributorSum> rest = {};
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

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCorellatedTransmission);}

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

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrientedLayer);}

				// you can set the children later
				inline COrientedLayer() = default;

				// These are same as the frontend's except that all the etas are oriented and reciprocated
				typed_pointer_type<const CContributorSum> brdfTop = {};
				// this node must be non-null until the last layer
				typed_pointer_type<const CCorellatedTransmission> firstTransmission = {};
		};

		// TODO: break this into UV-sample-able params and regular params
		template<uint8_t Channels>
		class CParameters final : public obj_pool_type::INonTrivial, public INode
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << paramSet;
					return true;
				}

			public:
				inline EFinalType getFinalType() const override
				{
					static_assert(1<=Channels && Channels<=4);
					return static_cast<EFinalType>(static_cast<uint8_t>(EFinalType::CParameters_1)-1+Channels);
				}

				// TODO: improve the token pasting here
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CParameters<Count>);}

				SParameterSet<Channels> paramSet = {};
		};
		
		// Unit Radiance emitter modulated by an IES profile
		class CEmitter final : public obj_pool_type::INonTrivial, public IContributor
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					hasher << profileTransform;
					// we ignore most of the sampler, needs to be set always the same
					const auto& sampler = profile.sampler;
					using namespace ::nbl::asset;
					if (sampler.BorderColor!=ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_TRANSPARENT_BLACK || sampler.MaxFilter!=ISampler::E_TEXTURE_FILTER::ETF_LINEAR)
						return false;
					using clamp_e = hlsl::TextureClamp;
					if (sampler.TextureWrapW!=clamp_e::ETC_CLAMP_TO_EDGE)
						return false;
					// there's a limited set of symmetries we can exploit in our IES tabulations, TODO: check which (probably not REPEAT)
					hasher << profile;
					return true;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CEmitter;}

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
		class IBxDF : public obj_pool_type::INonTrivial, public IContributor
		{
			public:
				inline bool isEmitter() const override {return false;}
		};
		class CDeltaTransmission final : public IBxDF
		{
				// nothing to do
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override {return true;}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CDeltaTransmission;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CDeltaTransmission);}

				inline CDeltaTransmission() = default;
		};
		class IBxDFWithNDF : public IBxDF
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					assert(false); // unimplemented
					return false;
				}

			public:
				// CParameters<4> ?
				//CNodePool::SBasicNDFParams ndfParams;
		};
		class COrenNayar final : public IBxDFWithNDF
		{
			public:
				inline EFinalType getFinalType() const override {return EFinalType::COrenNayar;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrenNayar);}

				inline COrenNayar() = default;
		};
		class CCookTorrance final : public IBxDFWithNDF
		{
			public:
				inline EFinalType getFinalType() const override {return EFinalType::CCookTorrance;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCookTorrance);}

				inline CCookTorrance() = default;
		};
		//! Parameter Nodes
		// ScalarConstant
		// SpectralConstant
		//! Basic factor nodes
		class IFactorLeaf : public IFactor {};
		// TODO use CParameters<1> or CParameters<3> + colorpsace semantics (part of `CSpectralVariable`)
		class CConstantFactor final : public obj_pool_type::INonTrivial, public IFactorLeaf
		{
				inline bool computeHash_impl(const obj_pool_type& pool, core::blake3_hasher& hasher) const override
				{
					assert(false);// TODO: hash the parameter
					return false;
				}

			public:
				inline EFinalType getFinalType() const override {return EFinalType::CConstantFactor;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CConstant);}

				// you can set the children later
				inline CConstantFactor() = default;
		};
#undef TYPE_NAME_STR
#undef HASH_THE_HASH
		
		//
		struct SBasicNodes final
		{
			public:
				inline bool operator==(const SBasicNodes& other) const = default;

				typed_pointer_type<const CContributorSum> blackHoleBxDF = {};
				typed_pointer_type<const CContributorSum> errorBxDF = {};

			private:
				friend class CTrueIR;
				NBL_API2 SBasicNodes(CTrueIR* ir);
		};
		const SBasicNodes& getBasicNodes() const {return m_basicNodes;}

		//
//		inline typed_pointer_type<const CConstant> createConstant()


		// Each material comes down to this, this is the only struct we don't de-duplicate
		struct SMaterial
		{
			// Stats needed by a renderer to skip loading parts of a material or remove expensive code altogether
			enum class EMetadataBits : uint16_t
			{
				None = 0,
				// if any such contributor present
				NotBlackhole = 0x1u<<0, // actually have a material
				NonDelta = 0x1u<<1, // can evaluate against point lights (or other samplings)
				DeltaTransmissive = 0x1u<<2, // can use stochastic transparency for closest hit rays and blending for anyhit 
				Emissive = 0x1u<<3, // maybe register for NEE, but definitely grab the emission
				NonSpatiallyVaryingEmissive = 0x1u<<4, // definitely register for NEE
				// TODO: 5,6 left
				// Bits that help us remove expensive code from impl
				DerivativeMap = 0x1u<<7,
				DirectionallyVaryingEmissive = 0x1u<<8, // IES profile
				SpatiallyVaryingEmissive = 0x1u<<9, // textured light
				Lambertian = 0x1u<<10,
				OrenNayar = 0x1u<<11,
				GGX = 0x1u<<12,
				AnisotropicGGX = 0x1u<<13,
				Beckmann = 0x1u<<14,
				AnisotropicBeckmann = 0x1u<<15,
			};
			//
			struct SOriented
			{
				// null means no material
				typed_pointer_type<const COrientedLayer> root = {};
				//
				constexpr static inline uint8_t MaxUVSlots = 32;
				std::bitset<MaxUVSlots> usedUVSlots = {};
				// the tangent frames are a subset of used UV slots, unless there's an anisotropic BRDF involved
				std::bitset<MaxUVSlots> usedTangentFrames = {};
				//
				core::bitflag<EMetadataBits> metadata = {};
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
				inline SDotPrinter(const CTrueIR* ir) : m_ir(ir) {}
				// assign in reverse because we want materials to print in order
				inline SDotPrinter(const CTrueIR* ir, std::span<const SMaterial> roots) : m_ir(ir)//, layerStack(roots.rbegin(),roots.rend())
				{
					// should probably size it better, if I knew total node count allocated or live
					visitedNodes.reserve(roots.size()<<4);
				}

				NBL_API2 void operator()(std::ostringstream& output);
				inline core::string operator()()
				{
					std::ostringstream tmp;
					operator()(tmp);
					return tmp.str();
				}
			
				core::unordered_set<typed_pointer_type<const INode>> visitedNodes;
#if 0
				// TODO: track layering depth and indent accordingly?
				core::vector<typed_pointer_type<const CLayer>> layerStack;
				core::stack<typed_pointer_type<const IExprNode>> exprStack;
#endif
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
				printMatrix(sstr,*reinterpret_cast<const decltype(_set.uvTransform)*>(_set.params+_count));
				sstr << "\"]";
				sstr << "\n\t" << selfID << " -> " << uvTransformID << "[label=\"UV Transform\"]";
			}
		}

	protected:
		NBL_API2 CTrueIR(creation_params_type&& params);

		// copies from other IR into ours and makes sure things get hashed properly
		NBL_API2 bool rewrite(SMaterial& material, CTrueIR* srcIR);

		core::vector<SMaterial> m_materials;
		// TODO: either we put the typeid in the hash, or we have a type of hashmap per type
		core::unordered_map<core::blake3_hash_t,typed_pointer_type<const INode>> m_uniqueNodes;
		friend struct SBasicNodes;
		const SBasicNodes m_basicNodes;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(CTrueIR::SMaterial::EMetadataBits)

inline void CTrueIR::SParameter::printDot(std::ostringstream& sstr, const core::string& selfID) const
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
		const auto& sampler = param.sampler;
		hasher << sampler.BorderColor;
		hasher << sampler.MaxFilter;
		using view_type_e = asset::IImageView<asset::ICPUImage>::E_TYPE;
		switch (viewParams.viewType)
		{
			case view_type_e::ET_3D:
				hasher << sampler.TextureWrapW;
				[[fallthrough]];
			case view_type_e::ET_2D: [[fallthrough]];
			case view_type_e::ET_2D_ARRAY: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP: [[fallthrough]];
			case view_type_e::ET_CUBE_MAP_ARRAY:
				hasher << sampler.TextureWrapV;
				[[fallthrough]];
			default:
				hasher << sampler.TextureWrapU;
				break;
		}
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
struct core::blake3_hasher::update_impl<CTrueIR::SBasicNDFParams,Dummy>
{
	using input_t = asset::material_compiler3::CTrueIR::SBasicNDFParams;

	static inline void __call(blake3_hasher& hasher, const input_t& input)
	{
		using type_e = input_t::EParamType;
		const type_e type = input.determineParamType();
		update_impl<uint8_t>::__call(hasher,static_cast<uint8_t>(type));
		update_impl<asset::material_compiler3::CTrueIR::SParameterSet<4>>::__call(hasher,*this);
		// reference stretch can be applied on non-mapped NDFs too
		if (!input.stretchInvariant())
			hasher << input.reference;
	}
};

} // namespace nbl::asset::material_compiler3

#endif
