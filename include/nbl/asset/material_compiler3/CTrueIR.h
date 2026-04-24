// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_TRUE_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CFrontendIR.h"


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

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				const auto& getHash() const {return hash;}

				// only call once the nodes underneath are linked up, returning empty hash means error/invalid node
				virtual core::blake3_hash_t computeHash() const = 0;

//				virtual void printDot(std::ostringstream& sstr, const core::string& selfID) const = 0;

			protected:
				inline bool recomputeHash()
				{
					hash = computeHash();
					return hash!=core::blake3_hash_t{};
				}

				// Each node is final and immutable, has a precomputed hash for the whole subtree beneath it.
				// Debug info does not form part of the hash, so can get wildly replaced.
				core::blake3_hash_t hash;
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = _typed_pointer_type<T>;
		// Contributors are things which can either be importance sampled to continue a path or impart a contribution ot the integral
		class IContributor : public INode
		{
			public:
				// ?
		};
		//
		class IExprNode : public INode
		{
			public:
				// Only sane child count allowed
				virtual uint8_t getChildCount() const = 0;
				inline _typed_pointer_type<IExprNode> getChildHandle(const uint8_t ix)
				{
					if (ix<getChildCount())
						return getChildHandle_impl(ix);
					return {};
				}
				inline _typed_pointer_type<const IExprNode> getChildHandle(const uint8_t ix) const
				{
					auto retval = const_cast<IExprNode*>(this)->getChildHandle(ix);
					return retval;
				}

				// A "contributor" of a term to the lighting equation: a BxDF (reflection or tranmission) or Emitter term
				// Contributors are not allowed to be multiplied together, but every additive term in the Expression must contain a contributor factor.
				enum class Type : uint8_t
				{
					Add = 0,
					Mul = 1,
					Other = 2
				};
				virtual inline Type getType() const {return Type::Other;}
				
			protected:
				// child managment
				virtual inline _typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const {assert(false); return {};}
		};
#define TYPE_NAME_STR(NAME) "nbl::asset::material_compiler3::CTrueIR::"#NAME
		// Note that this is not a root node, its a flipped leaf!
		class CWeightedContributor final : public obj_pool_type::INonTrivial, public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CWeightedContributor);}

				NBL_API2 core::blake3_hash_t computeHash() const;


				typed_pointer_type<const IContributor> contributor = {};
				// if null then assumed to be 1
				typed_pointer_type<const IExprNode> factor = {};
		};
		// One BRDF or BTDF component of a layer is represented as
		// 	   f(w_i,w_o) = Sum_i^N Product_j^{N_i} h_{ij}(w_i,w_o) l_i(w_i,w_o)
		class CContributorSum final : public obj_pool_type::INonTrivial, public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CContributorSum);}

				// CANONICALIZATION NOTE: The conntributors shall be ordered in following order:
				// - according to their type, emitters first, then BxDFs, within those
				//	+ Emitters with IES profile, then without 
				//	+ BxDFs by their type
				//		* within the same type, by the parameters (with/without bump, then rest
				// - all things being equal, order by hash
				NBL_API2 core::blake3_hash_t computeHash() const;


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
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCorellatedTransmission);}

				NBL_API2 core::blake3_hash_t computeHash() const;

				// you can set the children later
				inline CCorellatedTransmission() = default;

				// to get to the coated layer we must transmit through
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
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrientedLayer);}

				NBL_API2 core::blake3_hash_t computeHash() const;

				// you can set the children later
				inline COrientedLayer() = default;

				// These are same as the frontend's except that all the etas are oriented and reciprocated
				typed_pointer_type<const CContributorSum> brdfTop = {};
				// this node must be non-null until the last layer
				typed_pointer_type<const CCorellatedTransmission> firstTransmission = {};
		};

// TODO: Parameter Node
		
		// Unit Radiance emitter modulated by an IES profile
		class CEmitter final : public obj_pool_type::INonTrivial, public IContributor
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CEmitter);}

				NBL_API2 core::blake3_hash_t computeHash() const;

				// you can set the members later
				inline CEmitter() = default;

#if 0 // TODO: share with AST ? No parameter needs to be a node (so we can dedup and hash)!
				// This can be anything like an IES profile, if invalid, there's no directionality to the emission
				// `profile.scale` can still be used to influence the light strength without influencing NEE light picking probabilities
				SParameter profile = {};
				hlsl::float32_t3x3 profileTransform = hlsl::float32_t3x3(
					1,0,0,
					0,1,0,
					0,0,1
				);
				// TODO: semantic flags/metadata (symmetries of the profile)
#endif
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
		class IBxDF final : public obj_pool_type::INonTrivial, public IContributor
		{
			public:
				// TODO:
				// - Share ParamSet with AST
				// - Share SBasicNDFParams with AST
				// - hash NDF Params
				// - maybe share IBxDF
				// - and base BxDFs ?
		};


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
		// Returns indices into `this->getMaterials()` for every `forest->getMaterials()`
		// We take the trees from the forest, and canonicalize them into our weird Domain Specific IR with Upside down expression trees.
		// Process:
		// 1. Decompression (duplicating nodes, etc.)
		// 2. Canonicalize Expressions (Transform into Sum-Product form, DCE, etc.)
		// 3. Split BTDFs (front vs. back part), reciprocate Etas
		// 4. Simplify and Hoist Layer terms (delta sampling property)
		// 5. Subexpression elimination
		// It is the backend's job to handle:
		// - constant encoding precision (scale factors, UV matrices, IoRs)
		// - multiscatter compensation
		// - compilation failure to unsupported complex layering
		// - compilation failure to unsupported complex layering
		struct SAddMaterialsArgs
		{
			explicit inline operator bool() const {return forest;}

			const CFrontendIR* forest;
			system::logger_opt_ptr logger;

		};
		NBL_API2 core::vector<SMaterialHandle> addMaterials(const SAddMaterialsArgs& args);
		

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

	protected:
		using CNodePool::CNodePool;

		core::vector<SMaterial> m_materials;
		core::unordered_map<core::blake3_hash_t,typed_pointer_type<INode>> m_uniqueNodes;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(CTrueIR::SMaterial::EMetadataBits)

} // namespace nbl::asset::material_compiler3

#endif
