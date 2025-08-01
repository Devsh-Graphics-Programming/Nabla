// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CNodePool.h"

#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/ICPUImageView.h"



// temporary
#define NBL_API

namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material
// The materials are defined differently for front faces and back faces, you can be sure that GdotV>0 where G is the geometric normal.
// Basically V is always in the upper hemisphere, we can deduce transmission by just looking at the sign of GdotL.
// Because we implement Schussler et. al 2017 we also ensure that dot products with shading normals are identical to smooth normals.
// However the smooth normals are not identical to geometric normals, we reserve the right to use the "normal pull up trick" to make them consistent.
class CFrontendIR : public CNodePool
{
	public:
		// constructor
		inline core::smart_refctd_ptr<CFrontendIR> create(const uint8_t chunkSizeLog2=19, const uint8_t maxNodeAlignLog2=4, refctd_pmr_t&& _pmr={})
		{
			if (chunkSizeLog2<14 || maxNodeAlignLog2<4)
				return nullptr;
			if (!_pmr)
				_pmr = core::getDefaultMemoryResource();
			return core::smart_refctd_ptr<CFrontendIR>(new CFrontendIR(chunkSizeLog2,maxNodeAlignLog2,std::move(_pmr)),core::dont_grab);
		}
		
		struct SParameter
		{
			inline operator bool() const
			{
				return abs(scale)<std::numeric_limits<float>::infinity() && (!view || viewChannel<getFormatChannelCount(view->getCreationParameters().format));
			}

			// at this stage we store the multipliers in highest precision
			float scale = 1.f;
			// rest are ignored if the view is null
			uint8_t viewChannel : 2 = 0;
			uint8_t padding[3] = {0,0,0};
			core::smart_refctd_ptr<const ICPUImageView> view = {};
			// compare functions are ignored
			ICPUSampler::SParams sampler;
		};
		// in the forest, its not a node, we'll deduplicate later
		template<uint8_t Count>
		struct SParameterSet
		{
			SParameter params[Count];
			// identity transform by default, ignored if no UVs
			hlsl::float32_t2x3 uvTransform = hlsl::float32_t2x3(
				1,0,0,
				0,1,0
			);
			// ignored if no modulator textures
			uint8_t uvSlot = 0;
			uint8_t padding[3] = {0,0,0};
		};

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				inline bool isBxDFAllowedInSubtree(const uint8_t ix) const
				{
					if (ix<getChildCount())
						return isBxDFAllowedInSubtree_impl(ix);
					return false;
				}

				CNodePool::TypedHandle<CNodePool::CDebugInfo> debugInfo;

			protected:
				//
				virtual inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const {return false;}
		};
		template<typename T> requires std::is_base_of_v<INode, T>
		using TypedHandle = CNodePool::TypedHandle<T>;

		// This node could also represent non directional emission, but we have another node for that
		class CSpectralVariable final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CSpectralVariable";}
				// Variable length but has no children
				uint8_t getChildCount() const override {return 0;}

				enum class Semantics : uint8_t
				{
					// 3 knots, they're fixed at color primaries
					Fixed3_SRGB = 0,
					Fixed3_DCI_P3 = 1,
					Fixed3_BT2020 = 2,
					Fixed3_AdobeRGB = 3,
					Fixed3_AcesCG = 4,
					// Ideas: each node is described by (wavelength,value) pair
					// PairsLinear = 5, // linear interpolation
					// PairsLogLinear = 5, // linear interpolation in wavelenght log space
				};
				template<uint8_t Count>
				struct SCreationParams
				{
					SParameterSet<Count> knots = {};

					// a little bit of abuse and padding reuse
					template<bool Enable=true> requires (Enable==(Count>1))
					Semantics& getSemantics() {return reinterpret_cast<Semantics&>(nodes.params[1].padding[0]); }
					template<bool Enable=true> requires (Enable==(Count>1))
					const Semantics& getSemantics() const {return const_cast<const Semantics&>(const_cast<CSpectralVariable*>(this)->getSemantics());}
				};
				template<uint8_t Count>
				static inline uint32_t calc_size(const SCreationParams<Count>&)
				{
					return sizeof(CSpectralVariable)+sizeof(SCreationParams<Count>);
				}
				
				inline uint8_t getKnotCount() const
				{
					return reinterpret_cast<const SCreationParams<1>*>(this+1)->knots.params[0].padding[0];
				}
				inline uint32_t getSize() const override
				{
					auto pWonky = reinterpret_cast<const SCreationParams<1>*>(this+1);
					return calc_size(*pWonky)+(getKnotCount()-1)*sizeof(SParameter);
				}

				template<uint8_t Count>
				inline CSpectralVariable(SCreationParams<Count>&& params)
				{
					// back up the count
					params.knots.params[0].padding[0] = Count;
					std::construct_at(reinterpret_cast<SCreationParams<Count>*>(this+1),std::move(params));
				}

				inline operator bool() const
				{
					auto pWonky = reinterpret_cast<const SCreationParams<1>*>(this+1);
					for (auto i=0u; i<getKnotCount(); i++)
					if (!pWonky->knots.params[i])
						return false;
					return true;
				}

			protected:
				inline ~CSpectralVariable()
				{
					auto pWonky = reinterpret_cast<SCreationParams<1>*>(this+1);
					std::destroy_n(pWonky->knots.params,getKnotCount());
				}
		};
		//! Basic combiner nodes
		class CMul final : public INode
		{
			protected:
				//! NOTE: Only the "left" child subtree is allowed to contain BxDFs so we don't multiply them together!
				inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const override {return ix==0;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CMul";}
				inline uint8_t getChildCount() const override {return 2;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CMul);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CMul() = default;
		};
		class CAdd final : public INode
		{
			protected:
				inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CAdd";}
				inline uint8_t getChildCount() const override {return 2;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CAdd);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CAdd() = default;
		};
		// does `1-expression`
		class CComplement final : public INode
		{
			protected:
				inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CComplement";}
				inline uint8_t getChildCount() const override {return 1;}

				// you can set the children later
				static inline uint32_t calc_size() { return sizeof(CComplement); }
				inline uint32_t getSize() const override { return calc_size(); }
				inline CComplement() = default;
		};
		//! Basic Emitter
		class CEmitter final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CEmitter";}
				inline uint8_t getChildCount() const override {return 1;}

				// you can set the members later
				static inline uint32_t calc_size() {return sizeof(CEmitter);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CEmitter() = default;

				TypedHandle<CSpectralVariable> radiance;
				// This can be anything like an IES profile, if invalid, there's no directionality to the emission
				SParameter profile;
				hlsl::float32_t3x3 profileTransform = hlsl::float32_t3x3(
					1,0,0,
					0,1,0,
					0,0,1
				);
				// TODO: semantic flags/metadata (symmetries of the profile)

			protected:
				// not overriding the `inline bool isBxDFAllowedInSubtree_impl` the child is strongly typed
				inline Handle* getChildHandleStorage(const int16_t ix) override
				{
					return ix==0 ? (&radiance.untyped):nullptr;
				}
		};
		//! Special mul nodes meant to be used with Mul, as for the `N`, they uses the normal used by the Leaf BxDFs in its MUL node relative subgraph.
		//! Furthermore if the Leaf BXDF is Cook Torrance, the microfacet `H` normal will be used.
		//! If there are two BxDFs with different normals, theese nodes get split and duplicated into two in our final IR.
		// Assumes entry and exit through the same microfacet if microfacets are used, can only be used in:
		// - Cook Torrance BTDF expressions (because we're modelling distance to next interface)
		// - not last material layer
		class CBeer final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CBeer";}
				static inline uint32_t calc_size()
				{
					return sizeof(CBeer);
				}
				inline uint8_t getChildCount() const override {return 1;}

				// Effective transparency = exp2(log2(perpTransparency)/dot(refract(V,X,eta),X)) = exp2(log2(perpTransparency)*inversesqrt(1.f+(VdotX-1)*rcpEta)
				// Absorption and thickness of the interface combined into a single variable, eta is taken from the leaf BTDF node.
				TypedHandle<CSpectralVariable> perpTransparency;
		};
		// For BTDFs the fresnel is automatically turned into its complement and the eta is re-oriented when making the true IR
		class CFresnel final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CFresnel";}
				static inline uint32_t calc_size()
				{
					return sizeof(CFresnel);
				}
				inline uint8_t getChildCount() const override {return 2;}

				// Already pre-divided Index of Refraction, e.g. exterior/interior since VdotG>0 the ray always arrives from the exterior.
				TypedHandle<CSpectralVariable> orientedRealEta;
				// Specifying this turns your Fresnel into a conductor one
				TypedHandle<CSpectralVariable> orientedImagEta;
				// if you want to reuse the same parameter but want to flip the interfaces around
				uint8_t reciprocateEtas : 1 = false;
		};
		//! Basic BxDF nodes
		// Every BxDF leaf node is supposed to pass WFT test, color and extinction is added on later via multipliers
		class IBxDF : public INode
		{
			public:
				inline uint8_t getChildCount() const override final {return 0;}

				// Why are all of these kept together and forced to fetch from the same UV ?
				// Because they're supposed to be filtered together with the knowledge of the NDF
				// TODO: should really be 5 parameters (2+3) cause of rotatable anisotropic roughness
				struct SBasicNDFParams : SParameterSet<4>
				{
					inline auto getDerivMap() {return std::span<SParameter,2>(params,2);}
					inline auto getDerivMap() const {return std::span<const SParameter,2>(params,2);}
					inline auto getRougness() {return std::span<SParameter,2>(params+2,2);}
					inline auto getRougness() const {return std::span<const SParameter,2>(params+2,2);}

					// whether the derivative map and roughness is constant regardless of UV-space texture stretching
					inline bool stretchInvariant() const {return !(abs(hlsl::determinant(reference))>std::numeric_limits<float>::min());}

					// Ignored if not invertible, otherwise its the reference "stretch" (UV derivatives) at which identity roughness and normalmapping occurs
					hlsl::float32_t2x2 reference = hlsl::float32_t2x2(0,0,0,0);
				};

				// For Schussler et. al 2017 we'll spawn 2-3 additional BRDF leaf nodes in the proper IR for every normalmap present
		};
		// Only Special Node, because of how its useful for compiling Anyhit shaders, the rest can be done easily
		// - Delta Reflection -> Any Cook Torrance BxDF with roughness=0 attached as BRDF
		// - Smooth Conductor -> above multiplied with Conductor-Fresnel computed with N (more efficient to importance sample than with H despite same result)
		// - Smooth Dielectric -> Any Cook Torrance BxDF with roughness=0 attached as BRDF and BTDF
		// - Thindielectric -> Any Cook Torrance BxDF multiplied with Dielectric-Fresnel as BRDF and BTDF layering an identical Compound but with reciprocal Etas
		// - Plastic -> Above but we layer a Diffuse BRDF instead.
		class CDeltaTransmission final : public IBxDF
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CDeltaTransmission";}
				static inline uint32_t calc_size()
				{
					return sizeof(CDeltaTransmission);
				}
				uint32_t getSize() const override {return calc_size();}
		};
		// Because of Schussler et. al 2017 every one of these nodes splits into 2 (if no L dependence) or 3 during canonicalization
		class COrenNayar final : public IBxDF
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::COrenNayar";}
				static inline uint32_t calc_size()
				{
					return sizeof(COrenNayar);
				}
				uint32_t getSize() const override {return calc_size();}

				SBasicNDFParams ndParams;
		};
		// Supports anisotropy for all models
		class CCookTorrance final : public IBxDF
		{
			public:
				enum class NDF : uint8_t
				{
					GGX = 0,
					Beckmann = 1
				};

				inline const std::string_view getTypeName() const override {return "nbl::CCookTorrance";}
				static inline uint32_t calc_size()
				{
					return sizeof(CCookTorrance);
				}
				uint32_t getSize() const override {return calc_size();}

				SBasicNDFParams ndParams;
				// We need this eta to compute the refractions of `L` when importance sampling and the Jacobian during H to L generation for rough dielectrics
				// It does not mean we compute the Fresnel weights though! You might ask why we don't do that given that state of the art importance sampling
				// (at time of writing) is to decide upon reflection vs. refraction after the microfacet normal `H` is already sampled,
				// producing an estimator with just Masking and Shadowing function ratios. The reason is because we can simplify our IR by separating out
				// BRDFs and BTDFs components into separate expressions, and also importance sample much better, for details see commens in CTrueIR. 
				TypedHandle<CSpectralVariable> orientedRealEta;
				// 
				NDF ndf = NDF::GGX;
		};
		// All layers are modelled as coatings, most combinations are not feasible and what combos are feasible depend on the compiler backend you use.
		// We specifically disregard the layering proposed by Weidlich and Wilkie 2007 because it leads to pathologies like the glTF Spoecular+Diffuse model.
		// One CANNOT model layering through Cook Torrance BSDFs by only considering interactions with the microfacet aligning perfectly with the half-way vector.
		// Do not use Coatings for things which can be achieved with linear blends! (e.g. alpha transparency)
		class CLayer final : public INode
		{
			protected:
				inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CLayer";}
				// first child is BRDF, second is a BTDF [optional]
				// a null BRDF will not produce reflections, while a null BTDF will not allow any transmittance.
				inline uint8_t getChildCount() const override {return 2;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CLayer);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CLayer() = default;

				// Whether the layer is a BSDF depends on having a no-null 2nd child (a transmission component)
				// A whole material is a BSDF iff. all layers have a non-null BTDF
				inline bool isBSDF() const {return ;}

				// This layer can only coat another if its a BSDF (so coatees must have only BSDF layers above them)
				TypedHandle<CLayer> coated;
		};

		// Each material comes down to this
		inline std::span<const Handle> getMaterials() const {return m_rootNodes;}
		inline bool addMaterial(const Handle& rootNode)
		{
			if (valid(rootNode))
				m_rootNodes.push_back(rootNode);
		}

		// IMPORTANT: Two BxDFs are not allowed to be multiplied together.
		// NOTE: Right now all Spectral Variables are required to be Monochrome or 3 bucket fixed semantics, all the same wavelength.
		// There are certain things we're unable to check, like whether reciprocity is obeyed, as you're supposed to create
		// separate materials for a front-face and a back-face (with pre-divided IORs as oriented etas).
		NBL_API bool valid(const Handle& rootNode) const;
		// TODO: do a child validation thing, certain nodes need particular types of children

	protected:
		using CNodePool::CNodePool;

		core::vector<Handle> m_rootNodes;
};

//! DAG (baked)

} // namespace nbl::asset::material_compiler3

#endif