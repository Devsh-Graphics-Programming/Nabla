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
			// shadow comparison functions are ignored
			ICPUSampler::SParams sampler;
		};
		// In the forest, this is not a node, we'll deduplicate later
		template<uint8_t Count>
		struct SParameterSet
		{
			SParameter params[Count];
			// identity transform by default, ignored if no UVs
			hlsl::float32_t2x3 uvTransform = hlsl::float32_t2x3(
				1,0,0,
				0,1,0
			);

			// Ignored if no modulator textures
			uint8_t& uvSlot() {return params[0].padding[0];}
			const uint8_t& uvSlot() const {return params[0].padding[0];}
			// Note: the padding abuse
			static_assert(sizeof(SParameter::padding)>0);
		};

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				// Only sane child count allowed
				virtual uint8_t getChildCount() const = 0;
				inline TypedHandle<INode> getChild(const uint8_t ix) const
				{
					if (ix<getChildCount())
						return getChild_impl(ix);
					return {};
				}

				inline bool isContributorLeafAllowedInSubtree(const uint8_t ix) const
				{
					if (ix<getChildCount())
						return isContributorLeafAllowedInSubtree_impl(ix);
					return false;
				}

				CNodePool::TypedHandle<CNodePool::CDebugInfo> debugInfo;

			protected:
				virtual inline TypedHandle<INode> getChild_impl(const uint8_t ix) const {return {};}
				// by default we don't allow ContributorLeafs in subtrees, except on special nodes
				virtual inline bool isContributorLeafAllowedInSubtree_impl(const uint8_t ix) const {return false;}
		};
		template<typename T> requires std::is_base_of_v<INode,T>
		using TypedHandle = CNodePool::TypedHandle<T>;

		// This node could also represent non directional emission, but we have another node for that
		class CSpectralVariable final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CSpectralVariable";}
				// Variable length but has no children
				inline uint8_t getChildCount() const override {return 0;}

				enum class Semantics : uint8_t
				{
					// 3 knots, their wavelengths are implied and fixed at color primaries
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
					// Knots are "data points" on the (wavelength,value) plot, from which we can interpolate the rest of the spectrum
					SParameterSet<Count> knots = {};

					// a little bit of abuse and padding reuse
					static_assert(sizeof(SParameter::padding)>2);
					template<bool Enable=true> requires (Enable==(Count>1))
					Semantics& getSemantics() {return reinterpret_cast<Semantics&>(knots.params[0].padding[2]); }
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
					static_assert(sizeof(SParameter::padding)>1);
					return paramsBeginPadding()[1];
				}
				inline uint32_t getSize() const override
				{
					return sizeof(CSpectralVariable)+sizeof(SCreationParams<1>)+(getKnotCount()-1)*sizeof(SParameter);
				}

				template<uint8_t Count>
				inline CSpectralVariable(SCreationParams<Count>&& params)
				{
					// back up the count
					params.knots.params[0].padding[1] = Count;
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

			private:
				const uint8_t* paramsBeginPadding() const {return reinterpret_cast<const SCreationParams<1>*>(this+1)->knots.params[0].padding;}
		};
		//
		class IUnaryOp : public INode
		{
			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override final {return child;}
				
			public:
				inline uint8_t getChildCount() const override final {return 1;}

				TypedHandle<INode> child = {};
		};
		class IBinOp : public INode
		{
			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override final {return ix ? rhs:lhs;}
				
			public:
				inline uint8_t getChildCount() const override final {return 2;}

				TypedHandle<INode> lhs = {};
				TypedHandle<INode> rhs = {};
		};
		//! Basic combiner nodes
		class CMul final : public IBinOp
		{
			protected:
				//! NOTE: Only the "left" child subtree is allowed to contain ContributorLeafs so we don't multiply them together!
				inline bool isContributorLeafAllowedInSubtree_impl(const uint8_t ix) const override {return ix==0;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CMul";}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CMul);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CMul() = default;
		};
		class CAdd final : public IBinOp
		{
			protected:
				inline bool isContributorLeafAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CAdd";}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CAdd);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CAdd() = default;
		};
		// does `1-expression`
		class CComplement final : public IUnaryOp
		{
			protected:
				// TODO: explain why
				inline bool isContributorLeafAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CComplement";}

				// you can set the children later
				static inline uint32_t calc_size() { return sizeof(CComplement); }
				inline uint32_t getSize() const override { return calc_size(); }
				inline CComplement() = default;
		};
		//! Base class for leaf node quantities which contribute additively to the Lighting Integral
		class IContributorLeaf : public INode {};
		//! Basic Emitter
		class CEmitter final : public IContributorLeaf
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
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override {return radiance;}
				// not overriding the `inline bool isContributorLeafAllowedInSubtree_impl` the child is strongly typed
		};
		//! Special nodes meant to be used as `CMul::rhs`, as for the `N`, they use the normal used by the Leaf ContributorLeafs in its MUL node relative subgraph.
		//! However if the Leaf BXDF is Cook Torrance, the microfacet `H` normal will be used instead.
		//! If there are two BxDFs with different normals, theese nodes get split and duplicated into two in our Final IR.
		//! ----------------------------------------------------------------------------------------------------------------
		// Beer's Law Node, assumes entry and exit through the same microfacet if microfacets are used, can only be used in:
		// - Cook Torrance BTDF expressions (because we're modelling distance to next interface)
		// - not last material layer
		class CBeer final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CBeer";}
				inline uint8_t getChildCount() const override {return 1;}

				// you can set the members later
				static inline uint32_t calc_size() {return sizeof(CBeer);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CBeer() = default;

				// Effective transparency = exp2(log2(perpTransparency)/dot(refract(V,X,eta),X)) = exp2(log2(perpTransparency)*inversesqrt(1.f+(VdotX-1)*rcpEta))
				// Absorption and thickness of the interface combined into a single variable, eta is taken from the leaf BTDF node.
				TypedHandle<CSpectralVariable> perpTransparency = {};

			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override {return perpTransparency;}
				// not overriding the `inline bool isBxDFAllowedInSubtree_impl` the child is strongly typed
		};
		// The "oriented" in the Etas means from frontface to backface, so there's no need to reciprocate them when creating matching BTDF for BRDF
		class CFresnel final : public INode
		{
			public:
				inline uint8_t getChildCount() const override {return 2;}

				inline const std::string_view getTypeName() const override {return "nbl::CFresnel";}
				static inline uint32_t calc_size()
				{
					return sizeof(CFresnel);
				}
				inline CFresnel() = default;

				// Already pre-divided Index of Refraction, e.g. exterior/interior since VdotG>0 the ray always arrives from the exterior.
				TypedHandle<CSpectralVariable> orientedRealEta = {};
				// Specifying this turns your Fresnel into a conductor one
				TypedHandle<CSpectralVariable> orientedImagEta = {};
				// if you want to reuse the same parameter but want to flip the interfaces around
				uint8_t reciprocateEtas : 1 = false;

			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override {return ix ? orientedImagEta:orientedRealEta;}
				// not overriding the `inline bool isBxDFAllowedInSubtree_impl` the child is strongly typed
		};
		//! Basic BxDF nodes
		// Every BxDF leaf node is supposed to pass WFT test, color and extinction is added on later via multipliers
		class IBxDF : public IContributorLeaf
		{
			public
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
		// - Smooth Conductor -> above multiplied with Conductor-Fresnel
		// - Smooth Dielectric -> Any Cook Torrance BxDF with roughness=0 attached as BRDF and BTDF multiplied with Dielectric-Fresnel (no imaginary component)
		// - Thindielectric -> Any Cook Torrance BxDF multiplied with Dielectric-Fresnel as BRDF and BTDF layering an identical Compound but with reciprocal Etas
		// - Plastic -> Above but we layer over a Diffuse BRDF instead.
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
				inline uint8_t getChildCount() const override {return 1;}

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
				// BRDFs and BTDFs components into separate expressions, and also importance sample much better, for details see comments in CTrueIR. 
				TypedHandle<CSpectralVariable> orientedRealEta;
				// 
				NDF ndf = NDF::GGX;

			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override {return orientedRealEta;}
		};
		// All layers are modelled as coatings, most combinations are not feasible and what combos are feasible depend on the compiler backend you use.
		// We specifically disregard the layering proposed by Weidlich and Wilkie 2007 because it leads to pathologies like the glTF Specular+Diffuse model,
		// which is the assumption that light entering through a specular microfacet must also leave through it because microfacets are assumed to be much larger
		// than the thickness of the layer between the materials (this breaks foundational assumption of the facets being "micro" and multiple within same pixel/ray footprint).
		// One CANNOT model layering through Cook Torrance BSDFs by only considering interactions with the microfacet aligning perfectly with the half-way vector.
		// Do not use Coatings for things which can be achieved with linear blends! (e.g. alpha transparency)
		class CLayer final : public INode
		{
			protected:
				inline TypedHandle<INode> getChild_impl(const uint8_t ix) const override {return ix ? (ix!=1 ? coated:btdf):brdf;}
				inline bool isBxDFAllowedInSubtree_impl(const uint8_t ix) const override {return true;}

			public:
				inline const std::string_view getTypeName() const override {return "nbl::CLayer";}
				inline uint8_t getChildCount() const override {return 3;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CLayer);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CLayer() = default;

				// Whether the layer is a BSDF depends on having a non-null 2nd child (a transmission component)
				// A whole material is a BSDF iff. all layers have a non-null BTDF
				inline bool isBSDF() const {return bool(btdf);}

				// a null BRDF will not produce reflections, while a null BTDF will not allow any transmittance.
				TypedHandle<INode> brdf;
				TypedHandle<INode> btdf;
				// This layer can only coat another if its a BSDF, so coatees must have only layers with non-null BTDFs above them
				TypedHandle<CLayer> coated;
		};

		// Each material comes down to this
		inline std::span<const TypedHandle<CLayer>> getMaterials() const {return m_rootNodes;}
		inline bool addMaterial(const TypedHandle<CLayer>& rootNode)
		{
			if (valid(rootNode))
				m_rootNodes.push_back(rootNode);
		}

		// To quickly make a matching backface material from a frontface or vice versa
		NBL_API TypedHandle<CLayer> reciprocate(const TypedHandle<CLayer> other);

		// IMPORTANT: Two BxDFs are not allowed to be multiplied together.
		// NOTE: Right now all Spectral Variables are required to be Monochrome or 3 bucket fixed semantics, all the same wavelength.
		// There are certain things we're unable to check, like whether reciprocity is obeyed, as you're supposed to create separate materials
		// for a front-face and a back-face (with pre-divided IORs as oriented etas).
		NBL_API bool valid(const TypedHandle<CLayer>& rootNode) const;
		// TODO: do a child validation thing, certain nodes need particular types of children

	protected:
		using CNodePool::CNodePool;

		core::vector<TypedHandle<CLayer>> m_rootNodes;
};

//! DAG (baked)

} // namespace nbl::asset::material_compiler3

#endif