// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_


#include "nbl/system/ILogger.h"

#include "nbl/asset/material_compiler3/CNodePool.h"
#include "nbl/asset/format/EColorSpace.h"
#include "nbl/asset/ICPUImageView.h"



// temporary
#define NBL_API

namespace nbl::asset::material_compiler3
{

// You make the Materials with a classical expression IR, one Root Node per material.
// 
// Materials form a Layer Stack, each layer is statistically uncorrelated unlike `Weidlich, A., and Wilkie, A. Arbitrarily layered micro-facet surfaces` 2007
// we don't require that for a Microfacet Cook Torrance layer the ray must enter and exist through the same microfacet. Such an assumption only helps if you
// plan on ignoring every transmission through the microfacets within the statistical pixel footprint as given by the VNDF except the perfectly specular one.
// The energy loss from that leads to pathologies like the glGTF Specular+Diffuse model, comparison: https://x.com/DS2LightingMod/status/1961502201228267595
// 
// There's an implicit Top and Bottom on the layer stack, but thats only for the purpose of interpreting the Etas (predivided ratios of Indices of Refraction).
// We don't track the IoRs per layer because that would deprive us of the option to model each layer interface as a mixture of materials (metalness workflow).
// 
// If you don't plan on ignoring the actual convolution of incoming light by the VNDF, such an assumption only speeds up the Importance Sampling slightly as
// on the way back through a layer we don't consume another 2D random variable, instead transforming the ray deterministically. This however would require one
// to keep a stack of cached interactions with each layer, and its just simpler to run local path tracing through layers which can account for multiple scattering
// through a medium layer, etc.
// 
// Because we implement Schussler et. al 2017 we also ensure that signs of dot products with shading normals are identical to smooth normals.
// However the smooth normals are not identical to geometric normals, we reserve the right to use the "normal pull up trick" to make them consistent.
// Schussler can't help with disparity of Smooth Normal and Geometric Normal, it turns smooth surfaces into glistening "disco balls" really outlining the
// polygonization. Using PN-Triangles/displacement would be the optimal solution here. 
class CFrontendIR : public CNodePool
{
protected:
		template<typename T>
		using _TypedHandle = CNodePool::TypedHandle<T>;

	public:
		// constructor
		static inline core::smart_refctd_ptr<CFrontendIR> create(const uint8_t chunkSizeLog2=19, const uint8_t maxNodeAlignLog2=4, refctd_pmr_t&& _pmr={})
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
			// Ignored if no modulator textures
			uint8_t& uvSlot() {return params[0].padding[0];}
			const uint8_t& uvSlot() const {return params[0].padding[0];}
			// Note: the padding abuse
			static_assert(sizeof(SParameter::padding)>0);

			SParameter params[Count] = {};
			// identity transform by default, ignored if no UVs
			hlsl::float32_t2x3 uvTransform = hlsl::float32_t2x3(
				1,0,0,
				0,1,0
			);
		};

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				CNodePool::TypedHandle<CNodePool::CDebugInfo> debugInfo;
		};
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using TypedHandle = _TypedHandle<T>;
		
		class IExprNode;
		// All layers are modelled as coatings, most combinations are not feasible and what combos are feasible depend on the compiler backend you use.
		// Do not use Coatings for things which can be achieved with linear blends! (e.g. alpha transparency)
		class CLayer final : public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CLayer";}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CLayer);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CLayer() = default;

				// Whether the layer is a BSDF depends on having a non-null 2nd child (a transmission component)
				// A whole material is a BSDF iff. all layers have a non-null BTDF, otherwise its two separate layered BRDFs.
				inline bool isBSDF() const {return bool(btdf);}

				// A null BRDF will not produce reflections, while a null BTDF will not allow any transmittance.
				// The laws of BSDFs require reciprocity so we can only have one BTDF, but they allow separate/different BRDFs
				// Concrete example, think Vantablack stuck to a Aluminimum foil on the other side. 
				_TypedHandle<IExprNode> brdfTop = {};
				_TypedHandle<IExprNode> btdf = {};
				// when dealing with refractice indices, we expect the `brdfTop` and `brdfBottom` to be in sync (reciprocals of each other)
				_TypedHandle<IExprNode> brdfBottom = {};
				// The layer below us, if in the stack there's a layer with a null BTDF, we reserve the right to split up the material into two separate
				// materials, one for the front and one for the back face in the final IR. Everything between the first and last null BTDF will get discarded.
				_TypedHandle<CLayer> coated = {};
		};

		//
		class IExprNode : public INode
		{
			public:
				// Only sane child count allowed
				virtual uint8_t getChildCount() const = 0;
				inline _TypedHandle<IExprNode> getChildHandle(const uint8_t ix)
				{
					if (ix<getChildCount())
						return getChildHandle_impl(ix);
					return {};
				}
				inline _TypedHandle<const IExprNode> getChildHandle(const uint8_t ix) const
				{
					auto retval = const_cast<IExprNode*>(this)->getChildHandle(ix);
					return retval;
				}

				// A "contributor" of a term to the lighting equation: a BxDF (reflection or tranmission) or Emitter term
				// Contributors are not allowed to be multiplied together, but every additive term in the Expression must contain a contributor factor.
				enum class Type : uint8_t
				{
					Contributor = 0,
					Mul = 1,
					Add = 2,
					Other = 3
				};
				virtual inline Type getType() const {return Type::Other;}
				
			protected:
				friend class CFrontendIR;
				// default is no special checks beyond the above
				struct SInvalidCheckArgs
				{
					const CFrontendIR* pool;
					system::logger_opt_ptr logger;
					bool isBTDF;
					// there's space for 7 more bools
				};
				virtual inline bool invalid(const SInvalidCheckArgs&) const {return false;}
				virtual _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const = 0;
		};

		//! Base class for leaf node quantities which contribute additively to the Lighting Integral
		class IContributor : public IExprNode
		{
			public:
				inline Type getType() const override final {return Type::Contributor;}
		};

		// This node could also represent non directional emission, but we have another node for that
		class CSpectralVariable final : public IExprNode
		{
			public:
				inline uint8_t getChildCount() const override final { return 0; }
				inline const std::string_view getTypeName() const override {return "nbl::CSpectralVariable";}
				// Variable length but has no children

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

				inline operator bool() const {return !invalid(SInvalidCheckArgs{.pool=nullptr,.logger=nullptr});}

			protected:
				inline ~CSpectralVariable()
				{
					auto pWonky = reinterpret_cast<SCreationParams<1>*>(this+1);
					std::destroy_n(pWonky->knots.params,getKnotCount());
				}

				inline _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return {};}
				inline bool invalid(const SInvalidCheckArgs& args) const override
				{
					auto pWonky = reinterpret_cast<const SCreationParams<1>*>(this+1);
					for (auto i=0u; i<getKnotCount(); i++)
					if (!pWonky->knots.params[i])
					{
						args.logger.log("Knot %u parameters invalid",system::ILogger::ELL_ERROR,i);
						return false;
					}
					return true;
				}

			private:
				const uint8_t* paramsBeginPadding() const {return reinterpret_cast<const SCreationParams<1>*>(this+1)->knots.params[0].padding;}
		};
		//
		class IUnaryOp : public IExprNode
		{
			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override final {return child;}
				
			public:
				inline uint8_t getChildCount() const override final {return 1;}

				TypedHandle<IExprNode> child = {};
		};
		class IBinOp : public IExprNode
		{
			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override final {return ix ? rhs:lhs;}
				
			public:
				inline uint8_t getChildCount() const override final {return 2;}

				TypedHandle<IExprNode> lhs = {};
				TypedHandle<IExprNode> rhs = {};
		};
		//! Basic combiner nodes
		class CMul final : public IBinOp
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CMul";}
				inline Type getType() const override {return Type::Mul;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CMul);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CMul() = default;
		};
		class CAdd final : public IBinOp
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CAdd";}
				inline Type getType() const override {return Type::Add;}

				// you can set the children later
				static inline uint32_t calc_size() {return sizeof(CAdd);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CAdd() = default;
		};
		// does `1-expression`
		class CComplement final : public IUnaryOp
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CComplement";}

				// you can set the children later
				static inline uint32_t calc_size() { return sizeof(CComplement); }
				inline uint32_t getSize() const override { return calc_size(); }
				inline CComplement() = default;
		};
		// Compute Inifinite Scatter and extinction between two parallel infinite planes
		// Reflective Component is: R, T E R E T, T E (R E)^3 T, T E (R E)^5 T, ... 
		// Transmissive Component is: T E T, T E (R E)^2 T, T E (R E)^4 T, ... 
		// Note: This node can be also used to model non-linear color shifts of Diffuse BRDF multiple scattering if one plugs in the albedo as the reflectance.
		class CThinInfiniteScatterCorrection final : public IExprNode
		{
			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override final {return ix ? (ix!=1 ? extinction:transmittance):reflectance;}
				
			public:
				inline uint8_t getChildCount() const override final {return 3;}
				inline const std::string_view getTypeName() const override { return "nbl::CThinInfiniteScatterCorrection"; }

				// you can set the children later
				static inline uint32_t calc_size() { return sizeof(CThinInfiniteScatterCorrection); }
				inline uint32_t getSize() const override { return calc_size(); }
				inline CThinInfiniteScatterCorrection() = default;

				TypedHandle<IExprNode> reflectance = {};
				TypedHandle<IExprNode> transmittance = {};
				TypedHandle<IExprNode> extinction = {};
				// Whether to compute reflectance or transmittance
				uint8_t computeTransmittance : 1 = false;
		};
		// Emission nodes are only allowed in BRDF expressions, not BTDF. To allow different emission on both sides, expressed unambigously.
		// Basic Emitter
		class CEmitter final : public IContributor
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CEmitter";}
				inline uint8_t getChildCount() const override {return 1;}

				// you can set the members later
				static inline uint32_t calc_size() {return sizeof(CEmitter);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CEmitter() = default;

				TypedHandle<CSpectralVariable> radiance = {};
				// This can be anything like an IES profile, if invalid, there's no directionality to the emission
				SParameter profile = {};
				hlsl::float32_t3x3 profileTransform = hlsl::float32_t3x3(
					1,0,0,
					0,1,0,
					0,0,1
				);
				// TODO: semantic flags/metadata (symmetries of the profile)

			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return radiance;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
		};
		//! Special nodes meant to be used as `CMul::rhs`, as for the `N`, they use the normal used by the Leaf ContributorLeafs in its MUL node relative subgraph.
		//! However if the Leaf BXDF is Cook Torrance, the microfacet `H` normal will be used instead.
		//! If there are two BxDFs with different normals, these nodes get split and duplicated into two in our Final IR.
		//! ----------------------------------------------------------------------------------------------------------------
		// Beer's Law Node, behaves differently depending on where it is:
		// - to get a scattering medium, multiply it with CDeltaTransmission BTDF placed between two BRDFs in the same medium
		// - to get a scattering medium between two Layers, create a layer with the above
		// - to apply the beer's law on a single microfacet or a BRDF or BTDF multiply it with a BxDF
		class CBeer final : public IExprNode
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CBeer";}
				inline uint8_t getChildCount() const override {return 1;}

				// you can set the members later
				static inline uint32_t calc_size() {return sizeof(CBeer);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CBeer() = default;

				// Effective transparency = exp2(log2(perpTransparency)/dot(refract(V,X,eta),X)) = exp2(log2(perpTransparency)*inversesqrt(1.f+(LdotX-1)*rcpEta))
				// Absorption and thickness of the interface combined into a single variable, eta and `LdotX` is taken from the leaf BTDF node.
				// With refractions from Dielectrics, we get just `1/LdotX`, for Delta Transmission we get `1/VdotN` since its the same
				TypedHandle<CSpectralVariable> perpTransparency = {};

			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return perpTransparency;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
		};
		// The "oriented" in the Etas means from frontface to backface, so there's no need to reciprocate them when creating matching BTDF for BRDF
		class CFresnel final : public IExprNode
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
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return ix ? orientedImagEta:orientedRealEta;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
		};
		// @kept_secret TODO: Thin Film Interference Fresnel
		//! Basic BxDF nodes
		// Every BxDF leaf node is supposed to pass WFT test, color and extinction is added on later via multipliers
		class IBxDF : public IContributor
		{
			public:
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
		// Delta Transmission is the only Special Delta Distribution Node, because of how useful it is for compiling Anyhit shaders, the rest can be done easily with:
		// - Delta Reflection -> Any Cook Torrance BxDF with roughness=0 attached as BRDF
		// - Smooth Conductor -> above multiplied with Conductor-Fresnel
		// - Smooth Dielectric -> Any Cook Torrance BxDF with roughness=0 attached as BRDF on both sides (bottom side having a reciprocated Eta) of a Layer and BTDF multiplied with Dielectric-Fresnel (no imaginary component)
		// - Thindielectric -> Any Cook Torrance BxDF multiplied with Dielectric-Fresnel as BRDF in both sides and a Delta Transmission BTDF
		// - Plastic -> Can layer the above over Diffuse BRDF, but its faster to cook a mixture of Diffuse and Smooth Conductor BRDFs, weighing the diffuse by Fresnel complements.
		//				If one wants to emulate non-linear diffuse TIR color shifts, abuse `CThinInfiniteScatterCorrection` 
		class CDeltaTransmission final : public IBxDF
		{
			public:
				inline uint8_t getChildCount() const override {return 0;}
				inline const std::string_view getTypeName() const override {return "nbl::CDeltaTransmission";}
				static inline uint32_t calc_size()
				{
					return sizeof(CDeltaTransmission);
				}
				inline uint32_t getSize() const override {return calc_size();}
				inline CDeltaTransmission() = default;

			protected:
				inline _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return {};}
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
				inline uint32_t getSize() const override {return calc_size();}
				inline COrenNayar() = default;

				SBasicNDFParams ndParams;

			protected:
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
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
				inline uint32_t getSize() const override {return calc_size();}
				inline CCookTorrance() = default;

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
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return orientedRealEta;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
		};

		//
		template<typename T, typename... Args>
		inline CNodePool::TypedHandle<T> _new(Args&&... args)
		{
			return CNodePool::_new<T,Args...>(std::forward<Args>(args)...);
		}

		//
		template<typename T>
		inline void _delete(const CNodePool::TypedHandle<T> h)
		{
			return CNodePool::_delete<T>(h);
		}

		// Each material comes down to this
		inline std::span<const TypedHandle<const CLayer>> getMaterials() {return m_rootNodes;}
		inline bool addMaterial(const TypedHandle<const CLayer> rootNode, system::logger_opt_ptr logger)
		{
			if (valid(rootNode,logger))
			{
				m_rootNodes.push_back(rootNode);
				return true;
			}
			return false;
		}

		// To quickly make a matching backface material from a frontface or vice versa
		NBL_API TypedHandle<IExprNode> reciprocate(const TypedHandle<const IExprNode> other);

		// IMPORTANT: Two BxDFs are not allowed to be multiplied together.
		// NOTE: Right now all Spectral Variables are required to be Monochrome or 3 bucket fixed semantics, all the same wavelength.
		// Some things we can't check such as the compatibility of the BTDF with the BRDF (matching indices of refraction, etc.)
		bool valid(const TypedHandle<const CLayer> rootHandle, system::logger_opt_ptr logger) const;

		// For Debug Visualization (TODO: refactor to allow printing invalid nodes not in the `m_rootNodes` -> `printDotTree(std::ostringstream&,TypedHandle<const INode>)`)
		NBL_API void printDotGraph(std::ostringstream& str) const;
		inline core::string printDotGraph() const
		{
			std::ostringstream tmp;
			printDotGraph(tmp);
			return tmp.str();
		}

	protected:
		using CNodePool::CNodePool;

		inline core::string getNodeID(const TypedHandle<const INode> handle) const {return core::string("_")+std::to_string(handle.untyped.value);}
		inline core::string getLabelledNodeID(const TypedHandle<const INode> handle) const
		{
			const INode* node = deref(handle);
			core::string retval = getNodeID(handle);
			retval += " [label=\"";
			retval += node->getTypeName();
			if (const auto* debug=deref(node->debugInfo); debug && !debug->data().empty())
			{
				retval += "\\n";
				retval += std::string_view(reinterpret_cast<const char*>(debug->data().data()),debug->data().size()-1);
			}
			retval += "\"]";
			return retval;
		}

		core::vector<TypedHandle<const CLayer>> m_rootNodes;
};

inline bool CFrontendIR::valid(const TypedHandle<const CLayer> rootHandle, system::logger_opt_ptr logger) const
{
	constexpr auto ELL_ERROR = system::ILogger::E_LOG_LEVEL::ELL_ERROR;

	core::stack<const CLayer*> layerStack;
	auto pushLayer = [&](const TypedHandle<const CLayer> layerHandle)->bool
	{
		const auto* layer = deref(layerHandle);
		if (!layer)
		{
			logger.log("Layer node %u of type %s not a `CLayer` node!",ELL_ERROR,layerHandle.untyped.value,getTypeName(layerHandle).data());
			return false;
		}
		layerStack.push(layer);
		return true;
	};
	if (!pushLayer(rootHandle))
		return false;
			
	enum class SubtreeContributorState : uint8_t
	{
		Required,
		Forbidden
	};
	struct StackEntry
	{
		const IExprNode* node;
		TypedHandle<const IExprNode> handle;
		uint8_t contribSlot;
		SubtreeContributorState contribState = SubtreeContributorState::Required;
	};
	core::stack<StackEntry> exprStack;
	auto validateExpression = [&](const TypedHandle<const IExprNode> exprRoot, const bool isBTDF) -> bool
	{
		if (!exprRoot)
			return true;
		//
		const auto* root = deref(exprRoot);
		if (!root)
		{
			logger.log("Node %u is not an Expression Node, it's %s",ELL_ERROR,exprRoot.untyped.value,getTypeName(exprRoot).data());
			return false;
		}
		//
		constexpr uint8_t MaxContributors = 255;
		uint8_t contributorCount = 0;
		std::bitset<MaxContributors> contributorsFound;
		//
		exprStack.push({.node=root,.handle=exprRoot,.contribSlot=contributorCount++});
		const IExprNode::SInvalidCheckArgs invalidCheckArgs = {.pool=this,.logger=logger,.isBTDF=isBTDF};
		while (!exprStack.empty())
		{
			const StackEntry entry = exprStack.top();
			exprStack.pop();
			const auto* node = entry.node;
			const auto nodeType = node->getType();
			const bool nodeIsMul = nodeType==IExprNode::Type::Mul;
			const bool nodeIsAdd = nodeType==IExprNode::Type::Add;
			const auto childCount = node->getChildCount();
			bool takeOverContribSlot = true; // first add child can do this
			for (auto childIx=0; childIx<childCount; childIx++)
			{
				const auto childHandle = node->getChildHandle(childIx);
				if (const auto child=deref(childHandle); child)
				{
					const bool noContribBelow = entry.contribState==SubtreeContributorState::Forbidden || childIx!=0 && nodeIsMul;
					StackEntry newEntry = {.node=child,.handle=childHandle};
					if (noContribBelow)
					{
						if (child->getType()==IExprNode::Type::Contributor)
						{
							logger.log("Contibutor node %u of type %s not allowed in this subtree!",ELL_ERROR,childHandle,getTypeName(childHandle).data());
							return false;
						}
						newEntry.contribSlot = MaxContributors;
						newEntry.contribState = SubtreeContributorState::Forbidden;
					}
					else if (takeOverContribSlot)
					{
						assert(entry.contribSlot<MaxContributors);
						newEntry.contribSlot = entry.contribSlot;
						takeOverContribSlot = false;
					}
					else
						newEntry.contribSlot = contributorCount++;
					if (contributorCount>MaxContributors)
					{
						logger.log("Expression too complex, more than %d contributors encountered",ELL_ERROR,MaxContributors);
						return false;
					}
					exprStack.push(newEntry);
				}
				else if (childHandle)
				{
					logger.log(
						"Node %u of type %s has a %u th child %u which doesn't cast to `IExprNode`, its type is %s instead!",ELL_ERROR,
						entry.handle.untyped.value,node->getTypeName().data(),childIx,childHandle,getTypeName(childHandle).data()
					);
					return false;
				}
			}
			// check only after we know all children are OK
			if (node->invalid(invalidCheckArgs))
			{
				logger.log("Node %u of type %s is invalid!",ELL_ERROR,entry.handle.untyped.value,node->getTypeName().data());
				return false;
			}
			if (entry.contribSlot<MaxContributors)
				contributorsFound.set(entry.contribSlot);
		}
		for (uint8_t i=0; i<contributorCount; i++)
		if (!contributorsFound.test(i))
		{
			logger.log("Expression starting with node %u does not have a Contributor Leaf Node in all of its additively distributive subtrees",ELL_ERROR,exprRoot.untyped.value);
			return false;
		}
		return true;
	};
	while (!layerStack.empty())
	{
		const auto* layer = layerStack.top();
		layerStack.pop();
		if (layer->coated && !pushLayer(layer->coated))
		{
			logger.log("\tcoatee %d was specificed but is of wrong type",ELL_ERROR,layer->coated);
			return false;
		}
		if (!layer->brdfTop && !layer->btdf && !layer->brdfBottom)
		{
			logger.log("At least one BRDF or BTDF in the Layer is required.",ELL_ERROR);
			return false;
		}
		if (!validateExpression(layer->brdfTop,false))
			return false;
		if (!validateExpression(layer->btdf,true))
			return false;
		if (!validateExpression(layer->brdfBottom,false))
			return false;
	}
	return true;
}

} // namespace nbl::asset::material_compiler3

#endif