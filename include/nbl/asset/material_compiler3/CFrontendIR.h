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
// If you don't plan on ignoring the actual convolution of incoming light by the BSDF, such an assumption only speeds up the Importance Sampling slightly as
// on the way back through a layer we don't consume another 2D random variable, instead transforming the ray deterministically. This however would require one
// to keep a stack of cached interactions with each layer, and its just simpler to run local path tracing through layers which can account for multiple scattering
// through a medium layer, etc.
// 
// Our Frontend is built around the IR, which wants to perform the following canonicalization of a BSDF Layer (not including emission):
// 
// 	   f(w_i,w_o) = Sum_i^N Product_j^{N_i} h_{ij}(w_i,w_o) l_i(w_i,w_o)
// 
// Where `l(w_i,w_o)` is a Contributor Node BxDF such as Oren Nayar or Cook-Torrance, which is doesn't model absorption and is usually Monochrome.
// These are assumed to be 100% valid BxDFs with White Furnace Test <= 1 and obeying Helmholtz Reciprocity. This is why you can't multiply two "Contributor Nodes" together.
// We make an attempt to implement Energy Normalized versions of `l_i` but not always, so there might be energy loss due to single scattering assumptions.
// 
// This convention greatly simplifies the Layering of BSDFs as when we model two layers combined we need only consider the Sum terms which are Products of a BTDF contibutor
// in convolution with the layer below or above. For emission this is equivalent to convolving the emission with BTDFs producing a custom emission profile.
// Some of these combinations can be approximated or solved outright without resolving to frequency space approaches or path tracing within the layers.
// 
// To obtain a valid BxDF for the canonical expression, each product of weights also needs to exhibit Helmholtz Reciprocity:
// 
// 	   Product_j^{N_i} h(w_i,w_o) = Product_j^{N_i} h(w_o,w_i)
// 
// Which means that direction dependant weight nodes need to know the underlying contributor they are weighting to determine their semantics, e.g. a Fresnel on:
// - Cook Torrance will use the Microfacet Normal for any calculation as that is symmetric between `w_o` and `w_i`
// - Diffuse will use both `NdotV` and `NdotL` (also known as `theta_i` and `theta_o`) symmetrically
// - A BTDF will use the compliments (`1-x`) of the Fresnels
// 
// We cannot derive BTDF factors from top and bottom BRDF as the problem is underconstrained, we don't know which factor models absorption and which part transmission.
// 
// Helmholtz Reciprocity allows us to use completely independent BRDFs per hemisphere, when `w_i` and `w_o` are in the same hemisphere (reflection).
// Note that transmission only occurs when `w_i` and `w_o` are in opposite hemispheres and the reciprocity forces one BTDF.
// 
// There's an implicit Top and Bottom on the layer stack, but thats only for the purpose of interpreting the Etas (predivided ratios of Indices of Refraction),
// both the Top and Bottom BRDF treat the Eta as being the speed of light in the medium above over the speed of light in the medium below.
// This means that for modelling air-vs-glass you use the same Eta for the Top BRDF, the middle BTDF and Bottom BRDF.
// We don't track the IoRs per layer because that would deprive us of the option to model each layer interface as a mixture of materials (metalness workflow).
// 
// The backend can expand the Top BRDF, Middle BTDF, Bottom BRDF into 4 separate instruction streams for Front-Back BRDF and BTDF. This is because we can
// throw away the first or last BRDF+BTDF in the stack, as well as use different pre-computed Etas if we know the sign of `cos(theta_i)` as we interact with each layer.
// Whether the backend actually generates a separate instruction stream depends on the impact of Instruction Cache misses due to not sharing streams for layers.
// 
// Also note that a single null BTDF in the stack splits it into the two separate stacks, one per original interaction orientation.
// 
// I've considered expressing the layers using only a BTDF and BRDF (same top and bottom hemisphere) but that would lead to more layers in for materials,
// requiring the placing of a mirror then vantablack layer for most one-sided materials, and most importantly disallow the expression of certain front-back correlations.
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
		
		struct SParameter
		{
			inline operator bool() const
			{
				return abs(scale)<std::numeric_limits<float>::infinity() && (!view || viewChannel<getFormatChannelCount(view->getCreationParameters().format));
			}

			NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const;

			// at this stage we store the multipliers in highest precision
			float scale = std::numeric_limits<float>::infinity();
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
			private:
				friend class CSpectralVariable;
				template<typename StringConstIterator=const core::string*>
				inline void printDot(const uint8_t _count, std::ostringstream& sstr, const core::string& selfID, StringConstIterator paramNameBegin={}) const
				{
					bool imageUsed = false;
					for (uint8_t i=0; i<_count; i++)
					{
						const auto paramID = selfID+"_param"+std::to_string(i);
						if (params[i].view)
							imageUsed = true;
						params[i].printDot(sstr,paramID);
						sstr << "\n\t" << selfID << " -> " << paramID;
						if (paramNameBegin)
							sstr <<" [label=\"" << *(paramNameBegin++) << "\"]";
						else
							sstr <<" [label=\"Param " << std::to_string(i) <<"\"]";
					}
					if (imageUsed)
					{
						const auto uvTransformID = selfID+"_uvTransform";
						sstr << "\n\t" << uvTransformID << " [label=\"";
						printMatrix(sstr,*reinterpret_cast<const decltype(uvTransform)*>(params+_count));
						sstr << "\"]";
						sstr << "\n\t" << selfID << " -> " << uvTransformID << "[label=\"UV Transform\"]";
					}
				}

			public:
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

				template<typename StringConstIterator=const core::string*>
				inline void printDot(std::ostringstream& sstr, const core::string& selfID, StringConstIterator paramNameBegin={}) const
				{
					printDot<StringConstIterator>(Count,sstr,selfID,std::forward<StringConstIterator>(paramNameBegin));
				}

				SParameter params[Count] = {};
				// identity transform by default, ignored if no UVs
				hlsl::float32_t2x3 uvTransform = hlsl::float32_t2x3(
					1,0,0,
					0,1,0
				);

				// to make sure there will be no padding inbetween
				static_assert(alignof(SParameter)>=alignof(hlsl::float32_t2x3));
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
				// by default all children are mandatory
				virtual inline bool invalid(const SInvalidCheckArgs& args) const
				{
					const auto childCount = getChildCount();
					for (uint8_t i=0u; i<childCount; i++)
					if (const auto childHandle=getChildHandle_impl(i); !childHandle)
					{
						args.logger.log("Default `IExprNode::invalid` child #%u missing!",system::ILogger::ELL_ERROR,i);
						return true;
					}
					return false;
				}
				virtual _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const = 0;
				
				virtual inline core::string getLabelSuffix() const {return "";}
				virtual inline void printDot(std::ostringstream& sstr, const core::string& selfID) const {}
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

				//
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
					const Semantics& getSemantics() const {return const_cast<const Semantics&>(const_cast<SCreationParams<Count>*>(this)->getSemantics());}
				};
				//
				template<uint8_t Count>
				inline CSpectralVariable(SCreationParams<Count>&& params)
				{
					// back up the count
					params.knots.params[0].padding[1] = Count;
					// set it correctly for monochrome
					if constexpr (Count==1)
						params.knots.params[0].padding[2] = static_cast<uint8_t>(Semantics::NoneUndefined);
					else
					{
						assert(params.getSemantics()!=Semantics::NoneUndefined);
					}
					std::construct_at(reinterpret_cast<SCreationParams<Count>*>(this+1),std::move(params));
				}

				// encapsulation due to padding abuse
				inline uint8_t& uvSlot() {return pWonky()->knots.uvSlot();}
				inline const uint8_t& uvSlot() const {return pWonky()->knots.uvSlot();}

				// these getters are immutable
				inline uint8_t getKnotCount() const
				{
					static_assert(sizeof(SParameter::padding)>1);
					return paramsBeginPadding()[1];
				}
				inline Semantics getSemantics() const
				{
					static_assert(sizeof(SParameter::padding)>2);
					const auto retval = static_cast<Semantics>(paramsBeginPadding()[2]);
					assert((getKnotCount()==1)==(retval==Semantics::NoneUndefined));
					return retval;
				}

				//
				inline SParameter* getParam(const uint8_t i)
				{
					if (i<getKnotCount())
						return &pWonky()->knots.params[i];
					return nullptr;
				}
				inline const SParameter* getParam(const uint8_t i) const {return const_cast<const SParameter*>(const_cast<CSpectralVariable*>(this)->getParam(i));}

				//
				template<uint8_t Count>
				static inline uint32_t calc_size(const SCreationParams<Count>&)
				{
					return sizeof(CSpectralVariable)+sizeof(SCreationParams<Count>);
				}
				inline uint32_t getSize() const override
				{
					return sizeof(CSpectralVariable)+sizeof(SCreationParams<1>)+(getKnotCount()-1)*sizeof(SParameter);
				}

				inline operator bool() const {return !invalid(SInvalidCheckArgs{.pool=nullptr,.logger=nullptr});}

			protected:
				inline ~CSpectralVariable()
				{
					std::destroy_n(pWonky()->knots.params,getKnotCount());
				}

				inline _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return {};}
				inline bool invalid(const SInvalidCheckArgs& args) const override
				{
					const auto knotCount = getKnotCount();
					// non-monochrome spectral variable 
					if (const auto semantic=getSemantics(); knotCount>1)
					switch (semantic)
					{
						case Semantics::Fixed3_SRGB: [[fallthrough]];
						case Semantics::Fixed3_DCI_P3: [[fallthrough]];
						case Semantics::Fixed3_BT2020: [[fallthrough]];
						case Semantics::Fixed3_AdobeRGB: [[fallthrough]];
						case Semantics::Fixed3_AcesCG:
							if (knotCount!=3)
							{
								args.logger.log("Semantic %d is only usable with 3 knots, this has %d knots",system::ILogger::ELL_ERROR,static_cast<uint8_t>(semantic),knotCount);
								return false;
							}
							break;
						default:
							args.logger.log("Invalid Semantic %d",system::ILogger::ELL_ERROR,static_cast<uint8_t>(semantic));
							return true;
					}
					for (auto i=0u; i<knotCount; i++)
					if (!*getParam(i))
					{
						args.logger.log("Knot %u parameters invalid",system::ILogger::ELL_ERROR,i);
						return true;
					}
					return false;
				}
				
				NBL_API core::string getLabelSuffix() const override;
				NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const override;

			private:
				SCreationParams<1>* pWonky() {return reinterpret_cast<SCreationParams<1>*>(this+1);}
				const SCreationParams<1>* pWonky() const {return reinterpret_cast<const SCreationParams<1>*>(this+1);}
				const uint8_t* paramsBeginPadding() const {return pWonky()->knots.params[0].padding; }
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
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
				inline void printDot(std::ostringstream& sstr, const core::string& selfID) const override
				{
					sstr << "\n\t" << selfID << " -> " << selfID << "_computeTransmittance [label=\"computeTransmittance = " << (computeTransmittance ? "true":"false") << "\"]";
				}
				
			public:
				inline uint8_t getChildCount() const override final {return 3;}
				inline const std::string_view getTypeName() const override {return "nbl::CThinInfiniteScatterCorrection";}

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
		// Basic Emitter - note that it is of unit radiance so its easier to importance sample
		class CEmitter final : public IContributor
		{
			public:
				inline const std::string_view getTypeName() const override {return "nbl::CEmitter";}
				inline uint8_t getChildCount() const override {return 0;}

				// you can set the members later
				static inline uint32_t calc_size() {return sizeof(CEmitter);}
				inline uint32_t getSize() const override {return calc_size();}
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

			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return {};}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
				NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const override;
		};
		//! Special nodes meant to be used as `CMul::rhs`, their behaviour depends on the IContributor in its MUL node relative subgraph.
		//! If you use a different contributor node type or normal for shading, these nodes get split and duplicated into two in our Final IR.
		//! Due to the Helmholtz Reciprocity handling outlined in the comments for the entire front-end you can usually count on these nodes
		//! getting applied once using `VdotH` for Cook-Torrance BRDF, twice using `VdotN` and `LdotN` for Diffuse BRDF, and using their
		//! complements before multiplication for BTDFs. 
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
				static inline uint32_t calc_size() {return sizeof(CFresnel);}
				inline uint32_t getSize() const override {return calc_size();}
				inline CFresnel() = default;

				// Already pre-divided Index of Refraction, e.g. exterior/interior since VdotG>0 the ray always arrives from the exterior.
				TypedHandle<CSpectralVariable> orientedRealEta = {};
				// Specifying this turns your Fresnel into a conductor one, note that currently these are disallowed on BTDFs!
				TypedHandle<CSpectralVariable> orientedImagEta = {};
				// if you want to reuse the same parameter but want to flip the interfaces around
				uint8_t reciprocateEtas : 1 = false;

			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return ix ? orientedImagEta:orientedRealEta;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
				NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const override;
		};
		// @kept_secret TODO: Thin Film Interference Fresnel
		//! Basic BxDF nodes
		// Every BxDF leaf node is supposed to pass WFT test and must not create energy, color and extinction is added on later via multipliers
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
					
					inline SBasicNDFParams()
					{
						// initialize with constant flat deriv map and smooth roughness
						for (auto& param : params)
							param.scale = 0.f;
					}

					// whether the derivative map and roughness is constant regardless of UV-space texture stretching
					inline bool stretchInvariant() const {return !(abs(hlsl::determinant(reference))>std::numeric_limits<float>::min());}

					NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const;

					// Ignored if not invertible, otherwise its the reference "stretch" (UV derivatives) at which identity roughness and normalmapping occurs
					hlsl::float32_t2x2 reference = hlsl::float32_t2x2(0,0,0,0);
				};

				// For Schussler et. al 2017 we'll spawn 2-3 additional BRDF leaf nodes in the proper IR for every normalmap present
		};
		// Delta Transmission is the only Special Delta Distribution Node, because of how useful it is for compiling Anyhit shaders, the rest can be done easily with:
		// - Delta Reflection -> Any Cook Torrance BxDF with roughness=0 attached as BRDF
		// - Smooth Conductor -> above multiplied with Conductor-Fresnel
		// - Smooth Dielectric -> Any Cook Torrance BxDF with roughness=0 attached as BRDF on both sides of a Layer and BTDF multiplied with Dielectric-Fresnel (no imaginary component)
		// - Thindielectric -> Any Cook Torrance BxDF multiplied with Dielectric-Fresnel as BRDF in both sides and a Delta Transmission BTDF with `CThinInfiniteScatterCorrection` on the fresnel
		// - Plastic -> Similar to layering the above over Diffuse BRDF, its of uttmost importance that the BTDF is Delta Transmission.
		//              If one wants to emulate non-linear diffuse TIR color shifts, abuse `CThinInfiniteScatterCorrection`.
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
		//! Because of Schussler et. al 2017 every one of these nodes splits into 2 (if no L dependence) or 3 during canonicalization
		// Base diffuse node
		class COrenNayar final : public IBxDF
		{
			public:
				inline uint8_t getChildCount() const override {return 0;}
				inline const std::string_view getTypeName() const override {return "nbl::COrenNayar";}
				static inline uint32_t calc_size() {return sizeof(COrenNayar);}
				inline uint32_t getSize() const override {return calc_size();}
				inline COrenNayar() = default;

				SBasicNDFParams ndParams = {};

			protected:
				inline _TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return {};}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;
				NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const override;
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

				SBasicNDFParams ndParams = {};
				// We need this eta to compute the refractions of `L` when importance sampling and the Jacobian during H to L generation for rough dielectrics
				// It does not mean we compute the Fresnel weights though! You might ask why we don't do that given that state of the art importance sampling
				// (at time of writing) is to decide upon reflection vs. refraction after the microfacet normal `H` is already sampled,
				// producing an estimator with just Masking and Shadowing function ratios. The reason is because we can simplify our IR by separating out
				// BRDFs and BTDFs components into separate expressions, and also importance sample much better, for details see comments in CTrueIR. 
				TypedHandle<CSpectralVariable> orientedRealEta = {};
				// 
				NDF ndf = NDF::GGX;

			protected:
				inline TypedHandle<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return orientedRealEta;}
				NBL_API bool invalid(const SInvalidCheckArgs& args) const override;

				inline core::string getLabelSuffix() const override {return ndf!=NDF::GGX ? "\\nNDF = Beckmann":"\\nNDF = GGX";}
				NBL_API void printDot(std::ostringstream& sstr, const core::string& selfID) const override;
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

		// Each material comes down to this, YOU MUST NOT MODIFY THE NODES AFTER ADDING THEIR PARENT TO THE ROOT NODES!
		// TODO: shall we copy and hand out a new handle?
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
		NBL_API TypedHandle<CFresnel> createNamedFresnel(const std::string_view name);

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
			if (const auto* expr=deref<const IExprNode>({.untyped=handle.untyped}); expr)
				retval += expr->getLabelSuffix();
			retval += "\"]";
			return retval;
		}

		core::vector<TypedHandle<const CLayer>> m_rootNodes;
};

inline bool CFrontendIR::valid(const TypedHandle<const CLayer> rootHandle, system::logger_opt_ptr logger) const
{
	constexpr auto ELL_ERROR = system::ILogger::E_LOG_LEVEL::ELL_ERROR;
			
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
	// unused yet
	core::unordered_set<TypedHandle<const INode>,HandleHash> visitedNodes;
	// should probably size it better, if I knew total node count allocated or live
	visitedNodes.reserve(m_rootNodes.size()<<3);
	//
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
					// cannot optimize with `unordered_set visitedNodes` because we need to check contributor slots, if we really wanted to we could do it with an
					// `unordered_map` telling us the contributor slot range remapping (and presence of contributor) but right now it would be premature optimization.
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

	core::vector<const CLayer*> layerStack;
	auto pushLayer = [&](const TypedHandle<const CLayer> layerHandle)->bool
	{
		const auto* layer = deref(layerHandle);
		if (!layer)
		{
			logger.log("Layer node %u of type %s not a `CLayer` node!",ELL_ERROR,layerHandle.untyped.value,getTypeName(layerHandle).data());
			return false;
		}
		auto found = std::find(layerStack.begin(),layerStack.end(),layer);
		if (found!=layerStack.end())
		{
			logger.log("Layer node %u is involved in a Cycle!",ELL_ERROR,layerHandle.untyped.value);
			return false;
		}
		layerStack.push_back(layer);
		return true;
	};
	if (!pushLayer(rootHandle))
		return false;
	while (true)
	{
		const auto* layer = layerStack.back();
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
		if (!layer->coated)
			break;
		if (!pushLayer(layer->coated))
		{
			logger.log("\tcoatee %d was specificed but is invalid!",ELL_ERROR,layer->coated);
			return false;
		}
	}
	return true;
}

} // namespace nbl::asset::material_compiler3

#endif