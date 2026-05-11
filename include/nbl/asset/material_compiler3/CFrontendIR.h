
// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CTrueIR.h"


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
// Where `l(w_i,w_o)` is a Contributor Node BxDF such as Oren Nayar or Cook-Torrance, which doesn't model absorption and is usually Monochrome.
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
// Because we implement Schussler et. al 2017 or Yining 2019 we also ensure that signs of dot products with shading normals are identical to smooth normals.
// However the smooth normals are not identical to geometric normals, we reserve the right to use the "normal pull up trick" to make them consistent.
// Schussler and Yining can't help with disparity of Smooth Normal and Geometric Normal, it turns smooth surfaces into glistening "disco balls" really outlining the
// polygonization. Using PN-Triangles/displacement would be the optimal solution here. 
class CFrontendIR final : public CNodePool
{
		using block_allocator_type = CNodePool::obj_pool_type::block_allocator_type;
		template<typename T>
		using _typed_pointer_type = CNodePool::obj_pool_type::mem_pool_type::typed_pointer_type<T>;

	public:
		// constructor
		using creation_params_type = typename obj_pool_type::creation_params_type;
		static inline core::smart_refctd_ptr<CFrontendIR> create(creation_params_type&& params)
		{
			if (params.composed.blockSizeKBLog2<4)
				return nullptr;
			return core::smart_refctd_ptr<CFrontendIR>(new CFrontendIR(std::move(params)),core::dont_grab);
		}

		// basic "built-in" nodes
		class INode : public CNodePool::INode
		{
			public:
				_typed_pointer_type<CNodePool::CDebugInfo> debugInfo;
		};
		//
		template<typename T> requires std::is_base_of_v<INode,std::remove_const_t<T>>
		using typed_pointer_type = _typed_pointer_type<T>;
		
		class IExprNode;
#define TYPE_NAME_STR(NAME) "nbl::asset::material_compiler3::CFrontendIR::"#NAME
		// All layers are modelled as coatings, most combinations are not feasible and what combos are feasible depend on the compiler backend you use.
		// Do not use Coatings for things which can be achieved with linear blends! (e.g. alpha transparency)
		// TODO: can we have an object with a v-table thats still trivially destructible (just to know the type, but not set-up/run down anything - so no alloc tracking)
		class CLayer final : public obj_pool_type::INonTrivial, public INode
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CLayer);}

				// you can set the children later
				inline CLayer() = default;

				// Whether the layer is a BSDF depends on having a non-null 2nd child (a transmission component)
				// A whole material is a BSDF iff. all layers have a non-null BTDF, otherwise its two separate layered BRDFs.
				inline bool isBSDF() const {return bool(btdf);}

				// A null BRDF will not produce reflections, while a null BTDF will not allow any transmittance.
				// The laws of BSDFs require reciprocity so we can only have one BTDF, but they allow separate/different BRDFs
				// Concrete example, think Vantablack stuck to a Aluminimum foil on the other side. 
				_typed_pointer_type<IExprNode> brdfTop = {};
				_typed_pointer_type<IExprNode> btdf = {};
				// when dealing with refractice indices, we expect the `brdfTop` and `brdfBottom` to be in sync (reciprocals of each other)
				_typed_pointer_type<IExprNode> brdfBottom = {};
				// The layer below us, if in the stack there's a layer with a null BTDF, we reserve the right to split up the material into two separate
				// materials, one for the front and one for the back face in the final IR. Everything between the first and last null BTDF will get discarded.
				_typed_pointer_type<CLayer> coated = {};
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

				// TODO: rename to `EType` and the `getType` to `getExprNodeType`
				// A "contributor" of a term to the lighting equation: a BxDF (reflection or tranmission) or Emitter term
				// Contributors are not allowed to be multiplied together, but every additive term in the Expression must contain a contributor factor.
				enum class Type : uint8_t
				{
					Contributor = 0,
					Mul = 1,
					Add = 2,
					Complement = 3,
					SpectralVariable = 4,
					Other = 5
				};
				virtual inline Type getType() const {return Type::Other;}
				
			protected:
				friend class CFrontendIR;
				// copy
				virtual _typed_pointer_type<IExprNode> copy(CFrontendIR* ir) const = 0;
#define COPY_DEFAULT_IMPL inline _typed_pointer_type<IExprNode> copy(CFrontendIR* ir) const override final \
				{ \
					auto& pool = ir->getObjectPool(); \
					const auto copyH = pool.emplace<std::remove_const_t<std::remove_pointer_t<decltype(this)> > >(); \
					if (auto* const copy = pool.deref(copyH); copyH) \
						*copy = *this; \
					return copyH; \
				}

				// child managment
				virtual inline _typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const {assert(false); return {};}
				inline void setChild(const uint8_t ix, _typed_pointer_type<IExprNode> newChild)
				{
					assert(ix<getChildCount());
					setChild_impl(ix,newChild);
				}
				virtual inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) {assert(false);}

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

				virtual bool inline reciprocatable() const {return false;}
				// unless you override it, you're not supposed to call it
				virtual void reciprocate(IExprNode* dst) const {assert(reciprocatable() && dst);}
				
				virtual inline core::string getLabelSuffix() const {return "";}
				virtual inline std::string_view getChildName_impl(const uint8_t ix) const {return "";}
				virtual inline void printDot(std::ostringstream& sstr, const core::string& selfID) const {}
		};

		//! Base class for leaf node quantities which contribute additively to the Lighting Integral
		class IContributor : public IExprNode
		{
			public:
				inline Type getType() const override final {return Type::Contributor;}

			protected:
				friend class CFrontendIR;
				using ir_contributor_handle_t = CTrueIR::typed_pointer_type<CTrueIR::IContributor>;
				virtual ir_contributor_handle_t createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const = 0;
		};

		// This node could also represent non directional emission, but we have another node for that
		class ISpectralVariableExpr : public CTrueIR::ISpectralVariable, public IExprNode
		{
			public:
				// Variable length but has no children
				inline uint8_t getChildCount() const override final {return 0;}

				//
				inline IExprNode::Type getType() const override final {return Type::SpectralVariable;}

				//
				inline operator bool() const {return valid(nullptr);}

			protected:
				inline _typed_pointer_type<IExprNode> copy(CFrontendIR* ir) const override final
				{
					return static_cast<const CSpectralVariableExpr*>(this)->copy(ir->getObjectPool());
				}

				//
				inline bool invalid(const SInvalidCheckArgs& args) const override final {return !valid(args.logger);}
				
				//
				NBL_API2 core::string getLabelSuffix() const override final;
				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override final;

				//
				friend class CFrontendIR;
				NBL_API2 CTrueIR::typed_pointer_type<CTrueIR::CSpectralVariableFactor> createIRNode(const CFrontendIR* ast, CTrueIR* ir) const;
		};
		using CSpectralVariableExpr = CTrueIR::CSpectralVariable<ISpectralVariableExpr>;
		//
		class IUnaryOp : public obj_pool_type::INonTrivial, public IExprNode
		{
			protected:
				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override final {return child;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override final {child = newChild;}
				
			public:
				inline uint8_t getChildCount() const override final {return 1;}

				typed_pointer_type<IExprNode> child = {};
		};
		class IBinOp : public obj_pool_type::INonTrivial, public IExprNode
		{
			protected:
				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override final {return ix ? rhs:lhs;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override final {*(ix ? &rhs:&lhs) = newChild;}

				inline std::string_view getChildName_impl(const uint8_t ix) const override final {return ix ? "rhs":"lhs";}
				
			public:
				inline uint8_t getChildCount() const override final {return 2;}

				typed_pointer_type<IExprNode> lhs = {};
				typed_pointer_type<IExprNode> rhs = {};
		};
		//! Basic combiner nodes
		class CMul final : public IBinOp
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CMul);}
				inline Type getType() const override {return Type::Mul;}

				// you can set the children later
				inline CMul() = default;

			protected:
				COPY_DEFAULT_IMPL
		};
		class CAdd final : public IBinOp
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CAdd);}
				inline Type getType() const override {return Type::Add;}

				// you can set the children later
				inline CAdd() = default;

			protected:
				COPY_DEFAULT_IMPL
		};
		// does `1-expression`
		class CComplement final : public IUnaryOp
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CComplement);}
				inline Type getType() const override {return Type::Complement;}

				// you can set the children later
				inline CComplement() = default;

			protected:
				COPY_DEFAULT_IMPL
		};
		// Emission nodes are only allowed in BRDF expressions, not BTDF. To allow different emission on both sides, expressed unambigously.
		// Basic Emitter - note that it is of unit radiance so its easier to importance sample
		class CEmitter final : public obj_pool_type::INonTrivial, public IContributor
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CEmitter);}
				inline uint8_t getChildCount() const override {return 0;}

				// you can set the members later
				inline CEmitter() = default;

				// This can be anything like an IES profile, if invalid, there's no directionality to the emission
				// `profile.scale` can still be used to influence the light strength without influencing NEE light picking probabilities
				CTrueIR::SParameter profile = {};
				hlsl::float32_t3x3 profileTransform = hlsl::float32_t3x3(
					1,0,0,
					0,1,0,
					0,0,1
				);
				// TODO: semantic flags/metadata (symmetries of the profile)

			protected:
				COPY_DEFAULT_IMPL

				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override;

				NBL_API2 ir_contributor_handle_t createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const;
		};
		//! Special nodes meant to be used as `CMul::rhs`, their behaviour depends on the IContributor in its MUL node relative subgraph.
		//! If you use a different contributor node type or normal for shading, these nodes get split and duplicated into two in our Final IR.
		//! Due to the Helmholtz Reciprocity handling outlined in the comments for the entire front-end you can usually count on these nodes
		//! getting applied once using `VdotH` for Cook-Torrance BRDF, twice using `VdotN` and `LdotN` for Diffuse BRDF, and using their
		//! complements before multiplication for BTDFs. 
		class IContributorDependant : public obj_pool_type::INonTrivial, public IExprNode
		{
		};
		// Beer's Law Node, behaves differently depending on where it is:
		// - to get an extinction medium, multiply it with CDeltaTransmission BTDF placed between two BRDFs in the same medium
		// - to get a scattering medium between two Layers, create a layer with just a BTDF set up like above
		// - to apply the beer's law on a single microfacet or a BRDF or BTDF multiply it with a BxDF
		// Note: Even it makes little sense, Beer can be applied to the most outermost BRDF to simulate a correllated "foggy" coating without an extra BRDF layer.
		class CBeer final : public IContributorDependant
		{
			public:
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CBeer);}
				inline uint8_t getChildCount() const override {return 2;}

				// you can set the members later
				inline CBeer() = default;

				// Effective transparency = exp2(log2(perpTransmittance)*thickness/dot(refract(V,X,eta),X)) = exp2(log2(perpTransmittance)*thickness*inversesqrt(1.f+(LdotX-1)*rcpEta))
				// Eta and `LdotX` is taken from the leaf BTDF node. With refractions from Dielectrics, we get just `1/LdotX`, for Delta Transmission we get `1/VdotN` since its the same
				typed_pointer_type<CSpectralVariableExpr> perpTransmittance = {};
				typed_pointer_type<CSpectralVariableExpr> thickness = {};

			protected:
				COPY_DEFAULT_IMPL

				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return ix ? perpTransmittance:thickness;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override
				{
					*(ix ? &perpTransmittance:&thickness) = block_allocator_type::_static_cast<CSpectralVariableExpr>(newChild);
				}
				
				inline std::string_view getChildName_impl(const uint8_t ix) const override {return "Perpendicular\\nTransmittance";}
				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;
		};
		// The "oriented" in the Etas means from frontface to backface, so there's no need to reciprocate them when creating matching BTDF for BRDF
		// @kept_secret TODO: Thin Film Interference Fresnel
		class CFresnel final : public IContributorDependant
		{
			public:
				inline uint8_t getChildCount() const override {return 2;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CFresnel);}
				inline CFresnel() = default;

				// Already pre-divided Index of Refraction, e.g. exterior/interior since VdotG>0 the ray always arrives from the exterior.
				typed_pointer_type<CSpectralVariableExpr> orientedRealEta = {};
				// Specifying this turns your Fresnel into a conductor one, note that currently these are disallowed on BTDFs!
				typed_pointer_type<CSpectralVariableExpr> orientedImagEta = {};
				// if you want to reuse the same parameter but want to flip the interfaces around
				uint8_t reciprocateEtas : 1 = false;

			protected:
				COPY_DEFAULT_IMPL

				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return ix ? orientedImagEta:orientedRealEta;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override
				{
					*(ix ? &orientedImagEta:&orientedRealEta) = block_allocator_type::_static_cast<CSpectralVariableExpr>(newChild);
				}

				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;

				inline bool reciprocatable() const override {return true;}
				inline void reciprocate(IExprNode* dst) const override
				{
					(*static_cast<CFresnel*>(dst) = *this).reciprocateEtas = ~reciprocateEtas;
				}

				inline std::string_view getChildName_impl(const uint8_t ix) const override {return ix ? "Imaginary":"Real";}
				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override;
		};
		// Compute Inifinite Scatter and extinction between two parallel infinite planes.
		// 
		// It's a specialization of what would be a layer of two identical smooth BRDF and BTDF with arbitrary Fresnel function and beer's
		// extinction, all applied on a per micro-facet basis (layering per microfacet, not whole surface) so the NDFs of two layers would be correlated.
		// 
		// We actually allow you to use different reflectance nodes R_u and R_b, the NDFs of both layers remain the same but Reflectance Functions differ.
		// Note that e.g. using different Etas for the Fresnel used for the top and bottom reflectance will result in a compound Fresnel!=1.0 
		// meaning that in such case you can no longer optimize the BTDF contributor into a DeltaTransmission but need a zero-roughness CookTorrance with
		// an Eta equal to the ratio of the first Eta over the second Eta (note that when they're equal the ratio is 1 which turns into Delta Trans).
		// This will require you to make an AST that "seems wrong" that is where neither of the Etas of the CFresnel nodes match the Cook Torrance one.
		// 
		// Because we split BRDF and BTDF into separate expressions, what this node computes differs depending on where it gets used:
		// Note the transformation in the equations at the end just makes the prevention of 0/0 or 0*INF same as for a non-extinctive equation, just check `R_u*R_b < Threshold`
		// 
		// BRDF: R_u + (1-R_u)^2 E^2 R_b Sum_{i=0}^{\Inf}{(R_b R_u E^2)^i} = R_u + (1-R_u)^2 E^2 R_b / (1 - R_u R_b E^2) = R_u + (1-R_u)^2 R_b / (E^-2 - R_u R_b)
		// --------------------
		// Top BRDF as multiplied with CThinInfiniteScatterCorrection node with `reflectanceTop`
		// BTDF matching the BRDF above
		// Bottom BRDF matching Top (but corellated so you always hit the same microfacet going back)
		// Null BRDF
		// Delta Transmission Beer extinction
		// Null BRDF
		// Top Smooth BRDF with `reflectanceBottom` applied to a Delta Reflection
		// ------------------
		// 
		// BTDF: (1-R_u) E (1-R_b) Sum_{i=0}^{\Inf}{(R_b R_u E^2)^i} = (1-R_u) E^2 (1-R_b) / (1 - R_u R_b E^2) = (1-R_u) (1-R_b) / (E^-2 - R_u R_b)
		// --------------------
		// Bottom BRDF as multiplied with CThinInfiniteScatterCorrection node with `reflectanceTop`
		// Null BRDF
		// Delta Transmission Beer extinction
		// Null BRDF
		// Top BRDF as multiplied with CThinInfiniteScatterCorrection node but with `reflectanceBottom` (but corellated so you always hit the same microfacet leading to no refraction)
		// ------------------
		// 
		// The obvious downside of using this node for transmission is that its impossible to get "milky" glass because a spread of refractions is needed
		class CThinInfiniteScatterCorrection final : public obj_pool_type::INonTrivial, public IExprNode
		{
			protected:
				COPY_DEFAULT_IMPL

				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return ix ? (ix>1 ? reflectanceBottom:extinction):reflectanceTop;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override
				{
					*(ix ? (ix>1 ? &reflectanceBottom:&extinction):&reflectanceTop) = newChild;
				}

				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;
				
				inline std::string_view getChildName_impl(const uint8_t ix) const override {return ix ? (ix>1 ? "reflectanceBottom":"extinction"):"reflectanceTop";}
				
			public:
				inline uint8_t getChildCount() const override final {return 3;}
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CThinInfiniteScatterCorrection);}

				// you can set the children later
				inline CThinInfiniteScatterCorrection() = default;

				typed_pointer_type<IExprNode> reflectanceTop = {};
				// optional
				typed_pointer_type<IExprNode> extinction = {};
				typed_pointer_type<IExprNode> reflectanceBottom = {};
		};
		//! Basic BxDF nodes
		// Every BxDF leaf node is supposed to pass WFT test and must not create energy, color and extinction is added on later via multipliers
		class IBxDF : public obj_pool_type::INonTrivial, public IContributor
		{
			public:
				// ?
		};
		// Delta Transmission is the only Special Delta Distribution Node, because of how useful it is for compiling Anyhit shaders, the rest can be done easily with:
		// - Delta Reflection -> Any Cook Torrance BxDF with roughness=0 attached as BRDF
		// - Smooth Conductor -> above multiplied with Conductor-Fresnel
		// - Smooth Dielectric -> Any Cook Torrance BxDF with roughness=0 attached as BRDF on both sides of a Layer and BTDF multiplied with Dielectric-Fresnel (no imaginary component)
		// - Thindielectric Correlated -> Cook Torrance BxDF multiplied with Dielectric-Fresnel as top BRDF and its reciprocal as the bottom, then Delta Transmission as BTDF with fresnels of similar Eta
		// - Thindielectric Uncorrelated -> BRDF and BTDF same as above, no bottom BRDF, then another layer with delta transmission BTDF
		//		For Smooth dielectrics it makes sense because fresnel of the interface is the same (microfacet equals macro surface normal, no confusion)
		//		For Rough its a little more complicated, but using the same BTDF still makes sense.
		//		Why? Because you enter all microfacets at once with a ray packet, and because their backfaces are correlated you don't refract.
		//		If we then assume that they're quite big in relation to the thickness, most of the Total Internal Reflection stays within the same microfacet slab.
		//		So for a single microfacet we have the thindielectric infinite TIR equation with `R_u = (1-Fresnel(VdotH))` and `R_b = (1-Fresnel(-LdotH))`,
		//		which when convolved with the VNDF (integral of complete TIR equation over all H) can be approximated by substitution of `...dotH` with `...dotN`.
		//		It also wouldn't matter if we dictate each slab have uniform perpendicular or geometric normal thickness, as the VNDF keeps projected surface area proportional to microfacet angle.
		//		So the average VdotH or LdotH are equal to NdotV and NdotL respectively, which doesn't guarantee average `inversesqrt(1-VdotH*VdotH)` equals `inversesqrt(1-NdotV*NdotV)` but difference is small.
		// - Plastic -> Similar to layering the above over Diffuse BRDF, its of uttmost importance that the BTDF is Delta Transmission.
		class CDeltaTransmission final : public IBxDF
		{
			public:
				inline uint8_t getChildCount() const override {return 0;}
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CDeltaTransmission);}
				inline CDeltaTransmission() = default;

			protected:
				COPY_DEFAULT_IMPL

				NBL_API2 ir_contributor_handle_t createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const;
		};
		//! Because of Schussler et. al 2017 every one of these nodes splits into 2 (if no L dependence) or 3 during canonicalization
		// Base diffuse node
		class COrenNayar final : public IBxDF
		{
			public:
				inline uint8_t getChildCount() const override {return 0;}
				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(COrenNayar);}
				inline COrenNayar()
				{
					ndParams.getDistribution() = CTrueIR::SBasicNDFParams::EDistribution::Invalid;
				}

				CTrueIR::SBasicNDFParams ndParams = {};

			protected:
				COPY_DEFAULT_IMPL

				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;

				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override;

				NBL_API2 ir_contributor_handle_t createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const;
		};
		// Supports anisotropy for all models
		class CCookTorrance final : public IBxDF
		{
			public:
				inline uint8_t getChildCount() const override {return 1;}

				inline const std::string_view getTypeName() const override {return TYPE_NAME_STR(CCookTorrance);}

				inline CCookTorrance()
				{
					ndParams.getDistribution() = CTrueIR::SBasicNDFParams::EDistribution::GGX;
				}
				
				inline bool isEtaReciprocal() const {return ndParams.params[2].padding[0];}
				inline void setEtaReciprocal(const bool value) {ndParams.params[2].padding[0] = value;}

				CTrueIR::SBasicNDFParams ndParams = {};
				// See the comments in CTrueIR about this on a matching class 
				typed_pointer_type<CSpectralVariableExpr> orientedRealEta = {};

			protected:
				COPY_DEFAULT_IMPL

				inline typed_pointer_type<IExprNode> getChildHandle_impl(const uint8_t ix) const override {return orientedRealEta;}
				inline void setChild_impl(const uint8_t ix, _typed_pointer_type<IExprNode> newChild) override {orientedRealEta = block_allocator_type::_static_cast<CSpectralVariableExpr>(newChild);}

				NBL_API2 bool invalid(const SInvalidCheckArgs& args) const override;
				
				inline bool reciprocatable() const override {return true;}
				inline void reciprocate(IExprNode* dst) const override
				{
					(*static_cast<CCookTorrance*>(dst) = *this).setEtaReciprocal(!isEtaReciprocal());
				}

				inline core::string getLabelSuffix() const override
				{
					return ndParams.getDistribution()!=CTrueIR::SBasicNDFParams::EDistribution::GGX ? "\\nNDF = Beckmann":"\\nNDF = GGX";
				}
				inline std::string_view getChildName_impl(const uint8_t ix) const override {return "Oriented η";}
				NBL_API2 void printDot(std::ostringstream& sstr, const core::string& selfID) const override;

				NBL_API2 ir_contributor_handle_t createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const;
		};
#undef COPY_DEFAULT_IMPL
#undef TYPE_NAME_STR

		//
		inline void reset()
		{
			getObjectPool().reset();
		}

		// basic utilities
		inline typed_pointer_type<CMul> createMul(const typed_pointer_type<IExprNode> lhs, const typed_pointer_type<IExprNode> rhs)
		{
			if (!lhs || !rhs) // acceptable premaute optimization
				return {};
			const auto mulH = getObjectPool().emplace<CMul>();
			auto* const mul = getObjectPool().deref(mulH);
			mul->lhs = lhs;
			mul->rhs = rhs;
			return mulH;
		}
		inline typed_pointer_type<IExprNode> createAdd(const typed_pointer_type<IExprNode> lhs, const typed_pointer_type<IExprNode> rhs)
		{
			if (!lhs)
				return rhs;
			if (!rhs)
				return lhs;
			const auto addH = getObjectPool().emplace<CAdd>();
			auto* const add = getObjectPool().deref(addH);
			add->lhs = lhs;
			add->rhs = rhs;
			return addH;
		}
		inline typed_pointer_type<IExprNode> createFMA(const typed_pointer_type<IExprNode> a, const typed_pointer_type<IExprNode> b, const typed_pointer_type<IExprNode> c)
		{
			return createAdd(createMul(a,b),c);
		}
		inline typed_pointer_type<IExprNode> createWeightedSum(const typed_pointer_type<IExprNode> x0, const typed_pointer_type<IExprNode> w0, const typed_pointer_type<IExprNode> x1, const typed_pointer_type<IExprNode> w1)
		{
			return createAdd(createMul(x0,w0),createMul(x1,w1));
		}
		inline typed_pointer_type<CComplement> createComplement(const typed_pointer_type<IExprNode> child)
		{
			if (!child)
				return {};
			const auto complH = getObjectPool().emplace<CComplement>();
			getObjectPool().deref(complH)->child = child;
			return complH;
		}

		// To quickly make a fresnel
		NBL_API2 typed_pointer_type<CFresnel> createNamedFresnel(const std::string_view name);
		inline typed_pointer_type<CFresnel> createConstantMonochromeRealFresnel(const hlsl::float32_t orientedRealEta)
		{
			auto& pool = getObjectPool();
			const auto fresnelH = pool.emplace<CFresnel>();
			if (auto* const fresnel=pool.deref(fresnelH); fresnel)
			{
				fresnel->orientedRealEta = pool.emplace<CSpectralVariableExpr>(uint8_t(1));
				if (auto* const var=pool.deref<CSpectralVariableExpr>(fresnel->orientedRealEta); var)
					var->setParameter(0,{.scale=orientedRealEta});
				else
					return {};
			}
			return fresnelH;
		}
		
		// To copy every node in the tree keeping same dedup, optionally can take an `orig` from another AST/pool and have the reciprocal copy over to our pool
		NBL_API2 typed_pointer_type<const IExprNode> deepCopy(const typed_pointer_type<const IExprNode> orig, const CFrontendIR* pSourceIR=nullptr);

		// To quickly make a matching backface BxDF from a frontface or vice versa, optionally can take an `orig` from another AST/pool and have the reciprocal copy over to our pool
		NBL_API2 typed_pointer_type<const IExprNode> reciprocate(const typed_pointer_type<const IExprNode> orig, const CFrontendIR* pSourceIR=nullptr);

		// a deep copy of the layer stack, wont copy the BxDFs, optionally can take an `orig` from another AST/pool and have the reciprocal copy over to our pool
		NBL_API2 typed_pointer_type<CLayer> copyLayers(const typed_pointer_type<const CLayer> orig, const CFrontendIR* pSourceIR=nullptr);
		// Reverse the linked list of layers and reciprocate their Etas, optionally can take an `orig` from another AST/pool and have the reciprocal copy over to our pool
		NBL_API2 typed_pointer_type<CLayer> reverse(const typed_pointer_type<const CLayer> orig, const CFrontendIR* pSourceIR=nullptr);

		// first query, we check presence of btdf layers all the way through the layer stack
		inline bool transmissive(const typed_pointer_type<const CLayer> rootHandle) const
		{
			auto& pool = getObjectPool();
			for (auto layer=pool.deref(rootHandle); layer; layer=pool.deref(layer->coated))
			{
				// it takes only one layer without transmission to break the chain
				if (!layer->btdf)
					return false;
			}
			return true;
		}

		// IMPORTANT: Two BxDFs are not allowed to be multiplied together.
		// NOTE: Right now all Spectral Variables are required to be Monochrome or 3 bucket fixed semantics, all the same wavelength.
		// Some things we can't check such as the compatibility of the BTDF with the BRDF (matching indices of refraction, etc.)
		NBL_API2 bool valid(const typed_pointer_type<const CLayer> rootHandle, system::logger_opt_ptr logger) const;

		// Each material comes down to this, after lowering to the true IR ir the indices into `ir->getMaterials()` are returned
		// We take the trees from the forest, and canonicalize them into our weird Domain Specific IR with Upside down expression trees.
		// Process:
		// 1. Decompression (duplicating nodes, etc.)
		// 2. Canonicalize Expressions (Transform into Sum-Product form, DCE, etc.)
		// 3. Split BTDFs (front vs. back part), reciprocate Etas
		// 4. Simplify and Hoist Layer terms (delta sampling property)
		// 5. Subexpression elimination
		// Further transforms in the IR can be done by invoking IR passes
		struct SAddMaterialsArgs
		{
			explicit inline operator bool() const {return !rootNodes.empty() && ir && result;}

			std::span<const typed_pointer_type<const CLayer>> rootNodes;
			CTrueIR* ir;
			CTrueIR::SMaterialHandle* result;
			system::logger_opt_ptr logger;
		};
		// returns the number of materials successfully converted
		inline uint32_t addMaterials(const SAddMaterialsArgs args) const
		{
			uint32_t retval = 0;
			if (!args)
			{
				args.logger.log("Invalid Arguments to `CTrueIR::addMaterials`",system::ILogger::ELL_ERROR);
				return retval;
			}
			SAdd2IRSession session = {args};
			auto outIt = args.result;
			for (const auto& rootH : args.rootNodes)
			{
				if (!rootH) // its a valid material (blackhole)
					*outIt = CTrueIR::BlackholeMaterialHandle;
				else if (valid(rootH,args.logger))
					*outIt = session.makeFinalIR(rootH,this);
				// now check for failure
				if (*outIt)
					retval++;
			}
			return retval;
		}

		// For Debug Visualization
		struct SDotPrinter final
		{
			public:
				inline SDotPrinter() = default;
				inline SDotPrinter(const CFrontendIR* ir) : m_ir(ir) {}
				// assign in reverse because we want materials to print in order
				inline SDotPrinter(const CFrontendIR* ir, std::span<const typed_pointer_type<const CLayer>> roots) : m_ir(ir), layerStack(roots.rbegin(),roots.rend())
				{
					// should probably size it better, if I knew total node count allocated or live
					visitedNodes.reserve(roots.size()<<3);
				}

				inline void reset(const CFrontendIR* ir)
				{
					visitedNodes.clear();
					layerStack.clear();
					exprStack.clear();
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
				// TODO: track layering depth and indent  accordingly?
				core::vector<typed_pointer_type<const CLayer>> layerStack;
				core::vector<typed_pointer_type<const IExprNode>> exprStack;
			private:
				const CFrontendIR* m_ir = nullptr;
		};

	protected:
		using CNodePool::CNodePool;
		
		struct SAdd2IRSession final
		{
			public:
				inline SAdd2IRSession(const SAddMaterialsArgs& _args) : args(_args)
				{
					tmpAST = CFrontendIR::create({.composed={.blockSizeKBLog2=10},.maxBlocks=64});
					// give slightly more memory to IR, since the AST tends to be a bit more compact
					tmpIR = CTrueIR::create({.composed={.blockSizeKBLog2=12},.maxBlocks=64});
				}

				NBL_API2 CTrueIR::SMaterialHandle makeFinalIR(const typed_pointer_type<const CLayer> rootH, const CFrontendIR* ast);

			private:
				inline void printSubtree(const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> nodeH)
				{
					assert(astPrinter.exprStack.empty());
					astPrinter.exprStack.push_back(nodeH);
					args.logger.log("Subtree Dot3 : \n%s\n",system::ILogger::ELL_DEBUG,astPrinter().c_str());
					assert(astPrinter.exprStack.empty());
					astPrinter.visitedNodes.clear();
				}
				inline void printIRLayer(const CTrueIR::typed_pointer_type<const CTrueIR::COrientedLayer> layerH, const CTrueIR* ir)
				{
					irPrinter.reset(ir);
					irPrinter.layerStack.push_back(layerH);
					args.logger.log("IR Layer Dot3 : \n%s\n",system::ILogger::ELL_DEBUG,irPrinter().c_str());
					irPrinter.visitedNodes.clear();
				}
				
				using oriented_material_t = CTrueIR::SMaterial::SOriented;
				NBL_API2 oriented_material_t makeOrientedMaterial(const CFrontendIR::typed_pointer_type<const CFrontendIR::CLayer> rootH, const CFrontendIR* _srcAST);

				NBL_API2 CTrueIR::typed_pointer_type<const CTrueIR::CContributorSum> makeContributors(const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> bxdfRootH);

				NBL_API2 CTrueIR::typed_pointer_type<const CTrueIR::CWeightedContributor> popContributor();

				// inputs to the addMaterials function
				const SAddMaterialsArgs& args;
				// for rewriting AST expressions
				core::smart_refctd_ptr<CFrontendIR> tmpAST;
				// for making IR nodes before we Merkle Hash them and remove duplicates (so main IR doesn't get bloated)
				core::smart_refctd_ptr<CTrueIR> tmpIR;
				// changes dynamically
				const CFrontendIR* srcAST;
				SDotPrinter astPrinter;
				CTrueIR::SDotPrinter irPrinter;
				bool btdfSubtree = false;
				// for going over layers in the AST
				core::vector<const CLayer*> layerStack;
#if 0 // dead and wrong
				// Some of the things we must canonicalize:
				// A ( f_0 (B + C) + D f_1 ) = f_0 B A + f_0 C A + f_1 D A
				// Expression nodes of the Frontend AST really come in 4 variants:
				// - add
				// - mul
				// - complement, which is equivalent to 1 ADD (-1 MUL x)
				// - function/other
				// BRDFs can appear only under ADD and MUL nodes in the AST not the function/other/complement, so if we want to canonicalize:
				// 1. The Add above can be ignored, we form full multiplication chain to the top
				// 2. Adds in sibling nodes (below the last add) cause us to have to add a factored copy to the IR
				// DFS from right-to-left (inverse order of adding children to stack), would cause us to keep postifxes of the multiplier chain every time we descend into ADD.
				// We want to essentially visit the parent ADD node again after dealing with its subtree (in-order traversal) then mul chain can be reset just to the parent.
				// If we perform DFS stack push left-to-right, we'll know the contributor already for all the leaf nodes if we push it onto the stack.
				// Then for all other leaf nodes we can accumulate them in the MUL chain, and adding their weighted contributor whenever we're back at an ADD node (be it the ancestor or sibling/cousin).
				// If the contributor is null or multiplied with a null we can keep draining the stack until we're back at its immediate parent ADD node.
				struct SContributor
				{
					// the "active" contributor, basically the leftmost item in the subbranch below and ADD
					CTrueIR::typed_pointer_type<const CTrueIR::IContributor> contributor;
				};
				core::vector<SContributor> contributorStack;
				// Every time we encounter an AST leaf we must add the current contributor together with all the factors multiplied together
				struct SFactor
				{
					using handle_t = CTrueIR::typed_pointer_type<const CTrueIR::IFactorLeaf>;
					// We only track multiplicative factors, we break down every BRDF equally into the canonical form
					handle_t handle;
					uint8_t negate : 1 = false;
					uint8_t monochrome : 1 = true;
					// extend later when allowing variable bucket count
					uint8_t liveSpectralChannels : 3 = 0b111;
				};
				// here we keep the multiplication chain unsorted so its each to add/remove nodes as we encounter them
				core::vector<SFactor> mulChain;
				// scratch for sorting the mul chain before adding a contributor
				core::vector<SFactor> mulChainSortScratch;
				// By maintaining a hash map of AST nodes which simplify to a Constant (unity, or zero, or other) we could resolve the issue of the `nonMulImmediateAncestorStackEnd`
				// which has us adding the same non-mul node multiple times to stack during the traversal.
				// However how much of that would be moving IR manipulation into the AST ?
				struct StackEntry
				{
					constexpr static inline uint64_t DontAddContributor = (0x1u<<10)-1;

					inline bool notVisited() const {return !visited;}

					CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> nodeH;
					// the ancestor ADD node to go back to if we hit a 0 MUL, or if our ADD or any other node becomes 0
					uint64_t nonMulImmediateAncestorStackEnd : 11 = 0;
					// the start of the `mulChain`, basically the bits that don't cross an Other node
					uint64_t mulChainBegin : 21 = 0;
					// the length of the `mulChain` at the time we first visited the node, so that we may reset the prefix back to what it was before continuing down another leg of the ADD
					uint64_t mulChainPrefixEnd : 21 = 0;
					// only relevant for Add nodes, the value tells you what to trim the contributor stack to
					uint64_t contributorStackLen : 10 = DontAddContributor;
					//
					uint64_t visited : 1 = false;
				};
				core::vector<StackEntry> exprStack;
#endif
		};

		inline core::string getNodeID(const typed_pointer_type<const INode> handle) const {return core::string("_")+std::to_string(handle.value);}
		inline core::string getLabelledNodeID(const typed_pointer_type<const INode> handle) const
		{
			const INode* node = getObjectPool().deref(handle);
			core::string retval = getNodeID(handle);
			retval += " [label=\"";
			retval += node->getTypeName();
			if (const auto* debug=getObjectPool().deref<const CDebugInfo>(node->debugInfo); debug && !debug->data().empty())
			{
				retval += "\\n";
				retval += std::string_view(reinterpret_cast<const char*>(debug->data().data()),debug->data().size()-1);
			}
			if (const auto* expr=dynamic_cast<const IExprNode*>(node); expr)
				retval += expr->getLabelSuffix();
			retval += "\"]";
			return retval;
		}
};

template class CTrueIR::CSpectralVariable<CFrontendIR::ISpectralVariableExpr>;
}

// specialize the `to_string
namespace nbl::system::impl
{
template<>
struct to_string_helper<nbl::asset::material_compiler3::CFrontendIR::IExprNode::Type>
{
	using type = nbl::asset::material_compiler3::CFrontendIR::IExprNode::Type;

	static inline std::string __call(const type value)
	{
		switch (value)
		{
			case type::Contributor:
				return "Contributor";
			case type::Mul:
				return "Mul";
			case type::Complement:
				return "Complement";
			case type::SpectralVariable:
				return "SpectralVariable";
			case type::Other:
				return "Other";
			default:
				break;
		}
		return "";
	}
};
}

namespace nbl::asset::material_compiler3
{

inline bool CFrontendIR::valid(const typed_pointer_type<const CLayer> rootHandle, system::logger_opt_ptr logger) const
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
		typed_pointer_type<const IExprNode> handle;
		uint8_t contribSlot;
		SubtreeContributorState contribState = SubtreeContributorState::Required;
		// using post-order like stack but with a pre-order DFS
		uint8_t visited = false;
	};
	core::stack<StackEntry> exprStack;
	// why a separate stack to the main one? Because we don't push siblings.
	core::vector<typed_pointer_type<const IExprNode>> ancestorPrefix;
	// TODO: unused yet
	core::unordered_set<typed_pointer_type<const INode>> visitedNodes;
	visitedNodes.reserve(128);
	//
	auto validateExpression = [&](const typed_pointer_type<const IExprNode> exprRoot, const bool isBTDF) -> bool
	{
		if (!exprRoot)
			return true;
		//
		const auto* root = getObjectPool().deref<const IExprNode>(exprRoot);
		if (!root)
		{
			logger.log("Node %u is not an Expression Node, it's %s",ELL_ERROR,exprRoot.value,getTypeName(exprRoot).data());
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
			if (entry.visited)
			{
				exprStack.pop();
				// this is the whole reason why we're using a post-order like stack
				ancestorPrefix.pop_back();
				continue;
			}
			else
			{
				exprStack.top().visited = true;
				// push self into prefix so children can check against it
				ancestorPrefix.push_back(entry.handle);
			}
			const auto* node = entry.node;
			const auto nodeType = node->getType();
			const bool nodeIsMul = nodeType==IExprNode::Type::Mul;
			const bool nodeIsAdd = nodeType==IExprNode::Type::Add;
			const auto childCount = node->getChildCount();
			bool takeOverContribSlot = true; // first add child can do this
			for (auto childIx=0; childIx<childCount; childIx++)
			{
				const auto childHandle = node->getChildHandle(childIx);
				if (const auto child=getObjectPool().deref(childHandle); child)
				{
					// Only Add nodes can have Contributors in any subtree, Mul only the first, and others can't have them at all. Especially don't allow the complementing of a BxDF!
					const bool noContribBelow = entry.contribState==SubtreeContributorState::Forbidden || childIx!=0 && !nodeIsAdd || nodeType==IExprNode::Type::Other || nodeType==IExprNode::Type::Complement;
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
					// detect cycles
					const auto found = std::find(ancestorPrefix.begin(),ancestorPrefix.end(),childHandle);
					if (found!=ancestorPrefix.end())
					{
						logger.log("Expression contains a cycle involving node %d of type %s",ELL_ERROR,childHandle,getTypeName(childHandle).data());
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
						entry.handle.value,node->getTypeName().data(),childIx,childHandle,getTypeName(childHandle).data()
					);
					return false;
				}
			}
			// check only after we know all children are OK
			if (node->invalid(invalidCheckArgs))
			{
				logger.log("Node %u of type %s is invalid!",ELL_ERROR,entry.handle.value,node->getTypeName().data());
				return false;
			}
			if (nodeType==IExprNode::Type::Contributor)
			{
				assert(entry.contribSlot<MaxContributors);
				assert(entry.contribState==SubtreeContributorState::Required);
				contributorsFound.set(entry.contribSlot);
			}
		}
		for (uint8_t i=0; i<contributorCount; i++)
		if (!contributorsFound.test(i))
		{
			logger.log("Expression starting with node %u does not have a Contributor Leaf Node in all of its additively distributive subtrees",ELL_ERROR,exprRoot.value);
			return false;
		}
		return true;
	};

	core::vector<const CLayer*> layerStack;
	auto pushLayer = [&](const typed_pointer_type<const CLayer> layerHandle)->bool
	{
		const auto* layer = getObjectPool().deref(layerHandle);
		if (!layer)
		{
			logger.log("Layer node %u of type %s not a `CLayer` node!",ELL_ERROR,layerHandle.value,getTypeName(layerHandle).data());
			return false;
		}
		auto found = std::find(layerStack.begin(),layerStack.end(),layer);
		if (found!=layerStack.end())
		{
			logger.log("Layer node %u is involved in a Cycle!",ELL_ERROR,layerHandle.value);
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