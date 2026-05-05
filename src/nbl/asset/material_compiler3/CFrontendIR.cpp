// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#define _NBL_ASSET_MATERIAL_COMPILER3_C_FRONTEND_IR_CPP_
#include "nbl/asset/material_compiler3/CFrontendIR.h"

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"


namespace nbl::asset::material_compiler3
{

constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
constexpr auto ELL_DEBUG = nbl::system::ILogger::E_LOG_LEVEL::ELL_DEBUG;
using namespace nbl::system;

bool CFrontendIR::CEmitter::invalid(const SInvalidCheckArgs& args) const
{
	// not checking validty of profile because invalid means no emission profile
	// check for NaN and non invertible matrix
	if (profile && !(hlsl::determinant(profileTransform)>hlsl::numeric_limits<hlsl::float32_t>::min))
	{
		args.logger.log("Emission Profile's Transform is not an invertible matrix!");
		return true;
	}
	return false;
}

bool CFrontendIR::CBeer::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->getObjectPool().deref(perpTransmittance))
	{
		args.logger.log("Perpendicular Transparency node of correct type must be attached, but is %u of type %s",ELL_ERROR,perpTransmittance,args.pool->getTypeName(perpTransmittance).data());
		return true;
	}
	if (const auto* const thick=args.pool->getObjectPool().deref(thickness); !thick || thick->getKnotCount()!=1)
	{
		args.logger.log("Monochromatic Thickness node must be attached, but is %u of type %s",ELL_ERROR,thickness,args.pool->getTypeName(thickness).data());
		return true;
	}
	return false;
}

bool CFrontendIR::CFresnel::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->getObjectPool().deref(orientedRealEta))
	{
		args.logger.log("Oriented Real Eta node of correct type must be attached, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	if (const auto imagEta=args.pool->getObjectPool().deref(orientedImagEta); imagEta)
	{
		if (args.isBTDF)
		{
			const auto knotCount = imagEta->getKnotCount();
			for (uint8_t i=0; i<knotCount; i++)
			{
				const auto& param = imagEta->getParameter(i);
				if (param.scale==0.f)
					continue;
				args.logger.log("Fresnels used for BTDFs cannot have Imaginary Eta, scale must be 0.f for all knots, is %.*e at knot %u",ELL_ERROR,param.scale,i);
				return true;
			}
		}
	}
	else if (orientedImagEta)
	{
		args.logger.log("Oriented Imaginary Eta node of incorrect type attached, but is %u of type %s",ELL_ERROR,orientedImagEta,args.pool->getTypeName(orientedImagEta).data());
		return true;
	}
	return false;
}

bool CFrontendIR::CThinInfiniteScatterCorrection::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->getObjectPool().deref(reflectanceTop))
	{
		args.logger.log("Top reflectance node of correct type must be attached, but is %u of type %s",ELL_ERROR,reflectanceTop,args.pool->getTypeName(reflectanceTop).data());
		return true;
	}
	if (extinction && !args.pool->getObjectPool().deref(extinction))
	{
		args.logger.log("Extinction node of incorrect type attached, but is %u of type %s",ELL_ERROR,extinction,args.pool->getTypeName(extinction).data());
		return true;
	}
	if (!args.pool->getObjectPool().deref(reflectanceBottom))
	{
		args.logger.log("Top reflectance node of correct type must be attached, but is %u of type %s",ELL_ERROR,reflectanceBottom,args.pool->getTypeName(reflectanceBottom).data());
		return true;
	}
	return false;
}

bool CFrontendIR::COrenNayar::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	return false;
}

bool CFrontendIR::CCookTorrance::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	if (args.isBTDF && !args.pool->getObjectPool().deref(orientedRealEta))
	{
		args.logger.log("Cook Torrance BTDF requires the Index of Refraction to compute the refraction direction, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	return false;
}


auto CFrontendIR::deepCopy(const typed_pointer_type<const IExprNode> orig, const CFrontendIR* pSourceIR) -> typed_pointer_type<const IExprNode>
{
	auto& dstPool = getObjectPool();
	// if not explicitly other, then its ours
	if (!pSourceIR)
		pSourceIR = this;
	const auto& srcPool = pSourceIR->getObjectPool();

	core::vector<typed_pointer_type<const IExprNode>> stack;
	stack.reserve(32);
	stack.push_back(orig);
	// use a hashmap to not explore whole DAG
	core::unordered_map<typed_pointer_type<const IExprNode>,typed_pointer_type<IExprNode>> substitutions;
	while (!stack.empty())
	{
		const auto entry = stack.back();
		const auto* const node = srcPool.deref(entry);
		if (!node) // this is an error
			return {};
		const auto childCount = node->getChildCount();
		if (auto& copyH=substitutions[entry]; !copyH)
		{
			for (uint8_t c=0; c<childCount; c++)
			{
				const auto childH = node->getChildHandle(c);
				if (auto child=srcPool.deref(childH); !child)
					continue; // this is not an error
				stack.push_back(childH);
			}
			// copy copies everything including child handles
			copyH = node->copy(this);
			if (!copyH)
				return {};
		}
		else
		{
			auto* const copy = dstPool.deref(copyH);
			copy->debugInfo = copyDebugInfo(node->debugInfo,pSourceIR);
			for (uint8_t c=0; c<childCount; c++)
			{
				const auto childH = node->getChildHandle(c);
				if (!childH)
					continue;
				auto found = substitutions.find(childH);
				assert(found!=substitutions.end());
				copy->setChild(c,found->second);
			}
			stack.pop_back();
		}
	}
	return substitutions[orig];
}

auto CFrontendIR::reciprocate(const typed_pointer_type<const IExprNode> orig, const CFrontendIR* pSourceIR) -> typed_pointer_type<const IExprNode>
{
	if (!orig)
		return {};

	auto& dstPool = getObjectPool();
	// if not explicitly other, then its ours
	if (!pSourceIR)
		pSourceIR = this;
	const auto& srcPool = pSourceIR->getObjectPool();

	core::vector<typed_pointer_type<const IExprNode>> stack;
	stack.reserve(32);
	stack.push_back(orig);
	// use a hashmap to not explore whole DAG
	core::unordered_set<typed_pointer_type<const IExprNode>> visited;
	// use a hashmap because of holes in child arrays
	core::unordered_map<typed_pointer_type<const IExprNode>,typed_pointer_type<IExprNode>> substitutions;
	while (!stack.empty())
	{
		const auto entry = stack.back();
		const auto* const node = srcPool.deref(entry);
		if (!node) // this is an error
			return {};
		const auto childCount = node->getChildCount();
		if (auto [it,inserted] = visited.insert(entry); inserted)
		{
			for (uint8_t c=0; c<childCount; c++)
			{
				const auto childH = node->getChildHandle(c);
				if (auto child=srcPool.deref(childH); !child)
					continue; // this is not an error
				stack.push_back(childH);
			}
		}
		else
		{
			const bool needToReciprocate = node->reciprocatable();
			bool needToCopy = pSourceIR!=this || needToReciprocate;
			// if one descendant has changed then we need to copy node
			if (!needToCopy)
			{
				uint8_t c = 0;
				for (; c<childCount; c++)
				{
					if (auto found=substitutions.find(node->getChildHandle(c)); found!=substitutions.end())
						break;
				}
				needToCopy = c!=childCount;
			}
			if (needToCopy)
			{
				const auto copyH = node->copy(this);
				// copy copies everything including child handles
				auto* const copy = dstPool.deref(copyH);
				if (!copy)
					return {};
				if (needToReciprocate)
					node->reciprocate(copy);
				// reciprocate might take full copies, and copy pointers across, so do all modifications after
				if (pSourceIR!=this)
					copy->debugInfo = copyDebugInfo(node->debugInfo,pSourceIR);
				// only changed children need to be set
				for (uint8_t c=0; c<childCount; c++)
				{
					const auto childH = node->getChildHandle(c);
					if (!childH)
						continue;
					if (auto found=substitutions.find(childH); found!=substitutions.end())
						copy->setChild(c,found->second);
				}
				substitutions.insert({entry,copyH});
			}
			stack.pop_back();
		}
	}
	// there was nothing to reciprocate in the expression stack
	if (substitutions.empty())
		return orig;
	return substitutions[orig];
}

auto CFrontendIR::copyLayers(const typed_pointer_type<const CLayer> orig, const CFrontendIR* pSourceIR) -> typed_pointer_type<CLayer>
{
	auto& dstPool = getObjectPool();
	// if not explicitly other, then its ours
	if (!pSourceIR)
		pSourceIR = this;
	const auto& srcPool = pSourceIR->getObjectPool();

	auto copyH = dstPool.emplace<CLayer>();
	{
		auto* outLayer = dstPool.deref(copyH);
		for (const auto* layer=srcPool.deref(orig); true; layer=srcPool.deref(layer->coated))
		{
			*outLayer = *layer;
			// need to deep copy the nodes
			if (pSourceIR!=this)
			{
				outLayer->debugInfo = copyDebugInfo(layer->debugInfo,pSourceIR);
				outLayer->brdfBottom = deepCopy(layer->brdfBottom,pSourceIR)._const_cast();
				outLayer->btdf = deepCopy(layer->btdf,pSourceIR)._const_cast();
				outLayer->brdfTop = deepCopy(layer->brdfTop,pSourceIR)._const_cast();
			}
			if (!layer->coated)
			{
				// terminate the new stack
				outLayer->coated = {};
				break;
			}
			// continue the new stack
			outLayer->coated = dstPool.emplace<CLayer>();
			outLayer = dstPool.deref(outLayer->coated);
		}
	}
	return copyH;
}

auto CFrontendIR::reverse(const typed_pointer_type<const CLayer> orig, const CFrontendIR* pSourceIR) -> typed_pointer_type<CLayer>
{
	auto& dstPool = getObjectPool();
	// if not explicitly other, then its ours
	if (!pSourceIR)
		pSourceIR = this;
	const auto& srcPool = pSourceIR->getObjectPool();

	// we build the new linked list from the tail
	auto copyH = dstPool.emplace<CLayer>();
	{
		auto* outLayer = dstPool.deref(copyH);
		typed_pointer_type<CLayer> underLayerH={};
		std::string debugData; debugData.reserve(4096);
		for (const auto* layer=srcPool.deref(orig); true; layer=srcPool.deref(layer->coated))
		{
			debugData = "REVERSED {";
			if (auto* debugInfo=srcPool.deref(layer->debugInfo); debugInfo)
			{
				const auto span = debugInfo->data();
				debugData += std::string_view(reinterpret_cast<const char*>(span.data()),span.size()-1);
			}
			debugData += '}';
			outLayer->debugInfo = dstPool.emplace<CDebugInfo>(debugData);
			outLayer->coated = underLayerH;
			// we reciprocate everything because numerator and denominator switch (top and bottom of layer stack)
			outLayer->brdfBottom = reciprocate(layer->brdfTop,pSourceIR)._const_cast();
			outLayer->btdf = reciprocate(layer->btdf,pSourceIR)._const_cast();
			outLayer->brdfTop = reciprocate(layer->brdfBottom,pSourceIR)._const_cast();
			if (!layer->coated)
				break;
			underLayerH = copyH;
			copyH = dstPool.emplace<CLayer>();
			outLayer = dstPool.deref(copyH);
		}
	}
	return copyH;
}

auto CFrontendIR::createNamedFresnel(const std::string_view name) -> typed_pointer_type<CFresnel>
{
	using complex32_t = hlsl::complex_t<float>;
	using spectral_complex_t = hlsl::portable_vector_t<complex32_t,3>;
	const static core::map<std::string_view,spectral_complex_t> creationLambdas = {
#define SPECTRUM_MACRO(R,G,B,X,Y,Z) spectral_complex_t(complex32_t(R,X),complex32_t(G,Y),complex32_t(B,Z))
		{"a-C",				SPECTRUM_MACRO(1.6855f, 1.065f, 1.727f,		0.0f, 0.009f, 0.0263f)},			// there is no "a-C", but "a-C:H; data from palik"
		{"Ag",				SPECTRUM_MACRO(0.059481f, 0.055090f, 0.046878f,		4.1367f, 3.4574f, 2.8028f)},
		{"Al",				SPECTRUM_MACRO(1.3404f, 0.95151f, 0.68603f,		7.3509f, 6.4542f, 5.6351f)},
		{"AlAs",			SPECTRUM_MACRO(3.1451f, 3.2636f, 3.4543f,		0.0012319f, 0.0039041f, 0.012940f)},
		{"AlAs_palik",		SPECTRUM_MACRO(3.145f, 3.273f, 3.570f,		0.0f, 0.000275f, 1.56f)},
		{"Au",				SPECTRUM_MACRO(0.21415f, 0.52329f, 1.3319f,		3.2508f, 2.2714f, 1.8693f)},
		{"Be",				SPECTRUM_MACRO(3.3884f, 3.2860f, 3.1238f,		3.1692f, 3.1301f, 3.1246f)},
		{"Be_palik",		SPECTRUM_MACRO(3.46f, 3.30f, 3.19f,		3.18f, 3.18f, 3.16f)},
		{"Cr",				SPECTRUM_MACRO(3.2246f, 2.6791f, 2.1411f,		4.2684f, 4.1664f, 3.9300f)},
		{"CsI",				SPECTRUM_MACRO(1.7834f, 1.7978f, 1.8182f,		0.0f, 0.0f, 0.0f)},
		{"CsI_palik",		SPECTRUM_MACRO(1.78006f, 1.79750f, 1.82315,		0.0f, 0.0f, 0.0f)},
		{"Cu",				SPECTRUM_MACRO(0.32075f,1.09860f,1.2469f,		3.17900f,2.59220f,2.4562)},
		{"Cu_palik",		SPECTRUM_MACRO(0.32000f, 1.04f, 1.16f,		3.15000f, 2.59f, 2.4f)},
		{"Cu20",			SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},  // out of range beyond 2.5 um refractiveindex.info and similar websites, so data applied is same as from palik's data
		{"Cu20_palik",	SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},
		{"d-C",			SPECTRUM_MACRO(2.4123f, 2.4246f, 2.4349f,		0.0f, 0.0f, 0.0f)},
		{"d-C_palik",		SPECTRUM_MACRO(2.4137f, 2.4272f, 2.4446f,		0.0f, 0.0f, 0.0f)},
		{"Hg",			SPECTRUM_MACRO(1.8847f, 1.4764f, 1.1306f,		5.1147f, 4.5410f, 3.9896f)},
		{"Hg_palik",		SPECTRUM_MACRO(1.850f, 1.460f, 1.100f,		5.100f, 4.600f, 3.990f)},
		//{"HgTe",			SPECTRUM_MACRO(,,,		,,)},						// lack of length wave range for our purpose https://www.researchgate.net/publication/3714159_Dispersion_of_refractive_index_in_degenerate_mercury_cadmium_telluride
		//{"HgTe_palik",		SPECTRUM_MACRO(,,,		,,)},					// the same in palik (wavelength beyond 2 um)			
		{"Ir",			SPECTRUM_MACRO(2.4200f, 2.0795f, 1.7965f,		5.0665f, 4.6125f, 4.1120f)},
		{"Ir_palik",		SPECTRUM_MACRO(2.44f, 2.17f, 1.87f,		4.52f, 4.24f, 3.79f)},
		{"K",			SPECTRUM_MACRO(0.052350f, 0.048270f, 0.042580f,		1.6732f, 1.3919f, 1.1195f)},
		{"K_palik",		SPECTRUM_MACRO(0.0525f, 0.0483f, 0.0427f,		1.67f, 1.39f, 1.12f)},
		{"Li",			SPECTRUM_MACRO(0.14872f, 0.14726f, 0.19236f,		2.9594f, 2.5129f, 2.1144f)},
		{"Li_palik",		SPECTRUM_MACRO(0.218f, 0.2093f, 0.229f,		2.848f, 2.369f, 2.226f)},
		{"MgO",			SPECTRUM_MACRO(1.7357f, 1.7419f, 1.7501f,		0.0f, 0.0f, 0.0f)},
		{"MgO_palik",		SPECTRUM_MACRO(1.7355f, 1.7414f, 1.74975f,		0.0f, 0.0f, 1.55f)},				 // Handbook of optical constants of solids vol 2 page 951, weird k compoment alone, no measurements and resoults
		{"Mo",			SPECTRUM_MACRO(0.76709f, 0.57441f, 0.46711f,		8.5005f, 7.2352f, 6.1383f)},	   	// https://refractiveindex.info/?shelf=main&book=Mo&page=Werner comparing with palik - weird
		{"Mo_palik",		SPECTRUM_MACRO(3.68f, 3.77f, 3.175f,		3.51f, 3.624f, 3.56f)},
		{"Na_palik",			SPECTRUM_MACRO(0.0522f, 0.061f, 0.0667f,		2.535f, 2.196f, 1.861f)},
		{"Nb",			SPECTRUM_MACRO(2.2775f, 2.2225f, 2.0050f,		3.2500f, 3.1325f, 3.0100f)},
		{"Nb_palik",			SPECTRUM_MACRO(2.869f, 2.9235f, 2.738f,		2.867f, 2.8764f, 2.8983f)},
		{"Ni_palik",			SPECTRUM_MACRO(1.921f, 1.744f, 1.651f,		3.615f, 3.168f, 2.753f)},
		{"Rh",			SPECTRUM_MACRO(2.8490f, 2.6410f, 2.4310f,		3.5450f, 3.3150f, 3.1190f)},
		{"Rh_palik",		SPECTRUM_MACRO(2.092f, 1.934f, 1.8256f,		5.472f, 4.902f, 4.5181f)},
		{"Se",			SPECTRUM_MACRO(1.4420f, 1.4759f, 1.4501f,		0.018713f, 0.10233f, 0.18418f)},
		{"Se_palik",		SPECTRUM_MACRO(3.346f, 3.013f, 3.068f,		0.6402f, 0.6711f, 0.553f)},
		{"SiC",			SPECTRUM_MACRO(2.6398f, 2.6677f, 2.7069f,		0.0f, 0.0f, 0.0f)},
		{"SiC_palik",		SPECTRUM_MACRO(2.6412f, 2.6684f, 2.7077f,		0.0f, 0.0f, 0.0f)},
		{"SnTe",			SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},			   // no data except palik's resources, so data same as palik
		{"SnTe_palik",		SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},
		{"Ta",			SPECTRUM_MACRO(1.0683f, 1.1379f, 1.2243f,		5.5047f, 4.7432f, 4.0988f)},
		{"Ta_palik",		SPECTRUM_MACRO(1.839f, 2.5875f, 2.8211f,		1.997f, 1.8683f, 2.0514f)},
		{"Te",			SPECTRUM_MACRO(4.1277f, 3.2968f, 2.6239f,		2.5658f, 2.8789f, 2.7673f)},
		{"Te_palik",		SPECTRUM_MACRO(5.8101f, 4.5213f, 3.3682f,		2.9428f, 3.7289f, 3.6783f)},
		{"ThF4",			SPECTRUM_MACRO(1.5113f, 1.5152f, 1.5205f,		0.0f, 0.0f, 0.0f)},
		{"ThF4_palik",		SPECTRUM_MACRO(1.520f, 1.5125f, 1.524f,		0.0f, 0.0f, 0.0f)},
		{"TiC",			SPECTRUM_MACRO(3.0460f, 2.9815f, 2.8864f,		2.6585f, 2.4714f, 2.3987f)},
		{"TiC_palik",		SPECTRUM_MACRO(3.0454f, 2.9763, 2.8674f,		2.6589f, 2.4695f, 2.3959f)},
		{"TiO2",			SPECTRUM_MACRO(2.1362f, 2.1729f, 2.2298f,		0.0f, 0.0f, 0.0f)},
		{"TiO2_palik",		SPECTRUM_MACRO(2.5925f, 2.676f, 2.78f,		0.0f, 0.0f, 0.0f)},
		{"VC",			SPECTRUM_MACRO(3.0033f, 2.8936f, 2.8138f,		2.4981f, 2.3046f, 2.1913f)},
		{"VC_palik",		SPECTRUM_MACRO(3.0038f, 2.8951f, 2.8184f,		2.4923f, 2.3107f, 2.1902f)},
		{"V_palik",		SPECTRUM_MACRO(3.512f, 3.671f, 3.2178f,		2.9337, 3.069f, 3.3667f)},
		{"VN",			SPECTRUM_MACRO(2.3429f, 2.2268f, 2.1550f,		2.4506f, 2.1345f, 1.8753f)},
		{"VN_palik",		SPECTRUM_MACRO(2.3418f, 2.2239f, 2.1539f,		2.4498f, 2.1371f, 1.8776f)},		
		{"W",		SPECTRUM_MACRO(0.96133f, 1.5474f, 2.1930f,		6.2902f, 5.1052f, 5.0325f)},
		{"none",			SPECTRUM_MACRO(0.f,0.f,0.f,		0.f,0.f,0.f)}
#undef SPECTRUM_MACRO
	};
	//
	const auto found = creationLambdas.find(name);
	if (found==creationLambdas.end())
		return {};
	//
	const auto frH = getObjectPool().emplace<CFrontendIR::CFresnel>();
	auto* fr = getObjectPool().deref(frH);
	fr->debugInfo = getObjectPool().emplace<CNodePool::CDebugInfo>(found->first);
	fr->orientedRealEta = getObjectPool().emplace<CSpectralVariableExpr>(3);
	{
		auto* const eta = getObjectPool().deref(fr->orientedRealEta);
		eta->setSemantics(CTrueIR::ISpectralVariable::ESemantics::Fixed3_SRGB);
		for (uint8_t c=0; c<3; c++)
			eta->setParameter(c,{.scale=found->second[c].real()});
	}
	fr->orientedImagEta = getObjectPool().emplace<CSpectralVariableExpr>(3);
	{
		auto* const eta = getObjectPool().deref(fr->orientedImagEta);
		eta->setSemantics(CTrueIR::ISpectralVariable::ESemantics::Fixed3_SRGB);
		for (uint8_t c=0; c<3; c++)
			eta->setParameter(c,{.scale=found->second[c].imag()});
	}
	return frH;
}

void CFrontendIR::SDotPrinter::operator()(std::ostringstream& str)
{
	str << "digraph {\n";

	auto drainExprStack = [&]()->void
	{			
		while (!exprStack.empty())
		{
			const auto entry = exprStack.back();
			exprStack.pop_back();
			const auto nodeID = m_ir->getNodeID(entry);
			str << "\n\t" << m_ir->getLabelledNodeID(entry);
			const auto* node = m_ir->getObjectPool().deref(entry);
			const auto childCount = node->getChildCount();
			if (childCount)
			{
				for (auto childIx=0; childIx<childCount; childIx++)
				{
					const auto childHandle = node->getChildHandle(childIx);
					if (const auto child=m_ir->getObjectPool().deref(childHandle); child)
					{
						str << "\n\t" << nodeID << " -> " << m_ir->getNodeID(childHandle) << "[label=\"" << node->getChildName_impl(childIx) << "\"]";
						const auto visited = visitedNodes.find(childHandle);
						if (visited!=visitedNodes.end())
							continue;
						exprStack.push_back(childHandle);
						visitedNodes.insert(childHandle);
					}
				}
			}
			// special printing
			node->printDot(str,nodeID);
		}
	};
	drainExprStack();
	
	while (!layerStack.empty())
	{
		const auto layerHandle = layerStack.back();
		layerStack.pop_back();
		// don't print layer nodes multiple times
		const auto visited = visitedNodes.find(layerHandle);
		if (visited!=visitedNodes.end())
			continue;
		visitedNodes.insert(layerHandle);
		const auto* layerNode = m_ir->getObjectPool().deref(layerHandle);
		//
		const auto layerID = m_ir->getNodeID(layerHandle);
		str << "\n\t" << m_ir->getLabelledNodeID(layerHandle);
		//
		if (layerNode->coated)
		{
			str << "\n\t" << layerID << " -> " << m_ir->getNodeID(layerNode->coated) << "[label=\"coats\"]\n";
			layerStack.push_back(layerNode->coated);
		}
		auto pushExprRoot = [&](const typed_pointer_type<const IExprNode> root, const std::string_view edgeLabel)->void
		{
			if (!root)
				return;
			// print the link from the layer to the expression
			str << "\n\t" << layerID << " -> " << m_ir->getNodeID(root) << "[label=\"" << edgeLabel << "\"]";
			// but not the expression again
			const auto visited = visitedNodes.find(root);
			if (visited!=visitedNodes.end())
				return;
			exprStack.push_back(root);
			visitedNodes.insert(root);
		};
		pushExprRoot(layerNode->brdfTop,"Top BRDF");
		pushExprRoot(layerNode->btdf,"BTDF");
		pushExprRoot(layerNode->brdfBottom,"Bottom BRDF");
		drainExprStack();
	}

	// TODO: print image views

	str << "\n}\n";
}

core::string CFrontendIR::ISpectralVariableExpr::getLabelSuffix() const
{
	if (getKnotCount()<2)
		return "";
	constexpr const char* SemanticNames[] =
	{
		"", 
		"\\nSemantics = Fixed3_SRGB",
		"\\nSemantics = Fixed3_DCI_P3",
		"\\nSemantics = Fixed3_BT2020",
		"\\nSemantics = Fixed3_AdobeRGB",
		"\\nSemantics = Fixed3_AcesCG"
	};
	return SemanticNames[static_cast<uint8_t>(getSemantics())];
}
// TODO: move `printDot` to CNodePool ?
void CFrontendIR::ISpectralVariableExpr::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	CTrueIR::printDotParameterSet(*pWonky(),getKnotCount(),sstr,selfID,{});
}

void CFrontendIR::CEmitter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	if (profile)
		profile.printDot(sstr,selfID);
	if (profile.view)
	{
		const auto transformNodeID = selfID+"_pTform";
		sstr << "\n\t" << transformNodeID << " [label=\"";
		printMatrix(sstr,profileTransform);
		sstr << "\"]";
		// connect up
		sstr << "\n\t" << selfID << " -> " << transformNodeID << "[label=\"Profile Transform\"]";
	}
}

void CFrontendIR::CFresnel::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
}

void CFrontendIR::COrenNayar::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndParams.printDot(sstr,selfID);
}

void CFrontendIR::CCookTorrance::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndParams.printDot(sstr,selfID);
}


//! IR making
CTrueIR::SMaterialHandle CFrontendIR::SAdd2IRSession::makeFinalIR(const typed_pointer_type<const CLayer> rootH, const CFrontendIR* ast)
{
	const auto& astPool = ast->getObjectPool();
	const auto* astRoot = astPool.deref<const CFrontendIR::CLayer>(rootH);
	// no material
	if (!astRoot)
		return CTrueIR::BlackholeMaterialHandle;
	tmpIR->reset();
	// reverse AST into another tree
	tmpAST->reset();
	const auto backRootH = tmpAST->reverse(rootH,ast);
	// do not attempt and cache these, a normal person will not send the same material over and over to convert to IR
	CTrueIR::SMaterial material = {
		.front = makeOrientedMaterial(rootH,ast),
		.back = makeOrientedMaterial(backRootH,tmpAST.get())
	};

	const auto errorLayer = args.ir->getBasicNodes().errorLayer;
	auto printLayer = [&](const typed_pointer_type<const CLayer> _rootH, const CFrontendIR* _ast)->void
	{
		astPrinter.reset(_ast);
		astPrinter.layerStack.push_back(_rootH);
		args.logger.log("Subtree Dot3 : \n%s\n",ELL_DEBUG,astPrinter().c_str());
		assert(astPrinter.layerStack.empty());
	};
	if (material.front.root==errorLayer)
	{
		args.logger.log("Failed to create Frontface Material",ELL_ERROR);
		printLayer(rootH,ast);
		return {};
	}
	if (material.back.root==errorLayer)
	{
		args.logger.log("Failed to create Backface Material for reversed AST",ELL_ERROR);
		printLayer(backRootH,tmpAST.get());
		return {};
	}

	auto retval = args.ir->addMaterial(material,tmpIR.get());
	if (retval)
	{
		// TODO: better debug info (e.g. concat all the layer info during `makeOrientedMaterial` via the `session` object
		if (const auto* debug=astPool.deref<const CDebugInfo>(astRoot->debugInfo); debug && !debug->data().empty())
		{
			material.debugInfo = args.ir->getObjectPool().emplace<CNodePool::CDebugInfo>(debug->data().data(),static_cast<uint32_t>(debug->data().size()));
		}
	}
	return retval;
}

auto CFrontendIR::SAdd2IRSession::makeOrientedMaterial(const CFrontendIR::typed_pointer_type<const CFrontendIR::CLayer> rootH, const CFrontendIR* _srcAST) -> oriented_material_t
{
	oriented_material_t retval = {};

	// TODO: cache root expressions for layers since they tend to be reused quite a bit 
	// HOWEVER the `tmpAST` gets cleared every call to `makeOrientedMaterial` so only cache within a `makeOrientedMaterial` call,
	// so clear when AST changes or keep separate caches for original AST and tmp.
//	core::unordered_map<const CFrontendIR::CLayer*,CTrueIR::COrientedLayer> layers;
//	core::unordered_map<const CFrontendIR::IExprNode*,CTrueIR::CContributorSum> topBRDFcache;
//	core::unordered_map<const CFrontendIR::IExprNode*,CTrueIR::CContributorSum> BTDFcache;

	srcAST = _srcAST;
	astPrinter.reset(srcAST);

	const auto& astPool = srcAST->getObjectPool();
	assert(layerStack.empty());
	auto clearLayerStackOnExit = core::makeRAIIExiter([this]()->void{layerStack.clear();});

	auto& irPool = tmpIR->getObjectPool();
	// go down through layers and enqueue them so the layers can be added in reverse
	for (const auto* layer=astPool.deref(rootH); layer; layer=astPool.deref(layer->coated))
	{
		// TODO: actually re-check the expressions for being null after optimization
		bool noTopReflection = !layer->brdfTop;
		bool noTransmission = !layer->btdf;
		// if there's literally nothing on the top level, you can't get to the next layer to retroreflect from it
		if (noTopReflection && noTransmission)
		{
			if (layer->coated)
				args.logger.log("Skipping current layer and farther ones due to no transmission and reflection",ELL_DEBUG);
			break;
		}
		layerStack.push_back(layer);
		// find out rest of the layers don't matter because they're blocked from being seen, its not a complete check
		if (noTransmission)
		{
			if (layer->coated)
				args.logger.log("Skipping remaining layers due to no transmission",ELL_DEBUG);
			break;
		}
	}

	const auto errorBxDF = tmpIR->getBasicNodes().errorBxDF;
	const auto errorLayer = tmpIR->getBasicNodes().errorLayer;
	// Some metadata needed for us
	bool layersBelowCanScatterBack = false;
	// then go in reverse and do the layers
	for (auto layerIt=layerStack.rbegin(); layerIt!=layerStack.rend(); layerIt++)
	{
		const auto& inLayer = *layerIt;
		// allocate a layer
		const auto layerH = irPool.emplace<CTrueIR::COrientedLayer>();
		{
			auto* const outLayer = irPool.deref(layerH);
			// process the top BRDF
			outLayer->brdfTop = makeContributors(inLayer->brdfTop);
			if (outLayer->brdfTop==errorBxDF)
				return {.root=errorLayer};
			// process the BTDF
			btdfSubtree = true;
			const auto btdfH = makeContributors(inLayer->btdf);
			btdfSubtree = false;
			// because we're oriented, the bottom brdf can't exist without a BTDF on top (there's no ray that can reach it from our oriented side)
			if (btdfH)
			{
				if (btdfH==errorBxDF)
					return {.root=errorLayer};
				const auto transmissionH = irPool.emplace<CTrueIR::CCorellatedTransmission>();
				{
					auto* const transmission = irPool.deref(transmissionH);
					transmission->btdf = btdfH;
					// Only if we have a layer below us capable of reflecting the ray back, do we care about the bottom BRDF (you can't hit it otherwise)
					if (layersBelowCanScatterBack)
					{
						transmission->brdfBottom = makeContributors(inLayer->brdfBottom);
						if (transmission->brdfBottom==errorBxDF)
							return {.root=errorLayer};
					}
					// we check if previous layer didn't get oprimized away
					if (retval.root)
						transmission->coated = retval.root;
				}
			}
		}
		retval.root = layerH;
		// Now optimize everything inserting it into the proper IR
		{
			// temporary debug print
			constexpr bool DebugBeforeAndAfterOpt = true;
			if constexpr(DebugBeforeAndAfterOpt)
			{
				args.logger.log("Before optimization:",ELL_DEBUG);
				printIRLayer(layerH,tmpIR.get());
			}
			// avoid O(Layer^2) scaling by processing bottom layers over and over
			if (!tmpIR->rewriteSingleLayer(retval.root,retval.metadata,tmpIR.get()))
			{
				args.logger.log("Failed to rewrite and optimize IR layer (printing layer and everything it coats):\n",ELL_ERROR);
				printIRLayer(layerH,tmpIR.get());
				return {.root=errorLayer};
			}
			// Now remember that our rewriter can optimize us into a blackhole/null layer
			if constexpr(DebugBeforeAndAfterOpt)
			{
				args.logger.log("After optimization:",ELL_DEBUG);
				printIRLayer(retval.root,tmpIR.get());
			}
			// Set this for the next layer after us, we are reflective or previous layer scatters back towards us and we don't block it
			if (auto* const outLayer=irPool.deref(retval.root); outLayer)
				layersBelowCanScatterBack = bool(outLayer->brdfTop) || layersBelowCanScatterBack && outLayer->firstTransmission;
			else
				layersBelowCanScatterBack = false;
		}
	}
	// last checks
	if (retval.root)
	{
//		assert(retval.metadata.capabilities|CTrueIR::SMaterial::SMetadata::ECapabilityBits::NotBlackhole);
	}
	else
	{
		assert(!layersBelowCanScatterBack);
	}
	return retval;
}

// TODO: this really needs a visited AST nodes (or at least subtrees) cache!
auto CFrontendIR::SAdd2IRSession::makeContributors(const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> bxdfRootH) -> CTrueIR::typed_pointer_type<const CTrueIR::CContributorSum>
{	
	CTrueIR::typed_pointer_type<const CTrueIR::CContributorSum> headH = {};
	if (!bxdfRootH)
		return headH;
	// temporary debug for WIP
	printSubtree(bxdfRootH);

	// error on exit
	const auto errorRetval = tmpIR->getBasicNodes().errorBxDF;
	auto printFailAndCleanupOnExit = core::makeRAIIExiter([&]()->void
		{
			if (headH!=errorRetval)
				return;
			printSubtree(exprStack.back().nodeH);
			args.logger.log("Within BxDF:\n",ELL_DEBUG);
			printSubtree(bxdfRootH);
			// no point pushing an error contributor, don't want a best effort compilation within a layer, don't want contributors missing or substituted
			exprStack.clear();
			mulChain.clear();
			contributorStack.clear();
		}
	);

	auto& astPool = srcAST->getObjectPool();
	auto& irPool = tmpIR->getObjectPool();
	// scratches are initialized
	assert(mulChain.empty());
	assert(contributorStack.empty());
	exprStack.push_back({.nodeH=bxdfRootH});
	CTrueIR::typed_pointer_type<CTrueIR::CContributorSum> tailH = {};
	// the mul node gets visited after children are made
	while (!exprStack.empty())
	{
		auto pEntry = &exprStack.back();
		const auto* const node = astPool.deref(pEntry->nodeH);
		assert(node);
		//
		using ast_expr_type_e = CFrontendIR::IExprNode::Type;
		const ast_expr_type_e astExprType = node->getType();
		const bool isContributor = astExprType==ast_expr_type_e::Contributor;
		//
		if (pEntry->notVisited())
		{
			pEntry->visited = true;
			if (isContributor)
			{
				const auto contributorH = static_cast<const IContributor*>(node)->createIRNode(btdfSubtree,srcAST,tmpIR.get());
				// TODO: recompute instead of compute
				if (!contributorH || irPool.deref(contributorH)->computeHash(irPool)==core::blake3_hash_t{})
				{
					args.logger.log("Failed to Create IR Contributor from AST",ELL_ERROR);
					return (headH=errorRetval);
				}
				contributorStack.push_back({.contributor=contributorH});
				exprStack.pop_back();
				// shouldn't invalidate but lets play it safe
				pEntry = nullptr;
			}
			else
			{
				const bool isAdd = astExprType==ast_expr_type_e::Add;
				if (isAdd)
				{
					// There are no other nodes above current Add other than Mul, then we must add any potential contributors immediately below
					if (pEntry->nonMulImmediateAncestorStackEnd)
						pEntry->addContributor = true;
					// A contributor can only be in the leftmost branch of an Add node's subtree, it also cannot be under any other node than Add or Mul.
					// Mostly making sure that Add nodes within complex weighting functions don't add contributors all of the sudden.
					// Its like a flood fill on the AST, where any non-Mul and non-Add node stops filling below.
					else if (auto& nonMulAncestor=exprStack[pEntry->nonMulImmediateAncestorStackEnd-1]; nonMulAncestor.addContributor)
					{
						// So use this knowledge to our advantage, however if we ever alias `addContributor` with anything else we'd need to check the ancestor type
						assert(astPool.deref(nonMulAncestor.nodeH)->getType()==ast_expr_type_e::Add);
						pEntry->addContributor = true;
						// Current Add node will perform the job of the parent add node for this subtree, it takes over
						nonMulAncestor.addContributor = false;
					}
				}
				const bool notMul = astExprType!=ast_expr_type_e::Mul;
				// go through children
				const auto childCount = node->getChildCount();
				// pushing back invalidates iterators
				const auto entry = *pEntry;
				pEntry = nullptr;
				// making sure we visit this node again each time a subtree of an Add node is done
				bool pushedAChild = false;
				// add in reverse so stack processes in order
				for (auto childIx=childCount; childIx; )
				{
					const auto childH = node->getChildHandle(--childIx);
					// to be able to figure out which substring of the prefix applies to current contributor
					const auto mulChainLen = notMul ? static_cast<uint16_t>(mulChain.size()):entry.mulChainLen;
					// to be able to go back to the non mul that is supposed to add our subtree
					const auto nonMulImmediateAncestorStackEnd = notMul ? static_cast<uint16_t>(exprStack.size()):entry.nonMulImmediateAncestorStackEnd;
					// skip null child
					if (!childH)
					{
						// because non-contributors don't pop themselves when coming off the stack, we have this
						assert(nonMulImmediateAncestorStackEnd);
						// what should happen depends if we're in the middle of a MUL subtree
						if (notMul)
						{
							// Generally this is the same as having an OpUndef, for Add we can skip adding this branch which we'll handle outside the loop
							continue;
						}
						else // kill subtree
						{
							const auto logLevel = system::ILogger::ELL_WARNING;
							args.logger.log("A null immediate child was encountered in the Mul node forming the subtree:\n",logLevel);
							printSubtree(entry.nodeH);
							args.logger.log("Forms a 0 constant factor across its ancestors in the mul chain, within the BxDF:\n",logLevel);
							printSubtree(bxdfRootH);
							mulChain.resize(mulChainLen);
							// this is so we don't pop a contributor if we happen to be in the right hand subtree relative to the top contributor in the stack and below an `Other` function
							if (entry.addContributor)
								contributorStack.pop_back();
							exprStack.resize(nonMulImmediateAncestorStackEnd);
							break;
						}
						continue;
					}
					// making sure we visit this node again each time a subtree of an Add node is done
					if (isAdd && pushedAChild)
					{
						auto& extraEntry = exprStack.emplace_back(entry);
						assert(extraEntry.visited);
						extraEntry.addContributor = true;
					}
					pushedAChild = true;
					// regular exploration
					exprStack.push_back({.nodeH=childH,.nonMulImmediateAncestorStackEnd=nonMulImmediateAncestorStackEnd,.mulChainLen=mulChainLen});
				}
				// didn't manage to add any child, dead ADD node
				if (isAdd && !pushedAChild)
					exprStack.back().addContributor = false;
			}
		}
		else
		{
			assert(!isContributor);
			// do stuff now
			switch (astExprType)
			{
				case ast_expr_type_e::Mul:
				{
					// silently skip
					break;
				}
				case ast_expr_type_e::Add:
				{
					if (pEntry->addContributor)
					{
						// we visited the leftmost subtrees first so this is the right order
						{
							auto* const tail = irPool.deref(tailH);
							tailH = irPool.emplace<CTrueIR::CContributorSum>();
							if (tailH)
								tail->rest = tailH;
							else
								headH = tailH;
						}
						// add current contributor with weight to BxDF Sum
						if (const auto contributor=popContributor(); contributor)
							irPool.deref(tailH)->product = contributor;
						else
						{
							args.logger.log("Failed to Pop the Contributor from the Stack, most likely failed to create the factor node chain.",ELL_ERROR);
							return (headH=errorRetval);
						}
					}
					// When we are done we need to reset the mul chain back to its original state, even if we don't add a contributor.
					// Because this could have been MUL BXDF ADD F_0, F_1 in Reverse Polish Notation
					mulChain.resize(pEntry->mulChainLen);
					break;
				}
				case ast_expr_type_e::SpectralVariable:
				{
					const auto varH = static_cast<const CSpectralVariableExpr*>(node)->createIRNode(srcAST,tmpIR.get());
					auto* const var = irPool.deref(varH);
					if (!var)
					{
						args.logger.log("Failed to create the Spectral Variable.",ELL_ERROR);
						return (headH=errorRetval);
					}
					// Note that we push onto the mul-chain even if the first non-MUL ancestor node is not an ADD (e.g. Fresnel or other complex function).
					// Also we may have ADD nodes which are disconnected from a contributor by having an Other ancestor
					// TODO: mulChain probably needs to get renamed into something more semantically sound
					auto& factor = mulChain.emplace_back();
					factor.handle = varH;
					const auto channels = var->getSpectralBins();
					factor.monochrome = channels<2;
					// we only want the mul chain length till the mul subtree's root
					assert(exprStack.size()>=pEntry->nonMulImmediateAncestorStackEnd);
					const auto mulChainBegin = pEntry->nonMulImmediateAncestorStackEnd ? (exprStack[pEntry->nonMulImmediateAncestorStackEnd-1].mulChainLen):uint16_t(0);
					// this wont be affected by sorting, but we need to check if our prefix is large enough
					if (mulChain.size()>=mulChainBegin+2)
						factor.liveSpectralChannels = mulChain[mulChain.size()-2].liveSpectralChannels;
					for (uint8_t c=0; c<channels; c++)
					{
						const auto* const cVar = var;
						if (!(cVar->getParameter(c).scale>std::numeric_limits<float>::min()))
							factor.liveSpectralChannels &= ~(0b1<<c);
					}
					if (factor.liveSpectralChannels==0)
					{
						const auto logLevel = system::ILogger::ELL_PERFORMANCE;
						args.logger.log("The Node parent of the subtree:\n",logLevel);
						printSubtree(pEntry->nodeH);
						args.logger.log("Forms a 0 constant factor across its ancestors in the mul chain, within the BxDF:\n",logLevel);
						printSubtree(bxdfRootH);
						// cancel exploration of all descendands of our first non mul node
						mulChain.resize(mulChainBegin);
						exprStack.resize(pEntry->nonMulImmediateAncestorStackEnd);
						pEntry = nullptr;
						// if there was nothing else in the tree, we can bail out the whole material
						if (exprStack.empty())
						{
							// if there are no expression above, there must not be a mul chain
							assert(mulChain.empty());
							// for the MUL subtree to form the whole tree, there needs to be only one contributor in the stack
							assert(contributorStack.size()==1);
							contributorStack.clear();
							args.logger.log("This MUL subtree is the only subtree, returning no contributors for this Tree.",logLevel);
							return headH;
						}
						// don't add this MUL subtree island's contributor, it won't exist
						if (auto& nonMulAncestor=exprStack.back(); nonMulAncestor.addContributor)
						{
							// if we ever alias `addContributor` with anything else we'd need to check the ancestor type
							assert(astPool.deref(nonMulAncestor.nodeH)->getType()==ast_expr_type_e::Add);
							nonMulAncestor.addContributor = false;
							contributorStack.pop_back();
						}
					}
					break;
				}
				default:
				{
					// Due to the genious design of the AST, two BxDFs cannot be multiplied, and the BxDF must be in the leftmost branch of an Add.
					// This makes it impossible to have an `IContributorDependant` which is involved in a MUL with more than 1 contributor.
					// But one contributor can be multiplied with many `IContributorDependant` which is not a problem because they all go on the mulChain as needed.
					args.logger.log("Unsupported AST Expression type \"%s\"",ELL_ERROR,system::to_string(astExprType).c_str());
					return (headH=errorRetval);
				}
			}
			exprStack.pop_back();
			// shouldn't invalidate but lets play it safe
			pEntry = nullptr;
		}
	}
	// There was never an ADD node
	if (!contributorStack.empty())
	{
		// only one contributor could have been encountered
		assert(contributorStack.size()==1);
		// add the contributor
		headH = irPool.emplace<CTrueIR::CContributorSum>();
		if (const auto contributor=popContributor(); contributor)
			irPool.deref(headH._const_cast())->product = contributor;
		else
		{
			args.logger.log("Failed to Pop the Single Contributor from the Stack, most likely failed to create the factor node chain.",ELL_ERROR);
			return (headH=errorRetval);
		}
		mulChain.clear();
	}
	// we got all the AST ADD nodes on the way back out
	assert(mulChain.empty());
	return headH;
}

auto CFrontendIR::ISpectralVariableExpr::createIRNode(const CFrontendIR* ast, CTrueIR* ir) const -> CTrueIR::typed_pointer_type<CTrueIR::CSpectralVariableFactor>
{
	auto& irPool = ir->getObjectPool();
	uint8_t realCount = 1;
	for (uint8_t c=0; c<getKnotCount(); c++)
	if (getParameter(c)!=getParameter(0))
		realCount = 3;
	return irPool.emplace<CTrueIR::CSpectralVariableFactor>(realCount,*this);
}

//
auto CFrontendIR::SAdd2IRSession::popContributor() -> CTrueIR::typed_pointer_type<const CTrueIR::CWeightedContributor>
{
	auto& irPool = tmpIR->getObjectPool();
	//
	const auto retval = irPool.emplace<CTrueIR::CWeightedContributor>();
	auto* weighted = irPool.deref(retval);
	weighted->contributor = contributorStack.back().contributor;
	contributorStack.pop_back();
	// we only want the mul chain length till the mul subtree's root, this is the prefix
	const size_t mulChainBegin = exprStack.empty() ? 0ull:(exprStack.back().mulChainLen);
	if (mulChainBegin<mulChain.size())
	{
		assert(mulChain.back().liveSpectralChannels);
		// now scan the stuff
		const CTrueIR::CFactorCombiner::SState combinerState = {
			.type = CTrueIR::CFactorCombiner::Type::Mul,
			.childCount = mulChain.size()-mulChainBegin
		};
		// every contributor node gets its own SORTED ancestor prefix
		mulChainSortScratch = {mulChain.begin()+mulChainBegin,mulChain.end()};
		// Multiplication Chain need to be sorted in a canonical order so its easier to spot them being the same
		auto sortMuls = [](const SFactor& lhs, const SFactor& rhs)->bool
		{
			// monochrome is cheaper
			if (lhs.monochrome!=rhs.monochrome)
				return lhs.monochrome;
			// not doing a complement/negation is cheaper
			if (lhs.negate!=rhs.negate)
				return rhs.negate;
			// DO NOT sort by live spectral channels
			// but want negations to show up together in the sorted list so easier to put back together
			return lhs.handle.value<rhs.handle.value;
		};
		std::sort(mulChainSortScratch.begin(),mulChainSortScratch.end(),sortMuls);
		//
		const auto factorH = irPool.emplace<CTrueIR::CFactorCombiner>(combinerState);
		{
			auto* const factor = irPool.deref(factorH);
			auto i = 0;
			for (const auto& mul : mulChainSortScratch)
			{
				assert(!mul.negate); // TODO: handle later
				factor->setChildHandle(i++,mul.handle);
			}
		}
		weighted->factor = factorH;
	}
	return retval;
}

// AST Node -> IR methods
auto CFrontendIR::CEmitter::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> ir_contributor_handle_t
{
	assert(!forBTDF);
	auto& irPool = ir->getObjectPool();
	const auto retval = irPool.emplace<CTrueIR::CEmitter>();
	if (auto* const contributor=irPool.deref(retval); contributor)
	{
		contributor->profile = profile;
		contributor->profileTransform = profileTransform;
	}
	return retval;
}

auto CFrontendIR::CDeltaTransmission::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> ir_contributor_handle_t
{
	assert(forBTDF);
	return ir->getObjectPool().emplace<CTrueIR::CDeltaTransmission>();
}

auto CFrontendIR::COrenNayar::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> ir_contributor_handle_t
{
	if (ndParams.getDistribution()!=CTrueIR::SBasicNDFParams::EDistribution::Invalid)
		return {};
	auto& irPool = ir->getObjectPool();
	const auto retval = irPool.emplace<CTrueIR::COrenNayar>();
	if (auto* const contributor = irPool.deref(retval))
		contributor->ndfParams = ndParams; // the padding abuse is the same between the classes
	return retval;
}

auto CFrontendIR::CCookTorrance::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> ir_contributor_handle_t
{
	if (ndParams.getDistribution()>CTrueIR::SBasicNDFParams::EDistribution::Beckmann)
		return {};
	auto& irPool = ir->getObjectPool();
	CTrueIR::typed_pointer_type<const CTrueIR::CSpectralVariableFactor> etaH = {};
	if (forBTDF)
	{
		const auto* const srcEta = ast->getObjectPool().deref(orientedRealEta);
		if (!srcEta)
			return {};
		etaH = srcEta->createIRNode(ast,ir);
		if (!etaH)
			return {};
	}
	const auto retval = irPool.emplace<CTrueIR::CCookTorrance>();
	if (auto* const ct=irPool.deref(retval); ct)
	{
		ct->ndfParams = ndParams; // the padding abuse is the same between the classes
		ct->orientedRealEta = etaH;
	}
	return retval;
}

template class CTrueIR::CSpectralVariable<CFrontendIR::ISpectralVariableExpr>;
}