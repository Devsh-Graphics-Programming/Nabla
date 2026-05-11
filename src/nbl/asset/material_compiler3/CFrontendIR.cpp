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

	auto& astPool = srcAST->getObjectPool();
	auto& irPool = tmpIR->getObjectPool();

	// Simple and inefficient Factor Gather with no Contributors:
	// - want canonical form, so vector of mul chains
	// - Solution, distribute/hoist the ADD over the MUL. So replace a `MUL ADD A B C` with `ADD MUL A C MUL B C`
	// - However if there's a long `MUL` chain, then hoist all the way to the top above that MUL, then every unexplored (pushed) node goes in the mul chain
	// - Bottom up or Top Down?
	// 	   + Bottom up requires descent to bottom anyway to find the ADD within MUL islands
	// 	   + Top down we can find the first ADD, then rewrite it and continue rewriting subexpressions
	// - Visit Cache? Every original AST node has its own mul chain set ?
	// 	   + If 0 factor gets slapped on, chain dies, gets removed from sum
	// - linked lists needed ? Or just MUL in tmpAST or IR?
	// 1. somehow mark when an AST `IExprNode` doesn't need exploring anymore
	// 2. explore on repeat
	// Variant with contributors:
	// - just sort the contributor last in the linked list, problem solved

	// Multiplication Chain need to be sorted in a canonical order so its easier to spot them being the same
	auto sortMuls = [](const SFactor& lhs, const SFactor& rhs)->bool
	{
		// contributor goes first
		if (lhs.isContributor()!=rhs.isContributor())
			return lhs.isContributor();
		// only one contributor allowed per chain!
		assert(!lhs.isContributor() && !rhs.isContributor());
		// monochrome is cheaper
		if (lhs.factor.monochrome!=rhs.factor.monochrome)
			return lhs.factor.monochrome;
		// then by handle
		return lhs.typeless.value<rhs.typeless.value;
	};
	// Holds the single `Product_j` of full expression in the form:
	// 	   f(w_i,w_o) = Sum_i^N Product_j^{N_i} h_{ij}(w_i,w_o) l_i(w_i,w_o)
	// Everything on the `irChain` multiplies together, everything on the `astStack` before the current top is our relative through a MUL node.
	// CONTRIBUTOR and OTHER are leaf nodes which don't add any children onto the `astStack`. 
	// ADD node (and COMPLEMENT which is a specialization of `ADD 1 (-X)`) duplicates the `astStack`, the ADD node at the top of the stack,
	// itself was in a MUL relationship with all of the preceding AST nodes which are not explored yet, so its children will also be.
	// The key is to not add both children of the ADD onto the same `astStack` because they themselves are not MUL together.
	struct SCanonicalProduct
	{
		// Deal with optimizing this later on, not sure if `DoublyLinkedList` is appropriate, maybe I'd need a `DoublyLinkedBeadedCurtain` data structure
		// also the mulChain needs to be sorted later on, and doubly linked list is PITA to sort
		core::vector<typed_pointer_type<const IExprNode>> astStack = {}; // its also a stack
		core::vector<SFactor> irChain = {};
		uint8_t hasContributor : 1 = false;
		// extend later when allowing variable bucket count
		uint8_t negate : 3 = 0b000;
		uint8_t liveSpectralChannels : 3 = 0b111;
	};
	// TODO: for the visited cache, we'd somehow need to cache a "2D slice" from this, meaning skipping the `irChain` prefix of a node
	// (would need linked list IR so span/front-back of an irChain subsection can be kept)
	// but also keeping how many of these separate irChains get spawned by the AST node
	core::list<SCanonicalProduct> canonicalSum;
	
	// error value
	const auto errorRetval = tmpIR->getBasicNodes().errorBxDF;

	// good code start
	assert(canonicalSum.empty());
	// add the first term to explore
	canonicalSum.emplace_back().astStack.emplace_back() = bxdfRootH;
	// error on exit
	auto it=canonicalSum.begin();
	auto printFailAndCleanupOnExit = core::makeRAIIExiter([&]()->void
		{
			if (headH!=errorRetval)
				return;
			printSubtree(it->astStack.back());
			args.logger.log("Within BxDF:\n",ELL_DEBUG);
			printSubtree(bxdfRootH);
			// no point emitting an error contributor, don't want a best effort compilation within a layer, don't want contributors missing or substituted
			canonicalSum.clear();
		}
	);
	CTrueIR::typed_pointer_type<CTrueIR::CContributorSum> tailH = {};
	for (; it!=canonicalSum.end(); it++)
	{
		auto& irChain = it->irChain;
		auto& astStack = it->astStack;
		while (!astStack.empty())
		{
			auto* pEntry = &astStack.back();
			const auto nodeH = *pEntry;
			const auto* const node = astPool.deref(nodeH);
			// only non null nodes get pushed onto the stack
			assert(node);
			// depending on the type we have different things to do
			using ast_expr_type_e = CFrontendIR::IExprNode::Type;
			const ast_expr_type_e astExprType = node->getType();
			constexpr auto ELL_WARNING = system::ILogger::ELL_WARNING;
			switch (astExprType)
			{
				case ast_expr_type_e::Contributor:
				{
					// This must be the only contributor, as a contributor can only be in the leftmost branch of a Mul node's subtree, it also cannot be under any other node than Add or Mul.
					assert(!it->hasContributor);
					//
					astStack.pop_back();
					// shouldn't invalidate iterator, but underline our vector changes
					pEntry = nullptr;
					// push onto the irChain
					const auto contributorH = static_cast<const IContributor*>(node)->createIRNode(btdfSubtree,srcAST,tmpIR.get());
					// TODO: recompute instead of compute hash
					if (!contributorH || irPool.deref(contributorH)->computeHash(irPool)==core::blake3_hash_t{})
					{
						args.logger.log("Failed to Create IR Contributor from AST",ELL_ERROR);
						return (headH=errorRetval);
					}
					it->hasContributor = true;
					irChain.emplace_back().contributor = {.handle=contributorH};
					break;
				}
				case ast_expr_type_e::Mul: // pop self but push the two children
				{
					// add in reverse so children are visited left to right
					for (auto c=node->getChildCount(); c;)
					{
						const auto childH = node->getChildHandle(--c);
						// if any child is null, kill everything
						if (auto* const child=astPool.deref(childH); !child)
						{
							printSubtree(nodeH);
							constexpr bool HardFail = true;
							if constexpr (HardFail)
							{
								args.logger.log("Undef node in a MUL",ELL_ERROR);
								return (headH=errorRetval);
							}
							else
							{
								args.logger.log("Undef node in a MUL, turning the MUL into a 0",ELL_WARNING);
								irChain.clear();
								astStack.clear();
								break;
							}
						}
						// replace self with second child
						if (c)
							*pEntry = childH;
						else
						{
							astStack.emplace_back() = childH;
							pEntry = nullptr;
						}
					}
					break;
				}
				case ast_expr_type_e::Add: // start one new chain
				{
					constexpr bool HardFail = true;
					bool takeOverChain = true;
					// add in reverse so children are visited left to right
					for (auto c=node->getChildCount(); c;)
					{
						const auto childH = node->getChildHandle(--c);
						// if child is null, skip it
						if (auto* const child=astPool.deref(childH); !child)
						{
							printSubtree(nodeH);
							if constexpr (HardFail)
							{
								args.logger.log("Undef node in an ADD",ELL_ERROR);
								return (headH=errorRetval);
							}
							else
							{
								args.logger.log("Undef node in an ADD, substituting with a 0\n",ELL_WARNING);
								continue;
							}
						}
						if (takeOverChain)
						{
							// second child (but first to be pushed) takes over our ASTchain spot
							*pEntry = childH;
							// does not invalidate but changes contents
							pEntry = nullptr;
							takeOverChain = false;
						}
						else
						{
							// first child (second to be pushed) copies our chains and carries on
							auto afterIt = it;
							// but we need to remove the first child from the copy's astStack
							canonicalSum.insert(++afterIt,*it)->astStack.pop_back();
						}
					}
					// didn't push anything, need to kill current sum term
					if constexpr (!HardFail)
					if (takeOverChain)
					{
						args.logger.log("Both children of ADD are NULL, deleting whole Contributor Sum Term",ELL_WARNING);
						irChain.clear();
						astStack.clear();
					}
					break;
				}
				case ast_expr_type_e::Complement: // MUL PREFIX ADD 1.0 CHILD
				{
					constexpr bool HardFail = true;
					if (const auto childH=node->getChildHandle(0); astPool.deref(childH))
					{
						// insert a copy of the current chain before the current with AST cleared, so chain stopped (gets us our MUL PREFIX 1.0 term)
						canonicalSum.insert(it,*it)->astStack.clear();
						// start a new chain with a copy of ours but negate it (now `it` points to the copy)
						it->negate ^= 0b111;
						// now the expression which was getting complemented needs to be on the AST stack so we don't visit the complement again
						it->astStack.back() = childH;
						// need to finalize the irChain, so go back to the entry stopped
						it--;
					}
					else // the child is OpUndef, replace complement with 1 node
					{
						printSubtree(nodeH);
						if constexpr (HardFail)
						{
							args.logger.log("Child of COMPLEMENT is null,",ELL_ERROR);
							return (headH=errorRetval);
						}
						else
						{
							args.logger.log("Child of COMPLEMENT is null, replacing with 1.0",ELL_WARNING);
							// keep irChain but stop AST exploration
							astStack.clear();
						}
					}
					break;
				}
				case ast_expr_type_e::SpectralVariable:
				{
					const auto varH = static_cast<const CSpectralVariableExpr*>(node)->createIRNode(srcAST,tmpIR.get());
					auto* const var = irPool.deref(varH);
					// no soft fail, the node wasn't null to begin with
					if (!var)
					{
						printSubtree(nodeH);
						args.logger.log("Failed to create the Spectral Variable.",ELL_ERROR);
						return (headH=errorRetval);
					}
					// see what channels are dead
					uint8_t liveMask = 0b000;
					const auto channels = var->getSpectralBins();
					for (uint8_t c=0; c<channels; c++)
					{
						const auto* const cVar = var;
						const auto scale = cVar->getParameter(c).scale;
						if (std::numeric_limits<float>::min()<=scale && scale<std::numeric_limits<float>::infinity())
							liveMask |= uint8_t(1)<<c;
					}
					// promote monochromatic mask
					const bool monochrome = channels<2;
					if (monochrome && liveMask)
						liveMask = 0b111;
					// combine masks and check if we're dead
					if ((it->liveSpectralChannels&=liveMask)==0)
					{
						const auto logLevel = system::ILogger::ELL_PERFORMANCE;
						// if we REALLY want to we can print all the nodes in the `irChain` so far, but then the irChain needs to track debug data from AST
						args.logger.log("Product turns to 0 by multiplying the chain so far with the Spectral Variable:\n",logLevel);
						printSubtree(nodeH);
						args.logger.log("Forms a 0 constant factor across its ancestors in the mul chain, within the BxDF:\n",logLevel);
						printSubtree(bxdfRootH);
						// kill whole term
						irChain.clear();
						astStack.clear();
						pEntry = nullptr;
						break;
					}
					irChain.emplace_back().factor = {.handle=varH,.monochrome=monochrome};
					break;
				}
				case ast_expr_type_e::Other:
				{
					// TODO: We need to start a new `canonicalSum` and save a link to it in the current IR
					printSubtree(nodeH);
					args.logger.log("Unsupported AST Expression type \"%s\"",ELL_ERROR,system::to_string(astExprType).c_str());
					return (headH=errorRetval);
					break;
				}
				default:
					assert(false);
					return (headH=errorRetval);
			}
		}
		// only once the astChain is empty, if the ir chain is empty skip it, but don't remove it because some Other AST node might need it
		if (irChain.empty())
		{
			printSubtree(bxdfRootH);
			// can't really print the subtree, because a lot of ADDs and MULs have to collude to produce this
			args.logger.log("Empty IR mul chain encountered, skipping...",system::ILogger::ELL_PERFORMANCE);
			continue;
		}
		// should have gotten removed beforehand
		assert(it->liveSpectralChannels);
		// sort the factors in the mulchain
		std::sort(irChain.begin(),irChain.end(),sortMuls);
		// make the combiner
		if (it->hasContributor)
		{
			// if we have a contributor first node in the sorted chain must be the contributor
			assert(irChain.front().isContributor());
			// we visited the leftmost subtrees first so this is the right order
			{
				auto* const tail = irPool.deref(tailH);
				// allocate new tail
				tailH = irPool.emplace<CTrueIR::CContributorSum>();
				// append it
				if (tail)
					tail->rest = tailH;
				else
					headH = tailH;
			}
			// now get the true tail
			{
				auto* const tail = irPool.deref(tailH);
				// and slap the mul chain onto it
				const auto weightedContribH = irPool.emplace<CTrueIR::CWeightedContributor>();
				auto* const weightedContrib = irPool.deref(weightedContribH);
				weightedContrib->contributor = irChain.front().contributor.handle;
				// now the mul chain
				if (irChain.size()>1)
				{
					CTrueIR::CFactorCombiner::SState combinerState = {
						.type = CTrueIR::CFactorCombiner::Type::Mul,
						.childCount = irChain.size()
					};
					// if we negate we need to add one more mul factor, otherwise nothing
					if (it->negate==0)
						--combinerState.childCount;
					const auto factorH = irPool.emplace<CTrueIR::CFactorCombiner>(combinerState);
					{
						auto* const factor = irPool.deref(factorH);
						auto i = 0;
						// monochrome negation
						if (it->negate && it->negate==0b111)
						{
							// TODO: do with premade node so sorting is correct (lowest)
							assert(false);
						}
						bool monochromeNodes = true;
						for (auto itChain=irChain.begin()+1; itChain!=irChain.end(); itChain++)
						{
							// only one contributor per chain is allowed
							assert(!itChain->isContributor());
							if (monochromeNodes)
							{
								// not monochrome for the first time
								if (!itChain->factor.monochrome)
								{
									monochromeNodes = false;
									// make the non-monochrome negation node
									if (it->negate && it->negate!=0b111)
									{
										// TODO: do with premade node so sorting is correct (lowest)
										assert(false);
									}
								}
							}
							else
							{
								// once you go spectral, you never go back
								assert(!itChain->factor.monochrome);
							}
							factor->setChildHandle(i++,itChain->factor.handle);
						}
					}
					weightedContrib->factor = factorH;
				}
				tail->product = weightedContribH;
			}
		}
		else
		{
			// TODO: make without a contributor
			assert(false);
		}
	}

#if 0  // old and dead code

	// Either the same contributor has to be added multiple times and then later cleaned up - the `addContributor` solution
	// or, we properly add a single contributor and MUL its different additive factors - easily done during a rewrite
	// The former is better because it allows canonicalization of multiplicative factors and gather of identical contributors. To implement it:
	// - every ADD node that's reachable via ADD and MUL must add a fully weighted contributor when its visited again by its child
	//   (all the factors on the way from the ROOT to the ADD node + MUL subtree island below)
	// - therefore every ADD node must know the length of the mul prefix from itself to a non-ADD & non-MUL ancestor so it can both
	//		+ be reset so the MUL island of the child node doesn't affect its siblings
	//		+ we're not counting parts of the mul chain crossing an Other node (which models an arbitrary function - e.g. fresnel)
	// - islands of MUL nodes must be able to jump back to their first non-MUL ancestor (ADD or Other) and prevent adding of any contributors weighted by them
	// - when an ADD node doesn't add contributors it should still get its MUL Island properly taken care of

// Rewrite Rules
// (a+b)(c+d) = ac+bd+bc+ad


	// scratches are initialized
	assert(mulChain.empty());
	assert(contributorStack.empty());
	exprStack.push_back({.nodeH=bxdfRootH});
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
			else if (astExprType==ast_expr_type_e::SpectralVariable)
			{
				// this is just easier to read than going through the circus below to do nothing
			}
			else
			{
				const bool isAdd = astExprType==ast_expr_type_e::Add;
				if (isAdd)
				{
					// There are no other nodes above current Add other than Mul, then we must add any potential contributors immediately below
					if (!pEntry->nonMulImmediateAncestorStackEnd)
						pEntry->contributorStackLen = contributorStack.size();
					// A contributor can only be in the leftmost branch of a Mul node's subtree, it also cannot be under any other node than Add or Mul.
					// Mostly making sure that Add nodes within complex weighting functions don't add contributors all of the sudden.
					// Its like a flood fill on the AST, where any non-Mul and non-Add node stops filling below.
					else if (auto& nonMulAncestor=exprStack[pEntry->nonMulImmediateAncestorStackEnd-1]; nonMulAncestor.contributorStackLen!=StackEntry::DontAddContributor)
					{
						// So use this knowledge to our advantage, however if we ever alias `addContributor` with anything else we'd need to check the ancestor type
						assert(astPool.deref(nonMulAncestor.nodeH)->getType()==ast_expr_type_e::Add);
						// Current Add node will perform the job of the parent add node for this subtree, it takes over
						pEntry->contributorStackLen = contributorStack.size();
						// if we're in the leftmost subtree then reset to current, otherwise also reset to current because the contributor stack size will remain unchanged
						nonMulAncestor.contributorStackLen = StackEntry::DontAddContributor;
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
				// we don't push onto mulchain in the loop so can compute once
				const auto mulChainBegin = astExprType==ast_expr_type_e::Other ? static_cast<uint16_t>(mulChain.size()):entry.mulChainBegin;
				const auto mulChainPrefixEnd = notMul ? static_cast<uint16_t>(mulChain.size()):entry.mulChainPrefixEnd;
				assert(mulChainBegin<=mulChainPrefixEnd);
				// add in reverse so stack processes in order
				for (auto childIx=childCount; childIx;)
				{
					const auto childH = node->getChildHandle(--childIx);
					// to be able to go back to the non mul that is supposed to add our subtree
					const auto nonMulImmediateAncestorStackEnd = notMul ? static_cast<uint16_t>(exprStack.size()):entry.nonMulImmediateAncestorStackEnd;
					// skip null child
					if (!childH)
					{
						// what should happen depends if we're in the middle of a MUL subtree
						if (notMul)
						{
							// Generally this is the same as having an OpUndef, for Add we can skip adding this branch's contributor which we implicitly handle with `pushedAChild`
							continue;
						}
						else // kill MUL subtree
						{
							struct SHandleZeroMulArgs
							{
								std::string_view preSubtreePrintMsg;
								std::string_view preTreePrintMsg;
								std::string_view bailMsg;
								// whether to assert contributor stack against <=1 or ==1
								bool sureContributorAlreadyAdded;
							};

							const auto logLevel = system::ILogger::ELL_WARNING;
							args.logger.log("A null immediate child was encountered in the Mul node forming the subtree:\n",logLevel);
							printSubtree(entry.nodeH);
							args.logger.log("Forms a 0 constant factor across its ancestors in the mul chain, within the BxDF:\n",logLevel);
							printSubtree(bxdfRootH);
							// this is mulChainLen to at the first non-MUL ancestor
							mulChain.resize(mulChainPrefixEnd);
							// go back to the ancestor
							exprStack.resize(nonMulImmediateAncestorStackEnd);
							// we only had the root above our MUL node, we need to bail out of the whole material
							if (exprStack.empty())
							{
								// if there are no expression above, there must not be a mul chain
								assert(mulChain.empty());
								// for the MUL subtree to form the whole tree, there needs to be only one contributor in the stack, but we might have not added it yet
								assert(contributorStack.size()<=1);
								contributorStack.clear();
								args.logger.log("This MUL subtree is the only subtree, and one of the nodes is UNDEF, returning no contributors for this Tree.",logLevel);
								return headH;
							}
							// If we're in the MUL node with an ancestor, there are twp posibilities as to what's above us:
							// - Other, in which case its enough just to clear out the mul chain prefix and the expression stack, any node in our subtree couldn't have pushed a contributor
							// - ADD, then we need to distinguish between an ADD which can add a contributor or cannot, and this we'll know by looking at the node we went back to
							if (auto& nonMulAncestor=exprStack.back(); nonMulAncestor.contributorStackLen!=StackEntry::DontAddContributor)
							{
								// if we ever alias `addContributor` with anything else we'd need to check the ancestor type
								assert(astPool.deref(nonMulAncestor.nodeH)->getType()==ast_expr_type_e::Add);
								// now if we're here it means there's an ADD node above us which meant to add the contributor, but due to what we do before the loop at the very start of the else case
								for (auto ancestorEnd=nonMulAncestor.nonMulImmediateAncestorStackEnd; ancestorEnd; ancestorEnd=exprStack[ancestorEnd-1].nonMulImmediateAncestorStackEnd)
								{
									// all other ADDs above the first ADD above our MUL node CANNOT add a contributor (mul chain incomplete)
									assert(exprStack[ancestorEnd-1].contributorStackLen==StackEntry::DontAddContributor);
								}
								// In the subtree of our current MUL to its immediate ADD there can only be one contributor, if even pushed at all yet
								assert(contributorStack.size()==(nonMulAncestor.contributorStackLen+1) || contributorStack.size()==nonMulAncestor.contributorStackLen);
								contributorStack.resize(nonMulAncestor.contributorStackLen);
								// prevent the ADD we go back to from adding the contributor
								nonMulAncestor.contributorStackLen = StackEntry::DontAddContributor;
							}
							break;
						}
						continue;
					}
					// making sure we visit this node again each time a subtree of an Add node is done
					if (isAdd && pushedAChild)
					{
						auto& extraEntry = exprStack.emplace_back(entry);
						assert(extraEntry.visited);
						// the `contributorStackLen` stays the same
					}
					pushedAChild = true;
					// regular exploration
					exprStack.push_back({
						.nodeH = childH,
						.nonMulImmediateAncestorStackEnd = nonMulImmediateAncestorStackEnd,
						.mulChainBegin = mulChainBegin,
						.mulChainPrefixEnd = mulChainPrefixEnd,
						.contributorStackLen = contributorStackLen
					});
				}
				// didn't manage to add any child, dead ADD node, even if couldn't add a contributor in the first place, just disable speculatively
				if (isAdd && !pushedAChild)
					exprStack.back().contributorStackLen = StackEntry::DontAddContributor;
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
					if (pEntry->contributorStackLen!=StackEntry::DontAddContributor)
					{
// TODO: assert that contributor stack is at most one element more than current node's known contributor reset length
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
// TODO: let us resize the stack, not `popContributor`
					}
					// When we are done we need to reset the mul chain back to its original state, even if we don't add a contributor.
					// Because this could have been `MUL BXDF ADD F_0, F_1` in Polish Notation, and when we go onto the `F_1` branch we need to remove the `F_0`'s factors from the chain. 
					mulChain.resize(pEntry->mulChainPrefixEnd);
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
					const bool belowAnOther = pEntry->mulChainBegin>0;
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
						mulChain.resize(pEntry->mulChainPrefixEnd);
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
						if (auto& nonMulAncestor=exprStack.back(); nonMulAncestor.contributorStackLen!=StackEntry::DontAddContributor)
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
					// Due to the genious design of the AST, two BxDFs cannot be multiplied, and the BxDF must be in the leftmost branch of a Mul.
					// We just need to handle an `IContributorDependant` that is involved in a MUL with more than 1 contributor.
					// For example MUL ADD BXDF_0 BXDF_1 FRESNEL_0
					// One contributor can be multiplied with many `IContributorDependant` which is not a problem because they all go on the mulChain as needed.
					args.logger.log("Unsupported AST Expression type \"%s\"",ELL_ERROR,system::to_string(astExprType).c_str());
// TODO: add readymade node to `mulChain`

					return (headH=errorRetval);
				}
			}
			exprStack.pop_back();
			// shouldn't invalidate but lets play it safe
			pEntry = nullptr;
		}
	}
#endif


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