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
				// reciprocate only after all data is complete
				if (needToReciprocate)
					copy->reciprocate();
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
//	core::unordered_map<const CFrontendIR::CLayer*,CTrueIR::CCorellatedTransmission> transmissions;
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
	// then go in reverse and do the layers bottom up
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
					// we check if previous layer didn't get optimized away
					if (retval.root)
						transmission->coated = retval.root;
				}
				outLayer->firstTransmission = tmpIR->hashNCache(transmissionH);
			}
		}
		retval.root = tmpIR->hashNCache(layerH);
		// Now optimize everything inserting it into the proper IR
		{
			// extra debug print
			constexpr bool DebugBeforeAndAfterOpt = false;
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
			if (const auto* const outLayer=irPool.deref(retval.root); outLayer)
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


//
auto CFrontendIR::SAdd2IRSession::makeContributors(const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> bxdfRootH) -> CTrueIR::typed_pointer_type<const CTrueIR::CContributorSum>
{	
	CTrueIR::typed_pointer_type<const CTrueIR::CContributorSum> headH = {};
	if (!bxdfRootH)
		return headH;

	auto& astPool = srcAST->getObjectPool();
	auto& irPool = tmpIR->getObjectPool();

	// basic checks
	assert(canonicalSum.empty());

	using add_ir_t = CTrueIR::CFactorCombiner;
	// Multiplication Chain need to be sorted in a canonical order so its easier to spot them being the same
	auto sortMuls = [&irPool](const CTrueIR::typed_pointer_type<const CTrueIR::IFactorLeaf>& lhs, const CTrueIR::typed_pointer_type<const CTrueIR::IFactorLeaf>& rhs)->bool
	{
		// we are only sorting non-null nodes
		assert(lhs && rhs);
		// monochrome is cheaper
		const bool lhsScalar = irPool.deref(lhs)->isScalar();
		if (lhsScalar!=irPool.deref(rhs)->isScalar())
			return lhsScalar;
		// then by handle
		return lhs.value<rhs.value;
	};
	//
	auto hashIfFunction = [&](const CTrueIR::typed_pointer_type<const CTrueIR::IFactorLeaf> leafH)->CTrueIR::typed_pointer_type<const CTrueIR::IFactorLeaf>
	{
		if (const auto funcH=irPool._dynamic_cast<const CTrueIR::IFunctionNode>(leafH); funcH)
		{
			// not finalized yet (I know what I'm doing with the const_cast)
			if (auto* const func=const_cast<CTrueIR::IFunctionNode*>(irPool.deref(funcH)); func->getHash()==core::blake3_hash_t::EmptyInput())
			{
				// replace the argument linked list with a single add
				const uint8_t argCount = func->getChildCount();
				for (uint8_t a=0; a<argCount; a++)
				if (const auto firstAddH=func->getChildHandle(a); firstAddH)
				{
					// for logging
					const uint32_t arg32 = a;
					// count the add terms (note last node always has a NULL second child, there's exactly as many nodes as there's terms to sum)
					add_ir_t::SState state = {.type=add_ir_t::Type::Add,.childCount=0};
					for (auto* binAdd=irPool.deref(firstAddH); true; )
					{
						assert(binAdd->getChildHandle(0));
						assert(binAdd->getChildCount()==2);
						assert(binAdd->getFinalType()==CTrueIR::INode::EFinalType::CFactorCombiner);
						assert(static_cast<const add_ir_t*>(binAdd)->getState().type==add_ir_t::Type::Add);
						if ((state.childCount++)>>CTrueIR::CFactorCombiner::MaxChildCountLog2)
						{
							args.logger.log("Too many Sum terms in a Function node arg linked list %d",ELL_ERROR,arg32);
							return {};
						}
						const auto rhsH = binAdd->getChildHandle(1);
						if (!rhsH)
							break;
						binAdd = irPool.deref(rhsH);
					}
					assert(state.childCount);
					// special case skip the ADD node
					if (state.childCount==1)
					{
						func->setChild(irPool,a,irPool.deref(firstAddH)->getChildHandle(0));
						continue;
					}
					const auto argH = irPool.emplace<add_ir_t>(state);
					auto* const arg = irPool.deref(argH);
					if (!arg)
					{
						args.logger.log("Failed to create ADD Node to replace the Function argument %d linked list",ELL_ERROR,arg32);
						return {};
					}
					// now add all the nodes in
					{
						uint8_t c = 0;
						for (auto* binAdd=irPool.deref(func->getChildHandle(a)); true; )
						{
							arg->setChild(irPool,c++,binAdd->getChildHandle(0));
							const auto rhsH = binAdd->getChildHandle(1);
							if (!rhsH)
								break;
							binAdd = irPool.deref(rhsH);
						}
					}
					// and hashNCache
					const auto uniqueArgH = tmpIR->hashNCache(argH);
					if (!uniqueArgH)
					{
						args.logger.log("Couldn't hash a `CTrueIR::IFunction` argument %d node",ELL_ERROR,arg32);
						return {};
					}
					// connect the node as the argument to a function
					func->setChild(irPool,a,uniqueArgH);
				}
				else // just use NULL as the child
					func->setChild(irPool,a,{});
			}
			return tmpIR->hashNCache(funcH._const_cast());
		}
		return leafH;
	};
	
	// error value
	const auto errorRetval = tmpIR->getBasicNodes().errorBxDF;
	// negationNode
	const auto scalarNegation = tmpIR->getBasicNodes().scalarNegation;

	// good code start
	assert(canonicalSum.empty());
	// add the first term to explore
	canonicalSum.emplace_back().astStack.emplace_back() = bxdfRootH;
	// error on exit
	auto printFailAndCleanupOnExit = core::makeRAIIExiter([&]()->void
		{
			if (headH!=errorRetval)
				return;
			args.logger.log("Within BxDF:",ELL_DEBUG);
			printSubtree(bxdfRootH);
			// no point emitting an error contributor, don't want a best effort compilation within a layer, don't want contributors missing or substituted
			canonicalSum.clear();
		}
	);
	for (auto it=canonicalSum.begin(); it!=canonicalSum.end(); )
	{
		auto& irChain = it->irChain;
		bool goBack = false;
		auto& astStack = it->astStack;
		while (!it->astStack.empty())
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
					assert(!it->contribSumH);
					//
					astStack.pop_back();
					// shouldn't invalidate iterator, but underline our vector changes
					pEntry = nullptr;
					// create the contributor node and mark as found
					const auto contributorH = tmpIR->hashNCache(static_cast<const IContributor*>(node)->createIRNode(btdfSubtree,srcAST,tmpIR.get()));
					if (contributorH)
					{
						const auto weightedH = irPool.emplace<CTrueIR::CWeightedContributor>();
						if (weightedH)
						{
							it->contribSumH = irPool.emplace<CTrueIR::CContributorSum>();
							auto* const sumTerm = irPool.deref(it->contribSumH);
							if (sumTerm)
							{
								irPool.deref(weightedH)->contributor = contributorH;
								// note we don't hashNCache the `weightedH` node or any that has it as its child
								sumTerm->product = weightedH;
								it->hasContributor = true;
							}
						}
					}
					if (!it->contribSumH)
					{
						args.logger.log("Failed to Create IR Contributor from AST",ELL_ERROR);
						printSubtree(nodeH);
						return (headH=errorRetval);
					}
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
							afterIt = canonicalSum.insert(++afterIt,*it);
							// need to visit a different child
							afterIt->astStack.back() = childH;
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
					// not that since second child took over our chain and AST stack, we can continue iteration at `it` and current `astStack`
					break;
				}
				case ast_expr_type_e::Complement: // MUL PREFIX ADD 1.0 -CHILD
				{
					constexpr bool HardFail = true;
					if (const auto childH=node->getChildHandle(0); astPool.deref(childH))
					{
						auto afterIt = it; afterIt++;
						// insert a copy of the current chain AFTER the current
						afterIt = canonicalSum.insert(afterIt,*it);
						// with AST cleared, so chain is stopped (gets us our MUL PREFIX 1.0 term)
						afterIt->astStack.clear();
						// negate our current chain (get use our MUL PREFIX -CHILD)
						it->negate ^= 0b111;
						// now the expression which was getting complemented needs to be on the AST stack so we don't visit the complement again
						*pEntry = childH;
						// now we'll continue with the longer and negated product
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
					//
					astStack.pop_back();
					// shouldn't invalidate iterator, but underline our vector changes
					pEntry = nullptr;
					//
					const auto varH = tmpIR->hashNCache(static_cast<const CSpectralVariableExpr*>(node)->createIRNode(srcAST,tmpIR.get()));
					const auto* const var = irPool.deref(varH);
					// no soft fail, the node wasn't null to begin with
					if (!var)
					{
						args.logger.log("Failed to create the Spectral Variable.",ELL_ERROR);
						printSubtree(nodeH);
						return (headH=errorRetval);
					}
					// see what channels are dead
					uint8_t liveMask = 0b000;
					const auto channels = var->getSpectralBins();
					for (uint8_t c=0; c<channels; c++)
					{
						const auto* const cVar = var;
						const auto absScale = hlsl::abs(cVar->getParameter(c).scale);
						if (std::numeric_limits<float>::min()<=absScale && absScale<std::numeric_limits<float>::infinity())
							liveMask |= uint8_t(1)<<c;
					}
					// promote monochromatic mask
					const bool monochrome = channels<2;
					if (monochrome && liveMask)
						liveMask = 0b111;
					// combine masks and check if we're dead
					if ((it->liveSpectralChannels&=liveMask)==0)
					{
						// this prints a lot of these messages for 0 parameters of fresnel
						constexpr bool ExtraDebug = false;
						if constexpr (ExtraDebug)
						{
							const auto logLevel = system::ILogger::ELL_PERFORMANCE; 
							// if we REALLY want to we can print all the nodes in the `irChain` so far, but then the irChain needs to track debug data from AST
							args.logger.log("Product turns to 0 by multiplying the chain so far with the Spectral Variable:\n",logLevel);
							printSubtree(nodeH);
							args.logger.log("Forms a 0 constant factor across its ancestors in the mul chain, within the BxDF:\n",logLevel);
							printSubtree(bxdfRootH);
						}
						// kill whole term
						irChain.clear();
						astStack.clear();
						pEntry = nullptr;
						break;
					}
					irChain.emplace_back() = varH;
					break;
				}
				case ast_expr_type_e::Other: // the way this is written, we'd really benefit from a more "intimate" enforcing of the structure of Fresnel, Beer, etc. being identical between AST and IR
				{
					//
					astStack.pop_back();
					// shouldn't invalidate iterator, but underline our vector changes
					pEntry = nullptr;
					// do not hash yet! the children are invalid!
					const auto funcH = static_cast<const IFunctionNode*>(node)->createIRNode(btdfSubtree,srcAST,tmpIR.get());
					auto* const func = irPool.deref(funcH);
					if (!func)
					{
						args.logger.log("Failed to create Other IR Function from AST",ELL_ERROR);
						printSubtree(nodeH);
						return (headH=errorRetval);
					}
					//
					const uint8_t argCount = node->getChildCount();
					// need to start a new `canonicalSum` for every input term of the function
					// insert back to front so we get good `it` at the end
					for (uint8_t c=argCount; c;)
					{
						const auto astChild = node->getChildHandle(--c);
						// only add sum expressions for non-null children
						if (!astChild)
							continue;
						// start a new mul chain for the arg
						goBack = true;
						it = canonicalSum.emplace(it);
						it->astStack.push_back(astChild);
						it->funcH = funcH;
						it->targetArg = c;
					}
					// note that the `irChain` is that of the `it` before we started changing it in the loop above
					irChain.emplace_back() = funcH;
					break;
				}
				default:
					assert(false);
					printSubtree(nodeH);
					return (headH=errorRetval);
			}
			if (goBack)
				break;
		}
		// we're supposed to go back to something inserted before the sum term we started to process
		if (goBack)
			continue;
		// basic error checking, need to have a target node to add to
		if (!it->contribSumH || !it->funcH)
		{
			args.logger.log("This Mul chain has no Contributor Node to partner with or Function Node to be an argument of!",ELL_ERROR);
			return (headH=errorRetval);
		}
		// not printing an extra error message, should have already printed when found we mul to 0
		if (it->liveSpectralChannels==0)
		{
			it++;
			continue;
		}
		// sort the factors in the mulchain
		std::sort(irChain.begin(),irChain.end(),sortMuls);
		//
		bool monochromeFactor = true;
		// create the factor node - empty node means mul with 1.0 (no mul)
		CTrueIR::typed_pointer_type<const CTrueIR::IFactor> uniqueFactorH = {};
		if (!irChain.empty())
		{
			CTrueIR::CFactorCombiner::SState combinerState = {
				.type = CTrueIR::CFactorCombiner::Type::Mul,
				.childCount = irChain.size()
			};
			// one more needed for a negation of any channel
			if (it->negate)
				++combinerState.childCount;
			// no point making a mul node
			if (combinerState.childCount==1)
			{
				const auto leafH = irChain.front();
				monochromeFactor = irPool.deref(leafH)->isScalar();
				uniqueFactorH = hashIfFunction(leafH);
				if (!uniqueFactorH)
				{
					args.logger.log("Couldn't hash a `CTrueIR::IFunction` node",ELL_ERROR);
					return (headH=errorRetval);
				}
			}
			else
			{
				const auto factorH = irPool.emplace<CTrueIR::CFactorCombiner>(combinerState);
				if (auto* const factor=irPool.deref(factorH); factor)
				{
					auto i = 0;
					// monochrome negation
					if (it->negate==0b111)
						factor->setChildHandle(i++,scalarNegation);
					for (auto itChain=irChain.begin(); itChain!=irChain.end(); itChain++)
					{
						auto leafH = *itChain;
						const auto* const leaf = irPool.deref(leafH);
						assert(leaf);
						if (monochromeFactor)
						{
							// not monochrome for the first time
							if (!leaf->isScalar())
							{
								monochromeFactor = false;
								// make the non-monochrome negation node, not using premade cause that would tie me up with spectral buckets
								if (it->negate && it->negate!=0b111)
								{
									const auto negationH = irPool.emplace<CTrueIR::CSpectralVariableFactor>(uint8_t(3));
									if (auto* const negation=irPool.deref(negationH); negation)
									{
										for (uint8_t c=0; c<3; c++)
											negation->setParameter(c,{.scale=bool((it->negate>>c)&0x1u) ? (-1.f):1.f});
									}
									const auto uniqueH = tmpIR->hashNCache(negationH);
									if (!uniqueH)
									{
										args.logger.log("Couldn't create a unique spectral negation node",ELL_ERROR);
										return (headH=errorRetval);
									}
									factor->setChildHandle(i++,uniqueH);
								}
							}
						}
						else
						{
							// once you go spectral, you never go back
							assert(!leaf->isScalar());
						}
						// set the child
						if (leafH=hashIfFunction(leafH); leafH)
							factor->setChildHandle(i++,leafH);
						else
						{
							args.logger.log("Couldn't hash a `CTrueIR::IFunction` node",ELL_ERROR);
							return (headH=errorRetval);
						}
					}
				}
				uniqueFactorH = tmpIR->hashNCache(factorH);
				if (!uniqueFactorH)
				{
					args.logger.log("Couldn't allocate or hash a `CTrueIR::CFactorCombiner` Mul node",ELL_ERROR);
					return (headH=errorRetval);
				}
			}
		}
		// link up the factor node
		if (it->hasContributor)
		{
			auto* const sumTerm = irPool.deref(it->contribSumH);
			assert(sumTerm);
			{
				const auto weightedContribH = sumTerm->product._const_cast();
				// attach the factor
				irPool.deref(weightedContribH)->factor = uniqueFactorH;
				const auto uniqueWeightedH = tmpIR->hashNCache(weightedContribH);
				assert(uniqueWeightedH);
				if (!uniqueWeightedH)
				{
					args.logger.log("Couldn't hash a `CTrueIR::CWeightedContributor` node",ELL_ERROR);
					return (headH=errorRetval);
				}
				// replace the weighted contributor with unique hashed
				sumTerm->product = uniqueWeightedH;
				// note that we hold off on hashing the `sumTerm`, cause its subject to change (more terms attached to tail)
			}
			// now append it to previous with a contributor, if it exists
			for (auto prev=it; prev!=canonicalSum.begin(); prev--)
			{
				if (prev==it) // skip first item, its our current one
					continue;
				if (prev->hasContributor)
				{
					irPool.deref(prev->contribSumH)->rest = it->contribSumH;
					break;
				}
			}
		}
		else
		{
			auto* const func = irPool.deref(it->funcH);
			assert(func);
			// first time a function gets used, it gets hashedNCache-d, which finalizes it, can't be mutating its state afterwards!
			assert(func->getHash()==core::blake3_hash_t::EmptyInput());
			// make sure to update that factor is not scalar if we have any non-scalar factor term
			if (!monochromeFactor)
				func->scalar = false;
			// allocate space for 2 add factors, we'll clean up later to have no dangling binop
			const auto addTailH = irPool.emplace<add_ir_t>(add_ir_t::SState{.type=add_ir_t::Type::Add,.childCount=2});
			if (!addTailH)
			{
				args.logger.log("Failed to create ADD Node to serve as root for a function arg %d",ELL_ERROR,uint32_t(it->targetArg));
				return (headH=errorRetval);
			}
			auto* const addTail = irPool.deref(addTailH);
			addTail->setChildHandle(0,uniqueFactorH);
			// if there's already an add node waiting for us
			if (const auto prevH=func->getChildHandle(it->targetArg); prevH)
			{
				assert(irPool.deref(prevH)->getFinalType()==CTrueIR::INode::EFinalType::CFactorCombiner);
				// attach previous node as second sum term
				addTail->setChildHandle(1,block_allocator_type::_static_cast<CTrueIR::CFactorCombiner>(prevH));
			}
			// connect the node as the argument to a function
			func->setChild(irPool,it->targetArg,addTailH);
		}
		// progress onto next term
		it++;
	} 
	// now hash in reverse
	for (auto it=canonicalSum.rbegin(); it!=canonicalSum.rend(); it++)
	{
		// last contributor becomes the new head node, also hashNCache
		if (it->hasContributor)
		{
			const auto uniqueH = tmpIR->hashNCache(it->contribSumH);
			// replace reference to previous tail with hashed and cached tail
			if (headH)
				irPool.deref(headH._const_cast())->rest = uniqueH;
			headH = uniqueH;
		}
		// thankfully args to functions come before the expressions the functions are used in
	}
	// headH set by last in loop with contributor, so first with contributor in the sum linked list
	canonicalSum.clear();
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

//! record the original AST child handle in the IR node child, cleanup later
auto CFrontendIR::CBeer::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> CTrueIR::typed_pointer_type<CTrueIR::IFunctionNode>
{
	auto& irPool = ir->getObjectPool();
	auto retval = ir->getObjectPool().emplace<CTrueIR::CBeer>();
	return retval;
}
auto CFrontendIR::CFresnel::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> CTrueIR::typed_pointer_type<CTrueIR::IFunctionNode>
{
	auto& irPool = ir->getObjectPool();
	auto retval = ir->getObjectPool().emplace<CTrueIR::CFresnel>();
	if (auto* const leaf=irPool.deref(retval); leaf)
		leaf->setReciprocateEtas(this->reciprocateEtas);
	return retval;
}
auto CFrontendIR::CThinInfiniteScatterCorrection::createIRNode(const bool forBTDF, const CFrontendIR* ast, CTrueIR* ir) const -> CTrueIR::typed_pointer_type<CTrueIR::IFunctionNode>
{
	auto& irPool = ir->getObjectPool();
	auto retval = ir->getObjectPool().emplace<CTrueIR::CThinInfiniteScatterCorrection>();
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
		etaH = ir->hashNCache(srcEta->createIRNode(ast,ir));
		if (!etaH)
			return {};
	}
	const auto retval = irPool.emplace<CTrueIR::CCookTorrance>();
	if (auto* const ct=irPool.deref(retval); ct)
	{
		ct->ndfParams = ndParams; // the padding abuse is the same between the classes
		ct->orientedRealEta = etaH;
		// we don't flip depending on `forBTDF` because a BTDF can be hit from underside as long as its not the last BTDF
		ct->setEtaReciprocal(this->isEtaReciprocal());
	}
	return retval;
}

template class CTrueIR::CSpectralVariable<CFrontendIR::ISpectralVariableExpr>;
}