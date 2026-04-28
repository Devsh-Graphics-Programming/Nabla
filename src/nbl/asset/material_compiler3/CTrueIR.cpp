// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/material_compiler3/CTrueIR.h"

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"


namespace nbl::asset::material_compiler3
{

constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
constexpr auto ELL_DEBUG = nbl::system::ILogger::E_LOG_LEVEL::ELL_DEBUG;
using namespace nbl::system;

// TODO: Move these notes, they're for the backend.
// 
// Any material (unless its emission only) requires a shading normal to be computed and kept, this should be the first register.
// Materials are primitive agnostic, don't assume triangles, so any attribute interpolation happens in Material System attribute getting callbacks.
// In practice most renderers will apply them on triangles, the advice is to keep triangle indices in own live state raw or compressed (uint32_t base, and then uint16_t diffs).
// To use the material outside of Raytracing or Visibility Buffer one could use HW interpolators, so all interpolated attributes will sit in global input variables.
// Whenever Material Compiler calls `getUV` or `getTBN` its the user's choice how to implement that.
// 
// However whenever HW interpolators are not used, to compute a UV it costs:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent UV attribute fetch
// - 2 FMA per channel to interpolate
// And to compute a TBN its:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent fetch of Tangent (and optional Bitangent), or Rotation Around normal + scales
// - 2 FMA per channel to interpolate
// - 1 reconstruction of a TBN matrix (can be expensive, i.e. from a quaternion or include renormalizations of interpolated tangents)
// For this reason, it makes sense to spill interpolated UVs and Tangent+BitangentScale instead of recomputing.
// The material compiler will keep a bitmask of UVs and TBNs (TBN depends on UV set) already requested and will spill if user marks sets as expensive to recompute.
// 
// If everyone is using the same TBN set, we could back up the interaction and sample because these are just V and L transformed into the TBN.
// In theory this requires us to use anisotropic interactions whenever the isotropic BRDF has a normal perturbation which seems like an overhead.
// However specular AA mitigations which derive roughness from per-pixel surface curvature and path regularizations would force this upon us anyway.
// Problem is that importance sampling requires us to transform a sampled tangent space L into worldspace and that requires knowing the TBN.
// I can see some optimizations for TBN calc where we only recompute the tangent space .xy of L and V for each TBN since N stays constant and NdotX2 can be known.
// Since we must have full TBN for anisotropic BxDF, tangent space V can be computed quicker than fetching it. 
// 
// Pertubed normal can be spilled or refetched. If it gets refetched some coefficients need to be recomputed which allow for taking dot products of V and L with it.
// If the BRDF is anisotropic, a small nested TBN needs construction so perturbed normal is 3 coefficients.
// Spilling as a float16_t3 or float16_t4 is identical memory traffic to a bilinear derivative map tap assuming UNORM8 storage.
// Tangents consistent with unperturbed TBN can be worked out as `B_p = normalize(cross(N_p,T))` and `T_p = cross(B_p,N_p)`, unless `N_p==T` then cross N_p with B instead.
// If tangents want to be rotated/explicitly controlled, then the spilled normal plus a rotation are still 3 coefficients if encoded as a quaternion (painful to decode) or 4 coeffs.
// 
// Already computed shading normals and other BRDF parameters will be computed on demand but also cached (statically or dynamically) in registers.
// Register leftovers can be passed between command streams to not re-fetch or recompute BxDF parameters.
// Register allocation should spill whole quantities (all of RGB of a color or XYZ of a normal) not singular channels.
// Registers spilled should be the ones with least upcoming uses (should assume streams will run as AOV emission, AOV throughputs, Emission, NEE Eval, Generate, Quotient order)
// Relative cost to recompute or refetch should be included in the decision whether to keep around when register allocating.
// Perturbed Normals should be spilled to VRAM instead of recomputed as they require fetching at least a quaternion and a texture sample.
// Spill should be a big circular buffer allocated per-subgroup - hard to control in Pipeline Shaders https://github.com/KhronosGroup/Vulkan-Docs/issues/2717
// So need to allocate worst case spill for all rays in a dispatch (although can do persistent threads and reduce the dispatch size a little)


CTrueIR::CTrueIR(creation_params_type&& params) : CNodePool(std::move(params)), m_basicNodes(this)
{
	reset();
}

CTrueIR::SBasicNodes::SBasicNodes(CTrueIR* ir)
{
	auto& pool = ir->getObjectPool();
	//
	blackHoleBxDF = pool.emplace<CContributorSum>();
	{
		auto* const node = pool.deref(blackHoleBxDF._const_cast());
		node->product = {};
		const bool success = node->recomputeHash(pool);
		assert(success);
		ir->m_uniqueNodes[node->getHash()] = blackHoleBxDF;
	}
	//
	errorBxDF = pool.emplace<CContributorSum>();
	{
		auto* const node = pool.deref(errorBxDF._const_cast());
		node->product = {}; // TODO: magenta diffuse and rest of setup
		//const bool success = node->recomputeHash(pool);
		//assert(success);
	}
}


auto CTrueIR::SAddSession::makeOrientedMaterial(const CFrontendIR::typed_pointer_type<const CFrontendIR::CLayer> rootH) -> SMaterial::SOriented
{
	SMaterial::SOriented retval = {};
	assert(layerStack.empty());

	auto& astPool = args.forest->getObjectPool();
	auto& irPool = tmpIR->getObjectPool();
	// go down through layers and create all the dependencies
	for (const auto* layer=astPool.deref(rootH); layer; layer=astPool.deref(layer->coated))
	{
		// TODO: actually re-check the expressions for being null after optimization
		bool noTopReflection = !layer->brdfTop;
		bool noTransmission = !layer->btdf;
		// if there's literally nothing on the top level, you can't get to the next layer to retroreflect from it
		if (noTopReflection && noTransmission)
		{
			args.logger.log("Skipping current layer and farther ones due to no transmission and reflection",ELL_DEBUG);
			break;
		}
		layerStack.push_back(layer);
		// find out rest of the layers don't matter because they're blocked from being seen, its not a complete check
		if (noTransmission)
		{
			args.logger.log("Skipping remaining layers due to no transmission",ELL_DEBUG);
			break;
		}
// TODO: handle the rest of this stuff
//assert(false);
		// Only if we're not in the last layer do we care about the bottom BRDF (you can't hit it otherwise)
		// Note that this won't catch the next layer being a blackhole and needs to be undone if it is
		if (layer->coated && layer->brdfBottom)
		{
			// do stuff with brdfBottom
		}
	}
	if (!layerStack.empty())
		retval.metadata |= SMaterial::EMetadataBits::NotBlackhole;
	// then go back up and make the layers
	while (!layerStack.empty())
	{
		const auto* const inLayer = layerStack.back();
		layerStack.pop_back();
		// allocate a layer
		const auto layerH = irPool.emplace<COrientedLayer>();
		auto* const outLayer = irPool.deref(layerH);
		retval.root = layerH;
		// process the BTDF
		//...
		// process the top BRDF
		outLayer->brdfTop = makeContributors(inLayer->brdfTop);
		// if BTDF has delta transmissions, then via the sampling property hoist next layer into current layer BRDFs with the DeltaTransmission weights applied
		// hmm this would require decorrellation... because don't want rest of BTDF to affect
		//...
	}
	// skip replace delta transmissions by the layer undernearth, if null then keep as delta

	// AST is Sum Expression to the BRDF nodes
	// We need to keep the Ancestor prefix as an unrolled linked list
	return retval;
}

//
auto CTrueIR::SAddSession::makeContributors(const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> bxdfRootH) -> typed_pointer_type<const CContributorSum>
{
	// debug what we're processing
	auto printSubtree = [&](const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> nodeH)->void
	{
		CFrontendIR::SDotPrinter printer(args.forest);
		printer.exprStack.push(nodeH);
		args.logger.log("Subtree Dot3 : \n%s\n", ELL_DEBUG,printer().c_str());
	};
	typed_pointer_type<const CContributorSum> headH = {};
	if (!bxdfRootH)
		return headH;
	printSubtree(bxdfRootH);
	
	// Multiplication Chain need to be sorted in a canonical order so its easier to spot them being the same
	auto sortMuls = [](const SAddSession::SFactor& lhs, const SAddSession::SFactor& rhs)->bool
	{
		// monochrome is cheaper
		if (lhs.monochrome!=rhs.monochrome)
			return lhs.monochrome;
		// not doing a complement is cheaper
		if (lhs.handle.value==rhs.handle.value)
			return lhs.negate<rhs.negate;
		// but want negations to show up together in the sorted list so easier to put back together
		return lhs.handle.value<rhs.handle.value;
	};

	auto& astPool = args.forest->getObjectPool();
	auto& irPool = tmpIR->getObjectPool();
	// scratches are initialized
	assert(mulChain.empty());
	assert(contributorStack.empty());
	exprStack.push_back({.node=astPool.deref(bxdfRootH)});
	typed_pointer_type<CContributorSum> tailH = {};
	while (!exprStack.empty())
	{
		auto& entry = exprStack.back();
		using ast_expr_type_e = CFrontendIR::IExprNode::Type;
		const ast_expr_type_e astExprType = entry.node->getType();
		const bool isContributor = astExprType==CFrontendIR::IExprNode::Type::Contributor;
		//
		if (entry.notVisited())
		{
			if (isContributor)
			{
				// TODO actually make the contributor
				const auto contributorType = 0;
				switch (contributorType)
				{
					case 45:
					{
						// TODO: add to contributorStack
						contributorStack.push_back({.contributor={}});
						break;
					}
					// unsupported contributor
					default:
					{
						args.logger.log("Unsupported contributor type %d skipping subtree",ELL_ERROR,contributorType);
						exprStack.clear();
						mulChain.clear();
						contributorStack.clear();
						return tmpIR->getBasicNodes().errorBxDF;
					}
				}
				// dont want to deal with the contributor again
				exprStack.pop_back();
			}
			else
			{
#if 0 // TODO: Other factors
				// spot prefix being null/zero to stop exploring
				// can make the decision wholly on current factor
				bool continueExploring = true;
				if (continueExploring)
				{
					// make the actual factor
					auto derivedFactorH = getObjectPool().emplace<CConstant>();
					// TODO: find the place where to insert it
					{
						// shtuff
					}
					entry.factor = derivedFactorH;
					// add self after making self
					ancestors.push_back(getObjectPool().deref(entry.factor));
					// TODO: push the children nodes onto the stack
					continue;
				}
#endif
				const bool isAdd = astExprType==ast_expr_type_e::Add;
				if (isAdd)
				{
					entry.addContributor = true;
					// Current Add node will perform the job of the parent add node for this subtree
					if (entry.nonMulImmediateAncestorStackEnd)
						exprStack[entry.nonMulImmediateAncestorStackEnd-1].addContributor = false;
				}
				const bool notMul = astExprType!=ast_expr_type_e::Mul;
				// go through children
				const auto childCount = entry.node->getChildCount();
				// add in reverse so stack processes in order
				for (auto childIx=childCount; childIx; )
				{
					// making sure we visit this node again each time a subtree of an Add node is done
					if (isAdd && childIx!=childCount)
					{
						auto& extraEntry = exprStack.emplace_back(entry);
						extraEntry.visited = true;
						extraEntry.addContributor = true;
					}
					// regular exploration
					exprStack.push_back({
						.node = astPool.deref(entry.node->getChildHandle(--childIx)),
						// to be able to go back to the non mul that is supposed to add our subtree
						.nonMulImmediateAncestorStackEnd = notMul ? static_cast<uint16_t>(exprStack.size()):entry.nonMulImmediateAncestorStackEnd
					});
				}
			}
			entry.visited = true;
		}
		else
		{
			assert(!isContributor);
			// do stuff now
			switch (astExprType)
			{
				case ast_expr_type_e::Add:
				{
					// we visited the leftmost subtrees first so this is the right order
					{
						auto* const tail = irPool.deref(tailH);
						tailH = irPool.emplace<CContributorSum>();
						if (tailH)
							tail->rest = tailH;
						else
							headH = tailH;
					}
					// add current contributor with weight to BxDF Sum
					{
						const auto weightedH = irPool.emplace<CWeightedContributor>();
						irPool.deref(tailH)->product = weightedH;
						auto* weighted = irPool.deref(weightedH);
						weighted->contributor = contributorStack.back().contributor;
						if (!mulChain.empty())
						{
							const CFactorCombiner::SState combinerState = {
								.type = CFactorCombiner::Type::Mul,
								.childCount = mulChain.size()
							};
							// TODO: create the combiner node
							//const auto factorH = getObjectPool().emplace<CFactorCombiner>();
							{
								// every contributor node gets its own SORTED ancestor prefix
								mulChainSortScratch = mulChain;
								std::sort(mulChainSortScratch.begin(),mulChainSortScratch.end(),sortMuls);
								//auto oit = getObjectPool().deref(factorH)->child;
								//for (const auto& mul : mulChainSortScratch)
									//*(oit++) = mul.handle;
							}
							//weighted->factor = factorH;
						}
					}
					// when we are done we need to reset the mul chain back to its original state
					mulChain.resize(entry.mulChainLen);
					break;
				}
				default:
					break;
			}
			exprStack.pop_back();
		}
	}
	// we got all the AST ADD nodes on the way back out
	assert(mulChain.empty());
	assert(contributorStack.empty());
	return headH;
}

//
auto CTrueIR::addMaterials(const SAddMaterialsArgs& args) -> core::vector<SMaterialHandle>
{
	const auto logger = args.logger;
	if (!args)
	{
		logger.log("Invalid Arguments to `CTrueIR::addMaterials`",ELL_ERROR);
		return {};
	}
	const auto& astPool = args.forest->getObjectPool();


//	core::unordered_map<const CFrontendIR::IExprNode*,bool> brdfs;
//	core::unordered_map<const CFrontendIR::IExprNode*,bool> btdfs;

	SAddSession session = {args};
	const auto inputMaterials = args.forest->getMaterials();
	core::vector<SMaterialHandle> retval(inputMaterials.size(),{});
	auto outIt = retval.begin();
	for (const auto& rootH : inputMaterials)
	{
		auto& result = *(outIt++);

		const auto* astRoot = astPool.deref<const CFrontendIR::CLayer>(rootH);
		// no material
		if (!astRoot)
		{
			result = BlackholeMaterialHandle;
			continue;
		}
		session.tmpIR->reset();
		// reverse AST into another tree
		session.tmpAST->reset();
		const auto backRootH = session.tmpAST->reverse(rootH,args.forest);
		// TODO: hash, deduplicate, collect metadata and insert into current IR
		SMaterial material = {
			.front = session.makeOrientedMaterial(rootH),
			.back = session.makeOrientedMaterial(backRootH)
		};

		// TODO: better debug info
		if (const auto* debug=astPool.deref<const CDebugInfo>(astRoot->debugInfo); debug && !debug->data().empty())
		{
			material.debugInfo = getObjectPool().emplace<CNodePool::CDebugInfo>(debug->data().data(),debug->data().size());
		}

		//
		result.value = m_materials.size();
		m_materials.push_back(material);
	}

	return retval;
}


}