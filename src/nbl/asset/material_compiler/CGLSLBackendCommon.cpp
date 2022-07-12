// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nbl/asset/material_compiler/CGLSLBackendCommon.h>


namespace nbl::asset::material_compiler
{


using instr_stream = CGLSLBackendCommon::instr_stream;
using instr_t = instr_stream::instr_t;
using traversal_t = instr_stream::traversal_t;
// TODO: replace with hash consing
using tmp_bxdf_translation_cache_t = core::unordered_map<const IR::node_handle_t,IR::node_handle_t>;
// TODO: more extreme deduplication, use hash consing!
class CIdGenerator
{
	public:
		using id_t = instr_stream::instr_id_t;

		id_t get_id(const IR::node_handle_t _node)
		{
			if (auto found = m_cache.find(_node); found != m_cache.end())
				return found->second;

			id_t id = gen_id();
			m_cache.insert({ _node, id });

			return id;
		}

	private:
		id_t gen_id()
		{
			return gen++;
		}

		id_t gen = 0u;
		core::unordered_map<const IR::node_handle_t,id_t> m_cache;
};


// good idea to make this tree translator "catch" duplicate subtrees
class CInterpreter
{
	public:
		static IR::node_handle_t translateMixIntoBlends(IR* ir, const IR::IBSDFCombinerNode* _mix);

		struct codegen_t
		{
			instr_t instruction;
			IR::node_handle_t branchNode;
		};
		static codegen_t processSubtree(IR* ir, const IR::node_handle_t branch, tmp_bxdf_translation_cache_t* coatTranslationCache);

	protected:
		static inline IR::node_handle_t getCoatNode(IR* ir, tmp_bxdf_translation_cache_t* cache, const IR::node_handle_t combined_coating)
		{
			auto pCombinedCoating = ir->getNode<IR::CMicrofacetCoatingBSDFNode>(combined_coating);
			assert(pCombinedCoating);
			if (auto found = cache->find(combined_coating); found != cache->end())
				return found->second;

			// the coating is a dielectric, but it cannot transmit so make it a conductor
			auto coating = ir->allocTmpNode<IR::CMicrofacetSpecularBSDFNode>(1u);
			auto pCoating = ir->getNode<IR::CMicrofacetSpecularBSDFNode>(coating);
			{
				pCoating->alpha_u = pCombinedCoating->alpha_u;
				pCoating->alpha_v = pCombinedCoating->alpha_v;
				pCoating->eta = pCombinedCoating->eta;
				pCoating->ndf = pCombinedCoating->ndf;
			}
			cache->insert({ combined_coating, coating });

			return coating;
		}
		static inline IR::node_handle_t getDeltaTransmissionNode(IR* ir, tmp_bxdf_translation_cache_t* cache, const IR::node_handle_t opacity)
		{
			auto pOpacity = ir->getNode<IR::COpacityNode>(opacity);
			assert(pOpacity);
			if (auto found = cache->find(opacity); found != cache->end())
				return found->second;

			auto transmission = ir->allocTmpNode<IR::IBSDFNode>(IR::IBSDFNode::ET_DELTA_TRANSMISSION);
			cache->insert({ opacity, transmission });

			return transmission;
		}
};


// base class for the many traversals:
// - texture prefetch
// - normal precompute
// - importance sample generator
// - remainder and pdf computation
template <typename stack_el_t>
class ITraversalGenerator
{
	protected:
		using SContext = CGLSLBackendCommon::SContext;

		SContext* m_ctx;
		IR* m_ir;
		// merge ID geenrator and translation Cache
		CIdGenerator* m_id_gen;
		tmp_bxdf_translation_cache_t* m_translationCache;

		core::stack<stack_el_t> m_stack;

		//IDs in bumpmaps start with 0 (see writeBumpmapBitfields())
		//rem_and_pdf: instructions not preceded with OP_BUMPMAP (resulting from node without any bumpmap above in tree) will have normal ID = ~0
		uint32_t m_firstFreeNormalID = static_cast<uint32_t>(-1);

		// right now it only gets used to inherit normalID from parent instruction
		virtual void writeInheritableBitfields(instr_t& dst, instr_t parent) const
		{
		}

		// Extra operations performed on instruction just before it is pushed on stack
		virtual void onBeforeStackPush(instr_t& instr, const IR::node_handle_t node) const
		{
			instr_stream::instr_id_t id = m_id_gen->get_id(node);
			instr_stream::setInstrId(instr, id);
		}

		void writeBumpmapBitfields(instr_t& dst)
		{
			++m_firstFreeNormalID;
			dst = core::bitfieldInsert<instr_t>(dst, m_firstFreeNormalID, instr_stream::INSTR_NORMAL_ID_SHIFT, instr_stream::INSTR_NORMAL_ID_WIDTH);
		}

		// TODO: why would we even have NOOPs in the instruction stream!?
		void filterNOOPs(traversal_t& _traversal)
		{
			_traversal.erase(
				std::remove_if(_traversal.begin(), _traversal.end(), [](instr_t i) { return instr_stream::getOpcode(i) == instr_stream::OP_NOOP; }),
				_traversal.end()
			);
		}

		auto processSubtree(const IR::node_handle_t branch)
		{
			// TODO deduplication (find identical IR subtrees, make them share instruction streams), hash consing?
			// Merkle Tree, LLVM had some nice blogposts about how their LTO works with hashmaps that can match subtrees in the context of type definitions
			return CInterpreter::processSubtree(m_ir,branch,m_translationCache);
		}

		void setBSDFData(instr_stream::intermediate::SBSDFUnion& _dst, instr_stream::E_OPCODE _op, const IR::INode* _node)
		{
			switch (_op)
			{
			case instr_stream::OP_DIFFUSE:
			{
				auto* node = static_cast<const IR::CMicrofacetDiffuseBSDFNode*>(_node);
				if (!node->alpha_u.isConstant())
					_dst.diffuse.alpha.setTexture(packTexture(node->alpha_u.texture), node->alpha_u.texture.scale);
				else
					_dst.diffuse.alpha.setConst(node->alpha_u.constant);
				if (!node->reflectance.isConstant())
					_dst.diffuse.reflectance.setTexture(packTexture(node->reflectance.texture), node->reflectance.texture.scale);
				else
					_dst.diffuse.reflectance.setConst(node->reflectance.constant.pointer);
			}
			break;
			case instr_stream::OP_DIELECTRIC: [[fallthrough]];
			case instr_stream::OP_THINDIELECTRIC:
			{
				auto* node = static_cast<const IR::CMicrofacetDielectricBSDFNode*>(_node);

				if (!node->alpha_u.isConstant())
					_dst.dielectric.alpha_u.setTexture(packTexture(node->alpha_u.texture), node->alpha_u.texture.scale);
				else
					_dst.dielectric.alpha_u.setConst(node->alpha_u.constant);
				if (!node->alpha_v.isConstant())
					_dst.dielectric.alpha_v.setTexture(packTexture(node->alpha_v.texture), node->alpha_v.texture.scale);
				else
					_dst.dielectric.alpha_v.setConst(node->alpha_v.constant);
				_dst.dielectric.eta = core::rgb32f_to_rgb19e7(node->eta.pointer);
			}
			break;
			case instr_stream::OP_CONDUCTOR:
			{
				auto* node = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(_node);
				
				if (!node->alpha_u.isConstant())
					_dst.conductor.alpha_u.setTexture(packTexture(node->alpha_u.texture), node->alpha_u.texture.scale);
				else
					_dst.conductor.alpha_u.setConst(node->alpha_u.constant);
				if (!node->alpha_v.isConstant())
					_dst.conductor.alpha_v.setTexture(packTexture(node->alpha_v.texture), node->alpha_v.texture.scale);
				else
					_dst.conductor.alpha_v.setConst(node->alpha_v.constant);
				_dst.conductor.eta[0] = core::rgb32f_to_rgb19e7(node->eta.pointer);
				_dst.conductor.eta[1] = core::rgb32f_to_rgb19e7(node->etaK.pointer);
			}
			break;
			case instr_stream::OP_COATING:
			{
				auto* coat = static_cast<const IR::CMicrofacetCoatingBSDFNode*>(_node);

				/*
				if (!coat->alpha_u.isConstant())
					_dst.coating.alpha_u.setTexture(packTexture(coat->alpha_u.texture), coat->alpha_u.texture.scale);
				else
					_dst.coating.alpha_u.setConst(coat->alpha_u.constant);
				if (!coat->alpha_v.isConstant())
					_dst.coating.alpha_v.setTexture(packTexture(coat->alpha_v.texture), coat->alpha_v.texture.scale);
				else
					_dst.coating.alpha_v.setConst(coat->alpha_v.constant);
				*/
				if (!coat->thicknessSigmaA.isConstant())
					_dst.coating.sigmaA.setTexture(packTexture(coat->thicknessSigmaA.texture), coat->thicknessSigmaA.texture.scale);
				else
					_dst.coating.sigmaA.setConst(coat->thicknessSigmaA.constant.pointer);

				_dst.coating.eta = core::rgb32f_to_rgb19e7(coat->eta.pointer);
				//_dst.coating.thickness = coat->thickness;
			}
			break;
			case instr_stream::OP_BLEND:
			{
				auto* b = static_cast<const IR::CBSDFCombinerNode*>(_node);
				assert(b->type == IR::CBSDFCombinerNode::ET_WEIGHT_BLEND);
				auto* blend = static_cast<const IR::CBSDFBlendNode*>(b);

				if (!blend->weight.isConstant())
					_dst.blend.weight.setTexture(packTexture(blend->weight.texture), blend->weight.texture.scale);
				else
					_dst.blend.weight.setConst(blend->weight.constant.pointer);
			}
			break;
			case instr_stream::OP_DIFFTRANS:
			{
				auto* difftrans = static_cast<const IR::CMicrofacetDifftransBSDFNode*>(_node);

				if (!difftrans->alpha_u.isConstant())
					_dst.difftrans.alpha.setTexture(packTexture(difftrans->alpha_u.texture), difftrans->alpha_u.texture.scale);
				else
					_dst.difftrans.alpha.setConst(difftrans->alpha_u.constant);
				if (!difftrans->transmittance.isConstant())
					_dst.difftrans.transmittance.setTexture(packTexture(difftrans->transmittance.texture), difftrans->transmittance.texture.scale);
				else
					_dst.difftrans.transmittance.setConst(difftrans->transmittance.constant.pointer);
			}
			break;
			case instr_stream::OP_BUMPMAP:
			{
				const IR::CGeomModifierNode* bm = static_cast<const IR::CGeomModifierNode*>(_node);

				assert(bm->type == IR::CGeomModifierNode::ET_DERIVATIVE);

				_dst.bumpmap.derivmap.vtid = bm ? packTexture(bm->texture) : instr_stream::VTID::invalid();
				core::uintBitsToFloat(_dst.bumpmap.derivmap.scale) = bm ? bm->texture.scale : 0.f;
			}
			break;
			}
		}

		uint32_t getBSDFDataIndex(const instr_stream::E_OPCODE _op, const IR::node_handle_t _node)
		{
			switch (_op)
			{
				case instr_stream::OP_INVALID: [[fallthrough]];
				case instr_stream::OP_NOOP:
					return 0u;
				default:
					break;
			}

			// TODO: better deduplication
			auto found = m_ctx->bsdfDataIndexMap.find(_node);
			if (found != m_ctx->bsdfDataIndexMap.end())
				return found->second;

			instr_stream::intermediate::SBSDFUnion data;
			setBSDFData(data, _op, _node);
			size_t ix = m_ctx->bsdfData.size();
			m_ctx->bsdfDataIndexMap.insert({_node,ix});
			m_ctx->bsdfData.push_back(data);

			return ix;
		}

		instr_stream::VTID packTexture(const IR::INode::STextureSource& tex)
		{
			// cache, obviously
			if (auto found = m_ctx->VTallocMap.find({ tex.image.get(),tex.sampler.get() }); found != m_ctx->VTallocMap.end())
				return found->second;

			auto img = tex.image->getCreationParameters().image;
			img = m_ctx->vt.vt->createUpscaledImage(img.get());
			auto* sampler = tex.sampler.get();

			const auto& extent = img->getCreationParameters().extent;
			const auto uwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(sampler->getParams().TextureWrapU);
			const auto vwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(sampler->getParams().TextureWrapV);
			const auto border = static_cast<asset::ISampler::E_TEXTURE_BORDER_COLOR>(sampler->getParams().BorderColor);

			asset::IImage::SSubresourceRange subres;
			subres.baseArrayLayer = 0u;
			subres.layerCount = 1u;
			subres.baseMipLevel = 0u;
			const uint32_t mx = std::max(extent.width, extent.height);
			const uint32_t round = core::roundUpToPoT<uint32_t>(mx);
			const int32_t lsb = core::findLSB(round);
			subres.levelCount = static_cast<uint32_t>(lsb + 1);

			SContext::VT::alloc_t alloc;
			alloc.format = img->getCreationParameters().format;
			alloc.extent = img->getCreationParameters().extent;
			alloc.subresource = subres;
			alloc.uwrap = uwrap;
			alloc.vwrap = vwrap;
			auto addr = m_ctx->vt.alloc(alloc, std::move(img), border);

			std::pair<SContext::VTallocKey, instr_stream::VTID> item{{tex.image.get(),tex.sampler.get()}, addr};
			m_ctx->VTallocMap.insert(item);

			return addr;
		}

		// returns if the instruction actually got pushed
		template <typename ...Params>
		bool push(const CInterpreter::codegen_t& code, const instr_t _parent, Params&& ...args)
		{
			//  a copy of the instruction gets pushed (some flags get changed)
			instr_t instr = code.instruction;
			// right now this only deals with the normal ID
			writeInheritableBitfields(instr, _parent);
			switch (instr_stream::getOpcode(instr))
			{
				case instr_stream::OP_BUMPMAP:
					writeBumpmapBitfields(instr);
					[[fallthrough]];
				case instr_stream::OP_DIFFUSE: [[fallthrough]];
				case instr_stream::OP_DIFFTRANS: [[fallthrough]];
				case instr_stream::OP_DIELECTRIC: [[fallthrough]];
				case instr_stream::OP_THINDIELECTRIC: [[fallthrough]];
				case instr_stream::OP_CONDUCTOR: [[fallthrough]];
				case instr_stream::OP_COATING: [[fallthrough]];
				case instr_stream::OP_BLEND: [[fallthrough]];
				case instr_stream::OP_NOOP: [[fallthrough]];
				case instr_stream::OP_DELTRATRANS:
					onBeforeStackPush(instr,code.branchNode);
					m_stack.push(stack_el_t(instr,code.branchNode,std::forward<Params>(args)...));
					return true;
					break;
			}
			return false;
		}
		
		// set pointer to BSDF parameter data once its known
		instr_t finalizeInstr(instr_t _instr, const IR::node_handle_t _node, uint32_t _bsdfBufOffset)
		{
			constexpr instr_stream::E_NDF ndfMap[4]
			{
				instr_stream::NDF_BECKMANN,
				instr_stream::NDF_GGX,
				instr_stream::NDF_PHONG,
				instr_stream::NDF_PHONG
			};

			auto handleSpecularBitfields = [&ndfMap](instr_t dst, const IR::CMicrofacetSpecularBSDFNode* node) -> instr_t
			{
				dst = core::bitfieldInsert<instr_t>(dst, ndfMap[node->ndf], instr_stream::BITFIELDS_SHIFT_NDF, instr_stream::BITFIELDS_WIDTH_NDF);
				if (!node->alpha_v.isConstant())
					dst = core::bitfieldInsert<instr_t>(dst, 1u, instr_stream::BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
				if (!node->alpha_u.isConstant())
					dst = core::bitfieldInsert<instr_t>(dst, 1u, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);

				return dst;
			};

			const instr_stream::E_OPCODE op = instr_stream::getOpcode(_instr);
			switch (op)
			{
				case instr_stream::OP_DIFFUSE:
				{
					auto* node = ir->getNode<const IR::CMicrofacetDiffuseBSDFNode*>(_node);
					if (!node->alpha_u.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
					if (!node->reflectance.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_REFL_TEX, 1);
				}
				break;
				case instr_stream::OP_DIELECTRIC: [[fallthrough]];
				case instr_stream::OP_THINDIELECTRIC: [[fallthrough]];
				case instr_stream::OP_CONDUCTOR:
				{
					auto* node = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(_node);
					_instr = handleSpecularBitfields(_instr, node);
				}
				break;
				case instr_stream::OP_COATING:
				{
					auto* coat = static_cast<const IR::CMicrofacetCoatingBSDFNode*>(_node);

					//_instr = handleSpecularBitfields(_instr, coat);
					if (!coat->thicknessSigmaA.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_SIGMA_A_TEX, 1);
				}
				break;
				case instr_stream::OP_BLEND:
				{
					auto* blend = static_cast<const IR::CBSDFCombinerNode*>(_node);
					assert(blend->type == IR::CBSDFCombinerNode::ET_WEIGHT_BLEND);

					if (!static_cast<const IR::CBSDFBlendNode*>(_node)->weight.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_WEIGHT_TEX, 1);
				}
				case instr_stream::OP_DIFFTRANS:
				{
					auto* difftrans = static_cast<const IR::CMicrofacetDifftransBSDFNode*>(_node);

					if (!difftrans->alpha_u.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
					if (!difftrans->transmittance.isConstant())
						_instr = core::bitfieldInsert<instr_t>(_instr, 1u, instr_stream::BITFIELDS_SHIFT_TRANS_TEX, 1);
				}
				break;
			}

			_instr = core::bitfieldInsert<instr_t>(_instr, _bsdfBufOffset, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_SHIFT, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_WIDTH);

			return _instr;
		}

		ITraversalGenerator(SContext* _ctx, IR* _ir, CIdGenerator* _id_gen, tmp_bxdf_translation_cache_t* _cache) : 
			m_ctx(_ctx), m_ir(_ir), m_id_gen(_id_gen), m_translationCache(_cache) {}
};


namespace remainder_and_pdf
{

class CTraversalManipulator
{
	public:
		using substream_base_t = std::pair<size_t, size_t>;
		struct substream_t : private substream_base_t
		{
			substream_t(traversal_t::iterator _beg, traversal_t::iterator _end, traversal_t* _cont) : 
				substream_base_t(_beg-_cont->begin(), _end-_cont->begin()), my_cont(_cont)
			{}

			inline traversal_t::iterator begin() const { return my_cont->begin() + first; }
			inline traversal_t::iterator end()	 const { return my_cont->begin() + second; }

			inline size_t length() const { return second - first; }

		private:
			traversal_t* my_cont;
		};

		using id2pos_map_t = core::unordered_map<instr_stream::instr_id_t, uint32_t>;

	private:
		_NBL_STATIC_INLINE_CONSTEXPR instr_t SPECIAL_VAL = ~static_cast<instr_t>(0);

		static void setRegisters(instr_t& i, uint32_t rdst, uint32_t rsrc1 = 0u, uint32_t rsrc2 = 0u)
		{
			uint32_t& regDword = reinterpret_cast<uint32_t*>(&i)[1];

			constexpr uint32_t _2ND_DWORD_SHIFT = 32u;

			regDword = core::bitfieldInsert(regDword, rdst, instr_stream::remainder_and_pdf::INSTR_REG_DST_SHIFT-_2ND_DWORD_SHIFT, instr_stream::remainder_and_pdf::INSTR_REG_WIDTH);
			regDword = core::bitfieldInsert(regDword, rsrc1, instr_stream::remainder_and_pdf::INSTR_REG_SRC1_SHIFT-_2ND_DWORD_SHIFT, instr_stream::remainder_and_pdf::INSTR_REG_WIDTH);
			regDword = core::bitfieldInsert(regDword, rsrc2, instr_stream::remainder_and_pdf::INSTR_REG_SRC2_SHIFT-_2ND_DWORD_SHIFT, instr_stream::remainder_and_pdf::INSTR_REG_WIDTH);
		}

		struct has_different_normal_id
		{
			uint32_t n_id;

			bool operator()(const instr_t& i) const { return instr_stream::getNormalId(i) != n_id; };
		};

		traversal_t m_input;
		core::queue<uint32_t> m_streamLengths;
		const uint32_t m_regsPerResult;

		void reorderBumpMapStreams_impl(traversal_t& _input, traversal_t& _output, const substream_t& _stream);

	public:
		CTraversalManipulator(traversal_t&& _traversal, uint32_t _regsPerResult) : m_input(std::move(_traversal)), m_regsPerResult(_regsPerResult) {}

		traversal_t&& process(uint32_t regBudget, uint32_t& _out_usedRegs, id2pos_map_t& _out_id2pos)
		{
			reorderBumpMapStreams();
			_out_id2pos = specifyRegisters(regBudget, _out_usedRegs);
			// specify registers gives us the maximum used register offset, but we need "one past the end" to know the total consumption
			_out_usedRegs += m_regsPerResult;

			return std::move(m_input);
		}

	private:
		// reorders scattered bump-map streams (traversals of BSDF subtrees below bumpmap BSDF node) into continuous streams
		// and moves OP_BUMPMAPs to the beginning of their streams/traversals/subtrees (because obviously they must be executed before BSDFs using them)
		// leaves SPECIAL_VAL to mark original places of beginning of a stream (needed for function specifying registers)
		// WARNING: modifies m_input
		void reorderBumpMapStreams()
		{
			traversal_t result;
			reorderBumpMapStreams_impl(m_input, result, substream_t{ m_input.begin(), m_input.end(), &m_input });

			m_input = std::move(result);
		}

		id2pos_map_t specifyRegisters(uint32_t regCount, uint32_t& _out_maxUsedReg);
};


struct stack_el
{
	instr_t instr;
	IR::node_handle_t node;
	bool visited;
};
class CTraversalGenerator : public ITraversalGenerator<stack_el>
{
		using base_t = ITraversalGenerator<stack_el>;

		const uint32_t m_registerBudget;
		const uint32_t m_regsPerRes;
		CTraversalManipulator::id2pos_map_t m_id2pos;

	public:
		CTraversalGenerator(SContext* _ctx, IR* _ir, CIdGenerator* _id_gen, tmp_bxdf_translation_cache_t* _cache, uint32_t _regCount, uint32_t _regsPerResult) :
			base_t(_ctx, _ir, _id_gen, _cache), m_registerBudget(_regCount), m_regsPerRes(_regsPerResult) {}

		const auto& getId2PosMapping() const { return m_id2pos; }

		void writeInheritableBitfields(instr_t& dst, instr_t parent) const override
		{
			base_t::writeInheritableBitfields(dst, parent);
			
			dst = core::bitfieldInsert(dst, parent>>instr_stream::INSTR_NORMAL_ID_SHIFT, instr_stream::INSTR_NORMAL_ID_SHIFT, instr_stream::INSTR_NORMAL_ID_WIDTH);
		}

		traversal_t genTraversal(const IR::node_handle_t _root, uint32_t& _out_usedRegs);
};
}

namespace gen_choice
{

struct stack_el
{
	instr_t instr;
	IR::node_handle_t node;
	uint32_t parentIx;
};
class CTraversalGenerator : public ITraversalGenerator<stack_el>
{
		using base_t = ITraversalGenerator<stack_el>;

		uint32_t m_firstFreeNormalID = 0u;

	public:
		CTraversalGenerator(SContext* _ctx, IR* _ir, CIdGenerator* _id_gen, tmp_bxdf_translation_cache_t* _cache) : base_t(_ctx, _ir, _id_gen, _cache) {}

		traversal_t genTraversal(const IR::node_handle_t _root);
};

}
namespace tex_prefetch
{
	static instr_stream::tex_prefetch::prefetch_stream_t genTraversal(const traversal_t& _t, const core::vector<instr_stream::intermediate::SBSDFUnion>& _bsdfData, core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _tex2reg, uint32_t _firstFreeReg, uint32_t& _out_usedRegs, uint32_t& _out_regCntFlags);
}

// TODO: hoist to become a generalpurpose OP on the IR
IR::node_handle_t CInterpreter::translateMixIntoBlends(IR* ir, const IR::IBSDFCombinerNode* _mix)
{
	assert(_mix->getType() == IR::IBSDFCombinerNode::ET_MIX);
	auto* mix = static_cast<const IR::CBSDFMixNode*>(_mix);

	// TODO: translate it to use a stack
	struct q_el
	{
		const IR::INode* node;
		IR::INode::color_t weightsSum;
	};
	core::queue<q_el> q;
	for (uint32_t i = 0u; i < mix->children.count; ++i)
		q.push({ mix->children[i], IR::INode::color_t(mix->weights[i]) });

	while (q.size()>1ull)
	{
		auto* blend = ir->allocTmpNode<IR::CBSDFBlendNode>();

		auto left = q.front();
		q.pop();
		auto right = q.front();
		q.pop();

		q_el el{blend, left.weightsSum+right.weightsSum};
		blend->weight = right.weightsSum/el.weightsSum;
		blend->children = IR::INode::createChildrenArray(left.node,right.node);

		q.push(el);
	}

	return q.front().node;
}

CInterpreter::codegen_t CInterpreter::processSubtree(IR* ir, IR::node_handle_t branch, tmp_bxdf_translation_cache_t* cache)
{
	instr_t instr = instr_stream::OP_INVALID;
	switch (ir->getNode(branch)->getSymbol())
	{
		case IR::INode::ES_GEOM_MODIFIER:
		{
			auto* node = ir->getNode<const IR::CGeomModifierNode>(branch);

			if (node->type == IR::CGeomModifierNode::ET_DERIVATIVE)
				instr = instr_stream::OP_BUMPMAP;
			else
				instr = instr_stream::OP_INVALID;
			break;
		}
		case IR::INode::ES_BSDF_COMBINER:
		{
			auto* node = ir->getNode<const IR::IBSDFCombinerNode>(branch);
			switch (node->getType())
			{
				case IR::IBSDFCombinerNode::ET_WEIGHT_BLEND:
					instr = instr_stream::OP_BLEND;
					break;
				case IR::IBSDFCombinerNode::ET_MIX:
				{
					branch = translateMixIntoBlends(ir,node);
					instr = instr_stream::OP_BLEND;
					break;
				}
			}
			break;
		}
		case IR::INode::ES_OPACITY:
		{
			// original node
			auto* opacity = ir->getNode<const IR::COpacityNode>(branch);
			// pure transmission
			auto deltatrans = getDeltaTransmissionNode(ir,cache,branch);
			// blend between them
			branch = ir->allocTmpNode<IR::CBSDFBlendNode>(2u);
			auto* blend = ir->getNode<IR::CBSDFBlendNode>(branch);
			blend->weight = opacity->opacity;
			blend->getChildrenArray()[0] = deltatrans;
			blend->getChildrenArray()[1] = opacity->getChildrenArray()[0];
			instr = instr_stream::OP_BLEND;
			break;
		}
		case IR::INode::ES_BSDF:
		{
			auto* node = ir->getNode<const IR::IBSDFNode>(branch);
			switch (node->type)
			{
				case IR::IBSDFNode::ET_MICROFACET_DIFFTRANS:
					instr = instr_stream::OP_DIFFTRANS;
					break;
				case IR::IBSDFNode::ET_MICROFACET_DIFFUSE:
					instr = instr_stream::OP_DIFFUSE;
					break;
				case IR::IBSDFNode::ET_MICROFACET_SPECULAR:
					instr = instr_stream::OP_CONDUCTOR;
					break;
				case IR::IBSDFNode::ET_DELTA_TRANSMISSION:
					instr = instr_stream::OP_DELTRATRANS;
					break;
				case IR::IBSDFNode::ET_MICROFACET_COATING:
				{
					// original node
					auto* combined_coating = ir->getNode<const IR::CMicrofacetCoatingBSDFNode>(branch);
					// substrate
					auto coated = combined_coating->getChildrenArray()[0];
					auto pCoated = ir->getNode<const IR::IBSDFNode>(coated);
					if (pCoated)
					{
						os::Printer::log("Material compiler GLSL: Coating over non-BxDF IR node, FATAL ERROR!",ELL_ERROR);
						instr = instr_stream::OP_INVALID;
						break;
					}
					// we dont support coating over non-diffuse materials
					switch (pCoated->type)
					{
						case IR::IBSDFNode::ET_MICROFACET_DIFFTRANS: [[fallthorough]]
						case IR::IBSDFNode::ET_MICROFACET_DIFFUSE:
						{
							// translated coating layer
							auto coating = getCoatNode(ir, cache, branch);
							// blend between them
							branch = ir->allocTmpNode<IR::CBSDFBlendNode>(2u);
							auto* blend = ir->getNode<IR::CBSDFBlendNode>(branch);
							blend->weight = opacity->opacity;
							blend->getChildrenArray()[0] = coating;
							blend->getChildrenArray()[1] = coated;
							instr = instr_stream::OP_COATING;
							break;
						}
						// so we ignore coating layer and process only the coated material
						default:
							//assert(false);
							os::Printer::log("Material compiler GLSL: Coating over non-diffuse materials is not supported. Ignoring coating layer!", ELL_WARNING);
							return processSubtree(ir,coated,cache);
							break;
					}
					break;
				}
				case IR::IBSDFNode::ET_MICROFACET_DIELECTRIC:
				{
					auto* dielectric = static_cast<const IR::CMicrofacetDielectricBSDFNode*>(node);
					instr = dielectric->thin ? instr_stream::OP_THINDIELECTRIC:instr_stream::OP_DIELECTRIC;
				}
				break;
			}
		}
		break;
	}

	return {instr,branch};
}

void remainder_and_pdf::CTraversalManipulator::reorderBumpMapStreams_impl(traversal_t& _input, traversal_t& _output, const substream_t& _stream)
{
	// TODO: comments for this function!
	const uint32_t n_id = instr_stream::getNormalId(*(_stream.end() - 1));

	const size_t len = _stream.length();
	size_t subsLenAcc = 0ull;

	core::stack<substream_t> substreams;
	auto subBegin = std::find_if(_stream.begin(), _stream.end(), has_different_normal_id{ n_id });
	while (subBegin != _stream.end())
	{
		const uint32_t sub_n_id = instr_stream::getNormalId(*subBegin);
		decltype(subBegin) subEnd;
		for (subEnd = _stream.end() - 1; subEnd != subBegin; --subEnd)
			if (instr_stream::getNormalId(*subEnd) == sub_n_id)
				break;
		++subEnd;

		auto sub = substream_t{ subBegin, subEnd, &_input };
		//one place will be used for SPECIAL_VAL (hence -1)
		subsLenAcc += sub.length() - 1ull;
		substreams.push(sub);

		subBegin = std::find_if(subEnd, _stream.end(), has_different_normal_id{ n_id });
	}

	while (!substreams.empty())
	{
		reorderBumpMapStreams_impl(_input, _output, substreams.top());
		substreams.pop();
	}

	const uint32_t newlen = len - subsLenAcc;

	substream_t woSubs{ _stream.begin(),_stream.begin() + newlen, &_input };
	//move bumpmap instruction to the beginning of the stream
	auto lastInstr = *(_stream.begin() + newlen - 1);
	if (instr_stream::getOpcode(lastInstr) == instr_stream::OP_BUMPMAP)
	{
		_input.erase(_stream.begin() + newlen - 1);
		_input.insert(_stream.begin(), lastInstr);
	}

	//important for next stage of processing
	m_streamLengths.push(newlen);

	_output.insert(_output.end(), woSubs.begin(), woSubs.end());
	*woSubs.begin() = SPECIAL_VAL;
	//do not erase SPECIAL_VAL (hence +1)
	_input.erase(woSubs.begin() + 1, woSubs.end());
}

auto remainder_and_pdf::CTraversalManipulator::specifyRegisters(uint32_t regBudget, uint32_t& _out_maxUsedReg) -> id2pos_map_t
{
	core::stack<uint32_t> freeRegs;
	{
		// each instruction produces `m_regsPerResult` dwords which need contiguous storage
		regBudget /= m_regsPerResult;
		// add registers to stack back to front to make sure lowest number are popped first
		for (int32_t i=regBudget-1; i>=0; --i)
			freeRegs.push(m_regsPerResult*i);
	}
	//registers with result of bump-map substream
	core::stack<uint32_t> bmRegs;
	//registers waiting to be used as source
	core::stack<uint32_t> srcRegs;

	id2pos_map_t id2pos;

	// TODO: maybe refactor into nested loops so the fact that we deal with substreams assigned to normal IDs is clearer
	int32_t bmStreamEndCounter = 0;
	auto pushResultRegister = [&bmStreamEndCounter, &bmRegs, &srcRegs](uint32_t _resultReg)
	{
		core::stack<uint32_t>& stack = bmStreamEndCounter == 0 ? bmRegs : srcRegs;
		stack.push(_resultReg);
	};
	for (uint32_t j=0u; j<m_input.size();)
	{
		instr_t& i = m_input[j];

		// we need to use the persistent bump-map register as a source register to our next operation (setting the geometric normal)
		if (i == SPECIAL_VAL)
		{
			srcRegs.push(bmRegs.top());
			bmRegs.pop();
			// SPECIAL_VAL is a not an instruction, filter it out of the generated stream
			m_input.erase(m_input.begin() + j);

			continue; // do not increment j because we erased
		}
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(i);
		const instr_stream::instr_id_t id = instr_stream::getInstrId(i);
		id2pos.insert({ id, j });

		--bmStreamEndCounter;
		// next instruction is bump-map setting, record how long the instruction substream is that follows it
		if (op == instr_stream::OP_BUMPMAP)
		{
			bmStreamEndCounter = m_streamLengths.front() - 1u;
			m_streamLengths.pop();

			// OP_BUMPMAP doesnt care about usual registers, so dont set them (its basically a VM state toggle for a global persistent register)
			++j;
			continue;
		}
		// If bmStreamEndCounter reaches value of -1 and next instruction is not OP_BUMPMAP, then emit some SET_GEOM_NORMAL instr
		// but do not insert if we are at the very beginning - interaction is computed for geometry normal at the beginning of each stream implicitely anyway
		else if (bmStreamEndCounter==-1 && j!=0u)
		{
			//just opcode, no registers nor other bitfields in this instruction
			m_input.insert(m_input.begin() + j, instr_t(instr_stream::OP_SET_GEOM_NORMAL));

			++j;
			continue;
		}

		const uint32_t srcsNum = getNumberOfSrcRegsForOpcode(op);
		assert(srcsNum <= 2u);
		uint32_t srcs[2]{ 0u,0u };
		for (uint32_t k = 0u; k < srcsNum; ++k)
		{
			srcs[k] = srcRegs.top();
			srcRegs.pop();
		}

		_out_maxUsedReg = std::max(srcs[0],srcs[1]);

		// increment instruction interator
		++j;
		const bool notLastInstruction = j!=m_input.size();
		switch (srcsNum)
		{
			case 2u:
				{
					// in reverse order on purpose, because of stack pop ordering
					const uint32_t src2 = srcs[0];
					const uint32_t src1 = srcs[1];
					assert(src1<src2);
					const uint32_t dst = notLastInstruction ? src1:0u;
					pushResultRegister(dst);
					freeRegs.push(src2);
					setRegisters(i, dst, src1, src2);
				}
				break;
			case 1u:
				{
					const uint32_t src = srcs[0];
					const uint32_t dst = notLastInstruction ? src:0u;
					pushResultRegister(dst);
					setRegisters(i, dst, src);
				}
				break;
			case 0u:
				{
					assert(!freeRegs.empty());
					uint32_t dst = 0u;
					if (notLastInstruction)
					{
						dst = freeRegs.top();
						freeRegs.pop();
					}
					pushResultRegister(dst);
					setRegisters(i, dst);
				}
				break;
			default:
				break;
		}
	}

	return id2pos;
}

traversal_t remainder_and_pdf::CTraversalGenerator::genTraversal(const IR::node_handle_t _root, uint32_t& _out_usedRegs)
{
	traversal_t traversal;

	// push the whole IR tree while on-line translating into a binary tree, generating a DFS traversal on the stack
	{
		instr_t nullParent = {};
		push(processSubtree(_root), nullParent, false);
	}
	while (!m_stack.empty())
	{
		auto& top = m_stack.top();
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(top.instr);
		_NBL_DEBUG_BREAK_IF(op == instr_stream::OP_INVALID);
		auto children = m_ir->getNode(top.node)->getChildren();
		if (!children || top.visited)
		{
			// post-order traversal emits stuff after all children have already been visited
			const uint32_t bsdfBufIx = getBSDFDataIndex(instr_stream::getOpcode(top.instr), top.node);
			instr_t instr = finalizeInstr(top.instr, top.node, bsdfBufIx);
			traversal.push_back(instr);
			m_stack.pop();
		}
		else if (!top.visited)
		{
			top.visited = true;

			bool allGotPushed = true;
			// because we use a stack, children need pushing in reverse
			for (auto it=children.end()-1; it!=children.begin()-1; --it)
			{
				// we actually rewrite nodes with K>2 children into an unbalanced chain of 2-children nodes
				// TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/287#issuecomment-994732437
				allGotPushed = push(processSubtree(*it), top.instr, false) && allGotPushed;
			}
			_NBL_DEBUG_BREAK_IF(allGotPushed);
		}
	}

	//remove NOOPs
	filterNOOPs(traversal);

	traversal = std::move(CTraversalManipulator(std::move(traversal), m_regsPerRes).process(m_registerBudget, _out_usedRegs, m_id2pos));

	return traversal;
}

// generator choice doesn't use any registers because it chooses BxDF tree branches stochastically
traversal_t gen_choice::CTraversalGenerator::genTraversal(const IR::node_handle_t _root)
{
	constexpr uint32_t INVALID_INDEX = 0xdeadbeefu;

	traversal_t traversal;

	// push the whole IR tree while on-line translating into a binary tree, generating a DFS traversal on the stack
	{
		instr_t nullParent = {};
		push(processSubtree(_root), nullParent, INVALID_INDEX);
	}
	while (!m_stack.empty())
	{
		auto top = m_stack.top();
		m_stack.pop();
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(top.instr);

		const uint32_t bsdfBufIx = getBSDFDataIndex(op, top.node);
		const uint32_t currIx = traversal.size();
		traversal.push_back(finalizeInstr(top.instr, top.node, bsdfBufIx));
		_NBL_DEBUG_BREAK_IF(op == instr_stream::OP_INVALID);
		if (top.parentIx != INVALID_INDEX)
		{
			instr_t& parent = traversal[top.parentIx];
			const uint32_t rightJump = 
				currIx - 
				std::count_if(traversal.begin(), traversal.begin()+currIx, [](instr_stream::instr_t i) { return instr_stream::getOpcode(i) == instr_stream::OP_NOOP; });
			parent = core::bitfieldInsert<instr_t>(parent, rightJump, instr_stream::gen_choice::INSTR_RIGHT_JUMP_SHIFT, instr_stream::gen_choice::INSTR_RIGHT_JUMP_WIDTH);
		}
		
		bool allGotPushed = true;
		// because we use a stack, children need pushing in reverse
		auto children = m_ir->getNode(top.node)->getChildren();
		for (auto it=children.end()-1; it!=children.begin()-1; --it)
		{
			// we actually rewrite nodes with K>2 children into an unbalanced chain of 2-children nodes
			// TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/287#issuecomment-994732437
			allGotPushed = push(processSubtree(*it), top.instr, it==(children.begin()+1) ? currIx:INVALID_INDEX) && allGotPushed;
		}
		_NBL_DEBUG_BREAK_IF(allGotPushed);
	}

	//remove NOOPs
	filterNOOPs(traversal);

	return traversal;
}

// we record whats the most registers (channels) a texture that will get consumed as a parameter
// TODO: Investigate compressing 2-channel fetches to half floats and 3-channel to RGB9E5 or RGB19E7
instr_stream::tex_prefetch::prefetch_stream_t tex_prefetch::genTraversal(
	const traversal_t& _t, const core::vector<instr_stream::intermediate::SBSDFUnion>& _bsdfData,
	core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _out_tex2reg,
	uint32_t _firstFreeReg, uint32_t& _out_usedRegs, uint32_t& _out_regCntFlags
)
{
	core::unordered_set<instr_stream::STextureData, instr_stream::STextureData::hash> processed;
	uint32_t regNum = _firstFreeReg;

	instr_stream::tex_prefetch::prefetch_stream_t prefetch_stream;
		
	for (instr_t instr : _t)
	{
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(instr);

		if (op==instr_stream::OP_NOOP || op==instr_stream::OP_INVALID || op==instr_stream::OP_SET_GEOM_NORMAL)
			continue;

		const uint32_t bsdf_ix = core::bitfieldExtract(instr, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_SHIFT, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		const instr_stream::intermediate::SBSDFUnion& bsdf_data = _bsdfData[bsdf_ix];

		const uint32_t param_count = instr_stream::getParamCount(op);
		for (uint32_t param_i = 0u; param_i < param_count; ++param_i)
		{
			const uint32_t param_tex_shift = instr_stream::BITFIELDS_SHIFT_PARAM_TEX[param_i];
			// TODO: fetch normalmap/bumpmap/derivative map as regular textures, then replace the register contents in-place
			if (op != instr_stream::OP_BUMPMAP && core::bitfieldExtract(instr, param_tex_shift, 1) == 0ull)
				continue;

			// we dont fetch the same texel twice, cache helps us detect duplicates
			instr_stream::tex_prefetch::prefetch_instr_t prefetch_instr;
			prefetch_instr.tex_data = bsdf_data.common.param[param_i].tex;
			if (processed.find(prefetch_instr.tex_data) != processed.end())
				continue;
			processed.insert(prefetch_instr.tex_data);

			const uint32_t dst_reg = regNum;
			const uint32_t reg_cnt = instr_stream::getRegisterCountForParameter(op, param_i);
			// TODO: how is reg_cnt==0 handled!? Do redundant opcodes get emitted!?
			prefetch_instr.setRegCnt(reg_cnt);
			prefetch_instr.setDstReg(dst_reg);
			regNum += reg_cnt;

			prefetch_stream.push_back(prefetch_instr);

			_out_tex2reg.insert({ prefetch_instr.tex_data, dst_reg });

			_out_regCntFlags |= (1u << reg_cnt);
		}
	}

	_out_usedRegs = regNum-_firstFreeReg;

	return prefetch_stream;
}

std::string CGLSLBackendCommon::genPreprocDefinitions(const result_t& _res, E_GENERATOR_STREAM_TYPE _generatorChoiceStream)
{
	using namespace std::string_literals;

	std::string defs;
	defs += "\n#define REG_COUNT " + std::to_string(_res.usedRegisterCount);

	for (instr_stream::E_OPCODE op : _res.opcodes)
		defs += "\n#define "s + OPCODE_NAMES[op] + " " + std::to_string(op);
	defs += "\n#define OP_MAX_BRDF " + std::to_string(instr_stream::OP_MAX_BRDF);
	defs += "\n#define OP_MAX_BSDF " + std::to_string(instr_stream::OP_MAX_BSDF);

	for (instr_stream::E_NDF ndf : _res.NDFs)
		defs += "\n#define "s + NDF_NAMES[ndf] + " " + std::to_string(ndf);
	if (_res.NDFs.size()==1ull)
		defs += "\n#define ONLY_ONE_NDF";

	// TODO: dynamically size this?
	defs += "\n#define sizeof_bsdf_data " + std::to_string((sizeof(instr_stream::SBSDFUnion)+instr_stream::sizeof_uvec4-1u)/instr_stream::sizeof_uvec4);
	
	if (_generatorChoiceStream!=EGST_ABSENT)
		defs += "\n#define GEN_CHOICE_STREAM "+std::to_string(_generatorChoiceStream);
	if (!_res.noPrefetchStream)
		defs += "\n#define TEX_PREFETCH_STREAM";
	if (!_res.noNormPrecompStream)
		defs += "\n#define NORM_PRECOMP_STREAM";
	if (_res.allIsotropic)
		defs += "\n#define ALL_ISOTROPIC_BXDFS";
	if (_res.noBSDF)
		defs += "\n#define NO_BSDF";

	//instruction bitfields
	defs += "\n#define INSTR_OPCODE_MASK " + std::to_string(instr_stream::INSTR_OPCODE_MASK);
	defs += "\n#define INSTR_BSDF_BUF_OFFSET_SHIFT " + std::to_string(instr_stream::BITFIELDS_BSDF_BUF_OFFSET_SHIFT);
	defs += "\n#define INSTR_BSDF_BUF_OFFSET_MASK " + std::to_string(instr_stream::BITFIELDS_BSDF_BUF_OFFSET_MASK);
	defs += "\n#define INSTR_NDF_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_NDF);
	defs += "\n#define INSTR_NDF_MASK " + std::to_string(instr_stream::BITFIELDS_MASK_NDF);
	defs += "\n#define INSTR_ALPHA_U_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX);
	defs += "\n#define INSTR_ALPHA_V_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_ALPHA_V_TEX);
	defs += "\n#define INSTR_REFL_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_REFL_TEX);
	defs += "\n#define INSTR_TRANS_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_TRANS_TEX);
	defs += "\n#define INSTR_SIGMA_A_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_SIGMA_A_TEX);
	defs += "\n#define INSTR_WEIGHT_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_WEIGHT_TEX);
	defs += "\n#define INSTR_1ST_PARAM_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_1ST_PARAM_TEX);
	defs += "\n#define INSTR_2ND_PARAM_TEX_SHIFT " + std::to_string(instr_stream::BITFIELDS_SHIFT_2ND_PARAM_TEX);
	defs += "\n#define INSTR_NORMAL_ID_SHIFT " + std::to_string(instr_stream::INSTR_NORMAL_ID_SHIFT);
	defs += "\n#define INSTR_NORMAL_ID_MASK " + std::to_string(instr_stream::INSTR_NORMAL_ID_MASK);
	//remainder_and_pdf
	{
		defs += "\n#define INSTR_REG_MASK " + std::to_string(instr_stream::remainder_and_pdf::INSTR_REG_MASK);
		defs += "\n#define INSTR_REG_DST_SHIFT " + std::to_string(instr_stream::remainder_and_pdf::INSTR_REG_DST_SHIFT);
		defs += "\n#define INSTR_REG_SRC1_SHIFT " + std::to_string(instr_stream::remainder_and_pdf::INSTR_REG_SRC1_SHIFT);
		defs += "\n#define INSTR_REG_SRC2_SHIFT " + std::to_string(instr_stream::remainder_and_pdf::INSTR_REG_SRC2_SHIFT);
	}
	//gen_choice
	{
		defs += "\n#define INSTR_RIGHT_JUMP_SHIFT " + std::to_string(instr_stream::gen_choice::INSTR_RIGHT_JUMP_SHIFT);
		defs += "\n#define INSTR_RIGHT_JUMP_WIDTH " + std::to_string(instr_stream::gen_choice::INSTR_RIGHT_JUMP_WIDTH);

		defs += "\n#define INSTR_OFFSET_INTO_REMANDPDF_STREAM_SHIFT " + std::to_string(instr_stream::gen_choice::INSTR_OFFSET_INTO_REMANDPDF_STREAM_SHIFT);
		defs += "\n#define INSTR_OFFSET_INTO_REMANDPDF_STREAM_WIDTH " + std::to_string(instr_stream::gen_choice::INSTR_OFFSET_INTO_REMANDPDF_STREAM_WIDTH);
	}
	//tex_prefetch
	{
		defs += "\n#define PREFETCH_INSTR_REG_CNT_SHIFT " + std::to_string(instr_stream::tex_prefetch::prefetch_instr_t::DWORD4_REG_CNT_SHIFT);
		defs += "\n#define PREFETCH_INSTR_REG_CNT_WIDTH " + std::to_string(instr_stream::tex_prefetch::prefetch_instr_t::DWORD4_REG_CNT_WIDTH);
		defs += "\n#define PREFETCH_INSTR_DST_REG_SHIFT " + std::to_string(instr_stream::tex_prefetch::prefetch_instr_t::DWORD4_DST_REG_SHIFT);
		defs += "\n#define PREFETCH_INSTR_DST_REG_WIDTH " + std::to_string(instr_stream::tex_prefetch::prefetch_instr_t::DWORD4_DST_REG_WIDTH);

		if (_res.globalPrefetchRegCountFlags & (1u << 1))
			defs += "\n#define PREFETCH_REG_COUNT_1";
		if (_res.globalPrefetchRegCountFlags & (1u << 2))
			defs += "\n#define PREFETCH_REG_COUNT_2";
		if (_res.globalPrefetchRegCountFlags & (1u << 3))
			defs += "\n#define PREFETCH_REG_COUNT_3";
	}

	//parameter numbers
	{
		defs += "\n#define PARAMS_ALPHA_U_IX " + std::to_string(instr_stream::ALPHA_U_TEX_IX);
		defs += "\n#define PARAMS_ALPHA_V_IX " + std::to_string(instr_stream::ALPHA_V_TEX_IX);
		defs += "\n#define PARAMS_REFLECTANCE_IX " + std::to_string(instr_stream::REFLECTANCE_TEX_IX);
		defs += "\n#define PARAMS_TRANSMITTANCE_IX " + std::to_string(instr_stream::TRANSMITTANCE_TEX_IX);
		defs += "\n#define PARAMS_SIGMA_A_IX " + std::to_string(instr_stream::SIGMA_A_TEX_IX);
		defs += "\n#define PARAMS_WEIGHT_IX " + std::to_string(instr_stream::WEIGHT_TEX_IX);
	}

	for (uint32_t i = 0u; i < instr_stream::INSTR_MAX_PARAMETER_COUNT; ++i)
	{
		using namespace std::string_literals;
		if (_res.paramTexPresence[0][1] == 0u)
			defs += "\n#define PARAM"s + static_cast<char>(i+1u+'0') + "_NEVER_TEX";
		else if (_res.paramTexPresence[0][0] == 0u)
			defs += "\n#define PARAM"s + static_cast<char>(i+1u+'0') + "_ALWAYS_TEX";

		if (_res.paramConstants[i].first)
		{
			const auto& c = _res.paramConstants[i].second;
			defs += "\n#define PARAM"s + static_cast<char>(i+1u+'0') + "_ALWAYS_SAME_VALUE";
			defs += "\n#define PARAM"s + static_cast<char>(i+1u+'0') + "_VALUE vec3(" +
				std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + ")";
		}
	}


	defs += "\n";

	return defs;
}


auto CGLSLBackendCommon::compile(SContext* _ctx, IR* _ir, E_GENERATOR_STREAM_TYPE _generatorChoiceStream) -> result_t
{
	result_t res;
	res.noNormPrecompStream = true;
	res.noPrefetchStream = true;
	res.usedRegisterCount = 0u;
	res.globalPrefetchRegCountFlags = 0u;

	for (const auto root : _ir->getRootNodes())
	{
		uint32_t remainingRegisters = instr_stream::MAX_REGISTER_COUNT;

		const size_t interm_bsdf_data_begin_ix = _ctx->bsdfData.size();

		CIdGenerator id_gen;

		remainder_and_pdf::CTraversalManipulator::id2pos_map_t id2pos;
		tmp_bxdf_translation_cache_t translationCache;

		uint32_t usedRegs{};
		traversal_t rem_pdf_stream;
		{
			// TODO: investigate compression of return value registers from full float to shared exponent or half floats
			const uint32_t regsPerRes = [_generatorChoiceStream]() -> auto
			{
				// In case of presence of generator choice stream, remainder_and_pdf stream has 2 roles in raster backend:
				// * eval stream
				// * remainder-and-pdf stream (for use in multiple importance sampling, as an example); in which case instructions need to write their PDF as well
				// In raytracing backend _computeGenChoiceStream is always present
				switch (_generatorChoiceStream)
				{
					case EGST_PRESENT:
						return 4u;
						break;
					// When desiring Albedo and Normal Extraction, one needs to use extra registers for albedo, normal and throughput scale
					case EGST_PRESENT_WITH_AOV_EXTRACTION:
						// TODO: investigate whether using 10-16bit storage (fixed point or half float) makes execution faster, because 
						// albedo could fit in 1.5 DWORDs as 16bit (or 1 DWORDs as 10 bit), normal+throughput scale in 2 DWORDs as half floats or 16 bit snorm
						// and value/pdf is a low dynamic range so half float could be feasible! Giving us a total register count of 5 DWORDs.
						return 11u;
						break;
					default:
						break;
				}
				// only colour contribution
				return 3u; 
			}();

			remainder_and_pdf::CTraversalGenerator gen(_ctx, _ir, &id_gen, &translationCache, remainingRegisters, regsPerRes);
			rem_pdf_stream = gen.genTraversal(root, usedRegs);
			assert(usedRegs <= remainingRegisters);
			remainingRegisters -= usedRegs;
			id2pos = gen.getId2PosMapping();
		}
		traversal_t gen_choice_stream;
		if (_generatorChoiceStream!=EGST_ABSENT)
		{
			gen_choice::CTraversalGenerator gen(_ctx, _ir, &id_gen, &translationCache);
			// generator stream does not consume any registers
			uint32_t dummyUsedRegs;
			gen_choice_stream = gen.genTraversal(root);
			assert(dummyUsedRegs==0u);

			// final instructions in generator choice need to know which instruction in the remainder&pdf stream corresponds to the same BxDF
			for (auto& instr : gen_choice_stream)
			{
				const instr_stream::instr_id_t id = instr_stream::getInstrId(instr);
				uint32_t rnp_pos = static_cast<uint32_t>(-1);
				if (auto found = id2pos.find(id); found != id2pos.end())
					rnp_pos = found->second;
				instr_stream::gen_choice::setOffsetIntoRemAndPdfStream(instr, rnp_pos);
			}
		}

		// Texture Prefetch and Normal Precompute dont allocate their registers first because we count on 
		instr_stream::tex_prefetch::prefetch_stream_t tex_prefetch_stream;
		core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash> tex2reg;
		{
			tex_prefetch_stream = tex_prefetch::genTraversal(rem_pdf_stream, _ctx->bsdfData, tex2reg, instr_stream::MAX_REGISTER_COUNT-remainingRegisters, usedRegs, res.globalPrefetchRegCountFlags);
			assert(usedRegs <= remainingRegisters);
			remainingRegisters -= usedRegs;
		}

		traversal_t normal_precomp_stream;
		// register allocation for bumpmaps is a nice linear affair
		// TODO: investigate performance impact of quantizing normals to 16 or 21bit SNORM
		const uint32_t firstRegForBumpmaps = instr_stream::MAX_REGISTER_COUNT-remainingRegisters;
		{
			normal_precomp_stream.reserve(std::count_if(rem_pdf_stream.begin(), rem_pdf_stream.end(), [](instr_t i) {return instr_stream::getOpcode(i)==instr_stream::OP_BUMPMAP;}));
			assert(firstRegForBumpmaps+3u*normal_precomp_stream.capacity() <= instr_stream::MAX_REGISTER_COUNT);
			for (instr_t instr : rem_pdf_stream)
			{
				if (instr_stream::getOpcode(instr)==instr_stream::OP_BUMPMAP)
				{
					constexpr uint32_t REGS_FOR_NORMAL = 3u;
					//we can be sure that n_id is always in range [0,count of bumpmap instrs)
					const uint32_t n_id = instr_stream::getNormalId(instr);
					instr = core::bitfieldInsert<instr_t>(instr, firstRegForBumpmaps + REGS_FOR_NORMAL*n_id, instr_stream::normal_precomp::BITFIELDS_REG_DST_SHIFT, instr_stream::normal_precomp::BITFIELDS_REG_WIDTH);
					normal_precomp_stream.push_back(instr);
				}
			}
			remainingRegisters = instr_stream::MAX_REGISTER_COUNT - firstRegForBumpmaps - 3u*normal_precomp_stream.size();
		}

		//src1 reg for OP_BUMPMAPs is set to dst reg of corresponding instruction in normal precomp stream
		setSourceRegForBumpmaps(rem_pdf_stream, firstRegForBumpmaps);
		setSourceRegForBumpmaps(gen_choice_stream, firstRegForBumpmaps);

		for (auto it = _ctx->bsdfData.begin()+interm_bsdf_data_begin_ix; it != _ctx->bsdfData.end(); ++it)
		{
			const auto& interm_bsdf_data = *it;

			instr_stream::SBSDFUnion bsdf_data;
			for (uint32_t i = 0u; i < instr_stream::SBSDFUnion::MAX_TEXTURES; ++i)
			{
				auto found = tex2reg.find(interm_bsdf_data.common.param[i].tex);
				if (found != tex2reg.end())
					bsdf_data.common.param[i].setPrefetchReg(found->second);
				else
					bsdf_data.common.param[i].setConst(interm_bsdf_data.common.param[i].getConst());
			}
			bsdf_data.common.extras[0] = interm_bsdf_data.common.extras[0];
			bsdf_data.common.extras[1] = interm_bsdf_data.common.extras[1];

			res.bsdfData.push_back(bsdf_data);
		}

		result_t::instr_streams_t streams;
		{
			streams.offset = res.instructions.size();

			streams.rem_and_pdf_count = rem_pdf_stream.size();
			res.instructions.insert(res.instructions.end(), rem_pdf_stream.begin(), rem_pdf_stream.end());

			streams.gen_choice_count = gen_choice_stream.size();
			res.instructions.insert(res.instructions.end(), gen_choice_stream.begin(), gen_choice_stream.end());

			streams.norm_precomp_count = normal_precomp_stream.size();
			res.instructions.insert(res.instructions.end(), normal_precomp_stream.begin(), normal_precomp_stream.end());

			streams.prefetch_offset = res.prefetch_stream.size();
			streams.tex_prefetch_count = tex_prefetch_stream.size();
			res.prefetch_stream.insert(res.prefetch_stream.end(), tex_prefetch_stream.begin(), tex_prefetch_stream.end());
		}

		res.streams.insert({root,streams});

		res.noNormPrecompStream = res.noNormPrecompStream && (streams.norm_precomp_count==0u);
		res.noPrefetchStream = res.noPrefetchStream && (streams.tex_prefetch_count==0u);
		res.usedRegisterCount = std::max(res.usedRegisterCount, instr_stream::MAX_REGISTER_COUNT-remainingRegisters);
	}

	_ir->deinitTmpNodes();

	auto isAniso = [&res](instr_t _i) -> bool {
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(_i);
		if (!instr_stream::opHasSpecular(op))
			return false;

		const bool au_tex = core::bitfieldExtract(_i, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		const bool av_tex = core::bitfieldExtract(_i, instr_stream::BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
		if (au_tex != av_tex)
			return false;

		const uint32_t ix = instr_stream::getBSDFDataIx(_i);
		
		const auto& au = res.bsdfData[ix].common.param[0];
		const auto& av = res.bsdfData[ix].common.param[1];

		if (au_tex)
			return au.prefetch != av.prefetch;
		else
			return au.constant != av.constant;
	};

	for (auto& p : res.paramConstants)
	{
		const core::vector3df_SIMD nan = core::vector3df_SIMD(core::nan<float>());
		p = std::make_pair(true, nan);
	}

	res.allIsotropic = true;
	res.noBSDF = true;
	for (const auto& e : res.streams)
	{
		const result_t::instr_streams_t& streams = e.second;
		auto rem_and_pdf = streams.get_rem_and_pdf();
		for (uint32_t i = 0u; i < rem_and_pdf.count; ++i) 
		{
			const uint32_t first = rem_and_pdf.first;
			const instr_t instr = res.instructions[first+i];
			const instr_stream::E_OPCODE op = instr_stream::getOpcode(instr);
			const instr_stream::E_NDF ndf = instr_stream::getNDF(instr);

			const uint32_t paramCount = instr_stream::getParamCount(op);
			const uint32_t ix = instr_stream::getBSDFDataIx(instr);
			for (uint32_t i = 0u; i < paramCount; ++i)
			{
				const uint32_t shift = instr_stream::BITFIELDS_SHIFT_PARAM_TEX[i];
				const auto presence = core::bitfieldExtract(instr, shift, 1);
				res.paramTexPresence[i][presence]++;

				if (res.paramTexPresence[i][1] == 0u)
				{
					const auto& data = res.bsdfData[ix];
					const auto& param = data.common.param[i];
					const auto constant = param.getConst();
					if (!core::isnan(res.paramConstants[i].second.x))
						res.paramConstants[i].first = res.paramConstants[i].first && (res.paramConstants[i].second == constant).all();
					res.paramConstants[i].second = constant;
				}
				else
				{
					res.paramConstants[i].first = false;
				}
			}

			res.noBSDF = res.noBSDF && !instr_stream::opIsBSDF(op);

			res.opcodes.insert(op);
			if (instr_stream::opHasSpecular(op))
			{
				res.NDFs.insert(ndf);
				bool aniso = isAniso(instr);
				res.allIsotropic = res.allIsotropic && !aniso;
			}
		}
	}

	res.fragmentShaderSource_declarations =
		genPreprocDefinitions(res, _generatorChoiceStream) +
R"(
#include <nbl/builtin/glsl/material_compiler/common_declarations.glsl>
)";

	return res;
}


namespace remainder_and_pdf
{

// not dwords, full regs for all output
inline uint32_t getNumberOfSrcRegsForOpcode(instr_stream::E_OPCODE _op)
{
	if (_op == instr_stream::OP_BLEND || _op == instr_stream::OP_COATING)
		return 2u;
	else if (_op == instr_stream::OP_BUMPMAP)
		return 1u;
	return 0u;
}

inline core::vector3du32_SIMD getRegisters(const instr_t& i)
{
	return core::vector3du32_SIMD(
		(i>>instr_stream::remainder_and_pdf::INSTR_REG_DST_SHIFT),
		(i>>instr_stream::remainder_and_pdf::INSTR_REG_SRC1_SHIFT),
		(i>>instr_stream::remainder_and_pdf::INSTR_REG_SRC2_SHIFT)
	) & core::vector3du32_SIMD(instr_stream::remainder_and_pdf::INSTR_REG_MASK);
}

}


void CGLSLBackendCommon::debugPrint(std::ostream& _out, const result_t::instr_streams_t& _streams, const result_t& _res, const SContext* _ctx) const
{
	_out << "####### remainder_and_pdf stream\n";
	auto rem_and_pdf = _streams.get_rem_and_pdf();
	for (uint32_t i = 0u; i < rem_and_pdf.count; ++i)
	{
		using namespace remainder_and_pdf;

		const instr_t instr = _res.instructions[rem_and_pdf.first+i];
		const instr_stream::E_OPCODE op = instr_stream::getOpcode(instr);
		debugPrintInstr(_out, instr, _res, _ctx);
		auto regs = getRegisters(instr);
		auto rcnt = getNumberOfSrcRegsForOpcode(op);
		_out << "Registers:\n";
		if (op!=instr_stream::OP_BUMPMAP && op!=instr_stream::OP_SET_GEOM_NORMAL)
			_out << "DST  = " << regs.x << "\n";
		if (rcnt>0u || op==instr_stream::OP_BUMPMAP)
			_out << "SRC1 = " << regs.y << "\n";
		if (rcnt>1u)
			_out << "SRC2 = " << regs.z << "\n";
	}
	_out << "####### gen_choice stream\n";
	auto gen_choice = _streams.get_gen_choice();
	for (uint32_t i = 0u; i < gen_choice.count; ++i)
	{
		const instr_t instr = _res.instructions[gen_choice.first + i];
		debugPrintInstr(_out, instr, _res, _ctx);
		if (instr_stream::getOpcode(instr) == instr_stream::OP_BUMPMAP)
		{
			_out << "SRC1 = " << remainder_and_pdf::getRegisters(instr).y << "\n";
		}
		uint32_t rjump = core::bitfieldExtract(instr, instr_stream::gen_choice::INSTR_RIGHT_JUMP_SHIFT, instr_stream::gen_choice::INSTR_RIGHT_JUMP_WIDTH);
		_out << "Right jump " << rjump << "\n";
		uint32_t rnp_offset = instr_stream::gen_choice::getOffsetIntoRemAndPdfStream(instr);
		_out << "rem_and_pdf offset " << rnp_offset << "\n";
	}
	_out << "####### tex_prefetch stream\n";
	auto tex_prefetch = _streams.get_tex_prefetch();
	for (uint32_t i = 0u; i < tex_prefetch.count; ++i)
	{
		const instr_stream::tex_prefetch::prefetch_instr_t& instr = _res.prefetch_stream[tex_prefetch.first + i];
		const auto& vtid = instr.tex_data.vtid;

		_out << "### instr " << i << "\n";
		const uint32_t reg_cnt = instr.getRegCnt();
		const uint32_t reg = instr.getDstReg();
		uint32_t scale = instr.tex_data.scale;
		_out << "reg = " << reg << "\n";
		_out << "reg_count = " << reg_cnt << "\n";
		_out << "scale = " << core::uintBitsToFloat(scale) << "\n";
		_out << "pgtab coords = [ " << vtid.pgTab_x << ", " << vtid.pgTab_y << ", " << vtid.pgTab_layer << " ]\n";
		_out << "orig extent = { " << vtid.origsize_x << ", " << vtid.origsize_y << " }\n";
	}
	_out << "####### normal_precomp stream\n";
	auto norm_precomp = _streams.get_norm_precomp();
	for (uint32_t i = 0u; i < norm_precomp.count; ++i)
	{
		const instr_t instr = _res.instructions[norm_precomp.first + i];
		const uint32_t reg = core::bitfieldExtract(instr, instr_stream::normal_precomp::BITFIELDS_REG_DST_SHIFT, instr_stream::normal_precomp::BITFIELDS_REG_WIDTH);
		const uint32_t bsdf_ix = core::bitfieldExtract(instr, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_SHIFT, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		const instr_stream::SBSDFUnion& data = _res.bsdfData[bsdf_ix];

		_out << "### instr " << i << "\n";
		_out << reg << " <- perturbNormal( reg " << data.bumpmap.derivmap_prefetch_reg << " )\n";
	}
}

void CGLSLBackendCommon::debugPrintInstr(std::ostream& _out, instr_t instr, const result_t& _res, const SContext* _ctx) const
{
	auto texDataStr = [](const instr_stream::STextureData& td) {
		return "{ " + std::to_string(reinterpret_cast<const uint64_t&>(td.vtid)) + ", " + std::to_string(reinterpret_cast<const float&>(td.scale)) + " }";
	};
	auto paramVal3OrRegStr = [](const instr_stream::STextureOrConstant& tc, bool tex) {
		if (tex)
			return std::to_string(tc.prefetch);
		else {
			auto val = core::rgb19e7_to_rgb32f(tc.constant);
			return "{ " + std::to_string(val.x) + ", " + std::to_string(val.y) + ", " + std::to_string(val.z) + " }";
		}
	};
	auto paramVal1OrRegStr = [](const instr_stream::STextureOrConstant& tc, bool tex) {
		if (tex)
			return std::to_string(tc.prefetch);
		else {
			auto val = core::rgb19e7_to_rgb32f(tc.constant);
			return std::to_string(val.x);
		}
	};

	constexpr const char* ndf_names[4]{
		"BECKMANN",
		"GGX",
		"PHONG",
		"AS"
	};

	const auto op = instr_stream::getOpcode(instr);

	auto ndfAndAnisoAlphaTexPrint = [&_out,&paramVal1OrRegStr,&ndf_names](instr_t instr, const instr_stream::SBSDFUnion& data) {
		const instr_stream::E_NDF ndf = static_cast<instr_stream::E_NDF>(core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_NDF, instr_stream::BITFIELDS_WIDTH_NDF));
		_out << "NDF = " << ndf_names[ndf] << "\n";

		bool au = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		_out << "Alpha_u tex " << au << "\n";
		_out << "Alpha_u val/reg " << paramVal1OrRegStr(data.common.param[0], au) << "\n";

		bool av = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
		_out << "Alpha_v tex " << av << "\n";
		_out << "Alpha_v val/reg " << paramVal1OrRegStr(data.common.param[1], av) << "\n";
	};

	const uint32_t bsdf_ix = core::bitfieldExtract(instr, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_SHIFT, instr_stream::BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
	instr_stream::SBSDFUnion data;
	if (bsdf_ix < _res.bsdfData.size())
		data = _res.bsdfData[bsdf_ix];

	_out << "### " << OPCODE_NAMES[op] << "\n";
	_out << "BSDF data index = " << bsdf_ix << "\n";
	switch (op)
	{
	case instr_stream::OP_DIFFUSE:
	{
		bool alpha = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		_out << "Alpha tex " << alpha << "\n";
		_out << "Alpha val/reg " << paramVal1OrRegStr(data.diffuse.alpha, alpha) << "\n";
		bool refl = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_REFL_TEX, 1);
		_out << "Refl tex " << refl << "\n";
		_out << "Refl val/reg " << paramVal3OrRegStr(data.diffuse.reflectance, refl) << "\n";
	}
	break;
	case instr_stream::OP_DIELECTRIC: [[fallthrough]];
	case instr_stream::OP_THINDIELECTRIC:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);

		const auto eta = core::rgb19e7_to_rgb32f(data.dielectric.eta);

		_out << "Eta:  { " << eta.x << ", " << eta.y << ", " << eta.z << " }\n";
	}
		break;
	case instr_stream::OP_CONDUCTOR:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);

		const auto eta = core::rgb19e7_to_rgb32f(data.conductor.eta[0]);
		const auto etak = core::rgb19e7_to_rgb32f(data.conductor.eta[1]);

		_out << "Eta:  { " << eta.x << ", " << eta.y << ", " << eta.z << " }\n";
		_out << "EtaK: { " << etak.x << ", " << etak.y << ", " << etak.z << " }\n";
	}
	break;
	case instr_stream::OP_COATING:
	{
		//ndfAndAnisoAlphaTexPrint(instr, data);
		bool sigma = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_SIGMA_A_TEX, 1);
		_out << "thickness*SigmaA tex " << sigma << "\n";
		_out << "thickness*SigmaA val/reg " << paramVal3OrRegStr(data.coating.sigmaA, sigma) << "\n";

		const auto eta = core::rgb19e7_to_rgb32f(data.coating.eta);
		_out << "Eta:  { " << eta.x << ", " << eta.y << ", " << eta.z << " }\n";
		//_out << "Thickness: " << data.coating.thickness << "\n";
	}
	break;
	case instr_stream::OP_BLEND:
	{
		bool weight = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_WEIGHT_TEX, 1);
		_out << "Weight tex " << weight << "\n";
		_out << "Weight val/reg " << paramVal1OrRegStr(data.blend.weight, weight) << "\n";
	}
	break;
	case instr_stream::OP_DIFFTRANS:
	{
		bool trans = core::bitfieldExtract(instr, instr_stream::BITFIELDS_SHIFT_TRANS_TEX, 1);
		_out << "Trans tex " << trans << "\n";
		_out << "Trans val/reg " << paramVal3OrRegStr(data.difftrans.transmittance, trans) << "\n";
	}
	break;
	case instr_stream::OP_BUMPMAP:
		//_out << "Bump reg " << data.bumpmap.bumpmap.prefetch_reg << "\n";
	break;
	default: break;
}

}


}