#include <irr/asset/material_compiler/CMaterialCompilerGLSLBackendCommon.h>

#include <iostream>
#include <irr\asset\material_compiler\CMaterialCompilerGLSLBackendCommon.h>

namespace irr
{
namespace asset
{
namespace material_compiler
{

namespace instr_stream
{
	class CInterpreter
	{
	public:
		static const IR::INode* translateMixIntoBlends(IR* ir, const IR::INode* _mix);

		static std::pair<instr_t, const IR::INode*> processSubtree(IR* ir, const IR::INode* tree, bool thin, IR::INode::children_array_t& next);
	};

	template <typename stack_el_t>
	class ITraveralGenerator
	{
	protected:
		using SContext = CMaterialCompilerGLSLBackendCommon::SContext;

		SContext* m_ctx;
		IR* m_ir;

		core::stack<stack_el_t> m_stack;

		//IDs in bumpmaps start with 0 (see writeBumpmapBitfields())
		//rem_and_pdf: instructions not preceded with OP_BUMPMAP (resulting from node without any bumpmap above in tree) will have normal ID = ~0
		uint32_t m_firstFreeNormalID = static_cast<uint32_t>(-1);

		uint32_t m_registerPool;

		/*template <typename ...Params>
		static stack_el_t createStackEl(Params&& ...args)
		{
			return stack_el_t {std::forward<Params>(args)...};
		}*/

		virtual void writeInheritableBitfields(instr_t& dst, instr_t parent) const
		{
			dst = core::bitfieldInsert(dst, parent>>BITFIELDS_SHIFT_TWOSIDED, BITFIELDS_SHIFT_TWOSIDED, 1);
			dst = core::bitfieldInsert(dst, parent>>BITFIELDS_SHIFT_MASKFLAG, BITFIELDS_SHIFT_MASKFLAG, 1);
		}

		void writeBumpmapBitfields(instr_t& dst)
		{
			++m_firstFreeNormalID;
			dst = core::bitfieldInsert<instr_t>(dst, m_firstFreeNormalID, INSTR_NORMAL_ID_SHIFT, INSTR_NORMAL_ID_WIDTH);
		}

		void filterNOOPs(traversal_t& _traversal)
		{
			_traversal.erase(
				std::remove_if(_traversal.begin(), _traversal.end(), [](instr_t i) { return getOpcode(i) == OP_NOOP; }),
				_traversal.end()
			);
		}

		static IR::INode::SParameter<IR::INode::color_t> extractOpacity(const IR::INode* _root)
		{
			assert(_root->symbol==IR::INode::ES_MATERIAL);
			return static_cast<const IR::CMaterialNode*>(_root)->opacity;
		}

		std::pair<instr_t, const IR::INode*> processSubtree(const IR::INode* tree, bool thin, IR::INode::children_array_t& next)
		{
			//TODO deduplication
			return CInterpreter::processSubtree(m_ir, tree, thin, next);
		}

		void setBSDFData(SBSDFUnion& _dst, E_OPCODE _op, const IR::INode* _node, const IR::INode::SParameter<IR::INode::color_t>& _opacity)
		{
			switch (_op)
			{
			case OP_DIFFUSE:
			{
				auto* node = static_cast<const IR::CMicrofacetDiffuseBSDFNode*>(_node);
				if (node->alpha_u.source == IR::INode::EPS_TEXTURE)
					_dst.diffuse.alpha.setTexture(packTexture(node->alpha_u.value.texture), node->alpha_u.value.texture.scale);
				else
					_dst.diffuse.alpha.setConst(node->alpha_u.value.constant);
				if (node->reflectance.source == IR::INode::EPS_TEXTURE)
					_dst.diffuse.reflectance.setTexture(packTexture(node->reflectance.value.texture), node->reflectance.value.texture.scale);
				else
					_dst.diffuse.reflectance.setConst(node->reflectance.value.constant.pointer);
			}
			break;
			case OP_DIELECTRIC: _IRR_FALLTHROUGH;
			case OP_THINDIELECTRIC:
			{
				auto* blend = static_cast<const IR::CBSDFCombinerNode*>(_node);
				auto* refl = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(blend->children[0]);
				auto* trans = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(blend->children[1]);
				if (refl->scatteringMode == IR::CMicrofacetSpecularBSDFNode::ESM_TRANSMIT)
					std::swap(refl, trans);
				assert(refl->ndf == trans->ndf);
				assert(refl->alpha_u == trans->alpha_u);
				assert(refl->alpha_v == trans->alpha_v);

				if (refl->alpha_u.source == IR::INode::EPS_TEXTURE)
					_dst.dielectric.alpha_u.setTexture(packTexture(refl->alpha_u.value.texture), refl->alpha_u.value.texture.scale);
				else
					_dst.dielectric.alpha_u.setConst(refl->alpha_u.value.constant);
				if (refl->alpha_v.source == IR::INode::EPS_TEXTURE)
					_dst.dielectric.alpha_v.setTexture(packTexture(refl->alpha_v.value.texture), refl->alpha_v.value.texture.scale);
				else
					_dst.dielectric.alpha_v.setConst(refl->alpha_v.value.constant);
			}
			break;
			case OP_CONDUCTOR:
			{
				auto* node = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(_node);
				
				if (node->alpha_u.source == IR::INode::EPS_TEXTURE)
					_dst.conductor.alpha_u.setTexture(packTexture(node->alpha_u.value.texture), node->alpha_u.value.texture.scale);
				else
					_dst.conductor.alpha_u.setConst(node->alpha_u.value.constant);
				if (node->alpha_v.source == IR::INode::EPS_TEXTURE)
					_dst.conductor.alpha_v.setTexture(packTexture(node->alpha_v.value.texture), node->alpha_v.value.texture.scale);
				else
					_dst.conductor.alpha_v.setConst(node->alpha_v.value.constant);
				_dst.conductor.eta[0] = core::rgb32f_to_rgb19e7(node->eta.pointer);
				_dst.conductor.eta[1] = core::rgb32f_to_rgb19e7(node->etaK.pointer);
			}
			break;
			case OP_PLASTIC:
			{
				auto* node = static_cast<const IR::CBSDFCombinerNode*>(_node);
				assert(node->type == IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR);

				auto* d = static_cast<const IR::CBSDFNode*>(node->children[0]);
				auto* s = static_cast<const IR::CBSDFNode*>(node->children[1]);
				if (d->type == IR::CBSDFNode::ET_MICROFACET_SPECULAR)
					std::swap(d, s);

				auto* diffuse = static_cast<const IR::CMicrofacetDiffuseBSDFNode*>(d);
				auto* specular = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(s);

				if (diffuse->reflectance.source == IR::INode::EPS_TEXTURE)
					_dst.plastic.reflectance.setTexture(packTexture(diffuse->reflectance.value.texture), diffuse->reflectance.value.texture.scale);
				else
					_dst.plastic.reflectance.setConst(diffuse->reflectance.value.constant.pointer);
				if (specular->alpha_u.source == IR::INode::EPS_TEXTURE)
					_dst.plastic.alpha_u.setTexture(packTexture(specular->alpha_u.value.texture), specular->alpha_u.value.texture.scale);
				else
					_dst.plastic.alpha_u.setConst(specular->alpha_u.value.constant);
				if (specular->alpha_v.source == IR::INode::EPS_TEXTURE)
					_dst.plastic.alpha_v.setTexture(packTexture(specular->alpha_v.value.texture), specular->alpha_v.value.texture.scale);
				else
					_dst.plastic.alpha_v.setConst(specular->alpha_v.value.constant);

				_dst.plastic.eta = specular->eta.x;
			}
			break;
			case OP_COATING:
			{
				auto* coat = static_cast<const IR::CCoatingBSDFNode*>(_node);

				if (coat->alpha_u.source == IR::INode::EPS_TEXTURE)
					_dst.coating.alpha_u.setTexture(packTexture(coat->alpha_u.value.texture), coat->alpha_u.value.texture.scale);
				else
					_dst.coating.alpha_u.setConst(coat->alpha_u.value.constant);
				if (coat->alpha_v.source == IR::INode::EPS_TEXTURE)
					_dst.coating.alpha_v.setTexture(packTexture(coat->alpha_v.value.texture), coat->alpha_v.value.texture.scale);
				else
					_dst.coating.alpha_v.setConst(coat->alpha_v.value.constant);
				if (coat->sigmaA.source == IR::INode::EPS_TEXTURE)
					_dst.coating.sigmaA.setTexture(packTexture(coat->alpha_u.value.texture), coat->sigmaA.value.texture.scale);
				else
					_dst.coating.sigmaA.setConst(coat->sigmaA.value.constant.pointer);

				uint32_t thickness_eta;
				thickness_eta = core::Float16Compressor::compress(coat->thickness);
				thickness_eta |= static_cast<uint32_t>(core::Float16Compressor::compress(coat->eta.x))<<16;
				_dst.coating.thickness_eta = thickness_eta;
			}
			break;
			case OP_BLEND:
			{
				auto* b = static_cast<const IR::CBSDFCombinerNode*>(_node);
				assert(b->type == IR::CBSDFCombinerNode::ET_WEIGHT_BLEND);
				auto* blend = static_cast<const IR::CBSDFBlendNode*>(b);

				if (blend->weight.source == IR::INode::EPS_TEXTURE)
					_dst.blend.weight.setTexture(packTexture(blend->weight.value.texture), blend->weight.value.texture.scale);
				else
					_dst.blend.weight.setConst(blend->weight.value.constant);
			}
			break;
			case OP_DIFFTRANS:
			{
				auto* difftrans = static_cast<const IR::CDifftransBSDFNode*>(_node);

				if (difftrans->transmittance.source == IR::INode::EPS_TEXTURE)
					_dst.difftrans.transmittance.setTexture(packTexture(difftrans->transmittance.value.texture), difftrans->transmittance.value.texture.scale);
				else
					_dst.difftrans.transmittance.setConst(difftrans->transmittance.value.constant.pointer);
			}
			break;
			case OP_BUMPMAP:
			{
				const IR::CGeomModifierNode* bm = nullptr;
				if (_node->symbol == IR::INode::ES_MATERIAL)
				{
					size_t ix = 0u;
					if (_node->children.find(IR::INode::ES_GEOM_MODIFIER, &ix))
						bm = static_cast<const IR::CGeomModifierNode*>(_node->children[ix]);
				}
				else if (_node->symbol == IR::INode::ES_GEOM_MODIFIER)
					bm = static_cast<const IR::CGeomModifierNode*>(_node);

				assert(bm->type == IR::CGeomModifierNode::ET_DERIVATIVE);

				_dst.bumpmap.derivmap.vtid = bm ? packTexture(bm->texture) : VTID::invalid();
				core::uintBitsToFloat(_dst.bumpmap.derivmap.scale) = bm ? bm->texture.scale : 0.f;
			}
			break;
			}

			if (_op!=OP_BUMPMAP && _op!=OP_BLEND)
			{
				if (_opacity.source == IR::INode::EPS_TEXTURE)
					_dst.param[2].setTexture(packTexture(_opacity.value.texture), _opacity.value.texture.scale);
				else
					_dst.param[2].setConst(_opacity.value.constant.pointer);
			}
		}

		size_t getBSDFDataIndex(E_OPCODE _op, const IR::INode* _node, const IR::INode::SParameter<IR::INode::color_t>& _opacity)
		{
			switch (_op)
			{
			case OP_INVALID: _IRR_FALLTHROUGH;
			case OP_NOOP:
				return 0ull;
			default: break;
			}

			auto found = m_ctx->bsdfDataIndexMap.find(_node);
			if (found != m_ctx->bsdfDataIndexMap.end())
				return found->second;

			SBSDFUnion data;
			setBSDFData(data, _op, _node, _opacity);
			size_t ix = m_ctx->pBsdfData->size();
			m_ctx->bsdfDataIndexMap.insert({_node,ix});
			m_ctx->pBsdfData->push_back(data);

			return ix;
		}

		VTID packTexture(const IR::INode::STextureSource& tex)
		{
			auto found = m_ctx->VTallocMap.find({tex.image.get(),tex.sampler.get()});
			if (found != m_ctx->VTallocMap.end())
				return found->second;

			auto* img = tex.image->getCreationParameters().image.get();
			auto* sampler = tex.sampler.get();

			const auto& extent = img->getCreationParameters().extent;
			const auto uwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(sampler->getParams().TextureWrapU);
			const auto vwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(sampler->getParams().TextureWrapV);
			const auto border = static_cast<asset::ISampler::E_TEXTURE_BORDER_COLOR>(sampler->getParams().BorderColor);

			auto imgAndOrigSz = asset::ICPUVirtualTexture::createPoTPaddedSquareImageWithMipLevels(img, uwrap, vwrap, border);

			asset::IImage::SSubresourceRange subres;
			subres.baseArrayLayer = 0u;
			subres.layerCount = 1u;
			subres.baseMipLevel = 0u;
			auto mx = std::max(extent.width, extent.height);
			auto round = core::roundUpToPoT<uint32_t>(mx);
			auto lsb = core::findLSB(round);
			subres.levelCount = lsb + 1;

			auto addr = m_ctx->vt->alloc(img->getCreationParameters().format, imgAndOrigSz.second, subres, uwrap, vwrap);
			m_ctx->vt->commit(addr, imgAndOrigSz.first.get(), subres, uwrap, vwrap, border);

			std::pair<SContext::VTallocKey, VTID> item{{tex.image.get(),tex.sampler.get()}, addr};
			m_ctx->VTallocMap.insert(item);

			return addr;
		}

		template <typename ...Params>
		bool push(const instr_t _instr, const IR::INode* _node, const IR::INode::children_array_t& _children, instr_t _parent, Params&& ...args)
		{
			bool pushIt = false;
			instr_t instr = _instr;
			writeInheritableBitfields(instr, _parent);
			switch (getOpcode(instr))
			{
			case OP_BUMPMAP:
				writeBumpmapBitfields(instr);
				_IRR_FALLTHROUGH;
			case OP_DIFFUSE: _IRR_FALLTHROUGH;
			case OP_DIFFTRANS: _IRR_FALLTHROUGH;
			case OP_DIELECTRIC: _IRR_FALLTHROUGH;
			case OP_THINDIELECTRIC: _IRR_FALLTHROUGH;
			case OP_CONDUCTOR: _IRR_FALLTHROUGH;
			case OP_PLASTIC: _IRR_FALLTHROUGH;
			case OP_COATING: _IRR_FALLTHROUGH;
			case OP_BLEND: _IRR_FALLTHROUGH;
			case OP_NOOP:
				pushIt = true;
				break;
			}
			if (pushIt)
				m_stack.push(stack_el_t(instr, _node, _children, std::forward<Params>(args)...));

			return pushIt;
		}
		instr_t finalizeInstr(instr_t _instr, const IR::INode* _node, uint32_t _bsdfBufOffset)
		{
			constexpr E_NDF ndfMap[4]
			{
				NDF_BECKMANN,
				NDF_GGX,
				NDF_PHONG,
				NDF_PHONG
			};

			auto handleSpecularBitfields = [&ndfMap](instr_t dst, const IR::CMicrofacetSpecularBSDFNode* node) -> instr_t
			{
				dst = core::bitfieldInsert<instr_t>(dst, ndfMap[node->ndf], BITFIELDS_SHIFT_NDF, BITFIELDS_WIDTH_NDF);
				if (node->alpha_v.source == IR::INode::EPS_TEXTURE)
					dst = core::bitfieldInsert<instr_t>(dst, 1u, BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
				if (node->alpha_u.source == IR::INode::EPS_TEXTURE)
					dst = core::bitfieldInsert<instr_t>(dst, 1u, BITFIELDS_SHIFT_ALPHA_U_TEX, 1);

				return dst;
			};

			const E_OPCODE op = getOpcode(_instr);
			switch (op)
			{
			case OP_DIFFUSE:
			{
				auto* node = static_cast<const IR::CMicrofacetDiffuseBSDFNode*>(_node);
				if (node->alpha_u.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
				if (node->reflectance.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_REFL_TEX, 1);
			}
			break;
			case OP_DIELECTRIC: _IRR_FALLTHROUGH;
			case OP_THINDIELECTRIC:
			{
				auto* blend = static_cast<const IR::CBSDFCombinerNode*>(_node);
				auto* refl = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(blend->children[0]);
				auto* trans = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(blend->children[1]);
				if (refl->scatteringMode == IR::CMicrofacetSpecularBSDFNode::ESM_TRANSMIT)
					std::swap(refl, trans);
				assert(refl->ndf == trans->ndf);
				assert(refl->alpha_u == trans->alpha_u);
				assert(refl->alpha_v == trans->alpha_v);

				_instr = handleSpecularBitfields(_instr, refl);
			}
			break;
			case OP_CONDUCTOR:
			{
				auto* node = static_cast<const IR::CMicrofacetSpecularBSDFNode*>(_node);
				_instr = handleSpecularBitfields(_instr, node);
			}
			break;
			case OP_PLASTIC:
			{
				auto* node = static_cast<const IR::CBSDFCombinerNode*>(_node);
				assert(node->type == IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR);

				auto* diffuse = static_cast<const IR::CBSDFNode*>(node->children[0]);
				auto* specular = static_cast<const IR::CBSDFNode*>(node->children[1]);
				if (diffuse->type == IR::CBSDFNode::ET_MICROFACET_SPECULAR)
					std::swap(diffuse, specular);

				_instr = handleSpecularBitfields(_instr, static_cast<const IR::CMicrofacetSpecularBSDFNode*>(specular));
				if (static_cast<const IR::CMicrofacetDiffuseBSDFNode*>(diffuse)->reflectance.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_REFL_TEX, 1);
			}
			break;
			case OP_COATING:
			{
				auto* coat = static_cast<const IR::CCoatingBSDFNode*>(_node);

				_instr = handleSpecularBitfields(_instr, coat);
				if (coat->sigmaA.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_SIGMA_A_TEX, 1);
			}
			break;
			case OP_BLEND:
			{
				auto* blend = static_cast<const IR::CBSDFCombinerNode*>(_node);
				assert(blend->type == IR::CBSDFCombinerNode::ET_WEIGHT_BLEND);

				if (static_cast<const IR::CBSDFBlendNode*>(_node)->weight.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_WEIGHT_TEX, 1);
			}
			case OP_DIFFTRANS:
			{
				auto* difftrans = static_cast<const IR::CDifftransBSDFNode*>(_node);

				if (difftrans->transmittance.source == IR::INode::EPS_TEXTURE)
					_instr = core::bitfieldInsert<instr_t>(_instr, 1u, BITFIELDS_SHIFT_TRANS_TEX, 1);
			}
			break;
			}

			_instr = core::bitfieldInsert<instr_t>(_instr, _bsdfBufOffset, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);

			return _instr;
		}

	public:
		ITraveralGenerator(SContext* _ctx, IR* _ir, uint32_t _regCount) : m_ctx(_ctx), m_ir(_ir), m_registerPool(_regCount) {}

		virtual traversal_t genTraversal(const IR::INode* _root, uint32_t& _out_usedRegs) = 0;
	};

namespace remainder_and_pdf
{
	inline uint32_t getNumberOfSrcRegsForOpcode(E_OPCODE _op)
	{
		if (_op == OP_BLEND)
			return 2u;
		else if (_op == OP_BUMPMAP || _op == OP_COATING)
			return 1u;
		return 0u;
	}

	inline core::vector3du32_SIMD getRegisters(const instr_t& i)
	{
		return core::vector3du32_SIMD(
			(i>>INSTR_REG_DST_SHIFT),
			(i>>INSTR_REG_SRC1_SHIFT),
			(i>>INSTR_REG_SRC2_SHIFT)
		) & core::vector3du32_SIMD(INSTR_REG_MASK);
	}

	class CTraversalManipulator
	{
	public:
		using traversal_t = core::vector<instr_t>;
		using substream_t = std::pair<traversal_t::iterator, traversal_t::iterator>;

	private:
		_IRR_STATIC_INLINE_CONSTEXPR instr_t SPECIAL_VAL = ~static_cast<instr_t>(0);

		static void setRegisters(instr_t& i, uint32_t rdst, uint32_t rsrc1 = 0u, uint32_t rsrc2 = 0u)
		{
			uint32_t& regDword = reinterpret_cast<uint32_t*>(&i)[1];

			constexpr uint32_t _2ND_DWORD_SHIFT = 32u;

			regDword = core::bitfieldInsert(regDword, rdst, INSTR_REG_DST_SHIFT-_2ND_DWORD_SHIFT, INSTR_REG_WIDTH);
			regDword = core::bitfieldInsert(regDword, rsrc1, INSTR_REG_SRC1_SHIFT-_2ND_DWORD_SHIFT, INSTR_REG_WIDTH);
			regDword = core::bitfieldInsert(regDword, rsrc2, INSTR_REG_SRC2_SHIFT-_2ND_DWORD_SHIFT, INSTR_REG_WIDTH);
		}

		struct has_different_normal_id
		{
			uint32_t n_id;

			bool operator()(const instr_t& i) const { return getNormalId(i) != n_id; };
		};

		traversal_t m_input;
		core::queue<uint32_t> m_streamLengths;
		uint32_t m_regsPerResult;

		void reorderBumpMapStreams_impl(traversal_t& _input, traversal_t& _output, const substream_t& _stream);

	public:
		CTraversalManipulator(traversal_t&& _traversal, uint32_t _regsPerResult) : m_input(std::move(_traversal)), m_regsPerResult(_regsPerResult) {}

		traversal_t&& process(uint32_t regCount, uint32_t& _out_usedRegs)&&
		{
			reorderBumpMapStreams();
			specifyRegisters(regCount, _out_usedRegs);
			_out_usedRegs += m_regsPerResult;

			return std::move(m_input);
		}

	private:
		//reorders scattered bump-map streams (traversals of BSDF subtrees below bumpmap BSDF node) into continuous streams
		//and moves OP_BUMPMAPs to the beginning of their streams/traversals/subtrees (because obviously they must be executed before BSDFs using them)
		//leaves SPECIAL_VAL to mark original places of beginning of a stream (needed for function specifying registers)
		//WARNING: modifies m_input
		void reorderBumpMapStreams()
		{
			traversal_t result;
			reorderBumpMapStreams_impl(m_input, result, { m_input.begin(),m_input.end() });

			m_input = std::move(result);
		}

		void specifyRegisters(uint32_t regCount, uint32_t& _out_maxUsedReg);
	};


	struct stack_el
	{
		stack_el(instr_t i, const IR::INode* n, const IR::INode::children_array_t& ch, bool v) :
			instr(i), node(n), children(ch), visited(v)
		{}

		instr_t instr;
		const IR::INode* node;
		IR::INode::children_array_t children;
		bool visited;
	};
	class CTraversalGenerator : public ITraveralGenerator<stack_el>
	{
		using base_t = ITraveralGenerator<stack_el>;

		uint32_t m_regsPerRes;

	public:
		CTraversalGenerator(SContext* _ctx, IR* _ir, uint32_t _regCount, uint32_t _regsPerResult) :
			base_t(_ctx,_ir,_regCount), m_regsPerRes(_regsPerResult)
		{}

		void writeInheritableBitfields(instr_t& dst, instr_t parent) const override
		{
			base_t::writeInheritableBitfields(dst, parent);
			
			dst = core::bitfieldInsert(dst, parent>>INSTR_NORMAL_ID_SHIFT, INSTR_NORMAL_ID_SHIFT, INSTR_NORMAL_ID_WIDTH);
		}

		traversal_t genTraversal(const IR::INode* _root, uint32_t& _out_usedRegs) override;
	};
}

namespace gen_choice
{
	struct stack_el
	{
		stack_el(instr_t i, const IR::INode* n, const IR::INode::children_array_t& ch, uint32_t p) :
			instr(i), node(n), children(ch), parentIx(p)
		{}

		instr_t instr;
		const IR::INode* node;
		IR::INode::children_array_t children;
		uint32_t parentIx;
	};
	class CTraversalGenerator : public ITraveralGenerator<stack_el>
	{
		using base_t = ITraveralGenerator<stack_el>;

		uint32_t m_firstFreeNormalID = 0u;

	public:
		using base_t::base_t;

		traversal_t genTraversal(const IR::INode* _root, uint32_t& _out_usedRegs) override;
	};
}
namespace tex_prefetch
{
	static traversal_t genTraversal(const traversal_t& _t, const core::vector<instr_stream::SBSDFUnion>& _bsdfData, core::unordered_map<STextureData, uint32_t, STextureData::hash>& _tex2reg, uint32_t _firstFreeReg, uint32_t& _out_usedRegs, uint32_t& _out_regCntFlags, uint32_t& _out_prefetchFlags);
}
}


using namespace instr_stream;

const IR::INode* instr_stream::CInterpreter::translateMixIntoBlends(IR* ir, const IR::INode* _mix)
{
	auto* mix = static_cast<const IR::CBSDFMixNode*>(_mix);
	assert(mix->symbol == IR::INode::ES_BSDF_COMBINER);

	struct q_el
	{
		const IR::INode* node;
		float weightsSum;
	};
	core::queue<q_el> q;
	for (uint32_t i = 0u; i < mix->children.count; ++i)
		q.push({mix->children[i],mix->weights[i]});

	while (q.size()>1ull)
	{
		auto* blend = ir->allocTmpNode<IR::CBSDFBlendNode>();
		blend->weight.source = IR::INode::EPS_CONSTANT;

		auto left = q.front();
		q.pop();
		auto right = q.front();
		q.pop();

		q_el el{blend, left.weightsSum+right.weightsSum};
		blend->weight.value.constant = right.weightsSum/el.weightsSum;
		blend->children = IR::INode::createChildrenArray(left.node,right.node);

		q.push(el);
	}

	return q.front().node;
}

std::pair<instr_t, const IR::INode*> instr_stream::CInterpreter::processSubtree(IR* ir, const IR::INode* tree, bool thin, IR::INode::children_array_t& out_next)
{
	auto isTwosided = [](const IR::INode* _node, size_t _frontSurfIx, IR::INode::children_array_t& _frontSurfChildren) -> bool
	{
		bool ts = false;
		auto* front = _node->children[_frontSurfIx];
		if (_node->children.find(IR::INode::ES_BACK_SURFACE, &_frontSurfIx))
		{
			auto* back = _node->children[_frontSurfIx];
			assert(front->children == back->children);//TODO handle this in some other way
			ts = true;
		}
		_frontSurfChildren = front->children;

		return ts;
	};

	instr_t instr = OP_INVALID;
	switch (tree->symbol)
	{
	case IR::INode::ES_MATERIAL:
	{
		bool twosided = false;
		bool masked = false;

		auto* node = static_cast<const IR::CMaterialNode*>(tree);
		masked = (node->opacity.source==IR::INode::EPS_TEXTURE || (node->opacity.source==IR::INode::EPS_TEXTURE && (node->opacity.value.constant!=IR::INode::color_t(1.f)).xyzz().all()));

		instr_t retval = OP_INVALID;

		size_t ix = 0u;
		if (node->children.find(IR::INode::ES_GEOM_MODIFIER, &ix))
		{
			auto* geom_node = static_cast<const IR::CGeomModifierNode*>(node->children[ix]);
			out_next = geom_node->children;
			if (geom_node->children.find(IR::INode::ES_FRONT_SURFACE, &ix))
				twosided = isTwosided(geom_node, ix, out_next);
			retval = (geom_node->type != IR::CGeomModifierNode::ET_DERIVATIVE) ? OP_INVALID : OP_BUMPMAP;
		}
		else if (node->children.find(IR::INode::ES_FRONT_SURFACE, &ix))
		{
			twosided = isTwosided(node, ix, out_next);
			retval = OP_NOOP; //returning OP_NOOP causes not adding this instruction to resulting stream, but doesnt stop processing (effectively is used to forward flags only)
		}
		if (twosided)
			retval = core::bitfieldInsert<instr_t>(retval, 1u, BITFIELDS_SHIFT_TWOSIDED, 1);
		if (masked)
			retval = core::bitfieldInsert<instr_t>(retval, 1u, BITFIELDS_SHIFT_MASKFLAG, 1);
		instr = retval;
	}
	break;
	case IR::INode::ES_GEOM_MODIFIER:
	{
		bool twosided = false;
		auto* node = static_cast<const IR::CGeomModifierNode*>(tree);
		size_t ix;
		if (node->children.find(IR::INode::ES_FRONT_SURFACE, &ix))
			twosided = isTwosided(node, ix, out_next);

		instr_t retval;
		if (node->type == IR::CGeomModifierNode::ET_HEIGHT)
			retval = OP_BUMPMAP;
		else
			retval = OP_INVALID;

		if (twosided)
			retval = core::bitfieldInsert<instr_t>(retval, 1u, BITFIELDS_SHIFT_TWOSIDED, 1);

		instr = retval;
	}
	break;
	case IR::INode::ES_BSDF_COMBINER:
	{
		auto* node = static_cast<const IR::CBSDFCombinerNode*>(tree);
		switch (node->type)
		{
		case IR::CBSDFCombinerNode::ET_WEIGHT_BLEND:
			out_next = node->children;
			instr = OP_BLEND;
			break;
		case IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR:
		{
			assert(node->children.count==2u);
			assert(node->children[0]->symbol==IR::INode::ES_BSDF && node->children[1]->symbol==IR::INode::ES_BSDF);
			auto type = static_cast<const IR::CBSDFNode*>(node->children[0])->type;
			assert(type==IR::CBSDFNode::ET_MICROFACET_DIFFUSE || type==IR::CBSDFNode::ET_MICROFACET_SPECULAR);
			type = static_cast<const IR::CBSDFNode*>(node->children[1])->type;
			assert(type==IR::CBSDFNode::ET_MICROFACET_DIFFUSE || type==IR::CBSDFNode::ET_MICROFACET_SPECULAR);
			instr = OP_PLASTIC;
		}
		break;
		case IR::CBSDFCombinerNode::ET_FRESNEL_BLEND:
		{
			assert(node->children.count==2u);
			assert(node->children[0]->symbol==IR::INode::ES_BSDF && node->children[1]->symbol==IR::INode::ES_BSDF);
			auto type = static_cast<const IR::CBSDFNode*>(node->children[0])->type;
			assert(type == IR::CBSDFNode::ET_MICROFACET_SPECULAR);
			type = static_cast<const IR::CBSDFNode*>(node->children[1])->type;
			assert(type == IR::CBSDFNode::ET_MICROFACET_SPECULAR);
			instr = thin ? OP_THINDIELECTRIC : OP_DIELECTRIC;
		}
		break;
		case IR::CBSDFCombinerNode::ET_MIX:
		{
			tree = translateMixIntoBlends(ir, node);
			instr = OP_BLEND;
			out_next = tree->children;
		}
		break;
		}
	}
	break;
	case IR::INode::ES_BSDF:
	{
		auto* node = static_cast<const IR::CBSDFNode*>(tree);
		switch (node->type)
		{
		case IR::CBSDFNode::ET_DIFFTRANS:
			instr = OP_DIFFTRANS; break;
		case IR::CBSDFNode::ET_MICROFACET_DIFFUSE:
			instr = OP_DIFFUSE; break;
		case IR::CBSDFNode::ET_MICROFACET_SPECULAR:
			instr = OP_CONDUCTOR; break;
		case IR::CBSDFNode::ET_COATING:
			out_next = node->children;
			assert(node->children.count==1u);
			instr = OP_COATING;
			break;
		}
	}
	break;
	}

	return {instr,tree};
}

void instr_stream::remainder_and_pdf::CTraversalManipulator::reorderBumpMapStreams_impl(traversal_t& _input, traversal_t& _output, const substream_t& _stream)
{
	const uint32_t n_id = getNormalId(*(_stream.second - 1));

	const size_t len = _stream.second - _stream.first;
	size_t subsLenAcc = 0ull;

	core::stack<substream_t> substreams;
	auto subBegin = std::find_if(_stream.first, _stream.second, has_different_normal_id{ n_id });
	while (subBegin != _stream.second)
	{
		const uint32_t sub_n_id = getNormalId(*subBegin);
		decltype(subBegin) subEnd;
		for (subEnd = _stream.second - 1; subEnd != subBegin; --subEnd)
			if (getNormalId(*subEnd) == sub_n_id)
				break;
		++subEnd;

		//one place will be used for SPECIAL_VAL (hence -1)
		subsLenAcc += subEnd - subBegin - 1;
		substreams.push({ subBegin,subEnd });

		subBegin = std::find_if(subEnd, _stream.second, has_different_normal_id{ n_id });
	}

	while (!substreams.empty())
	{
		reorderBumpMapStreams_impl(_input, _output, substreams.top());
		substreams.pop();
	}

	const uint32_t newlen = len - subsLenAcc;

	substream_t woSubs{ _stream.first,_stream.first + newlen };
	//move bumpmap instruction to the beginning of the stream
	auto lastInstr = *(_stream.first + newlen - 1);
	if (getOpcode(lastInstr) == OP_BUMPMAP)
	{
		_input.erase(_stream.first + newlen - 1);
		_input.insert(_stream.first, lastInstr);
	}

	//important for next stage of processing
	m_streamLengths.push(newlen);

	_output.insert(_output.end(), woSubs.first, woSubs.second);
	*woSubs.first = SPECIAL_VAL;
	//do not erase SPECIAL_VAL (hence +1)
	_input.erase(woSubs.first + 1, woSubs.second);
}

void instr_stream::remainder_and_pdf::CTraversalManipulator::specifyRegisters(uint32_t regCount, uint32_t& _out_maxUsedReg)
{
	core::stack<uint32_t> freeRegs;
	{
		regCount /= m_regsPerResult;
		for (uint32_t i = 0u; i < regCount; ++i)
			freeRegs.push(m_regsPerResult*(regCount - 1u - i));
	}
	//registers with result of bump-map substream
	core::stack<uint32_t> bmRegs;
	//registers waiting to be used as source
	core::stack<uint32_t> srcRegs;

	int32_t bmStreamEndCounter = 0;
	auto pushResultRegister = [&bmStreamEndCounter, &bmRegs, &srcRegs](uint32_t _resultReg)
	{
		core::stack<uint32_t>& stack = bmStreamEndCounter == 0 ? bmRegs : srcRegs;
		stack.push(_resultReg);
	};
	for (uint32_t j = 0u; j < m_input.size();)
	{
		instr_t& i = m_input[j];

		if (i == SPECIAL_VAL)
		{
			srcRegs.push(bmRegs.top());
			bmRegs.pop();
			m_input.erase(m_input.begin() + j);

			continue;// do not increment j
		}
		const E_OPCODE op = getOpcode(i);

		--bmStreamEndCounter;
		if (op == OP_BUMPMAP)
		{
			bmStreamEndCounter = m_streamLengths.front() - 1u;
			m_streamLengths.pop();

			//OP_BUMPMAP doesnt care about usual registers, so dont set them
			++j;
			continue;
		}
		//if bmStreamEndCounter reaches value of -1 and next instruction is not OP_BUMPMAP, then emit some SET_GEOM_NORMAL instr
		else if (bmStreamEndCounter == -1)
		{
			//just opcode, no registers nor other bitfields in this instruction
			m_input.insert(m_input.begin() + j, instr_t(OP_SET_GEOM_NORMAL));

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

		switch (srcsNum)
		{
		case 2u:
		{
			const uint32_t src2 = srcs[0];
			const uint32_t src1 = srcs[1];
			const uint32_t dst = (j == m_input.size() - 1u) ? 0u : src1;
			pushResultRegister(dst);
			freeRegs.push(src2);
			setRegisters(i, dst, src1, src2);
		}
		break;
		case 1u:
		{
			const uint32_t src = srcs[0];
			const uint32_t dst = (j == m_input.size() - 1u) ? 0u : src;
			pushResultRegister(dst);
			setRegisters(i, dst, src);
		}
		break;
		case 0u:
		{
			assert(!freeRegs.empty());
			uint32_t dst = 0u;
			if (j < m_input.size() - 1u)
			{
				dst = freeRegs.top();
				freeRegs.pop();
			}
			pushResultRegister(dst);
			setRegisters(i, dst);
		}
		break;
		default: break;
		}

		++j;
	}
}

instr_stream::traversal_t instr_stream::remainder_and_pdf::CTraversalGenerator::genTraversal(const IR::INode* _root, uint32_t& _out_usedRegs)
{
	traversal_t traversal;

	assert(_root->symbol == IR::INode::ES_MATERIAL);
	const bool thin = static_cast<const IR::CMaterialNode*>(_root)->thin;

	auto opacity = extractOpacity(_root);
	{
		IR::INode::children_array_t next;
		const IR::INode* node;
		instr_t instr;
		std::tie(instr, node) = processSubtree(_root, thin, next);
		push(instr, node, next, instr, false);
	}
	while (!m_stack.empty())
	{
		auto& top = m_stack.top();
		const E_OPCODE op = getOpcode(top.instr);
		//const uint32_t srcRegCount = getNumberOfSrcRegsForOpcode(op);
		_IRR_DEBUG_BREAK_IF(op == OP_INVALID);
		if (top.children.count==0u || top.visited)
		{
			const uint32_t bsdfBufIx = getBSDFDataIndex(getOpcode(top.instr), top.node, opacity);
			instr_t instr = finalizeInstr(top.instr, top.node, bsdfBufIx);
			traversal.push_back(instr);
			m_stack.pop();
		}
		else if (!top.visited)
		{
			top.visited = true;

			size_t pushedCount = 0ull;
			for (auto it = top.children.end() - 1; it != top.children.begin() - 1; --it)
			{
				const IR::INode* node = nullptr;
				IR::INode::children_array_t next2;
				instr_t instr2;
				std::tie(instr2, node) = processSubtree(*it, thin, next2);
				pushedCount += push(instr2, node, next2, top.instr, false);
			}
			_IRR_DEBUG_BREAK_IF(pushedCount > 2ull);
		}
	}

	//remove NOOPs
	filterNOOPs(traversal);

	traversal = std::move(CTraversalManipulator(std::move(traversal), m_regsPerRes).process(m_registerPool, _out_usedRegs));

	return traversal;
}

instr_stream::traversal_t instr_stream::gen_choice::CTraversalGenerator::genTraversal(const IR::INode* _root, uint32_t& _out_usedRegs)
{
	constexpr uint32_t INVALID_INDEX = 0xdeadbeefu;

	assert(_root->symbol == IR::INode::ES_MATERIAL);
	const bool thin = static_cast<const IR::CMaterialNode*>(_root)->thin;

	traversal_t traversal;

	auto opacity = extractOpacity(_root);
	{
		IR::INode::children_array_t next;
		const IR::INode* node;
		instr_t instr;
		std::tie(instr, node) = processSubtree(_root, thin, next);
		push(instr, node, next, instr, INVALID_INDEX);
	}
	while (!m_stack.empty())
	{
		auto top = m_stack.top();
		m_stack.pop();
		const E_OPCODE op = getOpcode(top.instr);

		const uint32_t bsdfBufIx = getBSDFDataIndex(op, top.node, opacity);
		const uint32_t currIx = traversal.size();
		traversal.push_back(finalizeInstr(top.instr, top.node, bsdfBufIx));
		_IRR_DEBUG_BREAK_IF(op == OP_INVALID);
		if (top.parentIx != INVALID_INDEX)
		{
			instr_t& parent = traversal[top.parentIx];
			parent = core::bitfieldInsert<instr_t>(parent, currIx, INSTR_RIGHT_JUMP_SHIFT, INSTR_RIGHT_JUMP_WIDTH);
		}

		{
			size_t pushedCount = 0ull;
			for (auto it = top.children.end() - 1; it != top.children.begin() - 1; --it)
			{
				const IR::INode* node = nullptr;
				IR::INode::children_array_t next2;
				instr_t instr2;
				std::tie(instr2, node) = processSubtree(*it, thin, next2);
				pushedCount += push(instr2, node, next2, top.instr, it == top.children.begin()+1 ? currIx : INVALID_INDEX);
			}
			_IRR_DEBUG_BREAK_IF(pushedCount>2ull);
		}
	}

	//remove NOOPs
	filterNOOPs(traversal);
	if (getOpcode(traversal.front())!=OP_BUMPMAP)
		traversal.insert(traversal.begin(), OP_SET_GEOM_NORMAL);

	_out_usedRegs = 0u;

	return traversal;
}

instr_stream::traversal_t instr_stream::tex_prefetch::genTraversal(const traversal_t& _t, const core::vector<instr_stream::SBSDFUnion>& _bsdfData, core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _out_tex2reg, uint32_t _firstFreeReg, uint32_t& _out_usedRegs, uint32_t& _out_regCntFlags, uint32_t& _out_prefetchFlags)
{
	core::unordered_set<STextureData, STextureData::hash> processed;
	uint32_t regNum = _firstFreeReg;
	auto write_fetch_bitfields = [&processed,&regNum,&_out_regCntFlags,&_out_prefetchFlags,&_out_tex2reg](instr_t& i, const STextureData& tex, int32_t dstbit, int32_t srcbit, int32_t reg_bitfield_shift, int32_t reg_count_bitfield_shift, uint32_t reg_count) -> bool {
		auto bit = core::bitfieldExtract(i, srcbit, 1);
		uint32_t reg = 0u;
		if (bit)
		{
			if (processed.find(tex)==processed.end()) {
				processed.insert(tex);
				reg = regNum;
				regNum += reg_count;
			}
			else
				bit = 0;
		}
		i = core::bitfieldInsert(i, bit, dstbit, 1);
		assert(reg<(1u<<BITFIELDS_REG_WIDTH));
		i = core::bitfieldInsert<instr_t>(i, reg, reg_bitfield_shift, BITFIELDS_REG_WIDTH);
		i = core::bitfieldInsert<instr_t>(i, reg_count, reg_count_bitfield_shift, BITFIELDS_FETCH_TEX_REG_CNT_WIDTH);

		if (bit) {
			_out_tex2reg.insert({tex,reg});
			_out_regCntFlags |= (1u << reg_count);
			_out_prefetchFlags |= (1u << dstbit);
		}

		return bit;
	};


	traversal_t traversal;
		
	for (instr_t i : _t)
	{
		const E_OPCODE op = getOpcode(i);

		if (op==OP_NOOP || op==OP_INVALID || op==OP_SET_GEOM_NORMAL)
			continue;

		//zero-out fetch flags
		i = core::bitfieldInsert<instr_t>(i, 0, BITFIELDS_FETCH_TEX_0_SHIFT, 1);
		i = core::bitfieldInsert<instr_t>(i, 0, BITFIELDS_FETCH_TEX_1_SHIFT, 1);
		i = core::bitfieldInsert<instr_t>(i, 0, BITFIELDS_FETCH_TEX_2_SHIFT, 1);
		i = core::bitfieldInsert<instr_t>(i, 0, BITFIELDS_FETCH_TEX_3_SHIFT, 1);
			
		const uint32_t bsdf_ix = core::bitfieldExtract(i, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		const SBSDFUnion& bsdf_data = _bsdfData[bsdf_ix];

		switch (op)
		{
		case OP_PLASTIC:
			write_fetch_bitfields(i, bsdf_data.plastic.alpha_v.tex, BITFIELDS_FETCH_TEX_1_SHIFT, BITFIELDS_SHIFT_ALPHA_V_TEX, BITFIELDS_REG_1_SHIFT, BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT, 1u);
			_IRR_FALLTHROUGH;
		case OP_DIFFUSE:
			write_fetch_bitfields(i, bsdf_data.param[0].tex, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_SHIFT_ALPHA_U_TEX, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 1u);//alpha_u
			write_fetch_bitfields(i, bsdf_data.param[3].tex, BITFIELDS_FETCH_TEX_3_SHIFT, BITFIELDS_SHIFT_REFL_TEX, BITFIELDS_REG_3_SHIFT, BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT, 3u);//reflectance
			break;
		case OP_DIFFTRANS:
			write_fetch_bitfields(i, bsdf_data.difftrans.transmittance.tex, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_SHIFT_TRANS_TEX, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 3u);
			break;
		case OP_DIELECTRIC: _IRR_FALLTHROUGH;
		case OP_THINDIELECTRIC: _IRR_FALLTHROUGH;
		case OP_CONDUCTOR:
			write_fetch_bitfields(i, bsdf_data.param[0].tex, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_SHIFT_ALPHA_U_TEX, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 1u);//alpha_u
			write_fetch_bitfields(i, bsdf_data.param[1].tex, BITFIELDS_FETCH_TEX_1_SHIFT, BITFIELDS_SHIFT_ALPHA_V_TEX, BITFIELDS_REG_1_SHIFT, BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT, 1u);//alpha_v
			break;
		case OP_COATING:
			write_fetch_bitfields(i, bsdf_data.coating.alpha_u.tex, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_SHIFT_ALPHA_U_TEX, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 1u);
			write_fetch_bitfields(i, bsdf_data.coating.alpha_v.tex, BITFIELDS_FETCH_TEX_1_SHIFT, BITFIELDS_SHIFT_ALPHA_V_TEX, BITFIELDS_REG_1_SHIFT, BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT, 1u);
			write_fetch_bitfields(i, bsdf_data.coating.sigmaA.tex, BITFIELDS_FETCH_TEX_3_SHIFT, BITFIELDS_SHIFT_SIGMA_A_TEX, BITFIELDS_REG_3_SHIFT, BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT, 3u);
			break;
		case OP_BUMPMAP:
			i = core::bitfieldInsert<instr_t>(i, 1u, BITFIELDS_FETCH_TEX_0_SHIFT, 1);
			write_fetch_bitfields(i, bsdf_data.bumpmap.derivmap, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 2u);
			break;
		case OP_BLEND:
			write_fetch_bitfields(i, bsdf_data.blend.weight.tex, BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_SHIFT_WEIGHT_TEX, BITFIELDS_REG_0_SHIFT, BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, 1u);
			break;
		}
		//opacity is common for all
		write_fetch_bitfields(i, bsdf_data.param[2].tex, BITFIELDS_FETCH_TEX_2_SHIFT, BITFIELDS_SHIFT_OPACITY_TEX, BITFIELDS_REG_2_SHIFT, BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT, 3u);

		//do not add instruction to stream if it doesnt need any texture or all needed textures are already being fetched by previous instructions
		if (getTexFetchFlags(i) == 0u)
			continue;

		traversal.push_back(i);
	}

	_out_usedRegs = regNum-_firstFreeReg;

	return traversal;
}

std::string CMaterialCompilerGLSLBackendCommon::genPreprocDefinitions(const result_t& _res, bool _genChoiceStream)
{
	using namespace std::string_literals;

	std::string defs;
	defs += "\n#define REG_COUNT " + std::to_string(_res.usedRegisterCount);

	for (E_OPCODE op : _res.opcodes)
		defs += "\n#define "s + OPCODE_NAMES[op] + " " + std::to_string(op);
	defs += "\n#define OP_MAX_BRDF " + std::to_string(OP_COATING);
	defs += "\n#define OP_MAX_BSDF " + std::to_string(OP_THINDIELECTRIC);

	for (E_NDF ndf : _res.NDFs)
		defs += "\n#define "s + NDF_NAMES[ndf] + " " + std::to_string(ndf);
	if (_res.NDFs.size()==1ull)
		defs += "\n#define ONLY_ONE_NDF";

	defs += "\n#define sizeof_bsdf_data " + std::to_string((sizeof(SBSDFUnion)+instr_stream::sizeof_uvec4-1u)/instr_stream::sizeof_uvec4);

	if (_genChoiceStream)
		defs += "\n#define GEN_CHOICE_STREAM";
	if (!_res.noPrefetchStream)
		defs += "\n#define TEX_PREFETCH_STREAM";
	if (!_res.noNormPrecompStream)
		defs += "\n#define NORM_PRECOMP_STREAM";
	if (_res.allIsotropic)
		defs += "\n#define ALL_ISOTROPIC_BXDFS";
	if (_res.noTwosided)
		defs += "\n#define NO_TWOSIDED";

	//instruction bitfields
	defs += "\n#define INSTR_OPCODE_MASK " + std::to_string(INSTR_OPCODE_MASK);
	defs += "\n#define INSTR_BSDF_BUF_OFFSET_SHIFT " + std::to_string(BITFIELDS_BSDF_BUF_OFFSET_SHIFT);
	defs += "\n#define INSTR_BSDF_BUF_OFFSET_MASK " + std::to_string(BITFIELDS_BSDF_BUF_OFFSET_MASK);
	defs += "\n#define INSTR_NDF_SHIFT " + std::to_string(BITFIELDS_SHIFT_NDF);
	defs += "\n#define INSTR_NDF_MASK " + std::to_string(BITFIELDS_MASK_NDF);
	defs += "\n#define INSTR_ALPHA_U_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_ALPHA_U_TEX);
	defs += "\n#define INSTR_ALPHA_V_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_ALPHA_V_TEX);
	defs += "\n#define INSTR_REFL_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_REFL_TEX);
	defs += "\n#define INSTR_TRANS_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_TRANS_TEX);
	defs += "\n#define INSTR_SIGMA_A_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_SIGMA_A_TEX);
	defs += "\n#define INSTR_WEIGHT_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_WEIGHT_TEX);
	defs += "\n#define INSTR_OPACITY_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_OPACITY_TEX);
	defs += "\n#define INSTR_TWOSIDED_SHIFT " + std::to_string(BITFIELDS_SHIFT_TWOSIDED);
	defs += "\n#define INSTR_MASKFLAG_SHIFT " + std::to_string(BITFIELDS_SHIFT_MASKFLAG);
	defs += "\n#define INSTR_1ST_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_1ST_PARAM_TEX);
	defs += "\n#define INSTR_2ND_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_2ND_PARAM_TEX);
	defs += "\n#define INSTR_3RD_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_3RD_PARAM_TEX);
	defs += "\n#define INSTR_4TH_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_4TH_PARAM_TEX);
	defs += "\n#define INSTR_NORMAL_ID_SHIFT " + std::to_string(INSTR_NORMAL_ID_SHIFT);
	defs += "\n#define INSTR_NORMAL_ID_MASK " + std::to_string(INSTR_NORMAL_ID_MASK);
	//remainder_and_pdf
	{
		using namespace remainder_and_pdf;

		defs += "\n#define INSTR_REG_MASK " + std::to_string(INSTR_REG_MASK);
		defs += "\n#define INSTR_REG_DST_SHIFT " + std::to_string(INSTR_REG_DST_SHIFT);
		defs += "\n#define INSTR_REG_SRC1_SHIFT " + std::to_string(INSTR_REG_SRC1_SHIFT);
		defs += "\n#define INSTR_REG_SRC2_SHIFT " + std::to_string(INSTR_REG_SRC2_SHIFT);
	}
	//gen_choice
	{
		using namespace gen_choice;

		defs += "\n#define INSTR_RIGHT_JUMP_SHIFT " + std::to_string(INSTR_RIGHT_JUMP_SHIFT);
		defs += "\n#define INSTR_RIGHT_JUMP_WIDTH " + std::to_string(INSTR_RIGHT_JUMP_WIDTH);
	}
	//tex_prefetch
	{
		using namespace tex_prefetch;

		defs += "\n#define INSTR_FETCH_FLAG_TEX_0_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_0_SHIFT);
		defs += "\n#define INSTR_FETCH_FLAG_TEX_1_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_1_SHIFT);
		defs += "\n#define INSTR_FETCH_FLAG_TEX_2_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_2_SHIFT);
		defs += "\n#define INSTR_FETCH_FLAG_TEX_3_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_3_SHIFT);

		defs += "\n#define INSTR_FETCH_TEX_0_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT);
		defs += "\n#define INSTR_FETCH_TEX_1_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT);
		defs += "\n#define INSTR_FETCH_TEX_2_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT);
		defs += "\n#define INSTR_FETCH_TEX_3_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT);
		defs += "\n#define INSTR_FETCH_TEX_REG_CNT_MASK " + std::to_string(BITFIELDS_FETCH_TEX_REG_CNT_MASK);

		defs += "\n#define INSTR_PREFETCH_REG_MASK " + std::to_string((1u<<BITFIELDS_REG_WIDTH) - 1u);
		defs += "\n#define INSTR_PREFETCH_REG_0_SHIFT " + std::to_string(BITFIELDS_REG_0_SHIFT);
		defs += "\n#define INSTR_PREFETCH_REG_1_SHIFT " + std::to_string(BITFIELDS_REG_1_SHIFT);
		defs += "\n#define INSTR_PREFETCH_REG_2_SHIFT " + std::to_string(BITFIELDS_REG_2_SHIFT);
		defs += "\n#define INSTR_PREFETCH_REG_3_SHIFT " + std::to_string(BITFIELDS_REG_3_SHIFT);

		if (_res.globalPrefetchRegCountFlags & (1u << 1))
			defs += "\n#define PREFETCH_REG_COUNT_1";
		if (_res.globalPrefetchRegCountFlags & (1u << 2))
			defs += "\n#define PREFETCH_REG_COUNT_2";
		if (_res.globalPrefetchRegCountFlags & (1u << 3))
			defs += "\n#define PREFETCH_REG_COUNT_3";
		if (core::bitfieldExtract(_res.globalPrefetchFlags, BITFIELDS_FETCH_TEX_0_SHIFT, 1))
			defs += "\n#define PREFETCH_TEX_0";
		if (core::bitfieldExtract(_res.globalPrefetchFlags, BITFIELDS_FETCH_TEX_1_SHIFT, 1))
			defs += "\n#define PREFETCH_TEX_1";
		if (core::bitfieldExtract(_res.globalPrefetchFlags, BITFIELDS_FETCH_TEX_2_SHIFT, 1))
			defs += "\n#define PREFETCH_TEX_2";
		if (core::bitfieldExtract(_res.globalPrefetchFlags, BITFIELDS_FETCH_TEX_3_SHIFT, 1))
			defs += "\n#define PREFETCH_TEX_3";
	}

	//parameter numbers
	{
		defs += "\n#define PARAMS_ALPHA_U_IX " + std::to_string(ALPHA_U_TEX_IX);
		defs += "\n#define PARAMS_ALPHA_V_IX " + std::to_string(ALPHA_V_TEX_IX);
		defs += "\n#define PARAMS_REFLECTANCE_IX " + std::to_string(REFLECTANCE_TEX_IX);
		defs += "\n#define PARAMS_TRANSMITTANCE_IX " + std::to_string(TRANSMITTANCE_TEX_IX);
		defs += "\n#define PARAMS_SIGMA_A_IX " + std::to_string(SIGMA_A_TEX_IX);
		defs += "\n#define PARAMS_WEIGHT_IX " + std::to_string(WEIGHT_TEX_IX);
		defs += "\n#define PARAMS_OPACITY_IX " + std::to_string(OPACITY_TEX_IX);
	}

	defs += "\n";

	return defs;
}

core::unordered_map<uint32_t, uint32_t> CMaterialCompilerGLSLBackendCommon::createBsdfDataIndexMapForPrefetchedTextures(SContext* _ctx, const instr_stream::traversal_t& _tex_prefetch_stream, const core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _tex2reg) const
{
	core::unordered_map<uint32_t, uint32_t> ix2ix;

	for (instr_t instr : _tex_prefetch_stream)
	{
		const auto bsdf_ix = core::bitfieldExtract(instr, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		{
			auto found = ix2ix.find(bsdf_ix);
			if (found != ix2ix.end())
				continue;
		}

		const SBSDFUnion& bsdf_data = (*_ctx->pBsdfData)[bsdf_ix];
		auto new_bsdf_data = bsdf_data;
		constexpr uint32_t tex_fetch_flag_shift[tex_prefetch::BITFIELDS_FETCH_TEX_COUNT]{
			tex_prefetch::BITFIELDS_FETCH_TEX_0_SHIFT,
			tex_prefetch::BITFIELDS_FETCH_TEX_1_SHIFT,
			tex_prefetch::BITFIELDS_FETCH_TEX_2_SHIFT,
			tex_prefetch::BITFIELDS_FETCH_TEX_3_SHIFT
		};
		for (uint32_t i = 0u; i < SBSDFUnion::MAX_TEXTURES; ++i)
			if (core::bitfieldExtract(instr, tex_fetch_flag_shift[i], 1))
			{
				auto found = _tex2reg.find(bsdf_data.param[i].tex);
				if (found == _tex2reg.end())
					continue;

				new_bsdf_data.param[i].tex.prefetch_reg = found->second;
			}

		ix2ix.insert({ bsdf_ix, _ctx->pBsdfData->size() });
		_ctx->pBsdfData->push_back(new_bsdf_data);
	}

	return ix2ix;
}

auto CMaterialCompilerGLSLBackendCommon::compile(SContext* _ctx, IR* _ir, bool _computeGenChoiceStream) -> result_t
{
	result_t res;
	_ctx->pBsdfData = &res.bsdfData;
	res.noNormPrecompStream = true;
	res.noPrefetchStream = true;
	res.usedRegisterCount = 0u;
	res.globalPrefetchFlags = 0u;
	res.globalPrefetchRegCountFlags = 0u;

	for (const IR::INode* root : _ir->roots)
	{
		uint32_t registerPool = instr_stream::MAX_REGISTER_COUNT;

		uint32_t usedRegs{};
		traversal_t rem_pdf_stream;
		{
			//In case of presence of generator choice stream, remainder_and_pdf stream has 2 roles in raster backend:
			//* eval stream
			//* remainder-and-pdf stream (for use in envmap sampling, as an example); in which case instructions need to write their PDF as well
			//In raytracing backend _computeGenChoiceStream is always true
			const uint32_t regsPerRes = _computeGenChoiceStream ? 4u : 3u;

			remainder_and_pdf::CTraversalGenerator gen(_ctx, _ir, registerPool, regsPerRes);
			rem_pdf_stream = gen.genTraversal(root, usedRegs);
			assert(usedRegs <= registerPool);
			registerPool -= usedRegs;
		}
		traversal_t gen_choice_stream;
		if (_computeGenChoiceStream)
		{
			gen_choice::CTraversalGenerator gen(_ctx, _ir, registerPool);
			gen_choice_stream = gen.genTraversal(root, usedRegs);
			assert(usedRegs <= registerPool);
			registerPool -= usedRegs;
		}
		core::unordered_map<uint32_t, uint32_t> ix2ix;
		traversal_t tex_prefetch_stream;
		{
			core::unordered_map<STextureData, uint32_t, STextureData::hash> tex2reg;
			tex_prefetch_stream = tex_prefetch::genTraversal(rem_pdf_stream, res.bsdfData, tex2reg, MAX_REGISTER_COUNT-registerPool, usedRegs, res.globalPrefetchRegCountFlags, res.globalPrefetchFlags);
			assert(usedRegs <= registerPool);
			registerPool -= usedRegs;

			ix2ix = createBsdfDataIndexMapForPrefetchedTextures(_ctx, tex_prefetch_stream, tex2reg);
		}

		const uint32_t regNum = MAX_REGISTER_COUNT-registerPool;

		adjustBSDFDataIndices(rem_pdf_stream, ix2ix);
		adjustBSDFDataIndices(gen_choice_stream, ix2ix);

		traversal_t normal_precomp_stream;
		{
			normal_precomp_stream.reserve(std::count_if(rem_pdf_stream.begin(), rem_pdf_stream.end(), [](instr_t i) {return getOpcode(i)==OP_BUMPMAP;}));
			assert(regNum+3u*normal_precomp_stream.capacity() <= MAX_REGISTER_COUNT);
			for (instr_t i : rem_pdf_stream)
			{
				if (getOpcode(i)==OP_BUMPMAP)
				{
					//we can be sure that n_id is always in range [0,count of bumpmap instrs)
					const uint32_t n_id = getNormalId(i);
					i = core::bitfieldInsert<instr_t>(i, regNum+3u*n_id, normal_precomp::BITFIELDS_REG_DST_SHIFT, normal_precomp::BITFIELDS_REG_WIDTH);
					normal_precomp_stream.push_back(i);
				}
			}
			registerPool = MAX_REGISTER_COUNT - regNum - 3u*normal_precomp_stream.size();
		}

		//src1 reg for OP_BUMPMAPs is set to dst reg of corresponding instruction in normal precomp stream
		setSourceRegForBumpmaps(rem_pdf_stream, regNum);
		setSourceRegForBumpmaps(gen_choice_stream, regNum);

		result_t::instr_streams_t streams;
#define STREAM(first,count) result_t::instr_streams_t::stream_t {static_cast<uint32_t>(first),static_cast<uint32_t>(count)}
		streams.rem_and_pdf = STREAM(res.instructions.size(), rem_pdf_stream.size());
		res.instructions.insert(res.instructions.end(), rem_pdf_stream.begin(), rem_pdf_stream.end());
		streams.gen_choice = STREAM(res.instructions.size(), gen_choice_stream.size());
		res.instructions.insert(res.instructions.end(), gen_choice_stream.begin(), gen_choice_stream.end());
		streams.tex_prefetch = STREAM(res.instructions.size(), tex_prefetch_stream.size());
		res.instructions.insert(res.instructions.end(), tex_prefetch_stream.begin(), tex_prefetch_stream.end());
		streams.norm_precomp = STREAM(res.instructions.size(), normal_precomp_stream.size());
		res.instructions.insert(res.instructions.end(), normal_precomp_stream.begin(), normal_precomp_stream.end());
#undef STREAM

		res.streams.insert({root,streams});

		res.noNormPrecompStream = res.noNormPrecompStream && (streams.norm_precomp.count==0u);
		res.noPrefetchStream = res.noPrefetchStream && (streams.tex_prefetch.count==0u);
		res.usedRegisterCount = std::max(res.usedRegisterCount, instr_stream::MAX_REGISTER_COUNT-registerPool);
	}

	_ir->deinitTmpNodes();

	auto isAniso = [&res](instr_stream::instr_t _i) -> bool {
		const E_OPCODE op = getOpcode(_i);
		if (!instr_stream::opHasSpecular(op))
			return false;

		const bool au_tex = core::bitfieldExtract(_i, BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		const bool av_tex = core::bitfieldExtract(_i, BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
		if (au_tex != av_tex)
			return false;

		const uint32_t ix = getBSDFDataIx(_i);
		
		const auto& au = res.bsdfData[ix].param[0];
		const auto& av = res.bsdfData[ix].param[1];

		if (au_tex)
			return au.tex.prefetch_reg != av.tex.prefetch_reg;
		else
			return au.constant[0] != av.constant[0];
	};

	res.allIsotropic = true;
	res.noTwosided = true;
	for (const auto& e : res.streams)
	{
		const result_t::instr_streams_t& streams = e.second;
		for (uint32_t i = 0u; i < streams.rem_and_pdf.count; ++i) {
			const uint32_t first = streams.rem_and_pdf.first;
			const instr_t instr = res.instructions[first+i];

			const bool ts = core::bitfieldExtract(instr, BITFIELDS_SHIFT_TWOSIDED, 1);
			res.noTwosided = res.noTwosided && !ts;

			const E_OPCODE op = instr_stream::getOpcode(instr);
			const E_NDF ndf = instr_stream::getNDF(instr);
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
		genPreprocDefinitions(res, _computeGenChoiceStream) +
R"(
#include <irr/builtin/material_compiler/glsl/common_declarations.glsl>
)";

	return res;
}

}
void material_compiler::CMaterialCompilerGLSLBackendCommon::debugPrint(std::ostream& _out, const result_t::instr_streams_t& _streams, const result_t& _res, const SContext* _ctx) const
{
	using namespace instr_stream;

	_out << "####### remainder_and_pdf stream\n";
	for (uint32_t i = 0u; i < _streams.rem_and_pdf.count; ++i)
	{
		using namespace remainder_and_pdf;

		const instr_t instr = _res.instructions[_streams.rem_and_pdf.first+i];
		const E_OPCODE op = getOpcode(instr);
		debugPrintInstr(_out, instr, _res, _ctx);
		auto regs = getRegisters(instr);
		auto rcnt = getNumberOfSrcRegsForOpcode(op);
		_out << "Registers:\n";
		if (op!=OP_BUMPMAP && op!=OP_SET_GEOM_NORMAL)
			_out << "DST  = " << regs.x << "\n";
		if (rcnt>0u || op==OP_BUMPMAP)
			_out << "SRC1 = " << regs.y << "\n";
		if (rcnt>1u)
			_out << "SRC2 = " << regs.z << "\n";
	}
	_out << "####### gen_choice stream\n";
	for (uint32_t i = 0u; i < _streams.gen_choice.count; ++i)
	{
		using namespace gen_choice;

		const instr_t instr = _res.instructions[_streams.gen_choice.first + i];
		debugPrintInstr(_out, instr, _res, _ctx);
		if (getOpcode(instr) == OP_BUMPMAP)
		{
			_out << "SRC1 = " << remainder_and_pdf::getRegisters(instr).y << "\n";
		}
		uint32_t rjump = core::bitfieldExtract(instr, gen_choice::INSTR_RIGHT_JUMP_SHIFT, gen_choice::INSTR_RIGHT_JUMP_WIDTH);
		_out << "Right jump " << rjump << "\n";
	}
	_out << "####### tex_prefetch stream\n";
	for (uint32_t i = 0u; i < _streams.tex_prefetch.count; ++i)
	{
		using namespace tex_prefetch;

		const uint32_t flag_shift[BITFIELDS_FETCH_TEX_COUNT]{ BITFIELDS_FETCH_TEX_0_SHIFT, BITFIELDS_FETCH_TEX_1_SHIFT, BITFIELDS_FETCH_TEX_2_SHIFT, BITFIELDS_FETCH_TEX_3_SHIFT };
		const uint32_t regcnt_shift[BITFIELDS_FETCH_TEX_COUNT]{ BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT, BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT, BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT, BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT };
		const uint32_t reg_shift[BITFIELDS_FETCH_TEX_COUNT]{ BITFIELDS_REG_0_SHIFT, BITFIELDS_REG_1_SHIFT, BITFIELDS_REG_2_SHIFT, BITFIELDS_REG_3_SHIFT };

		const instr_t instr = _res.instructions[_streams.tex_prefetch.first + i];

		_out << "### instr " << i << "\n";
		for (uint32_t t = 0u; t < BITFIELDS_FETCH_TEX_COUNT; ++t)
		{
			_out << "tex " << t << ": ";
			if (core::bitfieldExtract(instr, flag_shift[t], 1))
			{
				uint32_t regcnt = core::bitfieldExtract(instr, regcnt_shift[t], BITFIELDS_FETCH_TEX_REG_CNT_WIDTH);
				uint32_t reg = core::bitfieldExtract(instr, reg_shift[t], BITFIELDS_REG_WIDTH);
				_out << "reg=" << reg << ", reg_count=" << regcnt;
			}
			_out << "\n";
		}
	}
	_out << "####### normal_precomp stream\n";
	for (uint32_t i = 0u; i < _streams.norm_precomp.count; ++i)
	{
		using namespace normal_precomp;

		const instr_t instr = _res.instructions[_streams.norm_precomp.first + i];
		const uint32_t reg = core::bitfieldExtract(instr, BITFIELDS_REG_DST_SHIFT, BITFIELDS_REG_WIDTH);
		const uint32_t bsdf_ix = core::bitfieldExtract(instr, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		const SBSDFUnion& data = _res.bsdfData[bsdf_ix];

		_out << "### instr " << i << "\n";
		_out << reg << " <- perturbNormal( reg " << data.bumpmap.derivmap.prefetch_reg << " )\n";
	}
}

void material_compiler::CMaterialCompilerGLSLBackendCommon::debugPrintInstr(std::ostream& _out, instr_stream::instr_t instr, const result_t& _res, const SContext* _ctx) const
{
	auto texDataStr = [](const STextureData& td) {
		return "{ " + std::to_string(reinterpret_cast<const uint64_t&>(td.vtid)) + ", " + std::to_string(reinterpret_cast<const float&>(td.scale)) + " }";
	};
	auto paramVal3OrRegStr = [](const STextureOrConstant& tc, bool tex) {
		if (tex)
			return std::to_string(tc.tex.prefetch_reg);
		else {
			const float* val = reinterpret_cast<const float*>(tc.constant);
			return "{ " + std::to_string(val[0]) + ", " + std::to_string(val[1]) + ", " + std::to_string(val[2]) + " }";
		}
	};
	auto paramVal1OrRegStr = [](const STextureOrConstant& tc, bool tex) {
		if (tex)
			return std::to_string(tc.tex.prefetch_reg);
		else {
			const float* val = reinterpret_cast<const float*>(tc.constant);
			return std::to_string(val[0]);
		}
	};

	constexpr const char* ndf_names[4]{
		"BECKMANN",
		"GGX",
		"PHONG",
		"AS"
	};

	const auto op = getOpcode(instr);

	auto ndfAndAnisoAlphaTexPrint = [&_out,&paramVal1OrRegStr,&ndf_names](instr_t instr, const SBSDFUnion& data) {
		const E_NDF ndf = static_cast<E_NDF>(core::bitfieldExtract(instr, BITFIELDS_SHIFT_NDF, BITFIELDS_WIDTH_NDF));
		_out << "NDF = " << ndf_names[ndf] << "\n";

		bool au = core::bitfieldExtract(instr, BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		_out << "Alpha_u tex " << au << "\n";
		_out << "Alpha_u val/reg " << paramVal1OrRegStr(data.param[0], au) << "\n";

		bool av = core::bitfieldExtract(instr, BITFIELDS_SHIFT_ALPHA_V_TEX, 1);
		_out << "Alpha_v tex " << av << "\n";
		_out << "Alpha_v val/reg " << paramVal1OrRegStr(data.param[1], av) << "\n";
	};

	const uint32_t bsdf_ix = core::bitfieldExtract(instr, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
	SBSDFUnion data;
	if (bsdf_ix < _res.bsdfData.size())
		data = _res.bsdfData[bsdf_ix];

	const bool masked = core::bitfieldExtract(instr, BITFIELDS_SHIFT_MASKFLAG, 1);
	const bool twosided = core::bitfieldExtract(instr, BITFIELDS_SHIFT_TWOSIDED, 1);

	_out << "### " << OPCODE_NAMES[op] << " " << (masked ? "M " : "") << (twosided ? "TS " : "") << (reinterpret_cast<uint32_t*>(&instr)[0]) << "\n";
	_out << "BSDF data index = " << bsdf_ix << "\n";
	switch (op)
	{
	case OP_DIFFUSE:
	{
		bool alpha = core::bitfieldExtract(instr, BITFIELDS_SHIFT_ALPHA_U_TEX, 1);
		_out << "Alpha tex " << alpha << "\n";
		_out << "Alpha val/reg " << paramVal1OrRegStr(data.diffuse.alpha, alpha) << "\n";
		bool refl = core::bitfieldExtract(instr, BITFIELDS_SHIFT_REFL_TEX, 1);
		_out << "Refl tex " << refl << "\n";
		_out << "Refl val/reg " << paramVal3OrRegStr(data.diffuse.reflectance, refl) << "\n";
	}
	break;
	case OP_DIELECTRIC: _IRR_FALLTHROUGH;
	case OP_THINDIELECTRIC:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);

		const auto& data = _res.bsdfData[bsdf_ix];
		const float eta = data.dielectric.eta;

		_out << "Eta: " << eta << "\n";
	}
		break;
	case OP_CONDUCTOR:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);

		const auto& data = _res.bsdfData[bsdf_ix];
		const auto eta = core::rgb19e7_to_rgb32f(data.conductor.eta[0]);
		const auto etak = core::rgb19e7_to_rgb32f(data.conductor.eta[1]);

		_out << "Eta:  { " << eta.x << ", " << eta.y << ", " << eta.z << " }\n";
		_out << "EtaK: { " << etak.x << ", " << etak.y << ", " << etak.z << " }\n";
	}
		break;
	case OP_PLASTIC:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);
		bool refl = core::bitfieldExtract(instr, BITFIELDS_SHIFT_REFL_TEX, 1);
		_out << "Refl tex " << refl << "\n";
		_out << "Refl val/reg " << paramVal3OrRegStr(data.diffuse.reflectance, refl) << "\n";

		const auto& data = _res.bsdfData[bsdf_ix];
		const float eta = data.plastic.eta;
		_out << "Eta: " << eta << "\n";
	}
	break;
	case OP_COATING:
	{
		ndfAndAnisoAlphaTexPrint(instr, data);
		bool sigma = core::bitfieldExtract(instr, BITFIELDS_SHIFT_SIGMA_A_TEX, 1);
		_out << "SigmaA tex " << sigma << "\n";
		_out << "SigmaA val/reg " << paramVal3OrRegStr(data.coating.sigmaA, sigma) << "\n";

		const auto& data = _res.bsdfData[bsdf_ix];
		const uint32_t thickness_eta = data.coating.thickness_eta;
		const float thickness = core::Float16Compressor::decompress(thickness_eta&0xffffu);
		const float eta = core::Float16Compressor::decompress(thickness_eta>>16);
		_out << "Eta: " << eta << "\n";
		_out << "Thickness: " << thickness << "\n";
	}
	break;
	case OP_BLEND:
	{
		bool weight = core::bitfieldExtract(instr, BITFIELDS_SHIFT_WEIGHT_TEX, 1);
		_out << "Weight tex " << weight << "\n";
		_out << "Weight val/reg " << paramVal1OrRegStr(data.blend.weight, weight) << "\n";
	}
	break;
	case OP_DIFFTRANS:
	{
		bool trans = core::bitfieldExtract(instr, BITFIELDS_SHIFT_TRANS_TEX, 1);
		_out << "Trans tex " << trans << "\n";
		_out << "Trans val/reg " << paramVal3OrRegStr(data.difftrans.transmittance, trans) << "\n";
	}
	break;
	case OP_BUMPMAP:
		//_out << "Bump reg " << data.bumpmap.bumpmap.prefetch_reg << "\n";
	break;
	default: break;
	}
}

}}

