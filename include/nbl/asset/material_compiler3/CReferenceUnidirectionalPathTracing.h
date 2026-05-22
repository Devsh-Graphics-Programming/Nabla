// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_H_INCLUDED_

#include "nbl/asset/material_compiler3/IBackend.h"

namespace nbl::asset::material_compiler3
{
    
//template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>
struct OrientedMaterial
{
    uint32_t emitter_id;
    uint32_t prefetch_offset;
    uint32_t prefetch_count;
    uint32_t instr_offset;
    uint32_t rem_pdf_count;
    uint32_t nprecomp_count;
    uint32_t genchoice_count;

    core::blake3_hash_t hash;

    struct stream_t
    {
        uint32_t first;
        uint32_t count;
    };
    stream_t get_rem_and_pdf() const { return { instr_offset, rem_pdf_count }; }
    stream_t get_gen_choice() const { return { instr_offset + rem_pdf_count, genchoice_count }; }
    stream_t get_norm_precomp() const { return { instr_offset + rem_pdf_count + genchoice_count, nprecomp_count }; }
    stream_t get_tex_prefetch() const { return { prefetch_offset, prefetch_count }; }
};

class CReferenceUnidirectionalPathTracing : public IBackend
{
public:
    struct CResult : public IBackend::IResult
    {
        // TODO need all this?
        //instr_stream::traversal_t instructions;
        //instr_stream::tex_prefetch::prefetch_stream_t prefetch_stream;
        //core::vector<instr_stream::SBSDFUnion> bsdfData;
        //core::vector<emitter_t> emitterData;

        //bool noPrefetchStream;
        //bool noNormPrecompStream;
        //bool allIsotropic;
        //bool noBSDF;
        //uint32_t usedRegisterCount;
        //uint32_t globalPrefetchRegCountFlags;
        //uint32_t paramTexPresence[instr_stream::SBSDFUnion::MAX_TEXTURES][2];
        //// always same value and the value
        //std::pair<bool, core::vector3df_SIMD> paramConstants[instr_stream::SBSDFUnion::MAX_TEXTURES];

        //core::unordered_set<instr_stream::E_OPCODE> opcodes;
        //core::unordered_set<instr_stream::E_NDF> NDFs;

        //one element for each input IR root node
        core::vector<const CTrueIR::INode*> materials;

        //has to go after #version and before required user-provided descriptors and functions
        std::string fragmentShaderSource_declarations;
        //has to go after required user-provided descriptors and functions and before the rest of shader (especially entry point function)
        std::string fragmentShaderSource;
    };

    CResult compile(const CTrueIR* ir, const std::span<const CTrueIR::SMaterialHandle> materials);

private:
    std::string getHashAs4UintsString(const CTrueIR::INode* node, const CTrueIR* ir, const std::string& separator = ",") const;

    void getAlbedoHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getNormalHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getTransparencyHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getGenerateHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getQuotientWeightHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getEvalWeightHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);
};

}

#endif
