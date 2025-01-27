// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/utils/IMeshPacker.h>

namespace nbl
{
namespace asset
{

class IMeshPackerV2Base
{
public:
    class SupportedFormatsContainer
    {
    public:
        template <typename MeshBufferIt>
        void insertFormatsFromMeshBufferRange(MeshBufferIt mbBegin, MeshBufferIt mbEnd)
        {
            for (auto it = mbBegin; it != mbEnd; it++)
            {
                const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
                for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
                {
                    if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
                        continue;

                    formats.insert(static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format));
                }
            }
        }

        inline void insert(E_FORMAT format)
        {
            formats.insert(format);
        }

        inline const core::unordered_set<E_FORMAT>& getFormats() const { return formats; }

    private:
        core::unordered_set<E_FORMAT> formats;
    };

public:
    enum E_UTB_ARRAY_TYPE : uint8_t
    {
        EUAT_FLOAT,
        EUAT_INT,
        EUAT_UINT,
        EUAT_UNKNOWN
    };
    struct VirtualAttribConfig
    {
        VirtualAttribConfig() = default;

        VirtualAttribConfig(const SupportedFormatsContainer& formats)
        {
            const auto& formatsSet = formats.getFormats();
            for (auto it = formatsSet.begin(); it != formatsSet.end(); it++)
                insertAttribFormat(*it);
        }

        VirtualAttribConfig(const VirtualAttribConfig& other)
        {
            std::copy_n(other.utbs, EUAT_UNKNOWN, utbs);

            isUintBufferUsed = other.isUintBufferUsed;
            isUvec2BufferUsed = other.isUvec2BufferUsed;
            isUvec3BufferUsed = other.isUvec3BufferUsed;
            isUvec4BufferUsed = other.isUvec4BufferUsed;
        }

        VirtualAttribConfig(VirtualAttribConfig&& other)
        {
            for (auto i = 0u; i < EUAT_UNKNOWN; i++)
                utbs[i] = std::move(other.utbs[i]);

            isUintBufferUsed = other.isUintBufferUsed;
            isUvec2BufferUsed = other.isUvec2BufferUsed;
            isUvec3BufferUsed = other.isUvec3BufferUsed;
            isUvec4BufferUsed = other.isUvec4BufferUsed;

            //other.utbs->clear();
            other.isUintBufferUsed = false;
            other.isUvec2BufferUsed = false;
            other.isUvec3BufferUsed = false;
            other.isUvec4BufferUsed = false;
        }

        core::unordered_map<E_FORMAT,uint8_t> utbs[EUAT_UNKNOWN];
        bool isUintBufferUsed = false;
        bool isUvec2BufferUsed = false;
        bool isUvec3BufferUsed = false;
        bool isUvec4BufferUsed = false;
    
        VirtualAttribConfig& operator=(const VirtualAttribConfig& other)
        {
            std::copy_n(other.utbs,EUAT_UNKNOWN,utbs);

            isUintBufferUsed = other.isUintBufferUsed;
            isUvec2BufferUsed = other.isUvec2BufferUsed;
            isUvec3BufferUsed = other.isUvec3BufferUsed;
            isUvec4BufferUsed = other.isUvec4BufferUsed;

            return *this;
        }
    
        VirtualAttribConfig& operator=(VirtualAttribConfig&& other)
        {
            for (auto i=0u; i<EUAT_UNKNOWN; i++)
                utbs[i] = std::move(other.utbs[i]);
        
            isUintBufferUsed = other.isUintBufferUsed;
            isUvec2BufferUsed = other.isUvec2BufferUsed;
            isUvec3BufferUsed = other.isUvec3BufferUsed;
            isUvec4BufferUsed = other.isUvec4BufferUsed;
        
            other.isUintBufferUsed = false;
            other.isUvec2BufferUsed = false;
            other.isUvec3BufferUsed = false;
            other.isUvec4BufferUsed = false;
        
            return *this;
        }
    
        inline bool insertAttribFormat(E_FORMAT format)
        {
            auto& utb = utbs[getUTBArrayTypeFromFormat(format)];
            auto lookupResult = utb.find(format);
            if (lookupResult!=utb.end())
                return true;
    
            utb.insert(std::make_pair(format,utb.size()));
    
            const uint32_t attribSize = asset::getTexelOrBlockBytesize(format);
            constexpr uint32_t uvec4Size = 4u * 4u;
            constexpr uint32_t uvec3Size = 4u * 3u;
            constexpr uint32_t uvec2Size = 4u * 2u;
            constexpr uint32_t uintSize  = 4u;
            switch (attribSize)
            {
                case uvec4Size:
                    isUvec4BufferUsed = true;
                    break;
                case uvec3Size:
                    isUvec3BufferUsed = true;
                    break;
                case uvec2Size:
                    isUvec2BufferUsed = true;
                    break;
                case uintSize:
                    isUintBufferUsed = true;
                    break;
                default:
                    assert(false);
                    return true; //tmp
            }
    
            return true;
        }
        
        inline bool isFormatSupported(E_FORMAT format) const
        {
            auto& utb = utbs[getUTBArrayTypeFromFormat(format)];
            auto lookupResult = utb.find(format);
            if (lookupResult != utb.end())
                return true;

            return false;
        }

        static inline E_UTB_ARRAY_TYPE getUTBArrayTypeFromFormat(E_FORMAT format)
        {
            switch (format)
            {
                 //float formats
                case EF_R8_UNORM:
                case EF_R8_SNORM:
                case EF_R8_USCALED:
                case EF_R8_SSCALED:
                case EF_R8G8_UNORM:
                case EF_R8G8_SNORM:
                case EF_R8G8_USCALED:
                case EF_R8G8_SSCALED:
                case EF_R8G8B8_UNORM:
                case EF_R8G8B8_SNORM:
                case EF_R8G8B8_USCALED:
                case EF_R8G8B8_SSCALED:
                case EF_R8G8B8A8_UNORM:
                case EF_R8G8B8A8_SNORM:
                case EF_R8G8B8A8_USCALED:
                case EF_R8G8B8A8_SSCALED:
                case EF_R16_UNORM:
                case EF_R16_SNORM:
                case EF_R16_USCALED:
                case EF_R16_SSCALED:
                case EF_R16_SFLOAT:
                case EF_R16G16_UNORM:
                case EF_R16G16_SNORM:
                case EF_R16G16_USCALED:
                case EF_R16G16_SSCALED:
                case EF_R16G16_SFLOAT:
                case EF_R16G16B16_UNORM:
                case EF_R16G16B16_SNORM:
                case EF_R16G16B16_USCALED:
                case EF_R16G16B16_SSCALED:
                case EF_R16G16B16_SFLOAT:
                case EF_R16G16B16A16_UNORM:
                case EF_R16G16B16A16_SNORM:
                case EF_R16G16B16A16_USCALED:
                case EF_R16G16B16A16_SSCALED:
                case EF_R16G16B16A16_SFLOAT:
                case EF_R32_SFLOAT:
                case EF_R32G32_SFLOAT:
                case EF_R32G32B32_SFLOAT:
                case EF_R32G32B32A32_SFLOAT:
                case EF_B10G11R11_UFLOAT_PACK32:
                case EF_A2B10G10R10_UNORM_PACK32:
                case EF_A8B8G8R8_UNORM_PACK32:
                case EF_A8B8G8R8_SNORM_PACK32:
                case EF_A8B8G8R8_USCALED_PACK32:
                case EF_A8B8G8R8_SSCALED_PACK32:
                    return E_UTB_ARRAY_TYPE::EUAT_FLOAT;
    
                 //int formats
                case EF_R8_SINT:
                case EF_R8G8_SINT:
                case EF_R8G8B8_SINT:
                case EF_R8G8B8A8_SINT:
                case EF_R16_SINT:
                case EF_R16G16_SINT:
                case EF_R16G16B16_SINT:
                case EF_R16G16B16A16_SINT:
                case EF_R32_SINT:
                case EF_R32G32_SINT:
                case EF_R32G32B32_SINT:
                case EF_R32G32B32A32_SINT:
                case EF_A8B8G8R8_SINT_PACK32:
                    return E_UTB_ARRAY_TYPE::EUAT_INT;
    
                 //uint formats
                case EF_R8_UINT:
                case EF_R8G8_UINT:
                case EF_R8G8B8_UINT:
                case EF_R8G8B8A8_UINT:
                case EF_R16_UINT:
                case EF_R16G16_UINT:
                case EF_R16G16B16_UINT:
                case EF_R16G16B16A16_UINT:
                case EF_R32_UINT:
                case EF_R32G32_UINT:
                case EF_R32G32B32_UINT:
                case EF_R32G32B32A32_UINT:
                case EF_A2B10G10R10_SNORM_PACK32:
                case EF_A2B10G10R10_UINT_PACK32:
                case EF_A8B8G8R8_UINT_PACK32:
                    return E_UTB_ARRAY_TYPE::EUAT_UINT;
    
                default:
                    return E_UTB_ARRAY_TYPE::EUAT_UNKNOWN;
            }
        }
    };
    
    const VirtualAttribConfig& getVirtualAttribConfig() const { return m_virtualAttribConfig; }
    
protected:
    explicit IMeshPackerV2Base(const SupportedFormatsContainer& formats) : m_virtualAttribConfig(formats) {}
    explicit IMeshPackerV2Base(const VirtualAttribConfig& cfg) : m_virtualAttribConfig(cfg) {}
    
    const VirtualAttribConfig m_virtualAttribConfig;
};

#if 0 // REWRITE
template <class BufferType, class DescriptorSetType, class MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPackerV2 : public IMeshPacker<MeshBufferType,MDIStructType>, public IMeshPackerV2Base
{
    static_assert(std::is_base_of<IBuffer,BufferType>::value);

    using AllocationParams = IMeshPackerBase::AllocationParamsCommon;

    using DescriptorSetLayoutType = typename DescriptorSetType::layout_t;
public:
	using base_t = IMeshPacker<MeshBufferType,MDIStructType>;
    struct AttribAllocParams
    {
        size_t offset = base_t::INVALID_ADDRESS;
        size_t size = 0ull;
    };
    
    //TODO: REDESIGN
    //mdi allocation offset and index allocation offset should be shared
    struct ReservedAllocationMeshBuffers : ReservedAllocationMeshBuffersBase
    {
        AttribAllocParams attribAllocParams[SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];
    };
    struct VirtualAttribute
    {
            VirtualAttribute() : va(0u) {};

            VirtualAttribute(uint16_t arrayElement, uint32_t offset)
                :va(0u)
            {
                assert((offset & 0xF0000000u) == 0u); 

                va |= static_cast<uint32_t>(arrayElement) << 28u;
                va |= offset;
            }
        
            inline uint32_t getArrayElement() const { return core::bitfieldExtract(va,28,4); }
            inline void setArrayElement(uint16_t arrayElement) { va = core::bitfieldInsert<uint32_t>(va,arrayElement,28,4); }

            inline uint32_t getOffset() const { return core::bitfieldExtract(va,0,28); }
            inline void setOffset(uint32_t offset) { va = core::bitfieldInsert<uint32_t>(va,offset,0,28); }

        private:
            uint32_t va;
    };

    struct CombinedDataOffsetTable
    {
        VirtualAttribute attribInfo[SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];
    };

    struct PackerDataStore : base_t::template PackerDataStoreCommon<BufferType>
    {
        core::smart_refctd_ptr<BufferType> vertexBuffer;
        core::smart_refctd_ptr<BufferType> indexBuffer;
    };

    inline uint32_t getFloatBufferBindingsCnt() const { return m_virtualAttribConfig.utbs[EUAT_FLOAT].size();  }
    inline uint32_t getIntBufferBindingsCnt() const { return m_virtualAttribConfig.utbs[EUAT_INT].size(); }
    // not including the UTB for the index buffer
    inline uint32_t getUintBufferBindingsCnt() const { return m_virtualAttribConfig.utbs[EUAT_UINT].size(); }

    // the following cannot be called before 'instantiateDataStorage'
    struct DSLayoutParamsUTB
    {
        uint32_t usamplersBinding = 0u;
        uint32_t fsamplersBinding = 1u;
        uint32_t isamplersBinding = 2u;
    };
    std::string getGLSLForUTB(uint32_t descriptorSet = 0u, const DSLayoutParamsUTB& params = {})
    {
        std::string result = "#define _NBL_VG_DESCRIPTOR_SET " + std::to_string(descriptorSet) + '\n';

        result += "#define _NBL_VG_UINT_BUFFERS\n";
        result += "#define _NBL_VG_UINT_BUFFERS_BINDING " + std::to_string(params.usamplersBinding) + '\n';
        result += "#define _NBL_VG_UINT_BUFFERS_COUNT " + std::to_string(1u+getUintBufferBindingsCnt()) + '\n';
        if (getFloatBufferBindingsCnt())
        {
            result += "#define _NBL_VG_FLOAT_BUFFERS\n";
            result += "#define _NBL_VG_FLOAT_BUFFERS_BINDING " + std::to_string(params.fsamplersBinding) + '\n';
            result += "#define _NBL_VG_FLOAT_BUFFERS_COUNT " + std::to_string(getFloatBufferBindingsCnt()) + '\n';
        }
        if (getIntBufferBindingsCnt())
        {
            result += "#define _NBL_VG_INT_BUFFERS\n";
            result += "#define _NBL_VG_INT_BUFFERS_BINDING " + std::to_string(params.isamplersBinding) + '\n';
            result += "#define _NBL_VG_INT_BUFFERS_COUNT " + std::to_string(getFloatBufferBindingsCnt()) + '\n';
        }

        return result;
    }

    inline uint32_t getDSlayoutBindingsForUTB(typename DescriptorSetLayoutType::SBinding* outBindings, const DSLayoutParamsUTB& params = {}) const
    {
        const uint32_t bindingCount = 1u + // for the always present uint index buffer
            (getFloatBufferBindingsCnt() ? 1u : 0u) +
            (getIntBufferBindingsCnt() ? 1u : 0u);

        if (outBindings)
        {
            auto* bnd = outBindings;
            auto fillBinding = [&bnd](uint32_t binding, uint32_t count)
            {
                bnd->binding = binding;
                bnd->count = count;
                bnd->stageFlags = asset::ISpecializedShader::ESS_ALL;
                bnd->type = asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
                bnd->samplers = nullptr;
                bnd++;
            };

            fillBinding(params.usamplersBinding,getUintBufferBindingsCnt()+1u);
            if (getFloatBufferBindingsCnt())
                fillBinding(params.fsamplersBinding,getFloatBufferBindingsCnt());
            if (getIntBufferBindingsCnt())
                fillBinding(params.isamplersBinding,getIntBufferBindingsCnt());
        }
        return bindingCount;
    }

    inline std::pair<uint32_t,uint32_t> getDescriptorSetWritesForUTB(
        typename DescriptorSetType::SWriteDescriptorSet* outWrites, typename DescriptorSetType::SDescriptorInfo* outInfo, DescriptorSetType* dstSet,
        std::function<core::smart_refctd_ptr<IDescriptor>(core::smart_refctd_ptr<BufferType>&&,E_FORMAT)> createBufferView, const DSLayoutParamsUTB& params = {}
    ) const
    {
        const uint32_t writeCount = getDSlayoutBindingsForUTB(nullptr);
        const uint32_t infoCount = 1u + // for the index buffer
            getFloatBufferBindingsCnt() + 
            getIntBufferBindingsCnt() + 
            getUintBufferBindingsCnt();
        if (!outWrites || !outInfo)
            return std::make_pair(writeCount, infoCount);

        auto* info = outInfo;
        auto fillInfoStruct = [&](E_UTB_ARRAY_TYPE utbArrayType)
        {
            for (auto virtualAttribData : m_virtualAttribConfig.utbs[utbArrayType])
            {
                E_FORMAT format = virtualAttribData.first;
                switch (format)
                {
                    case EF_A2B10G10R10_SNORM_PACK32:
                        format = EF_R32_UINT;
                        break;
                    default:
                        break;
                }
                info[virtualAttribData.second].desc = createBufferView(core::smart_refctd_ptr(m_packerDataStore.vertexBuffer),format);
                info[virtualAttribData.second].buffer.offset = 0u;
                info[virtualAttribData.second].buffer.size = m_packerDataStore.vertexBuffer->getSize();
            }
            info += m_virtualAttribConfig.utbs[utbArrayType].size();
        };

        auto* write = outWrites;
        auto writeBinding = [&write,dstSet,&info](uint32_t binding, uint32_t count)
        {
            write->binding = binding;
            write->arrayElement = 0u;
            write->count = count;
            write->descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
            write->dstSet = dstSet;
            write->info = info;
            write++;
        };

        writeBinding(params.usamplersBinding, 1u + getUintBufferBindingsCnt());
        if (getUintBufferBindingsCnt())
            fillInfoStruct(E_UTB_ARRAY_TYPE::EUAT_UINT);
        info->desc = createBufferView(core::smart_refctd_ptr(m_packerDataStore.indexBuffer),EF_R16_UINT);
        info->buffer.offset = 0u;
        info->buffer.size = m_packerDataStore.indexBuffer->getSize();
        info++;
        if (getFloatBufferBindingsCnt())
        {
            writeBinding(params.fsamplersBinding, getFloatBufferBindingsCnt());
            fillInfoStruct(E_UTB_ARRAY_TYPE::EUAT_FLOAT);
        }
        if (getIntBufferBindingsCnt())
        {
            writeBinding(params.isamplersBinding, getIntBufferBindingsCnt());
            fillInfoStruct(E_UTB_ARRAY_TYPE::EUAT_INT);
        }

        return std::make_pair(writeCount, infoCount);
    }

    // the following cannot be called before 'instantiateDataStorage'
    struct DSLayoutParamsSSBO
    {
        uint32_t uintBufferBinding = 0u;
        uint32_t uvec2BufferBinding = 1u;
        uint32_t uvec3BufferBinding = 2u;
        uint32_t uvec4BufferBinding = 3u;
        uint32_t indexBufferBinding = 4u;
    };
    inline std::string getGLSLForSSBO(uint32_t descriptorSet = 0u, const DSLayoutParamsSSBO& params = {})
    {
        std::string result = "#define _NBL_VG_USE_SSBO\n";
        result += "#define _NBL_VG_SSBO_DESCRIPTOR_SET " + std::to_string(descriptorSet) + '\n';

        if (m_virtualAttribConfig.isUintBufferUsed)
        {
            result += "#define _NBL_VG_USE_SSBO_UINT\n";
            result += "#define _NBL_VG_SSBO_UINT_BINDING " + std::to_string(params.uintBufferBinding) + '\n';
        }
        if (m_virtualAttribConfig.isUvec2BufferUsed)
        {
            result += "#define _NBL_VG_USE_SSBO_UVEC2\n";
            result += "#define _NBL_VG_SSBO_UVEC2_BINDING " + std::to_string(params.uvec2BufferBinding) + '\n';
        }
        if (m_virtualAttribConfig.isUvec3BufferUsed)
        {
            result += "#define _NBL_VG_USE_SSBO_UVEC3\n";
            result += "#define _NBL_VG_SSBO_UVEC3_BINDING " + std::to_string(params.uvec3BufferBinding) + '\n';
        }
        if (m_virtualAttribConfig.isUvec4BufferUsed)
        {
            result += "#define _NBL_VG_USE_SSBO_UVEC4\n";
            result += "#define _NBL_VG_SSBO_UVEC4_BINDING " + std::to_string(params.uvec4BufferBinding) + '\n';
        }

        result += "#define _NBL_VG_USE_SSBO_INDEX\n";
        result += "#define _NBL_VG_SSBO_INDEX_BINDING " + std::to_string(params.indexBufferBinding) + '\n';
        return result;
    }

    inline uint32_t getDSlayoutBindingsForSSBO(typename DescriptorSetLayoutType::SBinding* outBindings, const DSLayoutParamsSSBO& params = {}) const
    {
        const uint32_t bindingCount = 1 + // for the index buffer
            m_virtualAttribConfig.isUintBufferUsed +
            m_virtualAttribConfig.isUvec2BufferUsed +
            m_virtualAttribConfig.isUvec3BufferUsed +
            m_virtualAttribConfig.isUvec4BufferUsed;

        if (outBindings)
        {
            auto* bnd = outBindings;
            auto fillBinding = [&bnd](uint32_t binding)
            {
                bnd->binding = binding;
                bnd->count = 1u;
                bnd->stageFlags = asset::ISpecializedShader::ESS_ALL;
                bnd->type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
                bnd->samplers = nullptr;
                bnd++;
            };

            if (m_virtualAttribConfig.isUintBufferUsed)
                fillBinding(params.uintBufferBinding);
            if (m_virtualAttribConfig.isUvec2BufferUsed)
                fillBinding(params.uvec2BufferBinding);
            if (m_virtualAttribConfig.isUvec3BufferUsed)
                fillBinding(params.uvec3BufferBinding);
            if (m_virtualAttribConfig.isUvec4BufferUsed)
                fillBinding(params.uvec4BufferBinding);
            fillBinding(params.indexBufferBinding);
        }
        return bindingCount;
    }

    // info count is always 2
    inline uint32_t getDescriptorSetWritesForSSBO(
        typename DescriptorSetType::SWriteDescriptorSet* outWrites, typename DescriptorSetType::SDescriptorInfo* outInfo, DescriptorSetType* dstSet,
        const DSLayoutParamsSSBO& params = {}
    ) const
    {
        const uint32_t writeCount = getDSlayoutBindingsForSSBO(nullptr);
        if (!outWrites || !outInfo)
            return writeCount;

        auto* write = outWrites;
        auto info = outInfo;
        auto fillWriteStruct = [&](uint32_t binding)
        {
            write->binding = binding;
            write->arrayElement = 0u;
            write->count = 1u;
            write->descriptorType = IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
            write->dstSet = dstSet;
            write->info = info;
            write++;
        };
        if (m_virtualAttribConfig.isUintBufferUsed)
            fillWriteStruct(params.uintBufferBinding);
        if (m_virtualAttribConfig.isUvec2BufferUsed)
            fillWriteStruct(params.uvec2BufferBinding);
        if (m_virtualAttribConfig.isUvec3BufferUsed)
            fillWriteStruct(params.uvec3BufferBinding);
        if (m_virtualAttribConfig.isUvec4BufferUsed)
            fillWriteStruct(params.uvec4BufferBinding);
        info->desc = m_packerDataStore.vertexBuffer;
        info->buffer.offset = 0u;
        info->buffer.size = m_packerDataStore.vertexBuffer->getSize();
        info++;
        
        fillWriteStruct(params.indexBufferBinding);
        info->desc = m_packerDataStore.indexBuffer;
        info->buffer.offset = 0u;
        info->buffer.size = m_packerDataStore.indexBuffer->getSize();
        info++;

        return writeCount;
    }

	template <typename MeshBufferIterator>
	bool alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    void free(const ReservedAllocationMeshBuffers* rambIn, uint32_t meshBuffersToFreeCnt)
    {
        for (uint32_t i = 0u; i < meshBuffersToFreeCnt; i++)
        {
            const ReservedAllocationMeshBuffers& ramb = rambIn[i];

            const ReservedAllocationMeshBuffers* const ramb = rambIn + i;

            if (ramb->indexAllocationOffset != base_t::INVALID_ADDRESS)
                base_t::m_idxBuffAlctr.free_addr(ramb->indexAllocationOffset, ramb->indexAllocationReservedCnt);
            
            if (ramb->mdiAllocationOffset != base_t::INVALID_ADDRESS)
                base_t::m_MDIDataAlctr.free_addr(ramb->mdiAllocationOffset, ramb->mdiAllocationReservedCnt);
            
            for (uint32_t j = 0; j < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; j++)
            {
                const AttribAllocParams& attrAllocParams = ramb->attribAllocParams[j];
                if (attrAllocParams.offset != base_t::INVALID_ADDRESS)
                    base_t::m_vtxBuffAlctr.free_addr(attrAllocParams.offset, attrAllocParams.size);
            }
        }
    }

    inline const PackerDataStore& getPackerDataStore() const { return m_packerDataStore; }

    const core::GeneralpurposeAddressAllocator<uint32_t>& getMDIAllocator() const { return base_t::m_MDIDataAlctr; }
    const core::GeneralpurposeAddressAllocator<uint32_t>& getIndexAllocator() const { return base_t::m_idxBuffAlctr; }
    const core::GeneralpurposeAddressAllocator<uint32_t>& getVertexAllocator() const { return base_t::m_vtxBuffAlctr; }

protected:
	IMeshPackerV2(const AllocationParams& allocParams, const SupportedFormatsContainer& formats, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
		: base_t(minTriangleCountPerMDIData, maxTriangleCountPerMDIData), 
        IMeshPackerV2Base(formats)
	{
        base_t::initializeCommonAllocators(allocParams);
    };
    template<class OtherBufferType, class OtherDescriptorSetType, class OtherMeshBufferType>
	explicit IMeshPackerV2(const IMeshPackerV2<OtherBufferType,OtherDescriptorSetType,OtherMeshBufferType,MDIStructType>* otherMp)
		: base_t(otherMp->getMinTriangleCountPerMDI(),otherMp->getMaxTriangleCountPerMDI()),
        IMeshPackerV2Base(otherMp->getVirtualAttribConfig())
	{
        base_t::initializeCommonAllocators(
            otherMp->getMDIAllocator(),
            otherMp->getIndexAllocator(),
            otherMp->getVertexAllocator()
        );
    };

    template <typename MeshBufferIterator>
    void freeAllocatedAddressesOnAllocFail(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
    {
        size_t i = 0ull;
        for (auto it = mbBegin; it != mbEnd; it++)
        {
            ReservedAllocationMeshBuffers& ramb = *(rambOut + i);

            if (ramb.indexAllocationOffset == base_t::INVALID_ADDRESS)
                return;

            base_t::m_idxBuffAlctr.free_addr(ramb.indexAllocationOffset, ramb.indexAllocationReservedCnt);

            const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
            for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
            {
                if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
                    continue;

                if (ramb.attribAllocParams[location].offset == base_t::INVALID_ADDRESS)
                    return;

                base_t::m_vtxBuffAlctr.free_addr(ramb.attribAllocParams[location].offset, ramb.attribAllocParams[location].size);
            }

            if (ramb.mdiAllocationOffset == base_t::INVALID_ADDRESS)
                return;

            base_t::m_MDIDataAlctr.free_addr(ramb.mdiAllocationOffset, ramb.mdiAllocationReservedCnt);

            i++;
        }
    }

    PackerDataStore m_packerDataStore;

};

template <class BufferType, class DescriptorSetType, class MeshBufferType, typename MDIStructType>
template <typename MeshBufferIterator>
bool IMeshPackerV2<BufferType,DescriptorSetType,MeshBufferType,MDIStructType>::alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    size_t i = 0ull;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambOut + i);
        const size_t idxCnt = base_t::calcIdxCntAfterConversionToTriangleList(*it);
        const size_t maxVtxCnt = base_t::calcVertexCountBoundWithBatchDuplication(*it);
        const uint32_t insCnt = (*it)->getInstanceCount();

        //allocate indices
        ramb.indexAllocationOffset = base_t::m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
        if (ramb.indexAllocationOffset == base_t::INVALID_ADDRESS)
        {
            freeAllocatedAddressesOnAllocFail(rambOut, mbBegin, mbEnd);
            return false;
        }
        ramb.indexAllocationReservedCnt = idxCnt;

        //allocate vertices
        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
        {
            if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
                continue;

            const E_FORMAT attribFormat = static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format);

            if (!m_virtualAttribConfig.isFormatSupported(attribFormat))
                return false;

            const uint32_t attribSize = asset::getTexelOrBlockBytesize(attribFormat);
            const uint32_t binding = mbVtxInputParams.attributes[location].binding;
            const E_VERTEX_INPUT_RATE inputRate = mbVtxInputParams.bindings[binding].inputRate;

            if (inputRate == EVIR_PER_VERTEX)
            {
                const uint32_t allocByteSize = maxVtxCnt * attribSize;
                ramb.attribAllocParams[location].offset = base_t::m_vtxBuffAlctr.alloc_addr(allocByteSize, attribSize);
                ramb.attribAllocParams[location].size = allocByteSize;

                if(ramb.attribAllocParams[location].offset == base_t::INVALID_ADDRESS)
                {
                    freeAllocatedAddressesOnAllocFail(rambOut, mbBegin, mbEnd);
                    return false;
                }
            }
            else if (inputRate == EVIR_PER_INSTANCE)
            {
                const uint32_t allocByteSize = insCnt * attribSize;
                ramb.attribAllocParams[location].offset = base_t::m_vtxBuffAlctr.alloc_addr(allocByteSize, attribSize);
                ramb.attribAllocParams[location].size = allocByteSize;

                if (ramb.attribAllocParams[location].offset == base_t::INVALID_ADDRESS)
                {
                    freeAllocatedAddressesOnAllocFail(rambOut, mbBegin, mbEnd);
                    return false;
                }
            }

            if (ramb.attribAllocParams[location].offset == base_t::INVALID_ADDRESS)
                return false;
        }

        //allocate MDI structs
        const uint32_t minIdxCntPerPatch = base_t::m_minTriangleCountPerMDIData * 3;
        size_t possibleMDIStructsNeededCnt = (idxCnt + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

        ramb.mdiAllocationOffset = base_t::m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
        if (ramb.mdiAllocationOffset == base_t::INVALID_ADDRESS)
        {
            freeAllocatedAddressesOnAllocFail(rambOut, mbBegin, mbEnd);
            return false;
        }
        ramb.mdiAllocationReservedCnt = possibleMDIStructsNeededCnt;

        i++;
    }

    return true;
}
#endif
}
}

#endif