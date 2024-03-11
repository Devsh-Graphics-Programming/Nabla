// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_CPU_MESH_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_CPU_MESH_BUFFER_H_INCLUDED_

#include "nbl/asset/IMeshBuffer.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/bawformat/blobs/MeshBufferBlob.h"
#include "nbl/asset/bawformat/BlobSerializable.h"
#include "nbl/asset/format/decodePixels.h"
#include "nbl/asset/format/encodePixels.h"

namespace nbl::asset
{

// TODO: This should probably go somewhere else, DEFINITELY SHOULD GO SOMEWHERE ELSE @Crisspl
namespace impl
{
    inline E_FORMAT getCorrespondingIntegerFmt(E_FORMAT _scaledFmt)
    {
        switch (_scaledFmt)
        {
        case EF_R8_USCALED: return EF_R8_UINT;
        case EF_R8_SSCALED: return EF_R8_SINT;
        case EF_R8G8_USCALED: return EF_R8G8_UINT;
        case EF_R8G8_SSCALED: return EF_R8G8_SINT;
        case EF_R8G8B8_USCALED: return EF_R8G8B8_UINT;
        case EF_R8G8B8_SSCALED: return EF_R8G8B8_SINT;
        case EF_B8G8R8_USCALED: return EF_B8G8R8_UINT;
        case EF_B8G8R8_SSCALED: return EF_B8G8R8_SINT;
        case EF_R8G8B8A8_USCALED: return EF_R8G8B8A8_UINT;
        case EF_R8G8B8A8_SSCALED: return EF_R8G8B8A8_SINT;
        case EF_B8G8R8A8_USCALED: return EF_B8G8R8A8_UINT;
        case EF_B8G8R8A8_SSCALED: return EF_B8G8R8A8_SINT;
        case EF_A8B8G8R8_USCALED_PACK32: return EF_A8B8G8R8_UINT_PACK32;
        case EF_A8B8G8R8_SSCALED_PACK32: return EF_A8B8G8R8_SINT_PACK32;
        case EF_A2R10G10B10_USCALED_PACK32: return EF_A2R10G10B10_UINT_PACK32;
        case EF_A2R10G10B10_SSCALED_PACK32: return EF_A2R10G10B10_SINT_PACK32;
        case EF_A2B10G10R10_USCALED_PACK32: return EF_A2B10G10R10_UINT_PACK32;
        case EF_A2B10G10R10_SSCALED_PACK32: return EF_A2B10G10R10_SINT_PACK32;
        case EF_R16_USCALED: return EF_R16_UINT;
        case EF_R16_SSCALED: return EF_R16_SINT;
        case EF_R16G16_USCALED: return EF_R16G16_UINT;
        case EF_R16G16_SSCALED: return EF_R16G16_SINT;
        case EF_R16G16B16_USCALED: return EF_R16G16B16_UINT;
        case EF_R16G16B16_SSCALED: return EF_R16G16B16_SINT;
        case EF_R16G16B16A16_USCALED: return EF_R16G16B16A16_UINT;
        case EF_R16G16B16A16_SSCALED: return EF_R16G16B16A16_SINT;

        default: return EF_UNKNOWN;
        }
    }
}

class ICPUMeshBuffer final : public IMeshBuffer<ICPUBuffer,ICPUDescriptorSet,ICPURenderpassIndependentPipeline>, public BlobSerializable, public IAsset
{
        using base_t = IMeshBuffer<ICPUBuffer,ICPUDescriptorSet,ICPURenderpassIndependentPipeline>;
        // knowing the position attribute ID is important for AABB computations etc.
        uint32_t posAttrId : 5;
        uint32_t normalAttrId : 5;
        // by having one attribute only, we limit the number of bones per vertex to 4
        uint32_t jointIDAttrId : 5;
        uint32_t jointWeightAttrId : 5;

    protected:
        virtual ~ICPUMeshBuffer() = default;

    public:
        //! Default constructor (initializes pipeline, desc set and buffer bindings to nullptr)
        ICPUMeshBuffer() : base_t(nullptr, nullptr, nullptr, SBufferBinding<ICPUBuffer>{})
        {
            posAttrId = 0u;
            normalAttrId = MAX_VERTEX_ATTRIB_COUNT;
            jointIDAttrId = MAX_VERTEX_ATTRIB_COUNT;
            jointWeightAttrId = MAX_VERTEX_ATTRIB_COUNT;
        }
        template<typename... Args>
        ICPUMeshBuffer(Args&&... args) : base_t(std::forward<Args>(args)...)
        {
            posAttrId = 0u;
            normalAttrId = MAX_VERTEX_ATTRIB_COUNT;
            jointIDAttrId = MAX_VERTEX_ATTRIB_COUNT;
            jointWeightAttrId = MAX_VERTEX_ATTRIB_COUNT;
        }
        
        virtual void* serializeToBlob(void* _stackPtr = nullptr, const size_t& _stackSize = 0) const override
        {
#ifdef OLD_SHADERS
            return CorrespondingBlobTypeFor<ICPUMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
#else
            return nullptr;
#endif
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            core::unordered_map<ICPUBuffer*, core::smart_refctd_ptr<ICPUBuffer>> buffers;
            auto cloneBuf = [&buffers,_depth](ICPUBuffer* buf) -> core::smart_refctd_ptr<ICPUBuffer> {
                if (!buf)
                    return nullptr;
                if (!_depth)
                    return core::smart_refctd_ptr<ICPUBuffer>(buf);

                auto found = buffers.find(buf);
                if (found != buffers.end())
                    return found->second;

                auto cp = core::smart_refctd_ptr_static_cast<ICPUBuffer>(buf->clone(_depth-1u));
                buffers.insert({ buf, cp });
                return cp;
            };

            auto cp = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
            clone_common(cp.get());

            cp->boundingBox = boundingBox;

            cp->m_indexBufferBinding.offset = m_indexBufferBinding.offset;
            cp->m_indexBufferBinding.buffer = cloneBuf(m_indexBufferBinding.buffer.get());
            for (uint32_t i = 0u; i < MAX_ATTR_BUF_BINDING_COUNT; ++i)
            {
                cp->m_vertexBufferBindings[i].offset = m_vertexBufferBindings[i].offset;
                cp->m_vertexBufferBindings[i].buffer = cloneBuf(m_vertexBufferBindings[i].buffer.get());
            }

            cp->m_inverseBindPoseBufferBinding.offset = m_inverseBindPoseBufferBinding.offset;
            cp->m_inverseBindPoseBufferBinding.buffer = cloneBuf(m_inverseBindPoseBufferBinding.buffer.get());
            cp->m_jointAABBBufferBinding.offset = m_jointAABBBufferBinding.offset;
            cp->m_jointAABBBufferBinding.buffer = cloneBuf(m_jointAABBBufferBinding.buffer.get()); 
            
            cp->m_descriptorSet = (_depth > 0u && m_descriptorSet) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSet>(m_descriptorSet->clone(_depth - 1u)) : m_descriptorSet;

            memcpy(cp->m_pushConstantsData, m_pushConstantsData, sizeof(m_pushConstantsData));

            cp->m_pipeline = (_depth > 0u && m_pipeline) ? core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(m_pipeline->clone(_depth - 1u)) : m_pipeline;

            cp->indexCount = indexCount;
            cp->instanceCount = instanceCount;
            cp->baseVertex = baseVertex;
            cp->baseInstance = baseInstance;

            cp->posAttrId = posAttrId;
            cp->normalAttrId = normalAttrId;
            cp->jointIDAttrId = jointIDAttrId;
            cp->jointWeightAttrId = jointWeightAttrId;

            cp->jointCount = jointCount;
            cp->maxJointsPerVx = maxJointsPerVx;
            cp->indexType = indexType;

            return cp;
        }

        virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
        {
            convertToDummyObject_common(referenceLevelsBelowToConvert);

            if (referenceLevelsBelowToConvert)
            {
                --referenceLevelsBelowToConvert;
                for (auto i=0u; i<MAX_ATTR_BUF_BINDING_COUNT; i++)
                    if (m_vertexBufferBindings[i].buffer)
                        m_vertexBufferBindings[i].buffer->convertToDummyObject(referenceLevelsBelowToConvert);
                if (m_indexBufferBinding.buffer)
                    m_indexBufferBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
                if (m_inverseBindPoseBufferBinding.buffer)
                    m_inverseBindPoseBufferBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
                if (m_jointAABBBufferBinding.buffer)
                    m_jointAABBBufferBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
                if (m_descriptorSet)
                    m_descriptorSet->convertToDummyObject(referenceLevelsBelowToConvert);
                if (m_pipeline)
                    m_pipeline->convertToDummyObject(referenceLevelsBelowToConvert);
            }
        }

        _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_SUB_MESH;
        inline E_TYPE getAssetType() const override { return AssetType; }

        inline size_t conservativeSizeEstimate() const override { return sizeof(base_t) + sizeof(uint32_t); }
        
        inline bool setVertexBufferBinding(SBufferBinding<ICPUBuffer>&& bufferBinding, uint32_t bindingIndex)
        {
            assert(!isImmutable_debug());
            if(bufferBinding.buffer)
                bufferBinding.buffer->addUsageFlags(IBuffer::EUF_VERTEX_BUFFER_BIT);
            return base_t::setVertexBufferBinding(std::move(bufferBinding), bindingIndex);
        }

        inline void setIndexBufferBinding(SBufferBinding<ICPUBuffer>&& bufferBinding)
        {
            assert(!isImmutable_debug());
            if(bufferBinding.buffer)
                bufferBinding.buffer->addUsageFlags(IBuffer::EUF_INDEX_BUFFER_BIT);
            return base_t::setIndexBufferBinding(std::move(bufferBinding));
        }

        //! You need to set skeleton, bind poses and AABBs all at once
        inline bool setSkin(
            SBufferBinding<ICPUBuffer>&& _inverseBindPoseBufferBinding,
            SBufferBinding<ICPUBuffer>&& _jointAABBBufferBinding,
            const uint32_t _jointCount, const uint32_t _maxJointsPerVx
        ) override
        {
            assert(!isImmutable_debug());
            
            if(_inverseBindPoseBufferBinding.buffer)
                _inverseBindPoseBufferBinding.buffer->addUsageFlags(IBuffer::EUF_STORAGE_BUFFER_BIT);
            if(_jointAABBBufferBinding.buffer)
                _jointAABBBufferBinding.buffer->addUsageFlags(IBuffer::EUF_STORAGE_BUFFER_BIT);

            return base_t::setSkin(std::move(_inverseBindPoseBufferBinding),std::move(_jointAABBBufferBinding),_jointCount,_maxJointsPerVx);
        }

        //!
        inline const ICPUDescriptorSet* getAttachedDescriptorSet() const
        {
            return base_t::getAttachedDescriptorSet();
        }
        inline ICPUDescriptorSet* getAttachedDescriptorSet()
        {
            //assert(!isImmutable_debug()); // TODO? @Crisspl?
            return m_descriptorSet.get();
        }
        inline void setAttachedDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSet>&& descriptorSet)
        {
            assert(!isImmutable_debug());
            base_t::setAttachedDescriptorSet(std::move(descriptorSet));
        }

        //!
        inline const ICPURenderpassIndependentPipeline* getPipeline() const {return base_t::getPipeline();}
        inline ICPURenderpassIndependentPipeline* getPipeline()
        {
            assert(!isImmutable_debug());
            return m_pipeline.get();
        }
        inline void setPipeline(core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>&& pipeline) override final
        {
            assert(!isImmutable_debug());
            base_t::setPipeline(std::move(pipeline));
        }

        //
        inline uint32_t getIndexValue(uint32_t _i) const
        {
            if (!m_indexBufferBinding.buffer)
                return _i;
            switch (indexType)
            {
                case EIT_16BIT:
                    return reinterpret_cast<const uint16_t*>(getIndices())[_i];
                case EIT_32BIT:
                    return reinterpret_cast<const uint32_t*>(getIndices())[_i];
                default:
                    break;
            }
            return _i;
        }

        //! Returns id of position attribute.
        inline uint32_t getPositionAttributeIx() const { return posAttrId; }
        //! Sets id of position atrribute.
        inline void setPositionAttributeIx(const uint32_t attrId)
        {
            assert(!isImmutable_debug());

            if (attrId >= MAX_VERTEX_ATTRIB_COUNT)
            {
    #ifdef _NBL_DEBUG
                //os::Printer::log("MeshBuffer setPositionAttributeIx attribute ID out of range!\n",ELL_ERROR);
    #endif // _NBL_DEBUG
                return;
            }

            posAttrId = attrId;
        }

        //! Returns id of normal attribute.
        inline uint32_t getNormalAttributeIx() const { return normalAttrId; }

        //! Sets id of position atrribute.
        inline void setNormalAttributeIx(const uint32_t attrId)
        {
            assert(!isImmutable_debug());
            normalAttrId = attrId;
        }

        //! Returns id of jointID attribute.
        inline uint32_t getJointIDAttributeIx() const { return jointIDAttrId; }

        //! Sets id of joint atrribute.
        inline void setJointIDAttributeIx(const uint32_t attrId)
        {
            assert(!isImmutable_debug());
            jointIDAttrId = attrId;
        }

        //! Returns id of joint weight attribute.
        inline uint32_t getJointWeightAttributeIx() const { return jointWeightAttrId; }

        //! Sets id of joint's weight atrribute.
        inline void setJointWeightAttributeIx(const uint32_t attrId)
        {
            assert(!isImmutable_debug());
            jointWeightAttrId = attrId;
        }

        //! Deduces max joint influences from the formats used for the joint attributes
        inline uint32_t deduceMaxJointsPerVertex() const
        {
            auto safelyGetAttributeFormatChannelCount = [&](const uint32_t attrId) -> uint32_t
            {
                if (!isAttributeEnabled(attrId))
                    return 0u;
                return getFormatChannelCount(getAttribFormat(attrId));
            };
            return (core::min)(safelyGetAttributeFormatChannelCount(jointIDAttrId),safelyGetAttributeFormatChannelCount(jointWeightAttrId)+1u);
        }

        //! Tells us if the mesh is skinned
        inline bool isSkinned() const override
        {
            if (!base_t::isSkinned())
                return false;
            return deduceMaxJointsPerVertex()!=0u;
        }

        //! Get access to Indices.
        /** \return Pointer to indices array. */
        inline void* getIndices()
        {
            assert(!isImmutable_debug());

            if (!m_indexBufferBinding.buffer)
                return nullptr;

            return reinterpret_cast<uint8_t*>(m_indexBufferBinding.buffer->getPointer()) + m_indexBufferBinding.offset;
        }

        //! Get access to Indices.
        /** We only keep track of a position attribute, as every vertex needs to have at least a position to be displayed on the screen.
        Certain vertices may not have colors, normals, texture coords, etc. but a position is always present.
        \return Pointer to index array. */
        inline const void* getIndices() const
        {
            if (!m_indexBufferBinding.buffer)
                return nullptr;

            return reinterpret_cast<const uint8_t*>(m_indexBufferBinding.buffer->getPointer()) + m_indexBufferBinding.offset;
        }

        //! Accesses given index of mapped position attribute buffer.
        /** @param ix Index number of vertex which is to be returned.
        @returns `ix`th vertex of mapped attribute buffer or (0, 0, 0, 1) vector if an error occured (e.g. no such vertex).
        @see @ref getAttribute()
        */
        virtual core::vectorSIMDf getPosition(size_t ix) const
        {
            core::vectorSIMDf outPos(0.f, 0.f, 0.f, 1.f);
            bool success = getAttribute(outPos, posAttrId, ix);
            #ifdef _NBL_DEBUG
                if (!success)
                {
                    //os::Printer::log("SOME DEBUG MESSAGE!\n",ELL_ERROR);
                }
            #endif // _NBL_DEBUG
            return outPos;
        }

        //! Accesses data of buffer of attribute of given id
        /** Basically it will get the start of the array at the same point as OpenGL will get upon a glDraw*.
        @param attrId Attribute id.
        @returns Pointer to corresponding buffer's data incremented by `baseVertex` and by `bufferOffset`
        @see @ref getBaseVertex() setBaseVertex() getAttribute()
        */
        virtual uint8_t* getAttribPointer(uint32_t attrId)
        {
            assert(!isImmutable_debug());

            if (!m_pipeline)
                return nullptr;

            const auto& cachedParams = m_pipeline->getCachedCreationParams();
            const auto& vtxInputParams = cachedParams.vertexInput;
            if (!isAttributeEnabled(attrId))
                return nullptr;

            const uint32_t bindingNum = vtxInputParams.attributes[attrId].binding;
            if (!isVertexAttribBufferBindingEnabled(bindingNum))
                return nullptr;

            ICPUBuffer* mappedAttrBuf = m_vertexBufferBindings[bindingNum].buffer.get();
            if (!mappedAttrBuf)
                return nullptr;

            int64_t ix = vtxInputParams.bindings[bindingNum].inputRate!=SVertexInputBindingParams::EVIR_PER_VERTEX ? baseInstance:baseVertex;
            ix *= vtxInputParams.bindings[bindingNum].stride;
            ix += (m_vertexBufferBindings[bindingNum].offset + vtxInputParams.attributes[attrId].relativeOffset);
            if (ix < 0 || static_cast<uint64_t>(ix) >= mappedAttrBuf->getSize())
                return nullptr;

            return reinterpret_cast<uint8_t*>(mappedAttrBuf->getPointer()) + ix;
        }
        inline const uint8_t* getAttribPointer(uint32_t attrId) const
        {
            return const_cast<typename std::decay<decltype(*this)>::type&>(*this).getAttribPointer(attrId);
        }

        static inline bool getAttribute(core::vectorSIMDf& output, const void* src, E_FORMAT format)
        {
            if (!src)
                return false;

            bool scaled = false;
            if (!isNormalizedFormat(format) && !isFloatingPointFormat(format) && !(scaled = isScaledFormat(format)))
                return false;

            if (!scaled)
            {
                double output64[4]{ 0., 0., 0., 1. };
                decodePixels<double>(format, &src, output64, 0u, 0u);
                for (auto i=0u; i<4u; i++)
                    output[i] = static_cast<float>(output64[i]);
            }
            else
            {
                if (isSignedFormat(format))
                {
                    int64_t output64i[4]{ 0, 0, 0, 1 };
                    decodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), &src, output64i, 0u, 0u);
                    for (auto i=0u; i<4u; i++)
                        output[i] = static_cast<float>(output64i[i]);
                }
                else
                {
                    uint64_t output64u[4]{ 0u, 0u, 0u, 1u };
                    decodePixels<uint64_t>(impl::getCorrespondingIntegerFmt(format), &src, output64u, 0u, 0u);
                    for (auto i=0u; i<4u; i++)
                        output[i] = static_cast<float>(output64u[i]);
                }
            }

            return true;
        }

        //! Accesses vertex of given index of given vertex attribute. Index number is incremented by `baseVertex`. WARNING: NOT ALL FORMAT CONVERSIONS TO RGBA32F/XYZW32F ARE IMPLEMENTED!
        /** If component count of given attribute is less than 4, only first ones of output vector's members will be written.
        @param[out] output vectorSIMDf object to which index's value will be returned.
        @param[in] attrId Atrribute id.
        @param[in] ix Index which is to be accessed. Will be incremented by `baseVertex`.
        @returns true if successful or false if an error occured (e.g. `ix` out of range, no attribute specified/bound or given attribute's format conversion to vectorSIMDf unsupported).
        @see @ref getBaseVertex() setBaseVertex() getAttribute()
        */
        virtual bool getAttribute(core::vectorSIMDf& output, uint32_t attrId, size_t ix) const
        {
            if (!isAttributeEnabled(attrId))
                return false;

            const uint32_t bindingId = getBindingNumForAttribute(attrId);

            const uint8_t* src = getAttribPointer(attrId);
            src += ix * getAttribStride(attrId);
            if (src >= reinterpret_cast<const uint8_t*>(m_vertexBufferBindings[bindingId].buffer->getPointer()) + m_vertexBufferBindings[bindingId].buffer->getSize())
                return false;

            return getAttribute(output, src, getAttribFormat(attrId));
        }

        static inline bool getAttribute(uint32_t* output, const void* src, E_FORMAT format)
        {
            if (!src)
                return false;

            bool scaled = false;
            if ((scaled = isScaledFormat(format)) || isIntegerFormat(format))
            {
                if (isSignedFormat(format))
                {
                    int64_t output64[4]{0, 0, 0, 1};
                    decodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                    for (uint32_t i = 0u; i < getFormatChannelCount(format); ++i)
                        output[i] = static_cast<uint32_t>(output64[i]);
                }
                else
                {
                    uint64_t output64[4]{0u, 0u, 0u, 1u};
                    decodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                    for (uint32_t i = 0u; i < getFormatChannelCount(format); ++i)
                        output[i] = static_cast<uint32_t>(output64[i]);
                }
                return true;
            }

            return false;
        }

        //! Accesses vertex of given index of given vertex attribute. Index number is incremented by `baseVertex`. WARNING: NOT ALL FORMAT CONVERSIONS TO RGBA32F/XYZW32F ARE IMPLEMENTED!
        /** If component count of given attribute is less than 4, only first ones of output vector's members will be written.
        Attributes of integer types smaller than 32 bits are promoted to 32bit integer.
        @param[out] output Pointer to memory to which index's value will be returned.
        @param[in] attrId Atrribute id.
        @param[in] ix Index which is to be accessed. Will be incremented by `baseVertex`.
        @returns true if successful or false if an error occured (e.g. `ix` out of range, no attribute specified/bound or given attribute's format conversion to vectorSIMDf unsupported).
        @see @ref getBaseVertex() setBaseVertex() getAttribute()
        */
        virtual bool getAttribute(uint32_t* output, uint32_t attrId, size_t ix) const
        {
            if (!m_pipeline)
                return false;
            if (!isAttributeEnabled(attrId))
                return false;

            const uint8_t* src = getAttribPointer(attrId);
            src += ix * getAttribStride(attrId);
            const ICPUBuffer* buf = base_t::getAttribBoundBuffer(attrId).buffer.get();
            if (!buf || src >= reinterpret_cast<const uint8_t*>(buf->getPointer()) + buf->getSize())
                return false;

            return getAttribute(output, src, getAttribFormat(attrId));
        }

        static inline bool setAttribute(core::vectorSIMDf input, void* dst, E_FORMAT format)
        {
            bool scaled = false;
            if (!dst || (!isFloatingPointFormat(format) && !isNormalizedFormat(format) && !(scaled = isScaledFormat(format))))
                return false;

            double input64[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                input64[i] = input.pointer[i];

            if (!scaled)
                encodePixels<double>(format, dst, input64);
            else
            {
                if (isSignedFormat(format))
                {
                    int64_t input64i[4]{ static_cast<int64_t>(input64[0]), static_cast<int64_t>(input64[1]), static_cast<int64_t>(input64[2]), static_cast<int64_t>(input64[3]) };
                    encodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), dst, input64i);
                }
                else
                {
                    uint64_t input64u[4]{ static_cast<uint64_t>(input64[0]), static_cast<uint64_t>(input64[1]), static_cast<uint64_t>(input64[2]), static_cast<uint64_t>(input64[3]) };
                    encodePixels<uint64_t>(impl::getCorrespondingIntegerFmt(format), dst, input64u);
                }
            }

            return true;
        }

        //! Sets value of vertex of given index of given attribute. WARNING: NOT ALL FORMAT CONVERSIONS FROM RGBA32F/XYZW32F (vectorSIMDf) ARE IMPLEMENTED!
        /** @param input Value which is to be set.
        @param attrId Atrribute id.
        @param ix Index of vertex which is to be set. Will be incremented by `baseVertex`.
        @returns true if successful or false if an error occured (e.g. no such index).
        @see @ref getBaseVertex() setBaseVertex() getAttribute()
        */
        virtual bool setAttribute(core::vectorSIMDf input, uint32_t attrId, size_t ix)
        {
            assert(!isImmutable_debug());
            if (!m_pipeline)
                return false;
            if (!isAttributeEnabled(attrId))
                return false;

            uint8_t* dst = getAttribPointer(attrId);
            dst += ix * getAttribStride(attrId);
            ICPUBuffer* buf = getAttribBoundBuffer(attrId).buffer.get();
            if (!buf || dst >= ((const uint8_t*)(buf->getPointer())) + buf->getSize())
                return false;

            return setAttribute(input, dst, getAttribFormat(attrId));
        }

        static inline bool setAttribute(const uint32_t* _input, void* dst, E_FORMAT format)
        {
            const bool scaled = isScaledFormat(format);
            if (!dst || !(scaled || isIntegerFormat(format)))
                return false;
            uint8_t* vxPtr = (uint8_t*)dst;

            if (isSignedFormat(format))
            {
                int64_t input[4];
                for (uint32_t i = 0u; i < 4u; ++i)
                    input[i] = reinterpret_cast<const int32_t*>(_input)[i];
                encodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, dst, input);
            }
            else
            {
                uint64_t input[4];
                for (uint32_t i = 0u; i < 4u; ++i)
                    input[i] = _input[i];
                encodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, dst, input);
            }
            return true;
        }

        //! @copydoc setAttribute(core::vectorSIMDf, const E_VERTEX_ATTRIBUTE_ID, size_t)
        virtual bool setAttribute(const uint32_t* _input, uint32_t attrId, size_t ix)
        {
            assert(!isImmutable_debug());
            if (!m_pipeline)
                return false;
            if (!isAttributeEnabled(attrId))
                return false;

            uint8_t* dst = getAttribPointer(attrId);
            dst += ix * getAttribStride(attrId);
            ICPUBuffer* buf = getAttribBoundBuffer(attrId).buffer.get();
            if (dst >= ((const uint8_t*)(buf->getPointer())) + buf->getSize())
                return false;

            return setAttribute(_input, dst, getAttribFormat(attrId));
        }

        //!
        inline const core::matrix3x4SIMD* getInverseBindPoses() const
        {
            if (!m_inverseBindPoseBufferBinding.buffer)
                return nullptr;

            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_inverseBindPoseBufferBinding.buffer->getPointer());
            return reinterpret_cast<const core::matrix3x4SIMD*>(ptr+m_inverseBindPoseBufferBinding.offset);
        }
        inline core::matrix3x4SIMD* getInverseBindPoses()
        {
            assert(!isImmutable_debug());
            return const_cast<core::matrix3x4SIMD*>(const_cast<const ICPUMeshBuffer*>(this)->getInverseBindPoses());
        }

        //!
        inline const core::aabbox3df* getJointAABBs() const
        {
            if (!m_jointAABBBufferBinding.buffer)
                return nullptr;

            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_jointAABBBufferBinding.buffer->getPointer());
            return reinterpret_cast<const core::aabbox3df*>(ptr+ m_jointAABBBufferBinding.offset);
        }
        inline core::aabbox3df* getJointAABBs()
        {
            assert(!isImmutable_debug());
            return const_cast<core::aabbox3df*>(const_cast<const ICPUMeshBuffer*>(this)->getJointAABBs());
        }

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUMeshBuffer*>(_other);
            if (memcmp(m_pushConstantsData, other->m_pushConstantsData, sizeof(m_pushConstantsData)) != 0)
                return false;
            if (baseVertex != other->baseVertex)
                return false;
            if (indexCount != other->indexCount)
                return false;
            if (instanceCount != other->instanceCount)
                return false;
            if (baseInstance != other->baseInstance)
                return false;
            if (posAttrId != other->posAttrId)
                return false;
            if (normalAttrId != other->normalAttrId)
                return false;
            if (jointIDAttrId != other->jointIDAttrId)
                return false;
            if (jointWeightAttrId != other->jointWeightAttrId)
                return false;
            if (jointCount != other->jointCount)
                return false;
            if (maxJointsPerVx != other->maxJointsPerVx)
                return false;
            if (m_indexBufferBinding.offset != other->m_indexBufferBinding.offset)
                return false;
            if ((!m_indexBufferBinding.buffer) != (!other->m_indexBufferBinding.buffer))
                return false;
            if (m_indexBufferBinding.buffer && !m_indexBufferBinding.buffer->canBeRestoredFrom(other->m_indexBufferBinding.buffer.get()))
                return false;
            for (uint32_t i = 0u; i<MAX_ATTR_BUF_BINDING_COUNT; ++i)
            {
                if (m_vertexBufferBindings[i].offset != other->m_vertexBufferBindings[i].offset)
                    return false;
                if ((!m_vertexBufferBindings[i].buffer) != (!other->m_vertexBufferBindings[i].buffer))
                    return false;
                if (m_vertexBufferBindings[i].buffer && !m_vertexBufferBindings[i].buffer->canBeRestoredFrom(other->m_vertexBufferBindings[i].buffer.get()))
                    return false;
            }
            
            if (m_inverseBindPoseBufferBinding.offset != other->m_inverseBindPoseBufferBinding.offset)
                return false;
            if ((!m_inverseBindPoseBufferBinding.buffer) != (!other->m_inverseBindPoseBufferBinding.buffer))
                return false;
            if (m_inverseBindPoseBufferBinding.buffer && !m_inverseBindPoseBufferBinding.buffer->canBeRestoredFrom(other->m_inverseBindPoseBufferBinding.buffer.get()))
                return false;
            if (m_jointAABBBufferBinding.offset != other->m_jointAABBBufferBinding.offset)
                return false;
            if ((!m_jointAABBBufferBinding.buffer) != (!other->m_jointAABBBufferBinding.buffer))
                return false;
            if (m_jointAABBBufferBinding.buffer && !m_jointAABBBufferBinding.buffer->canBeRestoredFrom(other->m_jointAABBBufferBinding.buffer.get()))
                return false;

            if ((!m_descriptorSet) != (!other->m_descriptorSet))
                return false;
            if (m_descriptorSet && !m_descriptorSet->canBeRestoredFrom(other->m_descriptorSet.get()))
                return false;

            // pipeline is not optional
            if (!m_pipeline->canBeRestoredFrom(other->m_pipeline.get()))
                return false;

            return true;
        }

    protected:
        void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
        {
            auto* other = static_cast<ICPUMeshBuffer*>(_other);

            if (_levelsBelow)
            {
                --_levelsBelow;

                if (m_pipeline)
                    restoreFromDummy_impl_call(m_pipeline.get(), other->m_pipeline.get(), _levelsBelow);
                if (m_descriptorSet)
                    restoreFromDummy_impl_call(m_descriptorSet.get(), other->m_descriptorSet.get(), _levelsBelow);

                if (m_inverseBindPoseBufferBinding.buffer)
                    restoreFromDummy_impl_call(m_inverseBindPoseBufferBinding.buffer.get(), other->m_inverseBindPoseBufferBinding.buffer.get(), _levelsBelow);
                if (m_jointAABBBufferBinding.buffer)
                    restoreFromDummy_impl_call(m_jointAABBBufferBinding.buffer.get(), other->m_jointAABBBufferBinding.buffer.get(), _levelsBelow);

                for (uint32_t i = 0u; i < MAX_ATTR_BUF_BINDING_COUNT; ++i)
                    if (m_vertexBufferBindings[i].buffer)
                        restoreFromDummy_impl_call(m_vertexBufferBindings[i].buffer.get(), other->m_vertexBufferBindings[i].buffer.get(), _levelsBelow);
                if (m_indexBufferBinding.buffer)
                    restoreFromDummy_impl_call(m_indexBufferBinding.buffer.get(), other->m_indexBufferBinding.buffer.get(), _levelsBelow);
            }
        }

        bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
        {
            --_levelsBelow;
            if (m_pipeline && m_pipeline->isAnyDependencyDummy(_levelsBelow))
                return true;
            if (m_descriptorSet && m_descriptorSet->isAnyDependencyDummy(_levelsBelow))
                return true;

            if (m_inverseBindPoseBufferBinding.buffer && m_inverseBindPoseBufferBinding.buffer->isAnyDependencyDummy(_levelsBelow))
                return true;
            if (m_jointAABBBufferBinding.buffer && m_jointAABBBufferBinding.buffer->isAnyDependencyDummy(_levelsBelow))
                return true;

            for (uint32_t i = 0u; i < MAX_ATTR_BUF_BINDING_COUNT; ++i)
                if (m_vertexBufferBindings[i].buffer && m_vertexBufferBindings[i].buffer->isAnyDependencyDummy(_levelsBelow))
                    return true;

            return (m_indexBufferBinding.buffer && m_indexBufferBinding.buffer->isAnyDependencyDummy(_levelsBelow));
        }
};

}

#endif
