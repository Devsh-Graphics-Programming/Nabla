// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"

#include <iterator>


//#include "nbl/asset/asset.h"


#include "nbl/video/asset_traits.h"
#include "nbl/video/ISemaphore.h"
#include "nbl/video/ILogicalDevice.h"

#include "nbl/asset/ECommonEnums.h"

namespace nbl::video
{

namespace impl
{
// non-pointer iterator type is AssetBundleIterator<> (see IDriver.cpp)
template<typename iterator_type>
inline constexpr bool is_const_iterator_v =
    (std::is_pointer_v<iterator_type> && is_pointer_to_const_object_v<iterator_type>) ||
    (!std::is_pointer_v<iterator_type> && std::is_const_v<iterator_type>);

template<class AssetType>
struct AssetBundleIterator
{
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;

        AssetBundleIterator(const core::smart_refctd_ptr<asset::IAsset>* _ptr) : ptr(_ptr) {}

        // general operators
        inline AssetType* operator*()
        {
            return static_cast<AssetType*>(ptr->get());
        }
        inline const AssetType* operator*() const
        {
            return static_cast<const AssetType*>(ptr->get());
        }
        inline const core::smart_refctd_ptr<asset::IAsset>* operator->() const
        {
            return ptr;
        }

        // arithmetic operators
        inline AssetBundleIterator<AssetType>& operator++()
        {
            ++ptr;
            return *this;
        }
        inline AssetBundleIterator<AssetType> operator++(int)
        {
            return AssetBundleIterator<AssetType>(ptr++);
        }
        inline difference_type operator-(const AssetBundleIterator<AssetType>& other) const
        {
            return ptr - other.ptr;
        }

        // comparison operators
        inline bool operator!=(const AssetBundleIterator<AssetType>& other) const
        {
            return ptr != other.ptr;
        }

    private:
        const core::smart_refctd_ptr<asset::IAsset>* ptr;
};
}


#if 0
class IGPUObjectFromAssetConverter
{
    public:

        enum E_QUEUE_USAGE
        {
            EQU_COMPUTE = 0,
            EQU_TRANSFER,

            EQU_COUNT
        };

        struct SParams
        {
            struct SPerQueue
            {
                IQueue* queue = nullptr;

                core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf = nullptr;
                //! If not null, semaphore will be written here. Written semaphore will be signaled once last operations of corresponding type (transfer/compute) are finished.
                core::smart_refctd_ptr<IGPUSemaphore>* semaphore = nullptr;
                core::smart_refctd_ptr<IEvent>* event = nullptr;
            };

            //! Required not null if an ICPUImage or ICPUBuffer needs converting
            IUtilities* utilities = nullptr;
            //! Required not null
            ILogicalDevice* device = nullptr;
            //! Required not null
            asset::IAssetManager* assetManager = nullptr;
            IGPUPipelineCache* pipelineCache = nullptr;

            uint32_t finalQueueFamIx = 0u;

            // @sadiuk put here more parameters if needed

            SPerQueue perQueue[EQU_COUNT];
            
            inline void beginCommandBuffer(E_QUEUE_USAGE queueUsage)
            {
                uint32_t queue = static_cast<uint32_t>(queueUsage);
                if(perQueue[queue].cmdbuf)
                    perQueue[queue].cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            }

            inline void beginCommandBuffers()
            {
                for (auto i=0; i<EQU_COUNT; i++)
                {
                    beginCommandBuffer(static_cast<E_QUEUE_USAGE>(i));
                }
            }
            
            inline void resetCommandBuffers()
            {
                for (auto i=0; i<EQU_COUNT; i++)
                {
                    if(perQueue[i].cmdbuf)
                        perQueue[i].cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
                }
            }
            
            //
            inline void waitForCreationToComplete(bool resetFencesAfterWait = false)
            {
                IGPUFence* fence_ptrs[EQU_COUNT];
                uint32_t count = 0;
                for (auto i=0; i<EQU_COUNT; i++)
                {
                    auto pFence = fences[i];
                    if (pFence && pFence.get()) // user wanted, and something actually got submitted
                        fence_ptrs[count++] = pFence.get();
                }
                if (count)
                {
                    device->blockForFences(count,fence_ptrs);
                    if(resetFencesAfterWait)
                    {
                        device->resetFences(count, fence_ptrs);
                    }
                }
                setFencesToNull();
                resetCommandBuffers();
            }

        protected:
            friend class IGPUObjectFromAssetConverter;
            inline void setFencesToNull()
            {
                for (auto i=0; i<EQU_COUNT; i++)
                if (fences[i])
                    fences[i] = nullptr;
            }

            core::smart_refctd_ptr<IGPUFence> fences[EQU_COUNT] = {};
        };

	protected:
		template<typename AssetType, typename iterator_type>
		struct get_asset_raw_ptr
		{
			static inline AssetType* value(iterator_type it) { return static_cast<AssetType*>(it->get()); }
		};

		template<typename AssetType>
		struct get_asset_raw_ptr<AssetType, AssetType**>
		{
			static inline AssetType* value(AssetType** it) { return *it; }
		};
		template<typename AssetType>
		struct get_asset_raw_ptr<AssetType, AssetType*const*>
		{
			static inline AssetType* value(AssetType*const* it) { return *it; }
		};

        template<typename AssetType>
        struct Hash
        {
            inline std::size_t operator()(AssetType* asset) const
            {
                return std::hash<AssetType*>{}(asset);
            }
        };

        template<typename AssetType>
        struct KeyEqual
        {
            bool operator()(AssetType* lhs, AssetType* rhs) const { return lhs==rhs; }
        };

		inline virtual created_gpu_object_array<asset::ICPUBuffer>				            create(const asset::ICPUBuffer** const _begin, const asset::ICPUBuffer** const _end, SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUImage>				            create(const asset::ICPUImage** const _begin, const asset::ICPUImage** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUSampler>		                    create(const asset::ICPUSampler** const _begin, const asset::ICPUSampler** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUPipelineLayout>		            create(const asset::ICPUPipelineLayout** const _begin, const asset::ICPUPipelineLayout** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPURenderpassIndependentPipeline>	create(const asset::ICPURenderpassIndependentPipeline** const _begin, const asset::ICPURenderpassIndependentPipeline** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSetLayout>             create(const asset::ICPUDescriptorSetLayout** const _begin, const asset::ICPUDescriptorSetLayout** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUComputePipeline>				    create(const asset::ICPUComputePipeline** const _begin, const asset::ICPUComputePipeline** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUSpecializedShader>	            create(const asset::ICPUSpecializedShader** const _begin, const asset::ICPUSpecializedShader** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUShader>				            create(const asset::ICPUShader** const _begin, const asset::ICPUShader** const _end, SParams& _params);
		//inline virtual created_gpu_object_array<asset::ICPUSkeleton>			            create(const asset::ICPUSkeleton** const _begin, const asset::ICPUSkeleton** const _end, SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUMeshBuffer>			            create(const asset::ICPUMeshBuffer** const _begin, const asset::ICPUMeshBuffer** const _end, SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUMesh>				            create(const asset::ICPUMesh** const _begin, const asset::ICPUMesh** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUBufferView>		                create(const asset::ICPUBufferView** const _begin, const asset::ICPUBufferView** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUImageView>				        create(const asset::ICPUImageView** const _begin, const asset::ICPUImageView** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSet>				    create(const asset::ICPUDescriptorSet** const _begin, const asset::ICPUDescriptorSet** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUAnimationLibrary>			    create(const asset::ICPUAnimationLibrary** const _begin, const asset::ICPUAnimationLibrary** const _end, SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUAccelerationStructure>			create(const asset::ICPUAccelerationStructure** const _begin, const asset::ICPUAccelerationStructure** const _end, SParams& _params);


        //! iterator_type is always either `[const] core::smart_refctd_ptr<AssetType>*[const]*` or `[const] AssetType*[const]*`
		template<typename AssetType, typename iterator_type>
        std::enable_if_t<!impl::is_const_iterator_v<iterator_type>, created_gpu_object_array<AssetType>>
        getGPUObjectsFromAssets(iterator_type _begin, iterator_type _end, SParams& _params)
		{
			const auto assetCount = _end-_begin;
			auto res = core::make_refctd_dynamic_array<created_gpu_object_array<AssetType> >(assetCount);

			core::vector<AssetType*> notFound; notFound.reserve(assetCount);
			core::vector<size_t> pos; pos.reserve(assetCount);

			for (iterator_type it = _begin; it != _end; it++)
			{
				const auto index = it-_begin;

				//if (*it)
				//{
					auto gpu = _params.assetManager->findGPUObject(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
					if (!gpu)
					{
						if ((*it)->isADummyObjectForCache())
							notFound.push_back(nullptr);
						else
							notFound.push_back(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
						pos.push_back(index);
					}
					else
						res->operator[](index) = core::move_and_dynamic_cast<typename asset_traits<AssetType>::GPUObjectType>(std::move(gpu));
				//}
				//res->operator[](index) = nullptr;
			}

			if (notFound.size())
			{
				decltype(res) created = create(const_cast<const AssetType**>(notFound.data()), const_cast<const AssetType**>(notFound.data()+notFound.size()), _params);
				for (size_t i=0u; i<created->size(); ++i)
				{
					auto& input = created->operator[](i);
                    handleGPUObjCaching(_params.assetManager,notFound[i],input);
					res->operator[](pos[i]) = std::move(input); // ok to move because the `created` array will die after the next scope
				}
			}

			return res;
		}
        template<typename AssetType, typename const_iterator_type>
        std::enable_if_t<impl::is_const_iterator_v<const_iterator_type>, created_gpu_object_array<AssetType>>
            getGPUObjectsFromAssets(const_iterator_type _begin, const_iterator_type _end, SParams& _params)
        {
            if constexpr (std::is_pointer_v<const_iterator_type>)
            {
                using iterator_type = pointer_to_nonconst_object_t<const_iterator_type>;

                auto begin = const_cast<iterator_type>(_begin);
                auto end = const_cast<iterator_type>(_end);

                return getGPUObjectsFromAssets<AssetType, iterator_type>(begin, end, _params);
            }
            else
            {
                using iterator_type = std::remove_const_t<const_iterator_type>;

                iterator_type& begin = const_cast<iterator_type&>(_begin);
                iterator_type& end = const_cast<iterator_type&>(_end);

                return getGPUObjectsFromAssets<AssetType, iterator_type>(begin, end, _params);
            }
        }

        bool computeQueueSupportsGraphics(const SParams& _params)
        {
            // assert that ComputeQueue also supports Graphics
            const uint32_t transferFamIx = _params.perQueue[EQU_TRANSFER].queue ? _params.perQueue[EQU_TRANSFER].queue->getFamilyIndex():~0u;
            const uint32_t computeFamIx = _params.perQueue[EQU_COMPUTE].queue ? _params.perQueue[EQU_COMPUTE].queue->getFamilyIndex() : transferFamIx;

            auto queueFamProps = _params.device->getPhysicalDevice()->getQueueFamilyProperties();
			if (computeFamIx>=queueFamProps.size())
				return true; // no queue specified

            const auto& familyProperty = queueFamProps.begin()[computeFamIx];
            const bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
            return hasGraphicsFlag;
        }

    public:
        virtual ~IGPUObjectFromAssetConverter() = default;

        template<typename AssetType>
        created_gpu_object_array<AssetType> getGPUObjectsFromAssets(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>& _range, SParams& _params)
        {
            assert(computeQueueSupportsGraphics(_params));
            _params.setFencesToNull();
            impl::AssetBundleIterator<AssetType> begin(_range.begin());
            impl::AssetBundleIterator<AssetType> end(_range.end());
            return this->template getGPUObjectsFromAssets<AssetType, const impl::AssetBundleIterator<AssetType>>(begin, end, _params);
        }

        template<typename AssetType>
        created_gpu_object_array<AssetType> getGPUObjectsFromAssets(const AssetType* const* const _begin, const AssetType* const* const _end, SParams& _params)
        {
            assert(computeQueueSupportsGraphics(_params));
            _params.setFencesToNull();
            return this->template getGPUObjectsFromAssets<AssetType, const AssetType* const*>(_begin, _end, _params);
        }

        template<typename AssetType>
        created_gpu_object_array<AssetType> getGPUObjectsFromAssets(const core::smart_refctd_ptr<AssetType>* _begin, const core::smart_refctd_ptr<AssetType>* _end, SParams& _params)
        {
            assert(computeQueueSupportsGraphics(_params));
            _params.setFencesToNull();
            return this->template getGPUObjectsFromAssets<AssetType, const core::smart_refctd_ptr<AssetType>*>(_begin, _end, _params);
        }

	protected:
        virtual inline void handleGPUObjCaching(asset::IAssetManager* _amgr, asset::IAsset* _asset, const core::smart_refctd_ptr<core::IReferenceCounted>& _gpuobj)
        {
            if (_asset && _gpuobj)
                _amgr->convertAssetToEmptyCacheHandle(_asset,core::smart_refctd_ptr(_gpuobj));
        }

		//! TODO: Make this faster and not call any allocator
		template<typename T>
		static inline core::vector<size_t> eliminateDuplicatesAndGenRedirs(core::vector<T*>& _input)
		{
			core::vector<size_t> redirs;

			core::unordered_map<T*, size_t, Hash<T>, KeyEqual<T>> firstOccur;
			size_t i = 0u;
			for (T* el : _input)
			{
				if (!el)
				{
					redirs.push_back(0xdeadbeefu);
					continue;
				}

				auto r = firstOccur.insert({ el, i });
				redirs.push_back(r.first->second);
				if (r.second)
					++i;
			}
			for (const auto& p : firstOccur)
				_input.push_back(p.first);
			_input.erase(_input.begin(), _input.begin() + (_input.size() - firstOccur.size()));
			std::sort(_input.begin(), _input.end(), [&firstOccur](T* a, T* b) { return firstOccur[a] < firstOccur[b]; });

			return redirs;
		}
};


class CAssetPreservingGPUObjectFromAssetConverter : public IGPUObjectFromAssetConverter
{
    public:
        using IGPUObjectFromAssetConverter::IGPUObjectFromAssetConverter;

	protected:
        virtual inline void handleGPUObjCaching(asset::IAssetManager* _amgr, asset::IAsset* _asset, const core::smart_refctd_ptr<core::IReferenceCounted>& _gpuobj) override
        {
            if (_asset && _gpuobj)
                _amgr->insertGPUObjectIntoCache(_asset,core::smart_refctd_ptr(_gpuobj));
        }
};


// need to specialize outside because of GCC
template<>
struct IGPUObjectFromAssetConverter::Hash<const asset::ICPURenderpassIndependentPipeline>
{
    inline std::size_t operator()(const asset::ICPURenderpassIndependentPipeline* _ppln) const
    {
        constexpr size_t bytesToHash = 
            asset::SVertexInputParams::serializedSize()+
            asset::SBlendParams::serializedSize()+
            asset::SRasterizationParams::serializedSize()+
            asset::SPrimitiveAssemblyParams::serializedSize()+
            sizeof(void*)*asset::ICPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT+//shaders
            sizeof(void*);//layout
        uint8_t mem[bytesToHash]{};
        uint32_t offset = 0u;
        _ppln->getVertexInputParams().serialize(mem+offset);
        offset += asset::SVertexInputParams::serializedSize();
        _ppln->getBlendParams().serialize(mem+offset);
        offset += asset::SBlendParams::serializedSize();
        _ppln->getRasterizationParams().serialize(mem+offset);
        offset += sizeof(asset::SRasterizationParams);
        _ppln->getPrimitiveAssemblyParams().serialize(mem+offset);
        offset += sizeof(asset::SPrimitiveAssemblyParams);
        const asset::ICPUSpecializedShader** shaders = reinterpret_cast<const asset::ICPUSpecializedShader**>(mem+offset);
        for (uint32_t i = 0u; i < asset::ICPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT; ++i)
            shaders[i] = _ppln->getShaderAtIndex(i);
        offset += asset::ICPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT*sizeof(void*);
        reinterpret_cast<const asset::ICPUPipelineLayout**>(mem+offset)[0] = _ppln->getLayout();

        const std::size_t hs = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), bytesToHash));

        return hs;
    }
};
template<>
struct IGPUObjectFromAssetConverter::Hash<const asset::ICPUComputePipeline>
{
    inline std::size_t operator()(const asset::ICPUComputePipeline* _ppln) const
    {
        constexpr size_t bytesToHash = 
            sizeof(void*)+//shader
            sizeof(void*);//layout
        uint8_t mem[bytesToHash]{};

        reinterpret_cast<const asset::ICPUSpecializedShader**>(mem)[0] = _ppln->getShader();
        reinterpret_cast<const asset::ICPUPipelineLayout**>(mem+sizeof(void*))[0] = _ppln->getLayout();

        const std::size_t hs = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), bytesToHash));

        return hs;
    }
};

template<>
struct IGPUObjectFromAssetConverter::KeyEqual<const asset::ICPURenderpassIndependentPipeline>
{
    //equality depends on hash only
    bool operator()(const asset::ICPURenderpassIndependentPipeline* lhs, const asset::ICPURenderpassIndependentPipeline* rhs) const { return true; }
};
template<>
struct IGPUObjectFromAssetConverter::KeyEqual<const asset::ICPUComputePipeline>
{
    //equality depends on hash only
    bool operator()(const asset::ICPUComputePipeline* lhs, const asset::ICPUComputePipeline* rhs) const { return true; }
};


auto IGPUObjectFromAssetConverter::create(const asset::ICPUBuffer** const _begin, const asset::ICPUBuffer** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUBuffer> // TODO: improve for caches of very large buffers!!!
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBuffer> >(assetCount);

    const auto& limits = _params.device->getPhysicalDevice()->getLimits();

    const uint64_t alignment =
    std::max<uint64_t>(
        std::max<uint64_t>(limits.bufferViewAlignment,limits.minSSBOAlignment),
        std::max<uint64_t>(limits.minUBOAlignment, _NBL_SIMD_ALIGNMENT)
    );

    const uint64_t maxBufferSize = limits.maxBufferSize;
    auto out = res->begin();
    auto firstInBlock = out;
    auto newBlock = [&]() -> auto
    {
        return core::LinearAddressAllocator<uint64_t>(nullptr, 0u, 0u, alignment, maxBufferSize);
    };
    auto addrAllctr = newBlock();

    auto & fence = _params.fences[EQU_TRANSFER];
    fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
    core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf = _params.perQueue[EQU_TRANSFER].cmdbuf;

    IQueue::SSubmitInfo submit;
    {
        submit.commandBufferCount = 1u;
        submit.commandBuffers = &cmdbuf.get();
        // CPU to GPU upload doesn't need to wait for anything or narrow down the execution barrier
        // buffer and addresses we're writing into are fresh and brand new, don't need to synchronize the writing with anything
        submit.waitSemaphoreCount = 0u;
        submit.pWaitDstStageMask = nullptr;
        submit.pWaitSemaphores = nullptr;
    }
    
    assert(cmdbuf && cmdbuf->getState() == IGPUCommandBuffer::STATE::RECORDING);
    // cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

    auto finalizeBlock = [&]() -> void
    {
        auto bufferSize = addrAllctr.get_allocated_size();
        if (bufferSize==0u)
            return;
        
        IGPUBuffer::SCreationParams bufparams = {};
        bufparams.size = bufferSize;
        bufparams.usage = core::bitflag(IGPUBuffer::EUF_TRANSFER_DST_BIT);

        for (auto it = firstInBlock; it != out; it++)
        {
            auto cpubuffer = _begin[std::distance(res->begin(), it)];
            bufparams.usage |= cpubuffer->getUsageFlags();
        }

        uint32_t qfams[2]{ _params.perQueue[EQU_TRANSFER].queue->getFamilyIndex(), _params.finalQueueFamIx };
        bufparams.queueFamilyIndices = qfams;
        bufparams.queueFamilyIndexCount = (qfams[0] == qfams[1]) ? 0u : 2u;
        
        core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags(IDeviceMemoryAllocation::EMAF_NONE);
        if(bufparams.usage.hasFlags(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
            allocateFlags |= IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT;

        auto gpubuffer = _params.device->createBuffer(std::move(bufparams));
        auto gpubufferMemReqs = gpubuffer->getMemoryReqs();
        gpubufferMemReqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufferMem = _params.device->allocate(gpubufferMemReqs, gpubuffer.get(), allocateFlags);

        for (auto it = firstInBlock; it != out; it++)
        {
            if (auto output = *it)
            {
                auto cpubuffer = _begin[std::distance(res->begin(), it)];
                asset::SBufferRange<IGPUBuffer> bufrng;
                bufrng.offset = output->getOffset();
                bufrng.size = cpubuffer->getSize();
                bufrng.buffer = gpubuffer;
                output->setBuffer(core::smart_refctd_ptr(gpubuffer));
                submit = _params.utilities->updateBufferRangeViaStagingBuffer(
                    bufrng,cpubuffer->getPointer(),
                    _params.perQueue[EQU_TRANSFER].queue, fence.get(), submit
                );
            }
        }
    };
    for (auto it=_begin; it!=_end; it++,out++)
    {
        auto cpubuffer = *it;
        if (cpubuffer->getSize()>maxBufferSize)
            continue;

        uint64_t addr = addrAllctr.alloc_addr(cpubuffer->getSize(),alignment);
        if (addr==decltype(addrAllctr)::invalid_address)
        {
            finalizeBlock();
            firstInBlock = out;
            addrAllctr = newBlock();
            addr = addrAllctr.alloc_addr(cpubuffer->getSize(),alignment);
        }
        assert(addr != decltype(addrAllctr)::invalid_address);
        *out = core::make_smart_refctd_ptr<typename asset_traits<asset::ICPUBuffer>::GPUObjectType>(addr);
    }
    finalizeBlock();

    // TODO: submit outside of `create` and make the function take an already created semaphore to signal
    cmdbuf->end();
    core::smart_refctd_ptr<IGPUSemaphore> sem;
    if (_params.perQueue[EQU_TRANSFER].semaphore)
    {
        sem = _params.device->createSemaphore();
        submit.signalSemaphoreCount = 1u;
        submit.pSignalSemaphores = &sem.get();
    }
    else
    {
        submit.signalSemaphoreCount = 0u;
        submit.pSignalSemaphores = nullptr;
    }
    // transfer_fence needs to be signalled because of the streaming buffer uploads need to be fenced
    _params.perQueue[EQU_TRANSFER].queue->submit(1u,&submit,fence.get());
    if (_params.perQueue[EQU_TRANSFER].semaphore)
        _params.perQueue[EQU_TRANSFER].semaphore[0] = std::move(sem);

    return res;
}

namespace impl
{
template<typename MapIterator>
struct CustomBoneNameIterator
{
        inline CustomBoneNameIterator(const MapIterator& it) : m_it(it) {}
        inline CustomBoneNameIterator(MapIterator&& it) : m_it(std::move(it)) {}

        inline bool operator!=(const CustomBoneNameIterator<MapIterator>& other) const
        {
            return m_it!=other.m_it;
        }

        inline CustomBoneNameIterator<MapIterator>& operator++()
        {
            ++m_it;
            return *this;
        }
        inline CustomBoneNameIterator<MapIterator> operator++(int)
        {
            return m_it++;
        }

        inline const auto& operator*() const
        {
            return m_it->first;
        }
        inline auto& operator*()
        {
            return m_it->first;
        }

        using iterator_category = typename std::iterator_traits<MapIterator>::iterator_category;
        using difference_type = typename std::iterator_traits<MapIterator>::difference_type;
        using value_type = const char*;
        using reference = std::add_lvalue_reference_t<value_type>;
        using pointer = std::add_pointer_t<value_type>;

    private:
        MapIterator m_it;
};
}


auto IGPUObjectFromAssetConverter::create(const asset::ICPUMeshBuffer** _begin, const asset::ICPUMeshBuffer** _end, SParams& _params) -> created_gpu_object_array<asset::ICPUMeshBuffer>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMeshBuffer> >(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount * (asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT+1u));
    core::vector<const asset::ICPUDescriptorSet*> cpuDescSets;
    cpuDescSets.reserve(assetCount);
    core::vector<const asset::ICPURenderpassIndependentPipeline*> cpuPipelines;
    cpuPipelines.reserve(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUMeshBuffer* cpumb = _begin[i];

        if (cpumb->getPipeline())
            cpuPipelines.push_back(cpumb->getPipeline());
        if (cpumb->getAttachedDescriptorSet())
            cpuDescSets.push_back(cpumb->getAttachedDescriptorSet());

        if (const asset::ICPUBuffer* buf = cpumb->getInverseBindPoseBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);
        if (const asset::ICPUBuffer* buf = cpumb->getJointAABBBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);

        for (size_t b = 0ull; b < asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            if (const asset::ICPUBuffer* buf = cpumb->getVertexBufferBindings()[b].buffer.get())
                cpuBuffers.push_back(buf);
        }
        if (const asset::ICPUBuffer* buf = cpumb->getIndexBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;

    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
    redirs_t dsRedirs = eliminateDuplicatesAndGenRedirs(cpuDescSets);
    redirs_t pplnRedirs = eliminateDuplicatesAndGenRedirs(cpuPipelines);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size(), _params);
    _params.waitForCreationToComplete(false);
    _params.beginCommandBuffers();
    auto gpuDescSets = getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(cpuDescSets.data(), cpuDescSets.data()+cpuDescSets.size(), _params);
    _params.waitForCreationToComplete(false);
    _params.beginCommandBuffers();
    auto gpuPipelines = getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(cpuPipelines.data(), cpuPipelines.data()+cpuPipelines.size(), _params);

    size_t pplnIter = 0ull, dsIter = 0ull, skelIter = 0ull, bufIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUMeshBuffer* cpumb = _begin[i];

        IGPURenderpassIndependentPipeline* gpuppln = nullptr;
        if (cpumb->getPipeline())
            gpuppln = (*gpuPipelines)[pplnRedirs[pplnIter++]].get();
        IGPUDescriptorSet* gpuds = nullptr;
        if (cpumb->getAttachedDescriptorSet())
            gpuds = (*gpuDescSets)[dsRedirs[dsIter++]].get();
        
		asset::SBufferBinding<IGPUBuffer> invBindPoseBinding;
        if (cpumb->getInverseBindPoseBufferBinding().buffer)
        {
            invBindPoseBinding.offset = cpumb->getInverseBindPoseBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            invBindPoseBinding.offset += gpubuf->getOffset();
            invBindPoseBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
		asset::SBufferBinding<IGPUBuffer> jointAABBBinding;
        if (cpumb->getJointAABBBufferBinding().buffer)
        {
            jointAABBBinding.offset = cpumb->getJointAABBBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            jointAABBBinding.offset += gpubuf->getOffset();
            jointAABBBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        asset::SBufferBinding<IGPUBuffer> vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        for (size_t b = 0ull; b < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            const auto& cpubnd = cpumb->getVertexBufferBindings()[b];
            if (cpubnd.buffer) {
                vtxBindings[b].offset = cpubnd.offset;
                auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
                vtxBindings[b].offset += gpubuf->getOffset();
                vtxBindings[b].buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
            }
        }

		asset::SBufferBinding<IGPUBuffer> idxBinding;
        if (cpumb->getIndexBufferBinding().buffer)
        {
            idxBinding.offset = cpumb->getIndexBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            idxBinding.offset += gpubuf->getOffset();
            idxBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuppln_(gpuppln);
        core::smart_refctd_ptr<IGPUDescriptorSet> gpuds_(gpuds);
        (*res)[i] = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(gpuppln_), std::move(gpuds_), vtxBindings, std::move(idxBinding));
        (*res)[i]->setBoundingBox(cpumb->getBoundingBox());
        memcpy((*res)[i]->getPushConstantsDataPtr(), _begin[i]->getPushConstantsDataPtr(), IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
        //(*res)[i]->setSkin(std::move(invBindPoseBinding),std::move(jointAABBBinding),std::move(gpuskel_),(core::min)(cpumb->getMaxJointsPerVertex(),cpumb->deduceMaxJointsPerVertex()));
        (*res)[i]->setBaseInstance(_begin[i]->getBaseInstance());
        (*res)[i]->setBaseVertex(_begin[i]->getBaseVertex());
        (*res)[i]->setIndexCount(_begin[i]->getIndexCount());
        (*res)[i]->setInstanceCount(_begin[i]->getInstanceCount());
        (*res)[i]->setIndexType(_begin[i]->getIndexType());
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(const asset::ICPUMesh** const _begin, const asset::ICPUMesh** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUMesh>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMesh> >(assetCount);

	core::vector<const asset::ICPUMeshBuffer*> cpuDeps; cpuDeps.reserve(assetCount);
	for (auto i=0u; i<assetCount; i++)
	{
		auto* _asset = _begin[i];
        for (auto mesh : _asset->getMeshBuffers())
            cpuDeps.emplace_back(mesh);
    }

    core::vector<size_t> redir = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);
    for (size_t i=0u, j=0u; i<assetCount; ++i)
    {
		auto* _asset = _begin[i];
        auto cpuMeshBuffers = _asset->getMeshBuffers();

		auto& output = res->operator[](i);
        output = core::make_smart_refctd_ptr<IGPUMesh>(cpuMeshBuffers.size());
        output->setBoundingBox(_asset->getBoundingBox());

        auto gpuMeshBuffersIt = output->getMeshBufferIterator();
        for (auto mesh : cpuMeshBuffers)
		{
			*(gpuMeshBuffersIt++) = core::smart_refctd_ptr(gpuDeps->operator[](redir[j]));
			++j;
		}
    }

    return res;
}

// TODO: rewrite after GPU polyphase implementation
auto IGPUObjectFromAssetConverter::create(const asset::ICPUImage** const _begin, const asset::ICPUImage** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUImage>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImage> >(assetCount);

    // TODO: This should be the other way round because if a queue supports either compute or graphics but not the other way round
    const uint32_t transferFamIx = _params.perQueue[EQU_TRANSFER].queue->getFamilyIndex();
    const uint32_t computeFamIx = _params.perQueue[EQU_COMPUTE].queue ? _params.perQueue[EQU_COMPUTE].queue->getFamilyIndex() : transferFamIx;

    bool oneQueue = _params.perQueue[EQU_TRANSFER].queue == _params.perQueue[EQU_COMPUTE].queue;

    bool needToGenMips = false;
    
    core::unordered_map<const asset::ICPUImage*, core::smart_refctd_ptr<IGPUBuffer>> img2gpubuf;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        if (cpuimg->getRegions().size() == 0ull)
            continue;

        // TODO: Why isn't this buffer cached and why are we not going through recursive asset creation and getting ICPUBuffer equivalents? 
        //(we can always discard/not cache the GPU Buffers created only for image data upload)
        IGPUBuffer::SCreationParams params = {};
        params.usage = core::bitflag(IGPUBuffer::EUF_TRANSFER_SRC_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
        const auto& cpuimgParams = cpuimg->getCreationParameters();
        params.size = cpuimg->getBuffer()->getSize();

        auto gpubuf = _params.device->createBuffer(std::move(params));
        auto mreqs = gpubuf->getMemoryReqs();
        mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufMem = _params.device->allocate(mreqs, gpubuf.get());

        img2gpubuf.insert({ cpuimg, std::move(gpubuf) });

        const auto format = cpuimg->getCreationParameters().format;
        if (!asset::isIntegerFormat(format) && !asset::isBlockCompressionFormat(format))
            needToGenMips = true;
    }

    bool oneSubmitPerBatch = !needToGenMips || oneQueue;

    auto& transfer_fence = _params.fences[EQU_TRANSFER]; 
    auto cmdbuf_transfer = _params.perQueue[EQU_TRANSFER].cmdbuf;
    auto cmdbuf_compute = _params.perQueue[EQU_COMPUTE].cmdbuf;

    if (img2gpubuf.size())
    {
        transfer_fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));

        // User will call begin on cmdbuf now
        // cmdbuf_transfer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        assert(cmdbuf_transfer && cmdbuf_transfer->getState() == IGPUCommandBuffer::STATE::RECORDING);
        if (oneQueue)
        {
            cmdbuf_compute = cmdbuf_transfer;
        }
        else if (needToGenMips)
        {
            assert(cmdbuf_compute && cmdbuf_compute->getState() == IGPUCommandBuffer::STATE::RECORDING);
            // User will call begin on cmdbuf now
            // cmdbuf_compute->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        }
    }

    auto needToCompMipsForThisImg = [](const asset::ICPUImage* img) -> bool
    {
        if (img->getRegions().empty())
            return false;
        auto format = img->getCreationParameters().format;
        if (asset::isIntegerFormat(format) || asset::isBlockCompressionFormat(format))
            return false;
        // its enough to define a single mipmap region above the base level to prevent automatic computation
        for (auto& region : img->getRegions())
        if (region.imageSubresource.mipLevel)
            return false;
        return true;
    };
    
    IQueue::SSubmitInfo submit_transfer;
    {
        submit_transfer.commandBufferCount = 1u;
        submit_transfer.commandBuffers = &cmdbuf_transfer.get();
        // buffer and image written and copied to are fresh (or in the case of streaming buffer, at least fenced before freeing), no need to wait for anything external
        submit_transfer.waitSemaphoreCount = 0u;
        submit_transfer.pWaitSemaphores = nullptr;
        submit_transfer.pWaitDstStageMask = nullptr;
    }
    auto cmdComputeMip = [&](const asset::ICPUImage* cpuimg, IGPUImage* gpuimg, IGPUImage::LAYOUT newLayout) -> void
    {
        const auto& realParams = gpuimg->getCreationParameters();
        // TODO when we have compute shader mips generation:
        /*computeCmdbuf->bindPipeline();
        computeCmdbuf->bindDescriptorSets();
        computeCmdbuf->pushConstants();
        computeCmdbuf->dispatch();*/

        IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
        decltype(info)::image_barrier_t barrier = {};
        info.imgBarriers = &barrier;
        info.imgBarrierCount = 1u;

        barrier.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE;
        barrier.barrier.otherQueueFamilyIndex = cmdbuf_compute->getQueueFamilyIndex();
        barrier.image = gpuimg;
        // TODO this is probably wrong (especially in case of depth/stencil formats), but i think i can leave it like this since we'll never have any depth/stencil images loaded (right?)
        barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // not hardcode
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.layerCount = realParams.arrayLayers;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;

        IGPUCommandBuffer::SImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.srcSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.srcSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.srcOffsets[0] = { 0, 0, 0 };

        blitRegion.dstSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.dstSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.dstSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.dstOffsets[0] = { 0, 0, 0 };

        // Compute mips
        auto mipsize = cpuimg->getMipSize(cpuimg->getCreationParameters().mipLevels);
        uint32_t mipWidth = mipsize.x;
        uint32_t mipHeight = mipsize.y;
        uint32_t mipDepth = mipsize.z;
        for (uint32_t i = cpuimg->getCreationParameters().mipLevels; i < gpuimg->getCreationParameters().mipLevels; ++i)
        {
            const uint32_t srcLoD = i - 1u;

            // TODO: with compute blit these will have to be COMPUTE 2 COMPUTE barriers from DST to newLayout (with an intermediate transition to GENERAL for storage)
            barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
            barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
            barrier.newLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = srcLoD;
            barrier.subresourceRange.levelCount = 1u;

            cmdbuf_transfer->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT, asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
                static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u, &barrier);

            const auto srcMipSz = cpuimg->getMipSize(srcLoD);

            blitRegion.srcSubresource.mipLevel = srcLoD;
            blitRegion.srcOffsets[1] = { srcMipSz.x, srcMipSz.y, srcMipSz.z };

            blitRegion.dstSubresource.mipLevel = i;
            blitRegion.dstOffsets[1] = { mipWidth, mipHeight, mipDepth };

            // TODO: Remove the requirement that the transfer queue has graphics caps,
            cmdbuf_transfer->blitImage(gpuimg, IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL, gpuimg,
                IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, 1u, &blitRegion, asset::ISampler::ETF_LINEAR);

            if (mipWidth > 1u) mipWidth /= 2u;
            if (mipHeight > 1u) mipHeight /= 2u;
            if (mipDepth > 1u) mipDepth /= 2u;
        }
        
        barrier.subresourceRange.baseMipLevel = cpuimg->getCreationParameters().mipLevels - 1u;
        barrier.subresourceRange.levelCount = gpuimg->getCreationParameters().mipLevels - 1u;
        barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
        barrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = newLayout;

        cmdbuf_transfer->pipelineBarrier(
            asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
            finalStageMask,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            1u, &barrier);

        // Transition the last mip level to correct layout
        barrier.subresourceRange.baseMipLevel = gpuimg->getCreationParameters().mipLevels - 1u;
        barrier.subresourceRange.levelCount = 1u;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
        cmdbuf_transfer->pipelineBarrier(
            asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
            finalStageMask,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            1u, &barrier);
    };

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        IGPUImage::SCreationParams params = {};
        params = cpuimg->getCreationParameters();
        
        IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
        promotionRequest.originalFormat = params.format;
        promotionRequest.usages = {};

        // override the mip-count if its not an integer format and there was no mip-pyramid specified 
        if (params.mipLevels==1u && !asset::isIntegerFormat(params.format))
            params.mipLevels = 1u + static_cast<uint32_t>(std::log2(static_cast<float>(core::max<uint32_t>(core::max<uint32_t>(params.extent.width, params.extent.height), params.extent.depth))));

        if (cpuimg->getRegions().size())
            params.usage |= asset::IImage::EUF_TRANSFER_DST_BIT;
        
        const bool computeMips = needToCompMipsForThisImg(cpuimg);
        if (computeMips)
        {
            params.usage |= asset::IImage::EUF_TRANSFER_SRC_BIT; // this is for blit
            // I'm already adding usage flags for mip-mapping compute shader
            params.usage |= asset::IImage::EUF_SAMPLED_BIT; // to read source mips
            // but we don't add the STORAGE USAGE, yet
            // TODO: will change when we do the blit on compute shader.
            promotionRequest.usages.blitDst = true;
            promotionRequest.usages.blitSrc = true;
        }
        
        auto physDev = _params.device->getPhysicalDevice();
        promotionRequest.usages = promotionRequest.usages | params.usage;
        auto newFormat = physDev->promoteImageFormat(promotionRequest, IGPUImage::TILING::OPTIMAL);
        auto newFormatIsStorable = physDev->getImageFormatUsagesOptimalTiling()[newFormat].storageImage;
        
        // If Format Promotion failed try the same usages but with linear tiling.
        if (newFormat == asset::EF_UNKNOWN)
        {
            newFormat = physDev->promoteImageFormat(promotionRequest, IGPUImage::TILING::LINEAR);
            newFormatIsStorable = physDev->getImageFormatUsagesLinearTiling()[newFormat].storageImage;
            params.tiling = IGPUImage::TILING::LINEAR;
        }

        assert(newFormat != asset::EF_UNKNOWN); // No feasible supported format found for creating this image
        params.format = newFormat;

        // now add the STORAGE USAGE
        if (computeMips)
        {
            // formats like SRGB etc. can't be stored to
            params.usage |= asset::IImage::EUF_STORAGE_BIT;
            // but image views with formats that are store-able can be created
            if (!newFormatIsStorable)
            {
                params.flags |= asset::IImage::ECF_MUTABLE_FORMAT_BIT;
                params.flags |= asset::IImage::ECF_EXTENDED_USAGE_BIT;
                if (asset::isBlockCompressionFormat(newFormat))
                    params.flags |= asset::IImage::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
            }
        }

        auto gpuimg = _params.device->createImage(std::move(params));
        auto gpuimgMemReqs = gpuimg->getMemoryReqs();
        gpuimgMemReqs.memoryTypeBits &= physDev->getDeviceLocalMemoryTypeBits();
        auto gpuimgMem = _params.device->allocate(gpuimgMemReqs, gpuimg.get());

		res->operator[](i) = std::move(gpuimg);
    }

    if (img2gpubuf.size() == 0ull)
        return res;

    auto it = _begin;
    auto doBatch = [&]() -> void
    {
        constexpr uint32_t pipeliningDepth = 8u;

        IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
        decltype(info)::image_barrier_t imgbarriers[pipeliningDepth];
        info.imgBarriers = imgbarriers;

        const uint32_t n = it - _begin;

        auto oldIt = it;
        for (uint32_t i = 0u; i < pipeliningDepth && it != _end; ++i)
        {
            auto* cpuimg = *(it++);
            auto* gpuimg = (*res)[n+i].get();
            // Creation of a GPU image (on Vulkan) can fail for several reasons, one of them, for
            // example is unsupported formats
            if (!gpuimg)
                continue;

            // There should be no pipeline barrier before this, because the first usage implicitly acquires the resource for the queue!
            submit_transfer = _params.utilities->updateImageViaStagingBuffer(
                cpuimg->getBuffer(), cpuimg->getCreationParameters().format, gpuimg, IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, cpuimg->getRegions(),
                _params.perQueue[EQU_TRANSFER].queue, transfer_fence.get(), submit_transfer
            );
            
            IGPUImage::LAYOUT newLayout;
            const auto& realParams = gpuimg->getCreationParameters();
            if (realParams.usage.hasFlags(asset::IImage::EUF_SAMPLED_BIT))
                newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
            else
                newLayout = IGPUImage::LAYOUT::GENERAL;

            if (needToCompMipsForThisImg(cpuimg))
            {
                assert(_params.device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[realParams.format].sampledImage);
                assert(asset::isFloatingPointFormat(realParams.format) || asset::isNormalizedFormat(realParams.format));
                cmdComputeMip(cpuimg, gpuimg, newLayout);
            }
            else
            {
                auto& b = imgbarriers[info.imgBarrierCount++] = {};
                b.barrier.dep.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
                b.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT;
                if (ownershipTfer)
                {
                    b.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE;
                    b.barrier.otherQueueFamilyIndex = computeFamIx;
                }
                else
                {
                    b.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
                    b.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS|asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
                }
                b.image = gpuimg;
                if (asset::isDepthOrStencilFormat(realParams.format))
                {
                    if (!asset::isStencilOnlyFormat(realParams.format))
                        b.subresourceRange.aspectMask |= IGPUImage::EAF_DEPTH_BIT;
                    if (!asset::isDepthOnlyFormat(realParams.format))
                        b.subresourceRange.aspectMask |= IGPUImage::EAF_STENCIL_BIT;
                }
                else
                    b.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
                b.subresourceRange.levelCount = realParams.mipLevels;
                b.subresourceRange.layerCount = realParams.arrayLayers;
                b.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
                b.newLayout = newLayout;
            }
        }

        // ownership transition release or just a barrier
        cmdbuf_transfer->pipelineBarrier(asset::EDF_NONE,info);

        if (transferFamIx!=computeFamIx && cmdbuf_compute) // need to do ownership transition
        {
            // ownership transition acquire
            for (auto j=0u; j<info.imgBarrierCount; j++)
            {
                auto& b = imgbarriers[j];
                b.barrier.dep = {};
                b.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
                b.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS|asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
                b.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE;
                b.barrier.otherQueueFamilyIndex = transferFamIx;
            }
            cmdbuf_transfer->pipelineBarrier(asset::EDF_NONE,info);
        }

        cmdbuf_transfer->end();
        if (needToGenMips && !oneQueue)
            cmdbuf_compute->end();

        auto transfer_sem = _params.device->createSemaphore();
        auto* transfer_sem_ptr = transfer_sem.get();

        auto batch_final_fence = transfer_fence;

        submit_transfer.signalSemaphoreCount = 1u;
        submit_transfer.pSignalSemaphores = &transfer_sem_ptr;
        _params.perQueue[EQU_TRANSFER].queue->submit(1u, &submit_transfer, batch_final_fence.get());

        if (_params.perQueue[EQU_TRANSFER].semaphore)
            _params.perQueue[EQU_TRANSFER].semaphore[0] = transfer_sem;
        if (_params.perQueue[EQU_COMPUTE].semaphore && oneQueue && needToGenMips)
            _params.perQueue[EQU_COMPUTE].semaphore[0] = transfer_sem;

         // must be outside `if` scope to not get deleted after `batch_final_fence = compute_fence_ptr;` assignment
        auto & compute_fence = _params.fences[EQU_COMPUTE];
        if (!oneSubmitPerBatch)
        {
            core::smart_refctd_ptr<IGPUSemaphore> compute_sem;
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                compute_sem = _params.device->createSemaphore();
            auto* compute_sem_ptr = compute_sem.get();
            compute_fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* compute_fence_ptr = compute_fence.get();

            const auto dstWait = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            auto* cb_comp = cmdbuf_compute.get();
            IQueue::SSubmitInfo submit_compute;
            submit_compute.commandBufferCount = 1u;
            submit_compute.commandBuffers = &cb_comp;
            submit_compute.waitSemaphoreCount = 1u;
            submit_compute.pWaitDstStageMask = &dstWait;
            submit_compute.pWaitSemaphores = &transfer_sem_ptr;
            submit_compute.signalSemaphoreCount = compute_sem?1u:0u;
            submit_compute.pSignalSemaphores = compute_sem?&compute_sem_ptr:nullptr;
            _params.perQueue[EQU_COMPUTE].queue->submit(1u, &submit_compute, compute_fence_ptr);

            if (_params.perQueue[EQU_COMPUTE].semaphore)
                _params.perQueue[EQU_COMPUTE].semaphore[0] = compute_sem;

#if 0 //TODO: (!) enable when mips are in fact computed on `cmdbuf_compute` (currently they are done with blits on cmdbuf_transfer)
            batch_final_fence = compute_fence;
#endif
        }

        // wait to finish all batch work in order to safely reset command buffers
        _params.device->waitForFences(1u, &batch_final_fence.get(), false, 9999999999ull);

        if (!oneSubmitPerBatch)
            _params.device->waitForFences(1u, &compute_fence.get(), false, 9999999999ull);

        // separate cmdbufs per batch instead?
        if (it != _end)
        {
            cmdbuf_transfer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
            cmdbuf_transfer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            _params.device->resetFences(1u, &transfer_fence.get());
            
            if (!oneSubmitPerBatch)
            {
                cmdbuf_compute->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
                cmdbuf_compute->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
                _params.device->resetFences(1u, &compute_fence.get());
            }
        }
    };

    while (it != _end)
    {
        doBatch();
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUSampler> IGPUObjectFromAssetConverter::create(const asset::ICPUSampler** const _begin, const asset::ICPUSampler** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSampler> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUSampler* cpusmplr = _begin[i];
        res->operator[](i) = _params.device->createSampler(cpusmplr->getParams());
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUPipelineLayout> IGPUObjectFromAssetConverter::create(const asset::ICPUPipelineLayout** const _begin, const asset::ICPUPipelineLayout** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUPipelineLayout> >(assetCount);

    // TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
    // SOLUTION: a `creationCache` object as the last parameter to the `create` function
    core::vector<const asset::ICPUDescriptorSetLayout*> cpuDSLayouts;
    cpuDSLayouts.reserve(assetCount * asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT);

    for (const asset::ICPUPipelineLayout* pl : core::SRange<const asset::ICPUPipelineLayout*>(_begin, _end))
    {
        for (uint32_t ds = 0u; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
            if (pl->getDescriptorSetLayout(ds))
                cpuDSLayouts.push_back(pl->getDescriptorSetLayout(ds));
    }
    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDSLayouts);

    auto gpuDSLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuDSLayouts.data(), cpuDSLayouts.data() + cpuDSLayouts.size(), _params);

    size_t dslIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUPipelineLayout* cpupl = _begin[i];
        IGPUDescriptorSetLayout* dsLayouts[asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (size_t ds = 0ull; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
        {
            if (cpupl->getDescriptorSetLayout(ds))
                dsLayouts[ds] = (*gpuDSLayouts)[redirs[dslIter++]].get();
        }
        res->operator[](i) = _params.device->createPipelineLayout(
            cpupl->getPushConstantRanges().begin(), cpupl->getPushConstantRanges().end(),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[0]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[1]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[2]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[3])
        );
    }

    return res;
}

inline created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IGPUObjectFromAssetConverter::create(const asset::ICPURenderpassIndependentPipeline** const _begin, const asset::ICPURenderpassIndependentPipeline** const _end, SParams& _params)
{
    constexpr size_t GRAPHICS_SHADER_STAGE_COUNT = asset::ICPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT;

    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> >(assetCount);

    core::vector<const asset::ICPUPipelineLayout*> cpuLayouts;
    cpuLayouts.reserve(assetCount);
    core::vector<const asset::ICPUSpecializedShader*> cpuShaders;
    cpuShaders.reserve(assetCount * GRAPHICS_SHADER_STAGE_COUNT);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];
        cpuLayouts.push_back(cpuppln->getLayout());

        for (size_t s = 0ull; s < GRAPHICS_SHADER_STAGE_COUNT; ++s)
            if (const asset::ICPUSpecializedShader* shdr = cpuppln->getShaderAtIndex(s))
                cpuShaders.push_back(shdr);
    }

    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);

    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data() + cpuLayouts.size(), _params);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data() + cpuShaders.size(), _params);

    size_t shdrIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];

        IGPUPipelineLayout* layout = (*gpuLayouts)[layoutRedirs[i]].get();

        IGPUSpecializedShader* shaders[GRAPHICS_SHADER_STAGE_COUNT]{};
        size_t local_shdr_count = 0ull;
        for (size_t s = 0ull; s < GRAPHICS_SHADER_STAGE_COUNT; ++s)
            if (cpuppln->getShaderAtIndex(s))
                shaders[local_shdr_count++] = (*gpuShaders)[shdrRedirs[shdrIter++]].get();

        (*res)[i] = _params.device->createRenderpassIndependentPipeline(
            _params.pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>(layout),
            shaders, shaders + local_shdr_count,
            cpuppln->getVertexInputParams(),
            cpuppln->getBlendParams(),
            cpuppln->getPrimitiveAssemblyParams(),
            cpuppln->getRasterizationParams()
        );
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUDescriptorSetLayout** const _begin, const asset::ICPUDescriptorSetLayout** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUDescriptorSetLayout>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSetLayout> >(assetCount);

    // This is a descriptor set layout function, we only care about immutable samplers here.
    core::vector<const asset::ICPUSampler*> cpuSamplers;
    size_t maxSamplers = 0ull;
    size_t maxBindingsPerLayout = 0ull;
    size_t maxSamplersPerLayout = 0ull;
    for (auto dsl : core::SRange<const asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        const auto samplerCount = dsl->getImmutableSamplerRedirect().getTotalCount();
        maxSamplers += samplerCount;

        maxBindingsPerLayout = core::max<size_t>(maxBindingsPerLayout, dsl->getTotalBindingCount());
        maxSamplersPerLayout = core::max<size_t>(maxSamplersPerLayout, samplerCount);
    }
    cpuSamplers.reserve(maxSamplers);

    for (auto dsl : core::SRange<const asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        const auto& samplers = dsl->getImmutableSamplers();
        if (!samplers.empty())
        {
            for (auto& sampler : samplers)
                cpuSamplers.push_back(sampler.get());
        }
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data() + cpuSamplers.size(), _params);
    size_t gpuSmplrIter = 0ull;

    using gpu_bindings_array_t = core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>;
    auto tmpBindings = core::make_refctd_dynamic_array<gpu_bindings_array_t>(maxBindingsPerLayout);

    using samplers_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUSampler>>;
    auto tmpSamplers = core::make_refctd_dynamic_array<samplers_array_t>(maxSamplersPerLayout);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        core::smart_refctd_ptr<IGPUSampler>* smplr_ptr = tmpSamplers->data();
        const asset::ICPUDescriptorSetLayout* cpudsl = _begin[i];

        size_t gpuBindingCount = 0ull;
        const auto& samplerBindingRedirect = cpudsl->getImmutableSamplerRedirect();
        const auto samplerBindingCount = samplerBindingRedirect.getBindingCount();
        for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
            const auto& descriptorBindingRedirect = cpudsl->getDescriptorRedirect(type);
            const auto declaredBindingCount = descriptorBindingRedirect.getBindingCount();

            for (uint32_t b = 0; b < declaredBindingCount; ++b)
            {
                auto& gpuBinding = tmpBindings->begin()[gpuBindingCount++];
                gpuBinding.binding = descriptorBindingRedirect.getBinding(b).data;
                gpuBinding.type = type;
                gpuBinding.count = descriptorBindingRedirect.getCount(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ b });
                gpuBinding.stageFlags = descriptorBindingRedirect.getStageFlags(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ b });
                gpuBinding.samplers = nullptr;

                // If this DS layout has any immutable samplers..
                if ((gpuBinding.type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER) && (samplerBindingCount > 0))
                {
                    // If this binding number has any immutable samplers..
                    if (samplerBindingRedirect.findBindingStorageIndex(asset::ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ gpuBinding.binding }).data == samplerBindingRedirect.Invalid)
                        continue;

                    // Copy in tmpSamplers.
                    for (uint32_t s = 0; s < gpuBinding.count; ++s)
                        smplr_ptr[s] = (*gpuSamplers)[redirs[gpuSmplrIter++]];

                    gpuBinding.samplers = smplr_ptr;
                    smplr_ptr += gpuBinding.count;
                }
            }
        }
        assert(gpuBindingCount == cpudsl->getTotalBindingCount());

        (*res)[i] = _params.device->createDescriptorSetLayout(tmpBindings->begin(), tmpBindings->begin() + gpuBindingCount);
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUComputePipeline> IGPUObjectFromAssetConverter::create(const asset::ICPUComputePipeline** const _begin, const asset::ICPUComputePipeline** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUComputePipeline> >(assetCount);

    core::vector<const asset::ICPUPipelineLayout*> cpuLayouts;
    core::vector<const asset::ICPUSpecializedShader*> cpuShaders;
    cpuLayouts.reserve(res->size());
    cpuShaders.reserve(res->size());

    const asset::ICPUComputePipeline** it = _begin;
    while (it != _end)
    {
        cpuShaders.push_back((*it)->getShader());
        cpuLayouts.push_back((*it)->getLayout());
        ++it;
    }

    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);
    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data() + cpuShaders.size(), _params);
    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data() + cpuLayouts.size(), _params);

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        auto layout = (*gpuLayouts)[layoutRedirs[i]];
        auto shdr = (*gpuShaders)[shdrRedirs[i]];
        (*res)[i] = _params.device->createComputePipeline(_params.pipelineCache, std::move(layout), std::move(shdr));
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUSpecializedShader** const _begin, const asset::ICPUSpecializedShader** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUSpecializedShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSpecializedShader> >(assetCount);

    core::vector<const asset::ICPUShader*> cpuDeps;
    cpuDeps.reserve(res->size());

    const asset::ICPUSpecializedShader** it = _begin;
    while (it != _end)
    {
        cpuDeps.push_back((*it)->getUnspecialized());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUShader>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        auto unspecShader = gpuDeps->operator[](redirs[i]);
        if (unspecShader)
            res->operator[](i) = _params.device->createSpecializedShader(unspecShader.get(), _begin[i]->getSpecializationInfo());
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUShader** const _begin, const asset::ICPUShader** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUShader> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        res->operator[](i) = _params.device->createShader(core::smart_refctd_ptr<asset::ICPUShader>(const_cast<asset::ICPUShader*>(_begin[i])));
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUBufferView** const _begin, const asset::ICPUBufferView** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUBufferView>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBufferView> >(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBufs(assetCount, nullptr);
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
        cpuBufs[i] = _begin[i]->getUnderlyingBuffer();

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufs);
    auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufs.data(), cpuBufs.data()+cpuBufs.size(), _params);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUBufferView* cpubufview = _begin[i];
        IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[i]].get();
        (*res)[i] = _params.device->createBufferView(gpubuf->getBuffer(), cpubufview->getFormat(), gpubuf->getOffset() + cpubufview->getOffsetInBuffer(), cpubufview->getByteSize());;
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUImageView> IGPUObjectFromAssetConverter::create(const asset::ICPUImageView** const _begin, const asset::ICPUImageView** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImageView> >(assetCount);

    core::vector<asset::ICPUImage*> cpuDeps;
    cpuDeps.reserve(res->size());

    const asset::ICPUImageView** it = _begin;
    while (it != _end)
    {
        cpuDeps.push_back((*it)->getCreationParameters().image.get());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);

    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUImage>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);
    const auto physDev = _params.device->getPhysicalDevice();
    const auto& optimalUsages = physDev->getImageFormatUsagesOptimalTiling();
    const auto& linearUsages = physDev->getImageFormatUsagesLinearTiling();
    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        if (gpuDeps->begin()[redirs[i]])
        {
            const auto& cpuParams = _begin[i]->getCreationParameters();

            IGPUImageView::SCreationParams params = {};
            params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(cpuParams.flags);
            params.viewType = static_cast<IGPUImageView::E_TYPE>(cpuParams.viewType);
            params.image = (*gpuDeps)[redirs[i]];
            const auto& gpuImgParams = params.image->getCreationParameters();
            // override the view's format if the source image got promoted, a bit crude, but don't want to scratch my head about how to promote the views and guess semantics
            const bool formatGotPromoted = cpuParams.format!=gpuImgParams.format;
            params.format = formatGotPromoted ? gpuImgParams.format:cpuParams.format;
            params.subUsages = cpuParams.subUsages;
            // TODO: In Asset Converter 2.0 we'd pass through all descriptor sets etc and propagate the adding usages backwards to views, but here we need to trim the image's usages instead
            {
                IPhysicalDevice::SFormatImageUsages::SUsage validUsages(gpuImgParams.usage);
                if (params.image->getTiling()!=IGPUImage::TILING::LINEAR)
                    validUsages = validUsages & optimalUsages[params.format];
                else
                    validUsages = validUsages & linearUsages[params.format];
                // add them after trimming
                if (validUsages.sampledImage)
                    params.subUsages |= IGPUImage::EUF_SAMPLED_BIT;
                if (validUsages.storageImage)
                    params.subUsages |= IGPUImage::EUF_STORAGE_BIT;
                if (validUsages.attachment)
                    params.subUsages |= IGPUImage::EUF_RENDER_ATTACHMENT_BIT;
                if (validUsages.transferSrc)
                    params.subUsages |= IGPUImage::EUF_TRANSFER_SRC_BIT;
                if (validUsages.transferDst)
                    params.subUsages |= IGPUImage::EUF_TRANSFER_DST_BIT;
                // stuff thats not dependent on device caps
                const auto uncappedUsages = IGPUImage::EUF_TRANSIENT_ATTACHMENT_BIT|IGPUImage::EUF_INPUT_ATTACHMENT_BIT|IGPUImage::EUF_SHADING_RATE_ATTACHMENT_BIT|IGPUImage::EUF_FRAGMENT_DENSITY_MAP_BIT;
                params.subUsages |= gpuImgParams.usage&uncappedUsages;
            }
            memcpy(&params.components, &cpuParams.components, sizeof(params.components));
            params.subresourceRange = cpuParams.subresourceRange;
            // TODO: Undo this, make all loaders set the level and layer counts on image views to `ICPUImageView::remaining_...`
            params.subresourceRange.levelCount = gpuImgParams.mipLevels-params.subresourceRange.baseMipLevel;
            (*res)[i] = _params.device->createImageView(std::move(params));
        }
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUDescriptorSet> IGPUObjectFromAssetConverter::create(const asset::ICPUDescriptorSet** const _begin, const asset::ICPUDescriptorSet** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSet> >(assetCount);

    struct BindingDescTypePair_t{
        uint32_t binding;
        asset::IDescriptor::E_TYPE descType;
        size_t count;
    };
    auto isBufferDesc = [](asset::IDescriptor::E_TYPE t) {
        using namespace asset;
        switch (t) {
        case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER: [[fallthrough]];
        case IDescriptor::E_TYPE::ET_STORAGE_BUFFER: [[fallthrough]];
        case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC: [[fallthrough]];
        case IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
            return true;
            break;
        default:
            return false;
            break;
        }
    };
    auto isBufviewDesc = [](asset::IDescriptor::E_TYPE t) {
        using namespace asset;
        return t==IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER || t==IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
    };
    auto isSampledImgViewDesc = [](asset::IDescriptor::E_TYPE t) {
        return t==asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
    };
    auto isStorageImgDesc = [](asset::IDescriptor::E_TYPE t) {
        return t==asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
    };

	// TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
	core::vector<const asset::ICPUDescriptorSetLayout*> cpuLayouts;
	cpuLayouts.reserve(assetCount);
	uint32_t maxWriteCount = 0ull;
	uint32_t descCount = 0ull;
	uint32_t bufCount = 0ull;
	uint32_t bufviewCount = 0ull;
	uint32_t sampledImgViewCount = 0ull;
	uint32_t storageImgViewCount = 0ull;
    for (ptrdiff_t i=0u; i<assetCount; i++)
    {
        const asset::ICPUDescriptorSet* cpuds = _begin[i];
		cpuLayouts.push_back(cpuds->getLayout());

        for (uint32_t t = 0u; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
            
            // Since one binding can have multiple descriptors which will all be updated with a single SWriteDescriptorSet,
            // we add the binding count here not the descriptor count.
            maxWriteCount += cpuds->getLayout()->getDescriptorRedirect(type).getBindingCount();

            const auto count = cpuds->getLayout()->getTotalDescriptorCount(type);
            descCount += count;

            if (isBufferDesc(type))
                bufCount += count;
            else if (isBufviewDesc(type))
                bufCount += count;
            else if (isSampledImgViewDesc(type))
                sampledImgViewCount += count;
            else if (isStorageImgDesc(type))
                storageImgViewCount += count;
        }
    }
	
    core::vector<asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(bufCount);
    core::vector<asset::ICPUBufferView*> cpuBufviews;
    cpuBufviews.reserve(bufviewCount);
    core::vector<asset::ICPUImageView*> cpuImgViews;
    cpuImgViews.reserve(storageImgViewCount+sampledImgViewCount);
    core::vector<asset::ICPUSampler*> cpuSamplers;
    cpuSamplers.reserve(sampledImgViewCount);
    for (ptrdiff_t i=0u; i<assetCount; i++)
    {
        const asset::ICPUDescriptorSet* cpuds = _begin[i];

        for (uint32_t t = 0u; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
            if (cpuds->getDescriptorInfoStorage(type).empty())
                continue;

            for (uint32_t d = 0u; d < cpuds->getLayout()->getTotalDescriptorCount(type); ++d)
            {
                auto* info = cpuds->getDescriptorInfoStorage(type).begin() + d;
                auto descriptor = info->desc.get();
                if (isBufferDesc(type))
                {
                    auto cpuBuf = static_cast<asset::ICPUBuffer*>(descriptor);
                    if (type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER || type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
                        cpuBuf->addUsageFlags(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT);
                    else if (type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER || type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
                        cpuBuf->addUsageFlags(asset::IBuffer::EUF_STORAGE_BUFFER_BIT);
                    cpuBuffers.push_back(cpuBuf);
                }
                else if (isBufviewDesc(type))
                {
                    auto cpuBufView = static_cast<asset::ICPUBufferView*>(descriptor);
                    auto cpuBuf = cpuBufView->getUnderlyingBuffer();
                    if (cpuBuf && type == asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)
                        cpuBuf->addUsageFlags(asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT);
                    else if (cpuBuf && type == asset::IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER)
                        cpuBuf->addUsageFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT);
                    cpuBufviews.push_back(cpuBufView);
                }
                else if (isSampledImgViewDesc(type))
                {
                    auto cpuImgView = static_cast<asset::ICPUImageView*>(descriptor);
                    auto cpuImg = cpuImgView->getCreationParameters().image;
                    if (cpuImg)
                        cpuImg->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);
                    cpuImgViews.push_back(cpuImgView);
                    if (info->info.image.sampler)
                        cpuSamplers.push_back(info->info.image.sampler.get());
                }
                else if (isStorageImgDesc(type))
                {
                    auto cpuImgView = static_cast<asset::ICPUImageView*>(descriptor);
                    auto cpuImg = cpuImgView->getCreationParameters().image;
                    if (cpuImg)
                        cpuImg->addImageUsageFlags(asset::IImage::EUF_STORAGE_BIT);
                    cpuImgViews.push_back(cpuImgView);
                }
            }
        }
    }

	using redirs_t = core::vector<size_t>;
	redirs_t layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
	redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
	redirs_t bufviewRedirs = eliminateDuplicatesAndGenRedirs(cpuBufviews);
	redirs_t imgViewRedirs = eliminateDuplicatesAndGenRedirs(cpuImgViews);
	redirs_t smplrRedirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);

	auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size(), _params);
    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size(), _params);
    auto gpuBufviews = getGPUObjectsFromAssets<asset::ICPUBufferView>(cpuBufviews.data(), cpuBufviews.data()+cpuBufviews.size(), _params);
    auto gpuImgViews = getGPUObjectsFromAssets<asset::ICPUImageView>(cpuImgViews.data(), cpuImgViews.data()+cpuImgViews.size(), _params);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size(), _params);

    uint32_t dsCounts[] = {static_cast<uint32_t>(assetCount)};
    auto dsPool = _params.device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&gpuLayouts->begin()->get(),&gpuLayouts->end()->get(), dsCounts);

	core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes(maxWriteCount);
	auto write_it = writes.begin();
	core::vector<IGPUDescriptorSet::SDescriptorInfo> descInfos(descCount);
	{
		auto info = descInfos.begin();
		//iterators
		uint32_t bi = 0u, bvi = 0u, ivi = 0u, si = 0u;
		for (ptrdiff_t i = 0u; i < assetCount; i++)
		{
			IGPUDescriptorSetLayout* gpulayout = gpuLayouts->operator[](layoutRedirs[i]).get();
			res->operator[](i) = dsPool->createDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(gpulayout));
			auto gpuds = res->operator[](i).get();

            const asset::ICPUDescriptorSet* cpuds = _begin[i];

            for (uint32_t t = 0u; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
            {
                const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);

                const auto& descriptorBindingRedirect = cpuds->getLayout()->getDescriptorRedirect(type);
                const auto& mutableSamplerBindingRedirect = cpuds->getLayout()->getMutableSamplerRedirect();

                for (uint32_t b = 0u; b < descriptorBindingRedirect.getBindingCount(); ++b)
                {
                    write_it->dstSet = gpuds;
                    write_it->binding = descriptorBindingRedirect.getBinding(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ b }).data;
                    write_it->arrayElement = 0u;

                    const uint32_t descriptorCount = descriptorBindingRedirect.getCount(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ b });
                    write_it->count = descriptorCount;
                    write_it->descriptorType = type;
                    write_it->info = &(*info);

                    const uint32_t offset = descriptorBindingRedirect.getStorageOffset(asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ b }).data;

                    // It is better to use getDescriptorInfoStorage over getDescriptorInfos, because the latter does a binary search
                    // over the bindings, which is not really required given we have the index of binding number (since we're iterating
                    // over all the declared bindings).
                    auto descriptorInfos = cpuds->getDescriptorInfoStorage(type);

                    // Iterate through each descriptor in this binding to fill the info structs
                    bool allDescriptorsPresent = true;
                    for (uint32_t d = 0u; d < descriptorCount; ++d)
                    {
                        if (isBufferDesc(type))
                        {
                            core::smart_refctd_ptr<IGPUOffsetBufferPair> buffer = bufRedirs[bi] >= gpuBuffers->size() ? nullptr : gpuBuffers->operator[](bufRedirs[bi]);
                            if (buffer)
                            {
                                info->desc = core::smart_refctd_ptr<IGPUBuffer>(buffer->getBuffer());
                                info->info.buffer.offset = descriptorInfos.begin()[offset+d].info.buffer.offset + buffer->getOffset();
                                info->info.buffer.size = descriptorInfos.begin()[offset+d].info.buffer.size;
                            }
                            else
                            {
                                info->desc = nullptr;
                                info->info.buffer.offset = 0u;
                                info->info.buffer.size = 0u;
                            }
                            ++bi;
                        }
                        else if (isBufviewDesc(type))
                        {
                            info->desc = bufviewRedirs[bvi] >= gpuBufviews->size() ? nullptr : gpuBufviews->operator[](bufviewRedirs[bvi]);
                            ++bvi;
                        }
                        else if (isSampledImgViewDesc(type) || isStorageImgDesc(type))
                        {
                            info->desc = imgViewRedirs[ivi] >= gpuImgViews->size() ? nullptr : gpuImgViews->operator[](imgViewRedirs[ivi]);
                            ++ivi;
                            info->info.image.imageLayout = descriptorInfos[offset + d].info.image.imageLayout;
                            assert(info->info.image.imageLayout != asset::IImage::EL_UNDEFINED);

                            if (!isStorageImgDesc(type))
                            {
                                const bool isMutableSamplerBinding = (mutableSamplerBindingRedirect.findBindingStorageIndex(asset::ICPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ write_it->binding }).data != mutableSamplerBindingRedirect.Invalid);
                                if (isMutableSamplerBinding)
                                {
                                    assert(descriptorInfos.begin()[offset + d].info.image.sampler);
                                    info->info.image.sampler = gpuSamplers->operator[](smplrRedirs[si++]);
                                }
                            }
                        }
                        allDescriptorsPresent = allDescriptorsPresent && info->desc;
                        info++;
                    }

                    if (allDescriptorsPresent)
                        write_it++;
                }
            }
		}
	}

    _params.device->updateDescriptorSets(write_it - writes.begin(), writes.data(), 0u, nullptr);

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUAnimationLibrary** _begin, const asset::ICPUAnimationLibrary** _end, SParams& _params) -> created_gpu_object_array<asset::ICPUAnimationLibrary>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUAnimationLibrary> >(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount*3u);

    for (ptrdiff_t i=0u; i<assetCount; i++)
    {
        const asset::ICPUAnimationLibrary* cpuanim = _begin[i];
        cpuanim->getKeyframeStorageBinding().buffer->addUsageFlags(IGPUBuffer::EUF_STORAGE_BUFFER_BIT);
        cpuanim->getTimestampStorageBinding().buffer->addUsageFlags(IGPUBuffer::EUF_STORAGE_BUFFER_BIT);
        cpuBuffers.push_back(cpuanim->getKeyframeStorageBinding().buffer.get());
        cpuBuffers.push_back(cpuanim->getTimestampStorageBinding().buffer.get());
        if (const asset::ICPUBuffer* buf = cpuanim->getAnimationStorageRange().buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size(), _params);

    size_t bufIter = 0ull;
    for (ptrdiff_t i = 0u; i<assetCount; ++i)
    {
        const asset::IAnimationLibrary<asset::ICPUBuffer>* cpuanim = _begin[i];

		asset::SBufferBinding<IGPUBuffer> keyframeBinding,timestampBinding;
        {
            keyframeBinding.offset = cpuanim->getKeyframeStorageBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            keyframeBinding.offset += gpubuf->getOffset();
            keyframeBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        {
            timestampBinding.offset = cpuanim->getTimestampStorageBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            timestampBinding.offset += gpubuf->getOffset();
            timestampBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
		asset::SBufferRange<IGPUBuffer> animationRange;
        if (cpuanim->getAnimationStorageRange().buffer)
        {
            animationRange.offset = cpuanim->getAnimationStorageRange().offset;
            animationRange.size = cpuanim->getAnimationStorageRange().size;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            animationRange.offset += gpubuf->getOffset();
            animationRange.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        (*res)[i] = core::make_smart_refctd_ptr<IGPUAnimationLibrary>(std::move(keyframeBinding),std::move(timestampBinding),std::move(animationRange),cpuanim);
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUAccelerationStructure** _begin, const asset::ICPUAccelerationStructure** _end, SParams& _params) -> created_gpu_object_array<asset::ICPUAccelerationStructure>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUAccelerationStructure> >(assetCount);
	auto toCreateAndBuild = std::vector<const asset::ICPUAccelerationStructure*>();
    auto buildRangeInfos = std::vector<IGPUAccelerationStructure::BuildRangeInfo*>();
    toCreateAndBuild.reserve(assetCount);
    buildRangeInfos.reserve(assetCount);
    // Lambda function: creates the acceleration structure and It's buffer
    auto allocateBufferAndCreateAccelerationStructure = [&](size_t asSize, const asset::ICPUAccelerationStructure* cpuas)
    {
        // Create buffer with cpuas->getAccelerationStructureSize
        IGPUBuffer::SCreationParams gpuBufParams = {};
        gpuBufParams.size = asSize;
        gpuBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
        auto gpubuf = _params.device->createBuffer(std::move(gpuBufParams));
        auto mreqs = gpubuf->getMemoryReqs();
        mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufMem = _params.device->allocate(mreqs, gpubuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        assert(gpubufMem.isValid());

        // Create GPUAccelerationStructure with that buffer
        IGPUAccelerationStructure::SCreationParams creatationParams = {};
        creatationParams.bufferRange.buffer = gpubuf;
        creatationParams.bufferRange.offset = 0;
        creatationParams.bufferRange.size = asSize;
        creatationParams.flags = cpuas->getCreationParameters().flags;
        creatationParams.type = cpuas->getCreationParameters().type;
        return _params.device->createAccelerationStructure(std::move(creatationParams));
    };

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUAccelerationStructure* cpuas = _begin[i];

        if(cpuas->hasBuildInfo())
        {
            // Add to toBuild vector of ICPUAccelerationStructure
            toCreateAndBuild.push_back(cpuas);
            buildRangeInfos.push_back(const_cast<IGPUAccelerationStructure::BuildRangeInfo*>(cpuas->getBuildRanges().begin()));
        }
        else if(cpuas->getAccelerationStructureSize() > 0)
        {
            res->operator[](i) = allocateBufferAndCreateAccelerationStructure(cpuas->getAccelerationStructureSize(), cpuas);
        }
    }

    if(toCreateAndBuild.empty() == false)
    {
        bool hostBuildCommands = false; // get from SFeatures
        if(hostBuildCommands)
        {
            _NBL_TODO();
        }
        else
        {
            core::vector<const asset::ICPUBuffer*> cpuBufferDeps;
            constexpr uint32_t MaxGeometryPerBuildInfo = 16;
            constexpr uint32_t MaxBuffersPerGeometry = 3; // TrianglesData ->  vertex+index+transformation
            cpuBufferDeps.reserve(assetCount * MaxGeometryPerBuildInfo * MaxBuffersPerGeometry);

            // Get CPUBuffer Dependencies
            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto buildInfo = cpuas->getBuildInfo();
                assert(buildInfo != nullptr);

                auto geoms = buildInfo->getGeometries().begin();
                auto geomsCount = buildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& geom = geoms[g];
                    if(geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        if(geom.data.triangles.indexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.indexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.vertexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.vertexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.transformData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.transformData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        if(geom.data.aabbs.data.isValid())
                        {
                            auto cpuBuf = geom.data.aabbs.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        if(geom.data.instances.data.isValid())
                        {
                            auto cpuBuf = geom.data.instances.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                }
            }

            // Convert CPUBuffer Deps to GPUBuffers
            core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufferDeps);
            auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufferDeps.data(), cpuBufferDeps.data()+cpuBufferDeps.size(), _params);
            _params.waitForCreationToComplete();
            _params.beginCommandBuffers();
            size_t bufIter = 0ull;

            // Fill buildGeomInfos partially (to later ge Get AS Size before build command)
            std::vector<IGPUAccelerationStructure::DeviceBuildGeometryInfo> buildGeomInfos(toCreateAndBuild.size());
     
            using GPUGeometry = IGPUAccelerationStructure::Geometry<IGPUAccelerationStructure::DeviceAddressType>;
            std::vector<GPUGeometry> gpuGeoms;
            gpuGeoms.reserve(assetCount * MaxGeometryPerBuildInfo);

            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto cpuBuildInfo = cpuas->getBuildInfo();
                auto & gpuBuildInfo = buildGeomInfos[i];

                gpuBuildInfo.type = cpuBuildInfo->type;
                gpuBuildInfo.buildFlags = cpuBuildInfo->buildFlags;
                gpuBuildInfo.buildMode = cpuBuildInfo->buildMode;
                assert(cpuBuildInfo->buildMode == asset::IAccelerationStructure::EBM_BUILD);

                // Fill Later:
                gpuBuildInfo.srcAS = nullptr;
                gpuBuildInfo.dstAS = nullptr;
                gpuBuildInfo.scratchAddr = {};
                
                auto cpu_geoms = cpuBuildInfo->getGeometries().begin();
                auto geomsCount = cpuBuildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                size_t startGeom = gpuGeoms.size();
                size_t endGeom = gpuGeoms.size() + geomsCount;

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& cpu_geom = cpu_geoms[g];

                    GPUGeometry gpu_geom = {};
                    gpu_geom.type = cpu_geom.type;
                    gpu_geom.flags = cpu_geom.flags;

                    if(cpu_geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        gpu_geom.data.triangles.vertexFormat = cpu_geom.data.triangles.vertexFormat;
                        gpu_geom.data.triangles.vertexStride = cpu_geom.data.triangles.vertexStride;
                        gpu_geom.data.triangles.maxVertex = cpu_geom.data.triangles.maxVertex;
                        gpu_geom.data.triangles.indexType = cpu_geom.data.triangles.indexType;

                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.indexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.indexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.indexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.vertexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.vertexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.vertexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.transformData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.transformData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.transformData.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        gpu_geom.data.aabbs.stride = cpu_geom.data.aabbs.stride;
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.aabbs.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.aabbs.data.offset = gpubuf->getOffset() + cpu_geom.data.aabbs.data.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.instances.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.instances.data.offset = gpubuf->getOffset() + cpu_geom.data.instances.data.offset;
                        }
                    }

                    gpuGeoms.push_back(gpu_geom);
                }

                gpuBuildInfo.geometries = core::SRange<GPUGeometry>(gpuGeoms.data() + startGeom, gpuGeoms.data() + endGeom);
            }
            
            // Get SizeInfo for each CPUAS -> Create the AS -> Get Total Scratch Buffer Size 
            std::vector<IGPUAccelerationStructure::BuildSizes> buildSizes(toCreateAndBuild.size());
            uint64_t totalScratchBufferSize = 0ull;
            uint64_t maxScratchBufferSize = 0ull;
            for (ptrdiff_t i = 0u, toBuildIndex = 0u; i < assetCount; ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = _begin[i];
                if(cpuas->hasBuildInfo() == false)
                {
                    // Only those with buildInfo (index in toCreateAndBuild vector) will get passed
                    continue;
                }

                assert(cpuas == toCreateAndBuild[toBuildIndex]);
                assert(toBuildIndex < toCreateAndBuild.size());

                auto buildRanges = cpuas->getBuildRanges().begin();
                auto buildRangesCount = cpuas->getBuildRanges().size();

                auto & gpuBuildInfo = buildGeomInfos[toBuildIndex];
                
                std::vector<uint32_t> maxPrimCount(buildRangesCount);
                for(auto b = 0; b < buildRangesCount; b++)
                  maxPrimCount[b] = buildRanges[b].primitiveCount;

                auto buildSize = _params.device->getAccelerationStructureBuildSizes(gpuBuildInfo, maxPrimCount.data());
                buildSizes[i] = buildSize;

                auto gpuAS = allocateBufferAndCreateAccelerationStructure(buildSize.accelerationStructureSize, cpuas);
                res->operator[](i) = gpuAS;

                // complete the buildGeomInfos (now only thing left is to allocate and set scratchAddr.buffer)
                buildGeomInfos[toBuildIndex].dstAS = gpuAS.get();
                buildGeomInfos[toBuildIndex].scratchAddr.offset = totalScratchBufferSize;

                totalScratchBufferSize += buildSize.buildScratchSize;
                core::max(maxScratchBufferSize, buildSize.buildScratchSize); // maxScratchBufferSize has no use now (unless we changed this function to build 1 by 1 instead of batch builds or have some kind of memory limit?)
                ++toBuildIndex;
            }

            // Allocate Scratch Buffer
            IGPUBuffer::SCreationParams gpuScratchBufParams = {};
            gpuScratchBufParams.size = totalScratchBufferSize;
            gpuScratchBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT; 
            auto gpuScratchBuf = _params.device->createBuffer(std::move(gpuScratchBufParams));
            auto mreqs = gpuScratchBuf->getMemoryReqs();
            mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto gpuScratchBufMem = _params.device->allocate(mreqs, gpuScratchBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);


            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                auto & gpuBuildInfo = buildGeomInfos[i];
                gpuBuildInfo.scratchAddr.buffer = gpuScratchBuf;
            }

            // Record CommandBuffer for Building (We have Completed buildInfos + buildRanges for each CPUAS)
            auto & fence = _params.fences[EQU_COMPUTE];
            fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf = _params.perQueue[EQU_COMPUTE].cmdbuf;

            IQueue::SSubmitInfo submit;
            {
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &cmdbuf.get();
                submit.waitSemaphoreCount = 0u;
                submit.pWaitDstStageMask = nullptr;
                submit.pWaitSemaphores = nullptr;
                uint32_t waitSemaphoreCount = 0u;
            }
            
            assert(cmdbuf->getState() == IGPUCommandBuffer::STATE::RECORDING);
            cmdbuf->buildAccelerationStructures({buildGeomInfos.data(),buildGeomInfos.data()+buildGeomInfos.size()},buildRangeInfos.data());
            cmdbuf->end();

            // TODO for future to make this function more sophisticated: Compaction, MemoryLimit for Build

            core::smart_refctd_ptr<IGPUSemaphore> sem;
            
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                sem = _params.device->createSemaphore();

            auto* sem_ptr = sem.get();
            auto* fence_ptr = fence.get();

            submit.signalSemaphoreCount = sem_ptr?1u:0u;
            submit.pSignalSemaphores = sem_ptr?&sem_ptr:nullptr;

            _params.perQueue[EQU_COMPUTE].queue->submit(1u, &submit, fence_ptr);
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                _params.perQueue[EQU_COMPUTE].semaphore[0] = std::move(sem);
        }
    }

    return res;
}
#endif


}//nbl::video

#endif
