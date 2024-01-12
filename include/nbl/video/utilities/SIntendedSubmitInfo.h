#ifndef _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_
#define _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_


#include "nbl/video/IGPUCommandBuffer.h"


namespace nbl::video
{
    
//! Struct meant to be used with any Utility (not just `IUtilities`) which exhibits "submit on overflow" behaviour.
//! Such functions are non-blocking (unless overflow) and take `SIntendedSubmitInfo` by reference and patch it accordingly. 
//! MAKE SURE to do a submit to `queue` by yourself with a submit info obtained by casting `this` to `IQueue::SSubmitInfo` !
//!     for example: in the case the `frontHalf.waitSemaphores` were already waited upon, the struct will be modified to have it's `waitSemaphores` emptied.
struct SIntendedSubmitInfo final
{
    public:
        inline bool valid() const
        {
            if (!frontHalf.valid() || frontHalf.commandBuffers.empty() || signalSemaphores.empty())
                return false;
            const auto* scratch = frontHalf.getScratchCommandBuffer();
            // Must be resettable so we can end, submit, wait, reset and continue recording commands into it as-if nothing happened 
            if (!scratch->isResettable())
                return false;
            // It makes no sense to reuse the same commands for a second submission.
            // Moreover its dangerous because the utilities record their own internal commands which might use subresources for which
            // frees have already been latched on the scratch semaphore you must signal anyway.
            if (!scratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                return false;
            if (scratch->getState()!=IGPUCommandBuffer::STATE::INITIAL)
                return false;
            return true;
        }

        inline ISemaphore::SWaitInfo getScratchSemaphoreNextWait() const {return {signalSemaphores.front().semaphore,signalSemaphores.front().value};}

        inline operator IQueue::SSubmitInfo() const
        {
            return {
                .waitSemaphores = frontHalf.waitSemaphores,
                .commandBuffers = frontHalf.commandBuffers,
                .signalSemaphores = signalSemaphores
            };
        }

        // One thing you might notice is that this results in a few implicit Memory and Execution Dependencies
        // So there's a little bit of non-deterministic behaviour we won't fight (will not insert a barrier every time you "could-have" overflown)
        inline void overflowSubmit()
        {
            auto cmdbuf = frontHalf.getScratchCommandBuffer();
            auto& scratchSemaphore = signalSemaphores.front();
            // but first sumbit the already buffered up copies
            cmdbuf->end();
            IQueue::SSubmitInfo submit = *this;
            // we only signal the last semaphore which is used as scratch
            submit.signalSemaphores = {&scratchSemaphore,1};
            assert(submit.isValid());
            frontHalf.queue->submit({&submit,1});
            // We wait (stall) on the immediately preceeding submission timeline semaphore signal value and increase it for the next signaller
            {
                const ISemaphore::SWaitInfo info = {scratchSemaphore.semaphore,scratchSemaphore.value++};
                const_cast<ILogicalDevice*>(cmdbuf->getOriginDevice())->blockForSemaphores({&info,1});
            }
            // we've already waited on the Host for the semaphores, no use waiting twice
            frontHalf.waitSemaphores = {};
            // since all the commandbuffers have submitted already we only reuse the last one
            frontHalf.commandBuffers = {&frontHalf.commandBuffers.back(),1};
            // we will still signal the same set in the future
            cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
            cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        }


        //! The last CommandBuffer will be used to record the copy commands    
        struct SFrontHalf final
        {
            //! Need a valid queue and all the command buffers except the last one should be in `EXECUTABLE` state.
            inline bool valid() const
            {
                if (!queue)
                    return false;
                if (!commandBuffers.empty())
                for (size_t i=0; i<commandBuffers.size()-1; i++)
                if (commandBuffers[i].cmdbuf->getState()==IGPUCommandBuffer::STATE::EXECUTABLE)
                    return false;
                return true;
            }

            //! Little class to hold the storage for the modified commandbuffer span until submission time.
            class CRAIISpanPatch final : core::Uncopyable
            {
                public:
                    inline ~CRAIISpanPatch()
                    {
                        toNullify->commandBuffers = {};
                    }
                    inline CRAIISpanPatch(CRAIISpanPatch&& other) : CRAIISpanPatch() {operator=(std::move(other));}
                    inline CRAIISpanPatch& operator=(CRAIISpanPatch&& rhs)
                    {
                        commandBuffersStorage = std::move(rhs.commandBuffersStorage);
                        return *this;
                    }

                    inline operator bool() const {return m_recordingCommandBuffer.get();}

                private:
                    friend SFrontHalf;
                    inline CRAIISpanPatch() = default;
                    inline CRAIISpanPatch(SFrontHalf* _toNull) : commandBuffersStorage(_toNull->commandBuffers.size()+1), toNullify(_toNull) {}

                    core::vector<IQueue::SSubmitInfo::SCommandBufferInfo> commandBuffersStorage;
                    // If we made a new commandbuffer we need to nullify the span so it doesn't point at stale mem
                    SFrontHalf* toNullify = nullptr;
                    // If new one made, then need to hold reference to it, else its just an extra ref, but whatever
                    core::smart_refctd_ptr<IGPUCommandBuffer> m_recordingCommandBuffer;
            };
            //! Patches the `commandBuffers` and then makes sure the last command buffer is resettable, in recording state begun with ONE_TIME_SUBMIT
            //! If we can't make the last cmdbuffer that way, we make a new one and add it onto the end (hence the name "patching")
            //! If `commandBuffers.empty()`, it will create an implicit command buffer to use for recording commands,
            //! else if the last command buffer is not feasible to use as scratch for whatever reason,
            //! it will add another temporary command buffer to end of `commandBuffers` and use it for recording.
            //! WARNING: If patching occurs:
            //!     - a submission must occur before the return value goes out of scope!
            //!     - if `!commandBuffers.empty()`, the last CommandBuffer won't be in the same state as it was before entering the function,
            //!         because it needs to be `end()`ed before the submission
            //!     - the destructor of the return value will clear `commandBuffers` span
            //! For more info on command buffer states See `ICommandBuffer::E_STATE` comments.
            [[nodiscard("The RAII object returned by `patch()` provides lifetimes to your spans!")]]
            inline CRAIISpanPatch patch()
            {
                if (auto* candidateScratch = getScratchCommandBuffer(); candidateScratch && candidateScratch->isResettable())
                switch(candidateScratch->getState())
                {
                    case IGPUCommandBuffer::STATE::INITIAL:
                        if (!candidateScratch->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                            break;
                        [[fallthrough]];
                    case IGPUCommandBuffer::STATE::RECORDING:
                        if (!candidateScratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                            break;
                        {
                            CRAIISpanPatch retval;
                            retval.m_recordingCommandBuffer = core::smart_refctd_ptr<IGPUCommandBuffer>(candidateScratch);
                            return retval;
                        }
                        break;
                    default:
                        break;
                }

                CRAIISpanPatch retval(this);
                std::copy(commandBuffers.begin(),commandBuffers.end(),retval.commandBuffersStorage.begin());
                {
                    auto pool = const_cast<ILogicalDevice*>(queue->getOriginDevice())->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
                    if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&retval.m_recordingCommandBuffer,1}))
                        return {};
                    if (!retval.m_recordingCommandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                        return {};
                    retval.commandBuffersStorage.back().cmdbuf = retval.m_recordingCommandBuffer.get();
                }
                commandBuffers = retval.commandBuffersStorage;
                return retval;
            }

            // Use the last command buffer in intendedNextSubmit, it should be in recording state
            inline IGPUCommandBuffer* getScratchCommandBuffer() {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}
            inline const IGPUCommandBuffer* getScratchCommandBuffer() const {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}

            // This parameter is required but may be unused if there is no need to submit
            IQueue* queue = {};
            // Use this parameter to wait for previous operations to finish before whatever commands the Utility you're using records
            std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {};
            // Fill the commandbuffers you want to run before the first command the Utility records to run in the same submit,
            // for example baked command buffers with pipeline barrier commands.
            // Also remember that even though the last CommandBuffer is scratch, it you can record commands into it as well.
            std::span<IQueue::SSubmitInfo::SCommandBufferInfo> commandBuffers = {};
        } frontHalf = {};
        //! The first Semaphore will be used as a scratch, so don't choose the values for waits and signals yourself as we can advance the counter an arbitrary amount!
        //! You can actually examine the change in `signalSemaphore.front().value` to figure out how many overflows occurred.
        //! This semaphore is needed to "stitch together" additional submits if they occur so they occur before and after the original intended waits and signals.
        //! We use the first semaphore to keep the intended order of original semaphore signal and waits unchanged no matter how many overflows occur.
        //! You do however, NEED TO KEEP IT in the signal set of the last submit you're supposed to do manually, this allows freeing any resources used
        //! after the submit is done, indicating that your streaming routine is done.  
        //! * Also use this parameter to signal new semaphores so that other submits know your Utility method is done.
        std::span<IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores = {};

    private:
        friend class IUtilities;
        static const char* ErrorText;
};

}

#endif