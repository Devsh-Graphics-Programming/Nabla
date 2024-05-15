#ifndef _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_
#define _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_


#include "nbl/video/IGPUCommandBuffer.h"


namespace nbl::video
{
    
//! Struct meant to be used with any Utility (not just `IUtilities`) which exhibits "submit on overflow" behaviour.
//! Such functions are non-blocking (unless overflow) and take `SIntendedSubmitInfo` by reference and patch it accordingly. 
//!     for example: in the case the `waitSemaphores` were already waited upon, the struct will be modified to have it's `waitSemaphores` emptied.
//! MAKE SURE to do a submit to `queue` by yourself with the result of `popSubmit(...)` implicitly converted to `std::span<const IQueue::SSubmitInfo>` !
struct SIntendedSubmitInfo final : core::Uncopyable
{
    public:
        // This parameter is required but may be unused if there is no need (no overflow) to do submit
        IQueue* queue = nullptr;
        // Use this parameter to wait for previous operations to finish before whatever commands the Utility you're using records
        std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {};
        // Fill the commandbuffers you want to run before the first command the Utility records to run in the same submit,
        // for example baked command buffers with pipeline barrier commands.
        // The last CommandBuffer will be used to record the transient commands of the Utility,
        // but remember that even though the last CommandBuffer is scratch, you can record commands into it as well.
        std::span<IQueue::SSubmitInfo::SCommandBufferInfo> commandBuffers = {};
        // This parameter is required but may be unused if there is no need (no overflow) to do submit.
        // This semaphore is needed to "stitch together" additional submits if they occur so they occur before and after the original intended waits and signals.
        // The initial `scratchSemaphore.value` gets incremented and signalled on each submit, can start at 0 because an overflow will signal `value+1`.
        // NOTE: You should signal this semaphore as well when doing your last/tail submit! Why?
        // Utilities might use the `[Multi]TimelineEventHandler` to latch cleanups on `scratchSemaphore` where this value
        // is a value that they expect to actually get signalled in the future.
        // NOTE: Do not choose the values for waits and signals yourself, as the overflows may increment the counter by an arbitrary amount!
        // You can actually examine the change in `scratchSemaphore.value` to figure out how many overflows occurred.
        IQueue::SSubmitInfo::SSemaphoreInfo scratchSemaphore = {};
        

        // Use the last command buffer in intendedNextSubmit, it should be in recording state
        inline IGPUCommandBuffer* getScratchCommandBuffer() {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}
        inline const IGPUCommandBuffer* getScratchCommandBuffer() const {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}

        //
        inline ISemaphore::SWaitInfo getFutureScratchSemaphore() const {return {scratchSemaphore.semaphore,scratchSemaphore.value+1};}

        //  
        inline bool valid() const
        {
            if (!queue || commandBuffers.empty() || !scratchSemaphore.semaphore)
                return false;
            // All commandbuffers must be compatible with the queue we're about to submit to
            auto cmdbufNotSubmittableToQueue = [this](const IGPUCommandBuffer* cmdbuf)->bool
            {
                return !cmdbuf || cmdbuf->getPool()->getQueueFamilyIndex()!=queue->getFamilyIndex();
            };
            // All commandbuffers before the scratch must be executable (ready to be submitted)
            for (size_t i=0; i<commandBuffers.size()-1; i++)
            if (cmdbufNotSubmittableToQueue(commandBuffers[i].cmdbuf) || commandBuffers[i].cmdbuf->getState()!=IGPUCommandBuffer::STATE::EXECUTABLE)
                return false;
            const auto* scratch = getScratchCommandBuffer();
            // Must be resettable so we can end, submit, wait, reset and continue recording commands into it as-if nothing happened 
            if (cmdbufNotSubmittableToQueue(scratch) || !scratch->isResettable())
                return false;
            // It makes no sense to reuse the same commands for a second submission.
            // Moreover its dangerous because the utilities record their own internal commands which might use subresources for which
            // frees have already been latched on the scratch semaphore you must signal anyway.
            if (!scratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                return false;
            // Also the Scratch Command Buffer must be already begun
            if (scratch->getState()!=IGPUCommandBuffer::STATE::RECORDING)
                return false;
            return true;
        }

        //! xxxx
        class CSubmitStorage final : core::Uncopyable
        {
            public:
                inline CSubmitStorage() = default;
                inline ~CSubmitStorage() = default;
                inline CSubmitStorage(CSubmitStorage&& other) : self(other.self), m_semaphores(std::move(other.m_semaphores)) {}
                inline CSubmitStorage& operator=(CSubmitStorage&& rhs)
                {
                    self = std::move(self);
                    m_semaphores = std::move(rhs.m_semaphores);
                }

                inline operator std::span<const IQueue::SSubmitInfo>() const
                {
                    return {&self,1};
                }

            private:
                friend struct SIntendedSubmitInfo;
                inline CSubmitStorage(const SIntendedSubmitInfo& info, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores)
                    : m_semaphores(info.waitSemaphores.size()+signalSemaphores.size()+1)
                {
                    auto oit = m_semaphores.data();

                    self.waitSemaphores = {oit,info.waitSemaphores.size()};
                    for (auto sem : info.waitSemaphores)
                        *(oit++) = sem;
                    
                    self.commandBuffers = info.commandBuffers;

                    self.signalSemaphores = {oit,signalSemaphores.size()+1};
                    *(oit++) = {info.scratchSemaphore.semaphore,info.getFutureScratchSemaphore().value,info.scratchSemaphore.stageMask};
                    for (auto sem : signalSemaphores)
                        *(oit++) = sem;
                }

                IQueue::SSubmitInfo self;
                // Fully owning cause the semaphore fields in the `SIndendeSubmitInfo` are very mutable, so we copy
                core::vector<IQueue::SSubmitInfo::SSemaphoreInfo> m_semaphores;
        };
        
        // We assume you'll actually submit the return value before popping another one, hence we:
        // - increment the `scratchSemaphore.value` 
        // - set the `waitSemaphores` which we'll use in the future to `scratchSemaphore` that will be signalled by the popped submit
        inline CSubmitStorage popSubmit(const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores)
        {
            CSubmitStorage retval(*this,signalSemaphores);

            // If you keep reusing this SIntendedSubmitInfo for multiple utility calls, then we need to make sure that the future submit will
            // wait for this one. This achieves a command ordering in the cmdbuffer transparent to overflow submits.
            scratchSemaphore.value++;
            waitSemaphores = {&scratchSemaphore,1};

            return retval;
        }
        
        // One thing you might notice is that this results in a few implicit Memory and Execution Dependencies
        // So there's a little bit of non-deterministic behaviour we won't fight (will not insert a barrier every time you "could-have" overflown)
        inline void overflowSubmit()
        {
            auto cmdbuf = getScratchCommandBuffer();
            // first sumbit the already buffered up work
            cmdbuf->end();
            // we only signal the scratch semaphore when overflowing
            const auto submit = popSubmit({});
            queue->submit(submit);
            // We wait (stall) on the immediately preceeding submission, this could be improved in the future with multiple buffering of the commandbuffers
            {
                const ISemaphore::SWaitInfo info = {.semaphore=scratchSemaphore.semaphore,.value=scratchSemaphore.value};
                const_cast<ILogicalDevice*>(cmdbuf->getOriginDevice())->blockForSemaphores({&info,1});
            }
            // since all the commandbuffers have submitted already we only reuse the last one
            commandBuffers = {&commandBuffers.back(),1};
            // we will still signal the same set in the future
            cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
            cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        }

        // Error Text to Log/Display if you try to use an invalid `SIntendedSubmitInfo`
        static constexpr inline const char* ErrorText = R"===(Invalid `IUtilities::SIntendedSubmitInfo`, possible reasons are:
- No `commandBuffers` or `signalSemaphores` given in respective spans
- `commandBuffer.back()` is not Resettable
- `commandBuffer.back()` is not already begun with ONE_TIME_SUBMIT_BIT
- one of the `commandBuffer`s' Pool's Queue Family Index doesn't match `queue`'s Family
)===";
};

}

#endif