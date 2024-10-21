#ifndef _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_
#define _NBL_VIDEO_S_INTENDED_SUBMIT_INFO_H_INCLUDED_


#include "nbl/video/IGPUCommandBuffer.h"


namespace nbl::video
{
    
//! Struct meant to be used with any Utility (not just `IUtilities`) which exhibits "submit on overflow" behaviour.
//! Such functions are non-blocking (unless overflow) and take `SIntendedSubmitInfo` by reference and patch it accordingly. 
//!     for example: in the case the `waitSemaphores` were already waited upon, the struct will be modified to have it's `waitSemaphores` emptied.
//! User is supposed to begin EXACTLY ONE of the `scratchCommandBuffer` before first ever submit.
//!     Never `begin` or `end` scratch commandBuffers ever again.
//!     Always assume the current recording commandBuffer is in RECORDING state (except between `submit` and `beginNextCommandBuffer` call intended for ADVANCED users)
//!     The "current recording commandBuffer" is returned by `getCurrentRecordingCommandBufferInfo` or updated via the pointer ref in beginNextCommandBuffer or overflowSubmit)
struct SIntendedSubmitInfo final : core::Uncopyable
{
        // All commandbuffers must be compatible with the queue we're about to submit to
        bool cmdbufNotSubmittableToQueue(const IGPUCommandBuffer* cmdbuf) const
        {
            return !cmdbuf || cmdbuf->getPool()->getQueueFamilyIndex()!=queue->getFamilyIndex();
        }

        // Returns the scratch to use if valid, nullptr otherwise
        template<bool AllChecks>
        inline const IQueue::SSubmitInfo::SCommandBufferInfo* valid_impl() const
        {
            if (!queue || scratchCommandBuffers.empty() || !scratchSemaphore.semaphore)
                return nullptr;
            // the found scratch
            const IQueue::SSubmitInfo::SCommandBufferInfo* scratch = nullptr;
            // skip expensive stuff
            std::conditional_t<AllChecks,core::unordered_set<const IGPUCommandBuffer*>,const void*> uniqueCmdBufs;
            if constexpr (AllChecks)
            {
                // All commandbuffers before the scratch must be executable (ready to be submitted)
                for (const auto& info : prevCommandBuffers)
                if (cmdbufNotSubmittableToQueue(info.cmdbuf) || info.cmdbuf->getState()!=IGPUCommandBuffer::STATE::EXECUTABLE)
                    return nullptr;
                //
                uniqueCmdBufs.reserve(scratchCommandBuffers.size());
            }
            for (auto& info : scratchCommandBuffers)
            {
                if constexpr (AllChecks)
                {
                    // Must be resettable so we can end, submit, wait, reset and continue recording commands into it as-if nothing happened
                    if (cmdbufNotSubmittableToQueue(info.cmdbuf) || !info.cmdbuf->isResettable())
                        return nullptr;
                    uniqueCmdBufs.insert(info.cmdbuf);
                }
                // not our scratch
                if (info.cmdbuf->getState()!=IGPUCommandBuffer::STATE::RECORDING)
                    continue;
                // there can only be one scratch!
                if (scratch)
                    return nullptr;
                scratch = &info;
            }
            // a commandbuffer repeats itself
            if constexpr (AllChecks)
            if (uniqueCmdBufs.size()!=scratchCommandBuffers.size())
                return nullptr;
            // there is no scratch cmdbuf at all!
            if (!scratch)
                return nullptr;
            // It makes no sense to reuse the same commands for a second submission.
            // Moreover its dangerous because the utilities record their own internal commands which might use subresources for which
            // frees have already been latched on the scratch semaphore you must signal anyway.
            if (!scratch->cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                return nullptr;
            return scratch;
        }

    public:
        // This parameter is required but may be unused if there is no need (no overflow) to perform submit operations (no call to `overflowSubmit` or `submit`)
        IQueue* queue = nullptr;

        // Use this parameter to wait for previous operations to finish before whatever commands the Utility you're using records.
        std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {};

        // Fill the commandbuffers you want to run before the first command the Utility records to run in the same submit, 
        // for example, baked command buffers with pipeline barrier commands.
        std::span<const IQueue::SSubmitInfo::SCommandBufferInfo> prevCommandBuffers = {};

        // A set of command buffers the Utility can use in a round-robin manner for its transient commands. All command buffers must be individually resettable. 
        // Command buffers are cycled through for use and submission using a simple modular arithmetic fashion. 
        // EXACTLY ONE commandbuffer must be in recording state! This is the one the utilities will use immediately to record commands. 
        // However, even though it's scratch, you can record some of your own preceding commands into it as well.
        std::span<IQueue::SSubmitInfo::SCommandBufferInfo> scratchCommandBuffers = {};

        // This semaphore is needed to indicate when each sub-submit is complete and resources can be reclaimed.
        // It also ensures safe usage of the scratch command buffers by blocking the CPU if the next scratch command buffer to use is in a PENDING state and not safe to begin. 
        // The initial `scratchSemaphore.value` gets incremented and signaled on each submit. It can start at 0 because an overflow will signal `value+1`.
        // NOTE: You should signal this semaphore when doing your last/tail submit manually OR when not submitting the recorded scratch at all 
        //       (in which case, signal from Host or another submit). Why? The utilities that deal with `SIntendedSubmitInfo` might use the 
        //       `[Multi]TimelineEventHandler` to latch cleanups/deallocations on `scratchSemaphore`, and `getFutureScratchSemaphore()` 
        //       is the value they expect to get signaled in the future.
        // NOTE: Each overflow submit bumps the value to wait and signal by +1. A utility may overflow an arbitrary number of times. 
        //       Therefore, DO NOT choose the values for waits and signals manually, and do not modify the `scratchSemaphore` field after initialization or first use. 
        //       You can examine the change in `scratchSemaphore.value` to observe how many overflows or submits have occurred.
        IQueue::SSubmitInfo::SSemaphoreInfo scratchSemaphore = {};

        // Optional: If you had a semaphore whose highest pending signal is 45 but gave the scratch a value of 68 (causing 69 to get signaled in a popped submit), 
        // but only used 4 scratch command buffers, we'd wait for the semaphore to reach 66 before resetting the next scratch command buffer. 
        // This is obviously suboptimal if the next scratch command buffer wasn't pending with a signal of 67 at all (you'd be waiting until 70 gets signaled). 
        // Therefore, you would need to override this behavior to only wait for semaphore values higher than 68.
        size_t initialScratchValue = 0;

        // Optional: Callback to perform some other CPU work while blocking for one of the submitted scratch command buffers to complete execution. 
        // This callback may be called repeatedly! The argument provided is the scratch semaphore, allowing it to poll itself to determine when it can finish its work, 
        // preventing priority inversion.
        std::function<void(const ISemaphore::SWaitInfo&)> overflowCallback = {};

        //
        inline ISemaphore::SWaitInfo getFutureScratchSemaphore() const {return {scratchSemaphore.semaphore,scratchSemaphore.value+1};}
        
        //
        inline const auto* getCommandBufferForRecording() const {return valid_impl<false>();}

        //
        inline const auto* valid() const {return valid_impl<true>();}

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
                inline CSubmitStorage(const SIntendedSubmitInfo& info, IGPUCommandBuffer* scratch, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores)
                    : m_semaphores(info.waitSemaphores.size()+signalSemaphores.size()+1), m_cmdbufs(info.prevCommandBuffers.size()+1)
                {
                    auto oit = m_semaphores.data();

                    self.waitSemaphores = {oit,info.waitSemaphores.size()};
                    for (auto sem : info.waitSemaphores)
                        *(oit++) = sem;
                    
                    self.commandBuffers = m_cmdbufs;
                    std::copy(info.prevCommandBuffers.begin(),info.prevCommandBuffers.end(),m_cmdbufs.begin());
                    m_cmdbufs.back() = {.cmdbuf=scratch};

                    self.signalSemaphores = {oit,signalSemaphores.size()+1};
                    *(oit++) = {info.scratchSemaphore.semaphore,info.getFutureScratchSemaphore().value,info.scratchSemaphore.stageMask};
                    for (auto sem : signalSemaphores)
                        *(oit++) = sem;
                }

                IQueue::SSubmitInfo self = {};
                // Fully owning cause the semaphore fields in the `SIndendedSubmitInfo` are very mutable, so we copy
                core::vector<IQueue::SSubmitInfo::SSemaphoreInfo> m_semaphores;
                // thanks to this we no longer need the spans in the original `SIntendedSubmitInfo` to stay valid/unchanged
                core::vector<IQueue::SSubmitInfo::SCommandBufferInfo> m_cmdbufs;
        };
        
        // We assume you'll actually submit the return value before popping another one, hence we:
        // - increment the `scratchSemaphore.value` 
        // - clear the `waitSemaphores` which we'll use in the future because they will already be awaited on this `queue`
        inline CSubmitStorage popSubmit(IGPUCommandBuffer* recordingCmdBuf, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores)
        {
            assert(recordingCmdBuf);
            CSubmitStorage retval(*this,recordingCmdBuf,signalSemaphores);

            // If you want to wait for the result of this popped submit, you need to wait for this new value
            scratchSemaphore.value++;
            // The submits we wanted to wait on will have been awaited on this queue by the popped submit which is about to be submitted.
            waitSemaphores = {};
            // going to be submitted so forget as to not submit again
            prevCommandBuffers = {};

            return retval;
        }
        
        // ! [inout] recordingCmdBuf: current recording SCommandBufferInfo pointing to an element of `scratchCommandBuffers`.
        //      To achieve multiple submits in flight, the pointer may change and point to another element of `scratchCommandBuffers` when applicable.
        inline IQueue::RESULT overflowSubmit(const IQueue::SSubmitInfo::SCommandBufferInfo* &recordingCmdBuf)
        {
            IQueue::RESULT res = submit(recordingCmdBuf, {});
            if (res != IQueue::RESULT::SUCCESS)
                return res;
            if (!beginNextCommandBuffer(recordingCmdBuf))
                return IQueue::RESULT::OTHER_ERROR;
            return IQueue::RESULT::SUCCESS;
        }

        // Submits via the current recording command buffer
        // ! recordingCmdBuf: current recording SCommandBufferInfo pointing to an element of `scratchCommandBuffers`.
        // ! Optional: signalSemaphores: your signal semaphores that indicate the job is finished.
        // ! Don't attempt to use the `recordingCmdBuf` after calling this function and before calling `beginNextCommandBuffer` because it will be in PENDING state
        inline IQueue::RESULT submit(const IQueue::SSubmitInfo::SCommandBufferInfo* recordingCmdBuf, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores)
        {
            const auto pLast = &scratchCommandBuffers.back();
            if (recordingCmdBuf<scratchCommandBuffers.data() || recordingCmdBuf>pLast)
                return IQueue::RESULT::OTHER_ERROR;

            // First, submit the already buffered up work
            if (!recordingCmdBuf->cmdbuf->end())
                return IQueue::RESULT::OTHER_ERROR;
            
            // we only signal the scratch semaphore when overflowing
            const auto submit = popSubmit(recordingCmdBuf->cmdbuf,signalSemaphores);
            IQueue::RESULT res = queue->submit(submit);
            return res;
        }

        // If next command buffer is not PENDING and safe to use, it will reset and begin it, otherwise it will block on the scratch semaphore
        // ! recordingCmdBuf: previous recording SCommandBufferInfo pointing to an element of `scratchCommandBuffers`
        //      To achieve multiple submits in flight, the pointer may change and point to another element of `scratchCommandBuffers` when applicable.
        inline bool beginNextCommandBuffer(const IQueue::SSubmitInfo::SCommandBufferInfo* &recordingCmdBuf)
        {
            const auto pLast = &scratchCommandBuffers.back();
            if (recordingCmdBuf<scratchCommandBuffers.data() || recordingCmdBuf>pLast)
                return false;

            // Get the next command buffer to record
            if (recordingCmdBuf!=pLast)
                recordingCmdBuf++;
            else
                recordingCmdBuf = scratchCommandBuffers.data();
            // Now figure out if we need to block to reuse the next command buffer.
            const auto submitsInFlight = scratchCommandBuffers.size()-1;
            // We assume nobody was messing around with the scratchSemaphore too much and every popped submit increments value by 1
            if (scratchSemaphore.value>initialScratchValue+submitsInFlight)
            {
                const ISemaphore::SWaitInfo info = {.semaphore=scratchSemaphore.semaphore,.value=scratchSemaphore.value-submitsInFlight};
                // try to do some CPU work (or you can submit something else yourself to this queue or others from the callback)
                if (overflowCallback)
                    overflowCallback(info);
                // now wait before reset
                ISemaphore::WAIT_RESULT waitResult = const_cast<ILogicalDevice*>(queue->getOriginDevice())->blockForSemaphores({&info,1});
                if (waitResult!=ISemaphore::WAIT_RESULT::SUCCESS)
                {
                    assert(false);
                    return false;
                }
            }
            // could have just called begin to reset but also need to reset with the release resources flag
            recordingCmdBuf->cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
            recordingCmdBuf->cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

            return true;
        }
        
        // useful overload if you forgot what was command buffer to record.
        inline IQueue::RESULT overflowSubmit()
        {
            auto recordingCmdBuf = valid();
            return overflowSubmit(recordingCmdBuf);
        }

        // Error Text to Log/Display if you try to use an invalid `SIntendedSubmitInfo`
        static constexpr inline const char* ErrorText = R"===(Invalid `IUtilities::SIntendedSubmitInfo`, possible reasons are:
- No `scratchCommandBuffers` given in span or `scratchSemaphore` given
- some entry of `scratchCommandBuffers` is not Resettable
- there is not EXACTLY ONE already begun commandbuffer (with ONE_TIME_SUBMIT_BIT) in `scratchCommandBuffers` to use as scratch
- one of the `prevCommandBuffers` or `scratchCommandBuffers` Pool's Queue Family Index doesn't match `queue`'s Family
)===";
};

}

#endif