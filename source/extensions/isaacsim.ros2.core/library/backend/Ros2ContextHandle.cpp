// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Ros2Impl.h"

#include <carb/logging/Log.h>

#include <isaacsim/ros2/core/Ros2Macros.h>
#include <rcl/rcl.h>

namespace isaacsim
{
namespace ros2
{
namespace core
{

void* Ros2ContextHandleImpl::getContext()
{
    return m_context.get();
}

void Ros2ContextHandleImpl::init(int argc, char const* const* argv, bool setDomainId, size_t domainId)
{
    rcl_ret_t rc;
    // A previous init() call reaching this point (even one where the later
    // rcl_init() call itself failed, e.g. an invalid domain id) leaves
    // m_initOptions holding a live allocation from rcl_init_options_init()
    // below. shutdown() finalizes it and resets m_context to null, but if
    // init() is ever called again without an intervening shutdown() (e.g. a
    // caller retrying after isValid() reports false following a failed
    // rcl_init()), overwriting m_initOptions here would leak that allocation
    // every retry. Finalize it first whenever m_context is still set.
    if (m_context)
    {
        rcl_ret_t finiRc = rcl_init_options_fini(&m_initOptions);
        if (finiRc != RCL_RET_OK)
        {
            RCL_ERROR_MSG(Ros2ContextHandle, rcl_init_options_fini);
        }
    }
    // Initialize RCL init options and copy them
    m_initOptions = rcl_get_zero_initialized_init_options();
    rc = rcl_init_options_init(&m_initOptions, rcl_get_default_allocator());
    if (rc != RCL_RET_OK)
    {
        RCL_ERROR_MSG(Ros2ContextHandle, rcl_init_options_init);
        return;
    }
    if (setDomainId)
    {
        rcl_init_options_get_rmw_init_options(&m_initOptions)->domain_id = domainId;
    }

    m_context = std::shared_ptr<rcl_context_t>(new rcl_context_t,
                                               [this](rcl_context_t* context)
                                               {
                                                   if (nullptr != context->impl)
                                                   {
                                                       rcl_ret_t ret;
                                                       // Check the context pointer passed to the deleter, not
                                                       // this->isValid(): shutdown() invokes this deleter via
                                                       // m_context.reset(), by which point the m_context member is
                                                       // already empty, so isValid() would return false and skip
                                                       // rcl_shutdown/rcl_context_fini -> leaked context + DDS
                                                       // participant. (Using the local pointer also avoids touching a
                                                       // possibly-destroyed `this`.)
                                                       if (rcl_context_is_valid(context))
                                                       {
                                                           // shutdown first, if still valid
                                                           ret = rcl_shutdown(context);
                                                           if (RCL_RET_OK != ret)
                                                           {
                                                               RCL_ERROR_MSG(Ros2ContextHandle, rcl_shutdown);
                                                           }

                                                           ret = rcl_context_fini(context);
                                                           if (RCL_RET_OK != ret)
                                                           {
                                                               RCL_ERROR_MSG(Ros2ContextHandle, rcl_context_fini);
                                                           }
                                                       }
                                                   }
                                                   delete context;
                                               });

    // Init RCL Context
    *m_context = rcl_get_zero_initialized_context();
    rc = rcl_init(argc, argv, &m_initOptions, m_context.get());
    if (rc != RCL_RET_OK)
    {
        RCL_ERROR_MSG(Ros2ContextHandle, rcl_init);
        return;
    }
}

bool Ros2ContextHandleImpl::isValid()
{
    if (m_context.get())
    {
        return rcl_context_is_valid(m_context.get());
    }
    return false;
}

bool Ros2ContextHandleImpl::shutdown(const char* shutdownReason)
{
    // If the context is not valid, no need to do cleanup
    if (!m_context)
    {
        return true;
    }
    if (shutdownReason)
    {
        CARB_LOG_INFO("Ros2ContextHandle shutting down: %s", shutdownReason);
    }
    m_context.reset();
    // Finalize RCL options
    rcl_ret_t rc = rcl_init_options_fini(&m_initOptions);
    if (rc != RCL_RET_OK)
    {
        RCL_ERROR_MSG(~Ros2ContextHandle, rcl_init_options_fini);
        return false;
    }
    return true;
}

} // namespace core
} // namespace ros2
} // namespace isaacsim
