// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef _WIN32
#    pragma warning(push)
#    pragma warning(disable : 4996)
#endif

// clang-format off
#include <pch/UsdPCH.h>
// clang-format on

#include <carb/profiler/Profile.h>
#include <carb/tasking/ITasking.h>
#include <carb/tasking/TaskingUtils.h>

#include <isaacsim/ros2/core/Ros2Node.h>
#include <isaacsim/ros2/nodes/PointCloudPublisher.h>

#include <OgnROS2PublishPointCloudDatabase.h>

using namespace isaacsim::ros2::core;

class OgnROS2PublishPointCloud : public Ros2Node
{
public:
    static bool compute(OgnROS2PublishPointCloudDatabase& db)
    {
        auto& state = db.perInstanceState<OgnROS2PublishPointCloud>();
        const auto& nodeObj = db.abi_node();

        if (!state.m_pub.isInitialized())
        {
            const GraphContextObj& context = db.abi_context();
            long stageId = context.iContext->getStageId(context);
            auto stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

            if (!state.m_pub.initialize(
                    std::string(nodeObj.iNode->getPrimPath(nodeObj)),
                    collectNamespace(db.inputs.nodeNamespace(),
                                     stage->GetPrimAtPath(pxr::SdfPath(nodeObj.iNode->getPrimPath(nodeObj)))),
                    db.inputs.topicName(), db.inputs.frameId(), db.inputs.queueSize(), db.inputs.qosProfile(),
                    db.inputs.context()))
            {
                db.logError("Unable to create ROS2 publisher");
                return false;
            }

            carb::settings::ISettings* threadSettings = carb::getCachedInterface<carb::settings::ISettings>();
            static constexpr char s_kThreadDisable[] = "/exts/isaacsim.ros2.bridge/publish_multithreading_disabled";
            state.m_multithreadingDisabled = threadSettings->getAsBool(s_kThreadDisable);
            return true;
        }

        return state.publishLidar(db);
    }


    bool publishLidar(OgnROS2PublishPointCloudDatabase& db)
    {
        CARB_PROFILE_ZONE(0, "[IsaacSim] Lidar Point Cloud Pub");
        auto& state = db.perInstanceState<OgnROS2PublishPointCloud>();
        auto tasking = carb::getCachedInterface<carb::tasking::ITasking>();

        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] wait for previous publish");
            state.m_tasks.wait();
        }

        // Cartesian data + individual metadata pointers, gated by output* flags.
        // Each *BufferSize (bytes) is supplied by the upstream sensor node
        // alongside its pointer; validate it against the number of points about
        // to be published before trusting the pointer, since
        // fillPointCloudBufferHost()/publishFromDevice() read exactly numPoints
        // elements through it with no bounds check of their own -- a stale or
        // undersized buffer (e.g. left over from a previous, smaller-resolution
        // scan) would otherwise be over-read.
        const size_t numPoints =
            db.inputs.dataPtr() != 0 ? db.inputs.bufferSize() / sizeof(GfVec3f) : db.inputs.data.size();
        auto validateMetadataBuffer = [&](bool enabled, uint64_t bufferSizeBytes, size_t elementSize,
                                          const char* fieldName)
        {
            if (!enabled)
            {
                return false;
            }
            if (bufferSizeBytes < numPoints * elementSize)
            {
                db.logWarning(
                    "PointCloud metadata field '%s' buffer size (%llu bytes) is smaller than required for "
                    "%zu points (%zu bytes); omitting this field from the published message",
                    fieldName, static_cast<unsigned long long>(bufferSizeBytes), numPoints, numPoints * elementSize);
                return false;
            }
            return true;
        };

        isaacsim::ros2::nodes::PointCloudMetadata metadata;
        metadata.intensityPtr =
            validateMetadataBuffer(db.inputs.outputIntensity(), db.inputs.intensityBufferSize(), sizeof(float),
                                    "intensity")
                ? reinterpret_cast<float*>(db.inputs.intensityPtr())
                : nullptr;
        metadata.timestampPtr =
            validateMetadataBuffer(db.inputs.outputTimestamp(), db.inputs.timestampBufferSize(), sizeof(uint64_t),
                                    "timestamp")
                ? reinterpret_cast<uint64_t*>(db.inputs.timestampPtr())
                : nullptr;
        metadata.emitterIdPtr =
            validateMetadataBuffer(db.inputs.outputEmitterId(), db.inputs.emitterIdBufferSize(), sizeof(uint32_t),
                                    "emitterId")
                ? reinterpret_cast<uint32_t*>(db.inputs.emitterIdPtr())
                : nullptr;
        metadata.channelIdPtr =
            validateMetadataBuffer(db.inputs.outputChannelId(), db.inputs.channelIdBufferSize(), sizeof(uint32_t),
                                    "channelId")
                ? reinterpret_cast<uint32_t*>(db.inputs.channelIdPtr())
                : nullptr;
        metadata.materialIdPtr =
            validateMetadataBuffer(db.inputs.outputMaterialId(), db.inputs.materialIdBufferSize(), sizeof(uint32_t),
                                    "materialId")
                ? reinterpret_cast<uint32_t*>(db.inputs.materialIdPtr())
                : nullptr;
        metadata.tickIdPtr =
            validateMetadataBuffer(db.inputs.outputTickId(), db.inputs.tickIdBufferSize(), sizeof(uint32_t), "tickId")
                ? reinterpret_cast<uint32_t*>(db.inputs.tickIdPtr())
                : nullptr;
        metadata.hitNormalPtr =
            validateMetadataBuffer(db.inputs.outputHitNormal(), db.inputs.hitNormalBufferSize(), sizeof(GfVec3f),
                                    "hitNormal")
                ? reinterpret_cast<GfVec3f*>(db.inputs.hitNormalPtr())
                : nullptr;
        metadata.velocityPtr =
            validateMetadataBuffer(db.inputs.outputVelocity(), db.inputs.velocityBufferSize(), sizeof(GfVec3f),
                                    "velocity")
                ? reinterpret_cast<GfVec3f*>(db.inputs.velocityPtr())
                : nullptr;
        metadata.objectIdPtr =
            validateMetadataBuffer(db.inputs.outputObjectId(), db.inputs.objectIdBufferSize(), sizeof(uint32_t),
                                    "objectId")
                ? reinterpret_cast<uint32_t*>(db.inputs.objectIdPtr())
                : nullptr;
        metadata.echoIdPtr =
            validateMetadataBuffer(db.inputs.outputEchoId(), db.inputs.echoIdBufferSize(), sizeof(uint8_t), "echoId")
                ? reinterpret_cast<uint8_t*>(db.inputs.echoIdPtr())
                : nullptr;
        metadata.tickStatePtr =
            validateMetadataBuffer(db.inputs.outputTickState(), db.inputs.tickStateBufferSize(), sizeof(uint8_t),
                                    "tickState")
                ? reinterpret_cast<uint8_t*>(db.inputs.tickStatePtr())
                : nullptr;
        metadata.radialVelocityMSPtr =
            validateMetadataBuffer(db.inputs.outputRadialVelocityMS(), db.inputs.radialVelocityMSBufferSize(),
                                    sizeof(float), "radialVelocityMS")
                ? reinterpret_cast<float*>(db.inputs.radialVelocityMSPtr())
                : nullptr;

        if (db.inputs.cudaDeviceIndex() == -1)
        {
            if (db.inputs.dataPtr() != 0)
            {
                if (db.inputs.bufferSize() == 0)
                {
                    return false;
                }
                state.m_pub.getPointCloudMessage()->generateBuffer(
                    db.inputs.timeStamp(), state.m_pub.getFrameId(), db.inputs.bufferSize(), metadata.intensityPtr,
                    metadata.timestampPtr, metadata.emitterIdPtr, metadata.channelIdPtr, metadata.materialIdPtr,
                    metadata.tickIdPtr, metadata.hitNormalPtr, metadata.velocityPtr, metadata.objectIdPtr,
                    metadata.echoIdPtr, metadata.tickStatePtr, metadata.radialVelocityMSPtr);
                // Data is on host as ptr
                if (state.m_pub.getPointCloudMessage()->getOrderedFields().empty())
                {
                    // Direct copy when no metadata interleaving is needed
                    memcpy(state.m_pub.getPointCloudMessage()->getBufferPtr(),
                           reinterpret_cast<void*>(db.inputs.dataPtr()),
                           state.m_pub.getPointCloudMessage()->getTotalBytes());
                }
                else
                {
                    // Host interleave: fill buffer with xyz + metadata per point
                    isaacsim::ros2::nodes::fillPointCloudBufferHost(
                        reinterpret_cast<uint8_t*>(state.m_pub.getPointCloudMessage()->getBufferPtr()),
                        reinterpret_cast<const float3*>(db.inputs.dataPtr()),
                        state.m_pub.getPointCloudMessage()->getOrderedFields(),
                        state.m_pub.getPointCloudMessage()->getPointStep(),
                        state.m_pub.getPointCloudMessage()->getNumPoints());
                }
            }
            else
            {
                const size_t totalBytes = sizeof(GfVec3f) * db.inputs.data.size();
                if (totalBytes == 0)
                {
                    return false;
                }
                state.m_pub.getPointCloudMessage()->generateBuffer(
                    db.inputs.timeStamp(), state.m_pub.getFrameId(), totalBytes, metadata.intensityPtr,
                    metadata.timestampPtr, metadata.emitterIdPtr, metadata.channelIdPtr, metadata.materialIdPtr,
                    metadata.tickIdPtr, metadata.hitNormalPtr, metadata.velocityPtr, metadata.objectIdPtr,
                    metadata.echoIdPtr, metadata.tickStatePtr, metadata.radialVelocityMSPtr);
                // Data is on host as ogn data, copy from cpu. Mirror the dataPtr branch:
                // when metadata fields are enabled, point_step (and thus getTotalBytes())
                // exceeds the xyz-only source size, so a raw memcpy would over-read the
                // input buffer and emit a corrupt cloud. Interleave instead.
                if (state.m_pub.getPointCloudMessage()->getOrderedFields().empty())
                {
                    memcpy(state.m_pub.getPointCloudMessage()->getBufferPtr(),
                           reinterpret_cast<const uint8_t*>(db.inputs.data.cpu().data()),
                           state.m_pub.getPointCloudMessage()->getTotalBytes());
                }
                else
                {
                    isaacsim::ros2::nodes::fillPointCloudBufferHost(
                        reinterpret_cast<uint8_t*>(state.m_pub.getPointCloudMessage()->getBufferPtr()),
                        reinterpret_cast<const float3*>(db.inputs.data.cpu().data()),
                        state.m_pub.getPointCloudMessage()->getOrderedFields(),
                        state.m_pub.getPointCloudMessage()->getPointStep(),
                        state.m_pub.getPointCloudMessage()->getNumPoints());
                }
            }

            if (state.m_multithreadingDisabled)
            {
                CARB_PROFILE_ZONE(1, "[IsaacSim] Publish PCL");
                state.m_pub.send();
            }
            else
            {
                tasking->addTask(carb::tasking::Priority::eHigh, state.m_tasks,
                                 [&state]
                                 {
                                     CARB_PROFILE_ZONE(1, "[IsaacSim] Publish PCL Thread");
                                     state.m_pub.send();
                                 });
            }
        }
        else if (db.inputs.dataPtr() != 0)
        {
            const void* devicePtr = reinterpret_cast<const void*>(db.inputs.dataPtr());
            const size_t bufferSize = db.inputs.bufferSize();
            const double timestamp = db.inputs.timeStamp();
            const int cudaDeviceIndex = db.inputs.cudaDeviceIndex();

            if (state.m_multithreadingDisabled)
            {
                state.m_pub.publishFromDevice(devicePtr, bufferSize, timestamp, cudaDeviceIndex, metadata);
            }
            else
            {
                tasking->addTask(
                    carb::tasking::Priority::eHigh, state.m_tasks,
                    [&state, devicePtr, bufferSize, timestamp, cudaDeviceIndex, metadata]() mutable
                    { state.m_pub.publishFromDevice(devicePtr, bufferSize, timestamp, cudaDeviceIndex, metadata); });
            }
        }

        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2PublishPointCloudDatabase::sPerInstanceState<OgnROS2PublishPointCloud>(nodeObj, instanceId);
        state.reset();
    }

    void reset() override
    {
        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] wait for previous publish");
            m_tasks.wait();
        }
        m_pub.reset();
        Ros2Node::reset();
    }

private:
    isaacsim::ros2::nodes::PointCloudPublisher m_pub;

    carb::tasking::TaskGroup m_tasks;

    bool m_multithreadingDisabled = false;
};

REGISTER_OGN_NODE()

#ifdef _WIN32
#    pragma warning(pop)
#endif
