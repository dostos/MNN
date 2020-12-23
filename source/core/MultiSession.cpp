#include "MultiSession.hpp"
#include "MultiPipeline.hpp"
#include "Session.hpp"

namespace MNN {
SessionId MultiSession::sNextSessionId = 0;

SessionId MultiSession::addSession(Session* session) {
    SessionId currentId = sNextSessionId++;
    mSessions[currentId] = session;

    return currentId;
}

ErrorCode MultiSession::prepare() {
    // TODO : only consider sessions that has
    // samer backend type
    std::vector<SessionId> ids;
    for (auto session : mSessions) {
        ids.push_back(session.first);
        if (session.second->getNeedResize()) {
            auto code = session.second->resize();
            if (NO_ERROR != code) {
                return code;
            }
        }
    }

    mMultiSessionCaches.clear();
    // Subset execution
    //for (int numSessions = 1; numSessions <= ids.size(); numSessions++) {
    int numSessions = ids.size();
    std::vector<int> validIndexes(ids.size(), 0);
    for (int i = 0; i < numSessions; i++)
    {
        validIndexes[i] = 1;
    }

    std::sort(validIndexes.begin(), validIndexes.end());

    do {
        std::set<SessionId> idSet;
        std::vector<Session*> sessions;

        for(int i = 0 ; i < ids.size(); i++) {
            if (validIndexes[i]) {
                idSet.insert(ids[i]);
                sessions.push_back(mSessions[ids[i]]);
            }
        }

        mMultiSessionCaches[idSet] = std::make_shared<MultiSessionCache>(sessions);

    } while(std::next_permutation(validIndexes.begin(), validIndexes.end()));
    //}

    for (auto multiSessionCache : mMultiSessionCaches) {
        multiSessionCache.second->prepare();
    }


    return NO_ERROR;
}

ErrorCode MultiSession::runSequence(const std::set<SessionId> &requests, bool sync) {
    for (const SessionId& sessionId : requests) {
        if (mSessions.find(sessionId) != mSessions.end()) {
            ErrorCode error = mSessions[sessionId]->run();
            if (NO_ERROR != error) {
                return error;
            }
        }
    }

    if (sync) {
        for (const SessionId& sessionId : requests) {
            if (mSessions.find(sessionId) != mSessions.end()) {
                for (auto& bn : mSessions[sessionId]->mBackends) {
                    if(bn.second){
                        bn.second->onWaitFinish();
                    }
                }
            }
        }
    }

    return NO_ERROR;
}

ErrorCode MultiSession::runParallel(const std::set<SessionId> &requests, bool sync) {

    if (mMultiSessionCaches.find(requests) != mMultiSessionCaches.end()) {
        auto multiSessionCache = mMultiSessionCaches[requests];

        return multiSessionCache->run();
    }

    return NO_ERROR;
}


ErrorCode MultiSession::runWithCallBack(const std::set<SessionId> &requests, const TensorCallBackWithInfo& enterCallback, const TensorCallBackWithInfo& exitCallback, bool sync) {

    if (mMultiSessionCaches.find(requests) != mMultiSessionCaches.end()) {
        auto multiSessionCache = mMultiSessionCaches[requests];

        return multiSessionCache->runWithCallBack(enterCallback, exitCallback);
    }

    return NO_ERROR;
}

MultiSession::MultiSessionCache::MultiSessionCache(std::vector<Session *> sessions)
    :mSessions(sessions) {

    std::vector<Pipeline *> pipelines;
    int pipelineIndex = 0;
    do
    {
        pipelines.clear();

        for (auto session : sessions) {
            if (pipelineIndex < session->mPipelines.size())
                pipelines.push_back(session->mPipelines[pipelineIndex].get());
        }

        if (!pipelines.empty())
            mMultiPipelines.push_back(std::make_shared<MultiPipeline>(pipelines, sessions[0]->mBackends[MNN_FORWARD_OPENCL].get()));
        pipelineIndex++;
    } while (!pipelines.empty());
}

ErrorCode MultiSession::MultiSessionCache::prepare() {
    for(auto pipeline : mMultiPipelines) {
        auto code = pipeline->prepare();
        MNN_ASSERT(code == NO_ERROR);
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}

ErrorCode MultiSession::MultiSessionCache::run() {
    for(auto pipeline : mMultiPipelines) {
        auto code = pipeline->run();
        MNN_ASSERT(code == NO_ERROR);
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}

ErrorCode MultiSession::MultiSessionCache::runWithCallBack(const TensorCallBackWithInfo& enterCallback, const TensorCallBackWithInfo& exitCallback) {
    for(auto pipeline : mMultiPipelines) {
        auto code = pipeline->runWithCallBack(enterCallback, exitCallback);
        MNN_ASSERT(code == NO_ERROR);
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}
}