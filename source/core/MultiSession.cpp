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
    for (auto session : mSessions) {
        if (session.second->getNeedResize()) {
            auto code = session.second->resize();
            if (NO_ERROR != code) {
                return code;
            }
        }
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

        mMultiPipelines.push_back(std::make_shared<MultiPipeline>(pipelines));
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
}