#include "MultiSession.hpp"
#include "Session.hpp"

namespace MNN {
    SessionId MultiSession::sNextSessionId = 0;

    SessionId MultiSession::addSession(Session* session) {
        SessionId currentId = sNextSessionId++;
        mSessions[currentId] = session;
        return currentId;
    }

    ErrorCode MultiSession::runSequence(const std::vector<SessionId> &requests, bool sync) {
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

    ErrorCode MultiSession::runParallel(const std::vector<SessionId> &requests, bool sync) {
        return NO_ERROR;
    }
}