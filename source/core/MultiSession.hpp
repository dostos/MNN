
#ifndef MultiSession_hpp
#define MultiSession_hpp

#include <map>
#include <memory>
#include <vector>
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "Pipeline.hpp"
#include "Schedule.hpp"
#include "SizeComputer.hpp"
#include <MNN/Tensor.hpp>


namespace MNN {
class Session;
class MultiPipeline;
typedef int64_t SessionId;

class MNN_PUBLIC MultiSession {
public:
    SessionId addSession(Session *session);

    // Prepare : alloc tensors per session
    ErrorCode prepare();

    ErrorCode runSequence(const std::set<SessionId> &requests, bool sync = false);
    ErrorCode runParallel(const std::set<SessionId> &requests, bool sync = false);

private:
    static SessionId sNextSessionId;
    std::map<SessionId, Session *> mSessions;

    // Cache for multi-session execution
    class MultiSessionCache {
    public:
        MultiSessionCache(std::vector<Session *> sessions);

        ErrorCode prepare();
        ErrorCode run();

    private:
        // sessions that should run parallel 
        const std::vector<Session *> mSessions;
        std::vector<std::shared_ptr<MultiPipeline>> mMultiPipelines;
    };

    std::map<std::set<SessionId>, std::shared_ptr<MultiSessionCache>> mMultiSessionCaches;
};
} // namespace MNN

#endif // MultiSession_hpp