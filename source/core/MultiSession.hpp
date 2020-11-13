
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
typedef int64_t SessionId;

class MNN_PUBLIC MultiSession {
public:
    SessionId addSession(Session *session);

    ErrorCode runSequence(const std::vector<SessionId> &requests, bool sync = false);
    ErrorCode runParallel(const std::vector<SessionId> &requests, bool sync = false);

private:
    static SessionId sNextSessionId;
    std::map<SessionId, Session *> mSessions;
};
} // namespace MNN

#endif // MultiSession_hpp