#pragma once
#include "Exception.h"

namespace DC {

    /// @brief Node 组件专用异常，携带错误类型枚举以支持精确的错误分类处理。
    ///
    /// 所有公开的 Node API 在校验或执行失败时均可能抛出此类异常。
    /// 可通过 getErrorType() 获取具体错误类型进行针对性处理。
    class NodeException : public Exception {
    public:
        /// @brief 错误类型枚举，覆盖 Node 组件中常见的语义错误。
        enum class ErrorType {
            PortNotFound,      ///< 端口名不在 Schema 输入/输出端口列表中
            TaskNotFound,      ///< 指定 taskId 的任务不存在
            SchemaError,       ///< Schema 校验失败（端口名重复、typeSize 与类型不一致等）
            NotReady,          ///< 任务输入尚未全部就绪，不可执行
            Reentrant,         ///< 节点已在执行其他任务（Exclusive 策略下拒绝重入）
            ExecutionFailed,   ///< RunFn 执行过程中抛出异常或返回失败状态
            TypeMismatch,      ///< 输入值与端口声明的类型不一致
            OutputNotProduced, ///< RunFn 执行后未产出 Schema 声明的全部输出端口
            InternalError,     ///< 节点内部状态不一致或操作不合法
            Other              ///< 其他未分类的错误
        };

        /// @brief 构造 NodeException。
        /// @param errorType 错误类型（默认 Other）。
        /// @param source 错误来源组件名称（默认 "Unknown"）。
        /// @param message 附加错误消息（默认 "No message"）。
        /// @param level 严重级别（默认 Error）。
        NodeException(
            ErrorType errorType = ErrorType::Other,
            const std::string& source = "Unknown",
            const std::string& message = "No message",
            Level level = Level::Error
        ) : _errorType(errorType), Exception(source, composeMessage(errorType, message), level) {}

        /// @brief 获取错误类型。
        ErrorType getErrorType() const noexcept {
            return _errorType;
        }

    private:
        ErrorType _errorType;

        static std::string composeMessage(
            ErrorType errorType,
            const std::string& message
        ) {
            std::string errorStr;
            switch (errorType) {
            case ErrorType::PortNotFound:      errorStr = "Port Not Found";      break;
            case ErrorType::TaskNotFound:      errorStr = "Task Not Found";      break;
            case ErrorType::SchemaError:       errorStr = "Schema Error";        break;
            case ErrorType::NotReady:          errorStr = "Not Ready";           break;
            case ErrorType::Reentrant:         errorStr = "Reentrant";           break;
            case ErrorType::ExecutionFailed:   errorStr = "Execution Failed";    break;
            case ErrorType::TypeMismatch:      errorStr = "Type Mismatch";       break;
            case ErrorType::OutputNotProduced: errorStr = "Output Not Produced"; break;
            case ErrorType::InternalError:     errorStr = "Internal Error";      break;
            case ErrorType::Other:             errorStr = "Other";               break;
            }

            if (!message.empty()) {
                errorStr += " - " + message;
            }

            return errorStr;
        }
    };
}
