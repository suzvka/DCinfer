#pragma once
#include "Exception.h"

namespace DC {

/// @brief Node 组件专用异常，携带错误类型枚举供精确分类处理。
class NodeException : public Exception {
public:
	enum class ErrorType {
		PortNotFound, ///< 端口名不在 Schema 端口列表中
		TaskNotFound, ///< 指定 taskId 的任务不存在
		SchemaError, ///< Schema 校验失败
		NotReady, ///< 任务输入尚未全部就绪，不可执行
		Reentrant, ///< 节点已在执行其他任务（拒绝重入）
		ExecutionFailed, ///< RunFn 抛出异常或返回失败
		TypeMismatch, ///< 输入值与端口声明的类型不一致
		OutputNotProduced, ///< 未产出 Schema 声明的全部输出
		InternalError, ///< 节点内部状态不一致
		Frozen, ///< 节点已冻结，配置面不可变
		Other ///< 其他未分类的错误
	};

	NodeException(ErrorType errorType = ErrorType::Other, const std::string& source = "Unknown",
				  const std::string& message = "No message", Level level = Level::Error)
		: _errorType(errorType), Exception(source, composeMessage(errorType, message), level) {}

	ErrorType getErrorType() const noexcept {
		return _errorType;
	}

private:
	ErrorType _errorType;

	static std::string composeMessage(ErrorType errorType, const std::string& message) {
		std::string errorStr;
		switch (errorType) {
		case ErrorType::PortNotFound:
			errorStr = "Port Not Found";
			break;
		case ErrorType::TaskNotFound:
			errorStr = "Task Not Found";
			break;
		case ErrorType::SchemaError:
			errorStr = "Schema Error";
			break;
		case ErrorType::NotReady:
			errorStr = "Not Ready";
			break;
		case ErrorType::Reentrant:
			errorStr = "Reentrant";
			break;
		case ErrorType::ExecutionFailed:
			errorStr = "Execution Failed";
			break;
		case ErrorType::TypeMismatch:
			errorStr = "Type Mismatch";
			break;
		case ErrorType::OutputNotProduced:
			errorStr = "Output Not Produced";
			break;
		case ErrorType::InternalError:
			errorStr = "Internal Error";
			break;
		case ErrorType::Frozen:
			errorStr = "Frozen";
			break;
		case ErrorType::Other:
			errorStr = "Other";
			break;
		}

		if (!message.empty()) {
			errorStr += " - " + message;
		}

		return errorStr;
	}
};
} // namespace DC
