#pragma once
#include "Exception.h"

namespace DC {

/// @brief Node 组件专用异常，携带错误类型枚举供精确分类处理。
class NodeException : public Exception {
public:
	enum class ErrorType {
		PortNotFound,
		TaskNotFound,
		SchemaError,
		NotReady,
		Reentrant,
		ExecutionFailed,
		TypeMismatch,
		OutputNotProduced,
		InternalError,
		Frozen,
		Other
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
