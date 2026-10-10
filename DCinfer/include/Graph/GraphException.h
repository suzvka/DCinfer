#pragma once
#include "Exception.h"

namespace DC {

/// @brief Graph 组件专用异常，携带错误类型枚举供精确分类处理。
class GraphException : public Exception {
public:
	enum class ErrorType {
		NodeNotFound,
		DuplicateNode,
		PortNotFound,
		DirectConnect,
		NoDeclaration,
		DuplicateTask,
		DuplicateBinding,
		InvalidBinding,
		FeedFailed,
		Frozen,
		ExecutionFailed,
		PropagateFailed,
		UnreachableDeclaration,
		DuplicateEdge,
		NonTerminalPort,
		Other
	};

	GraphException(ErrorType errorType = ErrorType::Other, const std::string& source = "Unknown",
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
		case ErrorType::NodeNotFound:
			errorStr = "Node Not Found";
			break;
		case ErrorType::DuplicateNode:
			errorStr = "Duplicate Node";
			break;
		case ErrorType::PortNotFound:
			errorStr = "Port Not Found";
			break;
		case ErrorType::DirectConnect:
			errorStr = "Direct Connect Forbidden";
			break;
		case ErrorType::NoDeclaration:
			errorStr = "No Output Declaration";
			break;
		case ErrorType::DuplicateTask:
			errorStr = "Duplicate Task";
			break;
		case ErrorType::DuplicateBinding:
			errorStr = "Duplicate Binding";
			break;
		case ErrorType::InvalidBinding:
			errorStr = "Invalid Binding";
			break;
		case ErrorType::FeedFailed:
			errorStr = "Feed Failed";
			break;
		case ErrorType::Frozen:
			errorStr = "Graph Frozen";
			break;
		case ErrorType::ExecutionFailed:
			errorStr = "Execution Failed";
			break;
		case ErrorType::PropagateFailed:
			errorStr = "Propagate Failed";
			break;
		case ErrorType::UnreachableDeclaration:
			errorStr = "Unreachable Output Declaration";
			break;
		case ErrorType::DuplicateEdge:
			errorStr = "Edge Already Connected";
			break;
		case ErrorType::NonTerminalPort:
			errorStr = "Non-Terminal Output Port";
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
