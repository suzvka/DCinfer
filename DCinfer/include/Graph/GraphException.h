#pragma once
#include "Exception.h"

namespace DC {

/// @brief Graph 组件专用异常，携带错误类型枚举以支持精确的错误分类处理。
///
/// 所有公开的 Graph API 在拓扑校验或运行时数据传播失败时均可能抛出此类异常。
/// 可通过 getErrorType() 获取具体错误类型进行针对性处理。
class GraphException : public Exception {
public:
	/// @brief 错误类型枚举，覆盖 Graph 组件中常见的语义错误。
	enum class ErrorType {
		NodeNotFound,        ///< 目标节点在图不存在
		DuplicateNode,       ///< 同名节点重复添加
		PortNotFound,        ///< 连线时端口在目标节点 Schema 中不存在
		DirectConnect,       ///< 两个非 Connector 节点直连被拒
		NoDeclaration,       ///< submit 时未声明输出期望
		FeedFailed,          ///< feedInput 时调用 Node::setInput 失败
		ExecutionFailed,     ///< 线程池中 Node::tryExecute 抛出 NodeException
		PropagateFailed,     ///< 协程传播链中写下游输入失败
		Other                ///< 其他未分类的错误
	};

	/// @brief 构造 GraphException。
	/// @param errorType 错误类型（默认 Other）。
	/// @param source    错误来源组件名称（默认 "Unknown"）。
	/// @param message   附加错误消息（默认 "No message"）。
	/// @param level     严重级别（默认 Error）。
	GraphException(ErrorType errorType = ErrorType::Other, const std::string& source = "Unknown",
				   const std::string& message = "No message", Level level = Level::Error)
		: _errorType(errorType), Exception(source, composeMessage(errorType, message), level) {}

	/// @brief 获取错误类型。
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
		case ErrorType::FeedFailed:
			errorStr = "Feed Failed";
			break;
		case ErrorType::ExecutionFailed:
			errorStr = "Execution Failed";
			break;
		case ErrorType::PropagateFailed:
			errorStr = "Propagate Failed";
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
