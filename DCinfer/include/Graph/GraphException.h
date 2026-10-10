#pragma once
#include "Exception.h"

namespace DC {

/// @brief Graph 组件专用异常，携带错误类型枚举供精确分类处理。
class GraphException : public Exception {
public:
	enum class ErrorType {
		NodeNotFound,        ///< 目标节点不存在
		DuplicateNode,       ///< 同名节点重复添加
		PortNotFound,        ///< 端口不存在于节点 Schema
		DirectConnect,       ///< 两个非 Connector 节点直连被拒
		NoDeclaration,       ///< submit 时未声明输出期望
		DuplicateTask,       ///< 同一 taskId 的活动任务被重复提交
		DuplicateBinding,    ///< 图级绑定别名重复
		InvalidBinding,      ///< 图级绑定缺别名
		FeedFailed,          ///< feedInput 时 Node::setInput 失败
		Frozen,              ///< 图已冻结，构建 API 拒绝
		ExecutionFailed,     ///< 节点 tryExecute 抛出 NodeException
		PropagateFailed,     ///< 数据传播链中写下游输入失败
		UnreachableDeclaration, ///< 声明目标在拓扑上不可达
		DuplicateEdge,       ///< 端口已有连接
		NonTerminalPort,     ///< 输出取数端口非终端（有出边）
		Other                ///< 其他未分类的错误
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
