#pragma once
#include "Exception.h"

namespace DC {

/// @brief Tensor 组件专用异常，携带错误类型枚举供精确分类处理。
class TensorException : public Exception {
public:
	enum class ErrorType {
		TypeMismatch, ///< C++ 类型与张量声明的 typeSize 不一致
		ShapeMismatch, ///< 数据形状与规则形状或操作预期不符
		InvalidPath, ///< 维度越界、路径长度超出张量秩
		InvalidShape, ///< 形状参数本身不合法
		NotAScalar, ///< 非单元素子视图被作为标量读取
		NotData, ///< 访问尚未填充数据的张量
		Frozen, ///< 已冻结，拒绝突变
		Other ///< 其他未分类的错误
	};

	TensorException(ErrorType errorType = ErrorType::Other, const std::string& source = "Unknown",
					const std::string& message = "No message", Level level = Level::Error)
		: _errorType(errorType), Exception(source, composeMessage(errorType, message), level) {}

	ErrorType getErrorType() const noexcept {
		return _errorType;
	}

private:
	ErrorType _errorType;

	std::string composeMessage(ErrorType errorType, const std::string& message) {
		std::string errorStr;
		switch (errorType) {
		case ErrorType::TypeMismatch:
			errorStr = "Type Mismatch";
			break;
		case ErrorType::ShapeMismatch:
			errorStr = "Shape Mismatch";
			break;
		case ErrorType::InvalidPath:
			errorStr = "Invalid Path";
			break;
		case ErrorType::InvalidShape:
			errorStr = "Invalid Shape";
			break;
		case ErrorType::NotAScalar:
			errorStr = "Not a Scalar";
			break;
		case ErrorType::NotData:
			errorStr = "Not Data";
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
