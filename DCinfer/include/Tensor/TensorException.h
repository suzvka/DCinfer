#pragma once
#include "Exception.h"

namespace DC {

/// @brief Tensor 组件专用异常，携带错误类型枚举以支持精确的错误分类处理。
///
/// 所有公开的 Tensor API 在校验失败时均可能抛出此类异常。
/// 可通过 getErrorType() 获取具体错误类型进行针对性处理。
class TensorException : public Exception {
public:
	/// @brief 错误类型枚举，覆盖 Tensor 组件中常见的语义错误。
	enum class ErrorType {
		TypeMismatch, ///< 类型不匹配：传入的 C++ 类型与张量声明的 typeSize 不一致
		ShapeMismatch, ///< 形状不匹配：数据形状与规则形状或操作预期的形状不符
		InvalidPath, ///< 索引路径无效：维度越界、路径长度超出张量秩
		InvalidShape, ///< 无效形状：形状参数本身不合法
		NotAScalar, ///< 非标量：试图将非单元素的子视图作为标量读取
		NotData, ///< 无数据：尝试访问尚未填充数据的张量
		Other ///< 其他未分类的错误
	};

	/// @brief 构造 TensorException。
	/// @param errorType 错误类型（默认 Other）。
	/// @param source 错误来源组件名称（默认 "Unknown"）。
	/// @param message 附加错误消息（默认 "No message"）。
	/// @param level 严重级别（默认 Error）。
	TensorException(ErrorType errorType = ErrorType::Other, const std::string& source = "Unknown",
					const std::string& message = "No message", Level level = Level::Error)
		: _errorType(errorType), Exception(source, composeMessage(errorType, message), level) {}

	/// @brief 获取错误类型。
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
