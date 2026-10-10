#pragma once
#include "Exception.h"

namespace DC {

/// @brief Tensor 组件专用异常，携带错误类型枚举供精确分类处理。
class TensorException : public Exception {
public:
	enum class ErrorType {
		TypeMismatch,
		ShapeMismatch,
		InvalidPath,
		InvalidShape,
		NotAScalar,
		NotData,
		Frozen,
		Other
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
