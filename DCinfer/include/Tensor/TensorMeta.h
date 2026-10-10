#pragma once
#include <string>
#include <vector>

namespace DC {

/// @brief 张量元数据：类型标签、元素字节数、名称与规则形状。
///        shape 为空则跳过检查，-1 表示动态维度。
struct TensorMeta {

	enum class TensorType {
		Float,
		Int,
		Uint,
		Bool,
		Char,
		Data,
		Void
	};

public:
	TensorMeta();

	/// @brief 确保 C++ 类型到 TensorType 的映射已注册；线程安全，仅执行一次。
	static void ensureTypeMap();

	std::string name = "";

	size_t typeSize = 0;

	std::vector<int64_t> shape = {};

	TensorType type = TensorType::Void;

	bool checkShape(const std::vector<int64_t>& currentShape) const;

	static std::string typeToString(TensorType type);

	/// @brief 无法识别的字符串返回 Void。
	static TensorType stringToType(const std::string& str);

private:
	static void setTypeMap();
};
} // namespace DC