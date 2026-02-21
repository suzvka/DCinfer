#pragma once
#include <string>
#include <vector>

namespace DC {
	// 张量元数据
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

		enum class DataState {
			Overflow,
			TypeError,
			Empty,
			Ready
		};

		enum class MismatchPolicy {
			Auto,
			Truncate,
			Bitcast,
			Convert,
			Throw
		};

	public:
		TensorMeta();

		static void ensureTypeMap();

		std::string name = "";
		size_t typeSize = 0;
		// 规则形状（RuleShape）。若为空，则不进行形状检查。
		// 约定：某一维为 -1 表示该维度为动态维度，check() 时跳过该维度的比较。
		std::vector<int64_t> shape = {};
		TensorType type = TensorType::Void;

		// 检查形状是否符合规则
		bool check(const std::vector<int64_t>& currentShape) const;

	private:
		static void setTypeMap();
	};
} // namespace DC