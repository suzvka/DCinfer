#pragma once
#include <vector>
#include <cstdint>
#include <type_traits>

namespace DC {
	//推导输入张量形状==================================================================
	// 辅助模板：判断是否为std::vector-------------------------------------
	template <typename T>
	struct is_std_vector : std::false_type {};
	template <typename T, typename Alloc>
	struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};
	// 推导形状的核心递归模板----------------------------------------------
	template <typename T>
	std::vector<int64_t> inferShape(const T& value) {
		// 如果是标量，返回空形状
		return {};
	}
	template <typename T>
	std::vector<int64_t> inferShape(const std::vector<T>& vec) {
		// 当前维度是vector的大小
		std::vector<int64_t> shape = { static_cast<int64_t>(vec.size()) };
		// 如果内部还是vector，递归推导
		if constexpr (is_std_vector<T>::value) {
			std::vector<int64_t> subShape = inferShape(vec[0]);
			shape.insert(shape.end(), subShape.begin(), subShape.end());
		}
		return shape;
	}
	//==================================================================推导输入张量形状
}