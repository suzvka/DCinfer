#pragma once
#include <string>
#include <vector>

namespace DC {

	/// @brief 张量元数据，描述张量的逻辑类型、尺寸和命名信息。
	///
	/// TensorMeta 不持有实际数据，仅记录张量的类型标签、
	/// 元素字节数、约束形状和名称。RuleShape 用于在推理图
	/// 的上下游连接时校验形状兼容性：空 shape 表示跳过检查，
	/// -1 表示该维度为动态维度。
	struct TensorMeta {

		/// @brief 张量逻辑类型枚举。
		///
		/// 每个枚举值对应一类 C++ 类型族，映射关系在 setTypeMap() 中注册。
		enum class TensorType {
			Float,  ///< 浮点类型族：float、double
			Int,    ///< 有符号整数族：int8_t ~ int64_t
			Uint,   ///< 无符号整数族：uint8_t ~ uint64_t
			Bool,   ///< 布尔类型：bool
			Char,   ///< 字符类型：char、unsigned char
			Data,   ///< 原始字节块类型：std::vector<std::byte> 等
			Void    ///< 未指定 / 通配类型
		};

	public:
		/// @brief 默认构造，自动初始化类型映射表。
		TensorMeta();

		/// @brief 确保 C++ 类型到 TensorType 的映射已注册（线程安全，仅执行一次）。
		static void ensureTypeMap();

		/// @brief 张量名称，用于日志和错误消息中的标识。
		std::string name = "";

		/// @brief 单元素字节数。例如 float 为 4，double 为 8。
		size_t typeSize = 0;

		/// @brief 规则形状（RuleShape）。
		/// - 若为空，则不进行形状检查。
		/// - 某一维为 -1 表示该维度为动态维度，checkShape() 时跳过该维度的比较。
		std::vector<int64_t> shape = {};

		/// @brief 张量逻辑类型标签。
		TensorType type = TensorType::Void;

		/// @brief 检查给定形状是否符合规则形状的约束。
		/// @param currentShape 当前的动态形状。
		/// @return true 若形状匹配（空规则、维度数匹配且每维相等或规则维为 -1）。
		bool checkShape(const std::vector<int64_t>& currentShape) const;

	private:
		static void setTypeMap();
	};
} // namespace DC