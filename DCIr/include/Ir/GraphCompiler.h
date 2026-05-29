#pragma once

#include "InferGraph.h"

#include <string>
#include <string_view>

namespace DC::Ir {

/// @brief 从 JSON 推理图描述文件构建 InferGraph 对象
///
/// 图编译器负责解析 JSON 格式的推理图描述，将其转换为可执行的 InferGraph。
/// 解析过程会调用 EngineRegistry 解析节点类型和引擎引用。
class GraphCompiler {
public:
	/// @brief 从 JSON 文件路径构建推理图
	/// @param path JSON 图描述文件路径
	/// @return 构建好的 InferGraph 对象
	/// @throws GraphException 若 JSON 解析失败或图结构不合法
	static InferGraph compileFile(std::string_view path);

	/// @brief 从 JSON 字符串构建推理图
	/// @param json JSON 图描述字符串
	/// @return 构建好的 InferGraph 对象
	/// @throws GraphException 若 JSON 解析失败或图结构不合法
	static InferGraph compileString(std::string_view json);
};

} // namespace DC::Ir
