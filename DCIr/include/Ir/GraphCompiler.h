#pragma once

#include "InferGraph.h"

#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace DC::Ir {

/// @brief 推理图编译器：JSON ↔ InferGraph 双向转换
///
/// 支持两种文件格式：
/// - .json  纯 JSON 图描述文件
/// - .dcg   zip 打包的推理图（graph.json + model files）
///
/// modelPath 处理：
/// - 序列化时，绝对路径转为 zip 内相对路径（models/ 前缀）
/// - 反序列化时，相对路径拼接解压目录为绝对路径
class GraphCompiler {
public:
	// ── 反序列化 ──

	/// @brief 从文件构建推理图（支持 .json 和 .dcg）
	/// @param graph 输出参数，反序列化结果写入此对象
	/// @param path 图文件路径
	/// @throws GraphException 若 JSON 解析失败或图结构不合法
	static void compileFile(InferGraph& graph, std::string_view path);

	/// @brief 从 JSON 字符串构建推理图
	/// @param graph 输出参数，反序列化结果写入此对象
	/// @param json JSON 图描述字符串
	/// @param baseDir 模型文件基础目录（modelPath 为相对路径时拼接，默认为当前目录）
	/// @throws GraphException 若 JSON 解析失败或图结构不合法
	static void compileString(InferGraph& graph, std::string_view json,
							 std::filesystem::path baseDir = std::filesystem::current_path());

	// ── 序列化 ──

	/// @brief 将推理图序列化为文件（自动识别 .json 或 .dcg 扩展名）
	/// @param graph 推理图
	/// @param path 输出文件路径（.json → 纯 JSON；.dcg → ZIP 打包图+模型）
	static void serialize(const InferGraph& graph, std::string_view path);

private:
	// ── 反序列化辅助 ──

	/// @brief 从解析好的 JSON 填充 InferGraph
	static void buildGraph(InferGraph& graph, const nlohmann::json& root, const std::filesystem::path& baseDir);

	/// @brief 处理边的重连：按 mode 分组，重建连接器
	static void rebuildEdges(InferGraph& graph, const nlohmann::json& edgesJson);

	// ── 序列化辅助 ──

	/// @brief InferGraph → JSON
	static nlohmann::json graphToJson(const InferGraph& graph);

	/// @brief Node.Schema 端口 → JSON
	static nlohmann::json portToJson(const Node::Port& port);

	/// @brief 推断节点间边的 mode 并折叠连接器
	static nlohmann::json edgesToJson(const InferGraph& graph);
};

} // namespace DC::Ir
