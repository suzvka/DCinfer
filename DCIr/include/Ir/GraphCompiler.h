#pragma once

#include "InferGraph.h"

#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace DC::Ir {

/// @brief 推理图编译器：.json 与 .dcg 格式与 InferGraph 双向转换，.dcg 为 ZIP 打包图与模型。
///
/// 编译期只物化节点，不创建引擎实例；引擎实例由宿主在冻结前经 getOrCreateEngine
/// 与 Node::bindEngine 注入，缓存键为 modelPath。
/// modelPath 反序列化时原样透传，.dcg 序列化时替换为归档内 models/ 相对路径；
/// compileFile(.dcg) 只读取 graph.json，模型解压由宿主经 DcgArchive::extractOne 完成。
/// Node::Port::shape 以 int64_t 直通，-1 为动态维度，不得引入有符号与无符号转换；
/// TensorData::Shape 为 size_t，不能表达 -1，动态维度只存在于声明层。
class GraphCompiler {
public:
	/// @brief 从 .json 或 .dcg 文件构建推理图，失败抛 GraphException。
	static void compileFile(InferGraph& graph, std::string_view path);

	/// @brief 从 JSON 字符串构建推理图，失败抛 GraphException。
	static void compileString(InferGraph& graph, std::string_view json);

	/// @brief 将推理图序列化为文件；.json 为纯 JSON，.dcg 为 ZIP 打包图与模型。
	static void serialize(const InferGraph& graph, std::string_view path);

private:
	static void compileInternal(InferGraph& graph, const nlohmann::json& root);
	static void buildGraph(InferGraph& graph, const nlohmann::json& root);
	static void rebuildEdges(InferGraph& graph, const nlohmann::json& edgesJson);

	static nlohmann::json graphToJson(const InferGraph& graph);
	static nlohmann::json portToJson(const Node::Port& port);
	static nlohmann::json edgesToJson(const InferGraph& graph);
};

} // namespace DC::Ir
