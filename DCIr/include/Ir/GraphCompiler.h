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
///
/// 引擎节点（type 已注册 EngineRegistry）反序列化语义：
/// - createNode(engineType, name, modelPath) 一次加载并缓存引擎实例，
///   节点 Schema 从实例推导（引擎的 getInputPorts/getOutputPorts）——
///   JSON 中携带的 inputs/outputs 被引擎推导结果覆盖，不参与节点构造。
///   引擎未注册端口推导钩子时节点 Schema 为空，编译期输出警告。
/// - JSON 无 modelPath 字段的引擎节点：不以 "" 为缓存 key 创建引擎实例
///   （空路径加载模型会抛异常中断编译），回退为骨架节点并输出警告，
///   保留 JSON schema。
/// - createNode 失败（模型缺失 / 引擎未注册 createEngine）时节点被跳过，
///   输出含节点名、引擎类型、modelPath 的错误诊断，图不完整可感知。
///
/// 动态维度 shape 编码：
/// - Node::Port::shape 类型为 Tensor::Shape = std::vector<int64_t>
///   （DCinfer/include/Tensor/Tensor.hpp），本身即可表达 ONNX 动态维度 -1；
///   DCIr 序列化/反序列化对 shape 直接 int64_t 直通（JSON -1 ↔ 内存 -1），
///   roundtrip 稳定，不经过任何有符号/无符号转换。
///   （历史缺陷：jsonToPort 曾以 static_cast<size_t> 读入维度，把 JSON -1
///   转为 0xFFFFFFFFFFFFFFFF，与序列化端不对称——已修复并附测试。）
/// - 已知边界（设计决策，非缺陷）：TensorData::Shape（TensorData.h）=
///   std::vector<size_t> 只能表达确定形状——data 层的数据必然有确定尺寸，
///   负数尺寸无意义；-1 动态维度仅存在于 Tensor/schema 的"形状声明"层
///   （Tensor::Shape = std::vector<int64_t>）。若以含 -1 的 schema shape
///   构造实际 TensorData（如 ORT onnxToDC 输出动态形状张量），维度会隐式
///   转换为 size_t::max 且形状乘积溢出，属声明层与数据层的语义边界，
///   按负责人决策不纳入核心库改造（2026-08 评审结论）。
///
/// .dcg 与引擎实例缓存的生命周期：
/// - compileFile(.dcg) 解压模型到临时目录 → createNode 加载（实例缓存键 =
///   engineType + 临时绝对路径）→ 临时文件立即删除。
/// - 实例驻留内存不受删除影响；但缓存键指向的文件路径永久失效。重复编译
///   同一 .dcg 解压到新临时目录（键不同）→ 新增实例缓存条目，旧条目成为
///   永不命中的孤儿，直到 releaseAllEngines() 释放。
/// - 约束：dcg 重复编译后，旧实例缓存条目不可再命中；运行中的图保持实例
///   缓存（勿在中途 releaseAllEngines，节点持有非拥有实例指针），释放后
///   重新编译即可（会重新解压加载）。惰性加载引擎不受保护（临时文件已删）。
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
