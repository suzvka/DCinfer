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
/// modelPath 处理（引擎自定义不透明信息）：
/// - 反序列化时 modelPath 原样保留：GraphCompiler 不拼接、不校验、不做
///   任何文件系统解释——相对路径、URL、模型标识符均为合法取值，透传给
///   引擎适配层消费（与 engineConfig 的"一字段一语义"哲学一致）。
/// - 序列化（.dcg）时 modelPath 被替换为归档内相对路径（models/ 前缀）
///   以便模型文件随归档分发；.json 序列化原样写出。
///
/// 引擎节点（type 已注册 EngineRegistry）反序列化语义：
/// - 编译期只物化节点、不创建引擎实例（不调用 createEngine、不加载模型）：
///   节点经 EngineRegistry::createLazyNode 构造，schema 取 JSON 声明的
///   inputs/outputs（不做实例推导、不被覆盖），工厂提供引擎 RunFn。
///   引擎实例由宿主在冻结前经 getOrCreateEngine + Node::bindEngine 注入
///   （典型：执行机侧解析 modelPath 后加载）；实例缓存键为该字符串本身。
/// - JSON 声明的 schema 为空（无端口）时输出警告：接线与执行均以节点
///   schema 为准，空声明意味着该节点不可接线。
/// - 引擎未注册工厂时回退骨架节点（RunFn=nullptr）并输出警告，
///   保留 JSON 声明 schema，保证图结构完整可序列化。
///
/// .dcg 归档模型：compileFile(.dcg) 只读取 graph.json，不解压模型文件；
/// 资源就绪由宿主自行处理——DcgArchive::extractOne 提供归档内安全解压
/// （路径越界/符号链接/解包预算防御），供宿主在绑定期/执行期使用。
///
/// 动态维度 shape 编码：
/// - Node::Port::shape 类型为 Tensor::Shape = std::vector<int64_t>
///   （DCinfer/include/Tensor/Tensor.hpp），本身即可表达 ONNX 动态维度 -1；
///   DCIr 序列化/反序列化对 shape 直接 int64_t 直通（JSON -1 ↔ 内存 -1），
///   roundtrip 稳定，不经过任何有符号/无符号转换。
///   （两侧均须 int64_t 直通，不得引入 static_cast<size_t> 等
///   有符号/无符号转换——回归测试覆盖 -1 往返对称性。）
/// - 已知边界（设计决策，非缺陷）：TensorData::Shape（TensorData.h）=
///   std::vector<size_t> 只能表达确定形状——data 层的数据必然有确定尺寸，
///   负数尺寸无意义；-1 动态维度仅存在于 Tensor/schema 的"形状声明"层
///   （Tensor::Shape = std::vector<int64_t>）。若以含 -1 的 schema shape
///   构造实际 TensorData（如 ORT onnxToDC 输出动态形状张量），维度会隐式
///   转换为 size_t::max 且形状乘积溢出，属声明层与数据层的语义边界
///   （核心库保持现状，不改造数据层表达）。
///
/// 引擎实例生命周期：
/// - 编译期不创建实例；宿主经 getOrCreateEngine 显式加载（createEngine
///   钩子负责 modelPath 的解释与失败语义），节点经 Node::bindEngine 持有
///   共享句柄，实例存活期覆盖节点存活期。
/// - releaseEngine / releaseAllEngines 仅移除缓存条目：仍被节点持有的
///   实例安全存活，实际销毁发生在最后一个共享句柄释放时。
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
	/// @throws GraphException 若 JSON 解析失败或图结构不合法
	static void compileString(InferGraph& graph, std::string_view json);

	// ── 序列化 ──

	/// @brief 将推理图序列化为文件（自动识别 .json 或 .dcg 扩展名）
	/// @param graph 推理图
	/// @param path 输出文件路径（.json → 纯 JSON；.dcg → ZIP 打包图+模型）
	static void serialize(const InferGraph& graph, std::string_view path);

private:
	// ── 反序列化辅助 ──

	/// @brief 反序列化内部入口（compileString 与 compileFile(.dcg) 共用）
	static void compileInternal(InferGraph& graph, const nlohmann::json& root);

	/// @brief 从解析好的 JSON 填充 InferGraph
	static void buildGraph(InferGraph& graph, const nlohmann::json& root);

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
