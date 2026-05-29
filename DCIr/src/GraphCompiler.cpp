#include "Ir/GraphCompiler.h"

#include "GraphException.h"

namespace DC::Ir {

InferGraph GraphCompiler::compileFile(std::string_view path) {
	// TODO: 实现 JSON 文件解析 → InferGraph 构建
	throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::compileFile",
						 "not yet implemented");
}

InferGraph GraphCompiler::compileString(std::string_view json) {
	// TODO: 实现 JSON 字符串解析 → InferGraph 构建
	throw GraphException(GraphException::ErrorType::Other, "GraphCompiler::compileString",
						 "not yet implemented");
}

} // namespace DC::Ir
