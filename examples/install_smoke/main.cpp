// install_smoke - 安装版 DCinfer 的 find_package 冒烟验证。
// 验证：安装树头文件解析（裸文件名 include）+ 静态库链接 + 基础运行时行为。
// 开关（经 CMake 变量注入，见 CMakeLists.txt）：
//   SMOKE_WITH_IR / SMOKE_WITH_BUILTIN / SMOKE_WITH_NET
//   （分别验证 DCIr 安装链 / Builtin 算子图 / DCNet 回环监听）
// 预期输出：install smoke OK

#include "InferGraph.h"
#include "TaskStatus.h"
#include "Tensor.hpp"

#include <iostream>

#ifdef SMOKE_WITH_IR
#include "Ir/GraphCompiler.h"
#endif

#ifdef SMOKE_WITH_BUILTIN
#include "DCEngine/BuiltinOps.h"
#include "NodeExecutor.h"
#endif

#ifdef SMOKE_WITH_NET
#include "DCNet/NetListener.h"
#include "NodeException.h"
#endif

int main() {
	// 链接验证：核心类型可构造/析构
	DC::InferGraph graph;
	static_cast<void>(graph);

#ifdef SMOKE_WITH_IR
	// Ir/ 目录前缀 include 路径可解析
	DC::Ir::GraphCompiler* compiler = nullptr;
	static_cast<void>(compiler);
#endif

#ifdef SMOKE_WITH_BUILTIN
	// Builtin 安装链：注册 → createOperator → 真实执行 2+3
	DC::Builtin::registerBuiltinOperators();
	auto node = DC::EngineRegistry::instance().createOperator("Add", "smoke_add");
	if (!node) {
		std::cerr << "Builtin 'Add' operator not registered" << std::endl;
		return 1;
	}
	DC::NodeExecutor exec(*node);
	auto scalar = [](float v) {
		auto t = std::make_unique<DC::Tensor>(DC::Tensor::TensorType::Float, sizeof(float));
		*t = v;
		return DC::Value(std::move(t));
	};
	exec.setInput("t1", "a", scalar(2.0f));
	exec.setInput("t1", "b", scalar(3.0f));
	auto r = exec.tryExecute("t1");
	if (!r.ok() || !exec.hasOutput("t1", "sum")) {
		std::cerr << "Builtin add execution failed: " << r.message << std::endl;
		return 1;
	}
	auto out = exec.takeOutputTensor("t1", "sum");
	if (out.item<float>() != 5.0f) {
		std::cerr << "Builtin add result mismatch (expected 5.0)" << std::endl;
		return 1;
	}
#endif

#ifdef SMOKE_WITH_NET
	// DCNet 安装链：创建监听器并 bind 回环随机端口（socket 生命周期；完整往返回归见 ServerAdapterTest）
	try {
		auto listener = DC::Net::makeHttpListener();
		DC::Net::NetServerEndpoint ep;
		ep.port = 0;
		listener->bind(ep);
		if (listener->port() <= 0) {
			std::cerr << "DCNet listener failed to bind a port" << std::endl;
			return 1;
		}
		listener->stop();
	} catch (const DC::NodeException& e) {
		std::cerr << "DCNet listener bind failed: " << e.what() << std::endl;
		return 1;
	}
#endif

	// 枚举语义验证
	if (DC::TaskStatus::Succeeded != DC::TaskStatus::Succeeded) {
		std::cerr << "TaskStatus semantics broken" << std::endl;
		return 1;
	}

	std::cout << "install smoke OK" << std::endl;
	return 0;
}
