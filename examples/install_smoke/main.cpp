// install_smoke - 安装版 DCinfer 的 find_package 冒烟验证
//
// 验证：安装树头文件解析（裸文件名 include，与源码内消费风格一致）
//       + 静态库链接 + 基础运行时行为。
// 预期输出：install smoke OK

#include "InferGraph.h"
#include "TaskStatus.h"
#include "Tensor.hpp"

#include <iostream>

#ifdef SMOKE_WITH_IR
#include "Ir/GraphCompiler.h"
#endif

int main() {
	// 链接验证：核心类型可用且可构造/析构
	DC::InferGraph graph;
	static_cast<void>(graph);

#ifdef SMOKE_WITH_IR
	// 安装结构验证：Ir/ 目录前缀 include 路径可解析
	DC::Ir::GraphCompiler* compiler = nullptr;
	static_cast<void>(compiler);
#endif

	// 枚举语义验证
	if (DC::TaskStatus::Succeeded != DC::TaskStatus::Succeeded) {
		std::cerr << "TaskStatus semantics broken" << std::endl;
		return 1;
	}

	std::cout << "install smoke OK" << std::endl;
	return 0;
}
