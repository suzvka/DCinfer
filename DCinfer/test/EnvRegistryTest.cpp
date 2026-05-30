// EnvRegistry 单元测试：注册、创建、缓存、释放
#include <iostream>
#include <stdexcept>
#include <string>

#include "EnvRegistry.h"

using namespace DC;

// ── 辅助：带析构计数的 Mock 环境 ──
struct MockEnv {
	int id;
	int* destroyCount = nullptr;

	MockEnv(int id, int* counter) : id(id), destroyCount(counter) {}
	~MockEnv() {
		if (destroyCount)
			++(*destroyCount);
	}
};

static void runTests() {
	auto& reg = EnvRegistry::instance();

	// ── Test 1: 注册环境 ──
	{
		bool ok = reg.registerEnv("Mock", []() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(42, nullptr);
		});
		if (!ok)
			throw std::runtime_error("registerEnv failed");
	}
	std::cout << "Test 1 passed: register environment" << std::endl;

	// ── Test 2: 重复注册被拒绝 ──
	{
		bool ok = reg.registerEnv("Mock", []() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(0, nullptr);
		});
		if (ok)
			throw std::runtime_error("duplicate registration should fail");
	}
	std::cout << "Test 2 passed: duplicate registration rejected" << std::endl;

	// ── Test 3: 空 envType 注册被拒绝 ──
	{
		bool ok = reg.registerEnv("", []() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(0, nullptr);
		});
		if (ok)
			throw std::runtime_error("empty envType should be rejected");
	}
	std::cout << "Test 3 passed: empty envType rejected" << std::endl;

	// ── Test 4: 空工厂注册被拒绝 ──
	{
		bool ok = reg.registerEnv("NullFactory", nullptr);
		if (ok)
			throw std::runtime_error("null factory should be rejected");
	}
	std::cout << "Test 4 passed: null factory rejected" << std::endl;

	// ── Test 5: hasEnv 查询 ──
	{
		if (!reg.hasEnv("Mock"))
			throw std::runtime_error("hasEnv should be true for registered env");
		if (reg.hasEnv("Nonexistent"))
			throw std::runtime_error("hasEnv should be false for unknown env");
	}
	std::cout << "Test 5 passed: hasEnv query" << std::endl;

	// ── Test 6: getOrCreate 按名创建实例 ──
	{
		auto* env = reg.getOrCreate("Mock");
		if (!env)
			throw std::runtime_error("getOrCreate returned null");
		auto* mock = static_cast<MockEnv*>(env);
		if (mock->id != 42)
			throw std::runtime_error("mock env id mismatch: expected 42, got " + std::to_string(mock->id));
	}
	std::cout << "Test 6 passed: getOrCreate returns correct instance" << std::endl;

	// ── Test 7: 同一类型多次 getOrCreate 返回同一实例 ──
	{
		auto* env1 = reg.getOrCreate("Mock");
		auto* env2 = reg.getOrCreate("Mock");
		if (env1 != env2)
			throw std::runtime_error("getOrCreate should return same instance for same type");
		auto* mock1 = static_cast<MockEnv*>(env1);
		auto* mock2 = static_cast<MockEnv*>(env2);
		if (mock1->id != mock2->id)
			throw std::runtime_error("same instance should have same id");
	}
	std::cout << "Test 7 passed: getOrCreate returns cached instance" << std::endl;

	// ── Test 8: release 单环境（含 cleanup 调用）──
	{
		bool cleanupCalled = false;
		int destroyCount = 0;

		bool ok = reg.registerEnv("WithCleanup", [&destroyCount]() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(100, &destroyCount);
		},
								  [&cleanupCalled](void*) { cleanupCalled = true; });
		if (!ok)
			throw std::runtime_error("registerEnv with cleanup failed");

		// 先创建实例
		auto* env = reg.getOrCreate("WithCleanup");
		if (!env)
			throw std::runtime_error("getOrCreate WithCleanup failed");

		// 释放
		reg.release("WithCleanup");
		if (!cleanupCalled)
			throw std::runtime_error("cleanup should have been called on release");

		// 释放后 getOrCreate 应重新创建
		auto* env2 = reg.getOrCreate("WithCleanup");
		if (!env2)
			throw std::runtime_error("getOrCreate after release should create new instance");
		if (env2 == env)
			throw std::runtime_error("getOrCreate after release should return different instance");
	}
	std::cout << "Test 8 passed: release with cleanup" << std::endl;

	// ── Test 9: releaseAll 全部环境 ──
	{
		int destroyCountA = 0;
		int destroyCountB = 0;
		bool cleanupA = false;
		bool cleanupB = false;

		reg.registerEnv("RelA", [&destroyCountA]() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(1, &destroyCountA);
		}, [&cleanupA](void*) { cleanupA = true; });

		reg.registerEnv("RelB", [&destroyCountB]() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(2, &destroyCountB);
		}, [&cleanupB](void*) { cleanupB = true; });

		reg.getOrCreate("RelA");
		reg.getOrCreate("RelB");

		reg.releaseAll();

		if (!cleanupA || !cleanupB)
			throw std::runtime_error("releaseAll should call all cleanups");
		if (destroyCountA != 1 || destroyCountB != 1)
			throw std::runtime_error("releaseAll should destroy all instances");

		// 释放后 _instances 应为空（getOrCreate 应重新创建）
		auto* envA = reg.getOrCreate("RelA");
		if (!envA)
			throw std::runtime_error("getOrCreate after releaseAll should succeed");
	}
	std::cout << "Test 9 passed: releaseAll" << std::endl;

	// ── Test 10: 未注册类型 getOrCreate 返回 nullptr ──
	{
		auto* env = reg.getOrCreate("Ghost");
		if (env)
			throw std::runtime_error("getOrCreate for unregistered type should return nullptr");
	}
	std::cout << "Test 10 passed: unregistered getOrCreate returns nullptr" << std::endl;

	// ── Test 11: release 未缓存的类型不崩溃 ──
	{
		reg.registerEnv("NoCache", []() -> std::shared_ptr<void> {
			return std::make_shared<MockEnv>(0, nullptr);
		});
		// 从未调用 getOrCreate，直接 release
		reg.release("NoCache"); // should not crash
	}
	std::cout << "Test 11 passed: release uncached env is safe" << std::endl;

	std::cout << "\nAll EnvRegistry tests passed!" << std::endl;
}

int main() {
	try {
		runTests();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failure: " << e.what() << std::endl;
		return -1;
	}
}
