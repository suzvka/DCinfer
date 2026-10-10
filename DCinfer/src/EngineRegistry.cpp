#include "EngineRegistry.h"
#include "Node.h"

#include <stdexcept>
#include <memory>

namespace DC {

namespace {

/// 实例缓存容量上限：超限按 LRU 驱逐非 loading 槽位；被节点持有的实例由共享句柄保活。
constexpr std::size_t kMaxCachedEngineInstances = 64;

} // namespace

EngineCore::~EngineCore() {
	// 释放钩子：仅最后一个共享句柄析构时调用一次；先以原生指针有序清理，再正常析构。
	if (_desc && _core && _desc->releaseEngineCore) {
		_desc->releaseEngineCore(_core.get());
	}
}

EngineInstance::~EngineInstance() {
	// 释放钩子：仅最后一个共享句柄析构时调用一次；成员逆序析构，_engine 先于 _core 释放。
	if (_desc && _engine && _desc->releaseModel) {
		_desc->releaseModel(_engine.get());
	}
}

static Value builtinToNative(const Tensor& t) {
	return Value(std::make_unique<Tensor>(t));
}

static Tensor builtinToDC(const void* native) {
	return Tensor(*static_cast<const Tensor*>(native));
}

static void ensureBuiltinEngine(EngineRegistry& reg) {
	EngineDescriptor desc;
	desc.engineType = "Builtin";
	desc.converter = {builtinToNative, builtinToDC};
	// Builtin 节点不经工厂创建，由 createNode(name, schema, fn) 直接构造
	desc.factory = nullptr;
	reg.registerEngine(desc);
}

EngineRegistry& EngineRegistry::instance() {
	static EngineRegistry inst;
	static std::once_flag builtinFlag;
	std::call_once(builtinFlag, ensureBuiltinEngine, std::ref(inst));
	return inst;
}

bool EngineRegistry::registerEngine(const EngineDescriptor& desc) {
	std::lock_guard lk(_mutex);
	if (desc.engineType.empty()) {
		return false;
	}

	if (_engines.contains(desc.engineType)) {
		return false; // 不允许重复注册
	}

	_engines[desc.engineType] = desc;
	return true;
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& engineType, const std::string& nodeName,
												 const void* engineConfig) const {
	// 锁内拷贝工厂，锁外调用：factory 是用户回调，可能重入注册表
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(engineType);
		if (it == _engines.end() || !it->second.factory) {
			return nullptr;
		}
		factory = it->second.factory;
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	params.engineConfig = engineConfig;
	return factory(params);
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& nodeName, Node::Schema schema,
												 Node::RunFn fn) const {
	return std::make_unique<Node>("Builtin", nodeName, std::move(schema), std::move(fn),
								  ResourceClass::Operator);
}

std::unique_ptr<Node> EngineRegistry::createNode(const std::string& engineType, const std::string& nodeName,
												 const std::string& modelPath) {
	// 加载并缓存引擎实例，从实例推导 Schema，再由 factory 构造节点
	auto engineInstance = getOrCreateEngine(engineType, modelPath);
	if (!engineInstance)
		return nullptr;

	// 锁内拷贝钩子与工厂，锁外调用防重入死锁
	std::function<std::vector<Node::Port>(const EngineInstance&)> getInputs;
	std::function<std::vector<Node::Port>(const EngineInstance&)> getOutputs;
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(engineType);
		if (it == _engines.end() || !it->second.factory)
			return nullptr;
		factory = it->second.factory;
		getInputs = it->second.getInputPorts;
		getOutputs = it->second.getOutputPorts;
	}

	// 从实例推导 Schema；未注册推导钩子时留空，由工厂兜底
	Node::Schema schema;
	if (getInputs && getOutputs) {
		schema.inputs = getInputs(*engineInstance);
		schema.outputs = getOutputs(*engineInstance);
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	params.engineInstance = engineInstance; 
	params.schema = std::move(schema);
	params.modelPath = modelPath;

	auto node = factory(params);
	if (node)
		node->setModelPath(modelPath);
	return node;
}

std::unique_ptr<Node> EngineRegistry::createLazyNode(const std::string& engineType, const std::string& nodeName,
													 Node::Schema schema) const {
	// 锁内拷贝工厂，锁外调用：factory 是用户回调，可能重入注册表
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(engineType);
		if (it == _engines.end() || !it->second.factory) {
			return nullptr;
		}
		factory = it->second.factory;
	}

	// 不创建实例与缓存；schema 以调用方声明值交给工厂，不绑定实例。
	NodeFactoryParams params;
	params.nodeName = nodeName;
	params.schema = std::move(schema);
	return factory(params);
}

std::string EngineRegistry::_makeEngineKey(const std::string& engineType, const std::string& modelPath) {
	return engineType + ":" + modelPath;
}

EngineCoreHandle EngineRegistry::getOrCreateEngineCore(const std::string& engineType) {
	// 快路径：ready 缓存命中
	{
		std::lock_guard lk(_mutex);
		auto it = _engineCores.find(engineType);
		if (it != _engineCores.end() && it->second.ready) {
			return it->second.ready;
		}
	}

	// single-flight：首个调用者为领导者，其余为跟随者
	std::promise<EngineCoreHandle> promise;
	std::shared_future<EngineCoreHandle> myFuture = promise.get_future().share();
	bool leader = false;
	std::function<EngineCore()> createEngineCore;
	{
		std::lock_guard lk(_mutex);
		auto& slot = _engineCores[engineType]; // 按需创建空槽位
		if (slot.ready) {
			return slot.ready; // 双重检查：他人已完成创建
		}
		if (slot.loading.valid()) {
			myFuture = slot.loading; // 跟随者等待首创建者的同一结果
		} else {
			auto engIt = _engines.find(engineType);
			if (engIt == _engines.end()) {
				_engineCores.erase(engineType); // 未注册：不留空槽位
				return nullptr;
			}
			slot.loading = myFuture; // 领导者登记 loading 条目
			createEngineCore = engIt->second.createEngineCore;
			leader = true;
		}
	}

	if (!leader) {
		// 跟随者：阻塞等待，失败透传领导者异常
		return myFuture.get();
	}

	// 领导者：锁外执行初始化回调，可安全重入 registry；无钩子时合成空核心，
	// 钩子返回空核心视同初始化失败。
	EngineCoreHandle handle;
	std::exception_ptr error;
	try {
		if (createEngineCore) {
			EngineCore core = createEngineCore();
			if (core) // 空核心 = 失败
				handle = std::make_shared<EngineCore>(std::move(core));
		} else {
			handle = std::make_shared<EngineCore>(); // 无引擎级资源：合成空核心
		}
		// 注入所属描述符；find 内部加锁；_engines 注册后不擦除，指针可长存。
		if (handle) {
			if (const EngineDescriptor* desc = find(engineType))
				handle->setDescriptor(desc);
		}
	} catch (...) {
		error = std::current_exception();
	}

	// 发布：锁内写 ready 或清失败槽位；set_value 放锁外，避免被唤醒者抢锁
	{
		std::lock_guard lk(_mutex);
		if (handle) {
			auto& slot = _engineCores[engineType];
			slot.ready = handle;
			slot.loading = {}; // 清除 loading 条目
		} else {
			_engineCores.erase(engineType); // 失败：清槽位，后续重试
		}
	}
	// error 必须拷贝进 promise 而非 move：移动后源置空、判定恒 false，异常在 GCC/Clang 静默丢失。
	if (error) {
		promise.set_exception(error);
		std::rethrow_exception(error); // 初始化异常透传给首个调用者
	}
	promise.set_value(handle);
	return handle;
}

EngineHandle EngineRegistry::getOrCreateEngine(const std::string& engineType, const std::string& modelPath) {
	auto key = _makeEngineKey(engineType, modelPath);

	// 快路径：ready 缓存命中
	{
		std::lock_guard lk(_mutex);
		auto it = _engineInstances.find(key);
		if (it != _engineInstances.end() && it->second.ready) {
			it->second.lastAccess = std::chrono::steady_clock::now(); // LRU 触碰
			return it->second.ready;
		}
	}

	// 引擎未注册或未注册加载钩子：零副作用返回 nullptr，不为无法产生实例的
	// 引擎创建核心；锁内拷贝加载钩子、锁外调用防重入死锁。
	std::function<EngineInstance(const EngineCore&, const std::string&)> loadModel;
	{
		std::lock_guard lk(_mutex);
		auto engIt = _engines.find(engineType);
		if (engIt == _engines.end() || !engIt->second.loadModel)
			return nullptr;
		loadModel = engIt->second.loadModel;
	}

	// 引擎级核心就绪：每 engineType 恰好一次，失败返回 nullptr
	auto core = getOrCreateEngineCore(engineType);
	if (!core)
		return nullptr;

	// single-flight：同 key 首个调用者为领导者，其余为跟随者
	std::promise<EngineHandle> promise;
	std::shared_future<EngineHandle> myFuture = promise.get_future().share();
	bool leader = false;
	{
		std::lock_guard lk(_mutex);
		auto& slot = _engineInstances[key]; // 按需创建空槽位
		if (slot.ready) {
			return slot.ready; // 双重检查：他人已完成创建
		}
		if (slot.loading.valid()) {
			myFuture = slot.loading; // 跟随者等待首创建者的同一结果
		} else {
			slot.loading = myFuture; // 领导者登记 loading 条目
			leader = true;
		}
	}

	if (!leader) {
		// 跟随者：阻塞等待，失败透传领导者异常
		return myFuture.get();
	}

	// 领导者：锁外执行加载回调，backend 可安全重入 registry，其他 key 创建不阻塞。
	EngineHandle handle;
	std::exception_ptr error;
	try {
		auto instance = loadModel(*core, modelPath);
		if (instance) {
			handle = std::make_shared<EngineInstance>(std::move(instance));
			// 注入所属描述符；find 内部加锁；_engines 注册后不擦除，指针可长存。
			if (const EngineDescriptor* desc = find(engineType))
				handle->setDescriptor(desc);
			// 绑定核心：核心存活期覆盖实例
			handle->attachCore(core);
		}
	} catch (...) {
		error = std::current_exception();
	}

	// 发布：锁内写 ready 或清失败槽位；set_value 放锁外，避免被唤醒者抢锁
	std::vector<EngineHandle> overflow;
	{
		std::lock_guard lk(_mutex);
		if (handle) {
			auto& slot = _engineInstances[key];
			slot.ready = handle;
			slot.loading = {}; // 清除 loading 条目
			slot.lastAccess = std::chrono::steady_clock::now();
			overflow = _evictOverflowLocked(key); // 容量收敛
		} else {
			auto it = _engineInstances.find(key);
			if (it != _engineInstances.end() && !it->second.ready)
				_engineInstances.erase(it); // 失败：清槽位，后续重试
		}
	}
	// overflow 锁外析构：被逐句柄可能是最后持有者，析构链触发用户钩子，不持 _mutex。
	// error 不得 move 进 promise：拷贝并保留原值重抛，移动后异常会静默丢失。
	if (error) {
		promise.set_exception(error);
		std::rethrow_exception(error); // 加载异常透传给首个调用者
	}
	promise.set_value(handle);
	return handle;
}

std::vector<EngineHandle> EngineRegistry::_evictOverflowLocked(const std::string& keepKey) {
	std::vector<EngineHandle> doomed;
	while (_engineInstances.size() > kMaxCachedEngineInstances) {
		auto victim = _engineInstances.end();
		auto oldest = std::chrono::steady_clock::time_point::max();
		for (auto it = _engineInstances.begin(); it != _engineInstances.end(); ++it) {
			if (it->first == keepKey)
				continue; // 刚发布或命中的条目不逐
			if (it->second.loading.valid())
				continue; // single-flight 创建中：无可释放对象，保留
			if (it->second.lastAccess < oldest) {
				oldest = it->second.lastAccess;
				victim = it;
			}
		}
		if (victim == _engineInstances.end())
			break; // 全部创建中或仅剩 keepKey：暂超限，待后续收敛
		doomed.push_back(std::move(victim->second.ready));
		_engineInstances.erase(victim);
	}
	return doomed;
}

void EngineRegistry::releaseEngineCore(const std::string& engineType) {
	// 仅移除缓存条目：仍被持有的共享句柄保持核心存活，销毁发生在最后一个
	// 句柄释放时；loading 条目保留，待发布后由后续 release 生效。
	// 待析构句柄锁外释放，用户钩子不持 _mutex。
	EngineCoreHandle doomed;
	{
		std::lock_guard lk(_mutex);
		auto it = _engineCores.find(engineType);
		if (it == _engineCores.end() || it->second.loading.valid())
			return;
		doomed = std::move(it->second.ready);
		_engineCores.erase(it);
	}
}

void EngineRegistry::releaseEngine(const std::string& engineType, const std::string& modelPath) {
	// 仅移除缓存条目：被持有的句柄保持实例存活，销毁发生在最后一个句柄释放
	// 时，调用方无需释放顺序约定；loading 条目待发布后由后续 release 生效。
	// 待析构句柄锁外释放，用户钩子不持 _mutex。
	EngineHandle doomed;
	{
		std::lock_guard lk(_mutex);
		auto it = _engineInstances.find(_makeEngineKey(engineType, modelPath));
		if (it == _engineInstances.end() || it->second.loading.valid())
			return;
		doomed = std::move(it->second.ready);
		_engineInstances.erase(it);
	}
}

void EngineRegistry::releaseAllEngines() {
	// 语义同 releaseEngine/releaseEngineCore：移除缓存条目，实际销毁由引用计数
	// 决定；loading 条目跳过；待析构句柄锁外释放，用户钩子不持 _mutex。
	std::vector<EngineHandle> doomed;
	std::vector<EngineCoreHandle> doomedCores;
	{
		std::lock_guard lk(_mutex);
		doomed.reserve(_engineInstances.size());
		for (auto it = _engineInstances.begin(); it != _engineInstances.end();) {
			if (it->second.loading.valid()) {
				++it;
				continue;
			}
			doomed.push_back(std::move(it->second.ready));
			it = _engineInstances.erase(it);
		}
		doomedCores.reserve(_engineCores.size());
		for (auto it = _engineCores.begin(); it != _engineCores.end();) {
			if (it->second.loading.valid()) {
				++it;
				continue;
			}
			doomedCores.push_back(std::move(it->second.ready));
			it = _engineCores.erase(it);
		}
	}
}

const EngineDescriptor* EngineRegistry::find(const std::string& engineType) const {
	std::lock_guard lk(_mutex);
	auto it = _engines.find(engineType);
	if (it == _engines.end()) {
		return nullptr;
	}
	// 指针指向 map 节点；_engines 注册后不擦除，地址稳定
	return &it->second;
}

bool EngineRegistry::hasEngine(const std::string& engineType) const {
	std::lock_guard lk(_mutex);
	return _engines.contains(engineType);
}

std::vector<std::string> EngineRegistry::engineTypes() const {
	std::lock_guard lk(_mutex);
	std::vector<std::string> types;
	types.reserve(_engines.size());
	for (const auto& [type, desc] : _engines) {
		types.push_back(type);
	}
	return types;
}

bool EngineRegistry::registerOperator(const std::string& operatorName, Node::Schema schema, Node::RunFn fn) {
	if (operatorName.empty())
		return false;

	// 检查与插入合并为单一临界区：并发同名注册恰一方成功，不静默覆盖
	EngineDescriptor desc;
	desc.engineType = operatorName;
	desc.converter = {builtinToNative, builtinToDC};

	desc.factory = [schema = std::move(schema),
					fn = std::move(fn)](const NodeFactoryParams& p) -> std::unique_ptr<Node> {
		return std::make_unique<Node>("Builtin", p.nodeName, schema, fn, ResourceClass::Operator);
	};

	std::lock_guard lk(_mutex);
	if (_engines.contains(operatorName))
		return false;
	_engines[operatorName] = std::move(desc);
	return true;
}

std::unique_ptr<Node> EngineRegistry::createOperator(const std::string& operatorName,
													 const std::string& nodeName) const {
	NodeFactory factory;
	{
		std::lock_guard lk(_mutex);
		auto it = _engines.find(operatorName);
		if (it == _engines.end() || !it->second.factory) {
			return nullptr;
		}
		factory = it->second.factory;
	}

	NodeFactoryParams params;
	params.nodeName = nodeName;
	return factory(params);
}

} // namespace DC
