#include "TensorSlot.h"
#include "SlotType.h"

#include <iostream>
#include <string>

// 向全局注册表注册测试类型的映射（由 EngineRegistry.cpp 的静态初始化保障 DC::Tensor 和 NativeTensor）
// DummyExternalTensor 在此通过 ValiatorRegistry（不做校验）自动放行

static void runTensorSlotTests() {
	using namespace DC;

	struct DummyExternalTensor {
		std::string payload;
	};

	// Test 1: store Tensor and peek
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("in", TensorMeta::TensorType::Float, sizeof(float), {2, 2}, cfg);

		if (!slot.isInput())
			throw std::runtime_error("slot should be input");
		if (!slot.isType<float>())
			throw std::runtime_error("slot type should be float");

		// prepare tensor
		Tensor t = Tensor::Create<float>({2, 2});
		t.fill<float>(1.5f);

		slot.store(std::move(t)); // store via type-erased API

		if (!slot.hasData())
			throw std::runtime_error("slot should have data after store");

		// peek for read-only access
		auto* viewPtr = slot.peek<Tensor>();
		if (!viewPtr)
			throw std::runtime_error("peek<Tensor> returned null");

		auto sp = viewPtr->data<float>();
		if (sp.size() != 4)
			throw std::runtime_error("unexpected data size");
		for (auto v : sp) {
			if (v != 1.5f)
				throw std::runtime_error("unexpected value in tensor");
		}

		// view() backward compat
		const auto& v = slot.view();
		if (std::abs(v.item<float>() - 1.5f) < 1e-6f) { /* ok */
		}
	}

	// Test 2: default data and take output
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Output);

		TensorSlot slot("out", TensorMeta::TensorType::Float, sizeof(float), {1, 2}, cfg);

		Tensor def = Tensor::Create<float>({1, 2});
		def.fill<float>(2.5f);
		slot.setDefaultTensor(def);

		if (!slot.hasDefaultData())
			throw std::runtime_error("slot should have default data");

		// Take via view (backward compat for default data)
		const auto& out = slot.view();
		auto sp = out.data<float>();
		if (sp.size() != 2)
			throw std::runtime_error("unexpected default tensor size");
		for (auto v : sp)
			if (v != 2.5f)
				throw std::runtime_error("unexpected default tensor value");
	}

	// Test 3: shape mismatch should throw when storing
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("badshape", TensorMeta::TensorType::Float, sizeof(float), {2, 2}, cfg);
		Tensor t = Tensor::Create<float>({1, 2});
		t.fill<float>(0.0f);
		bool thrown = false;
		try {
			slot.store(std::move(t));
		} catch (const std::exception& e) {
			thrown = true;
		}
		if (!thrown)
			throw std::runtime_error("expected exception on shape mismatch");
	}

	// Test 4: store and take arbitrary external type (DummyExternalTensor)
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("ext", TensorMeta::TensorType::Float, sizeof(float), {1}, cfg);

		// store external type directly (no validator registered → pass-through)
		DummyExternalTensor ext{"moved"};
		slot.store(std::move(ext));

		if (!slot.hasData())
			throw std::runtime_error("slot should have external data");
		if (slot.storedType() == ensureSlotType<Tensor>())
			throw std::runtime_error("stored type should not be DCTensor");

		// take back
		auto got = slot.take<DummyExternalTensor>();
		if (got.payload != "moved")
			throw std::runtime_error("take<DummyExternalTensor> payload mismatch");

		// slot should be empty after take
		if (slot.hasData())
			throw std::runtime_error("slot should be empty after take");
	}

	// Test 5: type mismatch on take should throw
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Output);

		TensorSlot slot("typemismatch", TensorMeta::TensorType::Float, sizeof(float), {}, cfg);

		Tensor t(TensorMeta::TensorType::Float, sizeof(float));
		t = 42.0f;
		slot.store(std::move(t));

		if (slot.storedType() != ensureSlotType<Tensor>())
			throw std::runtime_error("expected DCTensor type");

		// Try to take as wrong type
		bool thrown = false;
		try {
			slot.take<DummyExternalTensor>();
		} catch (const std::exception&) {
			thrown = true;
		}
		if (!thrown)
			throw std::runtime_error("expected exception on type mismatch take");
	}

	// Test 6: store 构造抛出（拷贝构造可抛）时旧值保持完好（#5 强异常安全）
	{
		struct ThrowingCopy {
			std::string payload;
			bool throwOnCopy = false;

			ThrowingCopy() = default;
			ThrowingCopy(std::string p, bool bomb) : payload(std::move(p)), throwOnCopy(bomb) {}
			ThrowingCopy(const ThrowingCopy& o) : payload(o.payload), throwOnCopy(o.throwOnCopy) {
				if (o.throwOnCopy)
					throw std::runtime_error("copy boom");
			}
			ThrowingCopy(ThrowingCopy&&) noexcept = default;
			ThrowingCopy& operator=(const ThrowingCopy&) = delete;
			ThrowingCopy& operator=(ThrowingCopy&&) noexcept = default;
		};

		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);
		TensorSlot slot("safe", TensorMeta::TensorType::Float, sizeof(float), {1}, cfg);

		ThrowingCopy oldVal{"intact", false};
		slot.store(oldVal); // lvalue → 拷贝构造存储

		const auto* before = slot.peek<ThrowingCopy>();
		if (!before || before->payload != "intact")
			throw std::runtime_error("sanity: initial store failed");

		ThrowingCopy bomby{"replacement", true};
		bool thrown = false;
		try {
			slot.store(bomby); // 拷贝构造抛出：旧值必须保持完好（不得悬垂/双释放）
		} catch (const std::runtime_error&) {
			thrown = true;
		}
		if (!thrown)
			throw std::runtime_error("expected exception from throwing copy on store");
		if (!slot.hasData())
			throw std::runtime_error("old value must survive a throwing store");
		const auto* still = slot.peek<ThrowingCopy>();
		if (!still || still->payload != "intact")
			throw std::runtime_error("old value corrupted after throwing store (UAF/double-free)");

		// 后续正常 store 仍工作（槽位状态未被破坏）
		ThrowingCopy fresh{"fresh", false};
		slot.store(fresh);
		const auto* after = slot.peek<ThrowingCopy>();
		if (!after || after->payload != "fresh")
			throw std::runtime_error("slot must remain usable after a throwing store");
	}
}

int main() {
	try {
		runTensorSlotTests();

		std::cout << "TensorSlotBase tests passed" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
